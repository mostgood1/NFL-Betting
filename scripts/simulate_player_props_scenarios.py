"""Generate deterministic scenario-adjusted player props artifacts.

This script is intentionally conservative: it does not re-run scenario simulation.
Instead, it consumes the existing scenario outputs from:
  nfl_compare/data/backtests/{season}_wk{week}/sim_probs_scenarios.csv

Outputs:
  - player_props_scenarios.csv (player-level, scenario-adjusted)
  - player_props_scenarios_summary.csv (scenario x team rollups)
  - player_props_scenarios_meta.json (provenance)

Design goals:
  - Deterministic and reproducible (stable sorting, no RNG)
  - Best-effort fallbacks; never hard-fail on partial optional columns
  - Keep schema stable by reusing existing player props columns
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(REPO_ROOT / "nfl_compare" / "data"))).resolve()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "scenario"


def _coerce_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))


def _clip_scale(x: pd.Series, lo: float, hi: float) -> pd.Series:
    try:
        return x.clip(lower=float(lo), upper=float(hi))
    except Exception:
        return x


def _pow_scale(x: pd.Series, power: float) -> pd.Series:
    try:
        return np.power(x.astype(float), float(power))
    except Exception:
        return x


@dataclass
class ScaleConfig:
    baseline_scenario_id: str = "baseline"
    points_eps: float = 1.0
    vol_power: float = 0.55
    yds_power: float = 0.70
    td_power: float = 1.00
    vol_clip_lo: float = 0.70
    vol_clip_hi: float = 1.30
    yds_clip_lo: float = 0.60
    yds_clip_hi: float = 1.40
    td_clip_lo: float = 0.50
    td_clip_hi: float = 1.60


def _infer_set_slug(backtest_dir: Path) -> str:
    meta = _safe_read_json(backtest_dir / "sim_scenarios_meta.json")
    scfg = meta.get("scenario_config") if isinstance(meta, dict) else None
    if isinstance(scfg, dict):
        path = scfg.get("path")
        if path:
            try:
                return _slugify(Path(str(path)).stem)
            except Exception:
                pass
    return "player_props_scenarios"


def _build_team_points_long(sim: pd.DataFrame) -> pd.DataFrame:
    if sim is None or sim.empty:
        return pd.DataFrame()

    req = {"scenario_id", "game_id", "home_team", "away_team", "home_points_mean", "away_points_mean", "total_points_mean"}
    missing = [c for c in req if c not in sim.columns]
    if missing:
        return pd.DataFrame()

    out_rows = []
    for side, team_col, pts_col in [
        ("home", "home_team", "home_points_mean"),
        ("away", "away_team", "away_points_mean"),
    ]:
        tmp = sim[["scenario_id", "scenario_label", "season", "week", "game_id", team_col, pts_col, "total_points_mean"]].copy()
        tmp = tmp.rename(columns={team_col: "team", pts_col: "team_points_mean"})
        tmp["side"] = side
        out_rows.append(tmp)

    out = pd.concat(out_rows, ignore_index=True)
    # Normalize types
    for c in ["season", "week"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["game_id"] = out["game_id"].astype(str)
    out["scenario_id"] = out["scenario_id"].astype(str)
    out["team"] = out["team"].astype(str)
    out["team_points_mean"] = pd.to_numeric(out["team_points_mean"], errors="coerce")
    out["total_points_mean"] = pd.to_numeric(out["total_points_mean"], errors="coerce")
    return out


def _choose_baseline_id(sim: pd.DataFrame, requested: str) -> str:
    if sim is None or sim.empty or "scenario_id" not in sim.columns:
        return requested

    ids = sorted({str(x) for x in sim["scenario_id"].dropna().unique().tolist()})
    if requested in ids:
        return requested

    # Heuristic fallback: anything containing "baseline".
    for cand in ids:
        if "baseline" in cand.lower():
            return cand

    # Last resort: first by sort.
    return ids[0] if ids else requested


def _apply_team_scales(props: pd.DataFrame, scales: pd.DataFrame, cfg: ScaleConfig) -> pd.DataFrame:
    out = props.copy()

    # Join scales
    out["game_id"] = out["game_id"].astype(str)
    out["team"] = out["team"].astype(str)
    out = out.merge(
        scales[["game_id", "team", "points_scale", "vol_scale", "yds_scale", "td_scale"]],
        on=["game_id", "team"],
        how="left",
    )

    # Columns to scale (best-effort)
    vol_cols = [
        "team_plays",
        "team_pass_attempts",
        "team_rush_attempts",
        "pass_attempts",
        "rush_attempts",
        "targets",
        "receptions",
    ]
    yds_cols = [
        "team_pass_yards",
        "team_rush_yards",
        "pass_yards",
        "rush_yards",
        "rec_yards",
    ]
    td_cols = [
        "team_exp_pass_tds",
        "team_exp_rush_tds",
        "pass_tds",
        "rush_tds",
        "rec_tds",
    ]

    for c in vol_cols:
        if c in out.columns and "vol_scale" in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * pd.to_numeric(out["vol_scale"], errors="coerce")

    for c in yds_cols:
        if c in out.columns and "yds_scale" in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * pd.to_numeric(out["yds_scale"], errors="coerce")

    for c in td_cols:
        if c in out.columns and "td_scale" in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * pd.to_numeric(out["td_scale"], errors="coerce")

    # INTs: scale with volume if present
    if "interceptions" in out.columns and "vol_scale" in out.columns:
        out["interceptions"] = pd.to_numeric(out["interceptions"], errors="coerce") * pd.to_numeric(out["vol_scale"], errors="coerce")

    # Recompute any_td_prob from TD lambdas, matching player_props pipeline convention.
    if {"rush_tds", "rec_tds", "any_td_prob"}.issubset(out.columns):
        if "is_active" in out.columns:
            act = pd.to_numeric(out["is_active"], errors="coerce").fillna(1.0) > 0
        else:
            # If actives aren't present in the base props cache, treat all rows as active.
            act = pd.Series([True] * len(out), index=out.index)
        lam = pd.to_numeric(out.loc[act, "rush_tds"], errors="coerce").fillna(0.0) + pd.to_numeric(
            out.loc[act, "rec_tds"], errors="coerce"
        ).fillna(0.0)
        out.loc[act, "any_td_prob"] = (1.0 - np.exp(-lam)).astype(float)
        out["any_td_prob"] = pd.to_numeric(out["any_td_prob"], errors="coerce").clip(lower=0.0, upper=1.0)

    # Stable ordering
    sort_cols = [c for c in ["scenario_id", "game_id", "team", "position", "player"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--baseline-scenario-id", type=str, default="baseline")
    ap.add_argument("--out-dir", type=str, default="")

    # Scaling config knobs (kept here so we can pin them in meta and sweep later if desired)
    ap.add_argument("--vol-power", type=float, default=0.55)
    ap.add_argument("--yds-power", type=float, default=0.70)
    ap.add_argument("--td-power", type=float, default=1.00)
    ap.add_argument("--vol-clip", type=str, default="0.70,1.30")
    ap.add_argument("--yds-clip", type=str, default="0.60,1.40")
    ap.add_argument("--td-clip", type=str, default="0.50,1.60")
    ap.add_argument("--points-eps", type=float, default=1.0)

    args = ap.parse_args()

    s = int(args.season)
    w = int(args.week)
    backtest_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (DATA_DIR / "backtests" / f"{s}_wk{w}")
    backtest_dir.mkdir(parents=True, exist_ok=True)

    props_fp = DATA_DIR / f"player_props_{s}_wk{w}.csv"
    sim_fp = backtest_dir / "sim_probs_scenarios.csv"

    if not props_fp.exists():
        print(f"WARN: missing player props cache: {props_fp}")
        return 0

    if not sim_fp.exists():
        print(f"ERROR: missing scenario sim output: {sim_fp}")
        print("Run: python scripts/simulate_scenarios.py --season <S> --week <W> --scenario-set <name>")
        return 2

    props = _safe_read_csv(props_fp)
    sim = _safe_read_csv(sim_fp)
    if props.empty:
        print(f"WARN: empty props cache: {props_fp}")
        return 0
    if sim.empty:
        print(f"WARN: empty scenario sim output: {sim_fp}")
        return 0

    # Parse clip args
    def _parse_pair(s0: str, default: Tuple[float, float]) -> Tuple[float, float]:
        try:
            a, b = [float(x.strip()) for x in str(s0).split(",", 1)]
            return float(a), float(b)
        except Exception:
            return default

    vol_lo, vol_hi = _parse_pair(args.vol_clip, (0.70, 1.30))
    yds_lo, yds_hi = _parse_pair(args.yds_clip, (0.60, 1.40))
    td_lo, td_hi = _parse_pair(args.td_clip, (0.50, 1.60))

    cfg = ScaleConfig(
        baseline_scenario_id=str(args.baseline_scenario_id),
        points_eps=float(args.points_eps),
        vol_power=float(args.vol_power),
        yds_power=float(args.yds_power),
        td_power=float(args.td_power),
        vol_clip_lo=float(vol_lo),
        vol_clip_hi=float(vol_hi),
        yds_clip_lo=float(yds_lo),
        yds_clip_hi=float(yds_hi),
        td_clip_lo=float(td_lo),
        td_clip_hi=float(td_hi),
    )

    # Ensure required keys exist
    for c in ["season", "week", "game_id", "team", "player"]:
        if c not in props.columns:
            print(f"ERROR: props cache missing required column: {c}")
            return 3

    # Normalize sim scenario label
    if "scenario_label" not in sim.columns:
        sim["scenario_label"] = sim.get("scenario_id")

    baseline_id = _choose_baseline_id(sim, cfg.baseline_scenario_id)

    tp = _build_team_points_long(sim)
    if tp.empty:
        print("ERROR: sim_probs_scenarios.csv missing required team points columns")
        return 4

    # Build per-(scenario, game, team) scales
    base = tp[tp["scenario_id"].astype(str) == str(baseline_id)].copy()
    if base.empty:
        print(f"ERROR: baseline scenario not found in sim output: {baseline_id}")
        return 5

    base = base.rename(columns={"team_points_mean": "base_team_points_mean", "total_points_mean": "base_total_points_mean"})
    keys = ["game_id", "team"]
    base = base[keys + ["base_team_points_mean", "base_total_points_mean"]].drop_duplicates(subset=keys)

    cur = tp.copy()
    cur = cur.merge(base, on=keys, how="left")

    denom = pd.to_numeric(cur["base_team_points_mean"], errors="coerce")
    numer = pd.to_numeric(cur["team_points_mean"], errors="coerce")
    ok = denom.fillna(0.0) > float(cfg.points_eps)

    points_scale = pd.Series(np.ones(len(cur), dtype=float))
    points_scale.loc[ok] = (numer.loc[ok] / denom.loc[ok]).astype(float)
    points_scale = points_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    vol_scale = _clip_scale(_pow_scale(points_scale, cfg.vol_power), cfg.vol_clip_lo, cfg.vol_clip_hi)
    yds_scale = _clip_scale(_pow_scale(points_scale, cfg.yds_power), cfg.yds_clip_lo, cfg.yds_clip_hi)
    td_scale = _clip_scale(_pow_scale(points_scale, cfg.td_power), cfg.td_clip_lo, cfg.td_clip_hi)

    scales = cur[["scenario_id", "scenario_label", "game_id", "team"]].copy()
    scales["points_scale"] = points_scale.astype(float)
    scales["vol_scale"] = vol_scale.astype(float)
    scales["yds_scale"] = yds_scale.astype(float)
    scales["td_scale"] = td_scale.astype(float)

    # Generate scenario-adjusted props
    out_parts = []
    # Expand props for each scenario by joining per-game/team scale.
    # We keep this memory-friendly by iterating scenarios.
    scenario_ids = sorted({str(x) for x in scales["scenario_id"].dropna().unique().tolist()})
    for sid in scenario_ids:
        sc = scales[scales["scenario_id"].astype(str) == sid].copy()
        if sc.empty:
            continue
        p0 = props.copy()
        p0 = _apply_team_scales(p0, sc, cfg)
        p0.insert(0, "scenario_id", sid)
        # scenario_label might vary by game; attach a best-effort label
        try:
            lab = sc["scenario_label"].dropna().astype(str).unique().tolist()
            p0.insert(1, "scenario_label", lab[0] if lab else sid)
        except Exception:
            p0.insert(1, "scenario_label", sid)
        out_parts.append(p0)

    if not out_parts:
        print("WARN: no scenario-adjusted props produced")
        return 0

    out = pd.concat(out_parts, ignore_index=True)

    # Write outputs with archived copies
    set_slug = _infer_set_slug(backtest_dir)
    out_fp = backtest_dir / "player_props_scenarios.csv"
    out_arch_fp = backtest_dir / f"player_props_scenarios__{set_slug}.csv"
    out.to_csv(out_fp, index=False)
    try:
        out.to_csv(out_arch_fp, index=False)
    except Exception:
        pass

    # Scenario x team summary
    group_cols = [c for c in ["scenario_id", "scenario_label", "season", "week", "game_id", "team"] if c in out.columns]
    agg_map = {
        "pass_attempts": "sum",
        "rush_attempts": "sum",
        "targets": "sum",
        "receptions": "sum",
        "pass_yards": "sum",
        "rush_yards": "sum",
        "rec_yards": "sum",
        "pass_tds": "sum",
        "rush_tds": "sum",
        "rec_tds": "sum",
    }
    have = [c for c in agg_map.keys() if c in out.columns]
    if group_cols and have:
        summ = out.groupby(group_cols, dropna=False)[have].sum(numeric_only=True).reset_index()
    else:
        summ = pd.DataFrame()

    summ_fp = backtest_dir / "player_props_scenarios_summary.csv"
    summ_arch_fp = backtest_dir / f"player_props_scenarios_summary__{set_slug}.csv"
    if not summ.empty:
        summ.to_csv(summ_fp, index=False)
        try:
            summ.to_csv(summ_arch_fp, index=False)
        except Exception:
            pass

    meta = {
        "season": s,
        "week": w,
        "data_dir": str(DATA_DIR).replace("\\", "/"),
        "backtest_dir": str(backtest_dir).replace("\\", "/"),
        "inputs": {
            "player_props": {"path": str(props_fp), "sha256": _sha256_file(props_fp), "rows": int(props.shape[0]), "cols": int(props.shape[1])},
            "sim_probs_scenarios": {"path": str(sim_fp), "sha256": _sha256_file(sim_fp), "rows": int(sim.shape[0]), "cols": int(sim.shape[1])},
            "sim_scenarios_meta": {"path": str(backtest_dir / "sim_scenarios_meta.json"), "sha256": _sha256_file(backtest_dir / "sim_scenarios_meta.json")}
            if (backtest_dir / "sim_scenarios_meta.json").exists()
            else None,
        },
        "scaling": {
            "baseline_scenario_id_requested": str(cfg.baseline_scenario_id),
            "baseline_scenario_id_used": str(baseline_id),
            **{k: v for k, v in asdict(cfg).items() if k != "baseline_scenario_id"},
        },
        "outputs": {
            "player_props_scenarios_csv": {"path": str(out_fp), "sha256": _sha256_file(out_fp), "rows": int(out.shape[0]), "cols": int(out.shape[1]), "archive": str(out_arch_fp)},
            "player_props_scenarios_summary_csv": {
                "path": str(summ_fp),
                "sha256": _sha256_file(summ_fp) if summ_fp.exists() else None,
                "rows": int(summ.shape[0]) if not summ.empty else 0,
                "cols": int(summ.shape[1]) if not summ.empty else 0,
                "archive": str(summ_arch_fp) if not summ.empty else None,
            },
        },
    }

    meta_fp = backtest_dir / "player_props_scenarios_meta.json"
    meta_arch_fp = backtest_dir / f"player_props_scenarios_meta__{set_slug}.json"
    with meta_fp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        with meta_arch_fp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception:
        pass

    print(f"Wrote: {out_fp}")
    if summ_fp.exists():
        print(f"Wrote: {summ_fp}")
    print(f"Wrote: {meta_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
