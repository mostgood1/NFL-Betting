"""Evaluate scenario-adjusted player props vs actuals for a given week.

Inputs:
- nfl_compare/data/backtests/{season}_wk{week}/player_props_scenarios.csv
- nfl_compare/data/backtests/{season}_wk{week}/player_props_scenarios_meta.json
- nfl_compare/data/player_props_vs_actuals_{season}_wk{week}.csv

Outputs (default to backtests dir):
- player_props_scenarios_accuracy.csv  (scenario-level metrics)
- player_props_scenarios_coverage.csv  (scenario envelope coverage metrics)
- player_props_scenarios_accuracy.md   (small markdown summary)

Metrics:
- MAE for numeric props (pred vs *_act)
- Brier score for any_td_prob vs y_any_td (rush_tds_act + rec_tds_act > 0)
- Coverage: pct where actual falls within min..max across scenarios (per stat)

Notes:
- This is a *measurement* tool; it does not claim scenarios should be "more accurate".
  Scenarios are primarily for sensitivity / robustness. Coverage is often the more
  relevant diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(REPO_ROOT / "nfl_compare" / "data"))).resolve()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        yt = pd.to_numeric(y_true, errors="coerce").astype(float)
        yp = pd.to_numeric(y_pred, errors="coerce").astype(float)
        m = yt.notna() & yp.notna()
        if not m.any():
            return float("nan")
        return float((yt[m] - yp[m]).abs().mean())
    except Exception:
        return float("nan")


def _brier(p: pd.Series, y: pd.Series) -> float:
    try:
        pp = pd.to_numeric(p, errors="coerce").astype(float)
        yy = pd.to_numeric(y, errors="coerce").astype(float)
        m = pp.notna() & yy.notna()
        if not m.any():
            return float("nan")
        return float(((pp[m] - yy[m]) ** 2).mean())
    except Exception:
        return float("nan")


def _norm_keys(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def _infer_baseline_id(meta: Dict) -> Optional[str]:
    sc = meta.get("scaling") if isinstance(meta, dict) else None
    if isinstance(sc, dict):
        used = sc.get("baseline_scenario_id_used")
        if isinstance(used, str) and used.strip():
            return used.strip()
    return None


def _compute_y_any_td(actuals: pd.DataFrame) -> pd.Series:
    r = pd.to_numeric(actuals.get("rec_tds_act"), errors="coerce").fillna(0.0)
    u = pd.to_numeric(actuals.get("rush_tds_act"), errors="coerce").fillna(0.0)
    return ((r + u) > 0).astype(float)


def evaluate(
    props_scenarios: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    baseline_scenario_id: Optional[str],
    restrict_active: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    # Join on predicted player id
    if props_scenarios.empty or actuals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    ps = props_scenarios.copy()
    act = actuals.copy()

    # Normalize keys
    ps = _norm_keys(ps, ["game_id", "team", "player_id", "player"])
    if "player_id_x" in act.columns:
        act = act.rename(columns={"player_id_x": "player_id"})
    act = _norm_keys(act, ["game_id", "team", "player_id", "player"])

    join_keys = [k for k in ["game_id", "team", "player_id"] if k in ps.columns and k in act.columns]
    if not join_keys:
        # fallback to name join
        join_keys = [k for k in ["game_id", "team", "player"] if k in ps.columns and k in act.columns]

    # Keep only needed actual columns
    act_cols = [c for c in act.columns if c.endswith("_act")]
    act_keep = list(dict.fromkeys(join_keys + act_cols + [c for c in ["position", "is_active"] if c in act.columns]))
    act2 = act[act_keep].copy()

    j = ps.merge(act2, on=join_keys, how="left", suffixes=("", "_actfile"))

    # Optionally restrict to active predictions
    if restrict_active and "is_active" in j.columns:
        ia = pd.to_numeric(j["is_active"], errors="coerce")
        j = j[ia.fillna(0.0) > 0].copy()

    # Metrics to evaluate
    stats = [
        "pass_attempts",
        "pass_yards",
        "pass_tds",
        "interceptions",
        "rush_attempts",
        "rush_yards",
        "rush_tds",
        "targets",
        "receptions",
        "rec_yards",
        "rec_tds",
    ]

    rows = []
    for sid, g in j.groupby("scenario_id", dropna=False):
        sid_s = str(sid)
        row = {
            "scenario_id": sid_s,
            "scenario_label": str(g["scenario_label"].dropna().iloc[0]) if "scenario_label" in g.columns and g["scenario_label"].notna().any() else sid_s,
            "n_rows": int(len(g)),
        }

        # MAEs
        for stat in stats:
            pred_col = stat
            act_col = f"{stat}_act"
            if pred_col in g.columns and act_col in g.columns:
                row[f"mae_{stat}"] = _mae(g[act_col], g[pred_col])
                # coverage helper later
            else:
                row[f"mae_{stat}"] = float("nan")

        # any TD brier
        if "any_td_prob" in g.columns and ("rush_tds_act" in g.columns or "rec_tds_act" in g.columns):
            y_any = _compute_y_any_td(g)
            row["brier_any_td"] = _brier(g["any_td_prob"], y_any)
            row["any_td_rate"] = float(np.nanmean(y_any.values)) if len(y_any) else float("nan")
        else:
            row["brier_any_td"] = float("nan")
            row["any_td_rate"] = float("nan")

        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["scenario_id"]).reset_index(drop=True)

    # Scenario x position metrics
    pos_rows = []
    if "position" in j.columns:
        for (sid, pos), g in j.groupby(["scenario_id", "position"], dropna=False):
            sid_s = str(sid)
            pos_s = str(pos) if pos is not None else ""
            row = {
                "scenario_id": sid_s,
                "position": pos_s,
                "scenario_label": str(g["scenario_label"].dropna().iloc[0])
                if "scenario_label" in g.columns and g["scenario_label"].notna().any()
                else sid_s,
                "n_rows": int(len(g)),
            }
            for stat in stats:
                pred_col = stat
                act_col = f"{stat}_act"
                row[f"mae_{stat}"] = _mae(g[act_col], g[pred_col]) if (pred_col in g.columns and act_col in g.columns) else float("nan")
            if "any_td_prob" in g.columns and ("rush_tds_act" in g.columns or "rec_tds_act" in g.columns):
                y_any = _compute_y_any_td(g)
                row["brier_any_td"] = _brier(g["any_td_prob"], y_any)
                row["any_td_rate"] = float(np.nanmean(y_any.values)) if len(y_any) else float("nan")
            else:
                row["brier_any_td"] = float("nan")
                row["any_td_rate"] = float("nan")
            pos_rows.append(row)

    metrics_pos = pd.DataFrame(pos_rows)
    if not metrics_pos.empty:
        metrics_pos = metrics_pos.sort_values(["position", "scenario_id"]).reset_index(drop=True)

    # Coverage: compare actual within scenario min..max per player
    cov_rows = []
    id_cols = [c for c in ["game_id", "team", "player_id", "player", "position"] if c in j.columns]
    for stat in stats:
        act_col = f"{stat}_act"
        if act_col not in j.columns or stat not in j.columns:
            continue

        base = j[id_cols + ["scenario_id", stat, act_col]].copy()
        base[stat] = pd.to_numeric(base[stat], errors="coerce")
        base[act_col] = pd.to_numeric(base[act_col], errors="coerce")
        # Collapse to one actual per player-key (actual columns repeat across scenarios)
        grp_keys = [c for c in id_cols if c != "scenario_id"]
        if not grp_keys:
            continue

        # min/max across scenarios
        gmm = base.groupby(grp_keys, dropna=False)
        vmin = gmm[stat].min()
        vmax = gmm[stat].max()
        y = gmm[act_col].first()

        m = y.notna() & vmin.notna() & vmax.notna()
        if not m.any():
            continue

        inside = (y[m] >= vmin[m]) & (y[m] <= vmax[m])
        cov = float(inside.mean())

        # Also baseline MAE for reference
        mae_base = float("nan")
        if baseline_scenario_id is not None and baseline_scenario_id in set(j["scenario_id"].astype(str).unique().tolist()):
            bmask = j["scenario_id"].astype(str) == str(baseline_scenario_id)
            mae_base = _mae(j.loc[bmask, act_col], j.loc[bmask, stat])

        cov_rows.append(
            {
                "stat": stat,
                "n_players": int(m.sum()),
                "coverage_min_max": cov,
                "baseline_scenario_id": str(baseline_scenario_id) if baseline_scenario_id else None,
                "baseline_mae": mae_base,
            }
        )

    coverage = pd.DataFrame(cov_rows)
    if not coverage.empty:
        coverage = coverage.sort_values(["stat"]).reset_index(drop=True)

    # Best scenario by composite MAE
    core_mae_cols = [c for c in metrics.columns if c.startswith("mae_")]
    metrics2 = metrics.copy()
    metrics2["mae_mean"] = metrics2[core_mae_cols].mean(axis=1, numeric_only=True) if core_mae_cols else np.nan
    # Promote so it is persisted to CSV and consumable by downstream tooling
    metrics = metrics2

    # Also expose mae_mean on the per-position table (same definition)
    if metrics_pos is not None and not metrics_pos.empty:
        try:
            core_mae_cols_pos = [c for c in metrics_pos.columns if c.startswith("mae_")]
            if core_mae_cols_pos:
                metrics_pos = metrics_pos.copy()
                metrics_pos["mae_mean"] = metrics_pos[core_mae_cols_pos].mean(axis=1, numeric_only=True)
        except Exception:
            pass
    best_pick: Optional[str] = None
    try:
        if not metrics2.empty:
            best_pick = str(metrics2.sort_values(["mae_mean"], na_position="last").iloc[0]["scenario_id"])
    except Exception:
        best_pick = None

    # Top error contributors (baseline + best scenario)
    err_frames = []
    id_cols_for_err = [c for c in ["game_id", "team", "player", "player_id", "position"] if c in j.columns]

    def _top_errors_for_scenario(sid_pick: str, pick_label: str) -> None:
        if not sid_pick:
            return
        msk = j["scenario_id"].astype(str) == str(sid_pick)
        g = j.loc[msk].copy()
        if g.empty:
            return

        key_stats = ["rec_yards", "receptions", "targets", "rush_yards", "rush_attempts", "pass_yards", "pass_attempts"]
        key_stats = [s for s in key_stats if (s in g.columns and f"{s}_act" in g.columns)]
        if not key_stats:
            return

        comp = None
        for s in key_stats:
            e = (pd.to_numeric(g[s], errors="coerce") - pd.to_numeric(g[f"{s}_act"], errors="coerce")).abs()
            comp = e if comp is None else (comp + e)
        g["abs_error_sum"] = pd.to_numeric(comp, errors="coerce")

        keep = [c for c in ["scenario_id", "scenario_label", "abs_error_sum"] if c in g.columns] + id_cols_for_err
        for s in ["rec_yards", "targets", "rush_yards", "pass_yards"]:
            if s in g.columns and f"{s}_act" in g.columns:
                keep += [s, f"{s}_act"]

        out = g[keep].copy()
        out["pick"] = pick_label
        out = out.sort_values(["abs_error_sum"], ascending=False).head(30).reset_index(drop=True)
        err_frames.append(out)

    baseline_pick = str(baseline_scenario_id) if baseline_scenario_id else ""
    _top_errors_for_scenario(baseline_pick, "baseline")
    if best_pick and best_pick != baseline_pick:
        _top_errors_for_scenario(str(best_pick), "best_mae_mean")

    top_errors = pd.concat(err_frames, ignore_index=True) if err_frames else pd.DataFrame()

    summary = {
        "baseline_scenario_id": baseline_scenario_id,
        "n_scenarios": int(metrics.shape[0]) if not metrics.empty else 0,
        "n_rows_joined": int(len(j)),
        "restrict_active": bool(restrict_active),
        "join_keys": join_keys,
        "best_scenario_id_by_mae_mean": best_pick,
    }

    return metrics, metrics_pos, coverage, top_errors, summary


def _write_md(
    path: Path,
    metrics: pd.DataFrame,
    metrics_pos: pd.DataFrame,
    coverage: pd.DataFrame,
    top_errors: pd.DataFrame,
    summary: Dict,
) -> None:
    def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df is None or df.empty:
            return ""
        df2 = df.head(int(max_rows)).copy()
        cols = list(df2.columns)
        # stringify + keep it deterministic
        for c in cols:
            try:
                if pd.api.types.is_float_dtype(df2[c]) or pd.api.types.is_integer_dtype(df2[c]):
                    df2[c] = pd.to_numeric(df2[c], errors="coerce")
                    df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else (f"{float(x):.4f}" if abs(float(x)) < 1000 else f"{float(x):.2f}"))
                else:
                    df2[c] = df2[c].astype(str)
            except Exception:
                df2[c] = df2[c].astype(str)

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in df2.iterrows():
            rows.append("| " + " | ".join([str(r.get(c, "")) for c in cols]) + " |")
        return "\n".join([header, sep] + rows)

    lines = []
    lines.append(f"# Player Props Scenario Accuracy — {summary.get('baseline_scenario_id')}")
    lines.append("")
    lines.append(f"- scenarios: {summary.get('n_scenarios')}")
    lines.append(f"- joined rows: {summary.get('n_rows_joined')}")
    lines.append(f"- restrict_active: {summary.get('restrict_active')}")
    lines.append(f"- join_keys: {', '.join(summary.get('join_keys') or [])}")
    if summary.get("best_scenario_id_by_mae_mean"):
        lines.append(f"- best_scenario_id_by_mae_mean: {summary.get('best_scenario_id_by_mae_mean')}")
    lines.append("")

    if not metrics.empty:
        # Rank by a simple composite: avg MAE over core stats + brier_any_td
        core = [c for c in metrics.columns if c.startswith("mae_")]
        tmp = metrics.copy()
        tmp["mae_mean"] = tmp[core].mean(axis=1, numeric_only=True) if core else np.nan
        cols = ["scenario_id", "scenario_label", "n_rows", "mae_mean", "brier_any_td", "any_td_rate"]
        cols = [c for c in cols if c in tmp.columns]
        lines.append("## Scenario Summary")
        lines.append("")
        lines.append(_md_table(tmp[cols].sort_values(["mae_mean"], na_position="last"), max_rows=12))
        lines.append("")

    if not coverage.empty:
        lines.append("## Scenario Envelope Coverage (min..max across scenarios)")
        lines.append("")
        lines.append(_md_table(coverage, max_rows=50))
        lines.append("")

    if metrics_pos is not None and not metrics_pos.empty:
        lines.append("## By Position (baseline + best)")
        lines.append("")
        # Show baseline and best only to keep markdown compact
        picks = []
        b = summary.get("baseline_scenario_id")
        if b:
            picks.append(str(b))
        best = summary.get("best_scenario_id_by_mae_mean")
        if best and str(best) not in picks:
            picks.append(str(best))
        dfp = metrics_pos[metrics_pos["scenario_id"].astype(str).isin(picks)].copy()
        core = [c for c in dfp.columns if c.startswith("mae_")]
        if core:
            dfp["mae_mean"] = dfp[core].mean(axis=1, numeric_only=True)
        cols = ["scenario_id", "position", "n_rows", "mae_mean", "brier_any_td"]
        cols = [c for c in cols if c in dfp.columns]
        lines.append(_md_table(dfp[cols].sort_values(["position", "scenario_id"]), max_rows=200))
        lines.append("")

    if top_errors is not None and not top_errors.empty:
        lines.append("## Top Error Contributors")
        lines.append("")
        cols = [c for c in ["pick", "scenario_id", "game_id", "team", "player", "position", "abs_error_sum", "rec_yards", "rec_yards_act", "targets", "targets_act", "rush_yards", "rush_yards_act", "pass_yards", "pass_yards_act"] if c in top_errors.columns]
        lines.append(_md_table(top_errors[cols], max_rows=30))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate scenario-adjusted player props vs actuals")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--restrict-active", action="store_true", help="Only evaluate rows with is_active>0")
    args = ap.parse_args()

    s = int(args.season)
    w = int(args.week)
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (DATA_DIR / "backtests" / f"{s}_wk{w}")
    out_dir.mkdir(parents=True, exist_ok=True)

    props_fp = out_dir / "player_props_scenarios.csv"
    meta_fp = out_dir / "player_props_scenarios_meta.json"
    actuals_fp = DATA_DIR / f"player_props_vs_actuals_{s}_wk{w}.csv"

    props_scen = _safe_read_csv(props_fp)
    meta = _safe_read_json(meta_fp)
    actuals = _safe_read_csv(actuals_fp)

    if props_scen.empty:
        print(f"ERROR: missing or empty {props_fp}")
        return 2
    if actuals.empty:
        print(f"ERROR: missing or empty {actuals_fp}")
        return 3

    baseline_id = _infer_baseline_id(meta)

    metrics, metrics_pos, coverage, top_errors, summary = evaluate(
        props_scen,
        actuals,
        baseline_scenario_id=baseline_id,
        restrict_active=bool(args.restrict_active),
    )

    acc_fp = out_dir / "player_props_scenarios_accuracy.csv"
    acc_pos_fp = out_dir / "player_props_scenarios_accuracy_by_position.csv"
    cov_fp = out_dir / "player_props_scenarios_coverage.csv"
    top_fp = out_dir / "player_props_scenarios_top_errors.csv"
    md_fp = out_dir / "player_props_scenarios_accuracy.md"

    if not metrics.empty:
        metrics.to_csv(acc_fp, index=False)
        print(f"Wrote: {acc_fp}")
    if metrics_pos is not None and not metrics_pos.empty:
        metrics_pos.to_csv(acc_pos_fp, index=False)
        print(f"Wrote: {acc_pos_fp}")
    if not coverage.empty:
        coverage.to_csv(cov_fp, index=False)
        print(f"Wrote: {cov_fp}")
    if top_errors is not None and not top_errors.empty:
        top_errors.to_csv(top_fp, index=False)
        print(f"Wrote: {top_fp}")

    _write_md(md_fp, metrics, metrics_pos, coverage, top_errors, summary)
    print(f"Wrote: {md_fp}")

    # Also print baseline row summary if present
    if baseline_id and not metrics.empty:
        try:
            b = metrics[metrics["scenario_id"].astype(str) == str(baseline_id)].copy()
            if not b.empty:
                mae_cols = [c for c in b.columns if c.startswith("mae_")]
                mae_mean = float(b[mae_cols].mean(axis=1, numeric_only=True).iloc[0]) if mae_cols else float("nan")
                print(f"Baseline {baseline_id}: mae_mean={mae_mean:.3f} brier_any_td={float(b['brier_any_td'].iloc[0]):.4f}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
