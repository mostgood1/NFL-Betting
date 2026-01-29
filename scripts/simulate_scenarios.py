"""\
Generate scenario-based Monte Carlo simulation datasets.

Goal:
- Use existing model means + all available features (weather/injuries/etc.) to simulate a small,
  deterministic set of scenario perturbations.
- Write a single, stable dataset that can be consumed by analysis/backtests and by future
  higher-level simulation workflows.

Outputs (default under nfl_compare/data/backtests/{season}_wk{week}):
- sim_probs_scenarios.csv: concatenated sim_probs with scenario_id/label and scenario knobs
- sim_scenario_inputs.csv: per-game per-scenario input features after applying scenario perturbations
- sim_scenarios.json: scenario definitions
- sim_scenarios_meta.json: provenance (hashes, rows/cols, seed strategy)

Usage:
  python scripts/simulate_scenarios.py --season 2025 --week 20 --n-sims 2000

Notes:
- Scenarios are intentionally limited (single-knob perturbations) to avoid combinatorial blowups.
- This script does not mutate existing artifacts; it only writes new scenario outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.sim_engine import simulate_mc_probs, simulate_drive_timeline

DATA_DIR = REPO_ROOT / "nfl_compare" / "data"
SCENARIOS_DIR = REPO_ROOT / "scenarios"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    # Feature deltas applied to the eval frame (per game)
    wind_open_add_mph: float = 0.0
    wx_precip_add_pct: float = 0.0
    wx_temp_add_f: float = 0.0
    home_inj_starters_add: int = 0
    away_inj_starters_add: int = 0
    neutral_site_flag: Optional[int] = None
    # Opponent/context feature deltas
    rest_days_diff_add: float = 0.0
    elo_diff_add: float = 0.0
    net_margin_diff_add: float = 0.0
    home_def_sack_rate_add: float = 0.0
    away_def_sack_rate_add: float = 0.0
    home_def_ppg_add: float = 0.0
    away_def_ppg_add: float = 0.0
    # Market line movement scenarios (affects spread_ref/total_ref derivation)
    spread_home_add: float = 0.0
    total_add: float = 0.0
    # Direct mean shifts (useful for abstract opponent or "game script" deltas)
    pred_margin_add: float = 0.0
    pred_total_add: float = 0.0
    # Per-scenario sigma overrides (volatility / game state)
    ats_sigma_override: Optional[float] = None
    total_sigma_override: Optional[float] = None
    # Global sim knobs set as env vars for this scenario
    sim_market_blend_margin: Optional[float] = None
    sim_market_blend_total: Optional[float] = None
    # Additional SIM_* env overrides (advanced)
    sim_env: Optional[Dict[str, Any]] = None
    # Optional per-game application rules. If provided, feature deltas are applied only
    # to rows matching the conditions; other rows remain unchanged.
    # Note: sigma/env overrides are still scenario-global.
    apply_when: Optional[List[Dict[str, Any]]] = None
    apply_when_mode: str = "all"  # all|any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        if not path.exists() or not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _stable_int_hash(s: str) -> int:
    # Stable across runs/processes
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit-ish


def _sanitize_id_value(v: Any) -> str:
    if v is None:
        return "na"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if np.isfinite(x) and abs(x - int(round(x))) < 1e-9:
            return str(int(round(x)))
        # stable-ish decimal string
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        s = s.replace("-", "m").replace(".", "p")
        return s
    s = str(v).strip().lower()
    # Conservative slug
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out)[:64] or "val"


def _slugify_token(s: str) -> str:
    s = (s or "").strip().lower()
    out: list[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-", ":"}:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:80] or "run"


def _eval_where(conditions: list[dict[str, Any]], ctx: dict[str, Any]) -> bool:
    """Return True if all conditions pass.

    Supported ops: eq, ne, gt, ge, lt, le, in, not_in
    """

    def _cmp(a: Any, op: str, b: Any) -> bool:
        if op == "eq":
            return a == b
        if op == "ne":
            return a != b
        if op == "gt":
            return a > b
        if op == "ge":
            return a >= b
        if op == "lt":
            return a < b
        if op == "le":
            return a <= b
        if op == "in":
            return a in (b or [])
        if op == "not_in":
            return a not in (b or [])
        raise ValueError(f"Unsupported where op: {op}")

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        field = str(cond.get("field") or cond.get("name") or "").strip()
        if not field:
            continue
        op = str(cond.get("op") or "eq").strip().lower()
        val = cond.get("value")
        if field not in ctx:
            return False
        try:
            if not _cmp(ctx[field], op, val):
                return False
        except Exception:
            return False
    return True


def _row_mask(df: pd.DataFrame, conditions: Optional[List[Dict[str, Any]]], mode: str = "all") -> pd.Series:
    """Vectorized row mask for applying conditional scenarios.

    Supported ops: eq, ne, gt, ge, lt, le, in, not_in, isnull, notnull
    Values are compared against the eval frame column values (coerced to numeric when possible).
    """
    if not conditions:
        return pd.Series([True] * len(df), index=df.index)

    mode_l = (mode or "all").strip().lower()
    use_any = mode_l == "any"

    masks: List[pd.Series] = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        field = str(cond.get("field") or cond.get("name") or "").strip()
        if not field or field not in df.columns:
            # If a referenced column is missing, condition cannot match.
            masks.append(pd.Series([False] * len(df), index=df.index))
            continue

        op = str(cond.get("op") or "eq").strip().lower()
        val = cond.get("value")

        s_raw = df[field]
        s_num = pd.to_numeric(s_raw, errors="coerce")
        use_num = s_num.notna().any()
        s = s_num if use_num else s_raw.astype(str)

        try:
            if op == "eq":
                masks.append(s == val)
            elif op == "ne":
                masks.append(s != val)
            elif op == "gt":
                masks.append(s > val)
            elif op == "ge":
                masks.append(s >= val)
            elif op == "lt":
                masks.append(s < val)
            elif op == "le":
                masks.append(s <= val)
            elif op == "in":
                masks.append(s.isin(list(val or [])))
            elif op == "not_in":
                masks.append(~s.isin(list(val or [])))
            elif op == "isnull":
                masks.append(s_raw.isna())
            elif op == "notnull":
                masks.append(~s_raw.isna())
            else:
                raise ValueError(f"Unsupported where op: {op}")
        except Exception:
            masks.append(pd.Series([False] * len(df), index=df.index))

    if not masks:
        return pd.Series([True] * len(df), index=df.index)

    out = masks[0]
    for m in masks[1:]:
        out = (out | m) if use_any else (out & m)
    return out.fillna(False)


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(grid, dict):
        raise ValueError("grid must be an object")

    prefix = str(grid.get("prefix") or "grid").strip() or "grid"
    fixed = grid.get("fixed") or {}
    if fixed is not None and not isinstance(fixed, dict):
        raise ValueError("grid.fixed must be an object")

    axes = grid.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("grid.axes must be a non-empty list")

    parsed_axes: list[dict[str, Any]] = []
    for a in axes:
        if not isinstance(a, dict):
            continue
        field = str(a.get("field") or a.get("name") or "").strip()
        if not field:
            raise ValueError("Each grid axis must have 'field' (or 'name')")
        values = a.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid axis {field}: values must be a non-empty list")
        alias = str(a.get("alias") or field).strip()
        parsed_axes.append({"field": field, "alias": alias, "values": values})

    id_template = str(grid.get("id_template") or "").strip()
    label_template = str(grid.get("label_template") or "").strip()
    where = grid.get("where") or []
    if where is not None and not isinstance(where, list):
        raise ValueError("grid.where must be a list")

    max_scenarios = int(grid.get("max_scenarios") or 1000)
    truncate = bool(grid.get("truncate") or False)

    # Deterministic cartesian expansion (axis order + value order)
    out: list[dict[str, Any]] = []

    def _recurse(i: int, ctx: dict[str, Any]):
        nonlocal out
        if i >= len(parsed_axes):
            if where and not _eval_where(where, ctx):
                return

            scenario: dict[str, Any] = {}
            scenario.update(fixed or {})

            # Apply axis fields
            for ax in parsed_axes:
                scenario[ax["field"]] = ctx[ax["field"]]

            # scenario_id/label
            fmt_ctx: dict[str, Any] = {"prefix": prefix}
            for ax in parsed_axes:
                v = ctx[ax["field"]]
                fmt_ctx[ax["field"]] = v
                fmt_ctx[ax["alias"]] = v
                fmt_ctx[f"{ax['field']}_id"] = _sanitize_id_value(v)
                fmt_ctx[f"{ax['alias']}_id"] = _sanitize_id_value(v)

            if id_template:
                sid = id_template.format(**fmt_ctx)
            else:
                parts = [prefix]
                for ax in parsed_axes:
                    parts.append(f"{ax['alias']}{_sanitize_id_value(ctx[ax['field']])}")
                sid = "_".join(parts)

            if label_template:
                label = label_template.format(**fmt_ctx)
            else:
                label = sid

            scenario["scenario_id"] = str(sid)
            scenario["label"] = str(label)
            out.append(scenario)
            return

        ax = parsed_axes[i]
        field = ax["field"]
        for v in ax["values"]:
            ctx2 = dict(ctx)
            ctx2[field] = v
            _recurse(i + 1, ctx2)

    _recurse(0, {})

    if len(out) > max_scenarios:
        if truncate:
            out = out[:max_scenarios]
        else:
            raise ValueError(f"grid expansion produced {len(out)} scenarios; exceeds max_scenarios={max_scenarios}")

    return out


@contextmanager
def _temp_environ(kv: Dict[str, Optional[str]]):
    old = {k: os.environ.get(k) for k in kv.keys()}
    try:
        for k, v in kv.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v


def _coerce_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _apply_scenario(df: pd.DataFrame, scn: Scenario) -> pd.DataFrame:
    out = df.copy()

    # Conditional application mask
    mask = _row_mask(out, scn.apply_when, mode=scn.apply_when_mode)

    # Weather knobs
    if scn.wind_open_add_mph != 0.0 and "wind_open" in out.columns:
        w = _coerce_series(out, "wind_open")
        w2 = (w.fillna(0.0) + float(scn.wind_open_add_mph)).clip(lower=0.0)
        out.loc[mask, "wind_open"] = w2.loc[mask]

    if scn.wx_precip_add_pct != 0.0 and "wx_precip_pct" in out.columns:
        p = _coerce_series(out, "wx_precip_pct")
        p2 = (p.fillna(0.0) + float(scn.wx_precip_add_pct)).clip(lower=0.0, upper=100.0)
        out.loc[mask, "wx_precip_pct"] = p2.loc[mask]

    if scn.wx_temp_add_f != 0.0 and "wx_temp_f" in out.columns:
        t = _coerce_series(out, "wx_temp_f")
        out.loc[mask, "wx_temp_f"] = (t + float(scn.wx_temp_add_f)).loc[mask]

    # Injury knobs
    if scn.home_inj_starters_add != 0 and "home_inj_starters_out" in out.columns:
        x = _coerce_series(out, "home_inj_starters_out")
        out.loc[mask, "home_inj_starters_out"] = (x.fillna(0.0) + int(scn.home_inj_starters_add)).clip(lower=0.0).loc[mask]

    if scn.away_inj_starters_add != 0 and "away_inj_starters_out" in out.columns:
        x = _coerce_series(out, "away_inj_starters_out")
        out.loc[mask, "away_inj_starters_out"] = (x.fillna(0.0) + int(scn.away_inj_starters_add)).clip(lower=0.0).loc[mask]

    # Neutral-site override
    if scn.neutral_site_flag is not None:
        if "neutral_site_flag" in out.columns:
            out.loc[mask, "neutral_site_flag"] = int(scn.neutral_site_flag)
        else:
            out["neutral_site_flag"] = int(scn.neutral_site_flag)

    # Opponent/context deltas
    if scn.rest_days_diff_add != 0.0 and "rest_days_diff" in out.columns:
        x = _coerce_series(out, "rest_days_diff")
        out.loc[mask, "rest_days_diff"] = (x + float(scn.rest_days_diff_add)).loc[mask]

    if scn.elo_diff_add != 0.0 and "elo_diff" in out.columns:
        x = _coerce_series(out, "elo_diff")
        out.loc[mask, "elo_diff"] = (x + float(scn.elo_diff_add)).loc[mask]

    if scn.net_margin_diff_add != 0.0 and "net_margin_diff" in out.columns:
        x = _coerce_series(out, "net_margin_diff")
        out.loc[mask, "net_margin_diff"] = (x + float(scn.net_margin_diff_add)).loc[mask]

    # Defensive / opponent strength knobs (best-effort on any known column variants)
    for base, delta in [
        ("home_def_sack_rate_ema", scn.home_def_sack_rate_add),
        ("away_def_sack_rate_ema", scn.away_def_sack_rate_add),
        ("home_def_sack_rate", scn.home_def_sack_rate_add),
        ("away_def_sack_rate", scn.away_def_sack_rate_add),
        ("home_def_ppg", scn.home_def_ppg_add),
        ("away_def_ppg", scn.away_def_ppg_add),
    ]:
        if delta != 0.0 and base in out.columns:
            x = _coerce_series(out, base)
            out.loc[mask, base] = (x + float(delta)).loc[mask]

    # Market line movement
    if scn.spread_home_add != 0.0:
        for c in ["spread_home", "close_spread_home"]:
            if c in out.columns:
                x = _coerce_series(out, c)
                out.loc[mask, c] = (x + float(scn.spread_home_add)).loc[mask]

    if scn.total_add != 0.0:
        for c in ["total", "close_total"]:
            if c in out.columns:
                x = _coerce_series(out, c)
                out.loc[mask, c] = (x + float(scn.total_add)).loc[mask]

    # Direct mean shifts
    if scn.pred_margin_add != 0.0 and "pred_margin" in out.columns:
        x = _coerce_series(out, "pred_margin")
        out.loc[mask, "pred_margin"] = (x + float(scn.pred_margin_add)).loc[mask]
    if scn.pred_total_add != 0.0 and "pred_total" in out.columns:
        x = _coerce_series(out, "pred_total")
        out.loc[mask, "pred_total"] = (x + float(scn.pred_total_add)).loc[mask]

    return out


def default_scenarios_v1() -> List[Scenario]:
    # Keep these single-knob for interpretability and to avoid Cartesian blowup.
    return [
        Scenario("v1_baseline", "Baseline"),
        Scenario("v1_wind_plus10", "Wind +10mph (open)", wind_open_add_mph=10.0),
        Scenario("v1_wind_plus20", "Wind +20mph (open)", wind_open_add_mph=20.0),
        Scenario("v1_precip_plus25", "Precip +25%", wx_precip_add_pct=25.0),
        Scenario("v1_precip_plus50", "Precip +50%", wx_precip_add_pct=50.0),
        Scenario("v1_cold_minus15", "Temp -15F", wx_temp_add_f=-15.0),
        Scenario("v1_inj_home_plus1", "Home starters out +1", home_inj_starters_add=1),
        Scenario("v1_inj_home_plus2", "Home starters out +2", home_inj_starters_add=2),
        Scenario("v1_inj_away_plus1", "Away starters out +1", away_inj_starters_add=1),
        Scenario("v1_inj_away_plus2", "Away starters out +2", away_inj_starters_add=2),
        Scenario("v1_neutral_site", "Neutral site", neutral_site_flag=1),
        Scenario("v1_blend_10_20", "Market blend m=0.10 t=0.20", sim_market_blend_margin=0.10, sim_market_blend_total=0.20),
        Scenario("v1_blend_25_35", "Market blend m=0.25 t=0.35", sim_market_blend_margin=0.25, sim_market_blend_total=0.35),
    ]


def default_scenarios_v2() -> List[Scenario]:
    # More coverage: weather + injuries + opponent/context + market movement + volatility.
    return [
        Scenario("v2_baseline", "Baseline"),

        # Weather
        Scenario("v2_wind_plus10", "Wind +10mph (open)", wind_open_add_mph=10.0),
        Scenario("v2_wind_plus20", "Wind +20mph (open)", wind_open_add_mph=20.0),
        Scenario("v2_precip_plus25", "Precip +25%", wx_precip_add_pct=25.0),
        Scenario("v2_precip_plus50", "Precip +50%", wx_precip_add_pct=50.0),
        Scenario("v2_cold_minus15", "Temp -15F", wx_temp_add_f=-15.0),

        # Injuries
        Scenario("v2_inj_home_plus1", "Home starters out +1", home_inj_starters_add=1),
        Scenario("v2_inj_home_plus2", "Home starters out +2", home_inj_starters_add=2),
        Scenario("v2_inj_away_plus1", "Away starters out +1", away_inj_starters_add=1),
        Scenario("v2_inj_away_plus2", "Away starters out +2", away_inj_starters_add=2),

        # Opponent/context (best-effort; only applies when those feature columns exist)
        Scenario("v2_rest_plus7", "Home rest +7 days (diff)", rest_days_diff_add=7.0),
        Scenario("v2_rest_minus7", "Home rest -7 days (diff)", rest_days_diff_add=-7.0),
        Scenario("v2_elo_plus100", "Home Elo +100 (diff)", elo_diff_add=100.0),
        Scenario("v2_elo_minus100", "Home Elo -100 (diff)", elo_diff_add=-100.0),
        Scenario("v2_pressure_plus005", "Home/away pressure +0.05", home_def_sack_rate_add=0.05, away_def_sack_rate_add=0.05),

        # Market movement (line sensitivity)
        Scenario("v2_spread_home_minus3", "Spread home -3.0", spread_home_add=-3.0),
        Scenario("v2_spread_home_plus3", "Spread home +3.0", spread_home_add=3.0),
        Scenario("v2_total_minus3", "Total -3.0", total_add=-3.0),
        Scenario("v2_total_plus3", "Total +3.0", total_add=3.0),

        # Volatility (game state / uncertainty) - sigma overrides
        Scenario("v2_sigma_low", "Low volatility", ats_sigma_override=9.0, total_sigma_override=8.0),
        Scenario("v2_sigma_high", "High volatility", ats_sigma_override=16.0, total_sigma_override=15.0),

        # Neutral site and market blending
        Scenario("v2_neutral_site", "Neutral site", neutral_site_flag=1),
        Scenario("v2_blend_10_20", "Market blend m=0.10 t=0.20", sim_market_blend_margin=0.10, sim_market_blend_total=0.20),
    ]


def _load_scenarios_from_config(path: Path) -> tuple[List[Scenario], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Scenario config must be a JSON object")

    schema_version = str(obj.get("schema_version") or obj.get("version") or "1").strip()
    cfg_name = str(obj.get("name") or path.name).strip()

    raw_scenarios = obj.get("scenarios")
    if raw_scenarios is not None and not isinstance(raw_scenarios, list):
        raise ValueError("Scenario config 'scenarios' must be a list when provided")

    raw_grid = obj.get("grid")
    grid_scenarios: list[dict[str, Any]] = []
    if raw_grid is not None:
        # Allow a single grid object or a list of grids
        if isinstance(raw_grid, dict):
            grid_scenarios.extend(_expand_grid(raw_grid))
        elif isinstance(raw_grid, list):
            for g in raw_grid:
                grid_scenarios.extend(_expand_grid(g))
        else:
            raise ValueError("Scenario config 'grid' must be an object or a list of objects")

    # Combine explicit + grid-generated
    combined: list[dict[str, Any]] = []
    if isinstance(raw_scenarios, list):
        combined.extend([r for r in raw_scenarios if isinstance(r, dict)])
    combined.extend(grid_scenarios)
    if not combined:
        raise ValueError("Scenario config must include at least one scenario via 'scenarios' and/or 'grid'")

    scenarios: List[Scenario] = []
    seen: set[str] = set()
    for r in combined:
        sid = str(r.get("scenario_id") or r.get("id") or "").strip()
        label = str(r.get("label") or sid).strip()
        if not sid:
            raise ValueError("Each scenario must include 'scenario_id' (or 'id')")
        if sid in seen:
            raise ValueError(f"Duplicate scenario_id in config: {sid}")
        seen.add(sid)

        # Pull known fields; ignore unknown keys
        kw = {k: r.get(k) for k in Scenario.__dataclass_fields__.keys() if k in r}
        kw.setdefault("scenario_id", sid)
        kw.setdefault("label", label)

        # Normalize sim_env if present
        if "sim_env" in kw and kw["sim_env"] is not None and not isinstance(kw["sim_env"], dict):
            raise ValueError(f"Scenario {sid}: sim_env must be an object/dict")

        scenarios.append(Scenario(**kw))

    # Deterministic ordering by scenario_id
    scenarios = sorted(scenarios, key=lambda s: s.scenario_id)
    meta = {
        "schema_version": schema_version,
        "name": cfg_name,
        "path": str(path),
        "sha256": _sha256_file(path),
        "scenario_count": int(len(scenarios)),
        "has_grid": raw_grid is not None,
    }
    return scenarios, meta


def _scenario_set_config_path(set_name: str) -> Optional[Path]:
    ss = (set_name or "").strip().lower()
    if not ss:
        return None
    # Prefer stable, versioned configs in repo/scenarios
    cand = SCENARIOS_DIR / f"simulate_scenarios_{ss}.json"
    return cand if cand.exists() else None


def _pick_baseline_scenario_id(scenarios: List[Scenario], probs: pd.DataFrame, config_meta: Optional[dict[str, Any]]) -> Optional[str]:
    if config_meta:
        b = config_meta.get("baseline_scenario_id")
        if isinstance(b, str) and b.strip():
            b = b.strip()
            if any(s.scenario_id == b for s in scenarios):
                return b

    # Heuristic: prefer any scenario with "baseline" in the ID.
    for s in scenarios:
        if "baseline" in s.scenario_id.lower():
            return s.scenario_id

    # Fallback: first in deterministic ordering
    if scenarios:
        return scenarios[0].scenario_id

    # Final fallback: from probs
    if isinstance(probs, pd.DataFrame) and ("scenario_id" in probs.columns):
        try:
            v = str(probs["scenario_id"].dropna().astype(str).sort_values().iloc[0])
            return v
        except Exception:
            return None
    return None


def _write_csv_with_archive(df: pd.DataFrame, fp: Path, fp_arch: Path) -> None:
    df.to_csv(fp, index=False)
    print(f"Wrote {fp}")
    try:
        df.to_csv(fp_arch, index=False)
        print(f"Wrote {fp_arch}")
    except Exception:
        pass


def _load_player_props_cache(season: int, week: int) -> pd.DataFrame:
    fp = DATA_DIR / f"player_props_{int(season)}_wk{int(week)}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def run_drive_scenarios(
    df_eval: pd.DataFrame,
    scenarios: List[Scenario],
    *,
    season: int,
    week: int,
    n_sims: int,
    seed: Optional[int],
    ats_sigma: Optional[float],
    total_sigma: Optional[float],
    props_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    base_seed: Optional[int] = seed
    if base_seed is None:
        try:
            env_seed = os.environ.get("SIM_SEED")
            if env_seed is not None and str(env_seed).strip() != "":
                base_seed = int(float(str(env_seed).strip()))
        except Exception:
            base_seed = None

    for scn in scenarios:
        df_s = _apply_scenario(df_eval, scn)

        scn_seed = None
        if base_seed is not None:
            scn_seed = int(base_seed) + _stable_int_hash(scn.scenario_id)

        env_overrides: Dict[str, Optional[str]] = {}
        if scn.sim_market_blend_margin is not None:
            env_overrides["SIM_MARKET_BLEND_MARGIN"] = str(float(scn.sim_market_blend_margin))
        if scn.sim_market_blend_total is not None:
            env_overrides["SIM_MARKET_BLEND_TOTAL"] = str(float(scn.sim_market_blend_total))
        if scn.sim_env:
            for k, v in scn.sim_env.items():
                env_overrides[str(k)] = None if v is None else str(v)

        mask = _row_mask(df_eval, scn.apply_when, mode=scn.apply_when_mode)
        need_split = bool(scn.apply_when) and (mask.sum() not in (0, len(mask)))

        if need_split:
            df_yes = df_s.loc[mask].copy()
            df_no = df_eval.loc[~mask].copy()
            parts: List[pd.DataFrame] = []
            if not df_yes.empty:
                with _temp_environ(env_overrides):
                    d_yes = simulate_drive_timeline(
                        df_yes,
                        props_df=props_df,
                        n_sims=int(n_sims),
                        ats_sigma_override=scn.ats_sigma_override if scn.ats_sigma_override is not None else ats_sigma,
                        total_sigma_override=scn.total_sigma_override if scn.total_sigma_override is not None else total_sigma,
                        seed=scn_seed,
                        data_dir=DATA_DIR,
                        draws_by_game=None,
                    )
                if d_yes is not None and not d_yes.empty:
                    parts.append(d_yes)
            if not df_no.empty:
                d_no = simulate_drive_timeline(
                    df_no,
                    props_df=props_df,
                    n_sims=int(n_sims),
                    ats_sigma_override=ats_sigma,
                    total_sigma_override=total_sigma,
                    seed=scn_seed,
                    data_dir=DATA_DIR,
                    draws_by_game=None,
                )
                if d_no is not None and not d_no.empty:
                    parts.append(d_no)
            ddf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        else:
            with _temp_environ(env_overrides):
                ddf = simulate_drive_timeline(
                    df_s,
                    props_df=props_df,
                    n_sims=int(n_sims),
                    ats_sigma_override=scn.ats_sigma_override if scn.ats_sigma_override is not None else ats_sigma,
                    total_sigma_override=scn.total_sigma_override if scn.total_sigma_override is not None else total_sigma,
                    seed=scn_seed,
                    data_dir=DATA_DIR,
                    draws_by_game=None,
                )

        if ddf is None or ddf.empty:
            continue

        ddf = ddf.copy()
        ddf.insert(0, "scenario_id", scn.scenario_id)
        ddf.insert(1, "scenario_label", scn.label)
        rows.append(ddf)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ["scenario_id", "game_id", "drive_no"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _load_predictions_week(season: int, week: int) -> pd.DataFrame:
    fp = DATA_DIR / "predictions_week.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()

    if {"season", "week"}.issubset(df.columns):
        try:
            df["season"] = pd.to_numeric(df["season"], errors="coerce")
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df = df[(df["season"] == int(season)) & (df["week"] == int(week))].copy()
        except Exception:
            pass

    # Normalize means columns
    pm_col = "pred_margin" if "pred_margin" in df.columns else ("margin_pred" if "margin_pred" in df.columns else None)
    pt_col = "pred_total" if "pred_total" in df.columns else ("total_pred" if "total_pred" in df.columns else None)
    if pm_col and pm_col != "pred_margin":
        df = df.rename(columns={pm_col: "pred_margin"})
    if pt_col and pt_col != "pred_total":
        df = df.rename(columns={pt_col: "pred_total"})

    keep = [c for c in ["game_id", "pred_margin", "pred_total", "season", "week"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep].drop_duplicates(subset=["game_id"]) if "game_id" in keep else df[keep]


def build_eval_frame(season: int, week: int) -> pd.DataFrame:
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx).copy()
    for c in ("season", "week"):
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")

    eval_mask = (feat.get("season") == int(season)) & (feat.get("week") == int(week))
    df_eval = feat[eval_mask].copy()

    # Derived convenience columns for conditional scenarios
    try:
        if "spread_home" in df_eval.columns:
            sh = pd.to_numeric(df_eval["spread_home"], errors="coerce")
            df_eval["abs_spread_home"] = sh.abs()
        if "close_spread_home" in df_eval.columns:
            shc = pd.to_numeric(df_eval["close_spread_home"], errors="coerce")
            df_eval["abs_close_spread_home"] = shc.abs()
        if "wind_open" in df_eval.columns:
            df_eval["wind_open"] = pd.to_numeric(df_eval["wind_open"], errors="coerce")
        if "wx_precip_pct" in df_eval.columns:
            df_eval["wx_precip_pct"] = pd.to_numeric(df_eval["wx_precip_pct"], errors="coerce")
        if "wx_temp_f" in df_eval.columns:
            df_eval["wx_temp_f"] = pd.to_numeric(df_eval["wx_temp_f"], errors="coerce")
    except Exception:
        pass

    # Deduplicate by game_id
    try:
        if "game_id" in df_eval.columns:
            df_eval = df_eval.sort_values(["season", "week", "game_id"]).drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass

    # Join materialized week predictions means when present (preferred)
    pred = _load_predictions_week(season, week)
    if not pred.empty and "game_id" in pred.columns and "game_id" in df_eval.columns:
        left = df_eval.copy()
        left["game_id"] = left["game_id"].astype(str)
        right = pred.copy()
        right["game_id"] = right["game_id"].astype(str)
        left = left.merge(right[[c for c in ["game_id", "pred_margin", "pred_total"] if c in right.columns]], on="game_id", how="left")
        df_eval = left

    return df_eval


def run_scenarios(
    df_eval: pd.DataFrame,
    scenarios: List[Scenario],
    *,
    season: int,
    week: int,
    n_sims: int,
    seed: Optional[int],
    ats_sigma: Optional[float],
    total_sigma: Optional[float],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    base_seed: Optional[int] = seed
    if base_seed is None:
        try:
            env_seed = os.environ.get("SIM_SEED")
            if env_seed is not None and str(env_seed).strip() != "":
                base_seed = int(float(str(env_seed).strip()))
        except Exception:
            base_seed = None

    for scn in scenarios:
        df_s = _apply_scenario(df_eval, scn)

        scn_seed = None
        if base_seed is not None:
            scn_seed = int(base_seed) + _stable_int_hash(scn.scenario_id)

        env_overrides: Dict[str, Optional[str]] = {}
        if scn.sim_market_blend_margin is not None:
            env_overrides["SIM_MARKET_BLEND_MARGIN"] = str(float(scn.sim_market_blend_margin))
        if scn.sim_market_blend_total is not None:
            env_overrides["SIM_MARKET_BLEND_TOTAL"] = str(float(scn.sim_market_blend_total))
        if scn.sim_env:
            for k, v in scn.sim_env.items():
                # Allow explicit null to clear an env var
                env_overrides[str(k)] = None if v is None else str(v)

        # If apply_when is present, apply simulation overrides only to matching games.
        # Feature deltas are already applied only to mask rows in _apply_scenario.
        mask = _row_mask(df_eval, scn.apply_when, mode=scn.apply_when_mode)
        need_split = bool(scn.apply_when) and (mask.sum() not in (0, len(mask)))

        if need_split:
            df_yes = df_s.loc[mask].copy()
            df_no = df_eval.loc[~mask].copy()

            probs_parts: List[pd.DataFrame] = []
            if not df_yes.empty:
                with _temp_environ(env_overrides):
                    p_yes = simulate_mc_probs(
                        df_yes,
                        n_sims=int(n_sims),
                        ats_sigma_override=scn.ats_sigma_override if scn.ats_sigma_override is not None else ats_sigma,
                        total_sigma_override=scn.total_sigma_override if scn.total_sigma_override is not None else total_sigma,
                        seed=scn_seed,
                        data_dir=DATA_DIR,
                        draws_by_game=None,
                    )
                if p_yes is not None and not p_yes.empty:
                    probs_parts.append(p_yes)

            if not df_no.empty:
                # Baseline sim for non-applying games (no env/sigma overrides)
                p_no = simulate_mc_probs(
                    df_no,
                    n_sims=int(n_sims),
                    ats_sigma_override=ats_sigma,
                    total_sigma_override=total_sigma,
                    seed=scn_seed,
                    data_dir=DATA_DIR,
                    draws_by_game=None,
                )
                if p_no is not None and not p_no.empty:
                    probs_parts.append(p_no)

            probs = pd.concat(probs_parts, ignore_index=True) if probs_parts else pd.DataFrame()
        else:
            # No split: apply overrides (if any) to all games for this scenario.
            with _temp_environ(env_overrides):
                probs = simulate_mc_probs(
                    df_s,
                    n_sims=int(n_sims),
                    ats_sigma_override=scn.ats_sigma_override if scn.ats_sigma_override is not None else ats_sigma,
                    total_sigma_override=scn.total_sigma_override if scn.total_sigma_override is not None else total_sigma,
                    seed=scn_seed,
                    data_dir=DATA_DIR,
                    draws_by_game=None,
                )

        if probs is None or probs.empty:
            continue

        probs = probs.copy()
        probs.insert(0, "scenario_id", scn.scenario_id)
        probs.insert(1, "scenario_label", scn.label)
        probs["scenario_wind_open_add_mph"] = float(scn.wind_open_add_mph)
        probs["scenario_wx_precip_add_pct"] = float(scn.wx_precip_add_pct)
        probs["scenario_wx_temp_add_f"] = float(scn.wx_temp_add_f)
        probs["scenario_home_inj_starters_add"] = int(scn.home_inj_starters_add)
        probs["scenario_away_inj_starters_add"] = int(scn.away_inj_starters_add)
        probs["scenario_neutral_site_flag"] = int(scn.neutral_site_flag) if scn.neutral_site_flag is not None else np.nan
        probs["scenario_sim_market_blend_margin"] = float(scn.sim_market_blend_margin) if scn.sim_market_blend_margin is not None else np.nan
        probs["scenario_sim_market_blend_total"] = float(scn.sim_market_blend_total) if scn.sim_market_blend_total is not None else np.nan

        rows.append(probs)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    try:
        out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
        out["week"] = pd.to_numeric(out.get("week"), errors="coerce")
    except Exception:
        pass

    # Deterministic ordering
    sort_cols = [c for c in ["scenario_id", "game_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate scenario-based simulation datasets")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ats-sigma", type=float, default=None)
    ap.add_argument("--total-sigma", type=float, default=None)
    ap.add_argument("--scenario-set", type=str, default="v2", help="Scenario set name (currently: v1, v2)")
    ap.add_argument("--config", type=str, default="", help="Optional JSON config file defining scenarios")
    ap.add_argument("--drives", action="store_true", help="Also generate drive-level scenario artifacts")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (defaults to nfl_compare/data/backtests/{season}_wk{week})")
    args = ap.parse_args()

    scenario_config_meta: Optional[dict[str, Any]] = None
    if args.config:
        cfg_path = Path(args.config)
        scenarios, scenario_config_meta = _load_scenarios_from_config(cfg_path)
        scenario_set_name = f"config:{cfg_path.name}"
    else:
        ss = args.scenario_set.strip().lower()
        cfg_path = _scenario_set_config_path(ss)
        if cfg_path is not None:
            scenarios, scenario_config_meta = _load_scenarios_from_config(cfg_path)
            scenario_set_name = f"set:{ss}"
        else:
            # Backward-compatible fallback to code defaults
            if ss == "v1":
                scenarios = default_scenarios_v1()
            elif ss == "v2":
                scenarios = default_scenarios_v2()
            else:
                print(f"ERROR: unknown scenario-set: {args.scenario_set}")
                return 2
            scenario_set_name = ss

    df_eval = build_eval_frame(args.season, args.week)
    if df_eval is None or df_eval.empty:
        print("No eval rows; nothing to simulate.")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else (DATA_DIR / "backtests" / f"{int(args.season)}_wk{int(args.week)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = run_scenarios(
        df_eval,
        scenarios,
        season=args.season,
        week=args.week,
        n_sims=int(args.n_sims),
        seed=args.seed,
        ats_sigma=args.ats_sigma,
        total_sigma=args.total_sigma,
    )

    if probs is None or probs.empty:
        print("No scenario results; nothing written.")
        return 0

    set_slug = _slugify_token(scenario_set_name)

    probs_fp = out_dir / "sim_probs_scenarios.csv"
    probs_arch_fp = out_dir / f"sim_probs_scenarios__{set_slug}.csv"
    _write_csv_with_archive(probs, probs_fp, probs_arch_fp)

    # Emit per-scenario inputs dataset for downstream scenario modeling (best-effort).
    try:
        # Minimal but useful set of columns: keys + mean/line + key features sim_engine consumes
        cols_wanted = [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "pred_margin",
            "pred_total",
            "spread_home",
            "close_spread_home",
            "total",
            "close_total",
            "wind_open",
            "wx_wind_mph",
            "wx_precip_pct",
            "wx_temp_f",
            "roof_closed_flag",
            "roof_open_flag",
            "neutral_site_flag",
            "rest_days_diff",
            "elo_diff",
            "net_margin_diff",
            "home_inj_starters_out",
            "away_inj_starters_out",
            "home_def_ppg",
            "away_def_ppg",
            "home_def_sack_rate_ema",
            "away_def_sack_rate_ema",
            "home_def_sack_rate",
            "away_def_sack_rate",
        ]
        frames: List[pd.DataFrame] = []
        for scn in scenarios:
            df_s = _apply_scenario(df_eval, scn)
            applies = _row_mask(df_eval, scn.apply_when, mode=scn.apply_when_mode)
            keep = [c for c in cols_wanted if c in df_s.columns]
            base = df_s[keep].copy() if keep else pd.DataFrame(index=df_s.index)
            base.insert(0, "scenario_id", scn.scenario_id)
            base.insert(1, "scenario_label", scn.label)
            base["scenario_applies"] = applies.astype(int).values
            # Knob columns (so the dataset is self-contained)
            base["scenario_wind_open_add_mph"] = float(scn.wind_open_add_mph)
            base["scenario_wx_precip_add_pct"] = float(scn.wx_precip_add_pct)
            base["scenario_wx_temp_add_f"] = float(scn.wx_temp_add_f)
            base["scenario_home_inj_starters_add"] = int(scn.home_inj_starters_add)
            base["scenario_away_inj_starters_add"] = int(scn.away_inj_starters_add)
            base["scenario_neutral_site_flag"] = int(scn.neutral_site_flag) if scn.neutral_site_flag is not None else np.nan
            base["scenario_rest_days_diff_add"] = float(scn.rest_days_diff_add)
            base["scenario_elo_diff_add"] = float(scn.elo_diff_add)
            base["scenario_net_margin_diff_add"] = float(scn.net_margin_diff_add)
            base["scenario_home_def_sack_rate_add"] = float(scn.home_def_sack_rate_add)
            base["scenario_away_def_sack_rate_add"] = float(scn.away_def_sack_rate_add)
            base["scenario_spread_home_add"] = float(scn.spread_home_add)
            base["scenario_total_add"] = float(scn.total_add)
            base["scenario_pred_margin_add"] = float(scn.pred_margin_add)
            base["scenario_pred_total_add"] = float(scn.pred_total_add)
            base["scenario_ats_sigma_override"] = float(scn.ats_sigma_override) if scn.ats_sigma_override is not None else np.nan
            base["scenario_total_sigma_override"] = float(scn.total_sigma_override) if scn.total_sigma_override is not None else np.nan
            base["scenario_apply_when_mode"] = str(scn.apply_when_mode or "all")
            base["scenario_apply_when"] = json.dumps(scn.apply_when, sort_keys=True) if scn.apply_when else ""
            frames.append(base)

        inputs_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not inputs_df.empty:
            sort_cols = [c for c in ["scenario_id", "game_id"] if c in inputs_df.columns]
            if sort_cols:
                inputs_df = inputs_df.sort_values(sort_cols).reset_index(drop=True)
            inputs_fp = out_dir / "sim_scenario_inputs.csv"
            inputs_arch_fp = out_dir / f"sim_scenario_inputs__{set_slug}.csv"
            _write_csv_with_archive(inputs_df, inputs_fp, inputs_arch_fp)
    except Exception as e:
        print(f"WARN: failed writing sim_scenario_inputs.csv: {e}")

    scenarios_fp = out_dir / "sim_scenarios.json"
    scenarios_fp.write_text(json.dumps([asdict(s) for s in scenarios], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {scenarios_fp}")

    scenarios_arch_fp = out_dir / f"sim_scenarios__{set_slug}.json"
    try:
        scenarios_arch_fp.write_text(json.dumps([asdict(s) for s in scenarios], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {scenarios_arch_fp}")
    except Exception:
        pass

    meta = {
        "created_utc": _utc_now_iso(),
        "season": int(args.season),
        "week": int(args.week),
        "n_sims": int(args.n_sims),
        "seed": int(args.seed) if args.seed is not None else None,
        "seed_strategy": "seed + sha256(scenario_id)[:8] when seed is provided (or SIM_SEED env)",
        "scenario_set": scenario_set_name,
        "scenario_set_slug": set_slug,
        "scenario_config": scenario_config_meta,
        "inputs": {
            "predictions_week_csv": {
                "path": str(DATA_DIR / "predictions_week.csv"),
                "sha256": _sha256_file(DATA_DIR / "predictions_week.csv"),
            },
        },
        "output": {
            "sim_probs_scenarios_csv": {
                "path": str(probs_fp),
                "sha256": _sha256_file(probs_fp),
                "rows": int(probs.shape[0]),
                "cols": int(probs.shape[1]),
            }
        },
    }

    # Include archive locations (best-effort)
    meta["output"]["archive"] = {
        "sim_probs_scenarios_csv": str(probs_arch_fp),
        "sim_scenarios_json": str(scenarios_arch_fp),
    }

    # Add scenario inputs dataset if present
    try:
        inputs_fp = out_dir / "sim_scenario_inputs.csv"
        if inputs_fp.exists():
            try:
                df0 = pd.read_csv(inputs_fp)
                meta["output"]["sim_scenario_inputs_csv"] = {
                    "path": str(inputs_fp),
                    "sha256": _sha256_file(inputs_fp),
                    "rows": int(df0.shape[0]),
                    "cols": int(df0.shape[1]),
                }
            except Exception:
                meta["output"]["sim_scenario_inputs_csv"] = {
                    "path": str(inputs_fp),
                    "sha256": _sha256_file(inputs_fp),
                }
    except Exception:
        pass

    # Additional derived datasets: per-game deltas vs baseline + scenario summary
    deltas_fp = out_dir / "sim_scenario_deltas.csv"
    deltas_arch_fp = out_dir / f"sim_scenario_deltas__{set_slug}.csv"
    summary_fp = out_dir / "sim_scenario_summary.csv"
    summary_arch_fp = out_dir / f"sim_scenario_summary__{set_slug}.csv"
    baseline_id = _pick_baseline_scenario_id(scenarios, probs, scenario_config_meta)
    meta["baseline_scenario_id"] = baseline_id

    try:
        if baseline_id and ("scenario_id" in probs.columns) and ("game_id" in probs.columns):
            base = probs[probs["scenario_id"].astype(str) == str(baseline_id)].copy()
            if not base.empty:
                key = ["game_id"]
                cols_base = [
                    c
                    for c in [
                        "pred_margin",
                        "pred_total",
                        "spread_ref",
                        "total_ref",
                        "home_points_mean",
                        "away_points_mean",
                        "total_points_mean",
                        "prob_home_win_mc",
                        "prob_home_cover_mc",
                        "prob_over_total_mc",
                    ]
                    if c in base.columns
                ]
                base = base[[*key, *cols_base]].rename(columns={c: f"baseline_{c}" for c in cols_base})

                work = probs.merge(base, on=key, how="left")
                # Attach applies mask if inputs exists
                try:
                    inputs_fp0 = out_dir / "sim_scenario_inputs.csv"
                    if inputs_fp0.exists():
                        inp = pd.read_csv(inputs_fp0)
                        if {"scenario_id", "game_id", "scenario_applies"}.issubset(inp.columns):
                            work = work.merge(inp[["scenario_id", "game_id", "scenario_applies"]], on=["scenario_id", "game_id"], how="left")
                except Exception:
                    pass

                delta_cols = []
                for c in cols_base:
                    bcol = f"baseline_{c}"
                    dcol = f"delta_{c}"
                    if bcol in work.columns and c in work.columns:
                        work[dcol] = pd.to_numeric(work[c], errors="coerce") - pd.to_numeric(work[bcol], errors="coerce")
                        delta_cols.append(dcol)

                # Compact delta dataset
                keep = [c for c in ["season", "week", "scenario_id", "scenario_label", "game_id", "home_team", "away_team", "scenario_applies"] if c in work.columns]
                keep += [c for c in delta_cols if c in work.columns]
                deltas = work[keep].copy() if keep else pd.DataFrame()
                if not deltas.empty:
                    sort_cols = [c for c in ["scenario_id", "game_id"] if c in deltas.columns]
                    if sort_cols:
                        deltas = deltas.sort_values(sort_cols).reset_index(drop=True)
                    _write_csv_with_archive(deltas, deltas_fp, deltas_arch_fp)
                    meta["output"]["sim_scenario_deltas_csv"] = {
                        "path": str(deltas_fp),
                        "sha256": _sha256_file(deltas_fp),
                        "rows": int(deltas.shape[0]),
                        "cols": int(deltas.shape[1]),
                        "archive": str(deltas_arch_fp),
                    }

                # Summary
                if not deltas.empty and "scenario_id" in deltas.columns:
                    sum_cols = [c for c in deltas.columns if c.startswith("delta_")]
                    grp = deltas.groupby("scenario_id", dropna=False)
                    summary = grp[sum_cols].mean(numeric_only=True).reset_index() if sum_cols else grp.size().reset_index(name="n")
                    try:
                        if "scenario_applies" in deltas.columns:
                            applies = grp["scenario_applies"].sum(min_count=1).reset_index().rename(columns={"scenario_applies": "n_games_applies"})
                            summary = summary.merge(applies, on="scenario_id", how="left")
                        counts = grp.size().reset_index(name="n_games")
                        summary = summary.merge(counts, on="scenario_id", how="left")
                    except Exception:
                        pass
                    try:
                        labels = probs[["scenario_id", "scenario_label"]].drop_duplicates(subset=["scenario_id"])
                        summary = summary.merge(labels, on="scenario_id", how="left")
                    except Exception:
                        pass
                    _write_csv_with_archive(summary, summary_fp, summary_arch_fp)
                    meta["output"]["sim_scenario_summary_csv"] = {
                        "path": str(summary_fp),
                        "sha256": _sha256_file(summary_fp),
                        "rows": int(summary.shape[0]),
                        "cols": int(summary.shape[1]),
                        "archive": str(summary_arch_fp),
                    }
    except Exception as e:
        print(f"WARN: failed writing scenario deltas/summary: {e}")

    # Drive-level scenario artifacts (optional)
    if bool(args.drives):
        try:
            props_df = _load_player_props_cache(args.season, args.week)
            drives = run_drive_scenarios(
                df_eval,
                scenarios,
                season=args.season,
                week=args.week,
                n_sims=int(args.n_sims),
                seed=args.seed,
                ats_sigma=args.ats_sigma,
                total_sigma=args.total_sigma,
                props_df=props_df if not props_df.empty else None,
            )

            if drives is not None and not drives.empty:
                drives_fp = out_dir / "sim_drives_scenarios.csv"
                drives_arch_fp = out_dir / f"sim_drives_scenarios__{set_slug}.csv"
                _write_csv_with_archive(drives, drives_fp, drives_arch_fp)
                meta["output"]["sim_drives_scenarios_csv"] = {
                    "path": str(drives_fp),
                    "sha256": _sha256_file(drives_fp),
                    "rows": int(drives.shape[0]),
                    "cols": int(drives.shape[1]),
                    "archive": str(drives_arch_fp),
                }

                # Summary per (scenario_id, game_id)
                sum_cols = [c for c in ["drive_sec_mean", "drive_pts_mean", "p_drive_score", "p_drive_fg", "p_drive_td"] if c in drives.columns]
                keys = [c for c in ["scenario_id", "scenario_label", "season", "week", "game_id", "home_team", "away_team"] if c in drives.columns]
                if "scenario_id" in drives.columns and "game_id" in drives.columns:
                    g = drives.groupby(["scenario_id", "game_id"], dropna=False)
                    summ = g[sum_cols].mean(numeric_only=True).reset_index() if sum_cols else g.size().reset_index(name="n_drives")
                    # Attach basic identifiers
                    try:
                        ident = drives[keys].drop_duplicates(subset=["scenario_id", "game_id"]) if keys else None
                        if ident is not None and not ident.empty:
                            summ = summ.merge(ident, on=["scenario_id", "game_id"], how="left")
                    except Exception:
                        pass
                    # Include drive totals if available
                    try:
                        for c in ["drives_home", "drives_away", "drives_total"]:
                            if c in drives.columns:
                                x = g[c].max().reset_index().rename(columns={c: c})
                                summ = summ.merge(x, on=["scenario_id", "game_id"], how="left")
                    except Exception:
                        pass
                    summ_fp = out_dir / "sim_drives_scenarios_summary.csv"
                    summ_arch_fp = out_dir / f"sim_drives_scenarios_summary__{set_slug}.csv"
                    _write_csv_with_archive(summ, summ_fp, summ_arch_fp)
                    meta["output"]["sim_drives_scenarios_summary_csv"] = {
                        "path": str(summ_fp),
                        "sha256": _sha256_file(summ_fp),
                        "rows": int(summ.shape[0]),
                        "cols": int(summ.shape[1]),
                        "archive": str(summ_arch_fp),
                    }
        except Exception as e:
            print(f"WARN: failed writing drive-level scenario artifacts: {e}")

    meta_fp = out_dir / "sim_scenarios_meta.json"
    meta_fp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {meta_fp}")

    meta_arch_fp = out_dir / f"sim_scenarios_meta__{set_slug}.json"
    try:
        meta_arch_fp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {meta_arch_fp}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
