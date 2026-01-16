from __future__ import annotations

"""
Backtest player prop projections across a range of weeks in a season.

For each week, we compute projections via nfl_compare.src.player_props.compute_player_props,
then reconcile against actual weekly stats using nfl_compare.src.reconciliation helpers.
We aggregate mean absolute error (MAE) by position and metric across all evaluated weeks.

Usage:
  python scripts/backtest_props.py --season 2025 --start-week 1 --end-week 9 \
    --out nfl_compare/data/backtest_props_2025_wk9.csv

Outputs:
  - A CSV with MAE by position and metric across the selected span
  - Optional per-week summary printed to console

Dependencies:
  - nfl_data_py for actuals via reconciliation helpers
  - Existing priors and PBP parquet files improve projection quality but are not required
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nfl_compare.src.player_props import compute_player_props
try:
    from nfl_compare.src.reconciliation import reconcile_props as _reconcile, summarize_errors as _summarize
except Exception:
    # Fallback to script implementation if package path doesn't expose them
    from scripts.reconcile_props_vs_actuals import reconcile_props as _reconcile  # type: ignore
    from scripts.reconcile_props_vs_actuals import summarize_errors as _summarize  # type: ignore

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _week_summary(df: pd.DataFrame) -> pd.DataFrame:
    comp_cols = [
        "pass_attempts","pass_yards","pass_tds","interceptions",
        "rush_attempts","rush_yards","rush_tds",
        "targets","receptions","rec_yards","rec_tds",
    ]
    rows: List[Dict] = []
    for pos in ["QB","RB","WR","TE"]:
        sub = df[df.get("position", "").astype(str).str.upper() == pos].copy()
        if sub.empty:
            continue
        rec: Dict[str, float | str | int] = {"position": pos, "n": int(len(sub))}
        for c in comp_cols:
            c_act = f"{c}_act"
            if c in sub.columns and c_act in sub.columns:
                pred = pd.to_numeric(sub[c], errors="coerce")
                act = pd.to_numeric(sub[c_act], errors="coerce")
                err = (pred - act).abs()
                rec[f"{c}_MAE"] = float(np.nanmean(err)) if len(err) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def backtest(season: int, start_week: int, end_week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weeks = list(range(int(start_week), int(end_week) + 1))
    all_rows: List[pd.DataFrame] = []
    weekly: List[pd.DataFrame] = []
    for wk in weeks:
        print(f"[props-backtest] Season {season} Week {wk}...")
        # Projections
        proj = compute_player_props(season=season, week=wk)
        if proj is None or proj.empty:
            print(f"  no projections for week {wk}; skipping")
            continue
        # Reconcile vs actuals
        try:
            merged = _reconcile(int(season), int(wk))
        except Exception as e:
            print(f"  reconcile failed for week {wk}: {e}")
            continue
        if merged is None or merged.empty:
            print(f"  no reconciliation rows for week {wk}; skipping")
            continue
        summ = _summarize(merged)
        if summ is not None and not summ.empty:
            summ["season"] = int(season)
            summ["week"] = int(wk)
            all_rows.append(summ)
            # Keep per-week copies for standardized output
            weekly.append(summ.copy())
            try:
                print("  MAE by position (selected):")
                print(summ.to_string(index=False))
            except Exception:
                pass
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    res = pd.concat(all_rows, ignore_index=True)
    # Aggregate across weeks: mean of MAEs per position
    agg_cols = [c for c in res.columns if c.endswith("_MAE")]
    group = res.groupby("position", as_index=False)[agg_cols].mean(numeric_only=True)
    # Add counts/weeks evaluated
    # Robust count of rows per position without clobbering the 'position' column name
    # Using size().reset_index(name=...) avoids duplicate column renaming issues
    cnt = res.groupby("position").size().reset_index(name="weeks")
    out = group.merge(cnt, on="position", how="left")
    weekly_df = pd.concat(weekly, ignore_index=True) if weekly else pd.DataFrame()
    return out, weekly_df


def main():
    ap = argparse.ArgumentParser(description="Backtest player prop projections over a range of weeks")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="Output CSV path for summary")
    ap.add_argument("--out-dir", type=str, default=None, help="Standardized output directory (backtests/<season>_wk<end_week>/)")
    args = ap.parse_args()

    summ_df, weekly_df = backtest(args.season, args.start_week, args.end_week)
    if summ_df is None or summ_df.empty:
        print("No results; nothing written.")
        return 0

    # Standardized output dir
    std_dir: Optional[Path] = None
    if args.out_dir:
        std_dir = Path(args.out_dir)
    else:
        std_dir = DATA_DIR / "backtests" / f"{args.season}_wk{args.end_week}"
    try:
        std_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        std_dir = None

    # Legacy single CSV if --out provided
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(out_fp, index=False)
        print(f"Wrote props backtest summary -> {out_fp}")

    # Standardized files
    if std_dir is not None:
        try:
            summ_fp = std_dir / "props_summary.csv"
            summ_df.to_csv(summ_fp, index=False)
            print(f"Wrote {summ_fp}")
        except Exception as e:
            print(f"Failed writing props_summary.csv: {e}")
        try:
            if weekly_df is not None and not weekly_df.empty:
                wk_fp = std_dir / "props_weekly.csv"
                weekly_df.to_csv(wk_fp, index=False)
                print(f"Wrote {wk_fp}")
        except Exception as e:
            print(f"Failed writing props_weekly.csv: {e}")
        # Extend metrics.json
        try:
            import json as _json
            metrics_fp = std_dir / "metrics.json"
            metrics = {}
            if metrics_fp.exists():
                try:
                    metrics = _json.loads(metrics_fp.read_text(encoding='utf-8'))
                except Exception:
                    metrics = {}
            metrics['props'] = summ_df.mean(numeric_only=True).to_dict() if summ_df is not None else {}
            with open(metrics_fp, 'w', encoding='utf-8') as f:
                _json.dump(metrics, f, indent=2)
            print(f"Updated {metrics_fp}")
        except Exception as e:
            print(f"Failed updating metrics.json: {e}")

    try:
        print(summ_df.to_string(index=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
