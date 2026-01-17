"""Compare locked predictions vs materialized weekly predictions.

This is a debugging aid to quantify how much predictions_locked.csv differs from
predictions_week.csv for a season/week range.

Usage:
  python scripts/compare_predictions_locked_vs_week.py --season 2025 --start-week 12 --end-week 17

Outputs:
  nfl_compare/data/reports/pred_compare_<season>_wk<start>-<end>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"


def _coerce_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _keyed_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # Prefer game_id when present on both.
    if "game_id" in left.columns and "game_id" in right.columns:
        l = left.copy()
        r = right.copy()
        l["game_id"] = l["game_id"].astype(str)
        r["game_id"] = r["game_id"].astype(str)
        return l.merge(r, on="game_id", how="outer", suffixes=("_locked", "_week"))

    # Fallback: season/week/home/away
    need = {"season", "week", "home_team", "away_team"}
    if need.issubset(left.columns) and need.issubset(right.columns):
        return left.merge(right, on=["season", "week", "home_team", "away_team"], how="outer", suffixes=("_locked", "_week"))

    # Last resort: concat with no join
    return pd.concat([left.add_suffix("_locked"), right.add_suffix("_week")], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare predictions_locked.csv vs predictions_week.csv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    season = int(args.season)
    w0 = int(args.start_week)
    w1 = int(args.end_week)

    locked_fp = DATA_DIR / "predictions_locked.csv"
    week_fp = DATA_DIR / "predictions_week.csv"

    if not locked_fp.exists():
        raise SystemExit(f"Missing {locked_fp}")
    if not week_fp.exists():
        raise SystemExit(f"Missing {week_fp}")

    locked = _coerce_num(pd.read_csv(locked_fp), ["season", "week"])
    week = _coerce_num(pd.read_csv(week_fp), ["season", "week"])

    locked = locked[(locked.get("season") == season) & (locked.get("week").between(w0, w1))].copy()
    week = week[(week.get("season") == season) & (week.get("week").between(w0, w1))].copy()

    # Keep a compact column set (but preserve what's available)
    keep = ["game_id", "season", "week", "home_team", "away_team", "pred_home_points", "pred_away_points", "pred_total", "pred_margin", "prob_home_win"]
    locked_k = locked[[c for c in keep if c in locked.columns]].copy()
    week_k = week[[c for c in keep if c in week.columns]].copy()

    merged = _keyed_merge(locked_k, week_k)

    # Compute deltas where possible
    def _delta(col: str) -> None:
        a = pd.to_numeric(merged.get(f"{col}_locked"), errors="coerce")
        b = pd.to_numeric(merged.get(f"{col}_week"), errors="coerce")
        if a is None or b is None:
            return
        merged[f"delta_{col}"] = b - a
        merged[f"abs_delta_{col}"] = (b - a).abs()

    for c in ["pred_total", "pred_margin", "prob_home_win", "pred_home_points", "pred_away_points"]:
        if f"{c}_locked" in merged.columns and f"{c}_week" in merged.columns:
            _delta(c)

    # Flag obvious inconsistencies
    try:
        merged["flag_prob_shift_gt_0p2"] = (pd.to_numeric(merged.get("abs_delta_prob_home_win"), errors="coerce") > 0.20)
    except Exception:
        pass
    try:
        merged["flag_margin_shift_gt_3"] = (pd.to_numeric(merged.get("abs_delta_pred_margin"), errors="coerce") > 3.0)
    except Exception:
        pass

    out_dir = DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = Path(args.out) if args.out else (out_dir / f"pred_compare_{season}_wk{w0}-{w1}.csv")
    merged.to_csv(out_fp, index=False)
    print(f"Wrote {len(merged)} rows to {out_fp}")

    # Quick console summary
    try:
        n = len(merged)
        prob_flag = int(merged.get("flag_prob_shift_gt_0p2", pd.Series([False] * n)).sum())
        mar_flag = int(merged.get("flag_margin_shift_gt_3", pd.Series([False] * n)).sum())
        print(f"Flags: prob_shift_gt_0.20={prob_flag}/{n}, margin_shift_gt_3.0={mar_flag}/{n}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
