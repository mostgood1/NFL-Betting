from __future__ import annotations

"""
Summarize recommendation backtest details filtered by selected-side probability thresholds.

Usage:
  python scripts/summarize_recs_by_prob.py --details nfl_compare/data/backtests/2025_wk17/recs_backtest_details.csv \
    --p-ats 0.58 --p-total 0.60
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def summarize(details_fp: Path, p_ats: float, p_total: float) -> Dict[str, Any]:
    df = pd.read_csv(details_fp)
    out: Dict[str, Any] = {}
    def _metrics(sub: pd.DataFrame) -> Dict[str, Any]:
        if sub is None or sub.empty:
            return {"rows": 0}
        res = sub.get("result").astype(str).str.upper()
        wins = int(res.eq("WIN").sum())
        losses = int(res.eq("LOSS").sum())
        pushes = int(res.eq("PUSH").sum())
        denom = max(1, wins + losses)
        return {
            "rows": int(len(sub)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": float(wins) / float(denom),
            "ev_mean": float(pd.to_numeric(sub.get("ev_units"), errors="coerce").mean() or 0.0),
        }
    # Baseline
    out["baseline_spread"] = _metrics(df[df["type"].astype(str).str.upper().eq("SPREAD")])
    out["baseline_total"] = _metrics(df[df["type"].astype(str).str.upper().eq("TOTAL")])
    # Filtered
    ats = df[df["type"].astype(str).str.upper().eq("SPREAD")]
    ats_f = ats[pd.to_numeric(ats.get("prob_selected"), errors="coerce") >= float(p_ats)]
    out["filtered_spread"] = _metrics(ats_f)
    tot = df[df["type"].astype(str).str.upper().eq("TOTAL")]
    tot_f = tot[pd.to_numeric(tot.get("prob_selected"), errors="coerce") >= float(p_total)]
    out["filtered_total"] = _metrics(tot_f)
    out["thresholds"] = {"p_ats": float(p_ats), "p_total": float(p_total)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize recs by selected-side probability thresholds")
    ap.add_argument("--details", type=str, required=True)
    ap.add_argument("--p-ats", type=float, default=0.58)
    ap.add_argument("--p-total", type=float, default=0.60)
    args = ap.parse_args()

    res = summarize(Path(args.details), args.p_ats, args.p_total)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
