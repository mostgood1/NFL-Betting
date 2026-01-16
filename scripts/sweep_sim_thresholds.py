from __future__ import annotations

"""
Sweep MC probability thresholds for ML, ATS, and Total on sim_backtest_details.csv.

Find the smallest threshold that achieves ≥ target accuracy and report coverage.

Usage:
  python scripts/sweep_sim_thresholds.py --details nfl_compare/data/backtests/2025_wk18/sim_backtest_details.csv \
    --target 0.55 --out nfl_compare/data/backtests/2025_wk18/sim_thresholds_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _compute_outcomes(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    margin_act = pd.to_numeric(df.get("margin_actual"), errors="coerce")
    total_act = pd.to_numeric(df.get("total_actual"), errors="coerce")
    spread_ref = pd.to_numeric(df.get("spread_ref"), errors="coerce") if "spread_ref" in df.columns else pd.Series(index=df.index, dtype=float)
    total_ref = pd.to_numeric(df.get("total_ref"), errors="coerce") if "total_ref" in df.columns else pd.Series(index=df.index, dtype=float)
    y_ml = (margin_act > 0).astype(float)
    y_ats = (margin_act + spread_ref > 0).astype(float)
    y_tot = (total_act > total_ref).astype(float)
    return y_ml, y_ats, y_tot


def _acc_cov(prob: pd.Series, y: pd.Series, thresh: float) -> Dict[str, float]:
    p = pd.to_numeric(prob, errors="coerce").astype(float)
    yy = pd.to_numeric(y, errors="coerce").astype(float)
    m = p.notna() & yy.notna() & (p >= float(thresh))
    n = int(m.sum())
    if n == 0:
        return {"acc": float("nan"), "coverage": 0.0, "n": 0}
    acc = float(((p[m] > 0.5).astype(float) == yy[m]).mean())
    cov = float(n) / float(len(yy)) if len(yy) else 0.0
    return {"acc": acc, "coverage": cov, "n": n}


def sweep(details_fp: Path, target_acc: float = 0.55, min_thresh: float = 0.55, max_thresh: float = 0.70, step: float = 0.01) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(details_fp)
    y_ml, y_ats, y_tot = _compute_outcomes(df)
    p_ml = pd.to_numeric(df.get("prob_home_win_mc"), errors="coerce")
    p_ats = pd.to_numeric(df.get("prob_home_cover_mc"), errors="coerce")
    p_tot = pd.to_numeric(df.get("prob_over_total_mc"), errors="coerce")

    thresholds = np.round(np.arange(float(min_thresh), float(max_thresh) + 1e-9, float(step)), 2)
    res: Dict[str, Dict[str, float]] = {}
    for name, p, y in [("ML", p_ml, y_ml), ("ATS", p_ats, y_ats), ("TOTAL", p_tot, y_tot)]:
        best = None
        for th in thresholds:
            r = _acc_cov(p, y, th)
            if np.isfinite(r["acc"]) and r["acc"] >= float(target_acc):
                best = {"threshold": float(th), "acc": r["acc"], "coverage": r["coverage"], "n": r["n"]}
                break  # smallest threshold achieving target
        if best is None:
            # Report best observed accuracy even if it doesn't reach target
            candidates = [{"threshold": float(th), **_acc_cov(p, y, th)} for th in thresholds]
            best = max(candidates, key=lambda x: (x["acc"], x["coverage"]))
        res[name] = best
    return res


def main():
    ap = argparse.ArgumentParser(description="Sweep MC prob thresholds to hit target accuracy")
    ap.add_argument("--details", type=str, required=True)
    ap.add_argument("--target", type=float, default=0.55)
    ap.add_argument("--min", dest="min_thresh", type=float, default=0.55)
    ap.add_argument("--max", dest="max_thresh", type=float, default=0.70)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    details_fp = Path(args.details)
    out = sweep(details_fp, target_acc=args.target, min_thresh=args.min_thresh, max_thresh=args.max_thresh, step=args.step)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(Path(args.out), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
