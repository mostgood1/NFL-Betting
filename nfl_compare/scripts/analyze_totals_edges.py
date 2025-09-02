import os
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def ev_from_prob(p: float, dec: float) -> float:
    return p * (dec - 1.0) - (1.0 - p)


def main():
    base = Path(__file__).resolve().parents[1]
    pred_path = base / "data" / "predictions.csv"
    if not pred_path.exists():
        print({"error": f"predictions.csv not found at {pred_path}"})
        return

    df = pd.read_csv(pred_path)

    # Parameters (mirror app defaults)
    NFL_TOTAL_SIGMA = float(os.getenv("NFL_TOTAL_SIGMA", "10.0"))
    PROB_SHRINK = float(os.getenv("RECS_PROB_SHRINK", "0.50"))
    DEC_110 = 1.909090909
    MIN_EV_PCT = float(os.getenv("RECS_MIN_EV_PCT", "5.0"))

    # Pick best available market total column
    mcol = None
    for c in ["market_total", "total", "open_total", "close_total"]:
        if c in df.columns and df[c].notna().any():
            mcol = c
            break
    if mcol is None:
        print({"error": "No market total column found"})
        return

    evs = []
    details = []
    for _, r in df.iterrows():
        pt = r.get("pred_total")
        mt = r.get(mcol)
        if pd.notna(pt) and pd.notna(mt):
            edge = float(pt) - float(mt)
            p_over = sigmoid(edge / NFL_TOTAL_SIGMA)
            p_over = 0.5 + (p_over - 0.5) * (1.0 - PROB_SHRINK)
            e_over = ev_from_prob(p_over, DEC_110)
            e_under = ev_from_prob(1.0 - p_over, DEC_110)
            best = max(e_over, e_under)
            evs.append(best * 100.0)
            details.append({
                "home": r.get("home_team"),
                "away": r.get("away_team"),
                "pred_total": float(pt),
                "market_total": float(mt),
                "edge_pts": edge,
                "best_ev_pct": best * 100.0,
            })

    if not evs:
        print({
            "games_with_total": 0,
            "ev_ge_0": 0,
            "ev_ge_min": 0,
            "median_ev_pct": None,
            "p80_ev_pct": None,
            "market_col": mcol,
        })
        return

    evs_sorted = sorted(evs, reverse=True)
    summary = {
        "games_with_total": len(evs),
        "ev_ge_0": sum(1 for e in evs if e > 0),
        "ev_ge_min": sum(1 for e in evs if e >= MIN_EV_PCT),
        "median_ev_pct": float(np.median(evs)),
        "p80_ev_pct": float(np.percentile(evs, 80)),
        "market_col": mcol,
        "top5": details and sorted(details, key=lambda d: d["best_ev_pct"], reverse=True)[:5],
    }
    print(summary)


if __name__ == "__main__":
    main()
