from __future__ import annotations

"""
Summarize recommendation backtest details filtered by Monte Carlo simulation probability thresholds.

Joins sim_probs.csv onto recs_backtest_details.csv via season/week/game_id and computes
selected-side probabilities:
- SPREAD: if selection team == home_team, use prob_home_cover_mc; else 1 - prob_home_cover_mc
- TOTAL: if selection starts with Over, use prob_over_total_mc; if Under, use 1 - prob_over_total_mc

Usage:
  python scripts/summarize_recs_by_prob_mc.py --details nfl_compare/data/backtests/2025_wk17/recs_backtest_details.csv \
    --sim nfl_compare/data/backtests/2025_wk17/sim_probs.csv --p-ats 0.55 --p-total 0.60
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def _selected_prob_spread(row: pd.Series) -> float | None:
    p = pd.to_numeric(row.get("prob_home_cover_mc"), errors="coerce")
    if pd.isna(p):
        return None
    sel = str(row.get("selection") or "")
    home = str(row.get("home_team") or "")
    if not sel or not home:
        return None
    # selection format: "Team +/-X.X"
    try:
        sel_team = sel.rsplit(" ", 1)[0]
    except Exception:
        sel_team = sel
    return float(p) if sel_team.strip().lower() == home.strip().lower() else float(1.0 - float(p))


def _selected_prob_total(row: pd.Series) -> float | None:
    p = pd.to_numeric(row.get("prob_over_total_mc"), errors="coerce")
    if pd.isna(p):
        return None
    sel = str(row.get("selection") or "")
    if sel.lower().startswith("over"):
        return float(p)
    if sel.lower().startswith("under"):
        return float(1.0 - float(p))
    return None


def summarize(details_fp: Path, sim_fp: Path, p_ats: float, p_total: float) -> Dict[str, Any]:
    df = pd.read_csv(details_fp)
    # Prefer MC probabilities already attached to details; fallback to sim file join
    # Consider MC columns present if either market has attached MC probabilities
    have_mc_cols = (
        ("prob_home_cover_mc" in df.columns and not df["prob_home_cover_mc"].isna().all())
        or ("prob_over_total_mc" in df.columns and not df["prob_over_total_mc"].isna().all())
    )
    if have_mc_cols:
        work = df.copy()
    else:
        sim = pd.read_csv(sim_fp)
        # Join on season/week/game_id
        key_cols = ["season", "week", "game_id"]
        for c in key_cols:
            df[c] = df.get(c)
            sim[c] = sim.get(c)
        cols_to_merge = [c for c in ["prob_home_cover_mc", "prob_over_total_mc"] if c in sim.columns]
        work = df.merge(sim[[*key_cols, *cols_to_merge]], on=key_cols, how="left")
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
    out["baseline_spread"] = _metrics(work[work["type"].astype(str).str.upper().eq("SPREAD")])
    out["baseline_total"] = _metrics(work[work["type"].astype(str).str.upper().eq("TOTAL")])

    # MC-filtered
    ats = work[work["type"].astype(str).str.upper().eq("SPREAD")].copy()
    ats["prob_selected_mc"] = ats.apply(_selected_prob_spread, axis=1)
    ats_f = ats[pd.to_numeric(ats.get("prob_selected_mc"), errors="coerce") >= float(p_ats)]
    out["filtered_spread_mc"] = _metrics(ats_f)

    tot = work[work["type"].astype(str).str.upper().eq("TOTAL")].copy()
    tot["prob_selected_mc"] = tot.apply(_selected_prob_total, axis=1)
    tot_f = tot[pd.to_numeric(tot.get("prob_selected_mc"), errors="coerce") >= float(p_total)]
    out["filtered_total_mc"] = _metrics(tot_f)

    out["thresholds"] = {"p_ats": float(p_ats), "p_total": float(p_total)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize recs by MC selected-side probability thresholds")
    ap.add_argument("--details", type=str, required=True)
    ap.add_argument("--sim", type=str, required=True)
    ap.add_argument("--p-ats", type=float, default=0.58)
    ap.add_argument("--p-total", type=float, default=0.60)
    args = ap.parse_args()

    res = summarize(Path(args.details), Path(args.sim), args.p_ats, args.p_total)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
