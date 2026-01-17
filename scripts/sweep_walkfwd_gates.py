"""Sweep recommendation gate thresholds under walk-forward evaluation.

Goal: find thresholds that yield *reasonable pick volume* while maintaining
probability calibration (predictability) and avoiding obvious overconfidence.

This script:
- Builds a window view using the same plumbing as backtests (optionally walk-forward)
- Generates recommendations with strict_gates=True (so gates apply to finals too)
- Sweeps small grids of (delta-from-0.5) and EV% thresholds for ATS/TOTAL
- Reports per-market: rows, win_rate, profit, Brier, logloss, p_mean

Usage (PowerShell):
  $env:PRED_IGNORE_LOCKED='1'
  $env:PROB_CALIBRATION_FILE='nfl_compare/data/prob_calibration_walkfwd_view_2025_iso_full.json'
  & .\.venv\Scripts\python.exe scripts\sweep_walkfwd_gates.py --season 2025 --end-week 17 --lookback 17 --walk-forward --line-mode open

Notes:
- Keeps calibration file fixed for the sweep (app caches it).
- Focuses on ATS/TOTAL; ML is left enabled only if RECS_ALLOWED_MARKETS includes it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on path
BASE_DIR = Path(__file__).resolve().parents[1]
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import _compute_recommendations_for_row
from scripts.backtest_recommendations import _build_window, _summarize_recs


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _market_implied_from_odds(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    # If already decimal odds
    is_decimal = o.between(1.01, 10.0)
    p = pd.Series(np.nan, index=o.index, dtype="float")
    p.loc[is_decimal] = 1.0 / o.loc[is_decimal].astype(float)

    # American odds
    a = o.loc[~is_decimal]
    ap = pd.Series(np.nan, index=a.index, dtype="float")
    pos = a > 0
    neg = a < 0
    ap.loc[pos] = 100.0 / (a.loc[pos] + 100.0)
    ap.loc[neg] = (-a.loc[neg]) / ((-a.loc[neg]) + 100.0)
    p.loc[~is_decimal] = ap
    return p


def _brier_and_logloss(df: pd.DataFrame) -> Tuple[float, float, float]:
    # returns (p_mean, brier, logloss)
    if df is None or df.empty:
        return float("nan"), float("nan"), float("nan")
    p = pd.to_numeric(df.get("prob_selected"), errors="coerce")
    res = df.get("result", pd.Series([None] * len(df))).astype(str).str.upper()
    y = pd.Series(np.nan, index=df.index, dtype="float")
    y.loc[res.eq("WIN")] = 1.0
    y.loc[res.eq("LOSS")] = 0.0
    m = p.notna() & y.notna()
    if int(m.sum()) == 0:
        return float("nan"), float("nan"), float("nan")

    pp = p.loc[m].clip(1e-6, 1 - 1e-6)
    yy = y.loc[m]
    p_mean = float(pp.mean())
    brier = float(((pp - yy) ** 2).mean())
    logloss = float((-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp))).mean())
    return p_mean, brier, logloss


def _brier_implied_from_odds(sub: pd.DataFrame, p_imp: pd.Series) -> float:
    if sub is None or sub.empty:
        return float("nan")
    res = sub.get("result", pd.Series([None] * len(sub))).astype(str).str.upper()
    y = pd.Series(np.nan, index=sub.index, dtype=float)
    y.loc[res.eq("WIN")] = 1.0
    y.loc[res.eq("LOSS")] = 0.0
    m = p_imp.notna() & y.notna()
    if int(m.sum()) == 0:
        return float("nan")
    return float(((p_imp.loc[m].clip(1e-6, 1 - 1e-6) - y.loc[m]) ** 2).mean())


def _run_one(v: pd.DataFrame, line_mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in v.iterrows():
        try:
            picks = _compute_recommendations_for_row(row, strict_gates=True, line_mode=line_mode)
        except Exception:
            picks = []
        for r in picks:
            rec = dict(r)
            # ensure fields exist
            rec["season"] = row.get("season")
            rec["week"] = row.get("week")
            rec["game_id"] = row.get("game_id")
            rec["home_team"] = row.get("home_team")
            rec["away_team"] = row.get("away_team")
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep walk-forward recommendation gates")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=17)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--line-mode", choices=["auto", "open", "close"], default="open")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--min-picks", type=int, default=20)
    args = ap.parse_args()

    season = int(args.season)
    end_week = int(args.end_week)
    lookback = max(1, int(args.lookback))
    line_mode = str(args.line_mode).strip().lower()

    v = _build_window(season, end_week, lookback, compute_mc=False, walk_forward=bool(args.walk_forward))
    if v is None or v.empty:
        print("No window data.")
        return 0

    # Only final games with actual scores (avoid week 19 placeholder zeros etc.)
    hs = pd.to_numeric(v.get("home_score"), errors="coerce")
    aas = pd.to_numeric(v.get("away_score"), errors="coerce")
    is_final = hs.notna() & aas.notna() & ((hs + aas) > 0)
    v = v[is_final].copy()

    # Grid (keep modest)
    # Include 0.0 thresholds: with calibrated probabilities, true edges are often modest,
    # and totals especially may produce very few >=2% EV opportunities.
    min_ev_global_opts = [0.0, 1.0, 2.0]
    ats_delta_opts = [0.0, 0.03, 0.05, 0.07]
    total_delta_opts = [0.0, 0.03, 0.05, 0.07]
    ev_ats_opts = [0.0, 1.0, 2.0, 4.0]
    ev_total_opts = [0.0, 1.0, 2.0, 4.0]

    # Fixed knobs (can expand later)
    os.environ.setdefault("RECS_ALLOWED_MARKETS", "SPREAD,TOTAL")
    os.environ.setdefault("RECS_ONE_PER_MARKET", "true")
    os.environ.setdefault("RECS_ONE_PER_GAME", "false")

    rows_out: List[Dict[str, Any]] = []
    i = 0
    for min_ev_g in min_ev_global_opts:
        for atsd in ats_delta_opts:
            for td in total_delta_opts:
                for evats in ev_ats_opts:
                    for evtot in ev_total_opts:
                        i += 1
                        os.environ["RECS_MIN_EV_PCT"] = str(min_ev_g)
                        os.environ["RECS_MIN_ATS_DELTA"] = str(atsd)
                        os.environ["RECS_MIN_TOTAL_DELTA"] = str(td)
                        os.environ["RECS_MIN_EV_PCT_ATS"] = str(evats)
                        os.environ["RECS_MIN_EV_PCT_TOTAL"] = str(evtot)

                        df = _run_one(v, line_mode=line_mode)
                        summ = _summarize_recs(df)

                        sp = (summ.get("SPREAD") or {})
                        tot = (summ.get("TOTAL") or {})
                        sp_df = df[df.get("type", "").astype(str).str.upper().eq("SPREAD")] if not df.empty else pd.DataFrame()
                        tot_df = df[df.get("type", "").astype(str).str.upper().eq("TOTAL")] if not df.empty else pd.DataFrame()

                        sp_pmean, sp_brier, sp_ll = _brier_and_logloss(sp_df)
                        tot_pmean, tot_brier, tot_ll = _brier_and_logloss(tot_df)

                        # Market-implied baseline for calibration comparison
                        sp_imp = _market_implied_from_odds(sp_df.get("odds", pd.Series([np.nan] * len(sp_df)))) if not sp_df.empty else pd.Series([], dtype=float)
                        tot_imp = _market_implied_from_odds(tot_df.get("odds", pd.Series([np.nan] * len(tot_df)))) if not tot_df.empty else pd.Series([], dtype=float)

                        sp_brier_imp = _brier_implied_from_odds(sp_df, sp_imp)
                        tot_brier_imp = _brier_implied_from_odds(tot_df, tot_imp)

                        rec: Dict[str, Any] = {
                            "min_ev_global": float(min_ev_g),
                            "ats_delta": atsd,
                            "total_delta": td,
                            "ev_ats": evats,
                            "ev_total": evtot,
                            "sp_rows": int(sp.get("rows", 0) or 0),
                            "sp_win_rate": float(sp.get("win_rate", np.nan)),
                            "sp_profit_sum": float(sp.get("profit_sum_units", np.nan)),
                            "sp_roi": float(sp.get("roi_units_per_bet", np.nan)),
                            "sp_p_mean": sp_pmean,
                            "sp_brier": sp_brier,
                            "sp_brier_implied": sp_brier_imp,
                            "sp_logloss": sp_ll,
                            "tot_rows": int(tot.get("rows", 0) or 0),
                            "tot_win_rate": float(tot.get("win_rate", np.nan)),
                            "tot_profit_sum": float(tot.get("profit_sum_units", np.nan)),
                            "tot_roi": float(tot.get("roi_units_per_bet", np.nan)),
                            "tot_p_mean": tot_pmean,
                            "tot_brier": tot_brier,
                            "tot_brier_implied": tot_brier_imp,
                            "tot_logloss": tot_ll,
                            "rows": int(summ.get("rows", 0) or 0),
                        }
                        rows_out.append(rec)

    out_df = pd.DataFrame(rows_out)

    # Filter to configs with enough volume per market
    min_picks = int(args.min_picks)
    cand = out_df[(out_df["sp_rows"] >= min_picks) & (out_df["tot_rows"] >= min_picks)].copy()
    if cand.empty:
        print("No configs met min pick counts; writing full sweep only.")
    else:
        # Rank by (calibration quality first, then profit)
        cand["cal_gap_sp"] = (cand["sp_brier"] - cand["sp_brier_implied"]).abs()
        cand["cal_gap_tot"] = (cand["tot_brier"] - cand["tot_brier_implied"]).abs()
        cand["cal_gap"] = cand["cal_gap_sp"].fillna(999) + cand["cal_gap_tot"].fillna(999)
        cand = cand.sort_values(["cal_gap", "sp_profit_sum", "tot_profit_sum"], ascending=[True, False, False])
        print("\nTop configs (min picks per market met):")
        print(cand.head(12).to_string(index=False))

    # Write output
    out_path = Path(args.out) if args.out else (BASE_DIR / "nfl_compare" / "data" / "backtests" / f"{season}_wk{end_week}" / "walkfwd_gate_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote sweep: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
