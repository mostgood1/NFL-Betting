from __future__ import annotations

"""
Backtest Monte Carlo simulation accuracy over a week range.

Trains underlying game models (and uses any existing calibrations), runs simulations
to derive ML/ATS/Total probabilities, and evaluates:
- MAE for margin/total means vs actuals
- Brier scores for probabilities (ML, ATS, Total)
- Accuracy for ML/ATS/Total (threshold 0.5)

Usage:
  python scripts/backtest_simulation.py --season 2025 --start-week 1 --end-week 18 \
    --n-sims 2000 --out-dir nfl_compare/data/backtests/2025_wk18
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from scripts.simulate_games import simulate
except ImportError:
    # Fallback when running as a module
    from .simulate_games import simulate
from nfl_compare.src.data_sources import load_games

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


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


def _acc_from_prob(p: pd.Series, y: pd.Series) -> float:
    try:
        pp = pd.to_numeric(p, errors="coerce").astype(float)
        yy = pd.to_numeric(y, errors="coerce").astype(float)
        m = pp.notna() & yy.notna()
        if not m.any():
            return float("nan")
        return float(((pp[m] > 0.5).astype(float) == yy[m]).mean())
    except Exception:
        return float("nan")


def backtest_sim(season: int, start_week: int, end_week: int, n_sims: int = 2000, out_dir: Optional[Path] = None, pred_cache_fp: Optional[str] = None) -> Dict[str, float]:
    probs_df, summ_df = simulate(
        season,
        start_week,
        end_week,
        n_sims=n_sims,
        include_same=True,
        ats_sigma_override=None,
        total_sigma_override=None,
        pred_cache_fp=pred_cache_fp,
    )
    if probs_df is None or probs_df.empty:
        return {}

    games = load_games()
    # Coerce core columns
    for c in ("season","week","home_score","away_score"):
        if c in games.columns:
            games[c] = pd.to_numeric(games[c], errors="coerce")
    # Slice target weeks
    gmask = (games["season"] == int(season)) & (games["week"] >= int(start_week)) & (games["week"] <= int(end_week))
    eval_games = games[gmask].copy()
    # Use game_id to join if present; else inner concat by teams/date fallback
    try:
        if "game_id" in eval_games.columns and "game_id" in probs_df.columns:
            join = probs_df.merge(eval_games[["game_id","home_score","away_score"]], on="game_id", how="left")
        else:
            join = probs_df.copy()
            join = join.merge(eval_games[["home_team","away_team","home_score","away_score"]], on=["home_team","away_team"], how="left")
    except Exception:
        join = probs_df.copy()

    # Actual targets
    join["margin_actual"] = pd.to_numeric(join.get("home_score"), errors="coerce") - pd.to_numeric(join.get("away_score"), errors="coerce")
    join["total_actual"] = pd.to_numeric(join.get("home_score"), errors="coerce") + pd.to_numeric(join.get("away_score"), errors="coerce")
    # Reference lines from simulation output (spread_ref, total_ref)
    spread_ref = pd.to_numeric(join.get("spread_ref"), errors="coerce") if "spread_ref" in join.columns else pd.Series(index=join.index, dtype=float)
    total_ref = pd.to_numeric(join.get("total_ref"), errors="coerce") if "total_ref" in join.columns else pd.Series(index=join.index, dtype=float)

    # Outcomes: ML (home win), ATS (home cover vs spread_ref), TOTAL (Over vs total_ref)
    y_ml = (pd.to_numeric(join["margin_actual"], errors="coerce") > 0).astype(float)
    y_ats = (pd.to_numeric(join["margin_actual"], errors="coerce") + spread_ref > 0).astype(float)
    y_tot = (pd.to_numeric(join["total_actual"], errors="coerce") > total_ref).astype(float)

    # Predictions: means and probabilities
    m_pred = pd.to_numeric(join.get("pred_margin"), errors="coerce")
    t_pred = pd.to_numeric(join.get("pred_total"), errors="coerce")
    p_ml = pd.to_numeric(join.get("prob_home_win_mc"), errors="coerce")
    p_ats = pd.to_numeric(join.get("prob_home_cover_mc"), errors="coerce")
    p_tot = pd.to_numeric(join.get("prob_over_total_mc"), errors="coerce")

    # Metrics
    mae_margin = float((pd.to_numeric(join["margin_actual"], errors="coerce") - m_pred).abs().mean())
    mae_total = float((pd.to_numeric(join["total_actual"], errors="coerce") - t_pred).abs().mean())
    brier_ml = _brier(p_ml, y_ml)
    brier_ats = _brier(p_ats, y_ats)
    brier_total = _brier(p_tot, y_tot)
    acc_ml = _acc_from_prob(p_ml, y_ml)
    acc_ats = _acc_from_prob(p_ats, y_ats)
    acc_total = _acc_from_prob(p_tot, y_tot)

    metrics = {
        "season": int(season),
        "start_week": int(start_week),
        "end_week": int(end_week),
        "n_games": int(len(join)),
        "mae_margin": mae_margin,
        "mae_total": mae_total,
        "brier_ml": brier_ml,
        "brier_ats": brier_ats,
        "brier_total": brier_total,
        "acc_ml": acc_ml,
        "acc_ats": acc_ats,
        "acc_total": acc_total,
    }

    # Accuracy and coverage with probability gating (p >= 0.55)
    def _acc_cov(pp: pd.Series, yy: pd.Series, thresh: float) -> tuple[float, float, int]:
        try:
            p = pd.to_numeric(pp, errors="coerce").astype(float)
            y = pd.to_numeric(yy, errors="coerce").astype(float)
            m = p.notna() & y.notna() & (p >= float(thresh))
            n = int(m.sum())
            if n == 0:
                return float("nan"), 0.0, 0
            acc = float(((p[m] > 0.5).astype(float) == y[m]).mean())
            cov = float(n) / float(len(y)) if len(y) else 0.0
            return acc, cov, n
        except Exception:
            return float("nan"), 0.0, 0

    acc_ml_55, cov_ml_55, n_ml_55 = _acc_cov(p_ml, y_ml, 0.55)
    acc_ats_55, cov_ats_55, n_ats_55 = _acc_cov(p_ats, y_ats, 0.55)
    acc_tot_55, cov_tot_55, n_tot_55 = _acc_cov(p_tot, y_tot, 0.55)
    metrics.update({
        "acc_ml_p_ge_0.55": acc_ml_55, "coverage_ml_p_ge_0.55": cov_ml_55, "n_ml_p_ge_0.55": n_ml_55,
        "acc_ats_p_ge_0.55": acc_ats_55, "coverage_ats_p_ge_0.55": cov_ats_55, "n_ats_p_ge_0.55": n_ats_55,
        "acc_total_p_ge_0.55": acc_tot_55, "coverage_total_p_ge_0.55": cov_tot_55, "n_total_p_ge_0.55": n_tot_55,
    })

    # Write standardized outputs
    if out_dir is None:
        out_dir = DATA_DIR / "backtests" / f"{season}_wk{end_week}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with open(out_dir / "sim_backtest_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # Also write per-game details for inspection
        join.to_csv(out_dir / "sim_backtest_details.csv", index=False)
        print(f"Wrote {out_dir / 'sim_backtest_metrics.json'}")
        print(f"Wrote {out_dir / 'sim_backtest_details.csv'}")
    except Exception as e:
        print(f"Failed writing outputs: {e}")
    # Console summary
    try:
        print(pd.DataFrame([metrics]).to_string(index=False))
    except Exception:
        pass
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Backtest simulation accuracy over a week range")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--pred-cache", type=str, default=None, help="Path to cached predictions (games_details.csv) to avoid retraining")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    # If no explicit pred cache provided, try standardized backtests dir
    pred_cache_fp = args.pred_cache
    try:
        if pred_cache_fp is None and out_dir is not None:
            std_fp = out_dir / "games_details.csv"
            if std_fp.exists():
                pred_cache_fp = str(std_fp)
    except Exception:
        pass
    backtest_sim(args.season, args.start_week, args.end_week, n_sims=args.n_sims, out_dir=out_dir, pred_cache_fp=pred_cache_fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
