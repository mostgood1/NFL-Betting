from __future__ import annotations

"""Evaluate shipped sim/model accuracy by week.

This script is intentionally *read-only* with respect to sim artifacts: it reads the
already-shipped backtest outputs under:
  nfl_compare/data/backtests/{season}_wk{week}/sim_probs.csv

It joins to final scores from nfl_compare/data/games.csv and computes per-week metrics:
- MAE for margin and total (pred vs actual)
- Brier score and accuracy for MC probabilities (ML/ATS/TOTAL)

Usage:
  python scripts/weekly_sim_accuracy.py --season 2025 --end-week 19
  python scripts/weekly_sim_accuracy.py --season 2025 --start-week 1 --end-week 19 --out reports/weekly_sim_accuracy_2025.csv

Outputs:
- CSV (default: reports/weekly_sim_accuracy_{season}.csv)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "nfl_compare" / "data"
BACKTESTS_DIR = DATA_DIR / "backtests"
REPORTS_DIR = REPO_ROOT / "reports"


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


def _load_games() -> pd.DataFrame:
    fp = DATA_DIR / "games.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        g = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()

    for c in ("season", "week", "home_score", "away_score"):
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    # Deduplicate by game_id if present
    try:
        if "game_id" in g.columns:
            g = g.sort_values([c for c in ["season", "week", "game_id"] if c in g.columns]).drop_duplicates(
                subset=["game_id"], keep="first"
            )
    except Exception:
        pass

    return g


def _read_sim_probs(season: int, week: int) -> Optional[pd.DataFrame]:
    fp = BACKTESTS_DIR / f"{int(season)}_wk{int(week)}" / "sim_probs.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    # Normalize core fields
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def evaluate_week(season: int, week: int, games: pd.DataFrame) -> Optional[dict]:
    sim = _read_sim_probs(season, week)
    if sim is None or sim.empty:
        return None

    g = games
    if g is None or g.empty:
        return None

    gmask = (pd.to_numeric(g.get("season"), errors="coerce") == int(season)) & (
        pd.to_numeric(g.get("week"), errors="coerce") == int(week)
    )
    gw = g[gmask].copy()
    if gw.empty:
        return None

    # Join actuals
    joined: pd.DataFrame
    try:
        if "game_id" in sim.columns and "game_id" in gw.columns:
            joined = sim.merge(gw[["game_id", "home_score", "away_score"]], on="game_id", how="left")
        else:
            # Fallback join by home/away
            joined = sim.merge(gw[["home_team", "away_team", "home_score", "away_score"]], on=["home_team", "away_team"], how="left")
    except Exception:
        joined = sim.copy()

    hs = pd.to_numeric(joined.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(joined.get("away_score"), errors="coerce")
    have_final = hs.notna() & as_.notna()
    if not have_final.any():
        # Nothing final yet
        return {
            "season": int(season),
            "week": int(week),
            "n_games": int(len(sim)),
            "n_final": 0,
        }

    margin_actual = hs - as_
    total_actual = hs + as_

    # Predicted means
    margin_pred = pd.to_numeric(joined.get("pred_margin"), errors="coerce")
    total_pred = pd.to_numeric(joined.get("pred_total"), errors="coerce")

    # References
    spread_ref = pd.to_numeric(joined.get("spread_ref"), errors="coerce")
    total_ref = pd.to_numeric(joined.get("total_ref"), errors="coerce")

    # Outcomes (exclude pushes)
    y_ml = (margin_actual > 0).astype(float)
    y_ml = y_ml.where(margin_actual != 0, np.nan)

    ats_diff = margin_actual + spread_ref
    y_ats = (ats_diff > 0).astype(float)
    y_ats = y_ats.where(ats_diff != 0, np.nan)

    tot_diff = total_actual - total_ref
    y_tot = (tot_diff > 0).astype(float)
    y_tot = y_tot.where(tot_diff != 0, np.nan)

    # Predicted probabilities
    p_ml = pd.to_numeric(joined.get("prob_home_win_mc"), errors="coerce")
    p_ats = pd.to_numeric(joined.get("prob_home_cover_mc"), errors="coerce")
    p_tot = pd.to_numeric(joined.get("prob_over_total_mc"), errors="coerce")

    # Restrict to completed finals
    m_final = have_final

    def _mask_for(series: pd.Series) -> pd.Series:
        return m_final & series.notna()

    metrics = {
        "season": int(season),
        "week": int(week),
        "n_games": int(len(sim)),
        "n_final": int(m_final.sum()),
        "mae_margin": _mae(margin_actual[m_final], margin_pred[m_final]),
        "mae_total": _mae(total_actual[m_final], total_pred[m_final]),
        "brier_ml": _brier(p_ml[_mask_for(y_ml)], y_ml[_mask_for(y_ml)]),
        "brier_ats": _brier(p_ats[_mask_for(y_ats)], y_ats[_mask_for(y_ats)]),
        "brier_total": _brier(p_tot[_mask_for(y_tot)], y_tot[_mask_for(y_tot)]),
        "acc_ml": _acc_from_prob(p_ml[_mask_for(y_ml)], y_ml[_mask_for(y_ml)]),
        "acc_ats": _acc_from_prob(p_ats[_mask_for(y_ats)], y_ats[_mask_for(y_ats)]),
        "acc_total": _acc_from_prob(p_tot[_mask_for(y_tot)], y_tot[_mask_for(y_tot)]),
        "n_push_ats": int((m_final & ats_diff.notna() & (ats_diff == 0)).sum()),
        "n_push_total": int((m_final & tot_diff.notna() & (tot_diff == 0)).sum()),
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate shipped sim/model accuracy by week")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    games = _load_games()

    rows = []
    for wk in range(int(args.start_week), int(args.end_week) + 1):
        r = evaluate_week(int(args.season), int(wk), games)
        if r is None:
            continue
        rows.append(r)

    if not rows:
        print("No weeks evaluated (missing sim_probs or games).")
        return 2

    out_df = pd.DataFrame(rows).sort_values(["season", "week"]) if rows else pd.DataFrame()

    out_fp = Path(args.out) if args.out else (REPORTS_DIR / f"weekly_sim_accuracy_{int(args.season)}.csv")
    try:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    out_df.to_csv(out_fp, index=False)

    # Console summary
    try:
        show = [c for c in [
            "week","n_final","mae_margin","mae_total","brier_ml","brier_ats","brier_total","acc_ml","acc_ats","acc_total"
        ] if c in out_df.columns]
        print(f"Wrote {out_fp}")
        print(out_df[show].to_string(index=False))
    except Exception:
        print(f"Wrote {out_fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
