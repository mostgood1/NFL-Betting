from __future__ import annotations

"""
Backtest game-level models on a target season up to a given week.

We train on historical seasons (< target season) and optionally on same-season weeks prior to the cutoff,
then evaluate on the target season weeks [start_week..end_week]. Metrics include MAE for margin and total,
and accuracy for home team wins.

Usage:
  python scripts/backtest_games.py --season 2025 --start-week 1 --end-week 9 \
    --include-same-season --out nfl_compare/data/backtest_games_2025_wk9.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models, predict as model_predict
from nfl_compare.src.weather import load_weather_for_games

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    e = (yt - yp).abs().dropna()
    return float(e.mean()) if len(e) else float("nan")


def _acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    ok = y_true.notna() & y_pred.notna()
    return float((y_true[ok] == y_pred[ok]).mean()) if ok.any() else float("nan")


def backtest(season: int, start_week: int, end_week: int, include_same: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx)
    # Completed games only for target season slice
    feat = feat.copy()
    for c in ("season","week"):
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
    # Build train/eval splits
    hist = feat.dropna(subset=["home_score","away_score"]).copy()
    train_mask = hist["season"] < int(season)
    if include_same:
        train_mask = train_mask | ((hist["season"] == int(season)) & (hist["week"] < int(start_week)))
    df_train = hist[train_mask].copy()

    eval_mask = (hist["season"] == int(season)) & (hist["week"] >= int(start_week)) & (hist["week"] <= int(end_week))
    df_eval = hist[eval_mask].copy()

    # Deduplicate by game_id to avoid explosion from joins (e.g., multiple lines rows)
    try:
        if 'game_id' in df_train.columns:
            df_train = df_train.sort_values(['season','week','game_id']).drop_duplicates(subset=['game_id'], keep='first')
    except Exception:
        pass
    try:
        if 'game_id' in df_eval.columns:
            df_eval = df_eval.sort_values(['season','week','game_id']).drop_duplicates(subset=['game_id'], keep='first')
    except Exception:
        pass

    if df_train.empty or df_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    models = train_models(df_train)
    pred = model_predict(models, df_eval)

    # Actual targets
    pred["home_margin_actual"] = df_eval["home_score"] - df_eval["away_score"]
    pred["total_points_actual"] = df_eval["home_score"] + df_eval["away_score"]
    pred["home_win_actual"] = (pred["home_margin_actual"] > 0).astype(int)
    pred["home_win_pred"] = (pred["pred_margin"] > 0).astype(int)

    # Metrics
    row: Dict[str, float | int] = {
        "season": int(season),
        "start_week": int(start_week),
        "end_week": int(end_week),
        "n_games": int(len(df_eval)),
        "mae_margin": _mae(pred["home_margin_actual"], pred["pred_margin"]),
        "mae_total": _mae(pred["total_points_actual"], pred["pred_total"]),
        "acc_home_win": _acc(pred["home_win_actual"], pred["home_win_pred"]),
    }
    # Build per-game details for standardized reporting
    details_cols: List[str] = []
    base = df_eval.copy()
    keep_base = [c for c in ["season","week","game_id","game_date","date","home_team","away_team","home_score","away_score","spread_home","total"] if c in base.columns]
    if keep_base:
        base = base[keep_base].copy()
    else:
        base = pd.DataFrame()
    det = pred.copy()
    # Attach market lines if present in eval base
    try:
        if not base.empty:
            det = det.merge(base, left_index=True, right_index=True, how="left", suffixes=("",""))
    except Exception:
        pass
    # Compute residuals and normalize column names
    try:
        det["resid_margin"] = pd.to_numeric(det["home_margin_actual"], errors="coerce") - pd.to_numeric(det["pred_margin"], errors="coerce")
    except Exception:
        det["resid_margin"] = pd.NA
    try:
        det["resid_total"] = pd.to_numeric(det["total_points_actual"], errors="coerce") - pd.to_numeric(det["pred_total"], errors="coerce")
    except Exception:
        det["resid_total"] = pd.NA
    # Rename common columns for clarity
    det = det.rename(columns={
        "home_margin_actual": "margin_actual",
        "total_points_actual": "total_actual",
        "pred_margin": "margin_pred",
        "pred_total": "total_pred",
        "spread_home": "market_spread_home",
        "total": "market_total",
        "date": "game_date",
    })
    # Re-order detail columns when available
    prefer = [
        "season","week","game_id","game_date","home_team","away_team",
        "margin_pred","total_pred","market_spread_home","market_total",
        "margin_actual","total_actual","resid_margin","resid_total"
    ]
    cols = [c for c in prefer if c in det.columns] + [c for c in det.columns if c not in prefer]
    det = det[cols]
    return pd.DataFrame([row]), det


def main():
    ap = argparse.ArgumentParser(description="Backtest game models on a target season/week range")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--include-same-season", action="store_true", help="Include same-season weeks < start-week in training")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None, help="Standardized output directory (backtests/<season>_wk<end_week>/)")
    args = ap.parse_args()

    summ_df, det_df = backtest(args.season, args.start_week, args.end_week, include_same=args.include_same_season)
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
    # Write legacy single CSV if --out provided
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(out_fp, index=False)
        print(f"Wrote games backtest summary -> {out_fp}")
    # Write standardized files
    if std_dir is not None:
        try:
            summ_fp = std_dir / "games_summary.csv"
            summ_df.to_csv(summ_fp, index=False)
            print(f"Wrote {summ_fp}")
        except Exception as e:
            print(f"Failed writing games_summary.csv: {e}")
        try:
            if det_df is not None and not det_df.empty:
                det_fp = std_dir / "games_details.csv"
                det_df.to_csv(det_fp, index=False)
                print(f"Wrote {det_fp}")
        except Exception as e:
            print(f"Failed writing games_details.csv: {e}")
        # Minimal metrics.json for quick consumption
        try:
            import json as _json
            metrics = summ_df.iloc[0].to_dict()
            with open(std_dir / "metrics.json", "w", encoding="utf-8") as f:
                _json.dump({"games": metrics}, f, indent=2)
            print(f"Wrote {std_dir / 'metrics.json'}")
        except Exception as e:
            print(f"Failed writing metrics.json: {e}")
    # Console print
    try:
        print(summ_df.to_string(index=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
