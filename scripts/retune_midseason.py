from __future__ import annotations

"""
Mid-season retune orchestrator for game and player prop models.

Steps:
  1) Rebuild player usage priors for the current season
  2) Rebuild player efficiency priors from recent seasons' PBP
  3) Retrain game-level models using all historical completed games (optionally up to a cutoff)
  4) Fit totals calibration parameters from the last K completed weeks
  5) Optional: run game and props backtests and write summaries

Usage:
  python scripts/retune_midseason.py --season 2025 --week 10 --calib-weeks 4 \
    --run-backtests --props-end-week 9
"""

import argparse
from pathlib import Path
from typing import Optional

from joblib import dump as joblib_dump

import pandas as pd

# Local libraries
from nfl_compare.src.build_player_usage_priors import build_player_usage_priors
from nfl_compare.src.build_player_efficiency_priors import build_player_efficiency_priors
from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models
from nfl_compare.src.weather import load_weather_for_games

# Reuse calibration fitter from script module
from scripts.fit_totals_calibration import (
    fit_totals_calibration as _fit_totals,
    _get_data_dir as _cal_data_dir,
    _load_games as _cal_load_games,
    _load_predictions_any as _cal_load_preds,
    _load_lines as _cal_load_lines,
    _join_frames as _cal_join_frames,
)

# Optional backtests
try:
    from scripts.backtest_games import backtest as backtest_games
    from scripts.backtest_props import backtest as backtest_props
except Exception:
    backtest_games = None  # type: ignore
    backtest_props = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "nfl_compare" / "models"
DATA_DIR = BASE_DIR / "nfl_compare" / "data"
MODEL_FP = MODELS_DIR / "nfl_models.joblib"


def _recent_completed_weeks(k: int) -> list[tuple[int,int]]:
    games = _cal_load_games(_cal_data_dir())
    if games is None or games.empty:
        return []
    df = games.copy()
    for c in ("season","week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"home_score","away_score"}.issubset(df.columns):
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    if df.empty:
        return []
    df = df.sort_values(["season","week"]) if {"season","week"}.issubset(df.columns) else df
    uniq = df[["season","week"]].dropna().drop_duplicates().values.tolist()
    uniq = [(int(s), int(w)) for s, w in uniq]
    return uniq[-k:]


def main():
    ap = argparse.ArgumentParser(description="Mid-season retune for game and props models")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True, help="Current week (completed or upcoming)")
    ap.add_argument("--calib-weeks", type=int, default=4, help="Number of most recent completed weeks for totals calibration")
    ap.add_argument("--run-backtests", action="store_true", help="Run games/props backtests after retuning")
    ap.add_argument("--props-end-week", type=int, default=None, help="Last completed week to include in props backtest (defaults to week-1)")
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Usage priors for this season
    print(f"[retune] Building player usage priors for {season}...")
    usage = build_player_usage_priors(season)
    usage_fp = DATA_DIR / "player_usage_priors.csv"
    usage.to_csv(usage_fp, index=False)
    print(f"[retune] Wrote usage priors -> {usage_fp} ({len(usage)} rows)")

    # 2) Efficiency priors from recent seasons (last 3 + current when parquet present)
    print(f"[retune] Building player efficiency priors (recent seasons)...")
    eff = build_player_efficiency_priors(None)
    eff_fp = DATA_DIR / "player_efficiency_priors.csv"
    if eff is not None and not eff.empty:
        eff.to_csv(eff_fp, index=False)
        print(f"[retune] Wrote efficiency priors -> {eff_fp} ({len(eff)} rows)")
    else:
        print("[retune] Efficiency priors skipped (no PBP parquet files found)")

    # 3) Retrain game models on all completed games (up to current week in current season)
    print(f"[retune] Training game models...")
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)
    feat = merge_features(games, stats, lines, wx)
    df = feat.copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    # Keep only rows up to the current season/week
    for c in ("season","week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    mask = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    df = df[mask].copy()
    if df.empty:
        print("[retune] No rows available for training. Aborting.")
        return 2
    models = train_models(df)
    joblib_dump(models, MODEL_FP)
    print(f"[retune] Saved models -> {MODEL_FP}")

    # 4) Fit totals calibration from recent k completed weeks
    k = max(1, int(args.calib_weeks))
    weeks = _recent_completed_weeks(k)
    if weeks:
        print(f"[retune] Fitting totals calibration from last {k} completed weeks: {weeks}")
        games_cal = _cal_load_games(_cal_data_dir())
        preds_cal = _cal_load_preds(_cal_data_dir())
        lines_cal = _cal_load_lines(_cal_data_dir())
        df_join = _cal_join_frames(games_cal, preds_cal, lines_cal, weeks)
        fit = _fit_totals(df_join)
        if fit:
            out_path = DATA_DIR / "totals_calibration.json"
            payload = {
                "scale": fit.scale,
                "shift": fit.shift,
                "market_blend": fit.market_blend,
                "metrics": {"mse": fit.mse, "mae": fit.mae, "n": fit.n},
                "weeks_used": fit.weeks_used,
            }
            out_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
            print(f"[retune] Wrote totals calibration -> {out_path}")
        else:
            print("[retune] Totals calibration failed (insufficient rows)")
    else:
        print("[retune] No recent completed weeks inferred; skipping totals calibration")

    if args.run_backtests:
        # Games backtest: evaluate weeks up to week-1
        if backtest_games is not None:
            try:
                print(f"[retune] Backtesting games ({season} weeks 1..{max(1, week-1)})...")
                gdf = backtest_games(season, 1, max(1, week-1), include_same=True)
                if gdf is not None and not gdf.empty:
                    gfp = DATA_DIR / f"backtest_games_{season}_wk{max(1, week-1)}.csv"
                    gdf.to_csv(gfp, index=False)
                    print(f"[retune] Wrote games backtest -> {gfp}")
                    try:
                        print(gdf.to_string(index=False))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[retune] Games backtest failed: {e}")
        # Props backtest through provided end week or week-1
        if backtest_props is not None:
            end_w = int(args.props_end_week) if args.props_end_week else max(1, week-1)
            try:
                print(f"[retune] Backtesting props ({season} weeks 1..{end_w})...")
                pdf = backtest_props(season, 1, end_w)
                if pdf is not None and not pdf.empty:
                    pfp = DATA_DIR / f"backtest_props_{season}_wk{end_w}.csv"
                    pdf.to_csv(pfp, index=False)
                    print(f"[retune] Wrote props backtest -> {pfp}")
                    try:
                        print(pdf.to_string(index=False))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[retune] Props backtest failed: {e}")

    print("[retune] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
