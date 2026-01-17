"""
Train game-level models with current feature set (including injury features) and persist to nfl_compare/models/nfl_models.joblib.

Usage:
  python scripts/train_models.py [--season SEASON] [--week WEEK]

If season/week provided, filters training data to <= week within season (useful for quick iteration). By default trains on all available historical rows with scores.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure repo root is on sys.path so `nfl_compare` is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports
from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models

try:
    from joblib import dump as joblib_dump
except Exception:  # pragma: no cover
    joblib_dump = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "nfl_compare" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FP = MODELS_DIR / "nfl_models.joblib"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Train on a single season (<= week if provided)")
    ap.add_argument("--week", type=int, default=None, help="Train using rows up to this week (inclusive) for the chosen season")
    args = ap.parse_args(argv)

    # Load data
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)

    # Build features
    feat = merge_features(games, stats, lines, wx)

    # Filter to rows with targets available
    df = feat.copy()
    # Use rows where both scores exist
    for c in ("home_score","away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    # Optional temporal filter for fast iteration
    if args.season is not None:
        df = df[df["season"] == int(args.season)]
        if args.week is not None and "week" in df.columns:
            df = df[pd.to_numeric(df["week"], errors="coerce").fillna(0) <= int(args.week)]

    if df.empty:
        print("No training rows found after filters.")
        return 2

    # Train
    models = train_models(df)

    # Persist
    if joblib_dump is None:
        print("joblib not available; cannot save models.")
        return 3
    joblib_dump(models, MODEL_FP)
    print(f"Saved models to {MODEL_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
