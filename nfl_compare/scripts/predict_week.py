from __future__ import annotations

"""
Predict a specific season/week, including completed games, and write predictions_week.csv.

This complements the default predictions.csv (future games only) so the UI can show
model outputs for finals when needed (and on Render where on-the-fly prediction is disabled).

Usage:
  python -m nfl_compare.scripts.predict_week --season 2025 --week 1
"""

import argparse
from pathlib import Path
from joblib import load as joblib_load
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.models import predict as model_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    args = ap.parse_args()

    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = None

    models_path = Path(__file__).resolve().parents[2] / 'models' / 'nfl_models.joblib'
    if not models_path.exists():
        print('Model file not found. Run training first.')
        return
    models = joblib_load(models_path)

    feat = merge_features(games, team_stats, lines, wx)
    # Filter to requested season/week without excluding played games
    mask = (feat.get('season').astype(int) == int(args.season)) & (feat.get('week').astype(int) == int(args.week))
    sub = feat.loc[mask].copy()
    if sub.empty:
        print('No games found for requested season/week.')
        return

    pred = model_predict(models, sub)
    out_dir = Path(__file__).resolve().parents[2] / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / 'predictions_week.csv'
    pred.to_csv(out_fp, index=False)
    print(f'Wrote {out_fp} with {len(pred)} rows for season={args.season}, week={args.week}')


if __name__ == '__main__':
    main()
