from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure package import path
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / 'nfl_compare'
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.data_sources import load_games, load_team_stats, load_lines
from src.features import merge_features
from src.weather import load_weather_for_games
from src.models import train_models, predict


def main():
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = None
    df = merge_features(games, team_stats, lines, wx)

    # Use completed rows only for training; temporal split: train < 2024, test == 2024
    df_all = df.dropna(subset=['home_score', 'away_score']).copy()
    train = df_all[df_all['season'] < 2024]
    test = df_all[df_all['season'] == 2024]

    if train.empty or test.empty:
        print('Not enough data for 2024 OOT evaluation.')
        return

    models = train_models(train)
    preds = predict(models, test)

    # Errors
    preds['act_margin'] = preds['home_score'] - preds['away_score']
    preds['act_total'] = preds['home_score'] + preds['away_score']
    preds['err_margin'] = (preds['pred_margin'] - preds['act_margin']).abs()
    preds['err_total'] = (preds['pred_total'] - preds['act_total']).abs()
    mae_margin = float(preds['err_margin'].mean())
    mae_total = float(preds['err_total'].mean())
    within3 = float((preds['err_margin'] <= 3).mean())
    within7 = float((preds['err_margin'] <= 7).mean())

    # Home-win accuracy (threshold 0.5)
    preds['pred_home_cls'] = (preds['prob_home_win'] >= 0.5).astype(int)
    preds['act_home_win'] = (preds['act_margin'] > 0).astype(int)
    acc_home = float((preds['pred_home_cls'] == preds['act_home_win']).mean())

    # Print summary
    print('2024 out-of-time evaluation:')
    print(f'- MAE margin: {mae_margin:.3f}')
    print(f'- MAE total:  {mae_total:.3f}')
    print(f'- Home-win accuracy: {acc_home:.3f}')
    print(f'- |err_margin| <= 3: {within3:.3f}')
    print(f'- |err_margin| <= 7: {within7:.3f}')


if __name__ == '__main__':
    main()
