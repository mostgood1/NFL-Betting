import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _derive_predictions_from_market,
)
parser = argparse.ArgumentParser(description="Inspect merged week view with predictions and market synthesis.")
parser.add_argument('--season', type=int, help='Season to inspect (defaults to latest available)')
parser.add_argument('--week', type=int, help='Week to inspect (defaults to inferred or 1)')
args = parser.parse_args()

pred_df = _load_predictions()
games_df = _load_games()

# Determine season/week
season_i = args.season
if season_i is None:
    if games_df is not None and not games_df.empty and 'season' in games_df.columns and not games_df['season'].isna().all():
        season_i = int(games_df['season'].max())
    elif pred_df is not None and not pred_df.empty and 'season' in pred_df.columns and not pred_df['season'].isna().all():
        season_i = int(pred_df['season'].max())
week_i = args.week or 1

view = _build_week_view(pred_df, games_df, season_i, week_i)
view = _attach_model_predictions(view)
view = _derive_predictions_from_market(view)
print(f'Season {season_i} Week {week_i} rows in view:', 0 if view is None else len(view))
if view is None or view.empty:
    sys.exit(0)

cols = [
    'home_team','away_team','game_date','pred_home_points','pred_away_points','pred_total','pred_home_win_prob',
    'spread_home','market_spread_home','open_spread_home','close_spread_home',
    'total','market_total','open_total','close_total',
    'moneyline_home','moneyline_away'
]
cols = [c for c in cols if c in view.columns]
print('Columns present:', cols)
print('Counts:')
for c in cols:
    na = int(view[c].isna().sum())
    print(f"  {c}: {len(view)-na} non-null")

# List any pred_* and prob_* columns
extra = [c for c in view.columns if c.startswith('pred_') or c.startswith('prob_') or c in ('prediction_source','derived_from_market')]
print('Extra pred/prob/source cols:', [c for c in extra])
print(view[[c for c in extra if c in view.columns]].head(3))

# Source counts
if 'prediction_source' in view.columns:
    print('prediction_source counts:')
    print(view['prediction_source'].value_counts(dropna=False))
if 'derived_from_market' in view.columns:
    print('derived_from_market counts:')
    print(view['derived_from_market'].value_counts(dropna=False))
