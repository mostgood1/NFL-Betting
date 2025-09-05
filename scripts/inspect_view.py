import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_predictions, _load_games, _build_week_view, _attach_model_predictions

pred_df = _load_predictions()
games_df = _load_games()

# Default to latest season, week 1
season_i = None
if not games_df.empty and 'season' in games_df.columns and not games_df['season'].isna().all():
    season_i = int(games_df['season'].max())
week_i = 1

view = _build_week_view(pred_df, games_df, season_i, week_i)
view = _attach_model_predictions(view)
print('Rows in view:', 0 if view is None else len(view))
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
extra = [c for c in view.columns if c.startswith('pred_') or c.startswith('prob_')]
print('Extra pred/prob cols:', extra)
print(view[extra].head(3))
