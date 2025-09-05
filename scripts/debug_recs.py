import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_predictions, _load_games, _build_week_view, _attach_model_predictions, _compute_recommendations_for_row

pred_df = _load_predictions()
games_df = _load_games()
season_i = None
if not games_df.empty and 'season' in games_df.columns and not games_df['season'].isna().all():
    season_i = int(games_df['season'].max())
week_i = 1
view = _build_week_view(pred_df, games_df, season_i, week_i)
view = _attach_model_predictions(view)
print('Rows:', len(view))

for i, row in view.iterrows():
    recs = _compute_recommendations_for_row(row)
    if recs:
        print(row.get('away_team'), '@', row.get('home_team'), '->', len(recs), 'recs:', [(r['type'], r['selection'], round(r['ev_pct'],1) if r.get('ev_pct') is not None else None) for r in recs])
