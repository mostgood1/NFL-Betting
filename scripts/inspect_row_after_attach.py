import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app import _load_predictions, _load_games, _build_week_view, _attach_model_predictions

def main():
    pred = _load_predictions()
    games = _load_games()
    view = _build_week_view(pred, games, 2025, 8)
    out = _attach_model_predictions(view)
    row = out[(out['home_team']=='Kansas City Chiefs') & (out['away_team']=='Washington Commanders')].head(1)
    if row.empty:
        print('Row not found')
        return
    cols = ['season','week','game_id','market_spread_home','spread_home','open_spread_home','close_spread_home','market_total','total','open_total','close_total','moneyline_home','moneyline_away']
    cols = [c for c in cols if c in row.columns]
    print(row[cols].to_string(index=False))

if __name__ == '__main__':
    main()
