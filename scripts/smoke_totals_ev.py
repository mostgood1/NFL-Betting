import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _apply_totals_calibration_to_view,
    _derive_predictions_from_market,
    _build_cards,
)

# Simple smoke: build view, apply calibration, build cards, and print a few TOTAL recs

def main(season: int | None = None, week: int | None = 1, limit: int = 5):
    pred_df = _load_predictions()
    games_df = _load_games()

    # Infer latest season if not provided
    season_i = season
    if season_i is None:
        if games_df is not None and not games_df.empty and 'season' in games_df.columns and not games_df['season'].isna().all():
            season_i = int(games_df['season'].max())
        elif pred_df is not None and not pred_df.empty and 'season' in pred_df.columns and not pred_df['season'].isna().all():
            season_i = int(pred_df['season'].max())
    if week is None:
        week_i = 1
    else:
        week_i = int(week)

    view = _build_week_view(pred_df, games_df, season_i, week_i)
    view = _attach_model_predictions(view)
    view = _apply_totals_calibration_to_view(view)
    view = _derive_predictions_from_market(view)

    # Build cards and print basic TOTAL EV context
    cards = _build_cards(view)
    shown = 0
    print("Showing up to", limit, "cards' TOTAL EV context")
    for c in cards:
        if shown >= limit:
            break
        row = {
            'matchup': f"{c.get('away_team')} @ {c.get('home_team')}",
            'market_total': c.get('market_total'),
            'pred_total(card)': c.get('pred_total'),
            'edge_total': c.get('edge_total'),
            'rec_total_side': c.get('rec_total_side'),
            'rec_total_ev': c.get('rec_total_ev'),
        }
        print(row)
        shown += 1

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--season', type=int)
    p.add_argument('--week', type=int, default=1)
    p.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    main(args.season, args.week, args.limit)
