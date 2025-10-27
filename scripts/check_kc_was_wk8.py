import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app import app
import json

def main():
    with app.test_client() as c:
        r = c.get('/api/cards?season=2025&week=8')
        js = r.get_json() or {}
        cards = js.get('cards') or js.get('data') or []
        target = None
        for x in cards:
            if x.get('home_team')=='Kansas City Chiefs' and x.get('away_team')=='Washington Commanders':
                target = x
                break
        if not target:
            print('Target game not found in Week 8 cards.')
            return
        out = {
            'home': target.get('home_team'),
            'away': target.get('away_team'),
            'game_id': target.get('game_id'),
            'market_spread_home': target.get('market_spread_home'),
            'market_total': target.get('market_total'),
            'spread_home': target.get('spread_home'),
            'close_spread_home': target.get('close_spread_home'),
            'moneyline_home': target.get('moneyline_home'),
            'moneyline_away': target.get('moneyline_away'),
        }
        print(json.dumps(out, ensure_ascii=False))

if __name__ == '__main__':
    main()
