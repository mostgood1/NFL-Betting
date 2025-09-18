import os
import sys
from pathlib import Path
from collections import Counter
import json

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402

# Targets
LINE_REQUIRED = {"rec_yards","rush_yards","pass_yards","receptions","pass_attempts","rush_attempts","rush_rec_yards","pass_rush_yards","targets"}
EV_ALLOWED = {"any_td","interceptions","pass_tds","multi_tds"}


def main():
    with app.test_client() as c:
        r = c.get('/api/props/recommendations')
        if r.status_code != 200:
            print('FAIL: /api/props/recommendations ->', r.status_code)
            return 2
        js = r.get_json() or {}
        data = js.get('data') or []
        print('cards', len(data))
        # 1) No duplicate (player, team)
        pairs = [((d.get('player') or '').strip().lower(), (d.get('team') or '').strip().upper()) for d in data]
        dupes = [kv for kv, cnt in Counter(pairs).items() if cnt > 1]
        if dupes:
            print('FAIL: duplicate player/team cards:', dupes[:10])
            return 3
        # 2) No player names including trailing 'total' artifacts
        bad_names = [d.get('player') for d in data if d.get('player') and ' total ' in d.get('player').lower()]
        if bad_names:
            print('FAIL: names with trailing "total":', bad_names[:10])
            return 4
        # 3) No blank line-required plays
        missing_lines = []
        for d in data:
            plays = d.get('plays') or []
            for p in plays:
                mk = str(p.get('market') or '').strip().lower()
                # Normalize to internal key heuristically
                mk_map = {
                    'receiving yards':'rec_yards','receptions':'receptions','rushing yards':'rush_yards','passing yards':'pass_yards',
                    'passing tds':'pass_tds','pass tds':'pass_tds','pass touchdowns':'pass_tds',
                    'passing attempts':'pass_attempts','pass attempts':'pass_attempts','rushing attempts':'rush_attempts','rush attempts':'rush_attempts',
                    'interceptions':'interceptions','interceptions thrown':'interceptions',
                    'rush+rec yards':'rush_rec_yards','rushing + receiving yards':'rush_rec_yards','rush + rec yards':'rush_rec_yards',
                    'pass+rush yards':'pass_rush_yards','pass + rush yards':'pass_rush_yards','passing + rushing yards':'pass_rush_yards',
                    'targets':'targets','2+ touchdowns':'multi_tds','anytime td':'any_td','any time td':'any_td'
                }
                key = mk_map.get(mk, mk)
                if key in LINE_REQUIRED:
                    ln = p.get('line')
                    if ln is None or (isinstance(ln, float) and getattr(__import__('math'), 'isnan')(ln)):
                        missing_lines.append((d.get('player'), key))
        if missing_lines:
            print('FAIL: line-required plays missing line:', missing_lines[:10])
            return 5
        print('SMOKE SANITY OK')
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
