import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Ensure project root is on sys.path so `import app` works when invoked from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402

# Clear any env gates that could hide results in a smoke
for k in [
    'RECS_MIN_EV_PCT',
    'RECS_ONE_PER_GAME',
    'DISABLE_ON_REQUEST_PREDICTIONS',
]:
    os.environ.pop(k, None)

def main():
    with app.test_client() as c:
        r = c.get('/api/props/recommendations')
        print('status', r.status_code)
        if r.status_code != 200:
            print('ERROR: non-200 response body:', r.data[:500])
            return 2
        js = r.get_json() or {}
        season = js.get('season')
        week = js.get('week')
        data = js.get('data') or []
        print('season', season, 'week', week, 'rows', len(data))

        # Count duplicates by (player, team)
        key = lambda d: (((d.get('player') or '').strip().lower()), ((d.get('team') or '').strip().upper()))
        counts = Counter(key(d) for d in data)
        dupes = [(k, v) for k, v in counts.items() if v > 1]
        print('duplicate groups:', len(dupes))
        if dupes:
            # Show top 15 duplicate groups and their distinct markets/positions for debugging
            dupes.sort(key=lambda kv: kv[1], reverse=True)
            print('Top duplicates:')
            for (name, team), cnt in dupes[:15]:
                rows = [d for d in data if key(d) == (name, team)]
                pos_set = sorted({(d.get('position') or '').upper() for d in rows if d.get('position')})
                markets = sorted({(p.get('market') or '').lower() for d in rows for p in (d.get('plays') or []) if p})
                print(f'- {name}|{team}: {cnt} cards; positions={pos_set}; markets={markets[:8]}')

        # Basic assertion
        print('SMOKE OK')
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
