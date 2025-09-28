import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402

def main():
    with app.test_client() as c:
        r = c.get('/api/props/recommendations')
        js = r.get_json() or {}
        data = js.get('data') or []
        pairs = [((d.get('player') or '').strip().lower(), (d.get('team') or '').strip().upper()) for d in data]
        cnt = Counter(pairs)
        dups = [kv for kv, n in cnt.items() if n > 1]
        print('total cards:', len(data))
        print('duplicate pairs count:', len(dups))
        if not dups:
            print('no duplicates found')
            return 0
        # Show details for a few
        by_pair = defaultdict(list)
        for d in data:
            k = ((d.get('player') or '').strip().lower(), (d.get('team') or '').strip().upper())
            if k in dups:
                by_pair[k].append({
                    'event': d.get('event'),
                    'plays': len(d.get('plays') or []),
                    'projections_nonnull': sum(1 for k2, v in (d.get('projections') or {}).items() if v is not None),
                    'home_team': d.get('home_team'),
                    'away_team': d.get('away_team'),
                })
        for pair, rows in list(by_pair.items())[:5]:
            print('PAIR', pair)
            for row in rows:
                print('  -', row)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
