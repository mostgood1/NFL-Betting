import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import app` works when invoked from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

# Ensure default thresholds are in place for the smoke test
os.environ.pop('RECS_MIN_EV_PCT', None)
os.environ.pop('RECS_ONE_PER_GAME', None)

with app.test_client() as c:
    r = c.get('/api/recommendations')
    js = r.get_json()
    rows = js.get('rows') if isinstance(js, dict) else None
    print('API /api/recommendations rows:', rows)
    r_lo = c.get('/api/recommendations?min_ev=0')
    js_lo = r_lo.get_json()
    rows_lo = js_lo.get('rows') if isinstance(js_lo, dict) else None
    print('API /api/recommendations?min_ev=0 rows:', rows_lo)
    # Also hit HTML page
    r2 = c.get('/recommendations')
    print('HTML /recommendations status:', r2.status_code)
    txt = r2.data.decode('utf-8', errors='ignore')
    empty = 'No recommendations available.' in txt
    print('HTML shows empty message:', empty)
