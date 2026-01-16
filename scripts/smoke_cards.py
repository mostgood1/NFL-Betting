import os
import sys
import re
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

with app.test_client() as c:
    r = c.get('/')
    print('HTML / status:', r.status_code)
    html = r.data.decode('utf-8', errors='ignore')
    # Count game cards via data-game-id attribute within cards grid
    cards = re.findall(r'data-game-id=\"[^\"]+\"', html)
    print('HTML game cards:', len(cards))
    # Also check API cards for consistency
    ra = c.get('/api/cards')
    print('API /api/cards status:', ra.status_code)
    try:
        js = ra.get_json()
        rows = js.get('rows') if isinstance(js, dict) else None
    except Exception:
        rows = None
    print('API game cards rows:', rows)
