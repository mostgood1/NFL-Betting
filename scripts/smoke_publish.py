import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

# Conservative gates
os.environ['RECS_ALLOWED_MARKETS'] = 'MONEYLINE,SPREAD,TOTAL'
os.environ['RECS_ONE_PER_MARKET'] = 'true'
os.environ['RECS_ONE_PER_GAME'] = 'false'
os.environ['RECS_MIN_WP_DELTA'] = '0.12'
os.environ['RECS_MIN_EV_PCT_ML'] = '8.0'
os.environ['RECS_MIN_ATS_DELTA'] = '0.18'
os.environ['RECS_MIN_EV_PCT_ATS'] = '12.0'
os.environ['RECS_MIN_TOTAL_DELTA'] = '0.18'
os.environ['RECS_MIN_EV_PCT_TOTAL'] = '12.0'
os.environ['RECS_UPCOMING_CONF_MIN_ATS'] = 'High'
os.environ['RECS_UPCOMING_CONF_MIN_TOTAL'] = 'High'

with app.test_client() as c:
    r_api = c.get('/api/recommendations?season=2025&week=17&min_ev=8&one_per_game=false&per_market=3&min_conf=High')
    js = r_api.get_json() or {}
    print('API publish rows:', js.get('rows'))

    r_html = c.get('/publish?season=2025&week=17&min_ev=8&one_per_game=false&per_market=3&min_conf=High')
    print('HTML publish status:', r_html.status_code)
    txt = r_html.data.decode('utf-8', errors='ignore')
    print('HTML publish length:', len(txt))
