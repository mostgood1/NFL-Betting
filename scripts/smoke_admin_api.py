import os
import json

# Set a test admin key in-process
os.environ['ADMIN_KEY'] = 'test-admin-key'

import app as app_module  # imports the Flask app
app = app_module.app

print('Loaded app:', bool(app))

with app.test_client() as c:
    # Expect 401 without key
    r1 = c.get('/api/admin/daily-update/status')
    print('GET /api/admin/daily-update/status ->', r1.status_code)
    try:
        print('Body:', r1.get_json())
    except Exception:
        print('Body (text):', r1.data.decode('utf-8', 'ignore')[:200])

    # Expect 200 with key
    r2 = c.get('/api/admin/daily-update/status?tail=3&key=test-admin-key')
    print('GET /api/admin/daily-update/status?key=*** ->', r2.status_code)
    j2 = r2.get_json()
    print('Keys:', list(j2.keys()) if isinstance(j2, dict) else type(j2))
    assert isinstance(j2, dict) and 'running' in j2 and 'logs' in j2, 'Unexpected payload'

print('Smoke OK')
