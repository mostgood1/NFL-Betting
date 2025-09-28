# NFL-Betting (Flask)

Server-rendered NFL predictions and betting recommendations app.

- Pages: cards, table, recommendations
- EV-based picks for Moneyline, Spread, Total with Low/Medium/High confidence
- Real odds support, venue overrides (neutral-site), weather display
- Deployed on Render (Gunicorn)

## Run locally

Requirements: Python 3.11+, pip, virtualenv recommended.

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py  # http://localhost:5050
```
- GET /api/refresh-odds (requires ODDS_API_KEY)


1) Blueprint deploy
- Set required environment variables
- Deploy

2) Manual service
- Start Command:
  - bash start.sh

### Environment variables

- FLASK_ENV=production
- PYTHONUNBUFFERED=1
- ODDS_API_KEY=... (for /api/refresh-odds)
- OPENWEATHER_API_KEY=... (only if you wire weather refresh)
- Optional tuning:
  - RECS_MARKET_BLEND (default 0.50)
  - RECS_MARKET_BAND (default 0.10)
  - RECS_PROB_SHRINK (default 0.50)
  - RECS_EV_THRESH_LOW/MED/HIGH (defaults 4/8/15)
  - NFL_ATS_SIGMA (default 9.0)
  - NFL_TOTAL_SIGMA (default 10.0)
  - RECS_MIN_EV_PCT (default 3.0)
  - RECS_ONE_PER_GAME (true/false, default false)
  - Totals calibration (optional; helps align model totals to market while keeping signal):
    - NFL_TOTAL_SCALE (default 1.0) — multiplicative scale applied to model total before blending
    - NFL_TOTAL_SHIFT (default 0.0) — additive shift (points) applied before blending
    - NFL_MARKET_TOTAL_BLEND (default 0.60) — 0=no market anchor, 1=fully market total
  - NFL_TOTAL_SCALE=1.03
### Render Cron Job (server-side daily refresh)

If you want the live app to refresh odds, weather, predictions, and props daily without a redeploy, set up a Render Cron Job to call the built-in admin endpoint.

Two options:

1) Call the endpoint directly (no code changes required)

- Method: GET
- URL: `https://<your-app>.onrender.com/api/admin/daily-update?push=1&key=<ADMIN_KEY>`
- Schedule: daily at your preferred time (e.g., 10:00 UTC)
- Notes: `push=1` lets the server push updated data back to Git (optional).

2) Use the helper script in this repo

- Command: `python scripts/trigger_daily_update.py --base-url https://<your-app>.onrender.com --key <ADMIN_KEY> --push 1`
- Requirements: Python + `requests` available in the Cron environment.
- You can also set env vars so the command shortens to `python scripts/trigger_daily_update.py`:
  - `ADMIN_BASE_URL` or `BASE_URL` → `https://<your-app>.onrender.com`
  - `ADMIN_KEY` → your admin key

Validation:

- Check `GET /api/admin/daily-update/status?tail=200&key=<ADMIN_KEY>` to view the latest run log.
- Check `GET /api/data-status` to verify the server sees updated `predictions.csv`, `weather_*.csv`, and `real_betting_lines_*.json` files.
  - NFL_TOTAL_SHIFT=1.8
  - NFL_MARKET_TOTAL_BLEND=0.70
  - NFL_TOTAL_MARKET_BAND=6.5

### Notes

- Start script creates data/model folders if missing.
- If `nfl_compare/data/predictions.csv` is absent, pages will render with an empty state until you run predictions.
- The app avoids Streamlit; Gunicorn serves Flask with Jinja templates.

## Repo structure

- app.py: Flask app and routes
- templates/: HTML templates
- nfl_compare/src: data pipeline (training, prediction, odds/weather clients)
- render.yaml, Procfile, start.sh: deployment
- requirements.txt: runtime dependencies
