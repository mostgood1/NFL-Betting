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

Optional data refresh (training + predict):
- GET /api/refresh-data?train=true
- GET /api/refresh-odds (requires ODDS_API_KEY)

## Deploy on Render

This repo includes a Render blueprint (render.yaml). Two options:

1) Blueprint deploy
- In Render dashboard: New + → Blueprint → Give repo URL
- Set required environment variables
- Deploy

2) Manual service
- Create a new Web Service
- Runtime: Python
- Build Command:
  - pip install --upgrade pip
  - pip install -r requirements.txt
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
    - NFL_TOTAL_MARKET_BAND (default 5.0) — clamp final total within ±band points of market (0 disables clamp)

Typical balanced settings that reduced an “all unders” bias in testing:
  - NFL_TOTAL_SCALE=1.03
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
