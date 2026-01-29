# NFL-Betting (Flask)

Server-rendered NFL predictions and betting recommendations app.

 Pages: cards, table, recommendations
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

  - RECS_MARKET_BLEND (default 0.50)
  - RECS_MARKET_BAND (default 0.10)
  - RECS_PROB_SHRINK (default 0.50)
  - RECS_EV_THRESH_LOW/MED/HIGH (defaults 4/8/15)
  - NFL_ATS_SIGMA (default 9.0)
  - NFL_TOTAL_SIGMA (default 10.0)
  - RECS_MIN_EV_PCT (default 3.0)
  - RECS_ONE_PER_GAME (true/false, default false)
  - Card display:
    - CARD_TOP_PROP_EDGES (default 5) — max prop edges attached per game card
    - CARD_PROP_EDGES_INCLUDE_LADDERS (default 0) — set to 1 to include ladder/alt props in card highlights
  - Simulation (Monte Carlo) (optional):
    - SIM_COMPUTE_ON_REQUEST (default 0) — set to 1 to let the Flask app compute sim_probs when missing (local/debug only)
    - SIM_N_SIMS_ON_REQUEST (default 1000) — number of sims for on-request sim_probs (clamped 200–20000)
    - SIM_PLAYS_PER_DRIVE (default 6.2) — plays-per-drive used to estimate possessions from team_plays (drive timeline)
    - SIM_DRIVES_PER_TEAM_DEFAULT (default 11) — default drives per team when team_plays unavailable (drive timeline)
    - SIM_DRIVES_PER_TEAM_MIN (default 8) — minimum drives per team clamp (drive timeline)
    - SIM_DRIVES_PER_TEAM_MAX (default 16) — maximum drives per team clamp (drive timeline)
  - Injury features (optional):
    - INJURY_BASELINE_WEEK (default 1) — baseline week used to define "starter" identity for injury starter-out metrics
    - SIM_MEAN_MARGIN_K_NEUTRAL (default 0.0) — neutral-site adjustment to margin mean
    - SIM_MEAN_TOTAL_K_PRECIP (default 0.0) — precip adjustment to total mean (points per 100% precip)
    - SIM_MEAN_TOTAL_K_COLD (default 0.0) — cold adjustment to total mean (points per 10F below SIM_COLD_TEMP_F)
    - SIM_COLD_TEMP_F (default 45.0) — cold threshold temperature in Fahrenheit
    - SIM_SIGMA_K_NEUTRAL (default 0.0) — sigma scaling when neutral-site
  - Totals calibration (optional; helps align model totals to market while keeping signal):
    - NFL_TOTAL_SCALE (default 1.0) — multiplicative scale applied to model total before blending
    - NFL_TOTAL_SHIFT (default 0.0) — additive shift (points) applied before blending

  - PRED_IGNORE_LOCKED (default 0) — set to 1 to ignore nfl_compare/data/predictions_locked.csv and prefer materialized predictions_week.csv (useful for historical backtests)
    - NFL_MARKET_TOTAL_BLEND (default 0.60) — 0=no market anchor, 1=fully market total
  - NFL_TOTAL_SCALE=1.03

  - `PROB_CALIBRATION_FILE`: Override probability calibration file used by the app (defaults to `nfl_compare/data/prob_calibration.json`).
#### Game model ensembles

- Set `GAME_MODEL_N_ENSEMBLE` (default 1) to train multiple estimators with different seeds and average their predictions.
  - Works for both XGBoost (if available) and sklearn fallbacks.

#### Supervised props adjustment (WR rec yards)

- Enable with `PROPS_USE_SUPERVISED=1` to train a light Ridge model on the last N completed weeks and adjust this week’s WR receiving yards.
  - `PROPS_ML_LOOKBACK` (default 4)
  - `PROPS_ML_BLEND` (default 0.5) — blend weight for ML prediction vs heuristic
  - Output CSV: `nfl_compare/data/player_props_ml_<season>_wk<week>.csv`
### Render Cron Job (server-side daily refresh)

If you want the live app to refresh odds, weather, predictions, and props daily without a redeploy, set up a Render Cron Job to call the built-in admin endpoint.

Two options:

1) Call the endpoint directly (no code changes required)

- Method: GET
- URL: `https://<your-app>.onrender.com/api/admin/daily-update?push=1&light=1&key=<ADMIN_KEY>`
- Schedule: daily at your preferred time (e.g., 10:00 UTC)
- Notes: `push=1` lets the server push updated data back to Git (optional).
  - You can omit `&light=1` if you set `LIGHT_DAILY_UPDATE=1` on the service.

2) Use the helper script in this repo

- Command: `python scripts/trigger_daily_update.py --base-url https://<your-app>.onrender.com --key <ADMIN_KEY> --push 1`
  - To opt into light mode, add `--light 1` or set `LIGHT_DAILY_UPDATE=1` on the service.
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

## Totals calibration (O/U)

We support a simple global totals calibration to offset/scale model totals and optionally blend with market totals.

- Config file: `nfl_compare/data/totals_calibration.json` with keys `{scale, shift, market_blend}`
- Env overrides: `NFL_TOTAL_SCALE`, `NFL_TOTAL_SHIFT`, `NFL_MARKET_TOTAL_BLEND`
- Optional: set `NFL_CALIB_DIR` to load calibration files from a different directory (shipped-only bundles).
- If `nfl_compare/data/calibration_active.json` exists (written by the daily pipeline), the app will prefer its `bundle_dir` automatically when `NFL_CALIB_DIR` is not set.

When present, the app adds `pred_total_cal` to the weekly view and prefers it for O/U edge and EV.

Fit and write calibration from the last 4 completed weeks:

```
python scripts/fit_totals_calibration.py --weeks 4
```

Optionally override output path:

```
python scripts/fit_totals_calibration.py --weeks 6 --out nfl_compare/data/totals_calibration.json
```

Validate it’s picked up:

- GET `/api/health/calibration` should show the loaded values
- `/api/debug-week-view` should include `pred_total_cal` rows when predictions are present

## Standardized backtests and weekly report

We emit backtest artifacts in a predictable folder and a simple markdown report you can commit or share.

- Output directory: `nfl_compare/data/backtests/<season>_wk<week>/`
  - `games_summary.csv` — single-row summary (n_games, MAE margin/total, home-win accuracy)
  - `games_details.csv` — per-game predictions vs actuals with residuals and market lines
  - `props_summary.csv` — MAE by position aggregated across evaluated weeks
  - `props_weekly.csv` — per-week MAE by position
  - `metrics.json` — compact combined metrics for quick consumption
- Report: `reports/weekly_report_<season>_wk<week>.md`

Quick run (evaluate through previous week):

```powershell
# Games
./.venv/Scripts/python.exe scripts/backtest_games.py --season 2025 --start-week 1 --end-week 9 --include-same-season --out-dir nfl_compare/data/backtests/2025_wk10
# Props
./.venv/Scripts/python.exe scripts/backtest_props.py --season 2025 --start-week 1 --end-week 9 --out-dir nfl_compare/data/backtests/2025_wk10
# Report
./.venv/Scripts/python.exe scripts/generate_weekly_report.py --season 2025 --week 10
```

The daily automation can run these when `DAILY_UPDATE_RUN_BACKTESTS=1`.

### Daily update (scenario artifacts)

The PowerShell automation `daily_update.ps1` can optionally produce deterministic scenario artifacts (game-level, drive-level, and scenario-adjusted player props) under:

- `nfl_compare/data/backtests/<season>_wk<week>/`
  - `sim_probs_scenarios.csv`, `sim_scenarios_meta.json`, optional `sim_drives_scenarios.csv`
  - `player_props_scenarios.csv` (+ summary/meta)
  - prior-week eval outputs (if actuals exist): `player_props_scenarios_accuracy*.csv` and `.md`

Key toggles:

- `DAILY_UPDATE_RUN_SCENARIOS` (default 1)
- `DAILY_UPDATE_SCENARIO_SET` (default `v2`), `DAILY_UPDATE_SCENARIO_N_SIMS` (default 2000)
- `DAILY_UPDATE_SCENARIO_DRIVES` (default 0)
- `DAILY_UPDATE_SCENARIOS_INCLUDE_PRIOR` (default 1)
- `DAILY_UPDATE_RUN_PROPS_SCENARIOS` (default 1)
- `DAILY_UPDATE_EVAL_PROPS_SCENARIOS` (default 1)

Roster cache (week-accurate rosters/actives):

- `DAILY_UPDATE_FETCH_ROSTERS` (default 1) — prefetch nfl_data_py seasonal/weekly rosters into `nfl_compare/data/external/nfl_data_py/`
- `DAILY_UPDATE_REFRESH_ROSTERS` (default 0) — overwrite roster caches (useful if data source updates)
- `DAILY_UPDATE_ROSTER_TIMEOUT_SEC` (default 30)

Weekly automation equivalents:

- `WEEKLY_UPDATE_FETCH_ROSTERS` (default 1)
- `WEEKLY_UPDATE_REFRESH_ROSTERS` (default 0)
- `WEEKLY_UPDATE_ROSTER_TIMEOUT_SEC` (default 30)

### Team-level Ratings
Lightweight team ratings (offense/defense/net margin) are computed as exponential moving averages of finalized games and attached to the weekly view and model features. They provide a stable prior signal and are aligned to avoid leakage (for week W, ratings use weeks < W).

- Env knobs:
  - `TEAM_RATING_EMA` (default 0.60): EMA alpha.
  - `TEAM_RATINGS_OFF` (default 0): disable attaching ratings in the app.
