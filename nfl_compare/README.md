# NFL Compare

A self-contained NFL predictions and betting recommendations app, patterned after your NCAAF/MLB tools.

What’s included:
- Data pipeline: load historical results, team stats, and betting lines from CSV.
- Features: team strength ELO, pace, efficiency, QB adj, schedule-adjusted metrics.
- Models: score prediction (Poisson/GB), win prob, margin, O/U and spread outcomes, quarter/half splits baseline.
- UI: Streamlit dashboard mirroring MLB-Compare feel; matchup compare, projections, and picks.

Quick start
1. Create a venv and install requirements
2. Put CSVs into `./data/`
3. Train the model(s)
4. Launch the Streamlit UI

See detailed steps below.

## Data files (CSV)
Place in `./data/`:
- games.csv — historical game-level data
- team_stats.csv — per-team season/week stats
- lines.csv — betting lines and totals by game/week
- pbp_quarters.csv (optional) — quarterly scoring splits

Minimal schemas are in `src/schemas.py`. Extra columns will be ignored.

## How to run
- Install deps: `pip install -r requirements.txt`
- Train: `python -m src.train`
- Serve UI: `streamlit run ui/app.py`

### Real odds (Odds API)
- Set env var `ODDS_API_KEY` with your key. Optional: `ODDS_API_REGION` (default `us`), `ODDS_API_BOOKS` (comma list).
- Fetch and archive latest odds JSON: `python -m src.odds_api_client` (writes `data/real_betting_lines_YYYY_MM_DD.json`).
- The app will auto-load the latest `real_betting_lines_*.json` if present.

### Historical schedules and team stats (nflfastR)
- Install extras if not already: `pip install nfl-data-py pyarrow`
- Fetch schedules and team-week stats: `python -m src.fetch_nflfastr --range 2022-2025`
- Outputs `data/games.csv` and `data/team_stats.csv` using the project schemas.

### Weather (OpenWeather optional)
- Provide `data/stadium_meta.csv` with columns at least: `team,roof,surface,lat,lon,tz,altitude_ft`.
- Set env var `OPENWEATHER_API_KEY` (or `OPENWEATHER_KEY`).
- Fetch per-date weather snapshots: `python -m src.fetch_weather --date YYYY-MM-DD --kickoff-hour 16`.
- The app and features will consume `data/weather_YYYY-MM-DD.csv` automatically when present.

## Notes
- No external APIs are called by default; wire your providers in `src/data_sources.py`.
- Models saved in `./models/`.
- A simple baseline ships; swap in more advanced models as data allows.

## Player props additions
- EMA smoothing: `src/player_props.py` now blends a small EMA of team plays and pass rate (persisted snapshots `data/team_context_ema_<season>_wk<k>.csv`) using env `PROPS_EMA_BLEND` (0..1, default ~0.5).
- Position-vs-Defense: gentle WR/TE/RB defensive multipliers for target share and YPT are derived from season PBP through prior week and applied within clamps.
- Stability knobs: `PROPS_JITTER_SCALE`, `PROPS_OBS_BLEND`, and `PROPS_EMA_BLEND` can be tuned; daily_update.ps1 sets reasonable defaults.
