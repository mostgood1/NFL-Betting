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

## Notes
- No external APIs are called by default; wire your providers in `src/data_sources.py`.
- Models saved in `./models/`.
- A simple baseline ships; swap in more advanced models as data allows.
