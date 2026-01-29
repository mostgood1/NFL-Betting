# AI Assistant / Copilot Instructions for NFL-Betting

This document orients AI coding assistants working in this repository. It defines project context, data and model assumptions, coding conventions, and safe extension patterns. Follow it before proposing or applying changes.

---
## 1. Project Overview
A Flask-based web application plus analytics scripts that:
- Ingest & merge schedule (games), betting lines, weather, predictions, and player usage/efficiency priors
- Compute player props and betting edges (moneyline, spread, total)
- Render betting recommendation “cards” and player prop pages
- Provide reconciliation / validation endpoints (roster, props vs actuals)

Core logic lives in `app.py` (monolithic for now) and supporting code in `nfl_compare/src`. Data is stored in CSV / JSON / Parquet under `nfl_compare/data` (overridable via `NFL_DATA_DIR`).

---
## 2. Tech Stack
- Python 3.11
- Flask (single file app + Jinja templates under `templates/`)
- Pandas for data transforms
- Joblib for model object loading (if/when used)
- CSV/JSON/Parquet file storage (no DB currently)
- Deployment: Render (Procfile + `start.sh`), Gunicorn in production

---
## 3. Key Directories / Files
| Path | Purpose |
|------|---------|
| `app.py` | Main Flask routes, data loading, prediction attachment, card generation, props API |
| `nfl_compare/src` | Data sources, player props computation, priors, odds ingestion, weather, roster validation |
| `nfl_compare/data` | Canonical data store (games.csv, lines.csv, predictions.csv, player props caches, priors) |
| `scripts/` | One-off operational or inspection utilities (gen props, reconcile, debug recs, etc.) |
| `templates/` | Jinja HTML templates (index, props, recommendations, reconciliation) |
| `.github/copilot-instructions.md` | (This file) |

---
## 4. Data Files & Semantics
Core CSV / JSON assets (expected columns notable):
- `games.csv`: `season, week, game_id, date|game_date, home_team, away_team, home_score, away_score, ...`
- `lines.csv`: Betting market lines (`spread_home`, `total`, `moneyline_home`, `moneyline_away`, prices, open/close variants)
- `predictions.csv` (and `predictions_week.csv`, `predictions_locked.csv`): Model outputs and probabilistic metrics (prefixed `pred_`, `prob_`)
- `player_props_{season}_wk{week}.csv`: Cached computed props
- `nfl_team_assets.json`: Team metadata (logo/color/abbr)
- Priors / efficiency: `player_efficiency_priors.csv`, usage / EMA: `team_context_ema_{season}_wk{week}.csv`
- Depth / overrides: `depth_overrides.csv`, `predicted_not_active_{season}_wk{week}.csv`

Files may be partially absent; APIs must degrade gracefully.

---
## 5. Week / Season Inference & Fallbacks
- Primary inference uses `_infer_current_season_week` (considers explicit overrides via `current_week.json` or environment variables like `CURRENT_SEASON`, `CURRENT_WEEK`).
- Historically the UI broke when `games.csv` lacked future week rows while `predictions.csv` had them. A **fallback** was added in `_build_week_view`:
  1. Use filtered `games.csv` rows.
  2. Else synthesize from `lines.csv` (if present).
  3. Else synthesize minimal rows from `predictions` (season/week filtered).
- Always preserve original prediction metrics during merge.

When extending inference logic: avoid hard-coding week numbers; keep deterministic + override-friendly.

---
## 6. Player Props Pipeline (High-Level)
1. Load priors (usage, efficiency) & play-by-play historical stats
2. Adjust for week context (depth overrides, predicted inactive players)
3. Allocate target & rush shares; compute volume expectations
4. Apply efficiency priors -> yard, reception, TD projections
5. Write cache `player_props_{season}_wk{week}.csv`
6. API endpoints serve filtered JSON or CSV (with display name aliasing)

When modifying: maintain idempotence (reruns shouldn’t inflate). Avoid breaking column schema—front-end JS expects consistent keys.

---
## 7. Betting Card Logic
Located in `index()` route (`app.py`):
- Builds combined weekly view via `_build_week_view` + `_attach_model_predictions`.
- Computes model margin (`pred_home_points - pred_away_points`), predicted total, weather adjustments (precip & wind minor downward effect on totals if outdoors).
- Determines market vs model edges for spread & total, derives simple picks.
- EV / probability helpers exist (e.g., `_ev_from_prob_and_decimal`, `_american_to_decimal`).

If extending card metrics: keep calculations pure (no I/O), and add columns behind null checks. Provide safe guards for missing market lines.

---
## 8. Defensive Coding Patterns (Follow These)
- Never assume a file exists—always check `Path.exists()`.
- Wrap data loads in try/except; log once via `_log_once` for missing/exception states.
- Use numeric coercion (`pd.to_numeric(..., errors="coerce")`) for `season` and `week`.
- Preserve original data frames; operate on copies if mutating.
- Avoid hard-failing endpoints for partial data absence—return empty arrays with metadata (season/week) instead.
- Use layered fallbacks (games -> lines -> predictions -> assets) for any team/week lists.
- Keep new dependencies minimal; review if they belong in `requirements.txt` vs `requirements-web.txt`.

---
## 9. Naming & Style Conventions
- Functions: `snake_case`; internal helpers prefixed with underscore when not part of external API.
- Columns: prefer lowercase with underscores (`pred_home_points`), consistent prefixes (`pred_`, `prob_`).
- Keep route functions smallish; extract transformation helpers when adding complexity.
- Avoid global mutable state except controlled caches (`_recon_cache`).

---
## 10. Adding Routes / APIs
Checklist:
1. Validate authorization if admin-sensitive (reuse `_admin_auth_ok`).
2. Load data using existing helpers (`_load_predictions`, `_load_games`).
3. Apply fallbacks gracefully.
4. Return JSON with deterministic key ordering (Flask built-in is fine) and include context (`season`, `week`).
5. Avoid long-running model inference directly in request path (respect `DISABLE_ON_REQUEST_PREDICTIONS`).

---
## 11. Logging & Observability
- Use `_log_once(key, message)` for one-time warnings (missing file, parse error).
- Avoid noisy per-row logs; rely on health endpoint.
- New structured diagnostics can extend `/api/health/data`—keep response small (<15KB).

---
## 12. Environment Variables
| Variable | Purpose | Notes |
|----------|---------|-------|
| `NFL_DATA_DIR` | Override default data directory | Added for deployment flexibility |
| `SIM_SEED` | Deterministic seed for local simulation artifact generation | If set, sim engine uses it when no explicit seed is passed |
| `CURRENT_SEASON`, `CURRENT_WEEK` | Force season/week | Overrides inference logic |
| `DISABLE_ON_REQUEST_PREDICTIONS` | Skip on-demand model attachment | Default true on Render |
| `RENDER` | Deployment flag (truthy on Render) | Used to gate expensive ops |
| `ADMIN_TOKEN` | Protect admin endpoints | Required for sensitive routes |
| `PRED_IGNORE_LOCKED` | Ignore `predictions_locked.csv` when loading predictions | Default 0; set 1 for historical backtests/debug |
| `CARD_PROP_EDGES_INCLUDE_LADDERS` | Include ladder/alt player props in game card highlights | Default 0 (off); set to 1 to include |
| `INJURY_BASELINE_WEEK` | Baseline week to define "starter" identity for injury features | Default 1; higher values shift baseline later in season |
| Various `RECS_*`, `NFL_*_SIGMA` | Recommendation tuning | Used in EV/edge calculations |
| `DAILY_UPDATE_BUILD_MANIFEST` | Build per-week `manifests/{season}_wk{week}.json` during `daily_update.ps1` | Default on; set 0 to skip |
| `DAILY_UPDATE_PUBLISH_CAL_BUNDLE` | Snapshot calibration JSONs into a versioned bundle + write `calibration_active.json` | Default on; set 0 to skip |
| `DAILY_UPDATE_RUN_SCENARIOS` | Run scenario simulation (`scripts/simulate_scenarios.py`) for current week (and optionally prior week) | Default on; set 0 to skip |
| `DAILY_UPDATE_SCENARIO_SET` | Scenario set name passed to `simulate_scenarios.py --scenario-set` | Default `v2` |
| `DAILY_UPDATE_SCENARIO_N_SIMS` | Number of Monte Carlo sims per scenario | Default `2000` (min 200) |
| `DAILY_UPDATE_SCENARIO_DRIVES` | Also generate drive-level scenario artifacts (`--drives`) | Default `0` (off) |
| `DAILY_UPDATE_SCENARIOS_INCLUDE_PRIOR` | Also generate scenario artifacts for prior week (enables prior-week eval) | Default on |
| `DAILY_UPDATE_RUN_PROPS_SCENARIOS` | Generate scenario-adjusted player props (`scripts/simulate_player_props_scenarios.py`) | Default on |
| `DAILY_UPDATE_PROPS_SCENARIOS_BASELINE_ID` | Override baseline scenario id used for player-props scaling | Default inferred (e.g. `v2_baseline`) |
| `DAILY_UPDATE_EVAL_PROPS_SCENARIOS` | Evaluate scenario-adjusted props vs actuals for prior week (`scripts/player_props_scenarios_accuracy.py`) | Default on |
| `DAILY_UPDATE_FETCH_ROSTERS` | Prefetch nfl_data_py seasonal/weekly rosters into local cache (`scripts/fetch_rosters_cache.py`) | Default on; set 0 to skip |
| `DAILY_UPDATE_REFRESH_ROSTERS` | Force refresh/overwrite roster caches during daily update | Default off |
| `DAILY_UPDATE_ROSTER_TIMEOUT_SEC` | Timeout (seconds) used when fetching roster caches | Default 30 |
| `WEEKLY_UPDATE_FETCH_ROSTERS` | Prefetch nfl_data_py rosters into cache during `weekly_update.ps1` | Default on; set 0 to skip |
| `WEEKLY_UPDATE_REFRESH_ROSTERS` | Force refresh/overwrite roster caches during weekly update | Default off |
| `WEEKLY_UPDATE_ROSTER_TIMEOUT_SEC` | Timeout (seconds) used when fetching roster caches in weekly update | Default 30 |
| `RECS_MARKET_BLEND_MARGIN` | Blend model margin toward market-implied margin (-spread_home) for upcoming games | Default 0.0 (off); range 0–1 |
| `RECS_MARKET_BLEND_TOTAL` | Blend model total toward market total for upcoming games | Default 0.0 (off); range 0–1 |
| `RECS_UPCOMING_CONF_MIN_ATS` | Minimum confidence tier to publish upcoming ATS picks | Default High; values: Low, Medium, High |
| `RECS_UPCOMING_CONF_MIN_TOTAL` | Minimum confidence tier to publish upcoming Total picks | Default High; values: Low, Medium, High |
| `RECS_UPCOMING_MIN_EV_PCT_ATS` | Minimum EV percent to publish upcoming ATS picks | Default 0.0 (off); suggest 18.0–25.0 for precision |
| `RECS_UPCOMING_MIN_EV_PCT_TOTAL` | Minimum EV percent to publish upcoming Total picks | Default 0.0 (off); suggest 20.0–28.0 for precision |
| `RECS_USE_MC_PROBS_ATS` | Use Monte Carlo probabilities for ATS selected-side gating | Default 0 (off); reads sim_probs.csv from nfl_compare/data/backtests/{season}_wk{week} |
| `RECS_USE_MC_PROBS_TOTAL` | Use Monte Carlo probabilities for Totals selected-side gating | Default 0 (off); reads sim_probs.csv from nfl_compare/data/backtests/{season}_wk{week} |
| `RECS_USE_MC_PROBS_ML` | Use Monte Carlo probabilities for Moneyline selected-side gating | Default 0 (off); reads sim_probs.csv from nfl_compare/data/backtests/{season}_wk{week} |
| `RECS_MIN_P_ML` | Minimum selected-side probability for Moneyline gating | Default 0.0 (off); suggest 0.60–0.65 for precision |
| `RECS_MIN_P_ATS` | Minimum selected-side probability for ATS gating | Default 0.0 (off); suggest 0.80–0.90 for high-confidence coverage |
| `RECS_MIN_P_TOTAL` | Minimum selected-side probability for Totals gating | Default 0.0 (off); suggest 0.55–0.65 to retain coverage |
| `RECS_PROB_GATE_AND` | Require both MC and classifier probabilities to exceed min thresholds | Default 0 (off); applies to ML/ATS/TOTAL when enabled |
| `SIM_SIGMA_K_ROOF_CLOSED` | Per-game sigma multiplier for closed roof | Default -0.08 (reduces variance indoors) |
| `SIM_ATS_SIGMA_K_WIND` | ATS sigma wind sensitivity | Default 0.015 (per 10mph wind_open) |
| `SIM_TOTAL_SIGMA_K_WIND` | Total sigma wind sensitivity | Default 0.025 (per 10mph wind_open) |
| `SIM_SIGMA_K_REST` | ATS sigma rest-days differential sensitivity | Default 0.02 (per 7 days) |
| `SIM_TOTAL_SIGMA_K_LINE` | Total sigma sensitivity to total line level | Default 0.004 |
| `SIM_ATS_SIGMA_MIN` | Lower clamp for per-game ATS sigma | Default 7.0 |
| `SIM_ATS_SIGMA_MAX` | Upper clamp for per-game ATS sigma | Default 20.0 |
| `SIM_TOTAL_SIGMA_MIN` | Lower clamp for per-game Total sigma | Default 6.0 |
| `SIM_TOTAL_SIGMA_MAX` | Upper clamp for per-game Total sigma | Default 20.0 |
| `SIM_MARKET_BLEND_MARGIN` | Blend simulation margin mean toward market-implied (-spread_home) | Default 0.0 (off); range 0–1 |
| `SIM_MARKET_BLEND_TOTAL` | Blend simulation total mean toward market `total`/`close_total` | Default 0.0 (off); range 0–1 |
| `SIM_MEAN_TOTAL_K_WIND` | Adjustment to total mean per 10mph wind_open | Default -0.5 |
| `SIM_MEAN_MARGIN_K_REST` | Adjustment to margin mean per 7-day rest differential | Default 0.15 |
| `SIM_MEAN_MARGIN_K_ELO` | Adjustment to margin mean per 100 Elo diff (home-away) | Default 0.08 |
| `SIM_MEAN_TOTAL_K_INJ` | Adjustment to total mean per starter out (home+away) | Default -0.40 |
| `SIM_MEAN_TOTAL_K_DEFPPG` | Adjustment to total mean per 5 PPG vs league average defensive PPG | Default -0.40 |
| `SIM_MEAN_TOTAL_K_PRESSURE` | Adjustment to total mean per 0.05 above baseline combined defensive sack rate | Default -0.80 |
| `SIM_PRESSURE_BASELINE` | Baseline defensive sack rate used for pressure adjustments | Default 0.065 |
| `PROB_CALIBRATION_FILE` | Override probability calibration file used by `_apply_prob_calibration` | Default `nfl_compare/data/prob_calibration.json` |
| `NFL_CALIB_DIR` | Override directory for shipped calibration artifacts | If set, loads `sigma_calibration.json`, `totals_calibration.json`, and `prob_calibration.json` from this folder. If unset, the app/sim will prefer `nfl_compare/data/calibration_active.json` (its `bundle_dir`) when present |
| `SIM_MEAN_MARGIN_K_RATING` | Adjustment to margin mean per 1.0 of EMA net margin differential (home-away) | Default 0.08 |
| `SIM_CORR_MARGIN_TOTAL` | Correlation between simulated margin and total draws | Default 0.10 (range -0.3–0.3) |
| `SIM_COMPUTE_ON_REQUEST` | Allow Flask app to compute sim_probs when missing | Default 0 (off); set 1 for local/debug only |
| `SIM_N_SIMS_ON_REQUEST` | Number of Monte Carlo sims for on-request sim_probs | Default 1000; clamp 200–20000 |
| `SIM_PLAYS_PER_DRIVE` | Plays-per-drive used to estimate possessions from `team_plays` (drive timeline) | Default 6.2 |
| `SIM_DRIVES_PER_TEAM_DEFAULT` | Default drives per team when `team_plays` unavailable (drive timeline) | Default 11 |
| `SIM_DRIVES_PER_TEAM_MIN` | Minimum drives per team clamp (drive timeline) | Default 8 |
| `SIM_DRIVES_PER_TEAM_MAX` | Maximum drives per team clamp (drive timeline) | Default 16 |
| `SIM_DRIVE_TIME_MULT_SIGMA` | Lognormal sigma for per-drive duration multiplier (drive timeline) | Default 0.45 |
| `SIM_DRIVE_TIME_CAP_SEC` | Upper cap for simulated drive TOP seconds (drive timeline) | Default 780 |
| `SIM_DRIVE_MODE_SCORE_MIN` | Minimum `p_drive_score` to label a drive as TD/FG in `drive_outcome_mode` (drive timeline) | Default 0.28 |
| `SIM_MEAN_MARGIN_K_NEUTRAL` | Neutral-site adjustment to margin mean | Default 0.0 (off) |
| `SIM_MEAN_TOTAL_K_PRECIP` | Precip adjustment to total mean (points per 100% precip) | Default 0.0 (off) |
| `SIM_MEAN_TOTAL_K_COLD` | Cold adjustment to total mean (points per 10F below `SIM_COLD_TEMP_F`) | Default 0.0 (off) |
| `SIM_COLD_TEMP_F` | Cold threshold temperature in Fahrenheit | Default 45.0 |
| `SIM_SIGMA_K_NEUTRAL` | Sigma scaling when neutral-site | Default 0.0 (off) |
| `TEAM_RATING_EMA` | Exponential smoothing alpha for team ratings (off/def/net) | Default 0.60 (range 0–1) |
| `TEAM_RATINGS_OFF` | Disable attaching team ratings to weekly view/features | Default 0 (off/false) |
| `PROPS_QB_RZ_BASE` | Baseline QB red-zone rush rate for biasing | Default 0.10; affects QB any-time TD via RZ bias |
| `PROPS_QB_RZ_CAP` | Max QB RZ rush bias multiplier | Default 1.30 (was 1.25 hard-coded) |
| `PROPS_QB_RZ_SHARE_SCALE` | Scale to convert QB non-RZ rush share to RZ rush share | Default 0.95 (was 0.80 hard-coded) |
| `PROPS_QB_RZ_SHARE_MIN` | Minimum QB RZ rush share | Default 0.005 |
| `PROPS_QB_RZ_SHARE_MAX` | Maximum QB RZ rush share | Default 0.20 |
| `PROPS_ENFORCE_TEAM_USAGE` | Scale per-team targets/rush attempts to match team totals | Default 0 (off) |
| `PROPS_ENFORCE_TEAM_YARDS` | Scale per-team rec_yards/rush_yards to match team totals | Default 0 (off) |
| `PROPS_ENFORCE_TEAM_TDS` | Scale per-team rec_tds/rush_tds to match team totals | Default 0 (off) |
| `PROPS_TEAM_USAGE_SCALE_MIN` | Lower bound for usage scaling factor | Default 0.80 |
| `PROPS_TEAM_USAGE_SCALE_MAX` | Upper bound for usage scaling factor | Default 1.20 |
| `PROPS_TEAM_RECV_YDS_SCALE_MIN` | Lower bound for rec_yards scaling | Default 0.60 |
| `PROPS_TEAM_RECV_YDS_SCALE_MAX` | Upper bound for rec_yards scaling | Default 1.60 |
| `PROPS_TEAM_RUSH_YDS_SCALE_MIN` | Lower bound for rush_yards scaling | Default 0.60 |
| `PROPS_TEAM_RUSH_YDS_SCALE_MAX` | Upper bound for rush_yards scaling | Default 1.60 |
| `PROPS_TEAM_TDS_SCALE_MIN` | Lower bound for TDs scaling | Default 0.80 |
| `PROPS_TEAM_TDS_SCALE_MAX` | Upper bound for TDs scaling | Default 1.20 |
| `PROPS_POS_WR_REC_YDS` | Multiplier for WR receiving yards projections | Default 1.02 (range 0.80–1.20) |
| `PROPS_POS_TE_REC_YDS` | Multiplier for TE receiving yards projections | Default 0.98 (range 0.80–1.20) |
| `PROPS_POS_TE_REC` | Multiplier for TE receptions projections | Default 0.98 (range 0.80–1.20) |
| `PROPS_POS_QB_PASS_YDS` | Multiplier for QB passing yards projections | Default 0.97 (range 0.70–1.20) |
| `PROPS_POS_QB_PASS_TDS` | Multiplier for QB passing TD projections | Default 1.02 (range 0.70–1.20) |
| `PROPS_QB_TD_RATE_HI` | Upper cap for elite QB per-attempt TD rate prior | Default 0.075 (range 0.05–0.10) |
| `PROPS_QB_PRIOR_WEIGHT` | Blend weight for QB passing priors | Default 0.35 (range 0.0–0.8) |
| `PROPS_CALIB_ALPHA` | Calibration weight for volume (targets/rush attempts) from prior-week reconciliation | Default 0.35 (0–1) |
| `PROPS_CALIB_BETA` | Calibration weight for yards-per-volume (ypt/ypc proxy) | Default 0.30 (0–1) |
| `PROPS_CALIB_GAMMA` | Calibration weight for receptions via catch rate | Default 0.25 (0–1) |
| `PROPS_CALIB_QB` | Calibration weight for QB pass attempts/yards/TD/INT | Default 0.40 (0–1) |
| `PROPS_JITTER_SCALE` | Scale 0–1 to reduce/increase random jitter in outputs | Default 1.0 |
| `PROPS_OBS_BLEND` | Observed share blend for weeks > 1 (0–1) | Default 0.45 |
| `PROPS_EMA_BLEND` | EMA blend for team pass rate and plays (0–1) | Default 0.50 |
| `PROPS_QB_SHARE_MIN` | Lower clamp for QB rush share when deriving from priors | Default 0.015 |
| `PROPS_QB_SHARE_MAX` | Upper clamp for QB rush share when deriving from priors | Default 0.28 |
| `PROPS_QB_SHARE_DEFAULT` | Default QB rush share when no priors found | Default 0.07 |
| `PROPS_QB_OBS_BLEND` | Weight to blend observed SoD QB rush_share into week>1 estimates | Default 0.60 |
| `LEAGUE_RZ_PASS_RATE` | Baseline league red-zone pass rate used for scaling | Default 0.52 (range 0.30–0.80) |
| `PROPS_RZ_SHARE_W` | Weight to tilt WR/TE vs RB target shares by RZ pass tendency | Default 0.20 (0–0.50) |
| `PROPS_RZ_TD_W` | Weight to tilt team/QB pass TD expectation by RZ pass tendency | Default 0.30 (0–0.60) |
| `PROPS_PRESSURE_YPT_W` | Weight to scale YPT by opponent defensive sack rate (pressure) | Default 0.12 (0–0.50) |

When adding new env vars: document them here and in `README.md`.

---
## 13. Testing / Smoke Scripts
Existing helpful scripts:
- `scripts/gen_props.py <season> <week>`: Generate props cache.
- `scripts/inspect_week1.py`, `scripts/inspect_pos_counts.py`: Exploratory checks.
- `scripts/debug_recs.py`, `scripts/inspect_view.py`: Inspect view merge and recommendations.

Recommended quick smoke flow after changes:
```bash
python scripts/inspect_view.py          # Expect >0 rows
curl http://localhost:5000/api/props/teams?season=2025&week=1
curl http://localhost:5000/api/health/data
```
Add a new smoke script only if it provides unique coverage.

---
## 14. Safe Extension Examples
Do (good):
```python
# Add new prediction feature column if available
if 'pred_pressure_rate' in df.columns:
    out['pred_pressure_rate'] = df['pred_pressure_rate']
```
Don’t (bad):
```python
# Crash when column missing
out['pred_pressure_rate_normalized'] = df['pred_pressure_rate'] / 100.0  # KeyError risk
```

Do (good):
```python
p = df.get('prob_home_win')
if p is not None and pd.notna(p):
    edge = p - implied
```

---
## 15. Performance Notes
- Data sizes are modest; avoid O(n^2) merges or per-row Python loops when vectorizable.
- Use column selection before merge to reduce memory overhead.
- Delay expensive operations until absolutely required (lazy attachment of model predictions already implemented).

---
## 16. Deployment Considerations
- Gunicorn + Render: avoid introducing heavy warm-up code in module import scope.
- Any new on-disk artifacts should be written atomically (write to temp then `Path.rename`).
- Keep health endpoint fast (<100ms) & side-effect free.

---
## 17. Adding New Data Sources
1. Create loader in `nfl_compare/src/data_sources.py` (return DataFrame, coercing core numeric types).
2. Document expected columns.
3. Integrate into `_attach_model_predictions` if it enriches game view.
4. Add a single once-log for missing file.

---
## 18. Error Handling Philosophy
- Prefer empty sets / informative metadata to 500 errors for partial data lapses.
- Only raise (or return 500) on internal logic corruption, not on missing optional files.
- Keep user-facing JSON stable; add new keys instead of renaming existing ones.

---
## 19. Security & Safety
- Respect `ADMIN_TOKEN` for state-mutating endpoints.
- Never echo sensitive environment vars in logs.
- Avoid arbitrary shell execution; if adding system calls, sanitize inputs.

---
## 20. AI Assistant DO / DON'T Recap
DO:
- Check for file existence.
- Use existing helpers & naming patterns.
- Add fallbacks instead of hard failures.
- Keep diffs minimal & focused.
- Update this doc + README when introducing new env vars or core files.
DON'T:
- Introduce large dependencies without justification.
- Remove existing fallbacks.
- Refactor broadly inside a bug fix.
- Duplicate logic already present in helpers.

---
## 21. Future Refactor Targets (Non-blocking)
- Split `app.py` into blueprint modules (cards, props, admin) once churn stabilizes.
- Consolidate prediction merge logic into dedicated service class.
- Add lightweight persistence (SQLite or DuckDB) for historical queries.
- Introduce pytest-based regression suite for key endpoints.

---
## 22. Contact / Ownership
- Primary maintainer: (fill in) – Update when team changes.
- For urgent production issues: prioritize fallback robustness over schema purity.

---
*End of AI Assistant Guidance.*
