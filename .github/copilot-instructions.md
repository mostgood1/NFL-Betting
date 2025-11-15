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
| `CURRENT_SEASON`, `CURRENT_WEEK` | Force season/week | Overrides inference logic |
| `DISABLE_ON_REQUEST_PREDICTIONS` | Skip on-demand model attachment | Default true on Render |
| `RENDER` | Deployment flag (truthy on Render) | Used to gate expensive ops |
| `ADMIN_TOKEN` | Protect admin endpoints | Required for sensitive routes |
| Various `RECS_*`, `NFL_*_SIGMA` | Recommendation tuning | Used in EV/edge calculations |
| `RECS_MARKET_BLEND_MARGIN` | Blend model margin toward market-implied margin (-spread_home) for upcoming games | Default 0.0 (off); range 0–1 |
| `RECS_MARKET_BLEND_TOTAL` | Blend model total toward market total for upcoming games | Default 0.0 (off); range 0–1 |
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
