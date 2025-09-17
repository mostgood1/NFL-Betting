from __future__ import annotations

"""
Daily Updater

Steps:
- Determine current season and week from data/games.csv
- Get finals for any games played in the current week (via nfl_data_py if available)
- Update front end data (games.csv) with final scores
- Fetch updated betting odds (current + upcoming week covered by snapshot)
- Update weather for current and upcoming week games only (with backups)
- Confirm updated data presence
- Re-run predictions for unplayed games this week and next week

Usage (from repo root or nfl_compare):
  python -m nfl_compare.src.daily_updater
"""

import os
from datetime import date as _date, timedelta, datetime
from pathlib import Path
import shutil
from typing import Optional, Tuple

import pandas as pd

from .config import load_env
from .weather import DATA_DIR
from .auto_update import update_weather_for_date
from .odds_api_client import main as fetch_odds_main
try:
    from .predict import main as predict_main  # normal package context
except Exception:  # pragma: no cover - fallback when executed from repo root without installation
    # Inject parent directory so 'nfl_compare' becomes importable
    import sys
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.append(str(here))
    try:
        from nfl_compare.src.predict import main as predict_main  # type: ignore
    except Exception as _e:  # final fallback to relative path import
        try:
            from predict import main as predict_main  # type: ignore
        except Exception:
            raise ImportError(f"Could not import predict_main: {_e}")
from joblib import load as joblib_load
from .data_sources import load_games as ds_load_games, load_team_stats, load_lines
from .features import merge_features
from .weather import load_weather_for_games
from .models import predict as model_predict
from .data_sources import DATA_DIR as _DATA_DIR
from .data_sources import _try_load_latest_real_lines as _latest_json  # type: ignore
from .team_normalizer import normalize_team_name
def _import_backfill():
    try:
        from nfl_compare.scripts.backfill_close_lines import backfill_close_fields  # type: ignore
        return backfill_close_fields
    except Exception:
        # Attempt local path injection
        import sys
        here = Path(__file__).resolve().parents[1]
        scripts_dir = here / 'scripts'
        src_dir = here / 'src'
        for p in (scripts_dir, src_dir):
            sp = str(p)
            if sp not in sys.path:
                sys.path.append(sp)
        try:
            from backfill_close_lines import backfill_close_fields  # type: ignore
            return backfill_close_fields
        except Exception as e:
            raise ImportError(f'Could not import backfill_close_lines: {e}')

backfill_close_fields = _import_backfill()

# Locked predictions file path
LOCKED_FP = DATA_DIR / 'predictions_locked.csv'


def _print_guardrail(message: str) -> None:
    try:
        print(message)
    except Exception:
        # Fail-safe: avoid crashing updater on logging
        pass


def check_features_guardrail_for_week(season: int, week: int) -> None:
    """Guardrail: Build features for the given week and warn if per-team stats
    failed to attach (all-NaN) or if key diff features show near-zero variance.

    This helps catch future-week feature collapse (e.g., missing prior-week team_stats
    or an attach regression) before running predictions.
    """
    try:
        games = ds_load_games()
    except Exception as e:
        _print_guardrail(f"[GUARDRAIL] Skipped — could not load games.csv: {e}")
        return
    try:
        ts = load_team_stats()
    except Exception:
        ts = None
    try:
        lines = load_lines()
    except Exception:
        lines = pd.DataFrame()

    try:
        g_slice = games[(games.get('season').astype('Int64') == int(season)) & (games.get('week').astype('Int64') == int(week))].copy()
    except Exception:
        g_slice = pd.DataFrame(columns=['season','week','game_id','home_team','away_team','date'])

    n_games = len(g_slice)
    if n_games == 0:
        _print_guardrail(f"[GUARDRAIL] Season {season} Week {week}: no games found; skipping feature variance check.")
        return

    try:
        wx = load_weather_for_games(g_slice)
    except Exception:
        wx = pd.DataFrame()

    try:
        feats = merge_features(g_slice, ts if ts is not None else pd.DataFrame(), lines if lines is not None else pd.DataFrame(), wx)
    except Exception as e:
        _print_guardrail(f"[GUARDRAIL] Season {season} Week {week}: feature merge failed: {e}")
        return

    # Check per-team attach non-null counts
    home_nonnull = int(feats['home_off_epa'].notna().sum()) if 'home_off_epa' in feats.columns else 0
    away_nonnull = int(feats['away_off_epa'].notna().sum()) if 'away_off_epa' in feats.columns else 0

    # Compute std for key diffs
    diff_cols = ['off_epa_diff','def_epa_diff','pace_secs_play_diff','pass_rate_diff','rush_rate_diff','qb_adj_diff','sos_diff']
    stds = {}
    for c in diff_cols:
        if c in feats.columns:
            try:
                s = pd.to_numeric(feats[c], errors='coerce')
                stds[c] = float(s.std(ddof=0)) if len(s) > 0 else float('nan')
            except Exception:
                stds[c] = float('nan')

    # Determine warning conditions
    all_nan_attach = (home_nonnull == 0 and away_nonnull == 0)
    # Threshold for "near-zero" variance — generous; zero or extremely tiny dispersion is suspicious
    near_zero_threshold = 1e-6
    present_stds = [v for v in stds.values() if pd.notna(v)]
    low_variance_all = (len(present_stds) > 0) and all(abs(v) <= near_zero_threshold for v in present_stds)

    # Always print a concise summary
    std_summary = ', '.join(f"{k}={v:.6f}" for k, v in stds.items() if pd.notna(v)) or 'no-diffs'
    _print_guardrail(f"[GUARDRAIL] Season {season} Week {week}: games={n_games}, home_off_epa_nonnull={home_nonnull}, away_off_epa_nonnull={away_nonnull}, stds: {std_summary}")

    if all_nan_attach:
        _print_guardrail(f"[GUARDRAIL][WARNING] Season {season} Week {week}: prior-week team stats did not attach (all NaN). Check team_stats.csv for week {int(week)-1} and attachment logic.")
    elif low_variance_all:
        _print_guardrail(f"[GUARDRAIL][WARNING] Season {season} Week {week}: key diff features show near-zero variance. This may indicate attachment failure or identical inputs.")


def _week_has_started(season: int, week: int) -> bool:
    try:
        g = _load_games()
        if g is None or g.empty:
            return False
        gg = g[(g.get('season').astype('Int64') == int(season)) & (g.get('week').astype('Int64') == int(week))].copy()
        if gg is None or gg.empty:
            return False
        gg['date'] = pd.to_datetime(gg.get('date'), errors='coerce').dt.date
        today = _date.today()
        return bool((gg['date'].notna()) & (gg['date'] <= today)).any()
    except Exception:
        return False


def _lock_props_week(season: int, week: int) -> None:
    try:
        csv_fp = DATA_DIR / f"player_props_{int(season)}_wk{int(week)}.csv"
        lock_fp = DATA_DIR / f"props_lock_{int(season)}_wk{int(week)}.lock"
        if not lock_fp.exists():
            lock_fp.write_text('locked')
        # Write a frozen copy if the source CSV exists
        if csv_fp.exists():
            frozen_fp = DATA_DIR / f"player_props_{int(season)}_wk{int(week)}.locked.csv"
            try:
                df = pd.read_csv(csv_fp)
                df.to_csv(frozen_fp, index=False)
            except Exception:
                pass
    except Exception:
        pass


def _load_games() -> pd.DataFrame:
    fp = DATA_DIR / 'games.csv'
    if not fp.exists():
        raise FileNotFoundError('data/games.csv not found')
    df = pd.read_csv(fp)
    # Ensure expected columns exist
    for c in ['season','week','date','home_team','away_team','game_id','home_score','away_score']:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _save_games(df: pd.DataFrame) -> None:
    fp = DATA_DIR / 'games.csv'
    # Safety backup
    try:
        if fp.exists():
            shutil.copy2(fp, DATA_DIR / 'games.backup.csv')
    except Exception:
        pass
    df.to_csv(fp, index=False)


def _infer_current_season_week(games: pd.DataFrame) -> Tuple[int, int, Optional[int]]:
    today = _date.today()
    g = games.copy()
    g['date'] = pd.to_datetime(g['date'], errors='coerce').dt.date
    # Choose season with games near today; fallback to max season
    seasons = sorted(g['season'].dropna().astype(int).unique())
    if not seasons:
        raise RuntimeError('No seasons found in games.csv')
    # Score proximity by min days between today and that season's game dates
    def _season_score(s:int) -> int:
        sd = g[g['season']==s]['date'].dropna()
        if sd.empty:
            return 10**6
        return int(min(abs((d - today).days) for d in sd))
    season = min(seasons, key=_season_score)

    gs = g[g['season']==season].copy()
    if gs.empty:
        return season, 1, 2
    wk_ranges = (
        gs.groupby('week')['date']
          .agg(['min','max'])
          .dropna()
          .reset_index()
          .rename(columns={'min':'start','max':'end'})
    )
    # Normalize to date
    wk_ranges['start'] = pd.to_datetime(wk_ranges['start'], errors='coerce').dt.date
    wk_ranges['end'] = pd.to_datetime(wk_ranges['end'], errors='coerce').dt.date
    # Current week: window padded by 1 day on each side
    cur_row = wk_ranges[(wk_ranges['start'] - timedelta(days=1) <= today) & (today <= wk_ranges['end'] + timedelta(days=1))]
    if cur_row.empty:
        # Next upcoming week with start >= today
        up_row = wk_ranges[wk_ranges['start'] >= today].sort_values('start').head(1)
        if up_row.empty:
            # Late season; choose last
            cur_week = int(wk_ranges['week'].max())
        else:
            cur_week = int(up_row['week'].iloc[0])
    else:
        cur_week = int(cur_row['week'].iloc[0])
    # Upcoming week is the next one with start after current end
    try:
        cur_end = wk_ranges.loc[wk_ranges['week']==cur_week, 'end'].iloc[0]
        next_rows = wk_ranges[wk_ranges['start'] > cur_end].sort_values('start')
        up_week = int(next_rows['week'].iloc[0]) if not next_rows.empty else None
    except Exception:
        up_week = None
    return int(season), int(cur_week), (int(up_week) if up_week is not None else None)


def _update_finals_for_current_week(season: int, week: int) -> int:
    """Fetch updated finals/scores for current week using nfl_data_py schedules.
    Returns number of games updated.
    """
    try:
        # Lazy import so environments without the package still work
        from .fetch_nflfastr import fetch_games as _fetch_games
    except Exception as e:
        print(f"Finals update skipped (nfl_data_py not available): {e}")
        return 0
    try:
        sched = _fetch_games([int(season)])
    except Exception as e:
        print(f"Finals update failed to fetch schedules: {e}")
        return 0

    # Filter to this week and games with scores present (played)
    sched['date'] = pd.to_datetime(sched.get('date'), errors='coerce').dt.date
    today = _date.today()
    played = sched[(sched['week'] == week) & (sched['date'] <= today)]
    if played.empty:
        return 0

    local = _load_games()
    before = local.copy()

    # Merge by game_id when available, else by (date, home, away)
    updated = 0
    if 'game_id' in local.columns and 'game_id' in played.columns:
        m = local.set_index('game_id')
        for _, r in played.iterrows():
            gid = r.get('game_id')
            if pd.isna(gid) or gid not in m.index:
                continue
            hs = r.get('home_score'); as_ = r.get('away_score')
            if pd.notna(hs) or pd.notna(as_):
                # Only update if new info appears (was NA before)
                prev_hs = m.at[gid, 'home_score'] if 'home_score' in m.columns else pd.NA
                prev_as = m.at[gid, 'away_score'] if 'away_score' in m.columns else pd.NA
                if (pd.isna(prev_hs) and pd.notna(hs)) or (pd.isna(prev_as) and pd.notna(as_)):
                    m.at[gid, 'home_score'] = hs
                    m.at[gid, 'away_score'] = as_
                    updated += 1
        local = m.reset_index()
    else:
        key_cols = ['date','home_team','away_team']
        local['date'] = pd.to_datetime(local.get('date'), errors='coerce').dt.date
        played_lk = played[key_cols + ['home_score','away_score']].copy()
        local = local.merge(played_lk, on=key_cols, how='left', suffixes=('', '_new'))
        for c in ['home_score','away_score']:
            nc = f'{c}_new'
            if nc in local.columns:
                newly = local[c].isna() & local[nc].notna()
                updated += int(newly.sum())
                local.loc[newly, c] = local.loc[newly, nc]
        drop = [c for c in local.columns if c.endswith('_new')]
        if drop:
            local = local.drop(columns=drop)

    if updated > 0:
        _save_games(local)
    return updated


def _unique_dates_for_weeks(games: pd.DataFrame, season: int, weeks: list[int]) -> list[str]:
    g = games[(games['season']==season) & (games['week'].isin(weeks))].copy()
    g['date'] = pd.to_datetime(g['date'], errors='coerce').dt.date
    dates = sorted({d.isoformat() for d in g['date'].dropna().unique()})
    return dates


def _update_weather_for_weeks(games: pd.DataFrame, season: int, weeks: list[int]) -> int:
    dates = _unique_dates_for_weeks(games, season, weeks)
    if not dates:
        print('No dates found for weather update.')
        return 0
    total = 0
    for ds in dates:
        # Backup existing daily file if present, to preserve any manual edits
        fp = DATA_DIR / f"weather_{ds}.csv"
        if fp.exists():
            try:
                shutil.copy2(fp, DATA_DIR / f"weather_{ds}.backup.csv")
                print(f"Backed up {fp.name}")
            except Exception as e:
                print(f"Backup failed for {fp.name}: {e}")
        try:
            total += update_weather_for_date(ds)
        except Exception as e:
            print(f"Weather update failed for {ds}: {e}")
    return total


def main() -> None:
    load_env()

    # Load base games and infer current/upcoming week
    games = _load_games()
    season, cur_week, up_week = _infer_current_season_week(games)
    print(f"Season {season} — current week {cur_week} — upcoming week {up_week}")

    # 1) Finals for current week
    updated = _update_finals_for_current_week(season, cur_week)
    print(f"Finals updated for {updated} game(s) in current week.")

    # Reload in case games.csv changed
    games = _load_games()

    # 2) Fetch latest odds snapshot
    try:
        fetch_odds_main()
    except Exception as e:
        print(f"Odds fetch failed: {e}")

    # 3) Weather for current + upcoming week only
    weeks = [cur_week] + ([up_week] if up_week is not None else [])
    wx_rows = _update_weather_for_weeks(games, season, weeks)
    print(f"Weather updated: {wx_rows} rows across {len(weeks)} week(s).")

    # 3b) Append current-season play-by-play for archival/analysis
    try:
        from .fetch_pbp import append_current as _append_pbp
        try:
            out_fp = _append_pbp(int(season))
            print(f"PBP append: wrote/updated {out_fp.name}.")
        except Exception as e:
            print(f"PBP append failed: {e}")
    except Exception as e:
        print(f"PBP append skipped (fetch_pbp unavailable): {e}")

    # 4) Confirm inputs exist
    must_have = [
        DATA_DIR / 'games.csv',
        DATA_DIR / 'team_stats.csv',
        DATA_DIR / 'lines.csv',
    ]
    missing = [str(p) for p in must_have if not p.exists()]
    if missing:
        print(f"Warning: missing inputs: {missing}")

    # 4a) Feature variance/attachment guardrail for current and upcoming week
    try:
        check_features_guardrail_for_week(season, cur_week)
        if up_week is not None:
            check_features_guardrail_for_week(season, up_week)
    except Exception as e:
        print(f"Guardrail checks failed: {e}")

    # 4b) Ensure lines.csv has rows for current week games; enrich with latest JSON odds and backfill closing lines
    try:
        lines_fp = _DATA_DIR / 'lines.csv'
        if lines_fp.exists():
            lines_df = pd.read_csv(lines_fp)
        else:
            from .schemas import _field_names, LineRow  # type: ignore
            lines_df = pd.DataFrame(columns=_field_names(LineRow))

        # Normalize team names in existing lines
        if 'home_team' in lines_df.columns:
            lines_df['home_team'] = lines_df['home_team'].astype(str).apply(normalize_team_name)
        if 'away_team' in lines_df.columns:
            lines_df['away_team'] = lines_df['away_team'].astype(str).apply(normalize_team_name)

        g_slice = games[(games['season']==season) & (games['week']==cur_week)].copy()
        g_slice['home_team'] = g_slice['home_team'].astype(str).apply(normalize_team_name)
        g_slice['away_team'] = g_slice['away_team'].astype(str).apply(normalize_team_name)

        # Prepare new rows for any games not already in lines.csv by game_id
        have_ids = set(str(x) for x in (lines_df['game_id'].dropna().astype(str).unique() if 'game_id' in lines_df.columns else []))
        add = []
        for _, r in g_slice.iterrows():
            gid = str(r.get('game_id'))
            if gid and gid not in have_ids:
                add.append({
                    'season': int(r.get('season')) if pd.notna(r.get('season')) else None,
                    'week': int(r.get('week')) if pd.notna(r.get('week')) else None,
                    'game_id': gid,
                    'home_team': r.get('home_team'),
                    'away_team': r.get('away_team'),
                    'spread_home': pd.NA,
                    'total': pd.NA,
                    'moneyline_home': pd.NA,
                    'moneyline_away': pd.NA,
                    'close_spread_home': pd.NA,
                    'close_total': pd.NA,
                    'date': r.get('date'),
                })
        if add:
            lines_df = pd.concat([lines_df, pd.DataFrame(add)], ignore_index=True)

        # Enrich seeded/current rows with latest JSON odds for current market context
        try:
            j = _latest_json()
            if j is not None and not j.empty:
                j['home_team'] = j['home_team'].astype(str).apply(normalize_team_name)
                j['away_team'] = j['away_team'].astype(str).apply(normalize_team_name)
                key = ['home_team','away_team']
                cols = ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price']
                lines_df = lines_df.merge(j[key+cols], on=key, how='left', suffixes=('', '_json'))
                for c in cols:
                    jc = f'{c}_json'
                    if jc in lines_df.columns:
                        lines_df[c] = lines_df[c].where(lines_df[c].notna(), lines_df[jc])
                drop = [c for c in lines_df.columns if c.endswith('_json')]
                if drop:
                    lines_df = lines_df.drop(columns=drop)
        except Exception as e:
            print(f"JSON odds enrich skipped: {e}")

        # Backfill closing fields using snapshots and fallbacks
        lines_df2, report = backfill_close_fields(lines_df)
        lines_df2.to_csv(lines_fp, index=False)
        print(f"Seeded/enriched lines.csv for current week — {report}")
    except Exception as e:
        print(f"Seeding/backfilling lines.csv failed: {e}")

    # 5) Re-run predictions (future games only)
    try:
        predict_main()
    except Exception as e:
        print(f"Prediction run failed: {e}")

    # 6) Also write week-level predictions for the current week (includes finals)
    try:
        # Build features for current week and run models
        games_all = ds_load_games()
        stats = load_team_stats()
        lines = load_lines()
        try:
            wx_all = load_weather_for_games(games_all)
        except Exception:
            wx_all = None
        feat = merge_features(games_all, stats, lines, wx_all)
        sub = feat[(feat.get('season').astype('Int64') == int(season)) & (feat.get('week').astype('Int64') == int(cur_week))].copy()
        if sub is None or sub.empty:
            print('No feature rows for current week; skipping predictions_week.csv')
        else:
            models_path = Path(__file__).resolve().parents[1] / 'models' / 'nfl_models.joblib'
            if not models_path.exists():
                print('Model file missing; cannot write predictions_week.csv')
            else:
                models = joblib_load(models_path)
                pred_week = model_predict(models, sub)
                out_fp = DATA_DIR / 'predictions_week.csv'
                pred_week.to_csv(out_fp, index=False)
                print(f'Wrote {out_fp} with {len(pred_week)} rows for season={season}, week={cur_week}')
    except Exception as e:
        print(f"Week-level predictions failed: {e}")

    # 6b) Compute and save player props for current week
    try:
        # If games for the current week have started, lock props to prevent changes
        try:
            if _week_has_started(int(season), int(cur_week)):
                _lock_props_week(int(season), int(cur_week))
                print(f"Props auto-locked for season={season}, week={cur_week} (games started).")
        except Exception:
            pass

        # Build ESPN depth chart CSV for current week (for injury-aware depth)
        try:
            from .depth_charts import save_depth_chart as _save_depth
            dc_fp = _DATA_DIR / f"depth_chart_{int(season)}_wk{int(cur_week)}.csv"
            # Rebuild weekly to capture latest injuries; overwrite silently
            out_fp = _save_depth_chart(int(season), int(cur_week)) if False else _save_depth(int(season), int(cur_week))
            if dc_fp.exists():
                print(f"Depth chart ready: {dc_fp.name}")
            else:
                print(f"Depth chart built: {out_fp.name if hasattr(out_fp,'name') else out_fp}")
        except Exception as e:
            print(f"Depth chart build skipped/failed: {e}")

        # Build Week 1 central stats once per season (if not present)
        try:
            from .week1_central_stats import build_week1_central_stats as _build_central
            cfp = DATA_DIR / f"week1_central_stats_{int(season)}.csv"
            if not cfp.exists():
                df_c = _build_central(int(season))
                if df_c is not None and not df_c.empty:
                    print(f"Central Week 1 stats built: {len(df_c)} rows.")
                else:
                    print("Central Week 1 stats: no rows (skip).")
            else:
                print("Central Week 1 stats already present; skipping rebuild.")
        except Exception as e:
            print(f"Central Week 1 stats build skipped/failed: {e}")

        # Rebuild usage priors from latest rosters/depth charts to keep depth accurate
        try:
            from .build_player_usage_priors import build_player_usage_priors as _build_priors
            priors_df = _build_priors(int(season))
            if priors_df is not None and not priors_df.empty:
                priors_fp = DATA_DIR / 'player_usage_priors.csv'
                priors_df.to_csv(priors_fp, index=False)
                print(f"Updated {priors_fp.name} with {len(priors_df)} rows.")
            else:
                print('Usage priors: no rows built; keeping existing file if present.')
        except Exception as e:
            print(f"Usage priors build skipped/failed: {e}")

        # Build player efficiency priors from PBP archives (best effort)
        try:
            from .build_player_efficiency_priors import build_player_efficiency_priors as _build_eff
            # Use last 3 seasons + current if available via files
            eff_df = _build_eff()
            if eff_df is not None and not eff_df.empty:
                eff_fp = DATA_DIR / 'player_efficiency_priors.csv'
                eff_fp.parent.mkdir(parents=True, exist_ok=True)
                eff_df.to_csv(eff_fp, index=False)
                print(f"Updated {eff_fp.name} with {len(eff_df)} rows.")
            else:
                print('Efficiency priors: no rows built (missing PBP files?).')
        except Exception as e:
            print(f"Efficiency priors build skipped/failed: {e}")

        try:
            from .player_props import compute_player_props as _compute_player_props
            from .util import silence_stdout_stderr
        except Exception:
            _compute_player_props = None

        if _compute_player_props is not None:
            # Respect per-week props lock
            lock_marker = DATA_DIR / f"props_lock_{int(season)}_wk{int(cur_week)}.lock"
            locked_csv = DATA_DIR / f"player_props_{int(season)}_wk{int(cur_week)}.locked.csv"
            if lock_marker.exists() or locked_csv.exists():
                print(f"Player props locked for season={season}, week={cur_week}; skipping props regen.")
            else:
                with silence_stdout_stderr():
                    props_df = _compute_player_props(int(season), int(cur_week))
                if props_df is not None and not props_df.empty:
                    out_props = DATA_DIR / f"player_props_{season}_wk{cur_week}.csv"
                    props_df.to_csv(out_props, index=False)
                    print(f"Wrote {out_props} with {len(props_df)} rows of player props.")
                else:
                    print("Player props: no rows computed.")
        else:
            print("Player props: module unavailable; skipped.")

        # Build roster validation report (external check vs nfl_data_py rosters/depth charts)
        try:
            from .roster_validation import build_roster_validation as _build_roster_validation
            det, summ = _build_roster_validation(int(season), int(cur_week))
            if summ is not None and not summ.empty:
                print(
                    f"Roster validation wrote summary and details for season={season}, week={cur_week}"
                )
            else:
                print("Roster validation: no summary produced.")
        except Exception as e:
            print(f"Roster validation failed/skipped: {e}")
    except Exception as e:
        print(f"Player props step failed: {e}")

    print('Daily update complete.')

    # 7) Lock predictions for tracking & update finals
    try:
        _lock_predictions(season)
    except Exception as e:
        print(f"Lock predictions step failed: {e}")


def _lock_predictions(current_season: int) -> None:
    """Maintain a cumulative predictions_locked.csv that:
    - Stores the first (pre-game) predictions for every game encountered (one row per game_id)
    - When a game becomes final (scores present in games.csv) fills in final scores & closing lines
    - Never mutates the originally locked prediction values (pred_*, prob_*)
    """
    pred_fp = DATA_DIR / 'predictions.csv'
    week_fp = DATA_DIR / 'predictions_week.csv'
    games_fp = DATA_DIR / 'games.csv'
    lines_fp = DATA_DIR / 'lines.csv'

    if not games_fp.exists():
        print('Lock: games.csv missing; skipping.')
        return
    games = pd.read_csv(games_fp)
    # Ensure date typed
    games['date'] = pd.to_datetime(games.get('date'), errors='coerce')

    # Load existing locked file
    if LOCKED_FP.exists():
        locked = pd.read_csv(LOCKED_FP)
    else:
        locked = pd.DataFrame()

    # Build an index of already locked game_ids
    have_ids = set(str(x) for x in (locked['game_id'].astype(str).unique() if not locked.empty and 'game_id' in locked.columns else []))

    # Add new future-game predictions (those in predictions.csv) that are not yet locked
    if pred_fp.exists():
        try:
            preds = pd.read_csv(pred_fp)
        except Exception:
            preds = pd.DataFrame()
        if not preds.empty:
            new_rows = preds[~preds['game_id'].astype(str).isin(have_ids)].copy()
            if not new_rows.empty:
                new_rows['locked_at'] = datetime.utcnow().isoformat()
                locked = pd.concat([locked, new_rows], ignore_index=True) if not locked.empty else new_rows
                have_ids.update(new_rows['game_id'].astype(str).tolist())
                print(f"Lock: added {len(new_rows)} new game prediction(s).")

    # If week-level predictions exist, attempt to backfill early lock for games that already started before first lock
    if week_fp.exists():
        try:
            week_pred = pd.read_csv(week_fp)
        except Exception:
            week_pred = pd.DataFrame()
        if not week_pred.empty:
            need_lock = week_pred[~week_pred['game_id'].astype(str).isin(have_ids)].copy()
            # Only lock rows that have not yet started (no final scores) — but since week file may include finals we still lock if not already locked
            if not need_lock.empty:
                need_lock['locked_at'] = datetime.utcnow().isoformat()
                locked = pd.concat([locked, need_lock], ignore_index=True) if not locked.empty else need_lock
                have_ids.update(need_lock['game_id'].astype(str).tolist())
                print(f"Lock: backfilled {len(need_lock)} week prediction(s).")

    if locked.empty:
        print('Lock: no predictions to lock.')
        return

    # Update finals & closing lines for games now completed
    try:
        lines = pd.read_csv(lines_fp) if lines_fp.exists() else pd.DataFrame()
    except Exception:
        lines = pd.DataFrame()
    if not lines.empty and 'game_id' in lines.columns:
        # Ensure we don't duplicate game_id dtype issues
        lines['game_id'] = lines['game_id'].astype(str)
        key_cols = ['game_id','close_spread_home','close_total']
        close_map = lines[key_cols].drop_duplicates(subset=['game_id']) if set(key_cols).issubset(lines.columns) else pd.DataFrame()
    else:
        close_map = pd.DataFrame()

    # Identify finals in games
    finals = games[(games['home_score'].notna()) & (games['away_score'].notna())].copy()
    finals['game_id'] = finals['game_id'].astype(str)
    if not finals.empty:
        locked['game_id'] = locked['game_id'].astype(str)
        before_updates = 0
        # Merge finals data
        locked = locked.merge(finals[['game_id','home_score','away_score']], on='game_id', how='left', suffixes=('', '_final'))
        # For each score column, fill only if original missing
        for c in ['home_score','away_score']:
            fc = f'{c}_final'
            if fc in locked.columns:
                needs = locked[c].isna() & locked[fc].notna()
                if needs.any():
                    locked.loc[needs, c] = locked.loc[needs, fc]
                    before_updates += int(needs.sum())
        # Drop helper columns
        drop = [c for c in locked.columns if c.endswith('_final')]
        if drop:
            locked = locked.drop(columns=drop)
        if before_updates:
            print(f"Lock: updated final scores for {before_updates} game(s).")

    # Attach closing lines if available and not already present in locked
    if not close_map.empty:
        locked = locked.merge(close_map, on='game_id', how='left', suffixes=('', '_close2'))
        for c in ['close_spread_home','close_total']:
            alt = f'{c}_close2'
            if alt in locked.columns:
                locked[c] = locked[c].where(locked[c].notna(), locked[alt])
        drop2 = [c for c in locked.columns if c.endswith('_close2')]
        if drop2:
            locked = locked.drop(columns=drop2)

    # Reorder columns lightly: ensure locked_at near front
    cols = list(locked.columns)
    if 'locked_at' in cols:
        # Move locked_at after game_id
        cols.remove('locked_at')
        if 'game_id' in cols:
            gi_idx = cols.index('game_id')
            cols = cols[:gi_idx+1] + ['locked_at'] + cols[gi_idx+1:]
        else:
            cols = ['locked_at'] + cols
        locked = locked.reindex(columns=cols)

    # Persist
    locked.to_csv(LOCKED_FP, index=False)
    print(f"Lock: wrote {len(locked)} rows to {LOCKED_FP.name}.")


if __name__ == '__main__':
    main()
