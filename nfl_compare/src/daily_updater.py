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
from datetime import date as _date, timedelta
from pathlib import Path
import shutil
from typing import Optional, Tuple

import pandas as pd

from .config import load_env
from .weather import DATA_DIR
from .auto_update import update_weather_for_date
from .odds_api_client import main as fetch_odds_main
from .predict import main as predict_main
from joblib import load as joblib_load
from .data_sources import load_games as ds_load_games, load_team_stats, load_lines
from .features import merge_features
from .weather import load_weather_for_games
from .models import predict as model_predict
from .data_sources import DATA_DIR as _DATA_DIR
from .data_sources import _try_load_latest_real_lines as _latest_json  # type: ignore
from .team_normalizer import normalize_team_name
from nfl_compare.scripts.backfill_close_lines import backfill_close_fields  # type: ignore


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

    # 4) Confirm inputs exist
    must_have = [
        DATA_DIR / 'games.csv',
        DATA_DIR / 'team_stats.csv',
        DATA_DIR / 'lines.csv',
    ]
    missing = [str(p) for p in must_have if not p.exists()]
    if missing:
        print(f"Warning: missing inputs: {missing}")

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

    print('Daily update complete.')


if __name__ == '__main__':
    main()
