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

    # 5) Re-run predictions (future games only)
    try:
        predict_main()
    except Exception as e:
        print(f"Prediction run failed: {e}")

    print('Daily update complete.')


if __name__ == '__main__':
    main()
