from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from .player_props import DATA_DIR, normalize_team_name


def _load_pbp(season: int) -> pd.DataFrame:
    pbp = pd.DataFrame()
    try:
        fp = DATA_DIR / f"pbp_{int(season)}.parquet"
        if fp.exists():
            pbp = pd.read_parquet(fp)
    except Exception:
        pbp = pd.DataFrame()
    if pbp is None or pbp.empty:
        try:
            import nfl_data_py as nfl  # type: ignore
            pbp = nfl.import_pbp_data([int(season)])
        except Exception:
            pbp = pd.DataFrame()
    return pbp if pbp is not None else pd.DataFrame()


def build_week1_central_stats(season: int) -> pd.DataFrame:
    """Build per-player per-game Week 1 stats from PBP: targets, pass att, rush att, yards, TDs.
    Returns the DataFrame and writes CSV to DATA_DIR/week1_central_stats_{season}.csv.
    """
    df = _load_pbp(int(season))
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter to regular season week 1
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    elif 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    if 'week' not in df.columns:
        return pd.DataFrame()
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    df = df[df['week'] == 1].copy()
    if df.empty:
        return pd.DataFrame()

    # Columns
    tcol = 'posteam' if 'posteam' in df.columns else ('pos_team' if 'pos_team' in df.columns else None)
    dcol = 'defteam' if 'defteam' in df.columns else ('def_team' if 'def_team' in df.columns else None)
    if tcol is None:
        return pd.DataFrame()

    # Normalize base per-play
    base = df.copy()
    base['team'] = base[tcol].astype(str).apply(normalize_team_name)
    if dcol:
        base['opponent'] = base[dcol].astype(str).apply(normalize_team_name)
    else:
        base['opponent'] = None
    base['game_id'] = base.get('game_id', pd.NA)
    base['yards_gained'] = pd.to_numeric(base.get('yards_gained'), errors='coerce').fillna(0.0)

    # Receiving: targets, rec_yards, rec_tds
    pass_mask = None
    if 'pass' in base.columns:
        pass_mask = pd.to_numeric(base['pass'], errors='coerce').fillna(0).astype(int).eq(1)
    elif 'pass_attempt' in base.columns:
        pass_mask = pd.to_numeric(base['pass_attempt'], errors='coerce').fillna(0).astype(int).eq(1)
    rec_id_col = 'receiver_player_id' if 'receiver_player_id' in base.columns else ('receiver_id' if 'receiver_id' in base.columns else None)
    rec_name_col = None
    for c in ['receiver_player_name','receiver_name','receiver']:
        if c in base.columns:
            rec_name_col = c; break
    rec_stats = pd.DataFrame(columns=['game_id','team','player_id','player','targets','rec_yards','rec_tds'])
    if pass_mask is not None and (rec_id_col or rec_name_col):
        sub = base[pass_mask].copy()
        # identify targets: pass plays with receiver recorded
        has_any = None
        if rec_id_col:
            has_any = sub[rec_id_col].astype(str).str.len() > 0
        if rec_name_col:
            has_name = sub[rec_name_col].astype(str).str.strip().str.len() > 0
            has_any = has_name if has_any is None else (has_any | has_name)
        sub = sub[has_any.fillna(False)].copy()
        # completed catches for yards; incomplete -> 0
        comp_mask = pd.to_numeric(sub.get('complete_pass'), errors='coerce').fillna(0).astype(int).eq(1) if 'complete_pass' in sub.columns else None
        yards = sub['yards_gained'] if comp_mask is None else (sub['yards_gained'] * comp_mask.astype(int))
        rec_tds = pd.to_numeric(sub.get('pass_touchdown'), errors='coerce').fillna(0).astype(int)
        tmp = pd.DataFrame({
            'game_id': sub['game_id'],
            'team': sub['team'],
            'player_id': sub[rec_id_col] if rec_id_col else sub[rec_name_col],
            'player': sub[rec_name_col] if rec_name_col else sub[rec_id_col],
            'targets': 1,
            'rec_yards': yards,
            'rec_tds': rec_tds,
        })
        rec_stats = tmp.groupby(['game_id','team','player_id','player'], as_index=False).agg(
            targets=('targets','sum'),
            rec_yards=('rec_yards','sum'),
            rec_tds=('rec_tds','sum'),
        )

    # Rushing: attempts, yards, TDs
    rush_id_col = 'rusher_player_id' if 'rusher_player_id' in base.columns else ('rusher_id' if 'rusher_id' in base.columns else None)
    rush_name_col = None
    for c in ['rusher_player_name','rusher_name','rusher']:
        if c in base.columns:
            rush_name_col = c; break
    ru_stats = pd.DataFrame(columns=['game_id','team','player_id','player','rush_att','rush_yards','rush_tds'])
    if rush_id_col or rush_name_col:
        subr = base.copy()
        has_any = None
        if rush_id_col:
            has_any = subr[rush_id_col].astype(str).str.len() > 0
        if rush_name_col:
            has_name = subr[rush_name_col].astype(str).str.strip().str.len() > 0
            has_any = has_name if has_any is None else (has_any | has_name)
        subr = subr[has_any.fillna(False)].copy()
        rtd = pd.to_numeric(subr.get('rush_touchdown'), errors='coerce').fillna(0).astype(int)
        tmp = pd.DataFrame({
            'game_id': subr['game_id'],
            'team': subr['team'],
            'player_id': subr[rush_id_col] if rush_id_col else subr[rush_name_col],
            'player': subr[rush_name_col] if rush_name_col else subr[rush_id_col],
            'rush_att': 1,
            'rush_yards': subr['yards_gained'],
            'rush_tds': rtd,
        })
        ru_stats = tmp.groupby(['game_id','team','player_id','player'], as_index=False).agg(
            rush_att=('rush_att','sum'),
            rush_yards=('rush_yards','sum'),
            rush_tds=('rush_tds','sum'),
        )

    # Passing: attempts, yards, TDs
    pass_id_col = 'passer_player_id' if 'passer_player_id' in base.columns else ('passer_id' if 'passer_id' in base.columns else None)
    pass_name_col = None
    for c in ['passer_player_name','passer_name','passer']:
        if c in base.columns:
            pass_name_col = c; break
    pa_stats = pd.DataFrame(columns=['game_id','team','player_id','player','pass_att','pass_yards','pass_tds'])
    if pass_id_col or pass_name_col:
        subp = base.copy()
        # attempts mask
        if 'pass' in subp.columns:
            mask_pa = pd.to_numeric(subp['pass'], errors='coerce').fillna(0).astype(int).eq(1)
        elif 'pass_attempt' in subp.columns:
            mask_pa = pd.to_numeric(subp['pass_attempt'], errors='coerce').fillna(0).astype(int).eq(1)
        else:
            mask_pa = subp['yards_gained'].notna()  # fallback
        subp = subp[mask_pa].copy()
        has_any = None
        if pass_id_col:
            has_any = subp[pass_id_col].astype(str).str.len() > 0
        if pass_name_col:
            has_name = subp[pass_name_col].astype(str).str.strip().str.len() > 0
            has_any = has_name if has_any is None else (has_any | has_name)
        subp = subp[has_any.fillna(False)].copy()
        ptd = pd.to_numeric(subp.get('pass_touchdown'), errors='coerce').fillna(0).astype(int)
        tmp = pd.DataFrame({
            'game_id': subp['game_id'],
            'team': subp['team'],
            'player_id': subp[pass_id_col] if pass_id_col else subp[pass_name_col],
            'player': subp[pass_name_col] if pass_name_col else subp[pass_id_col],
            'pass_att': 1,
            'pass_yards': subp['yards_gained'],
            'pass_tds': ptd,
        })
        pa_stats = tmp.groupby(['game_id','team','player_id','player'], as_index=False).agg(
            pass_att=('pass_att','sum'),
            pass_yards=('pass_yards','sum'),
            pass_tds=('pass_tds','sum'),
        )

    # Merge roles
    out = rec_stats.merge(ru_stats, on=['game_id','team','player_id','player'], how='outer')
    out = out.merge(pa_stats, on=['game_id','team','player_id','player'], how='outer')
    out['season'] = int(season)
    out['week'] = 1

    # Bring opponent and date per game
    meta_cols = ['game_id','team','opponent']
    if 'game_date' in base.columns:
        base['date'] = pd.to_datetime(base['game_date'], errors='coerce').dt.date
        meta_cols.append('date')
    gm = base[meta_cols].drop_duplicates()
    out = out.merge(gm, on=['game_id','team'], how='left')

    # Clean types and order columns
    for c in ['targets','rec_yards','rec_tds','rush_att','rush_yards','rush_tds','pass_att','pass_yards','pass_tds']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(float)
    pref = ['season','week','date','game_id','team','opponent','player','player_id',
            'targets','rec_yards','rec_tds','rush_att','rush_yards','rush_tds','pass_att','pass_yards','pass_tds']
    cols = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref]
    out = out[cols]

    # Persist
    fp = DATA_DIR / f"week1_central_stats_{int(season)}.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(fp, index=False)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    args = ap.parse_args()
    df = build_week1_central_stats(args.season)
    print(f"Wrote {len(df)} rows to {DATA_DIR / f'week1_central_stats_{int(args.season)}.csv'}")
