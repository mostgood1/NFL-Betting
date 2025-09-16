from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'nfl_compare' / 'data'

# Normalizers
try:
    from .team_normalizer import normalize_team_name  # type: ignore
except Exception:  # Fallback for direct script runs
    def normalize_team_name(name: str) -> str:  # type: ignore
        return str(name or '').strip()
try:
    from .name_normalizer import (
        normalize_name as _norm_name,
        normalize_name_loose as _norm_name_loose,
        normalize_alias_init_last as _norm_alias_init_last,
    )
except Exception:
    # Minimal inline fallback
    def _norm_name(s: str) -> str:
        return str(s or '').strip().lower()
    def _norm_name_loose(s: str) -> str:
        t = str(s or '').lower()
        return ''.join(ch for ch in t if ch.isalnum())
    def _norm_alias_init_last(s: str) -> str:
        t = str(s or '').strip()
        if not t:
            return ''
        for ch in ['\t', '\n', '\r', '.', ',', '-', '_']:
            t = t.replace(ch, ' ')
        parts = [p for p in t.split(' ') if p]
        if not parts:
            return ''
        first = parts[0]
        last = parts[-1]
        key = (first[:1] + last).lower()
        return ''.join(ch for ch in key if ch.isalnum())


def load_weekly_actuals(season: int, week: int) -> pd.DataFrame:
    """Load weekly actual player stats; prefer nfl-data-py weekly, fallback to local Parquet PBP.

    Returns a DataFrame with canonical columns for comparison:
      - player_id (when available), player (display name), team
      - pass_attempts, pass_yards, pass_tds, interceptions
      - rush_attempts, rush_yards, rush_tds
      - targets, receptions, rec_yards, rec_tds
    Aggregated per player.
    """
    # Try nfl-data-py weekly first
    try:
        import nfl_data_py as nfl  # type: ignore
        df = nfl.import_weekly_data([int(season)])
        if df is not None and not df.empty and 'week' in df.columns:
            wk = df.copy()
            if 'season_type' in wk.columns:
                wk = wk[wk['season_type'].astype(str).str.upper() == 'REG']
            elif 'game_type' in wk.columns:
                wk = wk[wk['game_type'].astype(str).str.upper() == 'REG']
            wk['week'] = pd.to_numeric(wk['week'], errors='coerce').fillna(0).astype(int)
            wk = wk[wk['week'] == int(week)].copy()
            if not wk.empty:
                out = _normalize_weekly_frame(wk)
                # Add normalized team and alias keys
                if 'team' in out.columns:
                    out['team'] = out['team'].astype(str)
                    out['team_norm'] = out['team'].map(normalize_team_name)
                else:
                    out['team'] = ''
                    out['team_norm'] = ''
                if 'player' in out.columns:
                    out['_alias_init_last'] = out['player'].astype(str).map(_norm_alias_init_last)
                return out
    except Exception:
        pass

    # Fallback: derive weekly actuals from local PBP parquet
    pbp_fp = DATA_DIR / f"pbp_{int(season)}.parquet"
    if not pbp_fp.exists():
        return pd.DataFrame()
    try:
        pbp = pd.read_parquet(pbp_fp)
    except Exception:
        return pd.DataFrame()
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    # Filter by week if available
    if 'week' in pbp.columns:
        pbp['week'] = pd.to_numeric(pbp['week'], errors='coerce').fillna(0).astype(int)
        pbp = pbp[pbp['week'] == int(week)].copy()
        if pbp.empty:
            return pd.DataFrame()

    # Helper: first non-empty value across candidates
    def pick(row, keys):
        for k in keys:
            v = row.get(k)
            if v is not None and pd.notna(v) and str(v).strip():
                return str(v)
        return ''

    out_rows: List[Dict] = []
    # Passing
    if 'pass' in pbp.columns:
        p = pbp[pbp['pass'] == 1].copy()
        if not p.empty:
            p['pass_attempts'] = 1
            py = next((c for c in ['passing_yards','yards_gained'] if c in p.columns), None)
            p['pass_yards'] = pd.to_numeric(p[py], errors='coerce').fillna(0.0) if py else 0.0
            # TDs & INTs
            td_flag = None
            for c in ['pass_touchdown','touchdown']:
                if c in p.columns:
                    td_flag = c; break
            p['pass_tds'] = pd.to_numeric(p[td_flag], errors='coerce').fillna(0).astype(int) if td_flag else 0
            int_flag = None
            for c in ['interception','interception_player_name','interception_player_id']:
                if c in p.columns:
                    int_flag = c; break
            if int_flag and p[int_flag].dtype.kind in {'i','b','f'}:
                p['interceptions'] = (pd.to_numeric(p[int_flag], errors='coerce').fillna(0) > 0).astype(int)
            else:
                p['interceptions'] = (p[int_flag].astype(str).str.len() > 0).astype(int) if int_flag else 0
            p['passer_name_'] = p.apply(lambda r: pick(r, ['passer_player_name','passer','passer_name']), axis=1)
            p['team_'] = p.get('posteam', p.get('pos_team', p.get('offense_team', '')))
            g = (
                p.groupby(['passer_name_', 'team_'], as_index=False)
                 .agg(pass_attempts=('pass_attempts','sum'), pass_yards=('pass_yards','sum'), pass_tds=('pass_tds','sum'), interceptions=('interceptions','sum'))
            )
            for _, r in g.iterrows():
                out_rows.append({'player': r['passer_name_'], 'team': r['team_'], 'pass_attempts': r['pass_attempts'], 'pass_yards': r['pass_yards'], 'pass_tds': r['pass_tds'], 'interceptions': r['interceptions']})

    # Rushing
    if 'rush' in pbp.columns:
        r = pbp[pbp['rush'] == 1].copy()
        if not r.empty:
            r['rush_attempts'] = 1
            ry = next((c for c in ['rushing_yards','yards_gained'] if c in r.columns), None)
            r['rush_yards'] = pd.to_numeric(r[ry], errors='coerce').fillna(0.0) if ry else 0.0
            td_flag = None
            for c in ['rush_touchdown','touchdown']:
                if c in r.columns:
                    td_flag = c; break
            r['rush_tds'] = pd.to_numeric(r[td_flag], errors='coerce').fillna(0).astype(int) if td_flag else 0
            r['rusher_name_'] = r.apply(lambda x: pick(x, ['rusher_player_name','rusher','rusher_name']), axis=1)
            r['team_'] = r.get('posteam', r.get('pos_team', r.get('offense_team', '')))
            g = (
                r.groupby(['rusher_name_', 'team_'], as_index=False)
                 .agg(rush_attempts=('rush_attempts','sum'), rush_yards=('rush_yards','sum'), rush_tds=('rush_tds','sum'))
            )
            for _, x in g.iterrows():
                out_rows.append({'player': x['rusher_name_'], 'team': x['team_'], 'rush_attempts': x['rush_attempts'], 'rush_yards': x['rush_yards'], 'rush_tds': x['rush_tds']})

    # Receiving (targets)
    if 'pass' in pbp.columns:
        rec = pbp[pbp['pass'] == 1].copy()
        if not rec.empty:
            rec['receiver_name_'] = rec.apply(lambda r: pick(r, ['receiver_player_name','receiver','receiver_name']), axis=1)
            rec = rec[rec['receiver_name_'].astype(str).str.len() > 0]
            if not rec.empty:
                rec['targets'] = 1
                # completed? various flags
                comp_flag = next((c for c in ['complete_pass','is_complete'] if c in rec.columns), None)
                rec['receptions'] = pd.to_numeric(rec[comp_flag], errors='coerce').fillna(0).astype(int) if comp_flag else 0
                ry = next((c for c in ['receiving_yards','yards_gained','air_yards'] if c in rec.columns), None)
                rec['rec_yards'] = pd.to_numeric(rec[ry], errors='coerce').fillna(0.0) if ry else 0.0
                td_flag = None
                for c in ['receive_touchdown','pass_touchdown','touchdown']:
                    if c in rec.columns:
                        td_flag = c; break
                rec['rec_tds'] = pd.to_numeric(rec[td_flag], errors='coerce').fillna(0).astype(int) if td_flag else 0
                rec['team_'] = rec.get('posteam', rec.get('pos_team', rec.get('offense_team', '')))
                g = (
                    rec.groupby(['receiver_name_', 'team_'], as_index=False)
                       .agg(targets=('targets','sum'), receptions=('receptions','sum'), rec_yards=('rec_yards','sum'), rec_tds=('rec_tds','sum'))
                )
                for _, x in g.iterrows():
                    out_rows.append({'player': x['receiver_name_'], 'team': x['team_'], 'targets': x['targets'], 'receptions': x['receptions'], 'rec_yards': x['rec_yards'], 'rec_tds': x['rec_tds']})

    if not out_rows:
        return pd.DataFrame()
    # Merge rows by player name
    out = pd.DataFrame(out_rows)
    # Keep team in the key; a player appears for a single team in a given week
    if 'team' not in out.columns:
        out['team'] = ''
    out = out.groupby(['player','team'], as_index=False).sum(numeric_only=True)
    out['_name_norm'] = out['player'].astype(str).map(_norm_name)
    out['_name_norm_loose'] = out['player'].astype(str).map(_norm_name_loose)
    out['_alias_init_last'] = out['player'].astype(str).map(_norm_alias_init_last)
    # Provide foot_* merge keys even if props lacks separate football_name
    out['_foot_norm_loose'] = out['_name_norm_loose']
    out['_foot_alias_init_last'] = out['_alias_init_last']
    # team normalization
    out['team'] = out['team'].astype(str)
    out['team_norm'] = out['team'].map(normalize_team_name)
    return out


def _normalize_weekly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize nfl-data-py weekly frame columns to canonical output."""
    name_col = next((c for c in ['player_display_name','player_name','display_name','full_name','football_name'] if c in df.columns), None)
    id_col = next((c for c in ['player_id','gsis_id','nfl_id','pfr_id'] if c in df.columns), None)
    team_col = next((c for c in ['team','recent_team','team_abbr'] if c in df.columns), None)
    keep: List[str] = [c for c in [name_col, id_col, team_col] if c]
    metrics: Dict[str, List[str]] = {
        'pass_attempts': ['attempts','pass_attempts'],
        'pass_yards': ['passing_yards'],
        'pass_tds': ['passing_tds'],
        'interceptions': ['interceptions'],
        'rush_attempts': ['carries','rushing_attempts','rush_att','rushing_att'],
        'rush_yards': ['rushing_yards'],
        'rush_tds': ['rushing_tds'],
        'targets': ['targets'],
        'receptions': ['receptions'],
        'rec_yards': ['receiving_yards'],
        'rec_tds': ['receiving_tds'],
    }
    for out_col, cand in metrics.items():
        src = next((c for c in cand if c in df.columns), None)
        if src:
            keep.append(src)
        else:
            df[out_col] = 0.0
    sub = df[keep].copy()
    ren = {}
    for out_col, cand in metrics.items():
        src = next((c for c in cand if c in sub.columns), None)
        if src:
            ren[src] = out_col
    if ren:
        sub = sub.rename(columns=ren)
    for k in metrics.keys():
        if k in sub.columns:
            sub[k] = pd.to_numeric(sub[k], errors='coerce').fillna(0.0)
    grp_keys = [id_col] if id_col else [name_col]
    g = (
        sub.groupby(grp_keys, as_index=False)
           .agg({k: ('sum') for k in metrics.keys() if k in sub.columns})
    )
    if id_col and name_col:
        nm = (
            sub.groupby(id_col, as_index=False)[name_col]
               .agg(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else '')
        )
        g = g.merge(nm, on=id_col, how='left')
    if team_col:
        tm = (
            sub.groupby(grp_keys, as_index=False)[team_col]
               .agg(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else '')
        )
        g = g.merge(tm, on=grp_keys, how='left')
    if name_col in g.columns:
        g['_name_norm'] = g[name_col].astype(str).map(_norm_name)
        g['_name_norm_loose'] = g[name_col].astype(str).map(_norm_name_loose)
        g['_alias_init_last'] = g[name_col].astype(str).map(_norm_alias_init_last)
    out = g.rename(columns={id_col or 'player_id': 'player_id', name_col or 'player': 'player', team_col or 'team': 'team'})
    if 'team' in out.columns:
        out['team'] = out['team'].astype(str)
        out['team_norm'] = out['team'].map(normalize_team_name)
    else:
        out['team_norm'] = ''
    # Ensure props has foot_* alias keys used in merges
    if '_foot_norm_loose' not in out.columns:
        out['_foot_norm_loose'] = out['_name_norm_loose']
    if '_foot_alias_init_last' not in out.columns:
        out['_foot_alias_init_last'] = out['_alias_init_last']
    return out


def attach_player_ids(props: pd.DataFrame, season: int) -> pd.DataFrame:
    """Attach player_id to props using seasonal rosters for the season."""
    try:
        import nfl_data_py as nfl  # type: ignore
        ros = nfl.import_seasonal_rosters([int(season)])
    except Exception:
        ros = pd.DataFrame()
    if ros is None or ros.empty:
        out = props.copy()
        out['player_id'] = pd.NA
        out['_name_norm'] = out['player'].astype(str).map(_norm_name)
        out['_name_norm_loose'] = out['player'].astype(str).map(_norm_name_loose)
        return out
    name_col = next((c for c in ['player_display_name','player_name','display_name','full_name','football_name'] if c in ros.columns), None)
    id_col = next((c for c in ['gsis_id','player_id','nfl_id','pfr_id'] if c in ros.columns), None)
    first_col = 'first_name' if 'first_name' in ros.columns else None
    last_col = 'last_name' if 'last_name' in ros.columns else None
    if not name_col and not id_col:
        out = props.copy()
        out['player_id'] = pd.NA
        out['_name_norm'] = out['player'].astype(str).map(_norm_name)
        out['_name_norm_loose'] = out['player'].astype(str).map(_norm_name_loose)
        return out
    # Include optional football_name if present for alternate matching (disabled for ID attach to avoid dup complications)
    foot_col = 'football_name' if 'football_name' in ros.columns else None
    team_col = next((c for c in ['recent_team','team','team_abbr'] if c in ros.columns), None)
    m = ros[[c for c in [name_col, id_col, first_col, last_col, team_col] if c]].copy()
    m['_name_norm'] = m[name_col].astype(str).map(_norm_name)
    m['_name_norm_loose'] = m[name_col].astype(str).map(_norm_name_loose)
    # Also include football_name / abbreviated form if present
    # (foot_* keys skipped for ID attach)
    if first_col and last_col:
        m['_alias_init_last'] = (m[first_col].astype(str).str[:1] + m[last_col].astype(str)).map(_norm_name_loose)
    else:
        m['_alias_init_last'] = m[name_col].astype(str).map(_norm_alias_init_last)
    # Normalize roster team for team-aware merges
    if team_col:
        m['team_norm'] = m[team_col].astype(str).map(normalize_team_name)
    else:
        m['team_norm'] = ''
    out = props.copy()
    out['_orig_idx'] = np.arange(len(out))
    out['_name_norm'] = out['player'].astype(str).map(_norm_name)
    out['_name_norm_loose'] = out['player'].astype(str).map(_norm_name_loose)
    out['_alias_init_last'] = out['player'].astype(str).map(_norm_alias_init_last)
    # Normalize team in props for later team-aware merges
    if 'team' in out.columns:
        out['team'] = out['team'].astype(str)
        out['team_norm'] = out['team'].map(normalize_team_name)
    else:
        out['team_norm'] = ''
    # Helper: safe left-merge with right deduplicated on keys to avoid expanding rows
    def _safe_left_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, keys: list[str], value_col: str, out_col: str) -> pd.DataFrame:
        # If value_col missing, return left unchanged
        if value_col not in df_right.columns:
            if out_col not in df_left.columns:
                df_left[out_col] = pd.NA
            return df_left
        rsub = df_right.dropna(subset=keys + [value_col]).drop_duplicates(subset=keys, keep='first')
        merged = df_left.merge(rsub[keys + [value_col]], on=keys, how='left')
        # Fill only where missing
        if out_col not in merged.columns:
            # Create out_col from right value_col
            merged[out_col] = merged.get(value_col, pd.NA)
        else:
            mask = merged[out_col].isna() & merged[value_col].notna()
            merged.loc[mask, out_col] = merged.loc[mask, value_col]
        # drop helper value_col
        if value_col in merged.columns:
            merged = merged.drop(columns=[value_col])
        return merged

    # Start with blank player_id
    out['player_id'] = out.get('player_id', pd.NA)
    # Prefer team-aware matches first
    out = _safe_left_merge(out, m, ['_alias_init_last','team_norm'], id_col, 'player_id')
    out = _safe_left_merge(out, m, ['_name_norm','team_norm'], id_col, 'player_id')
    out = _safe_left_merge(out, m, ['_name_norm_loose','team_norm'], id_col, 'player_id')
    # Fallbacks without team
    out = _safe_left_merge(out, m, ['_alias_init_last'], id_col, 'player_id')
    out = _safe_left_merge(out, m, ['_name_norm'], id_col, 'player_id')
    out = _safe_left_merge(out, m, ['_name_norm_loose'], id_col, 'player_id')

    # Ensure we didn't expand rows; collapse to original order/count if any duplicates slipped through
    if out.duplicated('_orig_idx').any():
        out = out.sort_values('_orig_idx').drop_duplicates(subset=['_orig_idx'], keep='first')
    out = out.sort_values('_orig_idx').drop(columns=['_orig_idx'])
    return out


def reconcile_props(season: int, week: int) -> pd.DataFrame:
        """Return merged DataFrame joining props with weekly actuals, including _act and _err columns.

        Join priority:
            1) player_id if present on both sides
            2) team_norm + _name_norm_loose
            3) team_norm + _alias_init_last
            4) _name_norm_loose
            5) _alias_init_last
            6) _name_norm
        """
        props_fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
        if not props_fp.exists():
            raise FileNotFoundError(f"Missing projections file: {props_fp}")
        props = pd.read_csv(props_fp)
        props = attach_player_ids(props, season)
        actuals = load_weekly_actuals(season, week)
        if actuals is None or actuals.empty:
            raise RuntimeError("No weekly actuals loaded; cannot reconcile.")

        left = props.copy()
        right = actuals.copy()
        # Ensure keys and team normalization exist
        if '_name_norm' not in left.columns:
            left['_name_norm'] = left['player'].astype(str).map(_norm_name)
        if '_name_norm_loose' not in left.columns:
            left['_name_norm_loose'] = left['player'].astype(str).map(_norm_name_loose)
        if '_alias_init_last' not in left.columns:
            left['_alias_init_last'] = left['player'].astype(str).map(_norm_alias_init_last)
        if 'team_norm' not in left.columns:
            left['team_norm'] = left.get('team', '').astype(str).map(normalize_team_name)

        if '_name_norm' not in right.columns:
            right['_name_norm'] = right['player'].astype(str).map(_norm_name)
        if '_name_norm_loose' not in right.columns:
            right['_name_norm_loose'] = right['player'].astype(str).map(_norm_name_loose)
        if '_alias_init_last' not in right.columns:
            right['_alias_init_last'] = right['player'].astype(str).map(_norm_alias_init_last)
        if 'team_norm' not in right.columns:
            right['team_norm'] = right.get('team', '').astype(str).map(normalize_team_name)

        base = left.copy()
        merged = base.copy()

        def _do_merge(cur: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
            # Deduplicate right on the join keys to enforce 1:1 merge alignment
            if all(k in right.columns for k in keys):
                rsub = right.drop_duplicates(subset=keys, keep='first').copy()
            else:
                rsub = right.copy()
            m = cur.merge(rsub, on=keys, how='left', suffixes=('', '_act'))
            # Preserve original row order and index for positional-safe assignments later
            m = m.set_index(cur.index)
            # annotate strategy for debugging
            m['_match_strategy'] = ' & '.join(keys)
            return m

        def _fill(dst: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
            comp_cols = [
                'pass_attempts','pass_yards','pass_tds','interceptions',
                'rush_attempts','rush_yards','rush_tds',
                'targets','receptions','rec_yards','rec_tds'
            ]
            # compute where any _act will be newly provided
            any_new_mask = pd.Series(False, index=dst.index)
            for c in comp_cols:
                act_col = f"{c}_act"
                if act_col in src.columns:
                    if act_col not in dst.columns:
                        dst[act_col] = src[act_col]
                        # mark rows where src has non-null values
                        any_new_mask |= src[act_col].notna()
                    else:
                        cand = dst[act_col].isna() & src[act_col].notna()
                        if cand.any():
                            # Fill only the newly available entries from src
                            dst_vals = dst[act_col].to_numpy(copy=False)
                            src_vals = src[act_col].to_numpy()
                            mask_np = cand.to_numpy()
                            dst_vals[mask_np] = src_vals[mask_np]
                            dst[act_col] = dst_vals
                            any_new_mask |= cand
            # Copy meta columns for rows that just received any _act values
            for meta in ['player_act','team_act']:
                if meta in src.columns and meta not in dst.columns:
                    dst[meta] = None
            if any_new_mask.any():
                mask_np = any_new_mask.to_numpy()
                for meta in ['player_act','team_act']:
                    if meta in src.columns:
                        src_vals = src[meta].to_numpy()
                        # assign only to masked rows using positional indexing
                        dst_vals = dst[meta].to_numpy(dtype=object, copy=False)
                        dst_vals[mask_np] = src_vals[mask_np]
                        dst[meta] = dst_vals
                # record match strategy
                if '_match_strategy' in src.columns:
                    if 'match_strategy' not in dst.columns:
                        dst['match_strategy'] = None
                    src_ms = src['_match_strategy'].astype(object).to_numpy()
                    dst_ms = dst['match_strategy'].astype(object)
                    dst_ms_vals = dst_ms.to_numpy(copy=False)
                    dst_ms_vals[mask_np] = src_ms[mask_np]
                    dst['match_strategy'] = dst_ms_vals
            return dst

        priorities: List[List[str]] = []
        if 'player_id' in left.columns and 'player_id' in right.columns:
            priorities.append(['player_id'])
        priorities.extend([
            ['team_norm','_name_norm_loose'],
            ['team_norm','_alias_init_last'],
            ['_name_norm_loose'],
            ['_alias_init_last'],
            ['_name_norm']
        ])

        for keys in priorities:
            m = _do_merge(base, keys)
            merged = _fill(merged, m)
        comp_cols = [
            'pass_attempts','pass_yards','pass_tds','interceptions',
            'rush_attempts','rush_yards','rush_tds',
            'targets','receptions','rec_yards','rec_tds'
        ]
        for c in comp_cols:
            if c in merged.columns and f"{c}_act" in merged.columns:
                merged[f"{c}_err"] = pd.to_numeric(merged[c], errors='coerce') - pd.to_numeric(merged[f"{c}_act"], errors='coerce')
        return merged


def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    comp_cols = [
        'pass_attempts','pass_yards','pass_tds','interceptions',
        'rush_attempts','rush_yards','rush_tds',
        'targets','receptions','rec_yards','rec_tds'
    ]
    rows = []
    # Helper to safely compute nanmean without warnings on empty/all-NaN arrays
    def _nanmean_safe(arr) -> float:
        try:
            a = pd.to_numeric(arr, errors='coerce').to_numpy(dtype=float, copy=False)
        except Exception:
            # Fallback for non-numeric
            return float('nan')
        valid = ~(np.isnan(a))
        if a.size == 0 or not valid.any():
            return float('nan')
        return float(np.nanmean(a))
    for pos in ['QB','RB','WR','TE']:
        sub = df[df['position'].astype(str).str.upper() == pos].copy()
        if sub.empty:
            continue
        rec = {'position': pos, 'n': int(len(sub))}
        for c in comp_cols:
            if c in sub.columns and f"{c}_act" in sub.columns:
                pred = pd.to_numeric(sub[c], errors='coerce')
                act = pd.to_numeric(sub[f"{c}_act"], errors='coerce')
                err = (pred - act).abs()
                rec[f"{c}_MAE"] = _nanmean_safe(err)
                # Also include signed bias (mean error)
                rec[f"{c}_bias"] = _nanmean_safe(pd.to_numeric(sub[c], errors='coerce') - pd.to_numeric(sub[f"{c}_act"], errors='coerce'))
                rec[f"{c}_act_mean"] = _nanmean_safe(act)
                rec[f"{c}_pred_mean"] = _nanmean_safe(pred)
        rows.append(rec)
    return pd.DataFrame(rows)

