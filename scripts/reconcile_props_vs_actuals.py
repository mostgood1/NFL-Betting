from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'nfl_compare' / 'data'


try:
    # Prefer using the robust in-package reconciliation utilities,
    # which include a local PBP parquet fallback for actuals.
    from nfl_compare.src.reconciliation import (
        reconcile_props as _recon_from_pkg,
        summarize_errors as _summ_from_pkg,
    )
except Exception:
    _recon_from_pkg = None  # type: ignore
    _summ_from_pkg = None  # type: ignore

try:
    from nfl_compare.src.name_normalizer import (
        normalize_name as _norm_name,
        normalize_name_loose as _norm_name_loose,
        normalize_alias_init_last as _norm_alias_init_last,
    )
except Exception:
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
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception as e:
        raise SystemExit(f"nfl-data-py required: pip install nfl-data-py\n{e}")
    df = nfl.import_weekly_data([int(season)])
    if df is None or df.empty:
        return pd.DataFrame()
    # REG only
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    elif 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    if 'week' not in df.columns:
        return pd.DataFrame()
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    df = df[df['week'] == int(week)].copy()
    if df.empty:
        return pd.DataFrame()
    # Pick id, name, team columns flexibly
    id_col = None
    for c in ['player_id','gsis_id','nfl_id','pfr_id']:
        if c in df.columns:
            id_col = c; break
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in df.columns:
            name_col = c; break
    team_col = None
    for c in ['team','recent_team','team_abbr']:
        if c in df.columns:
            team_col = c; break

    keep = [c for c in [id_col, name_col, team_col] if c]
    # Metric columns
    metrics = {
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
        src = None
        for c in cand:
            if c in df.columns:
                src = c; break
        if src:
            keep.append(src)
        else:
            df[out_col] = 0.0

    sub = df[keep].copy()
    # Rename chosen metric columns to canonical names
    ren = {}
    for out_col, cand in metrics.items():
        src = None
        for c in cand:
            if c in sub.columns:
                src = c; break
        if src:
            ren[src] = out_col
    if ren:
        sub = sub.rename(columns=ren)
    # Coerce numeric
    for k in metrics.keys():
        if k in sub.columns:
            sub[k] = pd.to_numeric(sub[k], errors='coerce').fillna(0.0)
    # Aggregate per player id or name
    if id_col:
        grp_keys = [id_col]
    else:
        grp_keys = [name_col]
    g = (
        sub.groupby(grp_keys, as_index=False)
           .agg({k: ('sum') for k in metrics.keys() if k in sub.columns})
    )
    if id_col and name_col:
        # Keep a sample name for reporting
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
    # Normalize names for fallback join
    if name_col in g.columns:
        g['_name_norm'] = g[name_col].astype(str).map(_norm_name)
        g['_name_norm_loose'] = g[name_col].astype(str).map(_norm_name_loose)
        g['_alias_init_last'] = g[name_col].astype(str).map(_norm_alias_init_last)
    return g.rename(columns={id_col or 'player_id': 'player_id', name_col or 'player': 'player', team_col or 'team': 'team'})


def attach_player_ids(props: pd.DataFrame, season: int) -> pd.DataFrame:
    # Try to map props['player'] to player_id via seasonal rosters
    try:
        import nfl_data_py as nfl  # type: ignore
        ros = nfl.import_seasonal_rosters([int(season)])
    except Exception:
        ros = pd.DataFrame()
    if ros is None or ros.empty:
        props['player_id'] = pd.NA
        props['_name_norm'] = props['player'].astype(str).map(_norm_name)
        return props
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in ros.columns:
            name_col = c; break
    id_col = None
    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
        if c in ros.columns:
            id_col = c; break
    if not name_col and not id_col:
        props['player_id'] = pd.NA
        props['_name_norm'] = props['player'].astype(str).map(_norm_name)
        return props
    m = ros[[c for c in [name_col, id_col] if c]].copy()
    m['_name_norm'] = m[name_col].astype(str).map(_norm_name)
    m['_name_norm_loose'] = m[name_col].astype(str).map(_norm_name_loose)
    out = props.copy()
    out['_name_norm'] = out['player'].astype(str).map(_norm_name)
    out['_name_norm_loose'] = out['player'].astype(str).map(_norm_name_loose)
    # Prefer exact name_norm, fallback to name_norm_loose
    out = out.merge(m[['_name_norm', id_col]].rename(columns={id_col: 'player_id'}), on='_name_norm', how='left')
    if 'player_id' not in out.columns or out['player_id'].isna().any():
        out = out.merge(m[['_name_norm_loose', id_col]].rename(columns={id_col: 'player_id_loose'}), left_on='_name_norm_loose', right_on='_name_norm_loose', how='left')
        if 'player_id' not in out.columns:
            out['player_id'] = out.get('player_id_loose')
        else:
            mask = out['player_id'].isna() & out['player_id_loose'].notna()
            out.loc[mask, 'player_id'] = out.loc[mask, 'player_id_loose']
        if 'player_id_loose' in out.columns:
            out = out.drop(columns=['player_id_loose'])
    return out


def reconcile_props(season: int, week: int) -> pd.DataFrame:
    props_fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
    if not props_fp.exists():
        raise FileNotFoundError(f"Missing projections file: {props_fp}")
    props = pd.read_csv(props_fp)
    props = attach_player_ids(props, season)
    actuals = load_weekly_actuals(season, week)
    if actuals is None or actuals.empty:
        raise RuntimeError("No weekly actuals loaded; cannot reconcile.")

    # Join preference: by player_id when available, else by normalized name
    left = props.copy()
    right = actuals.copy()
    join_keys = []
    if 'player_id' in left.columns and 'player_id' in right.columns:
        join_keys = ['player_id']
    else:
        # build normalized name on both sides
        if '_name_norm' not in left.columns:
            left['_name_norm'] = left['player'].astype(str).map(_norm_name)
        if '_name_norm' not in right.columns:
            right['_name_norm'] = right['player'].astype(str).map(_norm_name)
        join_keys = ['_name_norm']

    merged = left.merge(
        right,
        on=join_keys,
        how='left',
        suffixes=('', '_act')
    )
    # Compute errors for a subset of columns
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
    for pos in ['QB','RB','WR','TE']:
        sub = df[df['position'].astype(str).str.upper() == pos].copy()
        if sub.empty:
            continue
        rec = {'position': pos, 'n': len(sub)}
        for c in comp_cols:
            if c in sub.columns and f"{c}_act" in sub.columns:
                pred = pd.to_numeric(sub[c], errors='coerce')
                act = pd.to_numeric(sub[f"{c}_act"], errors='coerce')
                err = (pred - act).abs()
                rec[f"{c}_MAE"] = float(np.nanmean(err)) if len(err) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Reconcile player props projections with weekly actuals')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    args = ap.parse_args(argv)
    # Prefer package implementation if available (handles local PBP fallback; avoids external 404s)
    if _recon_from_pkg is not None and _summ_from_pkg is not None:
        merged = _recon_from_pkg(args.season, args.week)
        summ = _summ_from_pkg(merged)
    else:
        # Fallback: use local implementations
        merged = reconcile_props(args.season, args.week)
        summ = summarize_errors(merged)
    out_fp = DATA_DIR / f"player_props_vs_actuals_{args.season}_wk{args.week}.csv"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_fp, index=False)
    print(f"Wrote reconciliation CSV -> {out_fp} ({len(merged)} rows)")
    if not summ.empty:
        print("\nMAE by position (selected stats):")
        print(summ.to_string(index=False))
    else:
        print("No summary computed.")


if __name__ == '__main__':
    main()
