from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

try:
    import nfl_data_py as nfl
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import nfl_data_py: {e}")

from .team_normalizer import normalize_team_name

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _safe_numeric(s) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _depth_sorted(df: pd.DataFrame, pos_prefix: str) -> pd.DataFrame:
    d = df.copy()
    # prefer depth chart order; fallback to None -> bottom
    d['depth_chart_order'] = pd.to_numeric(d.get('depth_chart_order'), errors='coerce')
    d['__order'] = d['depth_chart_order'].fillna(99).astype(int)
    name_col = 'player_name' if 'player_name' in d.columns else ('display_name' if 'display_name' in d.columns else 'gsis_id')
    d = d.sort_values(['__order', name_col]).copy()
    d['depth_label'] = [f"{pos_prefix}{i+1}" for i in range(len(d))]
    return d


def _weights_for_group(group: str, n: int, rz: bool = False) -> List[float]:
    # Heuristic depth-based weights; truncated to n and renormalized
    if group == 'RB':
        base = [0.60, 0.25, 0.10, 0.05] if not rz else [0.70, 0.25, 0.05]
    elif group == 'WR':
        base = [0.30, 0.25, 0.15, 0.10, 0.05] if not rz else [0.28, 0.25, 0.18, 0.10, 0.04]
    elif group == 'TE':
        base = [0.60, 0.25, 0.10, 0.05] if not rz else [0.65, 0.25, 0.10]
    elif group == 'QB':
        base = [0.10, 0.02, 0.01] if not rz else [0.10, 0.02, 0.01]  # rushing shares only
    else:
        base = [1.0]
    arr = base[:max(1, n)]
    arr = arr + [0.0] * (n - len(arr))
    s = sum(arr)
    if s <= 0:
        return [1.0 / n] * n
    return [w / s for w in arr]


def _build_team_usage(roster: pd.DataFrame, team_name: str, season: int) -> pd.DataFrame:
    rows: List[Dict] = []
    # Position grouping
    pos_col = 'depth_chart_position' if 'depth_chart_position' in roster.columns else ('position' if 'position' in roster.columns else None)
    rc = roster[pos_col].astype(str).str.upper().fillna('') if pos_col else pd.Series(['']*len(roster))
    qb = roster[rc == 'QB']
    # treat HB as RB; exclude FB from RB group
    rb = roster[rc.isin(['RB','HB'])]
    wr = roster[rc == 'WR']
    te = roster[rc == 'TE']

    # Depth sort
    qb = _depth_sorted(qb, 'QB') if not qb.empty else qb
    rb = _depth_sorted(rb, 'RB') if not rb.empty else rb
    wr = _depth_sorted(wr, 'WR') if not wr.empty else wr
    te = _depth_sorted(te, 'TE') if not te.empty else te

    # Assign weights
    qb_w = _weights_for_group('QB', len(qb), rz=False)
    rb_w = _weights_for_group('RB', len(rb), rz=False)
    wr_w = _weights_for_group('WR', len(wr), rz=False)
    te_w = _weights_for_group('TE', len(te), rz=False)
    rb_rz = _weights_for_group('RB', len(rb), rz=True)
    wr_rz = _weights_for_group('WR', len(wr), rz=True)
    te_rz = _weights_for_group('TE', len(te), rz=True)

    def _rows(df: pd.DataFrame, pos: str, w: List[float], rz: List[float]) -> List[Dict]:
        out = []
        for i, (_, r) in enumerate(df.iterrows()):
            def pick_name(row: pd.Series) -> str:
                # Best: explicit first + last when both present
                fn = row.get('first_name')
                ln = row.get('last_name')
                fn = str(fn).strip() if pd.notna(fn) else ''
                ln = str(ln).strip() if pd.notna(ln) else ''
                if fn and ln:
                    return f"{fn} {ln}".strip()

                # Next: any consolidated display/full name fields
                for key in ['player_name','player_display_name','display_name','full_name','football_name']:
                    val = row.get(key)
                    if pd.notna(val):
                        s = str(val).strip()
                        if s:
                            # If we only have first or last separately, try to merge
                            if fn and ' ' not in s:
                                # derive last from other name-like fields
                                for k2 in ['full_name','display_name','player_name','football_name']:
                                    v2 = row.get(k2)
                                    if pd.notna(v2):
                                        parts = str(v2).strip().split()
                                        if len(parts) >= 2:
                                            return f"{fn} {' '.join(parts[1:])}".strip()
                            if ln and ' ' not in s:
                                for k2 in ['full_name','display_name','player_name','football_name']:
                                    v2 = row.get(k2)
                                    if pd.notna(v2):
                                        parts = str(v2).strip().split()
                                        if len(parts) >= 2:
                                            return f"{' '.join(parts[:-1])} {ln}".strip()
                            return s

                # If only first OR last available, return what we have
                if fn or ln:
                    return f"{fn} {ln}".strip()

                # Fallback: placeholder with team/slot
                return f'{team_name} {pos}{i+1}'
            out.append({
                'season': season,
                'team': team_name,
                'player': pick_name(r),
                'position': pos,
                'rush_share': w[i] if pos in {'RB','QB'} else 0.0,
                'target_share': w[i] if pos in {'WR','TE','RB'} and pos != 'QB' else (0.0 if pos == 'QB' else w[i]),
                'rz_rush_share': rz[i] if pos in {'RB','QB'} else 0.0,
                'rz_target_share': rz[i] if pos in {'WR','TE','RB'} and pos != 'QB' else 0.0,
            })
        return out

    rows += _rows(qb, 'QB', qb_w, qb_w)
    rows += _rows(rb, 'RB', rb_w, rb_rz)
    rows += _rows(wr, 'WR', wr_w, wr_rz)
    rows += _rows(te, 'TE', te_w, te_rz)

    return pd.DataFrame(rows)


def build_player_usage_priors(season: int) -> pd.DataFrame:
    # seasonal rosters for names/positions; depth charts for ordering and role
    try:
        ros = nfl.import_seasonal_rosters([season])
    except Exception as e:
        raise SystemExit(f"import_seasonal_rosters failed: {e}")
    if ros is None or ros.empty:
        return pd.DataFrame(columns=['season','team','player','position','rush_share','target_share','rz_rush_share','rz_target_share'])

    try:
        dch = nfl.import_depth_charts([season])
    except Exception:
        dch = pd.DataFrame()

    # Normalize team names
    ros = ros.copy()
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if team_src:
        ros['team_norm'] = ros[team_src].astype(str).apply(normalize_team_name)
    else:
        ros['team_norm'] = ''

    # Prefer active players if status columns present, but be permissive (ACT/A01/etc.)
    if 'status' in ros.columns or 'status_description_abbr' in ros.columns:
        mask = pd.Series([True] * len(ros))
        if 'status' in ros.columns:
            s = ros['status'].astype(str).str.strip().str.lower()
            mask = mask & (s.isna() | s.isin(['act','active','unknown','']))
        if 'status_description_abbr' in ros.columns:
            s2 = ros['status_description_abbr'].astype(str).str.strip().str.upper()
            mask = mask & (s2.isna() | s2.isin(['A01','ACTIVE','UNKNOWN','']))
        ros = ros[mask]

    # Merge depth charts if available to bring in depth_chart_position/order
    if not dch.empty:
        dch = dch.copy()
        d_team_src = 'team' if 'team' in dch.columns else ('recent_team' if 'recent_team' in dch.columns else ('team_abbr' if 'team_abbr' in dch.columns else None))
        if d_team_src:
            dch['team_norm'] = dch[d_team_src].astype(str).apply(normalize_team_name)
        else:
            dch['team_norm'] = ''
        # determine player name columns for merge keys
        r_name = 'player_name' if 'player_name' in ros.columns else ('display_name' if 'display_name' in ros.columns else ('full_name' if 'full_name' in ros.columns else None))
        d_name = 'player_name' if 'player_name' in dch.columns else ('display_name' if 'display_name' in dch.columns else None)
        if r_name and d_name:
            # Map depth chart columns to expected names
            cols = ['team_norm', d_name]
            if 'pos_abb' in dch.columns:
                dch = dch.rename(columns={'pos_abb': 'depth_chart_position'})
            if 'pos_rank' in dch.columns:
                dch = dch.rename(columns={'pos_rank': 'depth_chart_order'})
            use_cols = cols + [c for c in ['depth_chart_position','depth_chart_order'] if c in dch.columns]
            ros = ros.merge(
                dch[use_cols],
                left_on=['team_norm', r_name], right_on=['team_norm', d_name], how='left',
                suffixes=('_ros','_dch')
            )
            # Drop duplicate right-side key first, then restore left names
            if d_name in ros.columns:
                ros = ros.drop(columns=[d_name])
            # Restore original name columns if suffixing occurred
            for nm in ['player_name','display_name','full_name']:
                left = f'{nm}_ros'
                if left in ros.columns:
                    ros = ros.rename(columns={left: nm})
                # drop right-side duplicate name column if present
                right = f'{nm}_dch'
                if right in ros.columns:
                    ros = ros.drop(columns=[right])
        else:
            # no reliable player name to join; keep roster as-is
            if 'depth_chart_position' not in ros.columns:
                ros['depth_chart_position'] = pd.NA
            if 'depth_chart_order' not in ros.columns:
                ros['depth_chart_order'] = pd.NA
    else:
        # ensure columns exist if depth charts unavailable
        if 'depth_chart_position' not in ros.columns:
            ros['depth_chart_position'] = pd.NA
        if 'depth_chart_order' not in ros.columns:
            ros['depth_chart_order'] = pd.NA

    all_rows: List[pd.DataFrame] = []
    for team, rdf in ros.groupby('team_norm'):
        if not team or pd.isna(team):
            continue
        # Keep relevant columns
        keep = ['player_name','player_display_name','display_name','full_name','football_name','first_name','last_name','gsis_id','position','depth_chart_position','depth_chart_order']
        for c in keep:
            if c not in rdf.columns:
                rdf[c] = pd.NA
        team_df = _build_team_usage(rdf[keep], team, season)
        if not team_df.empty:
            all_rows.append(team_df)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=['season','team','player','position','rush_share','target_share','rz_rush_share','rz_target_share'])
    # Clamp and renormalize per-team by column
    def _renorm(group: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        g = group.copy()
        for c in cols:
            g[c] = pd.to_numeric(g[c], errors='coerce').fillna(0.0).clip(lower=0.0)
            s = g[c].sum()
            if s > 0:
                g[c] = g[c] / s
        return g
    # Exclude grouping columns during apply to avoid pandas deprecation; fallback for older pandas
    try:
        out = out.groupby(['season','team'], as_index=False).apply(lambda g: _renorm(g, ['rush_share','target_share','rz_rush_share','rz_target_share']), include_groups=False).reset_index(drop=True)
    except TypeError:
        out = out.groupby(['season','team'], as_index=False, group_keys=False).apply(lambda g: _renorm(g, ['rush_share','target_share','rz_rush_share','rz_target_share'])).reset_index(drop=True)
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Build player usage priors (shares) from current rosters and depth charts.')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--out', type=str, default=None, help='Output CSV path; defaults to data/player_usage_priors.csv')
    args = ap.parse_args(argv)

    df = build_player_usage_priors(args.season)
    if df is None or df.empty:
        print('No roster data found; nothing written.')
        return
    out_fp = Path(args.out) if args.out else (DATA_DIR / 'player_usage_priors.csv')
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f'Wrote {len(df)} rows to {out_fp}')
    try:
        print(df.groupby(['team','position']).head(1)[['team','player','position','rush_share','target_share','rz_rush_share','rz_target_share']].to_string(index=False))
    except Exception:
        pass


if __name__ == '__main__':
    main()
