import argparse
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'nfl_compare' / 'data'


def normalize_team_name(name: str) -> str:
    # Lightweight normalization mirroring project behavior
    s = str(name or '').strip()
    if not s:
        return s
    # Known aliases map could be imported, but keep minimal here
    return s


def main():
    p = argparse.ArgumentParser(description='Audit props coverage vs ESPN depth')
    p.add_argument('season', type=int)
    p.add_argument('week', type=int)
    args = p.parse_args()

    depth_fp = DATA_DIR / f"depth_chart_{args.season}_wk{args.week}.csv"
    props_fp = DATA_DIR / f"player_props_{args.season}_wk{args.week}.csv"
    if not depth_fp.exists() or not props_fp.exists():
        print(f"Missing files. Depth: {depth_fp.exists()} Props: {props_fp.exists()}")
        sys.exit(1)

    depth = pd.read_csv(depth_fp)
    props = pd.read_csv(props_fp)

    keep_pos = {'QB','RB','WR','TE'}
    depth['team'] = depth['team'].astype(str).map(normalize_team_name)
    props['team'] = props['team'].astype(str).map(normalize_team_name)

    d = depth[depth['position'].astype(str).str.upper().isin(keep_pos)].copy()
    p = props[props['position'].astype(str).str.upper().isin(keep_pos)].copy()

    # Prepare comparable keys
    d['pos'] = d['position'].astype(str).str.upper()
    p['pos'] = p['position'].astype(str).str.upper()
    d['player_key'] = d['player'].astype(str).str.strip()
    p['player_key'] = p['player'].astype(str).str.strip()

    # Merge to find missing in props
    merged = d.merge(p[['team','pos','player_key']], on=['team','pos','player_key'], how='left', indicator=True)
    missing = merged[merged['_merge'] == 'left_only'].copy()
    missing = missing[['team','pos','player_key','depth_rank','depth_size','status','active']].sort_values(['team','pos','depth_rank'])

    # Per-team/pos counts
    coverage = (
        p.groupby(['team','pos'])['player_key']
         .nunique()
         .rename('props_count')
         .reset_index()
         .merge(
            d.groupby(['team','pos'])['player'].nunique().rename('depth_count').reset_index(),
            on=['team','pos'], how='outer'
         )
         .fillna(0)
         .astype({'props_count':int,'depth_count':int})
         .sort_values(['team','pos'])
    )

    out_missing = DATA_DIR / f"missing_players_{args.season}_wk{args.week}.csv"
    out_cov = DATA_DIR / f"coverage_counts_{args.season}_wk{args.week}.csv"
    missing.to_csv(out_missing, index=False)
    coverage.to_csv(out_cov, index=False)

    print(f"Saved missing players to {out_missing} ({len(missing)} rows)")
    print(f"Saved coverage counts to {out_cov} ({len(coverage)} rows)")

    # Quick screen summary: teams with big gaps
    gaps = coverage[coverage['props_count'] < coverage['depth_count']]
    if not gaps.empty:
        print('\nTeams with gaps (props < depth):')
        print(gaps.to_string(index=False))
    else:
        print('\nNo gaps detected (props >= depth for all team/pos).')


if __name__ == '__main__':
    main()
