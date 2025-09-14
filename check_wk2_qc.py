import pandas as pd, numpy as np
from pathlib import Path
fp = Path('c:/Users/mostg/OneDrive/Coding/NFL-Betting/nfl_compare/data/player_props_2025_wk2.csv')
df = pd.read_csv(fp)
# 1) QB per team and pass attempts integrity
qbs = df[df['position'].astype(str).str.upper()=='QB'].copy()
qbs['has_pa'] = pd.to_numeric(qbs['pass_attempts'], errors='coerce').fillna(0) > 0
print('QB rows:', len(qbs), 'teams:', qbs['team'].nunique())
# 2) Ensure one QB row per team
issues_qb_count = qbs.groupby('team').size().reset_index(name='n'); issues = issues_qb_count[issues_qb_count['n']!=1]
print('Teams with !=1 QB rows:', issues.to_dict(orient='records'))
# 3) Any QB with targets or rec yards
qbs_bad = qbs[(pd.to_numeric(qbs.get('targets',0), errors='coerce').fillna(0)>0) | (pd.to_numeric(qbs.get('rec_yards',0), errors='coerce').fillna(0)>0)]
print('QB receiving rows (should be 0):', len(qbs_bad))
# 4) Volume conservation: sum of targets vs team_pass_attempts per team
out = []
for team, g in df.groupby('team'):
    team_pass = float(pd.to_numeric(g['team_pass_attempts'], errors='coerce').fillna(0).iloc[0]) if 'team_pass_attempts' in g.columns else np.nan
    recv_mask = g['position'].astype(str).str.upper().isin(['WR','TE','RB'])
    sum_targets = float(pd.to_numeric(g.loc[recv_mask,'targets'], errors='coerce').fillna(0).sum())
    out.append((team, team_pass, sum_targets, sum_targets-team_pass))
print('Team target sums (team, team_pass_attempts, sum_targets, diff):')
print('\n'.join([str(x) for x in sorted(out)]))
# 5) Rushing conservation: sum rush_attempts approx team_rush_attempts
out2 = []
for team, g in df.groupby('team'):
    team_rush = float(pd.to_numeric(g['team_rush_attempts'], errors='coerce').fillna(0).iloc[0]) if 'team_rush_attempts' in g.columns else np.nan
    sum_rush = float(pd.to_numeric(g['rush_attempts'], errors='coerce').fillna(0).sum())
    out2.append((team, team_rush, sum_rush, sum_rush-team_rush))
print('Team rush sums (team, team_rush_attempts, sum_rush, diff):')
print('\n'.join([str(x) for x in sorted(out2)]))
