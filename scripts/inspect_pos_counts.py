from pathlib import Path
import pandas as pd

fp = Path(r'c:/Users/mostg/OneDrive/Coding/NFL-Betting/nfl_compare/data/player_props_2025_wk1.csv')
df = pd.read_csv(fp)
print('rows:', len(df))
print('positions:', df['position'].value_counts(dropna=False).to_dict())
wr = df[df['position'].astype(str).str.upper().eq('WR')].copy()
wr = wr.sort_values('targets', ascending=False)[['player','team','targets','is_active']].head(20)
print('\nTop WRs by targets:')
print(wr.to_string(index=False))
