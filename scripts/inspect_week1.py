from pathlib import Path
import pandas as pd

fp = Path(r'c:/Users/mostg/OneDrive/Coding/NFL-Betting/nfl_compare/data/player_props_2025_wk1.csv')
df = pd.read_csv(fp)
# Top WR/TE by targets
sub = df[df['position'].isin(['WR','TE'])].copy()
sub['targets'] = pd.to_numeric(sub['targets'], errors='coerce').fillna(0.0)
ranked = sub.sort_values('targets', ascending=False).head(20)
print(ranked[['player','team','position','targets','is_active']].to_string(index=False))
print('Any inactive in top 20?', bool((ranked['is_active'].astype('Int64').fillna(1)==0).any()))
