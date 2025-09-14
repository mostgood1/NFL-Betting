from pathlib import Path
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: qb_check.py <season> <week>")
    sys.exit(1)

season = int(sys.argv[1])
week = int(sys.argv[2])
fp = Path(__file__).resolve().parents[1] / 'nfl_compare' / 'data' / f'player_props_{season}_wk{week}.csv'
if not fp.exists():
    print(f"Missing file: {fp}")
    sys.exit(2)

df = pd.read_csv(fp)
df = df[df['position'].astype(str).str.upper()=='QB'].copy()
# Show per-team QB list and whether any has pass attempts assigned
agg = (
    df.assign(has_pa=df['pass_attempts'].notna())
      .groupby('team')
      .agg(players=('player', lambda s: list(s)), any_pa=('has_pa','any'))
      .reset_index()
      .sort_values('team')
)
print(agg.to_string(index=False))
