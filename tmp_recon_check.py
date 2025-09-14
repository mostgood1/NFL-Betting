import os, pandas as pd
from nfl_compare.src.reconciliation import reconcile_props, summarize_errors
season, week = 2025, 1
df = reconcile_props(season, week)
summ = summarize_errors(df)
print('recon_rows', len(df))
print(summ.to_string(index=False))
