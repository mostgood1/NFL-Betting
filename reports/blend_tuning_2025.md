# 2025 Blend Tuning Summary

Artifacts:
- Grid sweep: [nfl_compare/data/backtests/2025_wk17/blend_sweep_grid/blend_sweep.csv](nfl_compare/data/backtests/2025_wk17/blend_sweep_grid/blend_sweep.csv)
- Best summary: [nfl_compare/data/backtests/2025_wk17/blend_sweep_grid/blend_best_summary.csv](nfl_compare/data/backtests/2025_wk17/blend_sweep_grid/blend_best_summary.csv)
- Weekly (with splits): [nfl_compare/data/backtests/2025_wk17/games_weekly.csv](nfl_compare/data/backtests/2025_wk17/games_weekly.csv)

Findings:
- Max accuracy objectives favor no blend:
  - Winners/ATS/O-U best at blend_margin=0.00, blend_total=0.00 (acc_home_win=63.75%, ATS=64.58%, O/U=64.58%).
- MAE improvements:
  - Min MAE total at blend_total=0.15 (blend_margin=0.00): mae_total=10.68.
  - Min MAE margin at blend_margin=0.35 (blend_total=0.00): mae_margin=10.54, but winners accuracy drops (55.42%).

Recommendations:
- Keep margin blend off for classification tasks (winners/ATS). 
- Consider mild total blend (0.10–0.15) if optimizing MAE for totals without hurting O/U accuracy.
- Use weekly split metrics (acc_pred_home/away, acc_actual_home/away) to monitor bias; current season baseline is slightly home-leaning.

*Generated on 2025-12-25.*