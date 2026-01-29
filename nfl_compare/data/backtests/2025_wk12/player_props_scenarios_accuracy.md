# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 9131
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 397.0000 | 3.5190 | 0.1060 | 0.1184 |
| v2_spread_home_minus3 | Spread home -3.0 | 397.0000 | 3.5193 | 0.1093 | 0.1184 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 397.0000 | 3.5418 | 0.1075 | 0.1184 |
| v2_inj_home_plus2 | Home starters out +2 | 397.0000 | 3.5425 | 0.1078 | 0.1184 |
| v2_inj_away_plus2 | Away starters out +2 | 397.0000 | 3.5460 | 0.1077 | 0.1184 |
| v2_sigma_high | High volatility | 397.0000 | 3.5528 | 0.1084 | 0.1184 |
| v2_inj_away_plus1 | Away starters out +1 | 397.0000 | 3.5536 | 0.1078 | 0.1184 |
| v2_wind_plus10 | Wind +10mph (open) | 397.0000 | 3.5537 | 0.1078 | 0.1184 |
| v2_baseline | Baseline | 397.0000 | 3.5538 | 0.1084 | 0.1184 |
| v2_precip_plus25 | Precip +25% | 397.0000 | 3.5557 | 0.1085 | 0.1184 |
| v2_wind_plus20 | Wind +20mph (open) | 397.0000 | 3.5558 | 0.1076 | 0.1184 |
| v2_inj_home_plus1 | Home starters out +1 | 397.0000 | 3.5565 | 0.1080 | 0.1184 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 262.0000 | 0.9160 | v2_baseline | 0.0711 |
| pass_attempts | 262.0000 | 0.9008 | v2_baseline | 0.8459 |
| pass_tds | 262.0000 | 0.9122 | v2_baseline | 0.1140 |
| pass_yards | 262.0000 | 0.9008 | v2_baseline | 6.5654 |
| rec_tds | 262.0000 | 0.3244 | v2_baseline | 0.2752 |
| rec_yards | 262.0000 | 0.1985 | v2_baseline | 18.0707 |
| receptions | 262.0000 | 0.2405 | v2_baseline | 1.3524 |
| rush_attempts | 262.0000 | 0.5725 | v2_baseline | 1.7327 |
| rush_tds | 262.0000 | 0.7099 | v2_baseline | 0.1202 |
| rush_yards | 262.0000 | 0.6069 | v2_baseline | 8.0350 |
| targets | 262.0000 | 0.1679 | v2_baseline | 1.9091 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 28.0000 | 8.9156 | 0.0725 |
| v2_total_minus3 | QB | 28.0000 | 8.9106 | 0.0713 |
| v2_baseline | RB | 88.0000 | 4.4917 | 0.1555 |
| v2_total_minus3 | RB | 88.0000 | 4.4115 | 0.1502 |
| v2_baseline | TE | 110.0000 | 1.8331 | 0.0829 |
| v2_total_minus3 | TE | 110.0000 | 1.7966 | 0.0799 |
| v2_baseline | WR | 171.0000 | 2.8617 | 0.1064 |
| v2_total_minus3 | WR | 171.0000 | 2.8471 | 0.1057 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_12_TB_LA | Tampa Bay Buccaneers | Baker Mayfield | QB | 242.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.3000 | 0.0000 | 241.2000 | 41.0000 |
| baseline | v2_baseline | 2025_12_NYG_DET | Detroit Lions | Jahmyr Gibbs | RB | 217.6900 | 20.3000 | 45.0000 | 5.5300 | 12.0000 | 42.0000 | 219.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_BUF_HOU | Buffalo Bills | Joshua Palmer | WR | 204.4700 | 78.7000 | 13.0000 | 7.2300 | 3.0000 | 0.0000 | 116.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_PHI_DAL | Philadelphia Eagles | Jalen Hurts | QB | 143.5900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 37.5000 | 14.0000 | 193.3000 | 289.0000 |
| baseline | v2_baseline | 2025_12_MIN_GB | Green Bay Packers | Jordan Love | QB | 139.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.8000 | 3.0000 | 255.4000 | 139.0000 |
| baseline | v2_baseline | 2025_12_CAR_SF | Carolina Panthers | Trevor Etienne | RB | 138.0000 | 0.0000 | 30.0000 | 0.0000 | 4.0000 | 0.0000 | 86.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_SEA_TEN | Seattle Seahawks | Jaxon Smith-Njigba | WR | 133.1100 | 42.2000 | 167.0000 | 8.6800 | 10.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_CAR_SF | Carolina Panthers | Bryce Young | QB | 131.1900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.3000 | 0.0000 | 275.8000 | 169.0000 |
| baseline | v2_baseline | 2025_12_NYG_DET | New York Giants | Wan'Dale Robinson | WR | 125.9000 | 42.5000 | 156.0000 | 7.0300 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_PHI_DAL | Dallas Cowboys | George Pickens | WR | 118.1300 | 36.5000 | 146.0000 | 6.6800 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_IND_KC | Kansas City Chiefs | Kareem Hunt | RB | 114.9400 | 16.3000 | 26.0000 | 3.6000 | 4.0000 | 23.6000 | 104.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_PHI_DAL | Dallas Cowboys | Dak Prescott | QB | 113.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.3000 | 0.0000 | 261.6000 | 354.0000 |
| baseline | v2_baseline | 2025_12_NYG_DET | Detroit Lions | Amon-Ra St. Brown | WR | 111.4800 | 42.6000 | 149.0000 | 9.9100 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_MIN_GB | Minnesota Vikings | J.J. McCarthy | QB | 111.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4000 | 0.0000 | 184.6000 | 87.0000 |
| baseline | v2_baseline | 2025_12_NE_CIN | New England Patriots | Drake Maye | QB | 106.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.6000 | 0.0000 | 216.4000 | 294.0000 |
| baseline | v2_baseline | 2025_12_IND_KC | Kansas City Chiefs | Patrick Mahomes | QB | 104.2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.6000 | 0.0000 | 282.9000 | 352.0000 |
| baseline | v2_baseline | 2025_12_BUF_HOU | Buffalo Bills | James Cook III | RB | 98.5100 | 39.5000 | 21.0000 | 5.4000 | 4.0000 | 64.8000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_ATL_NO | New Orleans Saints | Alvin Kamara | RB | 96.0600 | 16.8000 | 4.0000 | 4.2300 | 2.0000 | 72.0000 | 11.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_JAX_ARI | Jacksonville Jaguars | Trevor Lawrence | QB | 95.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 7.0000 | 178.2000 | 256.0000 |
| baseline | v2_baseline | 2025_12_CAR_SF | San Francisco 49ers | Brock Purdy | QB | 94.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 25.7000 | 9.0000 | 263.5000 | 193.0000 |
| baseline | v2_baseline | 2025_12_JAX_ARI | Arizona Cardinals | Jacoby Brissett | QB | 94.0900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.3000 | 0.0000 | 261.7000 | 317.0000 |
| baseline | v2_baseline | 2025_12_IND_KC | Indianapolis Colts | Daniel Jones | QB | 92.6400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 33.8000 | 0.0000 | 227.6000 | 181.0000 |
| baseline | v2_baseline | 2025_12_TB_LA | Los Angeles Rams | Matthew Stafford | QB | 91.9800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.1000 | 0.0000 | 197.2000 | 273.0000 |
| baseline | v2_baseline | 2025_12_CLE_LV | Las Vegas Raiders | Geno Smith | QB | 89.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.4000 | -5.0000 | 232.5000 | 285.0000 |
| baseline | v2_baseline | 2025_12_NYJ_BAL | Baltimore Ravens | Lamar Jackson | QB | 86.4300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.9000 | 0.0000 | 198.8000 | 153.0000 |
| baseline | v2_baseline | 2025_12_ATL_NO | New Orleans Saints | Tyler Shough | QB | 85.7900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1000 | 10.0000 | 181.0000 | 243.0000 |
| baseline | v2_baseline | 2025_12_BUF_HOU | Buffalo Bills | James Cook III | RB | 85.7700 | 39.5000 | 13.0000 | 5.4000 | 3.0000 | 64.8000 | 116.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_MIN_GB | Green Bay Packers | Emanuel Wilson | RB | 85.7400 | 19.0000 | 18.0000 | 3.1400 | 2.0000 | 42.0000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_NE_CIN | Cincinnati Bengals | Chase Brown | RB | 84.9700 | 21.8000 | 6.0000 | 5.5200 | 4.0000 | 46.5000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_12_SEA_TEN | Tennessee Titans | Cam Ward | QB | 83.7600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.5000 | 7.0000 | 326.0000 | 256.0000 |

