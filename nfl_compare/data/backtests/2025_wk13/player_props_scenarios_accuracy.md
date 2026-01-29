# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10396
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 452.0000 | 3.3669 | 0.1078 | 0.1460 |
| v2_spread_home_plus3 | Spread home +3.0 | 452.0000 | 3.4028 | 0.1082 | 0.1460 |
| v2_wind_plus20 | Wind +20mph (open) | 452.0000 | 3.4127 | 0.1083 | 0.1460 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 452.0000 | 3.4166 | 0.1083 | 0.1460 |
| v2_inj_home_plus2 | Home starters out +2 | 452.0000 | 3.4188 | 0.1083 | 0.1460 |
| v2_inj_away_plus2 | Away starters out +2 | 452.0000 | 3.4190 | 0.1084 | 0.1460 |
| v2_wind_plus10 | Wind +10mph (open) | 452.0000 | 3.4253 | 0.1084 | 0.1460 |
| v2_inj_home_plus1 | Home starters out +1 | 452.0000 | 3.4290 | 0.1085 | 0.1460 |
| v2_inj_away_plus1 | Away starters out +1 | 452.0000 | 3.4306 | 0.1085 | 0.1460 |
| v2_precip_plus25 | Precip +25% | 452.0000 | 3.4328 | 0.1086 | 0.1460 |
| v2_baseline | Baseline | 452.0000 | 3.4334 | 0.1086 | 0.1460 |
| v2_precip_plus50 | Precip +50% | 452.0000 | 3.4345 | 0.1084 | 0.1460 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 299.0000 | 0.9030 | v2_baseline | 0.0637 |
| pass_attempts | 299.0000 | 0.9130 | v2_baseline | 0.6656 |
| pass_tds | 299.0000 | 0.9130 | v2_baseline | 0.0874 |
| pass_yards | 299.0000 | 0.9064 | v2_baseline | 6.2070 |
| rec_tds | 299.0000 | 0.3177 | v2_baseline | 0.2662 |
| rec_yards | 299.0000 | 0.2107 | v2_baseline | 16.9296 |
| receptions | 299.0000 | 0.2174 | v2_baseline | 1.4343 |
| rush_attempts | 299.0000 | 0.5886 | v2_baseline | 1.7657 |
| rush_tds | 299.0000 | 0.6957 | v2_baseline | 0.1127 |
| rush_yards | 299.0000 | 0.6020 | v2_baseline | 8.0937 |
| targets | 299.0000 | 0.1706 | v2_baseline | 2.1413 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 32.0000 | 7.9194 | 0.0569 |
| v2_total_minus3 | QB | 32.0000 | 7.4620 | 0.0560 |
| v2_baseline | RB | 101.0000 | 4.0883 | 0.1598 |
| v2_total_minus3 | RB | 101.0000 | 4.0768 | 0.1591 |
| v2_baseline | TE | 128.0000 | 2.0209 | 0.0686 |
| v2_total_minus3 | TE | 128.0000 | 1.9977 | 0.0671 |
| v2_baseline | WR | 191.0000 | 2.7374 | 0.1170 |
| v2_total_minus3 | WR | 191.0000 | 2.7074 | 0.1165 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_13_BUF_PIT | Buffalo Bills | Joshua Palmer | WR | 221.6400 | 71.2000 | 33.0000 | 8.2600 | 3.0000 | 0.0000 | 144.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_JAX_TEN | Tennessee Titans | Cam Ward | QB | 193.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4000 | 0.0000 | 327.9000 | 141.0000 |
| baseline | v2_baseline | 2025_13_NYG_NE | New York Giants | Jaxson Dart | QB | 143.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.5000 | 0.0000 | 253.8000 | 139.0000 |
| baseline | v2_baseline | 2025_13_CHI_PHI | Chicago Bears | Kyle Monangai | RB | 142.6100 | 18.4000 | 0.0000 | 3.4000 | 1.0000 | 25.2000 | 130.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_BUF_PIT | Buffalo Bills | Josh Allen | QB | 136.4900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.6000 | 16.0000 | 238.6000 | 123.0000 |
| baseline | v2_baseline | 2025_13_MIN_SEA | Seattle Seahawks | Sam Darnold | QB | 133.6600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.1000 | 0.0000 | 229.5000 | 128.0000 |
| baseline | v2_baseline | 2025_13_LV_LAC | Los Angeles Chargers | Kimani Vidal | RB | 131.1800 | 25.2000 | 11.0000 | 5.3400 | 1.0000 | 31.3000 | 126.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_BUF_PIT | Buffalo Bills | James Cook III | RB | 131.1500 | 32.4000 | 33.0000 | 5.7200 | 3.0000 | 39.3000 | 144.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_BUF_PIT | Pittsburgh Steelers | Aaron Rodgers | QB | 130.8900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.1000 | 0.0000 | 220.8000 | 117.0000 |
| baseline | v2_baseline | 2025_13_ATL_NYJ | Atlanta Falcons | Bijan Robinson | RB | 119.7700 | 16.2000 | 51.0000 | 4.1700 | 7.0000 | 64.2000 | 142.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_HOU_IND | Indianapolis Colts | Daniel Jones | QB | 116.5100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.7000 | 1.0000 | 276.0000 | 201.0000 |
| baseline | v2_baseline | 2025_13_HOU_IND | Indianapolis Colts | Michael Pittman Jr. | WR | 114.6900 | 116.0000 | 13.0000 | 10.3400 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_CHI_PHI | Chicago Bears | Caleb Williams | QB | 114.1500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.3000 | 5.0000 | 246.4000 | 154.0000 |
| baseline | v2_baseline | 2025_13_GB_DET | Green Bay Packers | Dontayvion Wicks | WR | 114.0000 | 0.0000 | 94.0000 | 0.0000 | 7.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_ARI_TB | Arizona Cardinals | Jacoby Brissett | QB | 109.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6000 | 0.0000 | 206.0000 | 301.0000 |
| baseline | v2_baseline | 2025_13_LV_LAC | Las Vegas Raiders | Geno Smith | QB | 107.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.7000 | -5.0000 | 239.8000 | 165.0000 |
| baseline | v2_baseline | 2025_13_GB_DET | Detroit Lions | Jameson Williams | WR | 107.2800 | 47.3000 | 144.0000 | 7.8800 | 10.0000 | 0.0000 | -5.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_KC_DAL | Dallas Cowboys | Dak Prescott | QB | 107.1500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.9000 | 0.0000 | 230.0000 | 320.0000 |
| baseline | v2_baseline | 2025_13_DEN_WAS | Denver Broncos | Bo Nix | QB | 105.5100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.3000 | 0.0000 | 246.8000 | 321.0000 |
| baseline | v2_baseline | 2025_13_LA_CAR | Carolina Panthers | Chuba Hubbard | RB | 104.3000 | 14.1000 | 41.0000 | 2.8000 | 2.0000 | 20.6000 | 83.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_CHI_PHI | Chicago Bears | D'Andre Swift | RB | 103.9900 | 26.5000 | 13.0000 | 5.4700 | 2.0000 | 46.2000 | 125.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_SF_CLE | San Francisco 49ers | Brock Purdy | QB | 103.6200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.3000 | 2.0000 | 234.1000 | 168.0000 |
| baseline | v2_baseline | 2025_13_NYG_NE | New England Patriots | Drake Maye | QB | 103.4400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8000 | 5.0000 | 196.0000 | 282.0000 |
| baseline | v2_baseline | 2025_13_LV_LAC | Los Angeles Chargers | Justin Herbert | QB | 100.5600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.5000 | 0.0000 | 220.7000 | 151.0000 |
| baseline | v2_baseline | 2025_13_DEN_WAS | Washington Commanders | Zach Ertz | TE | 93.6500 | 25.7000 | 106.0000 | 6.0600 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_CIN_BAL | Baltimore Ravens | Isaiah Likely | TE | 84.9500 | 16.6000 | 95.0000 | 3.3700 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_NYG_NE | New York Giants | Tyrone Tracy Jr. | RB | 81.7400 | 24.5000 | -3.0000 | 4.6300 | 1.0000 | 75.9000 | 36.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_NO_MIA | Miami Dolphins | De'Von Achane | RB | 80.0900 | 17.7000 | 0.0000 | 3.1800 | 1.0000 | 79.3000 | 134.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_ARI_TB | Arizona Cardinals | Michael Carter | RB | 80.0000 | 0.0000 | 47.0000 | 0.0000 | 6.0000 | 0.0000 | 17.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_13_CHI_PHI | Philadelphia Eagles | A.J. Brown | WR | 78.8800 | 59.1000 | 132.0000 | 10.7200 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

