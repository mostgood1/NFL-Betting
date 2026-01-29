# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10695
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 465.0000 | 3.3458 | 0.1029 | 0.1011 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 465.0000 | 3.3975 | 0.1052 | 0.1011 |
| v2_wind_plus20 | Wind +20mph (open) | 465.0000 | 3.3986 | 0.1051 | 0.1011 |
| v2_inj_home_plus2 | Home starters out +2 | 465.0000 | 3.4010 | 0.1051 | 0.1011 |
| v2_inj_away_plus2 | Away starters out +2 | 465.0000 | 3.4077 | 0.1053 | 0.1011 |
| v2_inj_away_plus1 | Away starters out +1 | 465.0000 | 3.4143 | 0.1057 | 0.1011 |
| v2_wind_plus10 | Wind +10mph (open) | 465.0000 | 3.4170 | 0.1055 | 0.1011 |
| v2_inj_home_plus1 | Home starters out +1 | 465.0000 | 3.4201 | 0.1058 | 0.1011 |
| v2_spread_home_minus3 | Spread home -3.0 | 465.0000 | 3.4212 | 0.1056 | 0.1011 |
| v2_precip_plus25 | Precip +25% | 465.0000 | 3.4223 | 0.1062 | 0.1011 |
| v2_rest_minus7 | Home rest -7 days (diff) | 465.0000 | 3.4257 | 0.1061 | 0.1011 |
| v2_baseline | Baseline | 465.0000 | 3.4265 | 0.1061 | 0.1011 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 295.0000 | 0.9254 | v2_baseline | 0.0483 |
| pass_attempts | 295.0000 | 0.9085 | v2_baseline | 0.7012 |
| pass_tds | 295.0000 | 0.9186 | v2_baseline | 0.0948 |
| pass_yards | 295.0000 | 0.9153 | v2_baseline | 6.1187 |
| rec_tds | 295.0000 | 0.3627 | v2_baseline | 0.2455 |
| rec_yards | 295.0000 | 0.1797 | v2_baseline | 17.0064 |
| receptions | 295.0000 | 0.1831 | v2_baseline | 1.4611 |
| rush_attempts | 295.0000 | 0.5898 | v2_baseline | 1.7380 |
| rush_tds | 295.0000 | 0.7220 | v2_baseline | 0.1140 |
| rush_yards | 295.0000 | 0.5797 | v2_baseline | 8.0849 |
| targets | 295.0000 | 0.1661 | v2_baseline | 2.0791 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 32.0000 | 8.7201 | 0.1008 |
| v2_total_minus3 | QB | 32.0000 | 8.2393 | 0.0998 |
| v2_baseline | RB | 108.0000 | 3.8892 | 0.1573 |
| v2_total_minus3 | RB | 108.0000 | 3.8320 | 0.1512 |
| v2_baseline | TE | 121.0000 | 1.8989 | 0.0977 |
| v2_total_minus3 | TE | 121.0000 | 1.8613 | 0.0956 |
| v2_baseline | WR | 204.0000 | 2.8599 | 0.0848 |
| v2_total_minus3 | WR | 204.0000 | 2.8240 | 0.0821 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_18_TEN_JAX | Tennessee Titans | Cam Ward | QB | 282.1700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 7.0000 | 291.0000 | 52.0000 |
| baseline | v2_baseline | 2025_18_DAL_NYG | Dallas Cowboys | Dak Prescott | QB | 227.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.2000 | 0.0000 | 255.4000 | 70.0000 |
| baseline | v2_baseline | 2025_18_NYJ_BUF | Buffalo Bills | Ray Davis | RB | 201.0000 | 0.0000 | 23.0000 | 0.0000 | 3.0000 | 0.0000 | 151.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_SEA_SF | San Francisco 49ers | Brock Purdy | QB | 176.8600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.2000 | 0.0000 | 265.1000 | 127.0000 |
| baseline | v2_baseline | 2025_18_LAC_DEN | Denver Broncos | Bo Nix | QB | 171.5700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 25.9000 | 0.0000 | 272.2000 | 141.0000 |
| baseline | v2_baseline | 2025_18_DET_CHI | Detroit Lions | Jared Goff | QB | 140.8900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.3000 | 0.0000 | 213.4000 | 331.0000 |
| baseline | v2_baseline | 2025_18_KC_LV | Kansas City Chiefs | Chris Oladokun | QB | 128.1800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.6000 | 0.0000 | 169.9000 | 58.0000 |
| baseline | v2_baseline | 2025_18_CAR_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 124.5900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.3000 | 0.0000 | 292.2000 | 203.0000 |
| baseline | v2_baseline | 2025_18_DET_CHI | Detroit Lions | Amon-Ra St. Brown | WR | 114.5500 | 36.6000 | 139.0000 | 8.0600 | 15.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_DET_CHI | Chicago Bears | Caleb Williams | QB | 111.6900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.0000 | 0.0000 | 291.0000 | 212.0000 |
| baseline | v2_baseline | 2025_18_NYJ_BUF | New York Jets | Brady Cook | QB | 110.6900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2000 | 2.0000 | 164.6000 | 60.0000 |
| baseline | v2_baseline | 2025_18_MIA_NE | New England Patriots | Rhamondre Stevenson | RB | 102.9300 | 10.4000 | 22.0000 | 2.4800 | 2.0000 | 42.5000 | 131.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_IND_HOU | Houston Texans | Xavier Hutchinson | WR | 99.0000 | 0.0000 | 84.0000 | 0.0000 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_CAR_TB | Carolina Panthers | Bryce Young | QB | 98.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.1000 | 0.0000 | 182.0000 | 266.0000 |
| baseline | v2_baseline | 2025_18_IND_HOU | Indianapolis Colts | Michael Pittman Jr. | WR | 97.8700 | 108.5000 | 20.0000 | 9.4600 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_BAL_PIT | Baltimore Ravens | Zay Flowers | WR | 95.5500 | 44.0000 | 138.0000 | 6.8500 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_BAL_PIT | Pittsburgh Steelers | Aaron Rodgers | QB | 91.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.4000 | 0.0000 | 221.5000 | 294.0000 |
| baseline | v2_baseline | 2025_18_IND_HOU | Indianapolis Colts | Alec Pierce | WR | 90.6300 | 42.6000 | 132.0000 | 7.1200 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_BAL_PIT | Pittsburgh Steelers | Kenneth Gainwell | RB | 86.7300 | 9.9000 | 64.0000 | 3.1600 | 10.0000 | 26.5000 | 10.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_WAS_PHI | Philadelphia Eagles | Tank Bigsby | RB | 82.3100 | 10.4000 | 31.0000 | 2.6900 | 1.0000 | 24.7000 | 75.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_DAL_NYG | New York Giants | Tyrone Tracy Jr. | RB | 82.0400 | 22.0000 | 56.0000 | 4.7200 | 10.0000 | 67.1000 | 103.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_NYJ_BUF | Buffalo Bills | James Cook III | RB | 81.0100 | 21.0000 | 1.0000 | 4.3200 | 3.0000 | 46.3000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_GB_MIN | Minnesota Vikings | Justin Jefferson | WR | 80.2600 | 32.2000 | 101.0000 | 7.0600 | 11.0000 | 0.0000 | 3.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_BAL_PIT | Baltimore Ravens | Derrick Henry | RB | 79.4700 | 15.2000 | 0.0000 | 3.5300 | 1.0000 | 69.6000 | 126.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_GB_MIN | Minnesota Vikings | Jordan Mason | RB | 76.4100 | 7.7000 | 0.0000 | 2.2500 | 0.0000 | 32.5000 | 94.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_ARI_LA | Los Angeles Rams | Tyler Higbee | TE | 76.0000 | 18.6000 | 91.0000 | 4.3500 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_KC_LV | Kansas City Chiefs | Brashard Smith | RB | 76.0000 | 0.0000 | 2.0000 | 0.0000 | 4.0000 | 0.0000 | 56.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_GB_MIN | Green Bay Packers | Chris Brooks | RB | 74.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 61.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_18_BAL_PIT | Baltimore Ravens | Lamar Jackson | QB | 72.0800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 45.4000 | 11.0000 | 217.6000 | 238.0000 |
| baseline | v2_baseline | 2025_18_CAR_TB | Tampa Bay Buccaneers | Bucky Irving | RB | 71.5300 | 24.0000 | 13.0000 | 5.6800 | 2.0000 | 46.2000 | 85.0000 | 0.0000 | 0.0000 |

