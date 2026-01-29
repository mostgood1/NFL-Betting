# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10143
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 441.0000 | 3.3039 | 0.1163 | 0.1587 |
| v2_wind_plus20 | Wind +20mph (open) | 441.0000 | 3.3276 | 0.1170 | 0.1587 |
| v2_inj_away_plus2 | Away starters out +2 | 441.0000 | 3.3278 | 0.1170 | 0.1587 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 441.0000 | 3.3298 | 0.1170 | 0.1587 |
| v2_inj_home_plus2 | Home starters out +2 | 441.0000 | 3.3316 | 0.1169 | 0.1587 |
| v2_spread_home_minus3 | Spread home -3.0 | 441.0000 | 3.3346 | 0.1165 | 0.1587 |
| v2_inj_away_plus1 | Away starters out +1 | 441.0000 | 3.3370 | 0.1171 | 0.1587 |
| v2_wind_plus10 | Wind +10mph (open) | 441.0000 | 3.3375 | 0.1171 | 0.1587 |
| v2_inj_home_plus1 | Home starters out +1 | 441.0000 | 3.3376 | 0.1170 | 0.1587 |
| v2_precip_plus25 | Precip +25% | 441.0000 | 3.3405 | 0.1172 | 0.1587 |
| v2_rest_minus7 | Home rest -7 days (diff) | 441.0000 | 3.3406 | 0.1173 | 0.1587 |
| v2_rest_plus7 | Home rest +7 days (diff) | 441.0000 | 3.3411 | 0.1171 | 0.1587 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 299.0000 | 0.8997 | v2_baseline | 0.0803 |
| pass_attempts | 299.0000 | 0.9064 | v2_baseline | 0.8177 |
| pass_tds | 299.0000 | 0.9097 | v2_baseline | 0.0685 |
| pass_yards | 299.0000 | 0.8963 | v2_baseline | 6.8719 |
| rec_tds | 299.0000 | 0.2943 | v2_baseline | 0.3024 |
| rec_yards | 299.0000 | 0.2174 | v2_baseline | 15.7175 |
| receptions | 299.0000 | 0.2140 | v2_baseline | 1.3021 |
| rush_attempts | 299.0000 | 0.5652 | v2_baseline | 1.6413 |
| rush_tds | 299.0000 | 0.6789 | v2_baseline | 0.1218 |
| rush_yards | 299.0000 | 0.5686 | v2_baseline | 7.9347 |
| targets | 299.0000 | 0.1806 | v2_baseline | 1.9278 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 31.0000 | 8.7761 | 0.0832 |
| v2_total_minus3 | QB | 31.0000 | 8.5900 | 0.0813 |
| v2_baseline | RB | 93.0000 | 3.9932 | 0.1712 |
| v2_total_minus3 | RB | 93.0000 | 3.9587 | 0.1700 |
| v2_baseline | TE | 122.0000 | 1.5916 | 0.0840 |
| v2_total_minus3 | TE | 122.0000 | 1.5464 | 0.0829 |
| v2_baseline | WR | 195.0000 | 2.5234 | 0.1179 |
| v2_total_minus3 | WR | 195.0000 | 2.5155 | 0.1172 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_04_IND_LA | Los Angeles Rams | Matthew Stafford | QB | 196.9200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 0.0000 | 206.0000 | 375.0000 |
| baseline | v2_baseline | 2025_04_GB_DAL | Green Bay Packers | Jordan Love | QB | 173.9100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.4000 | 0.0000 | 190.4000 | 337.0000 |
| baseline | v2_baseline | 2025_04_MIN_PIT | Minnesota Vikings | Carson Wentz | QB | 172.7400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.0000 | 0.0000 | 217.9000 | 350.0000 |
| baseline | v2_baseline | 2025_04_TEN_HOU | Tennessee Titans | Cam Ward | QB | 166.3100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0000 | 262.1000 | 108.0000 |
| baseline | v2_baseline | 2025_04_BAL_KC | Baltimore Ravens | Lamar Jackson | QB | 160.1700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.7000 | 7.0000 | 271.8000 | 147.0000 |
| baseline | v2_baseline | 2025_04_IND_LA | Los Angeles Rams | Puka Nacua | WR | 146.0200 | 38.1000 | 170.0000 | 8.3200 | 15.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_CIN_DEN | Denver Broncos | Bo Nix | QB | 138.0400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.6000 | 1.0000 | 210.3000 | 326.0000 |
| baseline | v2_baseline | 2025_04_LAC_NYG | New York Giants | Jaxson Dart | QB | 134.7300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.3000 | 31.0000 | 214.8000 | 111.0000 |
| baseline | v2_baseline | 2025_04_CHI_LV | Las Vegas Raiders | Geno Smith | QB | 126.6900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.5000 | 2.0000 | 209.0000 | 117.0000 |
| baseline | v2_baseline | 2025_04_WAS_ATL | Atlanta Falcons | Bijan Robinson | RB | 126.6200 | 18.1000 | 106.0000 | 5.5500 | 5.0000 | 40.5000 | 75.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_WAS_ATL | Atlanta Falcons | Michael Penix Jr. | QB | 111.3200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.2000 | 0.0000 | 227.0000 | 313.0000 |
| baseline | v2_baseline | 2025_04_LAC_NYG | Los Angeles Chargers | Omarion Hampton | RB | 108.3500 | 21.6000 | 37.0000 | 5.4600 | 5.0000 | 38.4000 | 128.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_JAX_SF | Jacksonville Jaguars | Travis Etienne Jr. | RB | 105.9200 | 26.7000 | 1.0000 | 5.7100 | 2.0000 | 56.6000 | 124.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_NYJ_MIA | Miami Dolphins | Tua Tagovailoa | QB | 103.3200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.6000 | 0.0000 | 257.8000 | 177.0000 |
| baseline | v2_baseline | 2025_04_WAS_ATL | Washington Commanders | Marcus Mariota | QB | 103.1600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41.7000 | -2.0000 | 199.9000 | 156.0000 |
| baseline | v2_baseline | 2025_04_NYJ_MIA | New York Jets | Adonai Mitchell | WR | 103.0000 | 0.0000 | 96.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_TEN_HOU | Houston Texans | Woody Marks | RB | 102.9900 | 12.7000 | 50.0000 | 3.5200 | 6.0000 | 19.4000 | 69.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_PHI_TB | Philadelphia Eagles | Jalen Hurts | QB | 102.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.7000 | 15.0000 | 211.3000 | 130.0000 |
| baseline | v2_baseline | 2025_04_MIN_PIT | Minnesota Vikings | Zavier Scott | RB | 101.7600 | 14.4000 | 43.0000 | 2.4100 | 9.0000 | 53.6000 | 1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_CIN_DEN | Denver Broncos | J.K. Dobbins | RB | 100.9800 | 21.0000 | 4.0000 | 5.4700 | 1.0000 | 31.2000 | 101.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_PHI_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 98.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.4000 | 0.0000 | 231.2000 | 289.0000 |
| baseline | v2_baseline | 2025_04_NO_BUF | New Orleans Saints | Spencer Rattler | QB | 98.3400 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10.7000 | 6.0000 | 210.5000 | 126.0000 |
| baseline | v2_baseline | 2025_04_PHI_TB | Tampa Bay Buccaneers | Bucky Irving | RB | 97.9500 | 19.9000 | 102.0000 | 4.3200 | 7.0000 | 73.2000 | 63.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_IND_LA | Los Angeles Rams | Tutu Atwell | WR | 92.0000 | 0.0000 | 88.0000 | 0.0000 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_SEA_ARI | Arizona Cardinals | Kyler Murray | QB | 91.7800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 23.4000 | 3.0000 | 251.2000 | 200.0000 |
| baseline | v2_baseline | 2025_04_CAR_NE | New England Patriots | Drake Maye | QB | 90.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.3000 | 0.0000 | 251.9000 | 203.0000 |
| baseline | v2_baseline | 2025_04_MIN_PIT | Minnesota Vikings | Justin Jefferson | WR | 90.2800 | 45.5000 | 126.0000 | 7.5800 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_IND_LA | Indianapolis Colts | Daniel Jones | QB | 89.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 37.0000 | 0.0000 | 223.9000 | 262.0000 |
| baseline | v2_baseline | 2025_04_MIN_PIT | Pittsburgh Steelers | DK Metcalf | WR | 88.2500 | 41.1000 | 126.0000 | 7.4400 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_04_SEA_ARI | Seattle Seahawks | Sam Darnold | QB | 87.7600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.2000 | 0.0000 | 195.7000 | 242.0000 |

