# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 9453
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_minus3 | Spread home -3.0 | 411.0000 | 3.6135 | 0.1052 | 0.1363 |
| v2_total_minus3 | Total -3.0 | 411.0000 | 3.6139 | 0.1044 | 0.1363 |
| v2_inj_away_plus2 | Away starters out +2 | 411.0000 | 3.6222 | 0.1049 | 0.1363 |
| v2_wind_plus20 | Wind +20mph (open) | 411.0000 | 3.6230 | 0.1049 | 0.1363 |
| v2_inj_home_plus2 | Home starters out +2 | 411.0000 | 3.6241 | 0.1050 | 0.1363 |
| v2_inj_home_plus1 | Home starters out +1 | 411.0000 | 3.6246 | 0.1052 | 0.1363 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 411.0000 | 3.6249 | 0.1050 | 0.1363 |
| v2_inj_away_plus1 | Away starters out +1 | 411.0000 | 3.6293 | 0.1053 | 0.1363 |
| v2_wind_plus10 | Wind +10mph (open) | 411.0000 | 3.6322 | 0.1051 | 0.1363 |
| v2_precip_plus25 | Precip +25% | 411.0000 | 3.6347 | 0.1055 | 0.1363 |
| v2_rest_plus7 | Home rest +7 days (diff) | 411.0000 | 3.6347 | 0.1056 | 0.1363 |
| v2_elo_plus100 | Home Elo +100 (diff) | 411.0000 | 3.6365 | 0.1055 | 0.1363 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 278.0000 | 0.8957 | v2_baseline | 0.0851 |
| pass_attempts | 278.0000 | 0.9065 | v2_baseline | 0.9848 |
| pass_tds | 278.0000 | 0.9029 | v2_baseline | 0.1076 |
| pass_yards | 278.0000 | 0.9137 | v2_baseline | 5.6914 |
| rec_tds | 278.0000 | 0.3165 | v2_baseline | 0.2740 |
| rec_yards | 278.0000 | 0.1799 | v2_baseline | 19.7586 |
| receptions | 278.0000 | 0.1942 | v2_baseline | 1.5169 |
| rush_attempts | 278.0000 | 0.6259 | v2_baseline | 1.5131 |
| rush_tds | 278.0000 | 0.6871 | v2_baseline | 0.1241 |
| rush_yards | 278.0000 | 0.6259 | v2_baseline | 7.6046 |
| targets | 278.0000 | 0.1295 | v2_baseline | 2.3420 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 30.0000 | 7.9131 | 0.0827 |
| v2_spread_home_minus3 | QB | 30.0000 | 7.8190 | 0.0832 |
| v2_baseline | RB | 89.0000 | 4.0958 | 0.1394 |
| v2_spread_home_minus3 | RB | 89.0000 | 4.0672 | 0.1380 |
| v2_baseline | TE | 115.0000 | 2.3866 | 0.0970 |
| v2_spread_home_minus3 | TE | 115.0000 | 2.4029 | 0.0979 |
| v2_baseline | WR | 177.0000 | 2.9739 | 0.0977 |
| v2_spread_home_minus3 | WR | 177.0000 | 2.9515 | 0.0972 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_07_IND_LAC | Los Angeles Chargers | Justin Herbert | QB | 232.6700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.1000 | 7.0000 | 229.8000 | 420.0000 |
| baseline | v2_baseline | 2025_07_PHI_MIN | Philadelphia Eagles | DeVonta Smith | WR | 158.5800 | 32.5000 | 183.0000 | 7.2300 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_CAR_NYJ | New York Jets | Justin Fields | QB | 151.8900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.2000 | 15.0000 | 163.6000 | 46.0000 |
| baseline | v2_baseline | 2025_07_LV_KC | Las Vegas Raiders | Geno Smith | QB | 151.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8000 | 0.0000 | 183.6000 | 67.0000 |
| baseline | v2_baseline | 2025_07_ATL_SF | San Francisco 49ers | Christian McCaffrey | RB | 146.9800 | 27.9000 | 72.0000 | 5.8300 | 8.0000 | 44.1000 | 129.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_PHI_MIN | Philadelphia Eagles | Jalen Hurts | QB | 141.8300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 22.0000 | -5.0000 | 227.9000 | 326.0000 |
| baseline | v2_baseline | 2025_07_MIA_CLE | Miami Dolphins | Tua Tagovailoa | QB | 141.8200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.8000 | 0.0000 | 223.8000 | 100.0000 |
| baseline | v2_baseline | 2025_07_MIA_CLE | Cleveland Browns | Dillon Gabriel | QB | 138.1500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.7000 | 0.0000 | 233.0000 | 116.0000 |
| baseline | v2_baseline | 2025_07_TB_DET | Detroit Lions | Jahmyr Gibbs | RB | 128.2100 | 26.7000 | 82.0000 | 5.3300 | 3.0000 | 66.7000 | 136.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_PIT_CIN | Cincinnati Bengals | Ja'Marr Chase | WR | 126.9500 | 58.0000 | 161.0000 | 10.6300 | 25.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_IND_LAC | Los Angeles Chargers | Oronde Gadsden II | TE | 126.4500 | 42.1000 | 164.0000 | 7.6200 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_LV_KC | Kansas City Chiefs | Brashard Smith | RB | 105.0000 | 0.0000 | 42.0000 | 0.0000 | 5.0000 | 0.0000 | 39.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_NE_TEN | New England Patriots | Brandon Smith | WR | 105.0000 | 0.0000 | 42.0000 | 0.0000 | 5.0000 | 0.0000 | 39.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_NYG_DEN | Denver Broncos | Bo Nix | QB | 104.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.0000 | 27.0000 | 204.3000 | 279.0000 |
| baseline | v2_baseline | 2025_07_PHI_MIN | Minnesota Vikings | Jordan Addison | WR | 103.8700 | 32.7000 | 128.0000 | 8.1100 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_PIT_CIN | Pittsburgh Steelers | Pat Freiermuth | TE | 102.9300 | 13.2000 | 111.0000 | 3.6400 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_HOU_SEA | Houston Texans | Nick Chubb | RB | 101.9400 | 28.1000 | -5.0000 | 4.8600 | 3.0000 | 69.0000 | 16.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_TB_DET | Tampa Bay Buccaneers | Baker Mayfield | QB | 100.6600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.6000 | 0.0000 | 178.5000 | 228.0000 |
| baseline | v2_baseline | 2025_07_PIT_CIN | Cincinnati Bengals | Chase Brown | RB | 99.2500 | 30.4000 | -8.0000 | 5.7600 | 4.0000 | 51.9000 | 108.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_IND_LAC | Los Angeles Chargers | Keenan Allen | WR | 98.6300 | 34.2000 | 119.0000 | 7.0000 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_WAS_DAL | Dallas Cowboys | Javonte Williams | RB | 97.2300 | 30.2000 | 2.0000 | 5.7400 | 4.0000 | 56.7000 | 116.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_CAR_NYJ | Carolina Panthers | Bryce Young | QB | 95.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.1000 | 0.0000 | 198.1000 | 138.0000 |
| baseline | v2_baseline | 2025_07_MIA_CLE | Cleveland Browns | Quinshon Judkins | RB | 93.9200 | 29.3000 | 0.0000 | 6.2100 | 0.0000 | 42.2000 | 84.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_NO_CHI | Chicago Bears | D'Andre Swift | RB | 92.8600 | 23.2000 | 14.0000 | 5.2800 | 2.0000 | 51.4000 | 124.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_NO_CHI | Chicago Bears | Caleb Williams | QB | 92.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 23.7000 | 0.0000 | 225.8000 | 172.0000 |
| baseline | v2_baseline | 2025_07_IND_LAC | Indianapolis Colts | Daniel Jones | QB | 91.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 33.6000 | 0.0000 | 238.4000 | 288.0000 |
| baseline | v2_baseline | 2025_07_HOU_SEA | Houston Texans | Jaylin Noel | WR | 88.0000 | 0.0000 | 77.0000 | 0.0000 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_NYG_DEN | Denver Broncos | Marvin Mims Jr. | WR | 83.6500 | 20.5000 | 85.0000 | 4.8500 | 7.0000 | 0.0000 | 13.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_HOU_SEA | Seattle Seahawks | Jaxon Smith-Njigba | WR | 82.8000 | 49.2000 | 123.0000 | 8.4900 | 15.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_07_PHI_MIN | Philadelphia Eagles | A.J. Brown | WR | 82.6000 | 43.3000 | 121.0000 | 9.6000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

