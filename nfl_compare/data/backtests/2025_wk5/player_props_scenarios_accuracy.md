# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 8809
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_plus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_plus3 | Spread home +3.0 | 383.0000 | 3.5895 | 0.1213 | 0.1671 |
| v2_elo_minus100 | Home Elo -100 (diff) | 383.0000 | 3.5966 | 0.1218 | 0.1671 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 383.0000 | 3.5991 | 0.1216 | 0.1671 |
| v2_inj_away_plus2 | Away starters out +2 | 383.0000 | 3.5994 | 0.1216 | 0.1671 |
| v2_inj_home_plus1 | Home starters out +1 | 383.0000 | 3.5999 | 0.1218 | 0.1671 |
| v2_rest_minus7 | Home rest -7 days (diff) | 383.0000 | 3.6004 | 0.1217 | 0.1671 |
| v2_wind_plus20 | Wind +20mph (open) | 383.0000 | 3.6022 | 0.1214 | 0.1671 |
| v2_precip_plus25 | Precip +25% | 383.0000 | 3.6028 | 0.1220 | 0.1671 |
| v2_neutral_site | Neutral site | 383.0000 | 3.6032 | 0.1218 | 0.1671 |
| v2_total_minus3 | Total -3.0 | 383.0000 | 3.6033 | 0.1210 | 0.1671 |
| v2_blend_10_20 | Market blend m=0.10 t=0.20 | 383.0000 | 3.6037 | 0.1219 | 0.1671 |
| v2_wind_plus10 | Wind +10mph (open) | 383.0000 | 3.6045 | 0.1217 | 0.1671 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 264.0000 | 0.8939 | v2_baseline | 0.0843 |
| pass_attempts | 264.0000 | 0.9053 | v2_baseline | 0.8764 |
| pass_tds | 264.0000 | 0.8939 | v2_baseline | 0.1264 |
| pass_yards | 264.0000 | 0.9053 | v2_baseline | 6.0175 |
| rec_tds | 264.0000 | 0.2652 | v2_baseline | 0.3288 |
| rec_yards | 264.0000 | 0.1818 | v2_baseline | 18.7974 |
| receptions | 264.0000 | 0.1742 | v2_baseline | 1.4263 |
| rush_attempts | 264.0000 | 0.5833 | v2_baseline | 1.6349 |
| rush_tds | 264.0000 | 0.6742 | v2_baseline | 0.1329 |
| rush_yards | 264.0000 | 0.5871 | v2_baseline | 8.2235 |
| targets | 264.0000 | 0.1591 | v2_baseline | 2.0030 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 28.0000 | 7.8388 | 0.0538 |
| v2_spread_home_plus3 | QB | 28.0000 | 7.7006 | 0.0530 |
| v2_baseline | RB | 82.0000 | 4.2367 | 0.1534 |
| v2_spread_home_plus3 | RB | 82.0000 | 4.2369 | 0.1500 |
| v2_baseline | TE | 107.0000 | 1.9820 | 0.1010 |
| v2_spread_home_plus3 | TE | 107.0000 | 1.9493 | 0.1002 |
| v2_baseline | WR | 166.0000 | 3.0224 | 0.1313 |
| v2_spread_home_plus3 | WR | 166.0000 | 3.0377 | 0.1322 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_05_SF_LA | San Francisco 49ers | Mac Jones | QB | 183.9300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9000 | 5.0000 | 182.8000 | 342.0000 |
| baseline | v2_baseline | 2025_05_MIA_CAR | Carolina Panthers | Rico Dowdle | RB | 172.0200 | 23.2000 | 28.0000 | 5.3600 | 5.0000 | 47.5000 | 206.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_TB_SEA | Seattle Seahawks | Sam Darnold | QB | 170.6100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 29.2000 | 0.0000 | 215.5000 | 341.0000 |
| baseline | v2_baseline | 2025_05_TB_SEA | Tampa Bay Buccaneers | Baker Mayfield | QB | 168.1300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.1000 | 0.0000 | 230.5000 | 379.0000 |
| baseline | v2_baseline | 2025_05_NE_BUF | New England Patriots | Kyle Williams | WR | 164.0000 | 0.0000 | 66.0000 | 0.0000 | 10.0000 | 0.0000 | 65.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_TB_SEA | Tampa Bay Buccaneers | Josh Williams | RB | 159.0000 | 0.0000 | 4.0000 | 0.0000 | 2.0000 | 0.0000 | 135.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DEN_PHI | Philadelphia Eagles | Jalen Hurts | QB | 138.3500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.9000 | 0.0000 | 204.4000 | 280.0000 |
| baseline | v2_baseline | 2025_05_SF_LA | Los Angeles Rams | Matthew Stafford | QB | 135.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.2000 | 0.0000 | 273.9000 | 389.0000 |
| baseline | v2_baseline | 2025_05_NE_BUF | New England Patriots | Drake Maye | QB | 134.9100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.1000 | 0.0000 | 180.5000 | 273.0000 |
| baseline | v2_baseline | 2025_05_TB_SEA | Tampa Bay Buccaneers | Emeka Egbuka | WR | 133.4400 | 33.5000 | 163.0000 | 7.4600 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DAL_NYJ | New York Jets | Justin Fields | QB | 124.4400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.2000 | 3.0000 | 202.3000 | 283.0000 |
| baseline | v2_baseline | 2025_05_NE_BUF | New England Patriots | Stefon Diggs | WR | 120.6500 | 37.8000 | 146.0000 | 6.3200 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_SF_LA | San Francisco 49ers | Kendrick Bourne | WR | 110.6300 | 39.3000 | 142.0000 | 7.9600 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_SF_LA | San Francisco 49ers | Christian McCaffrey | RB | 110.1800 | 16.3000 | 82.0000 | 4.1600 | 9.0000 | 33.9000 | 57.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DAL_NYJ | Dallas Cowboys | Ryan Flournoy | WR | 100.6200 | 32.5000 | 114.0000 | 4.8600 | 9.0000 | 0.0000 | 10.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_TEN_ARI | Tennessee Titans | Calvin Ridley | WR | 100.2700 | 33.9000 | 131.0000 | 7.9700 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_TEN_ARI | Tennessee Titans | Cam Ward | QB | 99.5800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.1000 | 0.0000 | 178.4000 | 265.0000 |
| baseline | v2_baseline | 2025_05_KC_JAX | Kansas City Chiefs | Tyquan Thornton | WR | 98.0000 | 0.0000 | 90.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DAL_NYJ | Dallas Cowboys | Javonte Williams | RB | 97.5500 | 25.7000 | 4.0000 | 5.3300 | 2.0000 | 65.8000 | 135.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DAL_NYJ | New York Jets | Breece Hall | RB | 97.2800 | 23.8000 | 42.0000 | 5.7700 | 6.0000 | 36.5000 | 113.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DEN_PHI | Philadelphia Eagles | DeVonta Smith | WR | 97.1500 | 28.9000 | 114.0000 | 5.3400 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_MIN_CLE | Minnesota Vikings | Carson Wentz | QB | 96.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.8000 | 0.0000 | 321.8000 | 236.0000 |
| baseline | v2_baseline | 2025_05_TB_SEA | Seattle Seahawks | Jaxon Smith-Njigba | WR | 92.9800 | 44.3000 | 132.0000 | 7.5900 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_LV_IND | Las Vegas Raiders | Geno Smith | QB | 90.8800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.0000 | 3.0000 | 168.9000 | 228.0000 |
| baseline | v2_baseline | 2025_05_HOU_BAL | Baltimore Ravens | Cooper Rush | QB | 90.2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.5000 | 0.0000 | 243.6000 | 179.0000 |
| baseline | v2_baseline | 2025_05_TEN_ARI | Arizona Cardinals | Emari Demercado | RB | 85.3400 | 15.7000 | 0.0000 | 3.8500 | 1.0000 | 19.7000 | 81.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_WAS_LAC | Los Angeles Chargers | Justin Herbert | QB | 83.0300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.7000 | 20.0000 | 233.3000 | 166.0000 |
| baseline | v2_baseline | 2025_05_MIA_CAR | Miami Dolphins | Jaylen Waddle | WR | 80.7700 | 32.7000 | 110.0000 | 7.8600 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_MIA_CAR | Miami Dolphins | De'Von Achane | RB | 79.6900 | 18.3000 | 30.0000 | 4.5000 | 8.0000 | 66.9000 | 16.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_05_DET_CIN | Detroit Lions | Jared Goff | QB | 79.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.0000 | 0.0000 | 201.6000 | 258.0000 |

