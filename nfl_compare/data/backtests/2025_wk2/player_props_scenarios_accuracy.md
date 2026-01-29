# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10511
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 457.0000 | 3.4804 | 0.1255 | 0.1707 |
| v2_wind_plus20 | Wind +20mph (open) | 457.0000 | 3.5174 | 0.1258 | 0.1707 |
| v2_inj_home_plus2 | Home starters out +2 | 457.0000 | 3.5218 | 0.1254 | 0.1707 |
| v2_spread_home_plus3 | Spread home +3.0 | 457.0000 | 3.5276 | 0.1250 | 0.1707 |
| v2_elo_minus100 | Home Elo -100 (diff) | 457.0000 | 3.5297 | 0.1257 | 0.1707 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 457.0000 | 3.5335 | 0.1260 | 0.1707 |
| v2_sigma_low | Low volatility | 457.0000 | 3.5338 | 0.1262 | 0.1707 |
| v2_precip_plus25 | Precip +25% | 457.0000 | 3.5346 | 0.1259 | 0.1707 |
| v2_inj_away_plus1 | Away starters out +1 | 457.0000 | 3.5353 | 0.1262 | 0.1707 |
| v2_wind_plus10 | Wind +10mph (open) | 457.0000 | 3.5356 | 0.1262 | 0.1707 |
| v2_cold_minus15 | Temp -15F | 457.0000 | 3.5399 | 0.1260 | 0.1707 |
| v2_precip_plus50 | Precip +50% | 457.0000 | 3.5422 | 0.1259 | 0.1707 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 308.0000 | 0.9058 | v2_baseline | 0.0815 |
| pass_attempts | 308.0000 | 0.8994 | v2_baseline | 1.0331 |
| pass_tds | 308.0000 | 0.8961 | v2_baseline | 0.1195 |
| pass_yards | 308.0000 | 0.9026 | v2_baseline | 8.4768 |
| rec_tds | 308.0000 | 0.2078 | v2_baseline | 0.3187 |
| rec_yards | 308.0000 | 0.1623 | v2_baseline | 17.2739 |
| receptions | 308.0000 | 0.1818 | v2_baseline | 1.4552 |
| rush_attempts | 308.0000 | 0.5942 | v2_baseline | 1.2850 |
| rush_tds | 308.0000 | 0.6818 | v2_baseline | 0.1116 |
| rush_yards | 308.0000 | 0.5812 | v2_baseline | 7.1152 |
| targets | 308.0000 | 0.1558 | v2_baseline | 1.8865 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | FB | 3.0000 | 1.0909 | 0.0000 |
| v2_total_minus3 | FB | 3.0000 | 1.0909 | 0.0000 |
| v2_baseline | QB | 44.0000 | 9.3853 | 0.0771 |
| v2_total_minus3 | QB | 44.0000 | 8.9516 | 0.0766 |
| v2_baseline | RB | 97.0000 | 3.7543 | 0.1361 |
| v2_total_minus3 | RB | 97.0000 | 3.7059 | 0.1340 |
| v2_baseline | TE | 119.0000 | 1.7347 | 0.0834 |
| v2_total_minus3 | TE | 119.0000 | 1.6791 | 0.0813 |
| v2_baseline | WR | 194.0000 | 2.8660 | 0.1606 |
| v2_total_minus3 | WR | 194.0000 | 2.8490 | 0.1614 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_02_JAX_CIN | Cincinnati Bengals | Jake Browning | QB | 283.7100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3000 | 1.0000 | 0.0000 | 241.0000 |
| baseline | v2_baseline | 2025_02_ATL_MIN | Atlanta Falcons | Michael Penix Jr. | QB | 232.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.8000 | 0.0000 | 337.3000 | 135.0000 |
| baseline | v2_baseline | 2025_02_NYG_DAL | New York Giants | Russell Wilson | QB | 220.5700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.6000 | 8.0000 | 238.2000 | 450.0000 |
| baseline | v2_baseline | 2025_02_JAX_CIN | Cincinnati Bengals | Joe Burrow | QB | 181.2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.5000 | 0.0000 | 226.4000 | 76.0000 |
| baseline | v2_baseline | 2025_02_BUF_NYJ | New York Jets | Justin Fields | QB | 167.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.0000 | 21.0000 | 163.7000 | 27.0000 |
| baseline | v2_baseline | 2025_02_BUF_NYJ | Buffalo Bills | Josh Allen | QB | 151.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.4000 | 7.0000 | 269.6000 | 148.0000 |
| baseline | v2_baseline | 2025_02_JAX_CIN | Cincinnati Bengals | Ja'Marr Chase | WR | 151.5700 | 33.1000 | 165.0000 | 6.3000 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_DEN_IND | Indianapolis Colts | Daniel Jones | QB | 151.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.5000 | -4.0000 | 216.6000 | 316.0000 |
| baseline | v2_baseline | 2025_02_PHI_KC | Philadelphia Eagles | Jalen Hurts | QB | 146.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.0000 | 6.0000 | 212.0000 | 101.0000 |
| baseline | v2_baseline | 2025_02_SEA_PIT | Seattle Seahawks | Sam Darnold | QB | 143.1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.2000 | 0.0000 | 169.0000 | 295.0000 |
| baseline | v2_baseline | 2025_02_BUF_NYJ | Buffalo Bills | James Cook | RB | 133.6300 | 25.2000 | 3.0000 | 5.6900 | 1.0000 | 38.7000 | 132.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_DEN_IND | Indianapolis Colts | Jonathan Taylor | RB | 131.3900 | 21.4000 | 50.0000 | 3.8100 | 3.0000 | 72.0000 | 165.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_PHI_KC | Kansas City Chiefs | Patrick Mahomes | QB | 116.5400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.6000 | 0.0000 | 278.9000 | 187.0000 |
| baseline | v2_baseline | 2025_02_NE_MIA | Miami Dolphins | Tua Tagovailoa | QB | 112.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.7000 | 0.0000 | 211.0000 | 315.0000 |
| baseline | v2_baseline | 2025_02_ATL_MIN | Atlanta Falcons | Bijan Robinson | RB | 111.4500 | 32.7000 | 25.0000 | 6.3300 | 5.0000 | 51.4000 | 143.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_WAS_GB | Washington Commanders | Jayden Daniels | QB | 109.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 25.8000 | 0.0000 | 263.8000 | 200.0000 |
| baseline | v2_baseline | 2025_02_NYG_DAL | New York Giants | Wan'Dale Robinson | WR | 109.1300 | 37.3000 | 142.0000 | 8.2500 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_CAR_ARI | Carolina Panthers | Bryce Young | QB | 107.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.0000 | 0.0000 | 255.1000 | 328.0000 |
| baseline | v2_baseline | 2025_02_NYG_DAL | New York Giants | Malik Nabers | WR | 105.7100 | 63.2000 | 167.0000 | 12.8100 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_NE_MIA | New England Patriots | Rhamondre Stevenson | RB | 102.7000 | 20.2000 | 88.0000 | 4.9100 | 5.0000 | 25.1000 | 54.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_TB_HOU | Tampa Bay Buccaneers | Baker Mayfield | QB | 100.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.7000 | 0.0000 | 289.4000 | 215.0000 |
| baseline | v2_baseline | 2025_02_DEN_IND | Denver Broncos | Bo Nix | QB | 99.0700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.5000 | -1.0000 | 270.9000 | 206.0000 |
| baseline | v2_baseline | 2025_02_NE_MIA | New England Patriots | Kyle Williams | WR | 97.2700 | 3.0000 | 14.0000 | 0.4700 | 2.0000 | 0.0000 | 66.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_CHI_DET | Detroit Lions | Amon-Ra St. Brown | WR | 96.7100 | 34.6000 | 115.0000 | 7.4600 | 11.0000 | 0.0000 | 7.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_JAX_CIN | Jacksonville Jaguars | Parker Washington | WR | 93.0000 | 0.0000 | 81.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_WAS_GB | Green Bay Packers | Tucker Kraft | TE | 89.0900 | 36.1000 | 124.0000 | 6.6500 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_CHI_DET | Chicago Bears | Caleb Williams | QB | 88.8300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 25.0000 | 8.0000 | 266.3000 | 207.0000 |
| baseline | v2_baseline | 2025_02_CHI_DET | Detroit Lions | Jahmyr Gibbs | RB | 88.6800 | 38.0000 | 10.0000 | 7.8200 | 3.0000 | 41.8000 | 94.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_02_NYG_DAL | Dallas Cowboys | Dak Prescott | QB | 88.2800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.4000 | 0.0000 | 295.0000 | 361.0000 |
| baseline | v2_baseline | 2025_02_NE_MIA | Miami Dolphins | De'Von Achane | RB | 87.0800 | 18.2000 | 92.0000 | 4.7300 | 10.0000 | 33.7000 | 30.0000 | 0.0000 | 0.0000 |

