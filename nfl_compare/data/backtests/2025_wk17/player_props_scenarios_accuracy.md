# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10511
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_plus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_plus3 | Spread home +3.0 | 457.0000 | 3.3036 | 0.1008 | 0.1225 |
| v2_total_minus3 | Total -3.0 | 457.0000 | 3.3185 | 0.0991 | 0.1225 |
| v2_wind_plus20 | Wind +20mph (open) | 457.0000 | 3.3388 | 0.1003 | 0.1225 |
| v2_sigma_high | High volatility | 457.0000 | 3.3397 | 0.1009 | 0.1225 |
| v2_inj_away_plus2 | Away starters out +2 | 457.0000 | 3.3402 | 0.1005 | 0.1225 |
| v2_inj_home_plus2 | Home starters out +2 | 457.0000 | 3.3410 | 0.1004 | 0.1225 |
| v2_neutral_site | Neutral site | 457.0000 | 3.3451 | 0.1009 | 0.1225 |
| v2_inj_away_plus1 | Away starters out +1 | 457.0000 | 3.3452 | 0.1008 | 0.1225 |
| v2_wind_plus10 | Wind +10mph (open) | 457.0000 | 3.3462 | 0.1005 | 0.1225 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 457.0000 | 3.3468 | 0.1005 | 0.1225 |
| v2_baseline | Baseline | 457.0000 | 3.3470 | 0.1008 | 0.1225 |
| v2_cold_minus15 | Temp -15F | 457.0000 | 3.3472 | 0.1010 | 0.1225 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 308.0000 | 0.9026 | v2_baseline | 0.0718 |
| pass_attempts | 308.0000 | 0.9156 | v2_baseline | 0.6701 |
| pass_tds | 308.0000 | 0.9091 | v2_baseline | 0.0873 |
| pass_yards | 308.0000 | 0.9123 | v2_baseline | 6.2878 |
| rec_tds | 308.0000 | 0.3474 | v2_baseline | 0.2394 |
| rec_yards | 308.0000 | 0.2013 | v2_baseline | 16.3891 |
| receptions | 308.0000 | 0.2305 | v2_baseline | 1.3325 |
| rush_attempts | 308.0000 | 0.6071 | v2_baseline | 1.5538 |
| rush_tds | 308.0000 | 0.7110 | v2_baseline | 0.1274 |
| rush_yards | 308.0000 | 0.6104 | v2_baseline | 8.1413 |
| targets | 308.0000 | 0.1721 | v2_baseline | 1.9162 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 32.0000 | 8.1646 | 0.0977 |
| v2_spread_home_plus3 | QB | 32.0000 | 7.8572 | 0.0989 |
| v2_baseline | RB | 109.0000 | 3.7598 | 0.1342 |
| v2_spread_home_plus3 | RB | 109.0000 | 3.7461 | 0.1356 |
| v2_baseline | TE | 118.0000 | 1.9815 | 0.0937 |
| v2_spread_home_plus3 | TE | 118.0000 | 1.9784 | 0.0933 |
| v2_baseline | WR | 198.0000 | 2.7045 | 0.0872 |
| v2_spread_home_plus3 | WR | 198.0000 | 2.6817 | 0.0865 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_17_BAL_GB | Baltimore Ravens | Derrick Henry | RB | 222.7200 | 19.6000 | 0.0000 | 4.3400 | 0.0000 | 44.0000 | 216.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_DET_MIN | Minnesota Vikings | Max Brosmer | QB | 173.6900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.6000 | 0.0000 | 204.0000 | 51.0000 |
| baseline | v2_baseline | 2025_17_TB_MIA | Tampa Bay Buccaneers | Baker Mayfield | QB | 173.3700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.9000 | 0.0000 | 216.2000 | 346.0000 |
| baseline | v2_baseline | 2025_17_SEA_CAR | Carolina Panthers | Bryce Young | QB | 173.1900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.3000 | -3.0000 | 200.8000 | 54.0000 |
| baseline | v2_baseline | 2025_17_LA_ATL | Atlanta Falcons | Bijan Robinson | RB | 160.3500 | 20.4000 | 34.0000 | 4.3300 | 8.0000 | 59.1000 | 195.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_CHI_SF | Chicago Bears | Caleb Williams | QB | 158.1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 23.9000 | 3.0000 | 209.3000 | 330.0000 |
| baseline | v2_baseline | 2025_17_DEN_KC | Kansas City Chiefs | Chris Oladokun | QB | 155.5200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.8000 | 0.0000 | 204.0000 | 66.0000 |
| baseline | v2_baseline | 2025_17_PHI_BUF | Buffalo Bills | Joshua Palmer | WR | 154.2200 | 58.6000 | 3.0000 | 5.7900 | 4.0000 | 0.0000 | 74.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_NO_TEN | New Orleans Saints | Tyler Shough | QB | 139.5400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.7000 | -2.0000 | 204.0000 | 333.0000 |
| baseline | v2_baseline | 2025_17_PHI_BUF | Philadelphia Eagles | Jalen Hurts | QB | 131.0900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 29.3000 | 2.0000 | 201.3000 | 110.0000 |
| baseline | v2_baseline | 2025_17_TB_MIA | Tampa Bay Buccaneers | Jalen McMillan | WR | 130.0000 | 0.0000 | 114.0000 | 0.0000 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_CHI_SF | Chicago Bears | Luther Burden III | WR | 123.8100 | 24.8000 | 138.0000 | 3.7300 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_CHI_SF | San Francisco 49ers | Christian McCaffrey | RB | 122.8100 | 19.0000 | 41.0000 | 4.3000 | 6.0000 | 51.3000 | 140.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_LA_ATL | Atlanta Falcons | Kirk Cousins | QB | 120.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.0000 | 1.0000 | 219.8000 | 126.0000 |
| baseline | v2_baseline | 2025_17_SEA_CAR | Carolina Panthers | Trevor Etienne | RB | 114.0000 | 0.0000 | 16.0000 | 0.0000 | 3.0000 | 0.0000 | 76.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_CHI_SF | San Francisco 49ers | Brock Purdy | QB | 112.2100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.3000 | 7.0000 | 217.4000 | 303.0000 |
| baseline | v2_baseline | 2025_17_PHI_BUF | Buffalo Bills | Brandin Cooks | WR | 111.0000 | 0.0000 | 101.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_JAX_IND | Jacksonville Jaguars | Parker Washington | WR | 110.6800 | 22.1000 | 115.0000 | 3.8100 | 10.0000 | 0.0000 | -5.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_DET_MIN | Minnesota Vikings | Jordan Addison | WR | 105.7000 | 31.5000 | 0.0000 | 5.7400 | 1.0000 | 0.0000 | 65.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_DAL_WAS | Dallas Cowboys | Dak Prescott | QB | 104.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.1000 | 6.0000 | 219.7000 | 307.0000 |
| baseline | v2_baseline | 2025_17_ARI_CIN | Cincinnati Bengals | Joe Burrow | QB | 100.7100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.6000 | 1.0000 | 220.5000 | 305.0000 |
| baseline | v2_baseline | 2025_17_TB_MIA | Tampa Bay Buccaneers | Chris Godwin Jr. | WR | 98.2000 | 18.6000 | 108.0000 | 3.6800 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_BAL_GB | Green Bay Packers | Josh Jacobs | RB | 97.1100 | 21.0000 | 0.0000 | 4.3300 | 1.0000 | 60.9000 | 3.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_DAL_WAS | Dallas Cowboys | Malik Davis | RB | 96.7800 | 16.5000 | 0.0000 | 2.6500 | 2.0000 | 37.0000 | 103.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_PHI_BUF | Buffalo Bills | James Cook III | RB | 94.9400 | 30.5000 | 12.0000 | 4.2700 | 3.0000 | 60.3000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_NE_NYJ | New England Patriots | Drake Maye | QB | 93.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.0000 | 2.0000 | 191.3000 | 256.0000 |
| baseline | v2_baseline | 2025_17_SEA_CAR | Seattle Seahawks | Zach Charbonnet | RB | 90.4000 | 12.0000 | 12.0000 | 2.7200 | 2.0000 | 30.3000 | 110.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_NYG_LV | New York Giants | Wan'Dale Robinson | WR | 82.0600 | 43.0000 | 113.0000 | 7.7000 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_NO_TEN | New Orleans Saints | Chris Olave | WR | 81.2800 | 44.3000 | 119.0000 | 7.5900 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_17_PIT_CLE | Cleveland Browns | Dylan Sampson | RB | 80.3500 | 22.9000 | 0.0000 | 4.3000 | 3.0000 | 75.9000 | 27.0000 | 0.0000 | 0.0000 |

