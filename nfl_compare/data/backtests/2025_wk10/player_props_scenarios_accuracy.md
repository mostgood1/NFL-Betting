# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 8924
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 388.0000 | 3.7436 | 0.1227 | 0.1624 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 388.0000 | 3.7892 | 0.1232 | 0.1624 |
| v2_inj_away_plus2 | Away starters out +2 | 388.0000 | 3.7893 | 0.1232 | 0.1624 |
| v2_wind_plus20 | Wind +20mph (open) | 388.0000 | 3.7920 | 0.1233 | 0.1624 |
| v2_inj_home_plus1 | Home starters out +1 | 388.0000 | 3.7983 | 0.1233 | 0.1624 |
| v2_inj_home_plus2 | Home starters out +2 | 388.0000 | 3.7994 | 0.1234 | 0.1624 |
| v2_wind_plus10 | Wind +10mph (open) | 388.0000 | 3.8023 | 0.1233 | 0.1624 |
| v2_inj_away_plus1 | Away starters out +1 | 388.0000 | 3.8087 | 0.1234 | 0.1624 |
| v2_cold_minus15 | Temp -15F | 388.0000 | 3.8121 | 0.1236 | 0.1624 |
| v2_spread_home_plus3 | Spread home +3.0 | 388.0000 | 3.8140 | 0.1238 | 0.1624 |
| v2_neutral_site | Neutral site | 388.0000 | 3.8144 | 0.1235 | 0.1624 |
| v2_elo_plus100 | Home Elo +100 (diff) | 388.0000 | 3.8147 | 0.1236 | 0.1624 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 261.0000 | 0.9042 | v2_baseline | 0.0672 |
| pass_attempts | 261.0000 | 0.9080 | v2_baseline | 0.7921 |
| pass_tds | 261.0000 | 0.9080 | v2_baseline | 0.0997 |
| pass_yards | 261.0000 | 0.9080 | v2_baseline | 6.1567 |
| rec_tds | 261.0000 | 0.3142 | v2_baseline | 0.2632 |
| rec_yards | 261.0000 | 0.1916 | v2_baseline | 18.8049 |
| receptions | 261.0000 | 0.2107 | v2_baseline | 1.4943 |
| rush_attempts | 261.0000 | 0.5709 | v2_baseline | 1.9247 |
| rush_tds | 261.0000 | 0.6858 | v2_baseline | 0.1494 |
| rush_yards | 261.0000 | 0.5785 | v2_baseline | 10.0966 |
| targets | 261.0000 | 0.1571 | v2_baseline | 2.1825 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 28.0000 | 8.4278 | 0.0645 |
| v2_total_minus3 | QB | 28.0000 | 7.9585 | 0.0618 |
| v2_baseline | RB | 87.0000 | 4.4897 | 0.1877 |
| v2_total_minus3 | RB | 87.0000 | 4.4352 | 0.1860 |
| v2_baseline | TE | 105.0000 | 2.4691 | 0.1087 |
| v2_total_minus3 | TE | 105.0000 | 2.4235 | 0.1073 |
| v2_baseline | WR | 168.0000 | 3.0188 | 0.1095 |
| v2_total_minus3 | WR | 168.0000 | 3.0047 | 0.1097 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_10_ATL_IND | Indianapolis Colts | Jonathan Taylor | RB | 259.0700 | 22.4000 | 42.0000 | 6.2600 | 4.0000 | 30.5000 | 244.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_CLE_NYJ | New York Jets | Justin Fields | QB | 217.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 33.0000 | 8.0000 | 223.5000 | 54.0000 |
| baseline | v2_baseline | 2025_10_BUF_MIA | Miami Dolphins | De'Von Achane | RB | 175.2400 | 21.2000 | 51.0000 | 6.0800 | 6.0000 | 39.4000 | 174.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_LV_DEN | Las Vegas Raiders | Geno Smith | QB | 156.5800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 0.0000 | 279.3000 | 143.0000 |
| baseline | v2_baseline | 2025_10_PIT_LAC | Pittsburgh Steelers | Aaron Rodgers | QB | 139.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3000 | 0.0000 | 282.8000 | 161.0000 |
| baseline | v2_baseline | 2025_10_NE_TB | New England Patriots | TreVeyon Henderson | RB | 138.8700 | 18.5000 | 3.0000 | 3.2500 | 1.0000 | 33.7000 | 147.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_NE_TB | Tampa Bay Buccaneers | Josh Williams | RB | 132.0000 | 0.0000 | 119.0000 | 0.0000 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_BUF_MIA | Buffalo Bills | Josh Allen | QB | 130.5100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 35.7000 | 17.0000 | 216.8000 | 306.0000 |
| baseline | v2_baseline | 2025_10_NE_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 117.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.4000 | 0.0000 | 196.6000 | 273.0000 |
| baseline | v2_baseline | 2025_10_LA_SF | San Francisco 49ers | Christian McCaffrey | RB | 117.3900 | 17.8000 | 66.0000 | 3.8700 | 10.0000 | 80.5000 | 30.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_ARI_SEA | Seattle Seahawks | Sam Darnold | QB | 114.9700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.7000 | 0.0000 | 240.1000 | 178.0000 |
| baseline | v2_baseline | 2025_10_DET_WAS | Detroit Lions | Jahmyr Gibbs | RB | 113.7600 | 21.4000 | 30.0000 | 5.8000 | 4.0000 | 41.6000 | 142.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_LV_DEN | Denver Broncos | Bo Nix | QB | 112.7600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2000 | 0.0000 | 230.2000 | 150.0000 |
| baseline | v2_baseline | 2025_10_BUF_MIA | Miami Dolphins | Tua Tagovailoa | QB | 112.2100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.0000 | 0.0000 | 253.4000 | 173.0000 |
| baseline | v2_baseline | 2025_10_JAX_HOU | Jacksonville Jaguars | Trevor Lawrence | QB | 111.0800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.9000 | 2.0000 | 245.3000 | 158.0000 |
| baseline | v2_baseline | 2025_10_NO_CAR | New Orleans Saints | Tyler Shough | QB | 109.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 182.3000 | 282.0000 |
| baseline | v2_baseline | 2025_10_BAL_MIN | Minnesota Vikings | Jalen Nailor | WR | 100.3000 | 27.8000 | 124.0000 | 4.3500 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_JAX_HOU | Houston Texans | Nico Collins | WR | 100.0400 | 42.9000 | 136.0000 | 10.0200 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_NO_CAR | Carolina Panthers | Trevor Etienne | RB | 99.0000 | 0.0000 | 19.0000 | 0.0000 | 4.0000 | 0.0000 | 58.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_ARI_SEA | Arizona Cardinals | Trey McBride | TE | 94.9500 | 42.3000 | 127.0000 | 7.4100 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_PHI_GB | Philadelphia Eagles | Jalen Hurts | QB | 93.2700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 40.6000 | 13.0000 | 234.8000 | 183.0000 |
| baseline | v2_baseline | 2025_10_BAL_MIN | Baltimore Ravens | Lamar Jackson | QB | 90.6400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41.2000 | 22.0000 | 236.7000 | 176.0000 |
| baseline | v2_baseline | 2025_10_DET_WAS | Detroit Lions | Jared Goff | QB | 90.5400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6000 | 0.0000 | 250.6000 | 320.0000 |
| baseline | v2_baseline | 2025_10_ATL_IND | Indianapolis Colts | Michael Pittman Jr. | WR | 90.4100 | 97.0000 | 19.0000 | 10.7600 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_NE_TB | Tampa Bay Buccaneers | Emeka Egbuka | WR | 90.3300 | 35.4000 | 115.0000 | 6.8600 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_NE_TB | New England Patriots | Mack Hollins | WR | 88.1400 | 26.4000 | 106.0000 | 4.6500 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_DET_WAS | Detroit Lions | Jameson Williams | WR | 84.4100 | 36.8000 | 119.0000 | 7.6200 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_BUF_MIA | Buffalo Bills | Joshua Palmer | WR | 81.3200 | 34.0000 | 24.0000 | 3.4800 | 5.0000 | 0.0000 | 53.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_BAL_MIN | Minnesota Vikings | Aaron Jones Sr. | RB | 80.4100 | 35.4000 | 22.0000 | 4.7600 | 7.0000 | 101.3000 | 47.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_10_NO_CAR | New Orleans Saints | Alvin Kamara | RB | 76.8500 | 14.8000 | 32.0000 | 4.4100 | 3.0000 | 35.6000 | 83.0000 | 0.0000 | 0.0000 |

