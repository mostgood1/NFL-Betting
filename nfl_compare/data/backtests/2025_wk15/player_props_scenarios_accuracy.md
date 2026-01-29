# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10442
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_minus3 | Spread home -3.0 | 454.0000 | 3.7535 | 0.1155 | 0.1432 |
| v2_total_minus3 | Total -3.0 | 454.0000 | 3.7589 | 0.1138 | 0.1432 |
| v2_wind_plus20 | Wind +20mph (open) | 454.0000 | 3.7615 | 0.1149 | 0.1432 |
| v2_inj_home_plus2 | Home starters out +2 | 454.0000 | 3.7638 | 0.1151 | 0.1432 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 454.0000 | 3.7656 | 0.1150 | 0.1432 |
| v2_inj_away_plus2 | Away starters out +2 | 454.0000 | 3.7660 | 0.1150 | 0.1432 |
| v2_wind_plus10 | Wind +10mph (open) | 454.0000 | 3.7682 | 0.1152 | 0.1432 |
| v2_inj_home_plus1 | Home starters out +1 | 454.0000 | 3.7688 | 0.1152 | 0.1432 |
| v2_inj_away_plus1 | Away starters out +1 | 454.0000 | 3.7689 | 0.1154 | 0.1432 |
| v2_cold_minus15 | Temp -15F | 454.0000 | 3.7734 | 0.1156 | 0.1432 |
| v2_elo_minus100 | Home Elo -100 (diff) | 454.0000 | 3.7743 | 0.1156 | 0.1432 |
| v2_rest_minus7 | Home rest -7 days (diff) | 454.0000 | 3.7749 | 0.1155 | 0.1432 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 295.0000 | 0.9017 | v2_baseline | 0.0713 |
| pass_attempts | 295.0000 | 0.9119 | v2_baseline | 0.7576 |
| pass_tds | 295.0000 | 0.9119 | v2_baseline | 0.1035 |
| pass_yards | 295.0000 | 0.9119 | v2_baseline | 6.4187 |
| rec_tds | 295.0000 | 0.3288 | v2_baseline | 0.3161 |
| rec_yards | 295.0000 | 0.2102 | v2_baseline | 19.7629 |
| receptions | 295.0000 | 0.2136 | v2_baseline | 1.5079 |
| rush_attempts | 295.0000 | 0.5763 | v2_baseline | 1.8392 |
| rush_tds | 295.0000 | 0.6847 | v2_baseline | 0.1379 |
| rush_yards | 295.0000 | 0.5763 | v2_baseline | 8.4167 |
| targets | 295.0000 | 0.1729 | v2_baseline | 2.2189 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 32.0000 | 8.1304 | 0.0752 |
| v2_spread_home_minus3 | QB | 32.0000 | 8.1292 | 0.0760 |
| v2_baseline | RB | 104.0000 | 4.0913 | 0.1629 |
| v2_spread_home_minus3 | RB | 104.0000 | 4.0587 | 0.1614 |
| v2_baseline | TE | 122.0000 | 2.5768 | 0.0914 |
| v2_spread_home_minus3 | TE | 122.0000 | 2.5927 | 0.0921 |
| v2_baseline | WR | 196.0000 | 3.1254 | 0.1122 |
| v2_spread_home_minus3 | WR | 196.0000 | 3.0828 | 0.1121 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_15_LAC_KC | Kansas City Chiefs | Gardner Minshew | QB | 236.4200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.8000 | 0.0000 | 211.1000 | 22.0000 |
| baseline | v2_baseline | 2025_15_BUF_NE | Buffalo Bills | Joshua Palmer | WR | 216.9000 | 85.0000 | 4.0000 | 7.2600 | 3.0000 | 0.0000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_IND_SEA | Indianapolis Colts | Philip Rivers | QB | 215.5800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7000 | -5.0000 | 310.9000 | 120.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Atlanta Falcons | Kirk Cousins | QB | 197.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.3000 | 0.0000 | 200.6000 | 373.0000 |
| baseline | v2_baseline | 2025_15_DET_LA | Los Angeles Rams | Matthew Stafford | QB | 167.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.8000 | 0.0000 | 230.6000 | 368.0000 |
| baseline | v2_baseline | 2025_15_CAR_NO | Carolina Panthers | Mitchell Evans | TE | 152.0000 | 0.0000 | 132.0000 | 0.0000 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Atlanta Falcons | David Sills V | WR | 148.1600 | 34.9000 | 166.0000 | 4.2400 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_DET_LA | Los Angeles Rams | Puka Nacua | WR | 147.8500 | 51.3000 | 181.0000 | 8.1800 | 13.0000 | 0.0000 | 8.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_CAR_NO | New Orleans Saints | Tyler Shough | QB | 144.6400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.4000 | 15.0000 | 150.0000 | 272.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Atlanta Falcons | Kyle Pitts Sr. | TE | 141.4300 | 39.1000 | 166.0000 | 6.0900 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_ARI_HOU | Houston Texans | Jawhar Jordan | RB | 137.0000 | 0.0000 | 17.0000 | 0.0000 | 2.0000 | 0.0000 | 101.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_NYJ_JAX | Jacksonville Jaguars | Trevor Lawrence | QB | 135.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.1000 | 3.0000 | 211.3000 | 330.0000 |
| baseline | v2_baseline | 2025_15_CAR_NO | Carolina Panthers | Trevor Etienne | RB | 124.0000 | 0.0000 | 73.0000 | 0.0000 | 4.0000 | 0.0000 | 32.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_DET_LA | Detroit Lions | Amon-Ra St. Brown | WR | 115.8400 | 61.1000 | 164.0000 | 10.5000 | 18.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 115.2200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.0000 | 0.0000 | 198.4000 | 277.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Atlanta Falcons | Bijan Robinson | RB | 111.0700 | 15.4000 | 82.0000 | 4.0300 | 11.0000 | 61.0000 | 93.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_BUF_NE | New England Patriots | TreVeyon Henderson | RB | 109.6400 | 18.6000 | 13.0000 | 4.6800 | 3.0000 | 47.2000 | 148.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_WAS_NYG | Washington Commanders | Jacory Croskey-Merritt | RB | 107.8900 | 10.7000 | 0.0000 | 2.4200 | 0.0000 | 15.9000 | 96.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_ARI_HOU | Arizona Cardinals | Trey McBride | TE | 106.4300 | 39.3000 | 134.0000 | 8.1900 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_MIA_PIT | Miami Dolphins | De'Von Achane | RB | 105.0500 | 16.2000 | 67.0000 | 3.4100 | 6.0000 | 97.0000 | 60.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_BUF_NE | New England Patriots | Kyle Williams | WR | 105.0000 | 0.0000 | 10.0000 | 0.0000 | 1.0000 | 0.0000 | 78.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_IND_SEA | Indianapolis Colts | Michael Pittman Jr. | WR | 103.5500 | 120.2000 | 26.0000 | 10.6700 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_NYJ_JAX | Jacksonville Jaguars | Travis Etienne Jr. | RB | 101.2600 | 25.1000 | 73.0000 | 4.3900 | 4.0000 | 79.3000 | 32.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_ATL_TB | Tampa Bay Buccaneers | Mike Evans | WR | 99.9200 | 38.7000 | 132.0000 | 8.6000 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_GB_DEN | Denver Broncos | Bo Nix | QB | 98.0300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.4000 | 8.0000 | 227.7000 | 302.0000 |
| baseline | v2_baseline | 2025_15_MIA_PIT | Pittsburgh Steelers | Kenneth Gainwell | RB | 94.5100 | 16.7000 | 46.0000 | 3.4000 | 7.0000 | 30.0000 | 80.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_BUF_NE | Buffalo Bills | James Cook III | RB | 91.7100 | 32.9000 | 16.0000 | 4.8100 | 3.0000 | 59.9000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_BUF_NE | Buffalo Bills | James Cook III | RB | 89.7300 | 32.9000 | 4.0000 | 4.8100 | 3.0000 | 59.9000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_15_TEN_SF | San Francisco 49ers | Brock Purdy | QB | 86.7900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 28.5000 | 5.0000 | 243.7000 | 295.0000 |
| baseline | v2_baseline | 2025_15_BAL_CIN | Baltimore Ravens | Derrick Henry | RB | 85.4300 | 14.5000 | 0.0000 | 3.7300 | 0.0000 | 36.0000 | 100.0000 | 0.0000 | 0.0000 |

