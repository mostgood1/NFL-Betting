# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10442
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 454.0000 | 3.0713 | 0.1082 | 0.1366 |
| v2_spread_home_minus3 | Spread home -3.0 | 454.0000 | 3.1033 | 0.1096 | 0.1366 |
| v2_wind_plus20 | Wind +20mph (open) | 454.0000 | 3.1111 | 0.1092 | 0.1366 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 454.0000 | 3.1126 | 0.1094 | 0.1366 |
| v2_inj_home_plus2 | Home starters out +2 | 454.0000 | 3.1148 | 0.1093 | 0.1366 |
| v2_inj_away_plus2 | Away starters out +2 | 454.0000 | 3.1151 | 0.1093 | 0.1366 |
| v2_wind_plus10 | Wind +10mph (open) | 454.0000 | 3.1234 | 0.1095 | 0.1366 |
| v2_inj_away_plus1 | Away starters out +1 | 454.0000 | 3.1256 | 0.1095 | 0.1366 |
| v2_inj_home_plus1 | Home starters out +1 | 454.0000 | 3.1277 | 0.1096 | 0.1366 |
| v2_cold_minus15 | Temp -15F | 454.0000 | 3.1299 | 0.1096 | 0.1366 |
| v2_sigma_high | High volatility | 454.0000 | 3.1321 | 0.1095 | 0.1366 |
| v2_precip_plus25 | Precip +25% | 454.0000 | 3.1325 | 0.1098 | 0.1366 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 297.0000 | 0.9192 | v2_baseline | 0.0664 |
| pass_attempts | 297.0000 | 0.9091 | v2_baseline | 0.8282 |
| pass_tds | 297.0000 | 0.9158 | v2_baseline | 0.1036 |
| pass_yards | 297.0000 | 0.9293 | v2_baseline | 5.0522 |
| rec_tds | 297.0000 | 0.3199 | v2_baseline | 0.2683 |
| rec_yards | 297.0000 | 0.2189 | v2_baseline | 16.1281 |
| receptions | 297.0000 | 0.2290 | v2_baseline | 1.4001 |
| rush_attempts | 297.0000 | 0.6061 | v2_baseline | 1.4540 |
| rush_tds | 297.0000 | 0.7003 | v2_baseline | 0.1204 |
| rush_yards | 297.0000 | 0.5993 | v2_baseline | 6.9796 |
| targets | 297.0000 | 0.1582 | v2_baseline | 2.0748 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 31.0000 | 6.8989 | 0.0341 |
| v2_total_minus3 | QB | 31.0000 | 6.6327 | 0.0330 |
| v2_baseline | RB | 95.0000 | 3.7130 | 0.1622 |
| v2_total_minus3 | RB | 95.0000 | 3.6656 | 0.1604 |
| v2_baseline | TE | 126.0000 | 1.6544 | 0.0864 |
| v2_total_minus3 | TE | 126.0000 | 1.6452 | 0.0854 |
| v2_baseline | WR | 202.0000 | 2.7401 | 0.1111 |
| v2_total_minus3 | WR | 202.0000 | 2.6842 | 0.1095 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_03_PIT_NE | Pittsburgh Steelers | Roman Wilson | WR | 251.9200 | 37.1000 | 0.0000 | 5.4400 | 0.0000 | 0.0000 | 3.0000 | 0.0000 | 160.0000 |
| baseline | v2_baseline | 2025_03_ATL_CAR | Carolina Panthers | Bryce Young | QB | 191.5400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.8000 | 0.0000 | 281.7000 | 121.0000 |
| baseline | v2_baseline | 2025_03_CIN_MIN | Cincinnati Bengals | Jake Browning | QB | 153.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.7000 | 3.0000 | 283.6000 | 140.0000 |
| baseline | v2_baseline | 2025_03_DEN_LAC | Denver Broncos | Bo Nix | QB | 148.7800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.5000 | -3.0000 | 262.3000 | 153.0000 |
| baseline | v2_baseline | 2025_03_DEN_LAC | Los Angeles Chargers | Justin Herbert | QB | 143.2900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.3000 | 2.0000 | 199.0000 | 300.0000 |
| baseline | v2_baseline | 2025_03_MIA_BUF | Miami Dolphins | Tua Tagovailoa | QB | 141.8100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.5000 | 2.0000 | 279.1000 | 146.0000 |
| baseline | v2_baseline | 2025_03_DAL_CHI | Chicago Bears | Luther Burden III | WR | 115.0000 | 0.0000 | 101.0000 | 0.0000 | 3.0000 | 0.0000 | 7.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_LV_WAS | Las Vegas Raiders | Tre Tucker | WR | 114.6100 | 38.6000 | 145.0000 | 8.7800 | 9.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_DET_BAL | Detroit Lions | David Montgomery | RB | 109.1900 | 12.5000 | 13.0000 | 2.6500 | 1.0000 | 46.7000 | 151.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_ARI_SF | Arizona Cardinals | Kyler Murray | QB | 103.6100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.2000 | 10.0000 | 243.2000 | 159.0000 |
| baseline | v2_baseline | 2025_03_PIT_NE | Pittsburgh Steelers | Aaron Rodgers | QB | 102.0700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.8000 | 0.0000 | 220.2000 | 139.0000 |
| baseline | v2_baseline | 2025_03_CIN_MIN | Minnesota Vikings | Jordan Mason | RB | 101.3400 | 22.5000 | 0.0000 | 4.9100 | 0.0000 | 47.2000 | 116.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_LA_PHI | Los Angeles Rams | Matthew Stafford | QB | 96.4200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 0.0000 | 272.6000 | 196.0000 |
| baseline | v2_baseline | 2025_03_DAL_CHI | Chicago Bears | Caleb Williams | QB | 94.6100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.9000 | 12.0000 | 223.2000 | 298.0000 |
| baseline | v2_baseline | 2025_03_ARI_SF | San Francisco 49ers | Christian McCaffrey | RB | 91.5300 | 22.4000 | 88.0000 | 4.9400 | 15.0000 | 47.5000 | 52.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_GB_CLE | Cleveland Browns | Quinshon Judkins | RB | 90.3500 | 23.5000 | 1.0000 | 5.6700 | 2.0000 | 39.1000 | 94.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_NYJ_TB | Tampa Bay Buccaneers | Sterling Shepard | WR | 90.0000 | 0.0000 | 80.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_LV_WAS | Washington Commanders | Deebo Samuel | WR | 88.3500 | 72.4000 | 11.0000 | 6.6600 | 3.0000 | 0.0000 | 18.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_LV_WAS | Washington Commanders | Jeremy McNichols | RB | 82.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 78.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_PIT_NE | New England Patriots | Drake Maye | QB | 81.9300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.4000 | 3.0000 | 230.6000 | 268.0000 |
| baseline | v2_baseline | 2025_03_PIT_NE | New England Patriots | Rhamondre Stevenson | RB | 78.7600 | 22.6000 | 38.0000 | 4.4000 | 3.0000 | 68.0000 | 18.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_DET_BAL | Baltimore Ravens | Zay Flowers | WR | 78.2300 | 77.9000 | 13.0000 | 11.3900 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_ATL_CAR | Carolina Panthers | Trevor Etienne | RB | 76.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 56.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_DEN_LAC | Los Angeles Chargers | Omarion Hampton | RB | 75.9500 | 20.1000 | 59.0000 | 4.2800 | 8.0000 | 45.7000 | 70.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_HOU_JAX | Jacksonville Jaguars | Trevor Lawrence | QB | 74.4700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.4000 | 0.0000 | 272.4000 | 222.0000 |
| baseline | v2_baseline | 2025_03_KC_NYG | New York Giants | Malik Nabers | WR | 74.2000 | 77.4000 | 13.0000 | 11.8100 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_DAL_CHI | Dallas Cowboys | KaVontae Turpin | WR | 74.0000 | 0.0000 | 64.0000 | 0.0000 | 3.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_LA_PHI | Philadelphia Eagles | A.J. Brown | WR | 73.1000 | 41.1000 | 109.0000 | 7.3400 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_HOU_JAX | Jacksonville Jaguars | Travis Etienne Jr. | RB | 72.8600 | 37.3000 | 0.0000 | 5.1700 | 2.0000 | 81.6000 | 56.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_03_ARI_SF | San Francisco 49ers | Ricky Pearsall | WR | 71.9300 | 51.4000 | 117.0000 | 8.4000 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

