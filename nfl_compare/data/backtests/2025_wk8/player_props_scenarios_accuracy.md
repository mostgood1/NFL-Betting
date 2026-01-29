# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 8280
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 360.0000 | 3.3701 | 0.1232 | 0.1611 |
| v2_wind_plus20 | Wind +20mph (open) | 360.0000 | 3.3955 | 0.1240 | 0.1611 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 360.0000 | 3.3999 | 0.1240 | 0.1611 |
| v2_inj_home_plus2 | Home starters out +2 | 360.0000 | 3.4032 | 0.1240 | 0.1611 |
| v2_inj_away_plus2 | Away starters out +2 | 360.0000 | 3.4061 | 0.1241 | 0.1611 |
| v2_inj_home_plus1 | Home starters out +1 | 360.0000 | 3.4062 | 0.1244 | 0.1611 |
| v2_wind_plus10 | Wind +10mph (open) | 360.0000 | 3.4072 | 0.1242 | 0.1611 |
| v2_inj_away_plus1 | Away starters out +1 | 360.0000 | 3.4128 | 0.1244 | 0.1611 |
| v2_elo_plus100 | Home Elo +100 (diff) | 360.0000 | 3.4154 | 0.1244 | 0.1611 |
| v2_precip_plus25 | Precip +25% | 360.0000 | 3.4155 | 0.1245 | 0.1611 |
| v2_cold_minus15 | Temp -15F | 360.0000 | 3.4166 | 0.1245 | 0.1611 |
| v2_precip_plus50 | Precip +50% | 360.0000 | 3.4168 | 0.1245 | 0.1611 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 251.0000 | 0.9243 | v2_baseline | 0.0541 |
| pass_attempts | 251.0000 | 0.9243 | v2_baseline | 0.5563 |
| pass_tds | 251.0000 | 0.9283 | v2_baseline | 0.1004 |
| pass_yards | 251.0000 | 0.9243 | v2_baseline | 4.8294 |
| rec_tds | 251.0000 | 0.3227 | v2_baseline | 0.2841 |
| rec_yards | 251.0000 | 0.2191 | v2_baseline | 15.7816 |
| receptions | 251.0000 | 0.2112 | v2_baseline | 1.4266 |
| rush_attempts | 251.0000 | 0.5976 | v2_baseline | 1.6231 |
| rush_tds | 251.0000 | 0.7131 | v2_baseline | 0.1360 |
| rush_yards | 251.0000 | 0.5936 | v2_baseline | 10.6886 |
| targets | 251.0000 | 0.1474 | v2_baseline | 2.1647 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 26.0000 | 7.9730 | 0.1051 |
| v2_total_minus3 | QB | 26.0000 | 7.6266 | 0.1041 |
| v2_baseline | RB | 79.0000 | 4.3597 | 0.1575 |
| v2_total_minus3 | RB | 79.0000 | 4.3074 | 0.1567 |
| v2_baseline | TE | 101.0000 | 1.9340 | 0.1030 |
| v2_total_minus3 | TE | 101.0000 | 1.8990 | 0.1017 |
| v2_baseline | WR | 154.0000 | 2.7570 | 0.1253 |
| v2_total_minus3 | WR | 154.0000 | 2.7495 | 0.1234 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_08_BUF_CAR | Buffalo Bills | Joshua Palmer | WR | 304.2900 | 59.5000 | 0.0000 | 5.2800 | 0.0000 | 0.0000 | 216.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_BUF_CAR | Buffalo Bills | James Cook III | RB | 202.7300 | 29.2000 | 0.0000 | 4.8900 | 0.0000 | 59.0000 | 216.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_BUF_CAR | Buffalo Bills | Josh Allen | QB | 166.0200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.9000 | 7.0000 | 285.2000 | 163.0000 |
| baseline | v2_baseline | 2025_08_GB_PIT | Green Bay Packers | Jordan Love | QB | 143.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2000 | 0.0000 | 230.0000 | 360.0000 |
| baseline | v2_baseline | 2025_08_NYJ_CIN | New York Jets | Breece Hall | RB | 124.1400 | 16.0000 | 14.0000 | 4.7900 | 3.0000 | 27.3000 | 133.0000 | 0.0000 | 4.0000 |
| baseline | v2_baseline | 2025_08_SF_HOU | Houston Texans | C.J. Stroud | QB | 122.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4000 | 5.0000 | 206.6000 | 318.0000 |
| baseline | v2_baseline | 2025_08_CLE_NE | Cleveland Browns | Quinshon Judkins | RB | 118.6500 | 18.8000 | -2.0000 | 3.4900 | 3.0000 | 100.3000 | 19.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_CHI_BAL | Chicago Bears | Caleb Williams | QB | 117.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 37.7000 | 0.0000 | 221.0000 | 285.0000 |
| baseline | v2_baseline | 2025_08_NYG_PHI | Philadelphia Eagles | Saquon Barkley | RB | 116.4500 | 12.5000 | 24.0000 | 3.8100 | 5.0000 | 48.4000 | 150.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_TEN_IND | Indianapolis Colts | Jonathan Taylor | RB | 113.0700 | 20.1000 | 21.0000 | 4.9100 | 2.0000 | 46.3000 | 153.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_GB_PIT | Green Bay Packers | Tucker Kraft | TE | 110.3500 | 35.6000 | 143.0000 | 7.9400 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_MIN_LAC | Los Angeles Chargers | Kimani Vidal | RB | 109.4900 | 19.8000 | 10.0000 | 4.4100 | 2.0000 | 36.1000 | 117.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_NYJ_CIN | Cincinnati Bengals | Joe Flacco | QB | 109.4700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.9000 | 1.0000 | 315.6000 | 223.0000 |
| baseline | v2_baseline | 2025_08_NYG_PHI | New York Giants | Jaxson Dart | QB | 103.8700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8000 | 9.0000 | 278.0000 | 193.0000 |
| baseline | v2_baseline | 2025_08_SF_HOU | Houston Texans | Woody Marks | RB | 99.1800 | 8.3000 | 49.0000 | 2.4700 | 4.0000 | 14.8000 | 62.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_SF_HOU | San Francisco 49ers | Christian McCaffrey | RB | 96.3800 | 13.2000 | 43.0000 | 3.2700 | 6.0000 | 78.5000 | 25.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_CLE_NE | Cleveland Browns | Dylan Sampson | RB | 96.1500 | 10.7000 | 29.0000 | 2.1700 | 6.0000 | 60.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_DAL_DEN | Denver Broncos | J.K. Dobbins | RB | 95.9800 | 25.6000 | 10.0000 | 5.2800 | 2.0000 | 41.1000 | 111.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_TB_NO | Tampa Bay Buccaneers | Baker Mayfield | QB | 93.2900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.9000 | 0.0000 | 219.2000 | 152.0000 |
| baseline | v2_baseline | 2025_08_NYG_PHI | Philadelphia Eagles | Tank Bigsby | RB | 90.8400 | 9.1000 | 0.0000 | 2.3700 | 0.0000 | 26.8000 | 104.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_DAL_DEN | Dallas Cowboys | Dak Prescott | QB | 89.7900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.5000 | 0.0000 | 264.0000 | 188.0000 |
| baseline | v2_baseline | 2025_08_NYJ_CIN | New York Jets | Isaiah Davis | RB | 89.6800 | 11.0000 | 44.0000 | 2.9800 | 6.0000 | 16.5000 | 65.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_WAS_KC | Kansas City Chiefs | Patrick Mahomes | QB | 89.0700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.2000 | 0.0000 | 239.3000 | 299.0000 |
| baseline | v2_baseline | 2025_08_DAL_DEN | Denver Broncos | Bo Nix | QB | 86.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.2000 | 0.0000 | 295.7000 | 247.0000 |
| baseline | v2_baseline | 2025_08_TB_NO | New Orleans Saints | Spencer Rattler | QB | 84.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.5000 | 3.0000 | 202.1000 | 136.0000 |
| baseline | v2_baseline | 2025_08_WAS_KC | Washington Commanders | Jeremy McNichols | RB | 84.3100 | 13.8000 | 64.0000 | 2.9900 | 6.0000 | 26.7000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_WAS_KC | Washington Commanders | Deebo Samuel | WR | 84.2000 | 89.3000 | 11.0000 | 7.7700 | 6.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_SF_HOU | Houston Texans | Xavier Hutchinson | WR | 81.0000 | 0.0000 | 69.0000 | 0.0000 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_08_CLE_NE | Cleveland Browns | Dillon Gabriel | QB | 79.6700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.2000 | 0.0000 | 224.1000 | 156.0000 |
| baseline | v2_baseline | 2025_08_TEN_IND | Tennessee Titans | Chimere Dike | WR | 77.3100 | 25.7000 | 93.0000 | 4.8300 | 9.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

