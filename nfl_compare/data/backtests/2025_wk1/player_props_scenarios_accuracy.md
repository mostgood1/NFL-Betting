# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 12857
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_minus3 | Spread home -3.0 | 559.0000 | 3.2619 | 0.0930 | 0.1127 |
| v2_total_minus3 | Total -3.0 | 559.0000 | 3.2780 | 0.0923 | 0.1127 |
| v2_baseline | Baseline | 559.0000 | 3.2798 | 0.0937 | 0.1127 |
| v2_precip_plus50 | Precip +50% | 559.0000 | 3.2920 | 0.0934 | 0.1127 |
| v2_blend_10_20 | Market blend m=0.10 t=0.20 | 559.0000 | 3.2930 | 0.0937 | 0.1127 |
| v2_wind_plus20 | Wind +20mph (open) | 559.0000 | 3.2955 | 0.0932 | 0.1127 |
| v2_sigma_low | Low volatility | 559.0000 | 3.2957 | 0.0937 | 0.1127 |
| v2_sigma_high | High volatility | 559.0000 | 3.2972 | 0.0934 | 0.1127 |
| v2_rest_minus7 | Home rest -7 days (diff) | 559.0000 | 3.2981 | 0.0934 | 0.1127 |
| v2_cold_minus15 | Temp -15F | 559.0000 | 3.2995 | 0.0935 | 0.1127 |
| v2_inj_away_plus2 | Away starters out +2 | 559.0000 | 3.3006 | 0.0932 | 0.1127 |
| v2_elo_minus100 | Home Elo -100 (diff) | 559.0000 | 3.3019 | 0.0938 | 0.1127 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 334.0000 | 0.9042 | v2_baseline | 0.0682 |
| pass_attempts | 334.0000 | 0.9102 | v2_baseline | 0.9953 |
| pass_tds | 334.0000 | 0.9102 | v2_baseline | 0.0947 |
| pass_yards | 334.0000 | 0.9132 | v2_baseline | 5.9207 |
| rec_tds | 334.0000 | 0.2575 | v2_baseline | 0.2473 |
| rec_yards | 334.0000 | 0.1976 | v2_baseline | 16.3420 |
| receptions | 334.0000 | 0.1766 | v2_baseline | 1.4201 |
| rush_attempts | 334.0000 | 0.5719 | v2_baseline | 1.4462 |
| rush_tds | 334.0000 | 0.6347 | v2_baseline | 0.1127 |
| rush_yards | 334.0000 | 0.5749 | v2_baseline | 7.4183 |
| targets | 334.0000 | 0.1856 | v2_baseline | 2.0122 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | FB | 3.0000 | 1.3939 | 0.0000 |
| v2_spread_home_minus3 | FB | 3.0000 | 1.3939 | 0.0000 |
| v2_baseline | QB | 32.0000 | 7.7881 | 0.1364 |
| v2_spread_home_minus3 | QB | 32.0000 | 7.6601 | 0.1357 |
| v2_baseline | RB | 112.0000 | 3.7707 | 0.1185 |
| v2_spread_home_minus3 | RB | 112.0000 | 3.7355 | 0.1178 |
| v2_baseline | TE | 149.0000 | 2.2034 | 0.0606 |
| v2_spread_home_minus3 | TE | 149.0000 | 2.2007 | 0.0605 |
| v2_baseline | WR | 263.0000 | 2.6338 | 0.0978 |
| v2_spread_home_minus3 | WR | 263.0000 | 2.6398 | 0.0967 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_01_CIN_CLE | Cleveland Browns | J.Flacco | RB | 347.2300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.8000 | 2.0000 | 0.0000 | 290.0000 |
| baseline | v2_baseline | 2025_01_PIT_NYJ | Pittsburgh Steelers | Roman Wilson | WR | 269.2800 | 35.9000 | 0.0000 | 5.2300 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 168.0000 |
| baseline | v2_baseline | 2025_01_BAL_BUF | Buffalo Bills | Josh Allen | QB | 230.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 29.4000 | 8.0000 | 211.3000 | 394.0000 |
| baseline | v2_baseline | 2025_01_LV_NE | Las Vegas Raiders | Geno Smith | QB | 176.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.7000 | 0.0000 | 214.5000 | 362.0000 |
| baseline | v2_baseline | 2025_01_CAR_JAX | Jacksonville Jaguars | T.Bigsby | RB | 165.9500 | 0.0000 | 13.0000 | 0.0000 | 3.0000 | 10.4000 | 143.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_CAR_JAX | Jacksonville Jaguars | Travis Etienne | RB | 150.7400 | 6.2000 | 13.0000 | 1.4600 | 3.0000 | 15.1000 | 143.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_KC_LAC | Los Angeles Chargers | Justin Herbert | QB | 147.6800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 23.7000 | 0.0000 | 209.8000 | 318.0000 |
| baseline | v2_baseline | 2025_01_LV_NE | New England Patriots | Drake Maye | QB | 146.5200 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | 23.6000 | 4.0000 | 191.7000 | 287.0000 |
| baseline | v2_baseline | 2025_01_BAL_BUF | Baltimore Ravens | Derrick Henry | RB | 140.2000 | 14.8000 | 13.0000 | 3.4700 | 1.0000 | 40.5000 | 169.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_CIN_CLE | Cincinnati Bengals | Joe Burrow | QB | 134.6400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 220.9000 | 113.0000 |
| baseline | v2_baseline | 2025_01_TB_ATL | Atlanta Falcons | Bijan Robinson | RB | 132.3800 | 13.1000 | 100.0000 | 2.8200 | 8.0000 | 56.5000 | 24.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_MIA_IND | Miami Dolphins | Tua Tagovailoa | QB | 124.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.6000 | 0.0000 | 220.1000 | 114.0000 |
| baseline | v2_baseline | 2025_01_TB_ATL | Atlanta Falcons | Michael Penix Jr. | QB | 123.8800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.7000 | 0.0000 | 204.4000 | 298.0000 |
| baseline | v2_baseline | 2025_01_SF_SEA | San Francisco 49ers | Christian McCaffrey | RB | 111.9200 | 13.7000 | 73.0000 | 3.2500 | 10.0000 | 40.6000 | 69.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_CAR_JAX | Jacksonville Jaguars | Travis Etienne Jr. | RB | 102.9100 | 19.7000 | 13.0000 | 3.2100 | 3.0000 | 53.2000 | 143.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_BAL_BUF | Baltimore Ravens | Zay Flowers | WR | 101.9300 | 53.0000 | 143.0000 | 10.7100 | 9.0000 | 0.0000 | 8.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_TEN_DEN | Tennessee Titans | Cam Ward | QB | 101.2700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.7000 | 0.0000 | 204.4000 | 112.0000 |
| baseline | v2_baseline | 2025_01_MIN_CHI | Minnesota Vikings | Aaron Jones Sr. | RB | 99.1800 | 19.8000 | 0.0000 | 3.1000 | 3.0000 | 62.7000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_MIN_CHI | Minnesota Vikings | Aaron Jones Sr. | RB | 99.1800 | 19.8000 | 0.0000 | 3.1000 | 3.0000 | 62.7000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_MIA_IND | Indianapolis Colts | Daniel Jones | QB | 98.2300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.1000 | 6.0000 | 205.0000 | 272.0000 |
| baseline | v2_baseline | 2025_01_SF_SEA | San Francisco 49ers | Brock Purdy | QB | 95.2800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.9000 | 0.0000 | 217.8000 | 277.0000 |
| baseline | v2_baseline | 2025_01_PIT_NYJ | New York Jets | Breece Hall | RB | 94.7900 | 14.8000 | 38.0000 | 3.4400 | 4.0000 | 42.8000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_LV_NE | Las Vegas Raiders | Brock Bowers | TE | 94.4800 | 15.8000 | 103.0000 | 3.3400 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_DAL_PHI | Philadelphia Eagles | Jalen Hurts | QB | 88.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.8000 | 2.0000 | 201.7000 | 152.0000 |
| baseline | v2_baseline | 2025_01_HOU_LA | Los Angeles Rams | Puka Nacua | WR | 85.5700 | 49.8000 | 130.0000 | 10.4900 | 11.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_KC_LAC | Kansas City Chiefs | Patrick Mahomes | QB | 83.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 22.8000 | 0.0000 | 215.9000 | 258.0000 |
| baseline | v2_baseline | 2025_01_NYG_WAS | Washington Commanders | Deebo Samuel Sr. | WR | 83.3100 | 23.8000 | 77.0000 | 4.8700 | 10.0000 | 0.0000 | 19.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_MIA_IND | Indianapolis Colts | A.Mitchell | WR | 82.5100 | 9.8000 | 80.0000 | 1.0500 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_CIN_CLE | Cleveland Browns | Dylan Sampson | RB | 81.5600 | 13.6000 | 64.0000 | 2.5900 | 8.0000 | 16.8000 | 29.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_01_CAR_JAX | Carolina Panthers | Bryce Young | QB | 81.4300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.6000 | 0.0000 | 201.1000 | 154.0000 |

