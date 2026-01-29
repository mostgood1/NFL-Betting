# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 3956
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_spread_home_plus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_spread_home_plus3 | Spread home +3.0 | 172.0000 | 3.7535 | 0.1105 | 0.1337 |
| v2_total_minus3 | Total -3.0 | 172.0000 | 3.7745 | 0.1093 | 0.1337 |
| v2_precip_plus50 | Precip +50% | 172.0000 | 3.7876 | 0.1111 | 0.1337 |
| v2_cold_minus15 | Temp -15F | 172.0000 | 3.7892 | 0.1112 | 0.1337 |
| v2_inj_away_plus1 | Away starters out +1 | 172.0000 | 3.7917 | 0.1111 | 0.1337 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 172.0000 | 3.7922 | 0.1107 | 0.1337 |
| v2_rest_minus7 | Home rest -7 days (diff) | 172.0000 | 3.7931 | 0.1113 | 0.1337 |
| v2_wind_plus20 | Wind +20mph (open) | 172.0000 | 3.7936 | 0.1106 | 0.1337 |
| v2_inj_home_plus1 | Home starters out +1 | 172.0000 | 3.7946 | 0.1111 | 0.1337 |
| v2_inj_home_plus2 | Home starters out +2 | 172.0000 | 3.7964 | 0.1107 | 0.1337 |
| v2_elo_plus100 | Home Elo +100 (diff) | 172.0000 | 3.7978 | 0.1115 | 0.1337 |
| v2_blend_10_20 | Market blend m=0.10 t=0.20 | 172.0000 | 3.8032 | 0.1115 | 0.1337 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 112.0000 | 0.9018 | v2_baseline | 0.0660 |
| pass_attempts | 112.0000 | 0.8750 | v2_baseline | 0.9056 |
| pass_tds | 112.0000 | 0.8929 | v2_baseline | 0.1386 |
| pass_yards | 112.0000 | 0.8839 | v2_baseline | 9.0357 |
| rec_tds | 112.0000 | 0.2857 | v2_baseline | 0.3176 |
| rec_yards | 112.0000 | 0.2232 | v2_baseline | 19.6759 |
| receptions | 112.0000 | 0.1964 | v2_baseline | 1.6310 |
| rush_attempts | 112.0000 | 0.5982 | v2_baseline | 1.3407 |
| rush_tds | 112.0000 | 0.6875 | v2_baseline | 0.1273 |
| rush_yards | 112.0000 | 0.5982 | v2_baseline | 6.2795 |
| targets | 112.0000 | 0.1339 | v2_baseline | 2.4104 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 12.0000 | 9.7391 | 0.0513 |
| v2_spread_home_plus3 | QB | 12.0000 | 9.3331 | 0.0483 |
| v2_baseline | RB | 39.0000 | 3.7341 | 0.1237 |
| v2_spread_home_plus3 | RB | 39.0000 | 3.7515 | 0.1231 |
| v2_baseline | TE | 47.0000 | 2.0801 | 0.0643 |
| v2_spread_home_plus3 | TE | 47.0000 | 2.0984 | 0.0639 |
| v2_baseline | WR | 74.0000 | 3.2509 | 0.1451 |
| v2_spread_home_plus3 | WR | 74.0000 | 3.1989 | 0.1435 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_19_GB_CHI | Green Bay Packers | Jordan Love | QB | 202.2100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.6000 | 0.0000 | 155.7000 | 323.0000 |
| baseline | v2_baseline | 2025_19_GB_CHI | Chicago Bears | Caleb Williams | QB | 200.5900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.3000 | 1.0000 | 197.7000 | 361.0000 |
| baseline | v2_baseline | 2025_19_HOU_PIT | Houston Texans | Christian Kirk | WR | 133.7000 | 20.5000 | 144.0000 | 4.2300 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_HOU_PIT | Pittsburgh Steelers | Aaron Rodgers | QB | 121.0800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.4000 | 0.0000 | 256.7000 | 146.0000 |
| baseline | v2_baseline | 2025_19_GB_CHI | Chicago Bears | Colston Loveland | TE | 118.8800 | 31.8000 | 137.0000 | 6.3800 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Buffalo Bills | Josh Allen | QB | 115.0800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 42.1000 | 20.0000 | 196.1000 | 273.0000 |
| baseline | v2_baseline | 2025_19_GB_CHI | Green Bay Packers | Romeo Doubs | WR | 112.9100 | 22.2000 | 124.0000 | 4.9000 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LA_CAR | Carolina Panthers | Jalen Coker | WR | 106.9600 | 37.7000 | 134.0000 | 7.3800 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LAC_NE | New England Patriots | Drake Maye | QB | 104.9700 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 28.5000 | 9.0000 | 197.8000 | 268.0000 |
| baseline | v2_baseline | 2025_19_SF_PHI | San Francisco 49ers | Demarcus Robinson | WR | 98.9100 | 18.2000 | 111.0000 | 4.3600 | 7.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_SF_PHI | Philadelphia Eagles | Saquon Barkley | RB | 97.1400 | 24.2000 | 25.0000 | 5.8400 | 7.0000 | 29.5000 | 106.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LA_CAR | Carolina Panthers | Bryce Young | QB | 95.9700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.0000 | 0.0000 | 192.4000 | 264.0000 |
| baseline | v2_baseline | 2025_19_GB_CHI | Green Bay Packers | Matthew Golden | WR | 93.0000 | 0.0000 | 84.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Jacksonville Jaguars | Parker Washington | WR | 89.2400 | 32.2000 | 107.0000 | 5.2900 | 12.0000 | 0.0000 | 2.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LAC_NE | Los Angeles Chargers | Omarion Hampton | RB | 86.0100 | 28.3000 | 0.0000 | 6.1100 | 0.0000 | 35.5000 | -1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LA_CAR | Los Angeles Rams | Puka Nacua | WR | 82.1800 | 57.8000 | 111.0000 | 9.4700 | 18.0000 | 0.0000 | 14.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LA_CAR | Los Angeles Rams | Matthew Stafford | QB | 81.9300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.1000 | 2.0000 | 245.2000 | 304.0000 |
| baseline | v2_baseline | 2025_19_HOU_PIT | Houston Texans | Woody Marks | RB | 81.1800 | 23.5000 | 0.0000 | 4.5900 | 1.0000 | 64.3000 | 112.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_SF_PHI | San Francisco 49ers | Brock Purdy | QB | 80.8900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.5000 | 14.0000 | 189.2000 | 262.0000 |
| baseline | v2_baseline | 2025_19_SF_PHI | San Francisco 49ers | Christian McCaffrey | RB | 80.7100 | 19.7000 | 66.0000 | 4.7800 | 8.0000 | 25.6000 | 48.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_LAC_NE | New England Patriots | Rhamondre Stevenson | RB | 79.8800 | 22.8000 | 75.0000 | 4.5300 | 4.0000 | 72.7000 | 53.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Buffalo Bills | James Cook III | RB | 77.7100 | 30.0000 | 5.0000 | 3.9100 | 3.0000 | 95.3000 | 46.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_HOU_PIT | Houston Texans | C.J. Stroud | QB | 77.1800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.3000 | 0.0000 | 199.0000 | 250.0000 |
| baseline | v2_baseline | 2025_19_LAC_NE | Los Angeles Chargers | Justin Herbert | QB | 66.1800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.8000 | 7.0000 | 211.8000 | 159.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Jacksonville Jaguars | Trevor Lawrence | QB | 63.2300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 3.0000 | 255.6000 | 207.0000 |
| baseline | v2_baseline | 2025_19_GB_CHI | Green Bay Packers | Emanuel Wilson | RB | 60.7200 | 10.7000 | 0.0000 | 2.3000 | 0.0000 | 38.2000 | 3.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_HOU_PIT | Houston Texans | Nick Chubb | RB | 58.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 48.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Jacksonville Jaguars | Brian Thomas Jr. | WR | 55.3900 | 63.9000 | 21.0000 | 10.5300 | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Jacksonville Jaguars | Bhayshul Tuten | RB | 53.7000 | 24.1000 | 0.0000 | 3.7200 | 0.0000 | 31.0000 | 51.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_19_BUF_JAX | Buffalo Bills | Khalil Shakir | WR | 53.1800 | 40.6000 | 82.0000 | 7.0800 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

