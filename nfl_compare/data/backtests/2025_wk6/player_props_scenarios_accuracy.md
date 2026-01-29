# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 9361
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 407.0000 | 3.4348 | 0.1142 | 0.1351 |
| v2_inj_away_plus2 | Away starters out +2 | 407.0000 | 3.4923 | 0.1155 | 0.1351 |
| v2_wind_plus20 | Wind +20mph (open) | 407.0000 | 3.4924 | 0.1155 | 0.1351 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 407.0000 | 3.4981 | 0.1156 | 0.1351 |
| v2_inj_home_plus2 | Home starters out +2 | 407.0000 | 3.5008 | 0.1158 | 0.1351 |
| v2_wind_plus10 | Wind +10mph (open) | 407.0000 | 3.5111 | 0.1157 | 0.1351 |
| v2_inj_home_plus1 | Home starters out +1 | 407.0000 | 3.5154 | 0.1160 | 0.1351 |
| v2_inj_away_plus1 | Away starters out +1 | 407.0000 | 3.5172 | 0.1160 | 0.1351 |
| v2_precip_plus50 | Precip +50% | 407.0000 | 3.5187 | 0.1161 | 0.1351 |
| v2_sigma_high | High volatility | 407.0000 | 3.5197 | 0.1161 | 0.1351 |
| v2_elo_minus100 | Home Elo -100 (diff) | 407.0000 | 3.5244 | 0.1161 | 0.1351 |
| v2_elo_plus100 | Home Elo +100 (diff) | 407.0000 | 3.5251 | 0.1161 | 0.1351 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 269.0000 | 0.8959 | v2_baseline | 0.0755 |
| pass_attempts | 269.0000 | 0.8922 | v2_baseline | 0.9160 |
| pass_tds | 269.0000 | 0.8996 | v2_baseline | 0.0939 |
| pass_yards | 269.0000 | 0.8959 | v2_baseline | 5.5836 |
| rec_tds | 269.0000 | 0.2491 | v2_baseline | 0.2978 |
| rec_yards | 269.0000 | 0.2156 | v2_baseline | 18.2625 |
| receptions | 269.0000 | 0.1896 | v2_baseline | 1.5043 |
| rush_attempts | 269.0000 | 0.5613 | v2_baseline | 1.5923 |
| rush_tds | 269.0000 | 0.6803 | v2_baseline | 0.1132 |
| rush_yards | 269.0000 | 0.5651 | v2_baseline | 8.1029 |
| targets | 269.0000 | 0.1747 | v2_baseline | 2.2760 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 30.0000 | 7.0207 | 0.0947 |
| v2_total_minus3 | QB | 30.0000 | 6.5220 | 0.0942 |
| v2_baseline | RB | 89.0000 | 4.0851 | 0.1386 |
| v2_total_minus3 | RB | 89.0000 | 4.0271 | 0.1346 |
| v2_baseline | TE | 113.0000 | 1.7640 | 0.1071 |
| v2_total_minus3 | TE | 113.0000 | 1.7534 | 0.1060 |
| v2_baseline | WR | 175.0000 | 3.1926 | 0.1146 |
| v2_total_minus3 | WR | 175.0000 | 3.1385 | 0.1126 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_06_DEN_NYJ | New York Jets | Justin Fields | QB | 249.7300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.8000 | 14.0000 | 272.0000 | 45.0000 |
| baseline | v2_baseline | 2025_06_DAL_CAR | Carolina Panthers | Rico Dowdle | RB | 164.3700 | 23.3000 | 56.0000 | 4.5900 | 5.0000 | 66.0000 | 183.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_BUF_ATL | Buffalo Bills | Joshua Palmer | WR | 155.1400 | 42.6000 | 0.0000 | 5.1500 | 0.0000 | 0.0000 | 87.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_SF_TB | San Francisco 49ers | Mac Jones | QB | 144.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.2000 | 5.0000 | 219.1000 | 347.0000 |
| baseline | v2_baseline | 2025_06_BUF_ATL | Atlanta Falcons | Bijan Robinson | RB | 142.2900 | 17.0000 | 68.0000 | 3.9100 | 8.0000 | 88.6000 | 170.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_BUF_ATL | Atlanta Falcons | Drake London | WR | 124.8200 | 47.2000 | 158.0000 | 7.3700 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_DAL_CAR | Dallas Cowboys | George Pickens | WR | 122.7400 | 52.0000 | 168.0000 | 9.0400 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_CHI_WAS | Chicago Bears | D'Andre Swift | RB | 114.1900 | 22.0000 | 67.0000 | 5.1400 | 3.0000 | 44.4000 | 108.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_LA_BAL | Baltimore Ravens | Cooper Rush | QB | 113.2300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.3000 | 0.0000 | 162.2000 | 72.0000 |
| baseline | v2_baseline | 2025_06_SF_TB | San Francisco 49ers | Kendrick Bourne | WR | 109.5900 | 34.1000 | 142.0000 | 7.4000 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_CHI_WAS | Washington Commanders | Deebo Samuel | WR | 106.5300 | 114.8000 | 15.0000 | 8.9100 | 6.0000 | 0.0000 | -1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_CHI_WAS | Washington Commanders | Jayden Daniels | QB | 106.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 35.9000 | -1.0000 | 273.0000 | 211.0000 |
| baseline | v2_baseline | 2025_06_DAL_CAR | Carolina Panthers | Bryce Young | QB | 101.0600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.4000 | 0.0000 | 263.1000 | 199.0000 |
| baseline | v2_baseline | 2025_06_ARI_IND | Arizona Cardinals | Jacoby Brissett | QB | 100.6600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.2000 | 7.0000 | 242.8000 | 320.0000 |
| baseline | v2_baseline | 2025_06_SEA_JAX | Seattle Seahawks | Jaxon Smith-Njigba | WR | 99.8100 | 65.3000 | 162.0000 | 10.3800 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_BUF_ATL | Buffalo Bills | James Cook III | RB | 98.7100 | 22.6000 | 60.0000 | 4.8400 | 2.0000 | 45.5000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_LAC_MIA | Miami Dolphins | De'Von Achane | RB | 97.6100 | 18.2000 | 22.0000 | 5.3900 | 7.0000 | 39.4000 | 128.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_LAC_MIA | Los Angeles Chargers | Kimani Vidal | RB | 95.9600 | 22.0000 | 14.0000 | 5.1500 | 4.0000 | 42.1000 | 124.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_SF_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 92.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.5000 | 0.0000 | 310.6000 | 256.0000 |
| baseline | v2_baseline | 2025_06_ARI_IND | Indianapolis Colts | Jonathan Taylor | RB | 90.8400 | 19.3000 | 14.0000 | 4.2100 | 4.0000 | 49.0000 | 123.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_BUF_ATL | Buffalo Bills | Josh Allen | QB | 90.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.8000 | 0.0000 | 238.9000 | 180.0000 |
| baseline | v2_baseline | 2025_06_DET_KC | Kansas City Chiefs | Patrick Mahomes | QB | 90.2800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.7000 | 1.0000 | 321.4000 | 257.0000 |
| baseline | v2_baseline | 2025_06_LA_BAL | Baltimore Ravens | Derrick Henry | RB | 89.3500 | 12.4000 | 8.0000 | 3.7100 | 2.0000 | 48.6000 | 122.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_TEN_LV | Las Vegas Raiders | Geno Smith | QB | 87.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.4000 | 0.0000 | 233.8000 | 174.0000 |
| baseline | v2_baseline | 2025_06_DEN_NYJ | Denver Broncos | Bo Nix | QB | 83.9700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.8000 | 6.0000 | 231.6000 | 174.0000 |
| baseline | v2_baseline | 2025_06_ARI_IND | Indianapolis Colts | Michael Pittman Jr. | WR | 83.6500 | 95.3000 | 20.0000 | 8.1200 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_06_PHI_NYG | New York Giants | Jaxson Dart | QB | 81.9500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.2000 | 18.0000 | 253.7000 | 195.0000 |
| baseline | v2_baseline | 2025_06_LAC_MIA | Miami Dolphins | Tua Tagovailoa | QB | 81.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.2000 | 0.0000 | 270.2000 | 205.0000 |
| baseline | v2_baseline | 2025_06_ARI_IND | Indianapolis Colts | Daniel Jones | QB | 79.2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30.3000 | 2.0000 | 250.8000 | 212.0000 |
| baseline | v2_baseline | 2025_06_NE_NO | New England Patriots | DeMario Douglas | WR | 79.0000 | 0.0000 | 71.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

