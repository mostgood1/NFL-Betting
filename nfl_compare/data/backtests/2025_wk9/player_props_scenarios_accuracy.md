# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 9016
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 392.0000 | 3.7061 | 0.1194 | 0.1454 |
| v2_spread_home_minus3 | Spread home -3.0 | 392.0000 | 3.7136 | 0.1219 | 0.1454 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 392.0000 | 3.7187 | 0.1204 | 0.1454 |
| v2_wind_plus20 | Wind +20mph (open) | 392.0000 | 3.7199 | 0.1204 | 0.1454 |
| v2_wind_plus10 | Wind +10mph (open) | 392.0000 | 3.7270 | 0.1206 | 0.1454 |
| v2_inj_home_plus2 | Home starters out +2 | 392.0000 | 3.7285 | 0.1205 | 0.1454 |
| v2_blend_10_20 | Market blend m=0.10 t=0.20 | 392.0000 | 3.7291 | 0.1211 | 0.1454 |
| v2_inj_away_plus2 | Away starters out +2 | 392.0000 | 3.7296 | 0.1204 | 0.1454 |
| v2_precip_plus25 | Precip +25% | 392.0000 | 3.7299 | 0.1211 | 0.1454 |
| v2_baseline | Baseline | 392.0000 | 3.7299 | 0.1209 | 0.1454 |
| v2_precip_plus50 | Precip +50% | 392.0000 | 3.7315 | 0.1210 | 0.1454 |
| v2_sigma_high | High volatility | 392.0000 | 3.7315 | 0.1210 | 0.1454 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 259.0000 | 0.9035 | v2_baseline | 0.0540 |
| pass_attempts | 259.0000 | 0.9035 | v2_baseline | 0.7527 |
| pass_tds | 259.0000 | 0.9073 | v2_baseline | 0.1305 |
| pass_yards | 259.0000 | 0.9112 | v2_baseline | 6.5498 |
| rec_tds | 259.0000 | 0.3127 | v2_baseline | 0.2923 |
| rec_yards | 259.0000 | 0.1815 | v2_baseline | 18.8502 |
| receptions | 259.0000 | 0.1969 | v2_baseline | 1.4326 |
| rush_attempts | 259.0000 | 0.5753 | v2_baseline | 1.9189 |
| rush_tds | 259.0000 | 0.6911 | v2_baseline | 0.1321 |
| rush_yards | 259.0000 | 0.5792 | v2_baseline | 8.8483 |
| targets | 259.0000 | 0.1583 | v2_baseline | 2.0680 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 28.0000 | 8.8444 | 0.1383 |
| v2_total_minus3 | QB | 28.0000 | 8.8288 | 0.1391 |
| v2_baseline | RB | 87.0000 | 4.4264 | 0.1686 |
| v2_total_minus3 | RB | 87.0000 | 4.3618 | 0.1644 |
| v2_baseline | TE | 107.0000 | 2.5999 | 0.0978 |
| v2_total_minus3 | TE | 107.0000 | 2.5781 | 0.0981 |
| v2_baseline | WR | 170.0000 | 2.7646 | 0.1081 |
| v2_total_minus3 | WR | 170.0000 | 2.7621 | 0.1064 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_09_CHI_CIN | Cincinnati Bengals | Joe Flacco | QB | 230.4600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.8000 | 0.0000 | 263.2000 | 470.0000 |
| baseline | v2_baseline | 2025_09_DEN_HOU | Houston Texans | C.J. Stroud | QB | 216.9800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.2000 | 0.0000 | 248.8000 | 79.0000 |
| baseline | v2_baseline | 2025_09_CHI_CIN | Chicago Bears | Kyle Monangai | RB | 186.6200 | 14.5000 | 22.0000 | 3.2500 | 5.0000 | 20.5000 | 176.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_LAC_TEN | Tennessee Titans | Cam Ward | QB | 174.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 301.7000 | 145.0000 |
| baseline | v2_baseline | 2025_09_SEA_WAS | Seattle Seahawks | Sam Darnold | QB | 174.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.2000 | 0.0000 | 187.0000 | 330.0000 |
| baseline | v2_baseline | 2025_09_KC_BUF | Buffalo Bills | Joshua Palmer | WR | 173.3300 | 38.3000 | 11.0000 | 3.7100 | 1.0000 | 0.0000 | 114.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_SF_NYG | San Francisco 49ers | Christian McCaffrey | RB | 156.0900 | 14.6000 | 67.0000 | 4.1100 | 6.0000 | 26.0000 | 106.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_CAR_GB | Carolina Panthers | Trevor Etienne | RB | 147.0000 | 0.0000 | 31.0000 | 0.0000 | 5.0000 | 0.0000 | 84.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_JAX_LV | Las Vegas Raiders | Brock Bowers | TE | 125.1600 | 25.2000 | 127.0000 | 5.8500 | 14.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_CAR_GB | Carolina Panthers | Bryce Young | QB | 119.9100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.2000 | 0.0000 | 191.2000 | 102.0000 |
| baseline | v2_baseline | 2025_09_IND_PIT | Indianapolis Colts | Daniel Jones | QB | 119.5300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.1000 | 3.0000 | 272.2000 | 342.0000 |
| baseline | v2_baseline | 2025_09_JAX_LV | Las Vegas Raiders | Geno Smith | QB | 111.1500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3000 | 3.0000 | 192.7000 | 284.0000 |
| baseline | v2_baseline | 2025_09_CHI_CIN | Chicago Bears | Caleb Williams | QB | 111.0200 | 0.0000 | 22.0000 | 0.0000 | 2.0000 | 16.4000 | 0.0000 | 221.7000 | 280.0000 |
| baseline | v2_baseline | 2025_09_ATL_NE | New England Patriots | DeMario Douglas | WR | 110.0000 | 0.0000 | 100.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_ATL_NE | New England Patriots | Drake Maye | QB | 109.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.1000 | 0.0000 | 183.2000 | 259.0000 |
| baseline | v2_baseline | 2025_09_SEA_WAS | Seattle Seahawks | Jaxon Smith-Njigba | WR | 103.8400 | 41.2000 | 129.0000 | 8.8600 | 11.0000 | 0.0000 | 11.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_DEN_HOU | Denver Broncos | Bo Nix | QB | 101.3200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.3000 | 11.0000 | 240.9000 | 173.0000 |
| baseline | v2_baseline | 2025_09_MIN_DET | Detroit Lions | Jared Goff | QB | 100.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.0000 | 0.0000 | 208.8000 | 284.0000 |
| baseline | v2_baseline | 2025_09_KC_BUF | Buffalo Bills | Josh Allen | QB | 97.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 42.1000 | 14.0000 | 213.2000 | 273.0000 |
| baseline | v2_baseline | 2025_09_CHI_CIN | Cincinnati Bengals | Chase Brown | RB | 95.5000 | 22.8000 | 75.0000 | 4.8400 | 14.0000 | 61.8000 | 37.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_NO_LA | Los Angeles Rams | Kyren Williams | RB | 95.4200 | 26.9000 | 0.0000 | 5.0700 | 0.0000 | 65.1000 | 114.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_MIN_DET | Minnesota Vikings | J.J. McCarthy | QB | 95.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8000 | 8.0000 | 229.3000 | 143.0000 |
| baseline | v2_baseline | 2025_09_CAR_GB | Carolina Panthers | Rico Dowdle | RB | 92.3800 | 16.0000 | 11.0000 | 4.2600 | 3.0000 | 54.0000 | 130.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_IND_PIT | Pittsburgh Steelers | Aaron Rodgers | QB | 89.1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3000 | 0.0000 | 282.6000 | 203.0000 |
| baseline | v2_baseline | 2025_09_LAC_TEN | Los Angeles Chargers | Justin Herbert | QB | 88.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 22.0000 | 11.0000 | 183.9000 | 250.0000 |
| baseline | v2_baseline | 2025_09_CHI_CIN | Cincinnati Bengals | Tee Higgins | WR | 84.2600 | 41.5000 | 121.0000 | 7.6600 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_ARI_DAL | Arizona Cardinals | Emari Demercado | RB | 82.9000 | 14.0000 | -1.0000 | 3.3000 | 1.0000 | 22.2000 | 79.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_CHI_CIN | Chicago Bears | Colston Loveland | TE | 82.7400 | 36.4000 | 118.0000 | 7.9700 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_JAX_LV | Jacksonville Jaguars | Parker Washington | WR | 81.9400 | 19.9000 | 90.0000 | 4.4600 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_09_DEN_HOU | Denver Broncos | RJ Harvey | RB | 81.0400 | 14.1000 | 51.0000 | 2.6900 | 5.0000 | 37.5000 | 5.0000 | 0.0000 | 0.0000 |

