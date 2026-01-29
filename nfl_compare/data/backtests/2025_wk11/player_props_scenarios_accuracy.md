# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 9752
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 424.0000 | 3.4079 | 0.0998 | 0.1156 |
| v2_inj_home_plus2 | Home starters out +2 | 424.0000 | 3.4273 | 0.1013 | 0.1156 |
| v2_wind_plus20 | Wind +20mph (open) | 424.0000 | 3.4291 | 0.1011 | 0.1156 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 424.0000 | 3.4312 | 0.1014 | 0.1156 |
| v2_wind_plus10 | Wind +10mph (open) | 424.0000 | 3.4328 | 0.1015 | 0.1156 |
| v2_inj_away_plus2 | Away starters out +2 | 424.0000 | 3.4341 | 0.1013 | 0.1156 |
| v2_inj_away_plus1 | Away starters out +1 | 424.0000 | 3.4378 | 0.1015 | 0.1156 |
| v2_baseline | Baseline | 424.0000 | 3.4420 | 0.1021 | 0.1156 |
| v2_inj_home_plus1 | Home starters out +1 | 424.0000 | 3.4420 | 0.1016 | 0.1156 |
| v2_rest_minus7 | Home rest -7 days (diff) | 424.0000 | 3.4431 | 0.1019 | 0.1156 |
| v2_precip_plus25 | Precip +25% | 424.0000 | 3.4434 | 0.1019 | 0.1156 |
| v2_precip_plus50 | Precip +50% | 424.0000 | 3.4441 | 0.1018 | 0.1156 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 280.0000 | 0.9036 | v2_baseline | 0.0730 |
| pass_attempts | 280.0000 | 0.9071 | v2_baseline | 0.9302 |
| pass_tds | 280.0000 | 0.9107 | v2_baseline | 0.1100 |
| pass_yards | 280.0000 | 0.9179 | v2_baseline | 7.5401 |
| rec_tds | 280.0000 | 0.3429 | v2_baseline | 0.2522 |
| rec_yards | 280.0000 | 0.2393 | v2_baseline | 16.8824 |
| receptions | 280.0000 | 0.2250 | v2_baseline | 1.3558 |
| rush_attempts | 280.0000 | 0.5821 | v2_baseline | 1.7567 |
| rush_tds | 280.0000 | 0.7107 | v2_baseline | 0.1401 |
| rush_yards | 280.0000 | 0.6071 | v2_baseline | 6.9595 |
| targets | 280.0000 | 0.1714 | v2_baseline | 1.8618 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 30.0000 | 9.8305 | 0.0838 |
| v2_total_minus3 | QB | 30.0000 | 9.6689 | 0.0832 |
| v2_baseline | RB | 96.0000 | 3.8129 | 0.1693 |
| v2_total_minus3 | RB | 96.0000 | 3.7703 | 0.1654 |
| v2_baseline | TE | 116.0000 | 1.8626 | 0.0649 |
| v2_total_minus3 | TE | 116.0000 | 1.8574 | 0.0628 |
| v2_baseline | WR | 182.0000 | 2.5949 | 0.0933 |
| v2_total_minus3 | WR | 182.0000 | 2.5797 | 0.0915 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_11_CAR_ATL | Carolina Panthers | Bryce Young | QB | 311.1100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.9000 | 0.0000 | 176.2000 | 448.0000 |
| baseline | v2_baseline | 2025_11_SF_ARI | Arizona Cardinals | Jacoby Brissett | QB | 291.8300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6000 | -2.0000 | 193.1000 | 452.0000 |
| baseline | v2_baseline | 2025_11_BAL_CLE | Cleveland Browns | Dillon Gabriel | QB | 240.5300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8000 | 1.0000 | 276.9000 | 68.0000 |
| baseline | v2_baseline | 2025_11_CAR_ATL | Atlanta Falcons | Kirk Cousins | QB | 192.4700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.4000 | 2.0000 | 209.7000 | 48.0000 |
| baseline | v2_baseline | 2025_11_LAC_JAX | Los Angeles Chargers | Justin Herbert | QB | 167.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.9000 | 0.0000 | 211.9000 | 81.0000 |
| baseline | v2_baseline | 2025_11_SF_ARI | Arizona Cardinals | Michael Wilson | WR | 159.1700 | 45.9000 | 185.0000 | 8.3400 | 18.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_SEA_LA | Seattle Seahawks | Sam Darnold | QB | 152.6100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 27.4000 | 3.0000 | 177.2000 | 279.0000 |
| baseline | v2_baseline | 2025_11_DET_PHI | Detroit Lions | Jahmyr Gibbs | RB | 145.7500 | 21.2000 | 107.0000 | 4.1800 | 8.0000 | 85.6000 | 39.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_KC_DEN | Denver Broncos | Bo Nix | QB | 142.2900 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 23.0000 | 1.0000 | 187.7000 | 295.0000 |
| baseline | v2_baseline | 2025_11_SEA_LA | Los Angeles Rams | Matthew Stafford | QB | 139.2200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.3000 | 0.0000 | 243.0000 | 130.0000 |
| baseline | v2_baseline | 2025_11_TB_BUF | Buffalo Bills | Josh Allen | QB | 138.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.0000 | 20.0000 | 187.3000 | 317.0000 |
| baseline | v2_baseline | 2025_11_LAC_JAX | Jacksonville Jaguars | Trevor Lawrence | QB | 123.3100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.6000 | 4.0000 | 252.3000 | 153.0000 |
| baseline | v2_baseline | 2025_11_TB_BUF | Tampa Bay Buccaneers | Sean Tucker | RB | 122.2400 | 12.5000 | 34.0000 | 3.0400 | 2.0000 | 19.5000 | 106.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CIN_PIT | Pittsburgh Steelers | Aaron Rodgers | QB | 116.4800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.6000 | 0.0000 | 206.4000 | 116.0000 |
| baseline | v2_baseline | 2025_11_DAL_LV | Dallas Cowboys | George Pickens | WR | 103.7100 | 47.3000 | 144.0000 | 8.4500 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_TB_BUF | Buffalo Bills | Tyrell Shavers | WR | 99.0000 | 0.0000 | 90.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CHI_MIN | Chicago Bears | Caleb Williams | QB | 98.8400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.3000 | 3.0000 | 265.1000 | 193.0000 |
| baseline | v2_baseline | 2025_11_TB_BUF | Buffalo Bills | Joshua Palmer | WR | 95.8700 | 34.8000 | 66.0000 | 4.3100 | 4.0000 | 0.0000 | 48.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CAR_ATL | Carolina Panthers | Tetairoa McMillan | WR | 94.8800 | 42.1000 | 130.0000 | 8.4400 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_SF_ARI | Arizona Cardinals | Trey McBride | TE | 90.8200 | 31.9000 | 115.0000 | 8.6600 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_SEA_LA | Seattle Seahawks | Jaxon Smith-Njigba | WR | 88.5900 | 38.6000 | 105.0000 | 7.0600 | 13.0000 | 0.0000 | 11.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CHI_MIN | Chicago Bears | D'Andre Swift | RB | 85.4900 | 26.0000 | 0.0000 | 5.6000 | 1.0000 | 48.7000 | 90.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_DAL_LV | Dallas Cowboys | Javonte Williams | RB | 83.3800 | 26.2000 | 0.0000 | 5.6200 | 1.0000 | 52.8000 | 93.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CIN_PIT | Cincinnati Bengals | Joe Flacco | QB | 82.9700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.4000 | 0.0000 | 273.1000 | 199.0000 |
| baseline | v2_baseline | 2025_11_LAC_JAX | Los Angeles Chargers | Kimani Vidal | RB | 82.3400 | 23.8000 | -1.0000 | 4.5800 | 2.0000 | 57.0000 | 13.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CIN_PIT | Pittsburgh Steelers | Kenneth Gainwell | RB | 81.7700 | 12.2000 | 81.0000 | 3.3000 | 10.0000 | 24.0000 | 24.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CAR_ATL | Atlanta Falcons | Drake London | WR | 81.6500 | 41.8000 | 119.0000 | 8.5700 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_WAS_MIA | Miami Dolphins | De'Von Achane | RB | 81.0500 | 13.9000 | 45.0000 | 3.5500 | 5.0000 | 75.0000 | 120.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_11_CHI_MIN | Minnesota Vikings | J.J. McCarthy | QB | 79.2300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7000 | 0.0000 | 215.9000 | 150.0000 |
| baseline | v2_baseline | 2025_11_SF_ARI | San Francisco 49ers | Christian McCaffrey | RB | 78.9500 | 16.6000 | 40.0000 | 4.0700 | 6.0000 | 33.4000 | 81.0000 | 0.0000 | 0.0000 |

