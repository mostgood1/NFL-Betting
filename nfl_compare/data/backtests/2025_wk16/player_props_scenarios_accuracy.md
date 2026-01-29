# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 10235
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 445.0000 | 3.6650 | 0.1210 | 0.1618 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 445.0000 | 3.6885 | 0.1213 | 0.1618 |
| v2_inj_away_plus2 | Away starters out +2 | 445.0000 | 3.6905 | 0.1215 | 0.1618 |
| v2_wind_plus20 | Wind +20mph (open) | 445.0000 | 3.6944 | 0.1214 | 0.1618 |
| v2_inj_home_plus2 | Home starters out +2 | 445.0000 | 3.6951 | 0.1215 | 0.1618 |
| v2_inj_home_plus1 | Home starters out +1 | 445.0000 | 3.6959 | 0.1216 | 0.1618 |
| v2_wind_plus10 | Wind +10mph (open) | 445.0000 | 3.6994 | 0.1216 | 0.1618 |
| v2_spread_home_minus3 | Spread home -3.0 | 445.0000 | 3.7027 | 0.1225 | 0.1618 |
| v2_inj_away_plus1 | Away starters out +1 | 445.0000 | 3.7028 | 0.1216 | 0.1618 |
| v2_rest_plus7 | Home rest +7 days (diff) | 445.0000 | 3.7031 | 0.1218 | 0.1618 |
| v2_elo_minus100 | Home Elo -100 (diff) | 445.0000 | 3.7037 | 0.1218 | 0.1618 |
| v2_sigma_low | Low volatility | 445.0000 | 3.7042 | 0.1217 | 0.1618 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 312.0000 | 0.9038 | v2_baseline | 0.0630 |
| pass_attempts | 312.0000 | 0.9071 | v2_baseline | 0.9869 |
| pass_tds | 312.0000 | 0.9071 | v2_baseline | 0.1342 |
| pass_yards | 312.0000 | 0.9071 | v2_baseline | 7.9715 |
| rec_tds | 312.0000 | 0.3333 | v2_baseline | 0.2729 |
| rec_yards | 312.0000 | 0.2244 | v2_baseline | 17.5244 |
| receptions | 312.0000 | 0.2051 | v2_baseline | 1.5275 |
| rush_attempts | 312.0000 | 0.5769 | v2_baseline | 1.6407 |
| rush_tds | 312.0000 | 0.6923 | v2_baseline | 0.1290 |
| rush_yards | 312.0000 | 0.5801 | v2_baseline | 8.3446 |
| targets | 312.0000 | 0.1603 | v2_baseline | 2.1585 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 32.0000 | 9.7099 | 0.0991 |
| v2_total_minus3 | QB | 32.0000 | 9.5700 | 0.0978 |
| v2_baseline | RB | 99.0000 | 4.1182 | 0.1789 |
| v2_total_minus3 | RB | 99.0000 | 4.0873 | 0.1785 |
| v2_baseline | TE | 121.0000 | 2.2080 | 0.0886 |
| v2_total_minus3 | TE | 121.0000 | 2.1215 | 0.0878 |
| v2_baseline | WR | 193.0000 | 2.7759 | 0.1169 |
| v2_total_minus3 | WR | 193.0000 | 2.7799 | 0.1162 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_16_GB_CHI | Green Bay Packers | Jordan Love | QB | 262.6900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4000 | 0.0000 | 304.9000 | 77.0000 |
| baseline | v2_baseline | 2025_16_MIN_NYG | New York Giants | Jaxson Dart | QB | 234.5200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.4000 | 3.0000 | 239.4000 | 33.0000 |
| baseline | v2_baseline | 2025_16_PHI_WAS | Washington Commanders | Marcus Mariota | QB | 207.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 39.2000 | 0.0000 | 237.3000 | 95.0000 |
| baseline | v2_baseline | 2025_16_LA_SEA | Los Angeles Rams | Matthew Stafford | QB | 205.6300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 0.0000 | 282.1000 | 457.0000 |
| baseline | v2_baseline | 2025_16_BUF_CLE | Buffalo Bills | Joshua Palmer | WR | 188.2300 | 65.9000 | 17.0000 | 5.8200 | 2.0000 | 0.0000 | 117.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_LA_SEA | Los Angeles Rams | Puka Nacua | WR | 185.5600 | 52.5000 | 225.0000 | 9.0300 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_TB_CAR | Tampa Bay Buccaneers | Baker Mayfield | QB | 181.9100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.8000 | 1.0000 | 299.2000 | 145.0000 |
| baseline | v2_baseline | 2025_16_NE_BAL | New England Patriots | Drake Maye | QB | 167.4200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.7000 | 19.0000 | 240.1000 | 380.0000 |
| baseline | v2_baseline | 2025_16_BUF_CLE | Buffalo Bills | Josh Allen | QB | 166.4600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 35.4000 | 7.0000 | 250.0000 | 130.0000 |
| baseline | v2_baseline | 2025_16_LV_HOU | Las Vegas Raiders | Ashton Jeanty | RB | 155.9200 | 19.0000 | 60.0000 | 4.9800 | 2.0000 | 32.0000 | 128.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_KC_TEN | Kansas City Chiefs | Chris Oladokun | QB | 148.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.2000 | 0.0000 | 234.5000 | 111.0000 |
| baseline | v2_baseline | 2025_16_JAX_DEN | Denver Broncos | Bo Nix | QB | 133.2200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30.0000 | 7.0000 | 260.5000 | 352.0000 |
| baseline | v2_baseline | 2025_16_MIN_NYG | Minnesota Vikings | J.J. McCarthy | QB | 132.0900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.5000 | 0.0000 | 224.4000 | 108.0000 |
| baseline | v2_baseline | 2025_16_BUF_CLE | Buffalo Bills | James Cook III | RB | 127.2600 | 31.2000 | 2.0000 | 4.3300 | 1.0000 | 77.9000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_JAX_DEN | Jacksonville Jaguars | Parker Washington | WR | 125.1600 | 29.0000 | 145.0000 | 4.9500 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_ATL_ARI | Atlanta Falcons | Bijan Robinson | RB | 120.5200 | 20.8000 | 92.0000 | 5.5600 | 11.0000 | 38.9000 | 76.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_LAC_DAL | Los Angeles Chargers | Justin Herbert | QB | 118.0900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.7000 | 2.0000 | 205.4000 | 300.0000 |
| baseline | v2_baseline | 2025_16_NE_BAL | Baltimore Ravens | Derrick Henry | RB | 115.1800 | 11.7000 | 0.0000 | 3.2200 | 0.0000 | 35.8000 | 128.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_NYJ_NO | New Orleans Saints | Chris Olave | WR | 114.5900 | 44.1000 | 148.0000 | 9.6800 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_PIT_DET | Detroit Lions | Jahmyr Gibbs | RB | 113.1000 | 26.0000 | 66.0000 | 5.3500 | 13.0000 | 55.0000 | 2.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_NE_BAL | Baltimore Ravens | Lamar Jackson | QB | 112.1400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.0000 | 7.0000 | 171.8000 | 101.0000 |
| baseline | v2_baseline | 2025_16_LA_SEA | Seattle Seahawks | Kenneth Walker III | RB | 110.8200 | 19.8000 | 64.0000 | 5.4500 | 3.0000 | 38.0000 | 100.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_SF_IND | Indianapolis Colts | Philip Rivers | QB | 109.4300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.1000 | 0.0000 | 178.5000 | 277.0000 |
| baseline | v2_baseline | 2025_16_LAC_DAL | Dallas Cowboys | George Pickens | WR | 100.5400 | 35.4000 | 130.0000 | 6.5700 | 9.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_PHI_WAS | Philadelphia Eagles | Jalen Hurts | QB | 97.5400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 46.3000 | 0.0000 | 214.5000 | 185.0000 |
| baseline | v2_baseline | 2025_16_NE_BAL | New England Patriots | TreVeyon Henderson | RB | 93.8400 | 26.9000 | 9.0000 | 4.5100 | 1.0000 | 64.2000 | 3.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_PIT_DET | Detroit Lions | Jared Goff | QB | 92.8100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.7000 | 0.0000 | 306.9000 | 364.0000 |
| baseline | v2_baseline | 2025_16_TB_CAR | Carolina Panthers | Trevor Etienne | RB | 92.0000 | 0.0000 | 16.0000 | 0.0000 | 6.0000 | 0.0000 | 50.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_NE_BAL | New England Patriots | Stefon Diggs | WR | 91.1400 | 51.8000 | 138.0000 | 8.4800 | 10.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_16_GB_CHI | Chicago Bears | Caleb Williams | QB | 90.5600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.9000 | 0.0000 | 199.5000 | 250.0000 |

