# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 1081
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 47.0000 | 4.8262 | 0.1245 | 0.1915 |
| v2_inj_home_plus2 | Home starters out +2 | 47.0000 | 4.8431 | 0.1247 | 0.1915 |
| v2_blend_10_20 | Market blend m=0.10 t=0.20 | 47.0000 | 4.8456 | 0.1247 | 0.1915 |
| v2_sigma_low | Low volatility | 47.0000 | 4.8493 | 0.1248 | 0.1915 |
| v2_inj_away_plus2 | Away starters out +2 | 47.0000 | 4.8502 | 0.1247 | 0.1915 |
| v2_inj_home_plus1 | Home starters out +1 | 47.0000 | 4.8506 | 0.1250 | 0.1915 |
| v2_wind_plus20 | Wind +20mph (open) | 47.0000 | 4.8547 | 0.1249 | 0.1915 |
| v2_rest_plus7 | Home rest +7 days (diff) | 47.0000 | 4.8559 | 0.1249 | 0.1915 |
| v2_precip_plus50 | Precip +50% | 47.0000 | 4.8596 | 0.1249 | 0.1915 |
| v2_elo_minus100 | Home Elo -100 (diff) | 47.0000 | 4.8617 | 0.1251 | 0.1915 |
| v2_spread_home_plus3 | Spread home +3.0 | 47.0000 | 4.8619 | 0.1253 | 0.1915 |
| v2_elo_plus100 | Home Elo +100 (diff) | 47.0000 | 4.8625 | 0.1250 | 0.1915 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 29.0000 | 0.8621 | v2_baseline | 0.0910 |
| pass_attempts | 29.0000 | 0.8966 | v2_baseline | 0.8083 |
| pass_tds | 29.0000 | 0.8621 | v2_baseline | 0.1593 |
| pass_yards | 29.0000 | 0.8621 | v2_baseline | 17.0483 |
| rec_tds | 29.0000 | 0.2069 | v2_baseline | 0.3166 |
| rec_yards | 29.0000 | 0.1724 | v2_baseline | 21.5138 |
| receptions | 29.0000 | 0.2414 | v2_baseline | 1.3579 |
| rush_attempts | 29.0000 | 0.5172 | v2_baseline | 2.2152 |
| rush_tds | 29.0000 | 0.6207 | v2_baseline | 0.1362 |
| rush_yards | 29.0000 | 0.5172 | v2_baseline | 8.1069 |
| targets | 29.0000 | 0.1724 | v2_baseline | 1.9614 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 4.0000 | 13.5398 | 0.1761 |
| v2_total_minus3 | QB | 4.0000 | 13.4131 | 0.1788 |
| v2_baseline | RB | 8.0000 | 3.6697 | 0.1687 |
| v2_total_minus3 | RB | 8.0000 | 3.6222 | 0.1600 |
| v2_baseline | TE | 14.0000 | 2.0047 | 0.0437 |
| v2_total_minus3 | TE | 14.0000 | 1.9045 | 0.0393 |
| v2_baseline | WR | 21.0000 | 3.9801 | 0.1534 |
| v2_total_minus3 | WR | 21.0000 | 3.9560 | 0.1574 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | Sam Darnold | QB | 199.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.2000 | -5.0000 | 188.3000 | 346.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Matthew Stafford | QB | 194.4600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.4000 | 0.0000 | 202.6000 | 374.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Puka Nacua | WR | 148.2200 | 31.9000 | 165.0000 | 8.1600 | 14.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | Jaxon Smith-Njigba | WR | 127.4400 | 35.3000 | 153.0000 | 8.3500 | 13.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Drake Maye | QB | 110.8800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 22.5000 | 11.0000 | 181.7000 | 86.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Jarrett Stidham | QB | 82.2900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.6000 | 8.0000 | 202.6000 | 133.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Davante Adams | WR | 68.2600 | 22.1000 | 89.0000 | 6.1400 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Rhamondre Stevenson | RB | 58.0100 | 17.4000 | 0.0000 | 4.9900 | 2.0000 | 46.0000 | 71.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Kyle Williams | WR | 51.2800 | 23.0000 | 22.0000 | 4.2600 | 3.0000 | 0.0000 | 39.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | Kenneth Walker III | RB | 50.6300 | 19.3000 | 49.0000 | 5.1700 | 4.0000 | 47.2000 | 62.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | TreVeyon Henderson | RB | 44.4900 | 12.3000 | 0.0000 | 3.1100 | 0.0000 | 26.4000 | 5.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Blake Corum | RB | 38.2100 | 13.7000 | 24.0000 | 3.2500 | 3.0000 | 28.6000 | 55.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | Rashid Shaheed | WR | 37.3700 | 17.1000 | 51.0000 | 4.1900 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Jaleel McLaughlin | RB | 37.0000 | 12.0000 | 2.0000 | 3.2300 | 2.0000 | 29.9000 | 11.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Colby Parkinson | TE | 36.4000 | 29.2000 | 62.0000 | 7.4300 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Kyren Williams | RB | 25.8900 | 19.6000 | 22.0000 | 5.2300 | 3.0000 | 51.4000 | 39.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Konata Mumpfield | WR | 25.4500 | 21.5000 | 0.0000 | 4.1000 | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | AJ Barner | TE | 24.7800 | 30.7000 | 13.0000 | 7.3600 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Pat Bryant | WR | 24.6100 | 20.3000 | 2.0000 | 4.1000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Hunter Henry | TE | 23.9100 | 29.3000 | 12.0000 | 7.3700 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | RJ Harvey | RB | 23.4800 | 20.9000 | 22.0000 | 5.1900 | 6.0000 | 53.5000 | 37.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Kayshon Boutte | WR | 22.5800 | 26.0000 | 6.0000 | 6.3900 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Courtland Sutton | WR | 20.8500 | 33.7000 | 17.0000 | 8.1600 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | Stefon Diggs | WR | 18.8400 | 33.3000 | 17.0000 | 8.4800 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Evan Engram | TE | 16.9900 | 29.0000 | 19.0000 | 7.4900 | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Seattle Seahawks | Cooper Kupp | WR | 12.0500 | 24.8000 | 36.0000 | 6.2900 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Tyler Higbee | TE | 6.3200 | 14.3000 | 12.0000 | 3.8400 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | Denver Broncos | Elijah Moore | WR | 6.0000 | 0.0000 | 4.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_New_England_Patriots_Denver_Broncos | New England Patriots | DeMario Douglas | WR | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_21_Los_Angeles_Rams_Seattle_Seahawks | Los Angeles Rams | Davis Allen | TE |  | 0.0000 |  | 0.0000 |  | 0.0000 |  | 0.0000 |  |

