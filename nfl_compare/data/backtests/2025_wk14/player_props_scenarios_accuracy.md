# Player Props Scenario Accuracy — v2_baseline

- scenarios: 23
- joined rows: 8947
- restrict_active: False
- join_keys: game_id, team, player_id
- best_scenario_id_by_mae_mean: v2_total_minus3

## Scenario Summary

| scenario_id | scenario_label | n_rows | mae_mean | brier_any_td | any_td_rate |
| --- | --- | --- | --- | --- | --- |
| v2_total_minus3 | Total -3.0 | 389.0000 | 3.4931 | 0.1103 | 0.1388 |
| v2_pressure_plus005 | Home/away pressure +0.05 | 389.0000 | 3.5159 | 0.1115 | 0.1388 |
| v2_wind_plus20 | Wind +20mph (open) | 389.0000 | 3.5175 | 0.1114 | 0.1388 |
| v2_inj_home_plus2 | Home starters out +2 | 389.0000 | 3.5224 | 0.1116 | 0.1388 |
| v2_wind_plus10 | Wind +10mph (open) | 389.0000 | 3.5257 | 0.1116 | 0.1388 |
| v2_inj_home_plus1 | Home starters out +1 | 389.0000 | 3.5281 | 0.1118 | 0.1388 |
| v2_inj_away_plus2 | Away starters out +2 | 389.0000 | 3.5289 | 0.1116 | 0.1388 |
| v2_inj_away_plus1 | Away starters out +1 | 389.0000 | 3.5303 | 0.1117 | 0.1388 |
| v2_baseline | Baseline | 389.0000 | 3.5359 | 0.1119 | 0.1388 |
| v2_rest_minus7 | Home rest -7 days (diff) | 389.0000 | 3.5368 | 0.1120 | 0.1388 |
| v2_precip_plus50 | Precip +50% | 389.0000 | 3.5384 | 0.1120 | 0.1388 |
| v2_precip_plus25 | Precip +25% | 389.0000 | 3.5391 | 0.1118 | 0.1388 |

## Scenario Envelope Coverage (min..max across scenarios)

| stat | n_players | coverage_min_max | baseline_scenario_id | baseline_mae |
| --- | --- | --- | --- | --- |
| interceptions | 280.0000 | 0.9000 | v2_baseline | 0.0657 |
| pass_attempts | 280.0000 | 0.9071 | v2_baseline | 0.8304 |
| pass_tds | 280.0000 | 0.9036 | v2_baseline | 0.1176 |
| pass_yards | 280.0000 | 0.9143 | v2_baseline | 7.7649 |
| rec_tds | 280.0000 | 0.3500 | v2_baseline | 0.2667 |
| rec_yards | 280.0000 | 0.2107 | v2_baseline | 16.9461 |
| receptions | 280.0000 | 0.2179 | v2_baseline | 1.3662 |
| rush_attempts | 280.0000 | 0.6107 | v2_baseline | 1.6754 |
| rush_tds | 280.0000 | 0.7036 | v2_baseline | 0.1245 |
| rush_yards | 280.0000 | 0.6143 | v2_baseline | 7.7837 |
| targets | 280.0000 | 0.1536 | v2_baseline | 1.9539 |

## By Position (baseline + best)

| scenario_id | position | n_rows | mae_mean | brier_any_td |
| --- | --- | --- | --- | --- |
| v2_baseline | QB | 28.0000 | 9.7621 | 0.0576 |
| v2_total_minus3 | QB | 28.0000 | 9.6623 | 0.0554 |
| v2_baseline | RB | 88.0000 | 3.9949 | 0.1773 |
| v2_total_minus3 | RB | 88.0000 | 3.9428 | 0.1764 |
| v2_baseline | TE | 105.0000 | 1.9460 | 0.0901 |
| v2_total_minus3 | TE | 105.0000 | 1.8941 | 0.0890 |
| v2_baseline | WR | 168.0000 | 2.6767 | 0.1002 |
| v2_total_minus3 | WR | 168.0000 | 2.6590 | 0.0981 |

## Top Error Contributors

| pick | scenario_id | game_id | team | player | position | abs_error_sum | rec_yards | rec_yards_act | targets | targets_act | rush_yards | rush_yards_act | pass_yards | pass_yards_act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | v2_baseline | 2025_14_MIA_NYJ | New York Jets | Tyrod Taylor | QB | 270.3100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3000 | 0.0000 | 234.2000 | 6.0000 |
| baseline | v2_baseline | 2025_14_WAS_MIN | Washington Commanders | Jayden Daniels | QB | 228.7400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 29.3000 | 0.0000 | 258.2000 | 78.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Cleveland Browns | Shedeur Sanders | QB | 216.8900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8000 | 0.0000 | 166.7000 | 364.0000 |
| baseline | v2_baseline | 2025_14_DAL_DET | Dallas Cowboys | Dak Prescott | QB | 160.6300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2000 | 1.0000 | 242.3000 | 376.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Tennessee Titans | Tony Pollard | RB | 158.8600 | 15.6000 | 0.0000 | 4.8100 | 0.0000 | 38.5000 | 161.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_NO_TB | Tampa Bay Buccaneers | Baker Mayfield | QB | 154.7200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.5000 | 4.0000 | 252.6000 | 122.0000 |
| baseline | v2_baseline | 2025_14_HOU_KC | Kansas City Chiefs | Patrick Mahomes | QB | 150.3100 | 0.0000 | -10.0000 | 0.0000 | 1.0000 | 14.8000 | 0.0000 | 279.1000 | 160.0000 |
| baseline | v2_baseline | 2025_14_PIT_BAL | Pittsburgh Steelers | Aaron Rodgers | QB | 141.8100 | 0.0000 | -9.0000 | 0.0000 | 1.0000 | 5.9000 | 0.0000 | 166.1000 | 284.0000 |
| baseline | v2_baseline | 2025_14_DAL_DET | Detroit Lions | Jared Goff | QB | 139.7800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.9000 | 0.0000 | 192.4000 | 309.0000 |
| baseline | v2_baseline | 2025_14_NO_TB | New Orleans Saints | Tyler Shough | QB | 139.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.5000 | 45.0000 | 225.6000 | 144.0000 |
| baseline | v2_baseline | 2025_14_CIN_BUF | Buffalo Bills | Joshua Palmer | WR | 126.8800 | 55.0000 | 31.0000 | 4.9400 | 2.0000 | 0.0000 | 80.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_PIT_BAL | Pittsburgh Steelers | DK Metcalf | WR | 125.2200 | 30.6000 | 148.0000 | 7.3000 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_LA_ARI | Los Angeles Rams | Puka Nacua | WR | 125.1900 | 46.6000 | 167.0000 | 8.0300 | 11.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_PHI_LAC | Philadelphia Eagles | Saquon Barkley | RB | 120.4700 | 21.8000 | 0.0000 | 4.9500 | 2.0000 | 41.0000 | 122.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_LA_ARI | Arizona Cardinals | Michael Wilson | WR | 118.2500 | 38.9000 | 142.0000 | 7.7200 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_SEA_ATL | Seattle Seahawks | Sam Darnold | QB | 116.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21.9000 | 3.0000 | 164.0000 | 249.0000 |
| baseline | v2_baseline | 2025_14_CIN_BUF | Buffalo Bills | Josh Allen | QB | 111.8800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 42.1000 | -3.0000 | 204.4000 | 251.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Cleveland Browns | Dylan Sampson | RB | 111.6000 | 8.5000 | 64.0000 | 2.4300 | 6.0000 | 42.8000 | 4.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_LA_ARI | Los Angeles Rams | Blake Corum | RB | 108.8000 | 13.2000 | 3.0000 | 2.5100 | 1.0000 | 35.2000 | 128.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_LA_ARI | Arizona Cardinals | Jacoby Brissett | QB | 105.3700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3000 | 5.0000 | 184.6000 | 271.0000 |
| baseline | v2_baseline | 2025_14_DAL_DET | Dallas Cowboys | Ryan Flournoy | WR | 101.6400 | 27.9000 | 115.0000 | 5.2800 | 14.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Cleveland Browns | Quinshon Judkins | RB | 99.0400 | 15.0000 | 58.0000 | 3.9000 | 3.0000 | 71.6000 | 26.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_DAL_DET | Detroit Lions | Jahmyr Gibbs | RB | 98.2800 | 14.0000 | 77.0000 | 4.0500 | 7.0000 | 64.0000 | 43.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Tennessee Titans | Cam Ward | QB | 96.4700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.3000 | 4.0000 | 201.5000 | 117.0000 |
| baseline | v2_baseline | 2025_14_TEN_CLE | Cleveland Browns | Harold Fannin Jr. | TE | 93.8700 | 29.5000 | 114.0000 | 6.5800 | 12.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_PIT_BAL | Baltimore Ravens | Derrick Henry | RB | 93.8200 | 17.1000 | 8.0000 | 5.0500 | 2.0000 | 29.8000 | 94.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_MIA_NYJ | Miami Dolphins | Jaylen Wright | RB | 93.0600 | 10.2000 | 0.0000 | 2.4500 | 3.0000 | 39.4000 | 107.0000 | 0.0000 | 0.0000 |
| baseline | v2_baseline | 2025_14_PHI_LAC | Los Angeles Chargers | Justin Herbert | QB | 93.0200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.8000 | 21.0000 | 201.7000 | 139.0000 |
| baseline | v2_baseline | 2025_14_IND_JAX | Indianapolis Colts | Riley Leonard | QB | 91.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0000 | 230.2000 | 145.0000 |
| baseline | v2_baseline | 2025_14_IND_JAX | Jacksonville Jaguars | Tim Patrick | WR | 89.0000 | 0.0000 | 78.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

