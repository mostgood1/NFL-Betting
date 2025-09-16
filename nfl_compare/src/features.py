import pandas as pd
import numpy as np

# ELO and rolling features

K = 20.0
HOME_ADV = 55.0


def initial_elo():
    return 1500.0


def expected_score(elo_a, elo_b, home_adv=0.0):
    return 1.0 / (1.0 + 10 ** (-(elo_a + home_adv - elo_b) / 400.0))


def update_elo(elo_a, elo_b, score_a, score_b, k=K, home=False):
    exp_a = expected_score(elo_a, elo_b, HOME_ADV if home else 0.0)
    result_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
    margin = abs(score_a - score_b)
    margin_mult = np.log(max(margin, 1) + 1.0)
    delta = k * margin_mult * (result_a - exp_a)
    return elo_a + delta


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    teams = pd.unique(pd.concat([games['home_team'], games['away_team']], ignore_index=True))
    elos = {t: initial_elo() for t in teams}
    rows = []
    games_sorted = games.sort_values(['season', 'week'])
    for _, g in games_sorted.iterrows():
        home, away = g['home_team'], g['away_team']
        hs, as_ = g['home_score'], g['away_score']
        elo_home, elo_away = elos.get(home, initial_elo()), elos.get(away, initial_elo())
        rows.append({
            'game_id': g['game_id'],
            'elo_home_pre': elo_home,
            'elo_away_pre': elo_away,
        })
        # update after game
        new_home = update_elo(elo_home, elo_away, hs, as_, home=True)
        new_away = update_elo(elo_away, elo_home, as_, hs, home=False)
        elos[home] = new_home
        elos[away] = new_away
    return pd.DataFrame(rows)


def merge_features(games: pd.DataFrame, team_stats: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    elo = compute_elo(games)
    df = games.merge(elo, on='game_id', how='left')

    # join team stats for both teams by season/week/team
    ts_home = team_stats.rename(columns={
        'team': 'home_team',
        'off_epa': 'home_off_epa', 'def_epa': 'home_def_epa', 'pace_secs_play': 'home_pace_secs_play',
        'pass_rate': 'home_pass_rate', 'rush_rate': 'home_rush_rate', 'qb_adj': 'home_qb_adj', 'sos': 'home_sos'
    })
    ts_away = team_stats.rename(columns={
        'team': 'away_team',
        'off_epa': 'away_off_epa', 'def_epa': 'away_def_epa', 'pace_secs_play': 'away_pace_secs_play',
        'pass_rate': 'away_pass_rate', 'rush_rate': 'away_rush_rate', 'qb_adj': 'away_qb_adj', 'sos': 'away_sos'
    })

    df = df.merge(ts_home, on=['season', 'week', 'home_team'], how='left')
    df = df.merge(ts_away, on=['season', 'week', 'away_team'], how='left')

    # lines
    df = df.merge(lines[['game_id', 'spread_home', 'total', 'close_spread_home', 'close_total']], on='game_id', how='left')

    # target engineering
    df['home_margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']

    # simple differentials
    df['elo_diff'] = df['elo_home_pre'] - df['elo_away_pre']
    for col in ['off_epa', 'def_epa', 'pace_secs_play', 'pass_rate', 'rush_rate', 'qb_adj', 'sos']:
        h, a = f'home_{col}', f'away_{col}'
        if h in df.columns and a in df.columns:
            df[f'{col}_diff'] = df[h].fillna(0) - df[a].fillna(0)

    # quarter/half aggregates if present
    if {'home_q1','home_q2','home_q3','home_q4'}.issubset(df.columns):
        df['home_first_half'] = df[['home_q1','home_q2']].sum(axis=1)
        df['home_second_half'] = df[['home_q3','home_q4']].sum(axis=1)
    if {'away_q1','away_q2','away_q3','away_q4'}.issubset(df.columns):
        df['away_first_half'] = df[['away_q1','away_q2']].sum(axis=1)
        df['away_second_half'] = df[['away_q3','away_q4']].sum(axis=1)

    return df
