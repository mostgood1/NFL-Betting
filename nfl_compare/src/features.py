import os
import pandas as pd
import numpy as np
from typing import Optional
from .weather import WeatherCols, DATA_DIR
from .priors import compute_team_priors

# ELO and rolling features

K = 20.0
HOME_ADV = 55.0


def initial_elo():
    return 1500.0


def expected_score(elo_a, elo_b, home_adv=0.0):
    return 1.0 / (1.0 + 10 ** (-(elo_a + home_adv - elo_b) / 400.0))


def update_elo(elo_a, elo_b, score_a, score_b, k=K, home=False, neutral=False):
    # No home advantage for neutral-site games
    adv = 0.0 if neutral else (HOME_ADV if home else 0.0)
    exp_a = expected_score(elo_a, elo_b, adv)
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
        neutral = False
        # Treat neutral-site if explicitly provided in games or weather (joined later in pipeline)
        for key in ['neutral_site','is_neutral']:
            if key in g and pd.notna(g[key]):
                val = str(g[key]).strip().lower()
                neutral = neutral or (val not in {'', '0', 'false', 'none', 'nan'})
        elo_home, elo_away = elos.get(home, initial_elo()), elos.get(away, initial_elo())
        rows.append({
            'game_id': g['game_id'],
            'elo_home_pre': elo_home,
            'elo_away_pre': elo_away,
        })
        # update after game
        new_home = update_elo(elo_home, elo_away, hs, as_, home=True, neutral=neutral)
        new_away = update_elo(elo_away, elo_home, as_, hs, home=False, neutral=neutral)
        elos[home] = new_home
        elos[away] = new_away
    return pd.DataFrame(rows)


def merge_features(games: pd.DataFrame, team_stats: pd.DataFrame, lines: pd.DataFrame, weather: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    elo = compute_elo(games)
    df = games.merge(elo, on='game_id', how='left')

    # join team stats for both teams by season/week/team
    ts_home = team_stats.rename(columns={
        'team': 'home_team',
        'off_epa': 'home_off_epa', 'def_epa': 'home_def_epa', 'pace_secs_play': 'home_pace_secs_play',
        'off_epa_1h': 'home_off_epa_1h', 'off_epa_2h': 'home_off_epa_2h',
        'def_epa_1h': 'home_def_epa_1h', 'def_epa_2h': 'home_def_epa_2h',
        'pass_rate': 'home_pass_rate', 'rush_rate': 'home_rush_rate', 'qb_adj': 'home_qb_adj', 'sos': 'home_sos'
    })
    ts_away = team_stats.rename(columns={
        'team': 'away_team',
        'off_epa': 'away_off_epa', 'def_epa': 'away_def_epa', 'pace_secs_play': 'away_pace_secs_play',
        'off_epa_1h': 'away_off_epa_1h', 'off_epa_2h': 'away_off_epa_2h',
        'def_epa_1h': 'away_def_epa_1h', 'def_epa_2h': 'away_def_epa_2h',
        'pass_rate': 'away_pass_rate', 'rush_rate': 'away_rush_rate', 'qb_adj': 'away_qb_adj', 'sos': 'away_sos'
    })

    df = df.merge(ts_home, on=['season', 'week', 'home_team'], how='left')
    df = df.merge(ts_away, on=['season', 'week', 'away_team'], how='left')

    # lines: prefer game_id join; if many NaNs, fall back to team-based join for current games
    line_cols = ['spread_home','total','moneyline_home','moneyline_away','close_spread_home','close_total',
                 'spread_home_price','spread_away_price','total_over_price','total_under_price']
    cols_present = [c for c in line_cols if c in lines.columns]

    if 'game_id' in lines.columns:
        df = df.merge(lines[['game_id'] + cols_present], on='game_id', how='left')
    else:
        df = df.copy()
    # If spread/total missing for many rows, try team-based left join as supplement
    if df['spread_home'].isna().mean() > 0.5 or df['total'].isna().mean() > 0.5:
        if {'home_team','away_team'}.issubset(lines.columns):
            suppl = lines[['home_team','away_team'] + cols_present].copy()
            df = df.merge(suppl, on=['home_team','away_team'], how='left', suffixes=('', '_sup'))
            for c in ['spread_home','total','moneyline_home','moneyline_away']:
                sup = f'{c}_sup'
                if sup in df.columns:
                    df[c] = df[c].where(df[c].notna(), df[sup])
            # drop helper columns
            drop_cols = [c for c in df.columns if c.endswith('_sup')]
            df = df.drop(columns=drop_cols)

    # Optional: merge weather/stadium data
    if weather is not None and not weather.empty:
        try:
            df = df.merge(
                weather[[
                    'game_id','date','home_team','away_team',
                    WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.roof, WeatherCols.surface, 'neutral_site'
                ]],
                on=['game_id','date','home_team','away_team'], how='left'
            )
        except Exception:
            pass
    # Apply per-game location overrides to set neutral_site if provided
    try:
        ovr_fp = DATA_DIR / 'game_location_overrides.csv'
        if ovr_fp.exists():
            expected = ['game_id','date','home_team','away_team','venue','city','country','tz','lat','lon','roof','surface','neutral_site','note']
            ovr = pd.read_csv(ovr_fp, comment='#', header=None, names=expected)
            # Prefer game_id join
            if 'game_id' in ovr.columns:
                cols = ['game_id','neutral_site']
                df = df.merge(ovr[cols], on='game_id', how='left', suffixes=('', '_ovr'))
                if 'neutral_site_ovr' in df.columns:
                    df['neutral_site'] = df['neutral_site_ovr'].where(df['neutral_site_ovr'].notna(), df.get('neutral_site'))
                    df = df.drop(columns=['neutral_site_ovr'])
            # Fallback: date + teams
            if df.get('neutral_site').isna().any() and {'date','home_team','away_team'}.issubset(ovr.columns):
                tmp = ovr[['date','home_team','away_team','neutral_site']].copy()
                df = df.merge(tmp, on=['date','home_team','away_team'], how='left', suffixes=('', '_ovr2'))
                if 'neutral_site_ovr2' in df.columns:
                    df['neutral_site'] = df['neutral_site_ovr2'].where(df['neutral_site_ovr2'].notna(), df.get('neutral_site'))
                    df = df.drop(columns=['neutral_site_ovr2'])
    except Exception:
        pass
    # Ensure weather columns exist
    for c in [WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.roof, WeatherCols.surface]:
        if c not in df.columns:
            df[c] = np.nan
    # Derive simple categorical flags
    def _is_dome(val):
        if pd.isna(val):
            return 0.0
        s = str(val).strip().lower()
        return 1.0 if s in {'dome','indoor','closed','retractable-closed'} else 0.0
    def _is_turf(val):
        if pd.isna(val):
            return 0.0
        s = str(val).strip().lower()
        return 1.0 if 'turf' in s else 0.0
    df['is_dome'] = df[WeatherCols.roof].apply(_is_dome)
    df['is_turf'] = df[WeatherCols.surface].apply(_is_turf)

    # target engineering
    df['home_margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    # Neutral-site indicator (1.0 if truthy)
    def _is_neutral(val):
        if pd.isna(val):
            return 0.0
        s = str(val).strip().lower()
        return 0.0 if s in {'', '0', 'false', 'none', 'nan'} else 1.0
    if 'neutral_site' in df.columns:
        df['is_neutral'] = df['neutral_site'].apply(_is_neutral)
    else:
        df['is_neutral'] = 0.0

    # Compute priors for current season from previous season stats and blend by week
    # weight w = min(week/6, 1) for current-year; (1-w) for prior
    if not team_stats.empty and {'season','team'}.issubset(team_stats.columns):
        # Build priors per season present in schedule
        seasons = sorted(df['season'].dropna().unique().tolist())
        priors_list = [compute_team_priors(team_stats, int(s)) for s in seasons if pd.notna(s)]
        priors_df = pd.concat([p for p in priors_list if p is not None and not p.empty], ignore_index=True) if priors_list else pd.DataFrame()
    else:
        priors_df = pd.DataFrame()

    if not priors_df.empty:
        hp = priors_df.rename(columns={
            'team': 'home_team',
            'off_epa_prior': 'home_off_epa_prior',
            'def_epa_prior': 'home_def_epa_prior',
            'pace_prior': 'home_pace_prior',
            'pass_rate_prior': 'home_pass_rate_prior',
            'rush_rate_prior': 'home_rush_rate_prior',
            'sos_prior': 'home_sos_prior',
            'qb_prior': 'home_qb_prior',
            'continuity_flag': 'home_continuity',
        })
        ap = priors_df.rename(columns={
            'team': 'away_team',
            'off_epa_prior': 'away_off_epa_prior',
            'def_epa_prior': 'away_def_epa_prior',
            'pace_prior': 'away_pace_prior',
            'pass_rate_prior': 'away_pass_rate_prior',
            'rush_rate_prior': 'away_rush_rate_prior',
            'sos_prior': 'away_sos_prior',
            'qb_prior': 'away_qb_prior',
            'continuity_flag': 'away_continuity',
        })
        df = df.merge(hp[['season','home_team','home_off_epa_prior','home_def_epa_prior','home_pace_prior','home_pass_rate_prior','home_rush_rate_prior','home_sos_prior','home_qb_prior','home_continuity']], on=['season','home_team'], how='left')
        df = df.merge(ap[['season','away_team','away_off_epa_prior','away_def_epa_prior','away_pace_prior','away_pass_rate_prior','away_rush_rate_prior','away_sos_prior','away_qb_prior','away_continuity']], on=['season','away_team'], how='left')

        # blending weight by week
        w = (df['week'].astype(float) / 6.0).clip(lower=0.0, upper=1.0)
        def _blend(curr: pd.Series, prior: pd.Series) -> pd.Series:
            return w * curr.fillna(0) + (1 - w) * prior.fillna(prior.mean())

        # create blended columns for key stats
        df['home_off_epa_blend'] = _blend(df.get('home_off_epa'), df.get('home_off_epa_prior')) if 'home_off_epa_prior' in df.columns else df.get('home_off_epa')
        df['away_off_epa_blend'] = _blend(df.get('away_off_epa'), df.get('away_off_epa_prior')) if 'away_off_epa_prior' in df.columns else df.get('away_off_epa')
        df['home_def_epa_blend'] = _blend(df.get('home_def_epa'), df.get('home_def_epa_prior')) if 'home_def_epa_prior' in df.columns else df.get('home_def_epa')
        df['away_def_epa_blend'] = _blend(df.get('away_def_epa'), df.get('away_def_epa_prior')) if 'away_def_epa_prior' in df.columns else df.get('away_def_epa')
        df['home_pace_blend'] = _blend(df.get('home_pace_secs_play'), df.get('home_pace_prior')) if 'home_pace_prior' in df.columns else df.get('home_pace_secs_play')
        df['away_pace_blend'] = _blend(df.get('away_pace_secs_play'), df.get('away_pace_prior')) if 'away_pace_prior' in df.columns else df.get('away_pace_secs_play')
        df['home_pass_rate_blend'] = _blend(df.get('home_pass_rate'), df.get('home_pass_rate_prior')) if 'home_pass_rate_prior' in df.columns else df.get('home_pass_rate')
        df['away_pass_rate_blend'] = _blend(df.get('away_pass_rate'), df.get('away_pass_rate_prior')) if 'away_pass_rate_prior' in df.columns else df.get('away_pass_rate')
        df['home_rush_rate_blend'] = _blend(df.get('home_rush_rate'), df.get('home_rush_rate_prior')) if 'home_rush_rate_prior' in df.columns else df.get('home_rush_rate')
        df['away_rush_rate_blend'] = _blend(df.get('away_rush_rate'), df.get('away_rush_rate_prior')) if 'away_rush_rate_prior' in df.columns else df.get('away_rush_rate')
        df['home_sos_blend'] = _blend(df.get('home_sos'), df.get('home_sos_prior')) if 'home_sos_prior' in df.columns else df.get('home_sos')
        df['away_sos_blend'] = _blend(df.get('away_sos'), df.get('away_sos_prior')) if 'away_sos_prior' in df.columns else df.get('away_sos')
        # qb prior and continuity do not have a current-year analog here; carry through
        if 'home_qb_prior' in df.columns and 'away_qb_prior' in df.columns:
            df['qb_prior_diff'] = df['home_qb_prior'].fillna(df['home_qb_prior'].mean()) - df['away_qb_prior'].fillna(df['away_qb_prior'].mean())
        else:
            df['qb_prior_diff'] = 0.0
        if 'home_continuity' in df.columns and 'away_continuity' in df.columns:
            df['continuity_diff'] = df['home_continuity'].fillna(1.0) - df['away_continuity'].fillna(1.0)
        else:
            df['continuity_diff'] = 0.0
    else:
        # Fallback: use currents directly
        df['home_off_epa_blend'] = df.get('home_off_epa')
        df['away_off_epa_blend'] = df.get('away_off_epa')
        df['home_def_epa_blend'] = df.get('home_def_epa')
        df['away_def_epa_blend'] = df.get('away_def_epa')
        df['home_pace_blend'] = df.get('home_pace_secs_play')
        df['away_pace_blend'] = df.get('away_pace_secs_play')
        df['home_pass_rate_blend'] = df.get('home_pass_rate')
        df['away_pass_rate_blend'] = df.get('away_pass_rate')
        df['home_rush_rate_blend'] = df.get('home_rush_rate')
        df['away_rush_rate_blend'] = df.get('away_rush_rate')
        df['home_sos_blend'] = df.get('home_sos')
        df['away_sos_blend'] = df.get('away_sos')
    df['qb_prior_diff'] = 0.0
    df['continuity_diff'] = 0.0

    # simple differentials (use Elo + blended stats)
    # Remove home-field advantage in market anchor by nudging market_home_prob toward 0.5 if neutral
    df['elo_diff'] = df['elo_home_pre'] - df['elo_away_pre']
    diff_map = [
        ('off_epa', 'off_epa_blend'),
        ('def_epa', 'def_epa_blend'),
    ('off_epa_1h', 'off_epa_1h'),
    ('off_epa_2h', 'off_epa_2h'),
    ('def_epa_1h', 'def_epa_1h'),
    ('def_epa_2h', 'def_epa_2h'),
        ('pace_secs_play', 'pace_blend'),
        ('pass_rate', 'pass_rate_blend'),
        ('rush_rate', 'rush_rate_blend'),
        ('qb_adj', 'qb_adj'),
        ('sos', 'sos_blend'),
    ]
    for base, col in diff_map:
        h, a = f'home_{col}', f'away_{col}'
        if h in df.columns and a in df.columns:
            df[f'{base}_diff'] = df[h].fillna(0) - df[a].fillna(0)

    # Market anchor: implied home win probability from moneylines, vig-adjusted
    def _american_to_prob(ml: float) -> Optional[float]:
        try:
            v = float(ml)
        except Exception:
            return None
        if v < 0:
            return (-v) / ((-v) + 100.0)
        else:
            return 100.0 / (v + 100.0)

    p_home = df['moneyline_home'].apply(_american_to_prob) if 'moneyline_home' in df.columns else pd.Series([None]*len(df))
    p_away = df['moneyline_away'].apply(_american_to_prob) if 'moneyline_away' in df.columns else pd.Series([None]*len(df))
    sump = (pd.Series(p_home).fillna(0) + pd.Series(p_away).fillna(0)).replace(0, np.nan)
    market_home_prob = pd.Series(p_home) / sump
    df['market_home_prob'] = market_home_prob.fillna(0.5)
    # Optional neutral adjustment on market-based anchor (very light)
    if 'is_neutral' in df.columns:
        df['market_home_prob'] = 0.5 * df['is_neutral'] + (1 - df['is_neutral']) * df['market_home_prob']

    # quarter/half aggregates if present
    if {'home_q1','home_q2','home_q3','home_q4'}.issubset(df.columns):
        df['home_first_half'] = df[['home_q1','home_q2']].sum(axis=1)
        df['home_second_half'] = df[['home_q3','home_q4']].sum(axis=1)
    if {'away_q1','away_q2','away_q3','away_q4'}.issubset(df.columns):
        df['away_first_half'] = df[['away_q1','away_q2']].sum(axis=1)
        df['away_second_half'] = df[['away_q3','away_q4']].sum(axis=1)

    # Rolling half/quarter performance features to shape 1H vs 2H predictability
    # Rolling window configurable via env (default 6)
    try:
        _WIN = int(os.getenv('NFL_HALF_ROLLING_WINDOW', '6'))
    except Exception:
        _WIN = 6

    def _compute_half_rolling(g: pd.DataFrame, window: int = _WIN) -> pd.DataFrame:
        g = g.copy()
        # ensure chronological order
        g = g.sort_values(['season','week','game_id'])
        # Build long team-game view
        rows = []
        for _, r in g.iterrows():
            if not set(['home_q1','home_q2','home_q3','home_q4','away_q1','away_q2','away_q3','away_q4']).issubset(r.index):
                continue
            if pd.isna(r['home_q1']) or pd.isna(r['away_q1']):
                continue
            home_1h = float(r['home_q1'] + r['home_q2'])
            home_2h = float(r['home_q3'] + r['home_q4'])
            away_1h = float(r['away_q1'] + r['away_q2'])
            away_2h = float(r['away_q3'] + r['away_q4'])
            rows.append({
                'game_id': r['game_id'], 'season': r['season'], 'week': r['week'],
                'team': r['home_team'], 'opp': r['away_team'], 'is_home': 1,
                'scored_1h': home_1h, 'scored_2h': home_2h,
                'allowed_1h': away_1h, 'allowed_2h': away_2h,
                'scored_q1': float(r['home_q1']), 'allowed_q1': float(r['away_q1']),
                'scored_q2': float(r['home_q2']), 'allowed_q2': float(r['away_q2']),
                'scored_q3': float(r['home_q3']), 'allowed_q3': float(r['away_q3']),
                'scored_q4': float(r['home_q4']), 'allowed_q4': float(r['away_q4'])
            })
            rows.append({
                'game_id': r['game_id'], 'season': r['season'], 'week': r['week'],
                'team': r['away_team'], 'opp': r['home_team'], 'is_home': 0,
                'scored_1h': away_1h, 'scored_2h': away_2h,
                'allowed_1h': home_1h, 'allowed_2h': home_2h,
                'scored_q1': float(r['away_q1']), 'allowed_q1': float(r['home_q1']),
                'scored_q2': float(r['away_q2']), 'allowed_q2': float(r['home_q2']),
                'scored_q3': float(r['away_q3']), 'allowed_q3': float(r['home_q3']),
                'scored_q4': float(r['away_q4']), 'allowed_q4': float(r['home_q4'])
            })
        if not rows:
            return pd.DataFrame()
        tg = pd.DataFrame(rows).sort_values(['team','season','week'])
        # Rolling means of past N games (prevent leakage with shift)
        for col in ['scored_1h','scored_2h','allowed_1h','allowed_2h',
                    'scored_q1','allowed_q1','scored_q2','allowed_q2','scored_q3','allowed_q3','scored_q4','allowed_q4']:
            tg[f'{col}_roll'] = (
                tg.groupby('team')[col]
                  .apply(lambda s: s.shift(1).rolling(window=window, min_periods=2).mean())
            )
        # Second-half adjustment tendency
        tg['delta_2h_minus_1h'] = tg['scored_2h'] - tg['scored_1h']
        tg['delta_2h_minus_1h_roll'] = (
            tg.groupby('team')['delta_2h_minus_1h']
              .apply(lambda s: s.shift(1).rolling(window=window, min_periods=2).mean())
        )
        # Pivot back to game_id for home/away roles
        home_feats = tg[tg['is_home'] == 1][['game_id','team',
            'scored_1h_roll','scored_2h_roll','allowed_1h_roll','allowed_2h_roll',
            'scored_q1_roll','allowed_q1_roll','scored_q2_roll','allowed_q2_roll','scored_q3_roll','allowed_q3_roll','scored_q4_roll','allowed_q4_roll',
            'delta_2h_minus_1h_roll'
        ]].rename(columns={
            'team': 'home_team',
            'scored_1h_roll': 'home_1h_scored_avg',
            'scored_2h_roll': 'home_2h_scored_avg',
            'allowed_1h_roll': 'home_1h_allowed_avg',
            'allowed_2h_roll': 'home_2h_allowed_avg',
            'scored_q1_roll': 'home_q1_scored_avg',
            'allowed_q1_roll': 'home_q1_allowed_avg',
            'scored_q2_roll': 'home_q2_scored_avg',
            'allowed_q2_roll': 'home_q2_allowed_avg',
            'scored_q3_roll': 'home_q3_scored_avg',
            'allowed_q3_roll': 'home_q3_allowed_avg',
            'scored_q4_roll': 'home_q4_scored_avg',
            'allowed_q4_roll': 'home_q4_allowed_avg',
            'delta_2h_minus_1h_roll': 'home_h2_minus_h1_avg'
        })
        away_feats = tg[tg['is_home'] == 0][['game_id','team',
            'scored_1h_roll','scored_2h_roll','allowed_1h_roll','allowed_2h_roll',
            'scored_q1_roll','allowed_q1_roll','scored_q2_roll','allowed_q2_roll','scored_q3_roll','allowed_q3_roll','scored_q4_roll','allowed_q4_roll',
            'delta_2h_minus_1h_roll'
        ]].rename(columns={
            'team': 'away_team',
            'scored_1h_roll': 'away_1h_scored_avg',
            'scored_2h_roll': 'away_2h_scored_avg',
            'allowed_1h_roll': 'away_1h_allowed_avg',
            'allowed_2h_roll': 'away_2h_allowed_avg',
            'scored_q1_roll': 'away_q1_scored_avg',
            'allowed_q1_roll': 'away_q1_allowed_avg',
            'scored_q2_roll': 'away_q2_scored_avg',
            'allowed_q2_roll': 'away_q2_allowed_avg',
            'scored_q3_roll': 'away_q3_scored_avg',
            'allowed_q3_roll': 'away_q3_allowed_avg',
            'scored_q4_roll': 'away_q4_scored_avg',
            'allowed_q4_roll': 'away_q4_allowed_avg',
            'delta_2h_minus_1h_roll': 'away_h2_minus_h1_avg'
        })
        # Return per-team rolling keyed by (team, season, week) for joining to both historical and future games
        return tg[['team','season','week',
                   'scored_1h_roll','scored_2h_roll','allowed_1h_roll','allowed_2h_roll',
                   'scored_q1_roll','allowed_q1_roll','scored_q2_roll','allowed_q2_roll','scored_q3_roll','allowed_q3_roll','scored_q4_roll','allowed_q4_roll',
                   'delta_2h_minus_1h_roll']]
    # end _compute_half_rolling

    try:
        half_feats = _compute_half_rolling(games)
        if half_feats is not None and not half_feats.empty:
            # Prepare asof merges for home and away to get last known rolling values at or before this week
            def _merge_asof_side(side: str):
                team_col = f'{side}_team'
                left = df[['game_id','season','week', team_col]].copy()
                left['week'] = pd.to_numeric(left['week'], errors='coerce')
                left = left.sort_values(['season', team_col, 'week'])
                right = half_feats.rename(columns={'team': team_col}).copy()
                right['week'] = pd.to_numeric(right['week'], errors='coerce')
                right = right.sort_values(['season', team_col, 'week'])
                merged = pd.merge_asof(
                    left,
                    right,
                    on='week', by=['season', team_col], direction='backward', allow_exact_matches=True
                )
                # Rename columns with side prefix
                rename_map = {
                    'scored_1h_roll': f'{side}_1h_scored_avg',
                    'scored_2h_roll': f'{side}_2h_scored_avg',
                    'allowed_1h_roll': f'{side}_1h_allowed_avg',
                    'allowed_2h_roll': f'{side}_2h_allowed_avg',
                    'scored_q1_roll': f'{side}_q1_scored_avg',
                    'allowed_q1_roll': f'{side}_q1_allowed_avg',
                    'scored_q2_roll': f'{side}_q2_scored_avg',
                    'allowed_q2_roll': f'{side}_q2_allowed_avg',
                    'scored_q3_roll': f'{side}_q3_scored_avg',
                    'allowed_q3_roll': f'{side}_q3_allowed_avg',
                    'scored_q4_roll': f'{side}_q4_scored_avg',
                    'allowed_q4_roll': f'{side}_q4_allowed_avg',
                    'delta_2h_minus_1h_roll': f'{side}_h2_minus_h1_avg',
                }
                merged = merged[['game_id'] + list(rename_map.keys())].rename(columns=rename_map)
                return merged

            mh = _merge_asof_side('home')
            ma = _merge_asof_side('away')
            df = df.merge(mh, on='game_id', how='left')
            df = df.merge(ma, on='game_id', how='left')

            # Build diffs used by models
            df['h1_scored_diff'] = (df['home_1h_scored_avg'] - df['away_1h_scored_avg']).fillna(0.0)
            df['h1_allowed_diff'] = (df['home_1h_allowed_avg'] - df['away_1h_allowed_avg']).fillna(0.0)
            df['h2_scored_diff'] = (df['home_2h_scored_avg'] - df['away_2h_scored_avg']).fillna(0.0)
            df['h2_allowed_diff'] = (df['home_2h_allowed_avg'] - df['away_2h_allowed_avg']).fillna(0.0)
            # Quarter-level diffs
            for q in ['q1','q2','q3','q4']:
                hs, as_ = f'home_{q}_scored_avg', f'away_{q}_scored_avg'
                ha, aa = f'home_{q}_allowed_avg', f'away_{q}_allowed_avg'
                sd, ad = f'{q}_scored_diff', f'{q}_allowed_diff'
                if hs in df.columns and as_ in df.columns:
                    df[sd] = (df[hs] - df[as_]).fillna(0.0)
                else:
                    df[sd] = 0.0
                if ha in df.columns and aa in df.columns:
                    df[ad] = (df[ha] - df[aa]).fillna(0.0)
                else:
                    df[ad] = 0.0
            df['h2_minus_h1_diff'] = (df['home_h2_minus_h1_avg'] - df['away_h2_minus_h1_avg']).fillna(0.0)
        else:
            for c in ['h1_scored_diff','h1_allowed_diff','h2_scored_diff','h2_allowed_diff',
                      'q1_scored_diff','q1_allowed_diff','q2_scored_diff','q2_allowed_diff','q3_scored_diff','q3_allowed_diff','q4_scored_diff','q4_allowed_diff',
                      'h2_minus_h1_diff']:
                df[c] = 0.0
    except Exception:
        for c in ['h1_scored_diff','h1_allowed_diff','h2_scored_diff','h2_allowed_diff',
                  'q1_scored_diff','q1_allowed_diff','q2_scored_diff','q2_allowed_diff','q3_scored_diff','q3_allowed_diff','q4_scored_diff','q4_allowed_diff',
                  'h2_minus_h1_diff']:
            if c not in df.columns:
                df[c] = 0.0

    return df
