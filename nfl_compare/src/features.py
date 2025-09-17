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
    # Handle missing/NaN scores: no update to ELO
    try:
        sa = float(score_a)
        sb = float(score_b)
        if not (np.isfinite(sa) and np.isfinite(sb)):
            return elo_a
    except Exception:
        return elo_a
    exp_a = expected_score(elo_a, elo_b, HOME_ADV if home else 0.0)
    result_a = 1.0 if sa > sb else (0.5 if sa == sb else 0.0)
    margin = abs(sa - sb)
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
        # Coerce scores to floats and handle missing
        try:
            hs = float(pd.to_numeric(g.get('home_score'), errors='coerce'))
        except Exception:
            hs = float('nan')
        try:
            as_ = float(pd.to_numeric(g.get('away_score'), errors='coerce'))
        except Exception:
            as_ = float('nan')
        elo_home, elo_away = elos.get(home, initial_elo()), elos.get(away, initial_elo())
        rows.append({
            'game_id': g['game_id'],
            'elo_home_pre': elo_home,
            'elo_away_pre': elo_away,
        })
        # update after game only if scores are present
        try:
            if pd.notna(hs) and pd.notna(as_):
                new_home = update_elo(elo_home, elo_away, hs, as_, home=True)
                new_away = update_elo(elo_away, elo_home, as_, hs, home=False)
                elos[home] = new_home
                elos[away] = new_away
        except Exception:
            # If any issue occurs, keep previous elos and continue
            pass
    return pd.DataFrame(rows)


def _attach_team_stats_prior(df: pd.DataFrame, team_stats: pd.DataFrame, side: str) -> pd.DataFrame:
    """Attach team-week stats for the given side using a prior-week fallback.

    For each game row (season, week, {side}_team), join the most recent stats row
    for that team where stats.week <= (game.week - 1). This avoids using same-week
    stats (leakage) and provides meaningful features for future weeks.

    If no prior-week stats exist in the same season, the row remains NaN for those
    fields (downstream code fills with 0 where needed).
    """
    if team_stats is None or team_stats.empty:
        return df

    # Standardize and subset right-hand stats table
    keep_cols = ['season', 'week', 'team', 'off_epa', 'def_epa', 'off_epa_1h', 'off_epa_2h',
                 'def_epa_1h', 'def_epa_2h', 'pace_secs_play', 'pass_rate', 'rush_rate', 'qb_adj', 'sos']
    ts = team_stats[[c for c in keep_cols if c in team_stats.columns]].copy()
    if ts.empty:
        return df

    side_team = f"{side}_team"
    # First perform an exact prior-week join (prev_week = week - 1)
    # Build left frame with explicit prev_week key
    left = df.copy()
    left['prev_week'] = pd.to_numeric(left.get('week'), errors='coerce') - 1
    # Ensure the side team column exists on the left
    side_team = f"{side}_team"
    if side_team not in left.columns:
        # Try to map from base team columns
        src_col = 'home_team' if side == 'home' else 'away_team'
        if src_col in left.columns:
            left[side_team] = left[src_col]
    right_prev = ts.copy()
    right_prev = right_prev.rename(columns={'team': side_team, 'week': 'prev_week'})
    # Columns to bring over from team_stats
    bring = [c for c in [side_team, 'season', 'prev_week', 'off_epa', 'def_epa', 'off_epa_1h', 'off_epa_2h', 'def_epa_1h', 'def_epa_2h', 'pace_secs_play', 'pass_rate', 'rush_rate', 'qb_adj', 'sos'] if c in right_prev.columns]
    right_prev = right_prev[bring].copy()
    rename_prev = {
        'off_epa': f'{side}_off_epa',
        'def_epa': f'{side}_def_epa',
        'off_epa_1h': f'off_epa_1h_{side[0]}',
        'off_epa_2h': f'off_epa_2h_{side[0]}',
        'def_epa_1h': f'def_epa_1h_{side[0]}',
        'def_epa_2h': f'def_epa_2h_{side[0]}',
        'pace_secs_play': f'{side}_pace_secs_play',
        'pass_rate': f'{side}_pass_rate',
        'rush_rate': f'{side}_rush_rate',
        'qb_adj': f'{side}_qb_adj',
        'sos': f'{side}_sos',
    }
    right_prev = right_prev.rename(columns=rename_prev)
    # Merge using left that contains prev_week and side_team
    merged = left.merge(right_prev, on=['season', side_team, 'prev_week'], how='left')
    if 'prev_week' in merged.columns:
        merged = merged.drop(columns=['prev_week'])

    # If still missing for some rows, attempt broader asof fallback
    needs_fallback = False
    metric_col = f'{side}_off_epa'
    if metric_col in merged.columns:
        needs_fallback = merged[metric_col].isna().any()
    if needs_fallback:
        # Prepare asof tables
        ts_side = ts.rename(columns={'team': side_team, 'week': 'ts_week'}).copy()
        for c in ['season', 'ts_week']:
            if c in ts_side.columns:
                ts_side[c] = pd.to_numeric(ts_side[c], errors='coerce')
        out = merged.copy()
        out['week_for_stats'] = pd.to_numeric(out.get('week'), errors='coerce').fillna(0) - 1
        out['week_for_stats'] = out['week_for_stats'].where(out['week_for_stats'] > 0, 0)
        if 'season' in out.columns:
            out['season'] = pd.to_numeric(out['season'], errors='coerce')
        out = out.sort_values(['season', side_team, 'week_for_stats'], kind='mergesort')
        ts_side = ts_side.sort_values([c for c in ['season', side_team, 'ts_week'] if c in ts_side.columns], kind='mergesort')
        try:
            merged = pd.merge_asof(
                out,
                ts_side,
                left_on='week_for_stats',
                right_on='ts_week',
                by=['season', side_team],
                direction='backward',
                allow_exact_matches=True
            )
            # Drop helper columns if present
            drop_cols = [c for c in ['week_for_stats', 'ts_week'] if c in merged.columns]
            if drop_cols:
                merged = merged.drop(columns=drop_cols)
        except Exception:
            pass

    return merged


def _attach_team_stats_exact_prev(df: pd.DataFrame, team_stats: pd.DataFrame, side: str) -> pd.DataFrame:
    """Attach team-week stats using an exact prior-week (week-1) match on season+team.

    This is a stricter fallback than the asof join and is used when we detect that
    the prior-week attach produced all-NaN values for the target subset (e.g.,
    when building features for a future week and asof grouping had dtype/sort issues).
    """
    if team_stats is None or team_stats.empty:
        return df
    side_team = f"{side}_team"
    if side_team not in df.columns:
        return df
    # Build left frame with explicit prev_week key
    left = df.copy()
    left['prev_week'] = pd.to_numeric(left.get('week'), errors='coerce') - 1
    right = team_stats.copy()
    right = right.rename(columns={'team': side_team, 'week': 'prev_week'})
    # Columns to bring over
    bring = [c for c in [side_team, 'season', 'prev_week', 'off_epa', 'def_epa', 'pace_secs_play', 'pass_rate', 'rush_rate', 'qb_adj', 'sos'] if c in right.columns]
    right = right[bring].copy()
    # Rename metrics to side-specific
    rename = {
        'off_epa': f'{side}_off_epa',
        'def_epa': f'{side}_def_epa',
        'pace_secs_play': f'{side}_pace_secs_play',
        'pass_rate': f'{side}_pass_rate',
        'rush_rate': f'{side}_rush_rate',
        'qb_adj': f'{side}_qb_adj',
        'sos': f'{side}_sos',
    }
    right = right.rename(columns=rename)
    # Ensure left has side team column
    if side_team not in left.columns:
        src_col = 'home_team' if side == 'home' else 'away_team'
        if src_col in left.columns:
            left[side_team] = left[src_col]
    # Merge exact prior-week stats
    merged = left.merge(right, on=['season', side_team, 'prev_week'], how='left')
    # Drop helper
    if 'prev_week' in merged.columns:
        merged = merged.drop(columns=['prev_week'])
    return merged


def merge_features(games: pd.DataFrame, team_stats: pd.DataFrame, lines: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    elo = compute_elo(games)
    df = games.merge(elo, on='game_id', how='left')

    # Attach team stats with prior-week fallback to avoid empty features for future weeks
    df = _attach_team_stats_prior(df, team_stats, 'home')
    df = _attach_team_stats_prior(df, team_stats, 'away')

    # lines
    df = df.merge(lines[['game_id', 'spread_home', 'total', 'close_spread_home', 'close_total']], on='game_id', how='left')

    # optional weather join by game_id
    if weather is not None and not weather.empty:
        # Accept common weather column names
        wcols = [c for c in weather.columns if c in ['game_id','wx_temp_f','wx_wind_mph','wx_precip_pct','wx_precip_type','wx_sky','roof','surface','neutral_site']]
        if 'game_id' in wcols:
            df = df.merge(weather[wcols], on='game_id', how='left')

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
    # Defensive: if prior-week stats failed to attach (all-NaN), retry attachment now
    try:
        if team_stats is not None and not team_stats.empty:
            need_home = ('home_off_epa' in df.columns) and (df['home_off_epa'].notna().sum() == 0)
            need_away = ('away_off_epa' in df.columns) and (df['away_off_epa'].notna().sum() == 0)
            if need_home:
                df = _attach_team_stats_prior(df, team_stats, 'home')
                if df['home_off_epa'].notna().sum() == 0:
                    df = _attach_team_stats_exact_prev(df, team_stats, 'home')
            if need_away:
                df = _attach_team_stats_prior(df, team_stats, 'away')
                if df['away_off_epa'].notna().sum() == 0:
                    df = _attach_team_stats_exact_prev(df, team_stats, 'away')
    except Exception:
        pass

    return df
