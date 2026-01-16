import pandas as pd
import numpy as np
import os

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
                 'def_epa_1h', 'def_epa_2h', 'pace_secs_play', 'pass_rate', 'rush_rate',
                 'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                 'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                 'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                 'qb_adj', 'sos']
    ts = team_stats[[c for c in keep_cols if c in team_stats.columns]].copy()
    if ts.empty:
        return df

    # Normalize team keys on both sides to avoid mismatches (e.g., 'JAX' vs 'JAC')
    try:
        from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
    except Exception:
        _norm_team = lambda s: str(s)
    if 'team' in ts.columns:
        ts['team'] = ts['team'].astype(str).apply(_norm_team)

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
    # Normalize left side team column
    if side_team in left.columns:
        left[side_team] = left[side_team].astype(str).apply(_norm_team)
    right_prev = ts.copy()
    right_prev = right_prev.rename(columns={'team': side_team, 'week': 'prev_week'})
    # Columns to bring over from team_stats
    bring = [c for c in [side_team, 'season', 'prev_week', 'off_epa', 'def_epa', 'off_epa_1h', 'off_epa_2h', 'def_epa_1h', 'def_epa_2h', 'pace_secs_play', 'pass_rate', 'rush_rate',
                         'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                         'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                         'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                         'qb_adj', 'sos'] if c in right_prev.columns]
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
        'off_sack_rate': f'{side}_off_sack_rate',
        'def_sack_rate': f'{side}_def_sack_rate',
        'off_rz_pass_rate': f'{side}_off_rz_pass_rate',
        'def_rz_pass_rate': f'{side}_def_rz_pass_rate',
        'def_wr_share_mult': f'{side}_def_wr_share_mult',
        'def_te_share_mult': f'{side}_def_te_share_mult',
        'def_rb_share_mult': f'{side}_def_rb_share_mult',
        'def_wr_ypt_mult': f'{side}_def_wr_ypt_mult',
        'def_te_ypt_mult': f'{side}_def_te_ypt_mult',
        'def_rb_ypt_mult': f'{side}_def_rb_ypt_mult',
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
    # Normalize team keys
    try:
        from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
    except Exception:
        _norm_team = lambda s: str(s)
    if side_team in right.columns:
        right[side_team] = right[side_team].astype(str).apply(_norm_team)
    # Columns to bring over
    bring = [c for c in [side_team, 'season', 'prev_week', 'off_epa', 'def_epa', 'pace_secs_play', 'pass_rate', 'rush_rate',
                         'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                         'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                         'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                         'qb_adj', 'sos'] if c in right.columns]
    right = right[bring].copy()
    # Rename metrics to side-specific
    rename = {
        'off_epa': f'{side}_off_epa',
        'def_epa': f'{side}_def_epa',
        'pace_secs_play': f'{side}_pace_secs_play',
        'pass_rate': f'{side}_pass_rate',
        'rush_rate': f'{side}_rush_rate',
        'off_sack_rate': f'{side}_off_sack_rate',
        'def_sack_rate': f'{side}_def_sack_rate',
        'off_rz_pass_rate': f'{side}_off_rz_pass_rate',
        'def_rz_pass_rate': f'{side}_def_rz_pass_rate',
        'def_wr_share_mult': f'{side}_def_wr_share_mult',
        'def_te_share_mult': f'{side}_def_te_share_mult',
        'def_rb_share_mult': f'{side}_def_rb_share_mult',
        'def_wr_ypt_mult': f'{side}_def_wr_ypt_mult',
        'def_te_ypt_mult': f'{side}_def_te_ypt_mult',
        'def_rb_ypt_mult': f'{side}_def_rb_ypt_mult',
        'qb_adj': f'{side}_qb_adj',
        'sos': f'{side}_sos',
    }
    right = right.rename(columns=rename)
    # Ensure left has side team column
    if side_team not in left.columns:
        src_col = 'home_team' if side == 'home' else 'away_team'
        if src_col in left.columns:
            left[side_team] = left[src_col]
    # Normalize left side team column
    if side_team in left.columns:
        left[side_team] = left[side_team].astype(str).apply(_norm_team)
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

    # Optional: attach season-to-date EMA features (computed up to prior week)
    def _attach_team_stats_ema(left: pd.DataFrame, right: pd.DataFrame, side: str, alpha: float = 0.60) -> pd.DataFrame:
        try:
            if right is None or right.empty:
                return left
            try:
                from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
            except Exception:
                _norm_team = lambda s: str(s)
            side_team = f'{side}_team'
            # Normalize team column
            r = right.copy()
            if 'team' not in r.columns:
                # If upstream stats use home/away naming, try to synthesize
                if side_team in left.columns:
                    r['team'] = left[side_team]
                else:
                    return left
            r['team'] = r['team'].astype(str).apply(_norm_team)
            # Ensure season/week numeric
            for c in ['season','week']:
                if c in r.columns:
                    r[c] = pd.to_numeric(r[c], errors='coerce')
            # Columns to EMA
            ema_cols = [
                'off_epa','def_epa','pace_secs_play','pass_rate','rush_rate',
                'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                'qb_adj','sos'
            ]
            present = [c for c in ema_cols if c in r.columns]
            if not present:
                return left
            # Compute per (season, team) EMA across weeks.
            # Use transform() to avoid groupby.apply() quirks / failures.
            r = r.sort_values(['season','team','week'])
            for c in present:
                try:
                    s = pd.to_numeric(r[c], errors='coerce')
                except Exception:
                    s = r[c]
                try:
                    r[f'{c}_ema'] = s.groupby([r['season'], r['team']]).transform(
                        lambda x: x.ewm(alpha=float(alpha), adjust=False).mean()
                    )
                except Exception:
                    r[f'{c}_ema'] = s
            # We want EMA as of prior week; so we'll merge on prev_week
            r['prev_week'] = r['week']
            # Prepare right frame with only EMA columns
            keep = ['season','team','prev_week'] + [f'{c}_ema' for c in present]
            r = r[keep].drop_duplicates()
            # Align left keys
            l = left.copy()
            if side_team not in l.columns:
                src_col = 'home_team' if side == 'home' else 'away_team'
                if src_col in l.columns:
                    l[side_team] = l[src_col]
            l[side_team] = l[side_team].astype(str).apply(_norm_team)
            if 'week' in l.columns:
                try:
                    l['prev_week'] = pd.to_numeric(l['week'], errors='coerce') - 1
                except Exception:
                    l['prev_week'] = None
            # Rename EMA columns to side-prefixed names
            rename = {f'{c}_ema': f'{side}_{c}_ema' for c in present}
            r = r.rename(columns={'team': side_team})
            r = r.rename(columns=rename)
            merged = l.merge(r, on=['season', side_team, 'prev_week'], how='left')
            try:
                merged = merged.drop(columns=['prev_week'])
            except Exception:
                pass
            return merged
        except Exception:
            return left

    try:
        # Use TEAM_STATS_EMA_ALPHA if provided, else fallback to TEAM_RATING_EMA or default 0.60
        alpha_env = os.environ.get('TEAM_STATS_EMA_ALPHA', os.environ.get('TEAM_RATING_EMA', '0.60'))
        alpha = float(alpha_env) if alpha_env is not None else 0.60
    except Exception:
        alpha = 0.60
    df = _attach_team_stats_ema(df, team_stats, 'home', alpha=alpha)
    df = _attach_team_stats_ema(df, team_stats, 'away', alpha=alpha)

    # lines
    # Prefer matching lines by (season, week, home_team, away_team) instead of `game_id`.
    # Rationale: game_id formats can differ by source (underscores vs hyphens), while team names are stable.
    try:
        df['_home_team_norm'] = df['home_team'].astype(str).apply(_norm_team)
        df['_away_team_norm'] = df['away_team'].astype(str).apply(_norm_team)

        ldf = lines.copy()
        ldf['_home_team_norm'] = ldf['home_team'].astype(str).apply(_norm_team) if 'home_team' in ldf.columns else pd.NA
        ldf['_away_team_norm'] = ldf['away_team'].astype(str).apply(_norm_team) if 'away_team' in ldf.columns else pd.NA

        ldf = ldf[['season', 'week', '_home_team_norm', '_away_team_norm', 'spread_home', 'total', 'close_spread_home', 'close_total']].copy()
        # Keep one row per matchup; if duplicates exist, keep the last (typically the most recent snapshot).
        ldf = ldf.dropna(subset=['season', 'week', '_home_team_norm', '_away_team_norm'])
        ldf = ldf.drop_duplicates(subset=['season', 'week', '_home_team_norm', '_away_team_norm'], keep='last')

        df = df.merge(ldf, on=['season', 'week', '_home_team_norm', '_away_team_norm'], how='left')
    except Exception:
        # Fallback: attempt legacy merge by game_id if normalization fails.
        df = df.merge(lines[['game_id', 'spread_home', 'total', 'close_spread_home', 'close_total']], on='game_id', how='left')
    finally:
        for c in ['_home_team_norm', '_away_team_norm']:
            if c in df.columns:
                try:
                    df = df.drop(columns=[c])
                except Exception:
                    pass

    # If open lines are missing, backfill from close lines (real data, but different semantics).
    try:
        df['spread_home_is_close_fallback'] = df['spread_home'].isna() & df['close_spread_home'].notna()
        df['total_is_close_fallback'] = df['total'].isna() & df['close_total'].notna()
        df['spread_home'] = df['spread_home'].where(df['spread_home'].notna(), df['close_spread_home'])
        df['total'] = df['total'].where(df['total'].notna(), df['close_total'])
    except Exception:
        if 'spread_home_is_close_fallback' not in df.columns:
            df['spread_home_is_close_fallback'] = False
        if 'total_is_close_fallback' not in df.columns:
            df['total_is_close_fallback'] = False

    # optional weather join by game_id
    if weather is not None and not weather.empty:
        # Accept common weather column names
        wcols = [c for c in weather.columns if c in ['game_id','wx_temp_f','wx_wind_mph','wx_precip_pct','wx_precip_type','wx_sky','roof','surface','neutral_site']]
        if 'game_id' in wcols:
            df = df.merge(weather[wcols], on='game_id', how='left')
        # Derive numeric-friendly weather features
        try:
            for c in ['wx_temp_f','wx_wind_mph','wx_precip_pct']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
        # Roof flags (closed/indoor vs open)
        # Data sources vary: stadium_meta/weather files may use values like "open", "fixed", "dome", "retractable".
        try:
            if 'roof' in df.columns:
                roof_s = df['roof'].astype(str).str.lower().fillna('')
                is_open = roof_s.str.contains(r"\b(?:open|outdoor|outdoors)\b", regex=True)
                is_closed = roof_s.str.contains(r"\b(?:closed|dome|indoor|fixed)\b", regex=True)
                # Treat generic "retractable" as closed unless explicitly marked open.
                is_retractable = roof_s.str.contains("retractable", regex=False)
                is_closed = is_closed | (is_retractable & (~roof_s.str.contains("open", regex=False)))
                df['roof_open_flag'] = is_open.astype(int)
                df['roof_closed_flag'] = (is_closed & (~is_open)).astype(int)
        except Exception:
            # Ensure columns exist
            if 'roof_closed_flag' not in df.columns:
                df['roof_closed_flag'] = 0
            if 'roof_open_flag' not in df.columns:
                df['roof_open_flag'] = 0
        # Wind impact interaction when not closed
        # Avoid silent fill-to-zero here; keep NaN when the underlying forecast is missing.
        try:
            if 'wx_wind_mph' in df.columns and 'roof_closed_flag' in df.columns:
                wind = pd.to_numeric(df['wx_wind_mph'], errors='coerce')
                roof_closed = pd.to_numeric(df['roof_closed_flag'], errors='coerce').fillna(0)
                df['wind_open'] = wind * (1 - roof_closed)
        except Exception:
            if 'wind_open' not in df.columns:
                df['wind_open'] = pd.NA

    # target engineering
    df['home_margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']

    # simple differentials
    df['elo_diff'] = df['elo_home_pre'] - df['elo_away_pre']
    for col in ['off_epa', 'def_epa', 'pace_secs_play', 'pass_rate', 'rush_rate',
                 'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                 'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                 'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                 'qb_adj', 'sos']:
        h, a = f'home_{col}', f'away_{col}'
        if h in df.columns and a in df.columns:
            df[f'{col}_diff'] = df[h].fillna(0) - df[a].fillna(0)

    # EMA differentials (if present)
    for col in ['off_epa', 'def_epa', 'pace_secs_play', 'pass_rate', 'rush_rate',
                 'off_sack_rate','def_sack_rate','off_rz_pass_rate','def_rz_pass_rate',
                 'def_wr_share_mult','def_te_share_mult','def_rb_share_mult',
                 'def_wr_ypt_mult','def_te_ypt_mult','def_rb_ypt_mult',
                 'qb_adj', 'sos']:
        h, a = f'home_{col}_ema', f'away_{col}_ema'
        if h in df.columns and a in df.columns:
            df[f'{col}_ema_diff'] = pd.to_numeric(df[h], errors='coerce').fillna(0) - pd.to_numeric(df[a], errors='coerce').fillna(0)

    # Derived pressure aggregates: combined defensive sack rate (EMA and non-EMA)
    # IMPORTANT: do not treat missing sack rates as 0 (that silently invents "no pressure").
    try:
        if 'home_def_sack_rate_ema' in df.columns and 'away_def_sack_rate_ema' in df.columns:
            h_ema = pd.to_numeric(df.get('home_def_sack_rate_ema'), errors='coerce')
            a_ema = pd.to_numeric(df.get('away_def_sack_rate_ema'), errors='coerce')
            df['def_pressure_avg_ema'] = (h_ema + a_ema) / 2.0
        else:
            if 'def_pressure_avg_ema' not in df.columns:
                df['def_pressure_avg_ema'] = np.nan
    except Exception:
        if 'def_pressure_avg_ema' not in df.columns:
            df['def_pressure_avg_ema'] = np.nan
    try:
        if 'home_def_sack_rate' in df.columns and 'away_def_sack_rate' in df.columns:
            h_raw = pd.to_numeric(df.get('home_def_sack_rate'), errors='coerce')
            a_raw = pd.to_numeric(df.get('away_def_sack_rate'), errors='coerce')
            df['def_pressure_avg'] = (h_raw + a_raw) / 2.0
        else:
            if 'def_pressure_avg' not in df.columns:
                df['def_pressure_avg'] = np.nan
    except Exception:
        if 'def_pressure_avg' not in df.columns:
            df['def_pressure_avg'] = np.nan

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

    # Injury features (optional): derive team-level starter activity and compute diffs
    try:
        use_inj = str(os.environ.get('INJURY_FEATURES_OFF', '0')).strip().lower() not in {'1','true','yes','on'}
    except Exception:
        use_inj = True
    if use_inj:
        try:
            # We will aggregate by (season, week, team) using ESPN weekly depth charts if present
            from .player_props import _load_weekly_depth_chart  # type: ignore
        except Exception:
            _load_weekly_depth_chart = None  # type: ignore
        try:
            from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
        except Exception:
            _norm_team = lambda s: str(s)
        if _load_weekly_depth_chart is not None:
            try:
                # Build a union map across present (season, week) pairs in df
                sw = df[['season','week']].dropna().drop_duplicates()
                inj_rows = []
                # Baseline week for "starter" identity (week 1 by default)
                try:
                    baseline_week = int(pd.to_numeric(os.environ.get('INJURY_BASELINE_WEEK', 1), errors='coerce'))
                    if not np.isfinite(baseline_week) or baseline_week < 1:
                        baseline_week = 1
                except Exception:
                    baseline_week = 1
                for _, row in sw.iterrows():
                    try:
                        s = int(pd.to_numeric(row['season'], errors='coerce'))
                        w = int(pd.to_numeric(row['week'], errors='coerce'))
                    except Exception:
                        continue
                    try:
                        dc = _load_weekly_depth_chart(s, w)
                    except Exception:
                        dc = None
                    if dc is None or dc.empty:
                        continue
                    d = dc.copy()
                    # Normalize
                    if 'team' in d.columns:
                        d['team'] = d['team'].astype(str).apply(_norm_team)
                    pos_col = 'position' if 'position' in d.columns else None
                    if not pos_col:
                        continue
                    if 'active' not in d.columns:
                        d['active'] = True
                    # Coerce active to real booleans (depth chart files can store strings like "False").
                    def _as_bool(v):
                        try:
                            if pd.isna(v):
                                return True
                        except Exception:
                            pass
                        if isinstance(v, bool):
                            return v
                        s = str(v).strip().lower()
                        if s in {'false','0','no','n','inactive','out'}:
                            return False
                        if s in {'true','1','yes','y','active','in'}:
                            return True
                        # Default: treat non-empty values as True
                        return True
                    try:
                        d['active'] = d['active'].map(_as_bool)
                    except Exception:
                        pass
                    dr_col = 'depth_rank' if 'depth_rank' in d.columns else None
                    if dr_col is None:
                        d['_rn'] = d.groupby(['team', pos_col]).cumcount() + 1
                        dr_col = '_rn'
                    d['pos_up'] = d[pos_col].astype(str).str.upper()
                    # Prefer a baseline definition of "starter" (from baseline_week), so
                    # missing season starters are counted even when ESPN elevates a backup.
                    baseline_dc = None
                    if baseline_week >= 1:
                        try:
                            baseline_dc = _load_weekly_depth_chart(s, baseline_week)
                        except Exception:
                            baseline_dc = None
                    use_baseline = baseline_dc is not None and (baseline_dc is not None) and (not baseline_dc.empty)

                    if use_baseline:
                        b = baseline_dc.copy()
                        if 'team' in b.columns:
                            b['team'] = b['team'].astype(str).apply(_norm_team)
                        if 'active' not in b.columns:
                            b['active'] = True
                        try:
                            b['active'] = b['active'].map(_as_bool)
                        except Exception:
                            pass
                        b_pos_col = 'position' if 'position' in b.columns else None
                        if not b_pos_col:
                            use_baseline = False
                        else:
                            b['pos_up'] = b[b_pos_col].astype(str).str.upper()
                            b_dr_col = 'depth_rank' if 'depth_rank' in b.columns else None
                            if b_dr_col is None:
                                b['_rn'] = b.groupby(['team', b_pos_col]).cumcount() + 1
                                b_dr_col = '_rn'

                    if use_baseline:
                        # Starter slots per position (baseline week)
                        slots = {'QB': 1, 'RB': 1, 'TE': 1, 'WR': 2}
                        bb = b[b['pos_up'].isin(list(slots.keys()))].copy()
                        if bb.empty:
                            use_baseline = False
                        else:
                            bb = bb.sort_values(['team', 'pos_up', b_dr_col, 'player'])
                            bb['_slot'] = bb.groupby(['team', 'pos_up']).cumcount() + 1
                            bb['_max_slot'] = bb['pos_up'].map(slots).fillna(1).astype(int)
                            bb = bb[bb['_slot'] <= bb['_max_slot']].copy()

                            # Current-week actives for lookup
                            cur = d.copy()
                            cur = cur[cur['pos_up'].isin(list(slots.keys()))].copy()
                            cur_act = cur[['team', 'pos_up', 'player', 'active']].drop_duplicates()
                            try:
                                cur_act['active'] = cur_act['active'].map(_as_bool)
                            except Exception:
                                pass

                            # Try overlay predicted-not-active (if present) as an additional "out" signal
                            pred_not_active = None
                            try:
                                from pathlib import Path
                                pred_fp = Path(__file__).resolve().parents[1] / 'data' / f'predicted_not_active_{int(s)}_wk{int(w)}.csv'
                                if pred_fp.exists():
                                    pred_not_active = pd.read_csv(pred_fp)
                            except Exception:
                                pred_not_active = None
                            pna_set = set()
                            try:
                                if pred_not_active is not None and not pred_not_active.empty:
                                    tcol = 'team' if 'team' in pred_not_active.columns else None
                                    pcol = 'player' if 'player' in pred_not_active.columns else None
                                    if tcol and pcol:
                                        tmp = pred_not_active[[tcol, pcol]].dropna().copy()
                                        tmp[tcol] = tmp[tcol].astype(str).apply(_norm_team)
                                        tmp[pcol] = tmp[pcol].astype(str)
                                        pna_set = set(zip(tmp[tcol].tolist(), tmp[pcol].tolist()))
                            except Exception:
                                pna_set = set()

                            # For each baseline starter, determine if they are out this week
                            m = bb[['team', 'pos_up', '_slot', 'player']].merge(
                                cur_act, on=['team', 'pos_up', 'player'], how='left'
                            )
                            # Missing in current depth chart => out
                            m['_active_now'] = m['active'].map(_as_bool) if 'active' in m.columns else True
                            m['_active_now'] = m['_active_now'].where(m['active'].notna(), False)
                            # Predicted not active overrides
                            if pna_set:
                                m['_pna'] = list(zip(m['team'].astype(str), m['player'].astype(str)))
                                m['_active_now'] = m['_active_now'] & (~pd.Series(m['_pna']).isin(pna_set).to_numpy())

                            # Reduce to per-team counts
                            def _slot_out(team_val: str, pos: str, slot: int) -> int:
                                try:
                                    mm = m[(m['team'] == team_val) & (m['pos_up'] == pos) & (m['_slot'] == slot)]
                                    if mm.empty:
                                        return 0
                                    return 0 if bool(mm.iloc[0].get('_active_now', True)) else 1
                                except Exception:
                                    return 0

                            for team_val in sorted(set(m['team'].astype(str).tolist())):
                                qb_out = _slot_out(team_val, 'QB', 1)
                                rb1_out = _slot_out(team_val, 'RB', 1)
                                te1_out = _slot_out(team_val, 'TE', 1)
                                wr1_out = _slot_out(team_val, 'WR', 1)
                                wr2_out = _slot_out(team_val, 'WR', 2)
                                wr_top2_out = int(wr1_out + wr2_out)
                                starters_out = int(qb_out + wr1_out + te1_out + rb1_out)
                                inj_rows.append({
                                    'season': s, 'week': w, 'team': str(team_val),
                                    'inj_qb_out': qb_out,
                                    'inj_wr1_out': wr1_out,
                                    'inj_te1_out': te1_out,
                                    'inj_rb1_out': rb1_out,
                                    'inj_wr_top2_out': wr_top2_out,
                                    'inj_starters_out': starters_out,
                                })
                    if not use_baseline:
                        # Fallback: "current starter" per position based on this week's depth chart
                        starters = (
                            d.sort_values(['team','pos_up', dr_col, 'player'])
                             .groupby(['team','pos_up'], as_index=False)
                             .first()[['team','pos_up','active']]
                             .rename(columns={'active':'starter_active'})
                        )
                        try:
                            starters['starter_active'] = starters['starter_active'].map(_as_bool)
                        except Exception:
                            pass
                        wr2 = d[d['pos_up']=='WR'].copy()
                        if not wr2.empty:
                            wr2 = wr2.sort_values(['team', dr_col, 'player'])
                            wr2['_rn2'] = wr2.groupby('team').cumcount()+1
                            wr2_agg = wr2[wr2['_rn2']<=2].groupby('team')['active'].apply(lambda s: list(s.astype(bool))).reset_index()
                            wr2_agg = wr2_agg.rename(columns={'active':'wr_top2_active'})
                        else:
                            wr2_agg = pd.DataFrame(columns=['team','wr_top2_active'])
                        piv = starters.pivot(index='team', columns='pos_up', values='starter_active').reset_index()
                        for c in ['QB','WR','TE','RB']:
                            if c not in piv.columns:
                                piv[c] = True
                        if not wr2_agg.empty:
                            piv = piv.merge(wr2_agg, on='team', how='left')
                        else:
                            piv['wr_top2_active'] = [[] for _ in range(len(piv))]
                        for _, tr in piv.iterrows():
                            team = str(tr['team'])
                            qb_out = 0 if bool(tr.get('QB', True)) else 1
                            wr1_out = 0 if bool(tr.get('WR', True)) else 1
                            te1_out = 0 if bool(tr.get('TE', True)) else 1
                            rb1_out = 0 if bool(tr.get('RB', True)) else 1
                            wr2_arr = tr.get('wr_top2_active')
                            wr_top2_out = 0
                            if isinstance(wr2_arr, list):
                                wr_top2_out = int(sum([0 if bool(x) else 1 for x in wr2_arr]))
                            starters_out = qb_out + wr1_out + te1_out + rb1_out
                            inj_rows.append({
                                'season': s, 'week': w, 'team': team,
                                'inj_qb_out': qb_out,
                                'inj_wr1_out': wr1_out,
                                'inj_te1_out': te1_out,
                                'inj_rb1_out': rb1_out,
                                'inj_wr_top2_out': wr_top2_out,
                                'inj_starters_out': starters_out,
                            })
                inj = pd.DataFrame(inj_rows)
                if not inj.empty:
                    # Merge home/away flags
                    for side in ['home','away']:
                        key = f'{side}_team'
                        cols = ['season','week','team','inj_qb_out','inj_wr1_out','inj_te1_out','inj_rb1_out','inj_wr_top2_out','inj_starters_out']
                        df = df.merge(inj[cols].rename(columns={'team': key, 'inj_qb_out': f'{side}_inj_qb_out', 'inj_wr1_out': f'{side}_inj_wr1_out', 'inj_te1_out': f'{side}_inj_te1_out', 'inj_rb1_out': f'{side}_inj_rb1_out', 'inj_wr_top2_out': f'{side}_inj_wr_top2_out', 'inj_starters_out': f'{side}_inj_starters_out'}),
                                       on=['season','week', key], how='left')
                    # Compute diffs (home - away)
                    df['inj_qb_out_diff'] = df.get('home_inj_qb_out', 0).fillna(0) - df.get('away_inj_qb_out', 0).fillna(0)
                    df['inj_wr1_out_diff'] = df.get('home_inj_wr1_out', 0).fillna(0) - df.get('away_inj_wr1_out', 0).fillna(0)
                    df['inj_te1_out_diff'] = df.get('home_inj_te1_out', 0).fillna(0) - df.get('away_inj_te1_out', 0).fillna(0)
                    df['inj_rb1_out_diff'] = df.get('home_inj_rb1_out', 0).fillna(0) - df.get('away_inj_rb1_out', 0).fillna(0)
                    df['inj_wr_top2_out_diff'] = df.get('home_inj_wr_top2_out', 0).fillna(0) - df.get('away_inj_wr_top2_out', 0).fillna(0)
                    df['inj_starters_out_diff'] = df.get('home_inj_starters_out', 0).fillna(0) - df.get('away_inj_starters_out', 0).fillna(0)
                else:
                    # Ensure columns exist with zeros
                    for c in ['inj_qb_out_diff','inj_wr1_out_diff','inj_te1_out_diff','inj_rb1_out_diff','inj_wr_top2_out_diff','inj_starters_out_diff']:
                        if c not in df.columns:
                            df[c] = 0
            except Exception:
                # Fail-safe: add zeros if anything goes wrong
                for c in ['inj_qb_out_diff','inj_wr1_out_diff','inj_te1_out_diff','inj_rb1_out_diff','inj_wr_top2_out_diff','inj_starters_out_diff']:
                    if c not in df.columns:
                        df[c] = 0
        else:
            for c in ['inj_qb_out_diff','inj_wr1_out_diff','inj_te1_out_diff','inj_rb1_out_diff','inj_wr_top2_out_diff','inj_starters_out_diff']:
                if c not in df.columns:
                    df[c] = 0

    # Playoff-aware and contextual features
    # 1) is_postseason flag (week >= 19)
    try:
        wk = pd.to_numeric(df.get('week'), errors='coerce')
        df['is_postseason'] = (wk >= 19).astype(int)
    except Exception:
        df['is_postseason'] = 0

    # 2) neutral_site_flag using weather metadata if present (else 0)
    if 'neutral_site' in df.columns:
        try:
            ns = df['neutral_site']
            if ns.dtype == 'bool':
                df['neutral_site_flag'] = ns.astype(int)
            else:
                df['neutral_site_flag'] = ns.astype(str).str.lower().isin(['true','1','yes','y']).astype(int)
        except Exception:
            df['neutral_site_flag'] = 0
    else:
        df['neutral_site_flag'] = 0

    # 3) rest_days_diff derived from prior game dates per team
    try:
        g = games.copy()
        # Normalize date
        if 'game_date' in g.columns:
            dt = pd.to_datetime(g['game_date'], errors='coerce', utc=True)
        else:
            dt = pd.to_datetime(g.get('date'), errors='coerce', utc=True)
        g['__dt'] = dt
        # Build per-team chronological sequence
        home = g[['game_id','season','week','home_team','__dt']].rename(columns={'home_team':'team'})
        away = g[['game_id','season','week','away_team','__dt']].rename(columns={'away_team':'team'})
        ta = pd.concat([home, away], ignore_index=True)
        ta = ta.dropna(subset=['team'])
        ta = ta.sort_values(['team','season','week','__dt'])
        # Prior date per team
        ta['__prev_dt'] = ta.groupby('team')['__dt'].shift(1)
        # Rest days per (team, game_id)
        ta['__rest_days'] = (ta['__dt'] - ta['__prev_dt']).dt.total_seconds() / (86400.0)
        # Aggregate back to game_id for home/away
        rest_home = ta.merge(g[['game_id','home_team']], left_on=['game_id','team'], right_on=['game_id','home_team'], how='inner')
        rest_home = rest_home[['game_id','__rest_days']].rename(columns={'__rest_days':'home_rest_days'})
        rest_away = ta.merge(g[['game_id','away_team']], left_on=['game_id','team'], right_on=['game_id','away_team'], how='inner')
        rest_away = rest_away[['game_id','__rest_days']].rename(columns={'__rest_days':'away_rest_days'})
        rest = pd.merge(rest_home, rest_away, on='game_id', how='outer')
        df = df.merge(rest, on='game_id', how='left')
        # Diff and fill
        df['home_rest_days'] = pd.to_numeric(df.get('home_rest_days'), errors='coerce')
        df['away_rest_days'] = pd.to_numeric(df.get('away_rest_days'), errors='coerce')
        df['rest_days_diff'] = df['home_rest_days'].fillna(0) - df['away_rest_days'].fillna(0)
        # Clean helpers if sneaked in
        for c in ['__dt','__prev_dt']:
            if c in df.columns:
                try:
                    df = df.drop(columns=[c])
                except Exception:
                    pass
    except Exception:
        # default zeros if anything fails
        for c in ['home_rest_days','away_rest_days','rest_days_diff']:
            if c not in df.columns:
                df[c] = 0.0

    # Optional: attach team ratings (EMA-based) for additional priors/features
    try:
        from .team_ratings import attach_team_ratings_to_view as _attach_ratings  # type: ignore
        try:
            from .data_sources import (
                load_pfr_drive_stats,
                load_redzone_splits,
                load_explosive_rates,
                load_penalties_stats,
                load_special_teams,
                load_officiating_crews,
                load_weather_noaa,
            )
        except Exception:
            load_pfr_drive_stats = lambda: pd.DataFrame()
            load_redzone_splits = lambda: pd.DataFrame()
            load_explosive_rates = lambda: pd.DataFrame()
            load_penalties_stats = lambda: pd.DataFrame()
            load_special_teams = lambda: pd.DataFrame()
            load_officiating_crews = lambda: pd.DataFrame()
            load_weather_noaa = lambda: pd.DataFrame()
    except Exception:
        _attach_ratings = None  # type: ignore
    # Respect TEAM_RATINGS_OFF env toggle
    try:
        ratings_off = str(os.environ.get('TEAM_RATINGS_OFF', '0')).strip().lower() in {'1','true','yes','on'}
    except Exception:
        ratings_off = False
    if (not ratings_off) and (_attach_ratings is not None):
        try:
            df = _attach_ratings(df)
        except Exception:
            # Non-fatal: keep base features
            pass

    # --- Extended features (optional; degrade gracefully) ---
    try:
        from .team_normalizer import normalize_team_name as _norm_team_ext  # type: ignore
    except Exception:
        _norm_team_ext = lambda s: str(s)

    try:
        drv = load_pfr_drive_stats()
        if drv is not None and not drv.empty:
            # Expect per team-week; attach prior-week drive metrics to each side
            for side in ['home','away']:
                key = f'{side}_team'
                key_norm = f'__{key}_norm'
                r = drv.copy()
                if 'team' in r.columns:
                    r = r.rename(columns={'team': key})
                # Prefer prior-week attach
                r['prev_week'] = pd.to_numeric(r.get('week'), errors='coerce')
                l = df.copy()
                l['prev_week'] = pd.to_numeric(l.get('week'), errors='coerce') - 1
                # Normalize team keys (many sources use abbreviations)
                l[key_norm] = l.get(key).astype(str).apply(_norm_team_ext)
                r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                bring = [c for c in [key,'season','prev_week','points_per_drive','td_per_drive','fg_per_drive','avg_start_fp','yards_per_drive','seconds_per_drive','drives'] if c in r.columns]
                r = r[bring]
                rename = {
                    'points_per_drive': f'{side}_ppd',
                    'td_per_drive': f'{side}_td_per_drive',
                    'fg_per_drive': f'{side}_fg_per_drive',
                    'avg_start_fp': f'{side}_avg_start_fp',
                    'yards_per_drive': f'{side}_yards_per_drive',
                    'seconds_per_drive': f'{side}_seconds_per_drive',
                    'drives': f'{side}_drives'
                }
                r = r.rename(columns=rename)
                # Ensure normalized merge key exists on right side
                if key_norm not in r.columns:
                    r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                # Avoid duplicating/suffixing the core team column in df.
                r = r.drop(columns=[key], errors='ignore')
                df = l.merge(r, on=['season', key_norm, 'prev_week'], how='left')
                df = df.drop(columns=[key_norm, 'prev_week'], errors='ignore')
            # Diffs
            for c in ['ppd','td_per_drive','fg_per_drive','avg_start_fp','yards_per_drive','seconds_per_drive','drives']:
                hc, ac = f'home_{c}', f'away_{c}'
                if hc in df.columns and ac in df.columns:
                    df[f'{c}_diff'] = pd.to_numeric(df[hc], errors='coerce') - pd.to_numeric(df[ac], errors='coerce')
    except Exception:
        pass
    try:
        rz = load_redzone_splits()
        if rz is not None and not rz.empty:
            for side in ['home','away']:
                key = f'{side}_team'
                key_norm = f'__{key}_norm'
                r = rz.copy()
                if 'team' in r.columns:
                    r = r.rename(columns={'team': key})
                r['prev_week'] = pd.to_numeric(r.get('week'), errors='coerce')
                l = df.copy()
                l['prev_week'] = pd.to_numeric(l.get('week'), errors='coerce') - 1
                l[key_norm] = l.get(key).astype(str).apply(_norm_team_ext)
                r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                bring = [c for c in [key,'season','prev_week','rzd_off_eff','rzd_def_eff','rzd_off_td_rate','rzd_def_td_rate'] if c in r.columns]
                r = r[bring]
                rename = {
                    'rzd_off_eff': f'{side}_rzd_off_eff',
                    'rzd_def_eff': f'{side}_rzd_def_eff',
                    'rzd_off_td_rate': f'{side}_rzd_off_td_rate',
                    'rzd_def_td_rate': f'{side}_rzd_def_td_rate',
                }
                r = r.rename(columns=rename)
                if key_norm not in r.columns:
                    r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                r = r.drop(columns=[key], errors='ignore')
                df = l.merge(r, on=['season', key_norm, 'prev_week'], how='left')
                df = df.drop(columns=[key_norm, 'prev_week'], errors='ignore')
            for c in ['rzd_off_eff','rzd_def_eff','rzd_off_td_rate','rzd_def_td_rate']:
                hc, ac = f'home_{c}', f'away_{c}'
                if hc in df.columns and ac in df.columns:
                    df[f'{c}_diff'] = pd.to_numeric(df[hc], errors='coerce') - pd.to_numeric(df[ac], errors='coerce')
    except Exception:
        pass
    try:
        expl = load_explosive_rates()
        if expl is not None and not expl.empty:
            for side in ['home','away']:
                key = f'{side}_team'
                key_norm = f'__{key}_norm'
                r = expl.copy()
                if 'team' in r.columns:
                    r = r.rename(columns={'team': key})
                r['prev_week'] = pd.to_numeric(r.get('week'), errors='coerce')
                l = df.copy()
                l['prev_week'] = pd.to_numeric(l.get('week'), errors='coerce') - 1
                l[key_norm] = l.get(key).astype(str).apply(_norm_team_ext)
                r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                bring = [c for c in [key,'season','prev_week','explosive_pass_rate','explosive_run_rate'] if c in r.columns]
                r = r[bring]
                rename = {
                    'explosive_pass_rate': f'{side}_explosive_pass_rate',
                    'explosive_run_rate': f'{side}_explosive_run_rate',
                }
                r = r.rename(columns=rename)
                if key_norm not in r.columns:
                    r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                r = r.drop(columns=[key], errors='ignore')
                df = l.merge(r, on=['season', key_norm, 'prev_week'], how='left')
                df = df.drop(columns=[key_norm, 'prev_week'], errors='ignore')
            for c in ['explosive_pass_rate','explosive_run_rate']:
                hc, ac = f'home_{c}', f'away_{c}'
                if hc in df.columns and ac in df.columns:
                    df[f'{c}_diff'] = pd.to_numeric(df[hc], errors='coerce') - pd.to_numeric(df[ac], errors='coerce')
    except Exception:
        pass
    try:
        pen = load_penalties_stats()
        if pen is not None and not pen.empty:
            for side in ['home','away']:
                key = f'{side}_team'
                key_norm = f'__{key}_norm'
                r = pen.copy()
                if 'team' in r.columns:
                    r = r.rename(columns={'team': key})
                r['prev_week'] = pd.to_numeric(r.get('week'), errors='coerce')
                l = df.copy()
                l['prev_week'] = pd.to_numeric(l.get('week'), errors='coerce') - 1
                l[key_norm] = l.get(key).astype(str).apply(_norm_team_ext)
                r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                bring = [c for c in [key,'season','prev_week','penalty_rate','turnover_adj_rate'] if c in r.columns]
                r = r[bring]
                rename = {
                    'penalty_rate': f'{side}_penalty_rate',
                    'turnover_adj_rate': f'{side}_turnover_adj_rate',
                }
                r = r.rename(columns=rename)
                if key_norm not in r.columns:
                    r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                r = r.drop(columns=[key], errors='ignore')
                df = l.merge(r, on=['season', key_norm, 'prev_week'], how='left')
                df = df.drop(columns=[key_norm, 'prev_week'], errors='ignore')
            for c in ['penalty_rate','turnover_adj_rate']:
                hc, ac = f'home_{c}', f'away_{c}'
                if hc in df.columns and ac in df.columns:
                    df[f'{c}_diff'] = pd.to_numeric(df[hc], errors='coerce') - pd.to_numeric(df[ac], errors='coerce')
    except Exception:
        pass
    try:
        st = load_special_teams()
        if st is not None and not st.empty:
            for side in ['home','away']:
                key = f'{side}_team'
                key_norm = f'__{key}_norm'
                r = st.copy()
                if 'team' in r.columns:
                    r = r.rename(columns={'team': key})
                r['prev_week'] = pd.to_numeric(r.get('week'), errors='coerce')
                l = df.copy()
                l['prev_week'] = pd.to_numeric(l.get('week'), errors='coerce') - 1
                l[key_norm] = l.get(key).astype(str).apply(_norm_team_ext)
                r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                bring = [c for c in [key,'season','prev_week','fg_acc','punt_epa','kick_return_epa','touchback_rate'] if c in r.columns]
                r = r[bring]
                rename = {
                    'fg_acc': f'{side}_fg_acc',
                    'punt_epa': f'{side}_punt_epa',
                    'kick_return_epa': f'{side}_kick_return_epa',
                    'touchback_rate': f'{side}_touchback_rate',
                }
                r = r.rename(columns=rename)
                if key_norm not in r.columns:
                    r[key_norm] = r.get(key).astype(str).apply(_norm_team_ext)
                r = r.drop(columns=[key], errors='ignore')
                df = l.merge(r, on=['season', key_norm, 'prev_week'], how='left')
                df = df.drop(columns=[key_norm, 'prev_week'], errors='ignore')
            for c in ['fg_acc','punt_epa','kick_return_epa','touchback_rate']:
                hc, ac = f'home_{c}', f'away_{c}'
                if hc in df.columns and ac in df.columns:
                    df[f'{c}_diff'] = pd.to_numeric(df[hc], errors='coerce') - pd.to_numeric(df[ac], errors='coerce')
    except Exception:
        pass
    try:
        oc = load_officiating_crews()
        if oc is not None and not oc.empty:
            # Merge by game_id when present
            if 'game_id' in df.columns and 'game_id' in oc.columns:
                df = df.merge(oc[['game_id','crew_penalty_rate','crew_dpi_rate','crew_pace_adj']], on='game_id', how='left')
    except Exception:
        pass
    try:
        noa = load_weather_noaa()
        if noa is not None and not noa.empty:
            if 'game_id' in df.columns and 'game_id' in noa.columns:
                df = df.merge(noa[['game_id','wx_gust_mph','wx_dew_point_f']], on='game_id', how='left')
    except Exception:
        pass

    # Phase A totals adjustment (lightweight, optional)
    # Compute a small additive delta to predicted totals using red-zone, explosives, penalties, and special teams diffs
    try:
        import os as _os
        w_rz = float(_os.environ.get('PHASEA_TOTALS_W_RZ', 3.0))
        w_expl = float(_os.environ.get('PHASEA_TOTALS_W_EXPL', 18.0))
        w_pen = float(_os.environ.get('PHASEA_TOTALS_W_PEN', -20.0))
        w_st = float(_os.environ.get('PHASEA_TOTALS_W_ST', 6.0))
        delta_max = float(_os.environ.get('PHASEA_TOTALS_DELTA_MAX', 5.0))

        def _num(s):
            return pd.to_numeric(df.get(s), errors='coerce') if s in df.columns else pd.Series(index=df.index, dtype=float)

        rz_off_td_diff = _num('rzd_off_td_rate_diff').fillna(0.0)
        rz_def_td_diff = _num('rzd_def_td_rate_diff').fillna(0.0)
        expl_pass_diff = _num('explosive_pass_rate_diff').fillna(0.0)
        expl_run_diff = _num('explosive_run_rate_diff').fillna(0.0)
        pen_rate_diff = _num('penalty_rate_diff').fillna(0.0)
        st_fg_acc_diff = _num('fg_acc_diff').fillna(0.0)
        st_kr_epa_diff = _num('kick_return_epa_diff').fillna(0.0)
        st_tb_rate_diff = _num('touchback_rate_diff').fillna(0.0)

        rz_term = w_rz * (rz_off_td_diff - rz_def_td_diff)
        expl_term = w_expl * (expl_pass_diff + expl_run_diff)
        pen_term = w_pen * pen_rate_diff
        st_term = w_st * (st_fg_acc_diff + 0.5 * st_kr_epa_diff - 0.5 * st_tb_rate_diff)
        phase_delta = rz_term + expl_term + pen_term + st_term
        try:
            phase_delta = phase_delta.clip(lower=-delta_max, upper=delta_max)
        except Exception:
            # Fallback clamp
            phase_delta = phase_delta.apply(lambda x: max(-delta_max, min(delta_max, float(x) if pd.notna(x) else 0.0)))
        df['phase_a_total_delta'] = phase_delta
    except Exception:
        # Non-fatal: omit delta if any issue
        if 'phase_a_total_delta' not in df.columns:
            df['phase_a_total_delta'] = 0.0

    return df
