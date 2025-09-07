import os
import pandas as pd
import numpy as np

# Betting recommendation heuristics inspired by MLB-Compare presentation layer.


def edge_to_units(edge: float) -> float:
    # Map model edge to suggested units (flat staking baseline)
    # 0.5-1.0 -> 0.25u, 1-2 -> 0.5u, 2-3 -> 0.75u, >3 -> 1u
    e = abs(edge)
    if e < 0.5:
        return 0.0
    if e < 1.0:
        return 0.25
    if e < 2.0:
        return 0.5
    if e < 3.0:
        return 0.75
    return 1.0


def american_to_decimal(ml: float) -> float:
    if ml is None or np.isnan(ml):
        return np.nan
    if ml > 0:
        return 1 + ml / 100.0
    else:
        return 1 + 100.0 / abs(ml)


def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1:
        return np.nan
    if p >= 0.5:
        return - (p / (1 - p)) * 100
    else:
        return (1 - p) / p * 100


def _implied_prob_from_american(ml: float) -> float:
    if ml is None or np.isnan(ml):
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


def make_recommendations(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Augment model predictions with betting recommendation metrics.

    Environment variables (optional):
      REC_EV_THRESHOLD      -> minimum percentage edge (e.g. 2.0) to flag value_* booleans
      REC_MIN_PROB_CLIP     -> lower probability clip for fair odds (default 0.01)
      REC_MAX_EDGE_CAP      -> cap absolute moneyline edge percentage (default 500)
      REC_FILTER_VALUE_ONLY -> if truthy, return only rows where any value flag true
    """
    df = df_pred.copy()

    # Configuration from environment
    try:
        ev_threshold = float(os.getenv('REC_EV_THRESHOLD', '2'))
    except Exception:
        ev_threshold = 2.0
    try:
        min_prob_clip = float(os.getenv('REC_MIN_PROB_CLIP', '0.01'))
    except Exception:
        min_prob_clip = 0.01
    try:
        edge_cap = float(os.getenv('REC_MAX_EDGE_CAP', '500'))
    except Exception:
        edge_cap = 500.0
    filter_value_only = os.getenv('REC_FILTER_VALUE_ONLY', '').lower() in {'1','true','yes','y'}

    # Moneyline edges
    dec_home = df['moneyline_home'].apply(american_to_decimal) if 'moneyline_home' in df else np.nan
    dec_away = df['moneyline_away'].apply(american_to_decimal) if 'moneyline_away' in df else np.nan

    # Fair odds (American) from model probabilities
    df['fair_home_ml'] = df['prob_home_win'].clip(1e-4, 1-1e-4).apply(prob_to_american)
    df['fair_away_ml'] = (1 - df['prob_home_win']).clip(1e-4, 1-1e-4).apply(prob_to_american)

    # Correct fair decimal odds: decimal = 1 / probability
    # (Old logic incorrectly inverted favorites/underdogs producing nonsensical EV.)
    # Apply probability clipping for fair odds display to avoid absurd huge lines
    p_clip = df['prob_home_win'].clip(min_prob_clip, 1 - min_prob_clip)
    df['fair_home_dec'] = 1.0 / p_clip
    df['fair_away_dec'] = 1.0 / (1 - p_clip)

    # If book odds present, compute edge in pct vs fair
    if isinstance(dec_home, pd.Series):
        # Edge defined as (book_decimal - fair_decimal)/fair_decimal * 100.
        # Positive edge => book offers a better (higher) payout than model's fair price.
        edge_home_raw = (dec_home - df['fair_home_dec']) / df['fair_home_dec'] * 100
        edge_away_raw = (dec_away - df['fair_away_dec']) / df['fair_away_dec'] * 100
        df['edge_home_ml_%_raw'] = edge_home_raw
        df['edge_away_ml_%_raw'] = edge_away_raw
        # Cap extreme edges (display only)
        df['edge_home_ml_%'] = edge_home_raw.clip(lower=-edge_cap, upper=edge_cap)
        df['edge_away_ml_%'] = edge_away_raw.clip(lower=-edge_cap, upper=edge_cap)
    else:
        df['edge_home_ml_%_raw'] = np.nan
        df['edge_away_ml_%_raw'] = np.nan
        df['edge_home_ml_%'] = np.nan
        df['edge_away_ml_%'] = np.nan

    # Spread edge: compare predicted margin vs book spread
    if 'spread_home' in df.columns:
        df['edge_spread_pts'] = df['pred_margin'] - df['spread_home']
        df['units_spread'] = df['edge_spread_pts'].apply(edge_to_units)
    else:
        df['edge_spread_pts'] = np.nan
        df['units_spread'] = 0.0

    # Total edge: compare predicted total vs book total
    if 'total' in df.columns:
        df['edge_total_pts'] = df['pred_total'] - df['total']
        df['units_total'] = df['edge_total_pts'].apply(edge_to_units)
    else:
        df['edge_total_pts'] = np.nan
        df['units_total'] = 0.0

    # Moneyline staking suggestions
    df['units_home_ml'] = df['edge_home_ml_%'].apply(lambda e: edge_to_units(e/100*3))  # scale heuristic
    df['units_away_ml'] = df['edge_away_ml_%'].apply(lambda e: edge_to_units(e/100*3))

    # Implied probabilities from market odds
    if 'moneyline_home' in df and 'moneyline_away' in df:
        df['implied_home_prob'] = df['moneyline_home'].apply(_implied_prob_from_american)
        df['implied_away_prob'] = df['moneyline_away'].apply(_implied_prob_from_american)
    else:
        df['implied_home_prob'] = np.nan
        df['implied_away_prob'] = np.nan

    # Value flags based on threshold
    df['value_home_ml'] = df['edge_home_ml_%'].ge(ev_threshold)
    df['value_away_ml'] = df['edge_away_ml_%'].ge(ev_threshold)
    df['value_any_ml'] = df['value_home_ml'] | df['value_away_ml']
    df['value_threshold_used'] = ev_threshold

    # First half and quarters suggested lines from team totals splits
    df['pred_1h_total'] = df['pred_home_1h'] + df['pred_away_1h']
    df['pred_2h_total'] = df['pred_home_2h'] + df['pred_away_2h']

    # Ranking for display: primary by absolute edge across markets
    df['max_edge_score'] = np.nanmax(
        np.vstack([
            df['units_spread'].fillna(0).values,
            df['units_total'].fillna(0).values,
            df['units_home_ml'].fillna(0).values,
            df['units_away_ml'].fillna(0).values,
        ]),
        axis=0,
    )

    # Confidence labels (Low/Medium/High) derived from suggested units
    def units_to_conf(u: float) -> str:
        try:
            if u is None or np.isnan(u) or u <= 0:
                return ''
            if u >= 1.0:
                return 'High'
            if u >= 0.5:
                return 'Medium'
            if u >= 0.25:
                return 'Low'
            return ''
        except Exception:
            return ''

    # Per-market confidence
    df['winner_confidence'] = np.vectorize(units_to_conf)(np.nanmax(np.vstack([
        df['units_home_ml'].fillna(0).values,
        df['units_away_ml'].fillna(0).values,
    ]), axis=0))
    df['spread_confidence'] = df['units_spread'].apply(units_to_conf)
    df['total_confidence'] = df['units_total'].apply(units_to_conf)
    # Overall game confidence as the max of any market
    df['game_confidence'] = np.vectorize(units_to_conf)(df['max_edge_score'])

    if filter_value_only:
        df = df[df['value_any_ml'] | (df['units_spread']>0) | (df['units_total']>0)].reset_index(drop=True)

    return df
