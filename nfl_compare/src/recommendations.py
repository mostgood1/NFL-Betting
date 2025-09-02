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


def make_recommendations(df_pred: pd.DataFrame) -> pd.DataFrame:
    df = df_pred.copy()

    # Moneyline edges
    dec_home = df['moneyline_home'].apply(american_to_decimal) if 'moneyline_home' in df else np.nan
    dec_away = df['moneyline_away'].apply(american_to_decimal) if 'moneyline_away' in df else np.nan

    df['fair_home_ml'] = df['prob_home_win'].clip(1e-4, 1-1e-4).apply(prob_to_american)
    df['fair_away_ml'] = (1 - df['prob_home_win']).clip(1e-4, 1-1e-4).apply(prob_to_american)

    # Convert fair odds back to decimal for EV calc
    df['fair_home_dec'] = df['prob_home_win'].apply(lambda p: 1/(1-p) if p>=0.5 else 1/p)
    df['fair_away_dec'] = (1 - df['prob_home_win']).apply(lambda p: 1/(1-p) if p>=0.5 else 1/p)

    # If book odds present, compute edge in pct vs fair
    if isinstance(dec_home, pd.Series):
        df['edge_home_ml_%'] = (df['fair_home_dec'] - dec_home) / dec_home * 100
        df['edge_away_ml_%'] = (df['fair_away_dec'] - dec_away) / dec_away * 100
    else:
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
    df['units_home_ml'] = df['edge_home_ml_%'].apply(lambda e: edge_to_units(e/100*3))  # scale
    df['units_away_ml'] = df['edge_away_ml_%'].apply(lambda e: edge_to_units(e/100*3))

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

    return df
