from __future__ import annotations

import pandas as pd


def _shrink(series: pd.Series, league_mean: float, n: pd.Series, alpha: float) -> pd.Series:
    """Empirical Bayes shrinkage toward league mean with weight alpha.
    series: per-team average last season
    n: number of observations (weeks)
    alpha: pseudo-count (weeks)
    """
    s = pd.to_numeric(series, errors='coerce')
    nn = pd.to_numeric(n, errors='coerce').fillna(0)
    return (nn * s + alpha * league_mean) / (nn + alpha).where((nn + alpha) != 0, other=1)


def compute_team_priors(team_stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute per-team priors for the given season using previous season stats.

        Returns columns:
            ['season','team','off_epa_prior','def_epa_prior','pace_prior','pass_rate_prior','rush_rate_prior','sos_prior','qb_prior','continuity_flag']
    """
    prev = season - 1
    if team_stats is None or team_stats.empty:
        return pd.DataFrame(columns=[
            'season','team','off_epa_prior','def_epa_prior','pace_prior','pass_rate_prior','rush_rate_prior','sos_prior','qb_prior','continuity_flag'
        ])
    ts = team_stats[team_stats['season'] == prev].copy()
    if ts.empty:
        return pd.DataFrame(columns=[
            'season','team','off_epa_prior','def_epa_prior','pace_prior','pass_rate_prior','rush_rate_prior','sos_prior','qb_prior','continuity_flag'
        ])

    grp = ts.groupby('team', as_index=False).agg(
        off_epa_avg=('off_epa', 'mean'),
        def_epa_avg=('def_epa', 'mean'),
        pace_avg=('pace_secs_play', 'mean'),
        pass_rate_avg=('pass_rate', 'mean'),
        rush_rate_avg=('rush_rate', 'mean'),
        sos_avg=('sos', 'mean'),
        weeks=('week', 'nunique'),
    )

    # League means for shrinkage
    lm_off = pd.to_numeric(grp['off_epa_avg'], errors='coerce').mean()
    lm_def = pd.to_numeric(grp['def_epa_avg'], errors='coerce').mean()
    lm_pace = pd.to_numeric(grp['pace_avg'], errors='coerce').mean()
    lm_pass = pd.to_numeric(grp['pass_rate_avg'], errors='coerce').mean()
    lm_rush = pd.to_numeric(grp['rush_rate_avg'], errors='coerce').mean()
    lm_sos = pd.to_numeric(grp['sos_avg'], errors='coerce').fillna(0).mean()

    # Pseudo-counts (in weeks)
    a_epa = 8.0
    a_pace = 6.0
    a_rate = 4.0
    a_sos = 6.0

    out = pd.DataFrame({
        'season': season,
        'team': grp['team'],
        'off_epa_prior': _shrink(grp['off_epa_avg'], lm_off, grp['weeks'], a_epa),
        'def_epa_prior': _shrink(grp['def_epa_avg'], lm_def, grp['weeks'], a_epa),
        'pace_prior': _shrink(grp['pace_avg'], lm_pace, grp['weeks'], a_pace),
        'pass_rate_prior': _shrink(grp['pass_rate_avg'], lm_pass, grp['weeks'], a_rate),
        'rush_rate_prior': _shrink(grp['rush_rate_avg'], lm_rush, grp['weeks'], a_rate),
        'sos_prior': _shrink(grp['sos_avg'].fillna(lm_sos), lm_sos, grp['weeks'], a_sos),
        # simple QB prior proxy and continuity default (1 = stable)
        'qb_prior': _shrink(grp['off_epa_avg'], lm_off, grp['weeks'], a_epa) * _shrink(grp['pass_rate_avg'], lm_pass, grp['weeks'], a_rate),
        'continuity_flag': 1.0,
    })

    return out
