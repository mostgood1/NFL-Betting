import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _attach_team_ratings_to_view,
)

try:
    # Optional data sources for richer feature checks
    from nfl_compare.src.data_sources import load_team_stats, load_lines  # type: ignore
except Exception:
    load_team_stats = None  # type: ignore
    load_lines = None  # type: ignore


def _nan_report(df: pd.DataFrame, label: str, cols: list[str]) -> None:
    if df is None or df.empty:
        print(f"[{label}] empty or missing")
        return
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[{label}] no target columns present")
        return
    total = len(df)
    print(f"[{label}] NaN counts (of {total} rows):")
    for c in cols:
        na = int(pd.to_numeric(df[c], errors='coerce').isna().sum()) if df[c].dtype != 'O' else int(df[c].isna().sum())
        pct = (na / total * 100.0) if total else 0.0
        print(f"  {c}: {na} ({pct:.1f}%)")


def main():
    p = argparse.ArgumentParser(description="Audit feature completeness for a given season/week.")
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--week', type=int, required=True)
    args = p.parse_args()

    pred = _load_predictions()
    games = _load_games()
    view = _build_week_view(pred, games, args.season, args.week)
    view = _attach_model_predictions(view)
    view = _attach_team_ratings_to_view(view)

    print(f"Season {args.season} Week {args.week} — rows: {0 if view is None else len(view)}")
    if view is None or view.empty:
        return

    # Core predictions/market fields
    _nan_report(view, 'predictions/market', [
        'pred_total','pred_total_cal','pred_margin','prob_home_win',
        'market_spread_home','market_total','moneyline_home','moneyline_away'
    ])

    # Team ratings fields (should be 0-filled after patch)
    _nan_report(view, 'team_ratings', [
        'home_off_ppg','home_def_ppg','home_net_margin',
        'away_off_ppg','away_def_ppg','away_net_margin',
        'off_ppg_diff','def_ppg_diff','net_margin_diff'
    ])

    # If team stats are available, check attached feature columns
    if load_team_stats is not None:
        try:
            ts = load_team_stats()
        except Exception:
            ts = None
        # Probe for typical feature columns if present later in pipelines
        stats_like = [
            'home_off_epa','away_off_epa','off_epa_diff',
            'home_def_epa','away_def_epa','def_epa_diff',
            'home_pace_secs_play','away_pace_secs_play','pace_secs_play_diff',
            'home_pass_rate','away_pass_rate','pass_rate_diff',
            'home_rush_rate','away_rush_rate','rush_rate_diff',
            'home_qb_adj','away_qb_adj','qb_adj_diff',
            'home_sos','away_sos','sos_diff'
        ]
        present = [c for c in stats_like if c in view.columns]
        if present:
            _nan_report(view, 'team_stats', present)
        else:
            print('[team_stats] no attached stats columns present in view (OK if upstream attaches later).')

    # Weather-related fields if present
    _nan_report(view, 'weather', [
        'wx_temp_f','wx_wind_mph','wx_precip_pct','wx_precip_type','wx_sky','roof','surface','neutral_site'
    ])


if __name__ == '__main__':
    main()
