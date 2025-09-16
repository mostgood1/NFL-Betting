"""Reconcile schedule (games.csv) vs predictions (predictions & locked) for a target season/week.

Outputs a JSON-style text summary of:
- games_only: game_ids present in games but absent in predictions
- predictions_only: game_ids present in predictions but absent in games
- id_format_mismatch: heuristic cases where hyphen/underscore variants could map
- team_mismatch_rows: rows where (home,away) differ for same inferred normalized id core

Usage:
    python scripts/reconcile_schedule_vs_predictions.py --season 2025 --week 1

If no week provided, processes all weeks for the season.
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_games, _load_predictions  # reuse existing loaders


def _norm_game_id(gid: str | None) -> str | None:
    if gid is None:
        return None
    if not isinstance(gid, str):
        gid = str(gid)
    g = gid.strip()
    if not g:
        return None
    # unify separators to underscore
    g = g.replace('-', '_')
    # collapse multiple underscores
    while '__' in g:
        g = g.replace('__', '_')
    return g.lower()


def _core_from_id(gid: str | None) -> str | None:
    g = _norm_game_id(gid)
    if g is None:
        return None
    # assume format season_week_home_away
    parts = g.split('_')
    if len(parts) < 4:
        return g
    return '_'.join(parts[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, help='Specific week; if omitted, all weeks in games file considered.')
    args = ap.parse_args()

    games = _load_games() or pd.DataFrame()
    preds = _load_predictions() or pd.DataFrame()

    # Filter season/week
    if not games.empty and 'season' in games.columns:
        games = games[games['season'].astype(str) == str(args.season)]
    if args.week and 'week' in games.columns:
        games = games[games['week'].astype(str) == str(args.week)]

    if not preds.empty and 'season' in preds.columns:
        preds = preds[preds['season'].astype(str) == str(args.season)]
    if args.week and 'week' in preds.columns:
        preds = preds[preds['week'].astype(str) == str(args.week)]

    # Extract game ids
    games_ids = set(games['game_id'].dropna().astype(str)) if 'game_id' in games.columns else set()
    pred_ids = set(preds['game_id'].dropna().astype(str)) if 'game_id' in preds.columns else set()

    norm_games = {_norm_game_id(g): g for g in games_ids}
    norm_preds = {_norm_game_id(g): g for g in pred_ids}

    games_only = []
    predictions_only = []
    id_format_mismatch = []

    for ng, orig in norm_games.items():
        if ng not in norm_preds:
            games_only.append(orig)
        else:
            # If original differs in format (hyphen vs underscore)
            if orig != norm_preds[ng]:
                id_format_mismatch.append({'games': orig, 'predictions': norm_preds[ng]})

    for npid, orig in norm_preds.items():
        if npid not in norm_games:
            predictions_only.append(orig)

    # Team mismatch detection: compare (home,away) for overlapping core ids
    team_mismatch_rows = []
    if not games.empty and not preds.empty:
        g_small = games[['game_id','home_team','away_team']].copy() if all(c in games.columns for c in ['game_id','home_team','away_team']) else pd.DataFrame()
        p_small = preds[['game_id','home_team','away_team']].copy() if all(c in preds.columns for c in ['game_id','home_team','away_team']) else pd.DataFrame()
        if not g_small.empty and not p_small.empty:
            g_small['core'] = g_small['game_id'].apply(_core_from_id)
            p_small['core'] = p_small['game_id'].apply(_core_from_id)
            merged = g_small.merge(p_small, on='core', suffixes=('_games','_preds'))
            for _, r in merged.iterrows():
                if (r['home_team_games'] != r['home_team_preds']) or (r['away_team_games'] != r['away_team_preds']):
                    team_mismatch_rows.append({
                        'core': r['core'],
                        'games_home': r['home_team_games'], 'games_away': r['away_team_games'],
                        'preds_home': r['home_team_preds'], 'preds_away': r['away_team_preds'],
                    })

    summary = {
        'season': args.season,
        'week': args.week,
        'games_count': len(games_ids),
        'predictions_count': len(pred_ids),
        'games_only': sorted(games_only),
        'predictions_only': sorted(predictions_only),
        'id_format_mismatch': id_format_mismatch,
        'team_mismatch_rows': team_mismatch_rows,
    }

    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
