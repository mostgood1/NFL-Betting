"""Generate synthetic market-based predictions for games that lack model outputs.

For each (season, week) specified, this script will:
 1. Load games, predictions, and lines data.
 2. Identify games present in games/lines but missing pred_home_points/pred_total.
 3. Derive synthetic predictions using the same logic as _derive_predictions_from_market.
 4. Append rows (with prediction_source='market_backfill') into a predictions_synth.csv file
    stored alongside existing prediction files.

Usage:
  python scripts/backfill_missing_week_predictions.py --season 2025 --week 1
  python scripts/backfill_missing_week_predictions.py --season 2025 --all-weeks

The resulting file predictions_synth.csv can be safely deleted/regenerated; it is additive and
non-destructive to original model predictions.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_games, _load_predictions, _derive_predictions_from_market

# Try to load lines lazily
try:
    from nfl_compare.src.data_sources import load_lines as _load_lines
except Exception:  # pragma: no cover
    _load_lines = lambda: pd.DataFrame()

DATA_DIR = Path('nfl_compare/data')
SYNTH_FILE = DATA_DIR / 'predictions_synth.csv'


def _filter_sw(df: pd.DataFrame, season: int | None, week: int | None) -> pd.DataFrame:
    out = df
    if season is not None and 'season' in out.columns:
        out = out[out['season'].astype(str) == str(season)]
    if week is not None and 'week' in out.columns:
        out = out[out['week'].astype(str) == str(week)]
    return out


def _derive_for_subset(base_rows: pd.DataFrame) -> pd.DataFrame:
    # Reuse helper; it expects absence of prediction columns to create them.
    out = _derive_predictions_from_market(base_rows)
    # Tag backfill variant (override source values for rows we produced)
    if 'prediction_source' not in out.columns:
        out['prediction_source'] = None
    mask = out.get('derived_from_market')
    if mask is not None:
        out.loc[mask == True, 'prediction_source'] = 'market_backfill'  # noqa: E712
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, help='Single week to backfill')
    ap.add_argument('--all-weeks', action='store_true', help='Process all weeks present in games file for the season')
    args = ap.parse_args()

    gtmp = _load_games()
    games = gtmp if gtmp is not None else pd.DataFrame()
    ptmp = _load_predictions()
    preds = ptmp if ptmp is not None else pd.DataFrame()
    ltmp = _load_lines()
    lines = ltmp if ltmp is not None else pd.DataFrame()

    # Determine weeks list
    if args.all_weeks:
        weeks = sorted({int(w) for w in games.loc[games['season'] == args.season, 'week'].dropna().unique()}) if not games.empty else []
    else:
        weeks = [args.week] if args.week is not None else []
    if not weeks:
        print('No weeks to process (provide --week or --all-weeks).')
        return

    synth_rows = []

    for wk in weeks:
        # Base candidate rows: from games or lines
        base = pd.DataFrame()
        g_sub = _filter_sw(games, args.season, wk)
        if not g_sub.empty:
            base = g_sub.copy()
        else:
            l_sub = _filter_sw(lines, args.season, wk)
            if not l_sub.empty:
                keep_cols = [c for c in ['season','week','game_id','game_date','date','home_team','away_team'] if c in l_sub.columns]
                if keep_cols:
                    base = l_sub[keep_cols].drop_duplicates()
                    if 'game_date' not in base.columns and 'date' in base.columns:
                        base = base.rename(columns={'date':'game_date'})
        if base.empty:
            print(f'Season {args.season} Week {wk}: no base rows found, skipping.')
            continue

        # Attach any existing prediction cols from model (to avoid overwriting)
        if not preds.empty:
            p_sub = _filter_sw(preds, args.season, wk)
            if not p_sub.empty and 'game_id' in base.columns and 'game_id' in p_sub.columns:
                core_keys = {'season','week','game_id','game_date','date','home_team','away_team','home_score','away_score'}
                pcols = [c for c in p_sub.columns if c not in core_keys]
                try:
                    right = p_sub[['game_id'] + pcols].drop_duplicates()
                    base = base.merge(right, on='game_id', how='left')
                except Exception:
                    pass

        # If already has prediction columns with any values skip derivation
        existing_pred = any(col in base.columns and base[col].notna().any() for col in ['pred_home_points','pred_total','pred_margin'])
        if existing_pred:
            print(f'Season {args.season} Week {wk}: model predictions already present; nothing to backfill.')
            continue

        derived = _derive_for_subset(base)
        # Only keep rows where we produced a total
        if 'pred_total' in derived.columns:
            produced = derived[derived['pred_total'].notna()].copy()
        else:
            produced = pd.DataFrame()
        if produced.empty:
            print(f'Season {args.season} Week {wk}: no synthetic predictions produced.')
            continue
        synth_rows.append(produced)
        print(f'Season {args.season} Week {wk}: produced {len(produced)} synthetic rows.')

    if not synth_rows:
        print('No synthetic rows to write.')
        return

    out_df = pd.concat(synth_rows, ignore_index=True)
    # Minimal column ordering preference
    preferred = ['season','week','game_id','home_team','away_team','pred_home_points','pred_away_points','pred_total','pred_margin','prob_home_win','prediction_source','derived_from_market']
    cols = preferred + [c for c in out_df.columns if c not in preferred]
    out_df = out_df[cols]

    # If file exists, append (dedupe by (season,week,game_id))
    if SYNTH_FILE.exists():
        try:
            prior = pd.read_csv(SYNTH_FILE)
            merged = pd.concat([prior, out_df], ignore_index=True)
            if all(c in merged.columns for c in ['season','week','game_id']):
                merged = merged.drop_duplicates(subset=['season','week','game_id'], keep='last')
            out_df = merged
        except Exception:
            pass

    SYNTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(SYNTH_FILE, index=False)
    print(f'Wrote synthetic predictions to {SYNTH_FILE} ({len(out_df)} total rows).')

if __name__ == '__main__':
    main()
