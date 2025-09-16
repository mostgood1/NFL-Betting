"""Normalize games.csv by merging with predictions_locked.csv to ensure complete weekly coverage.

This script will:
 1. Load existing games.csv (if present) and predictions_locked.csv.
 2. For a target season/week (or all weeks) build a union of game rows using locked predictions as authoritative
    when a game_id / matchup is missing from games.csv.
 3. Write/merge into games_normalized.csv (idempotent replacement of rows for processed weeks).

Usage:
  python scripts/normalize_games_from_locked.py --season 2025 --week 1
  python scripts/normalize_games_from_locked.py --season 2025 --all-weeks

Result file columns (subset preserved): season, week, game_id, game_date, home_team, away_team
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'nfl_compare' / 'data'
GAMES_FILE = DATA_DIR / 'games.csv'
LOCKED_FILE = DATA_DIR / 'predictions_locked.csv'
NORMALIZED_FILE = DATA_DIR / 'games_normalized.csv'


KEEP_COLS = ['season','week','game_id','game_date','date','home_team','away_team']


def _load_csv(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _filter_sw(df: pd.DataFrame, season: int, week: int | None):
    out = df.copy()
    if 'season' in out.columns:
        out = out[pd.to_numeric(out['season'], errors='coerce') == season]
    if week is not None and 'week' in out.columns:
        out = out[pd.to_numeric(out['week'], errors='coerce') == week]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, help='Single week to normalize')
    ap.add_argument('--all-weeks', action='store_true', help='Process all weeks present in locked predictions for season')
    args = ap.parse_args()

    games = _load_csv(GAMES_FILE)
    locked = _load_csv(LOCKED_FILE)

    if locked.empty:
        print('No locked predictions file found; nothing to do.')
        return

    # Determine weeks
    if args.all_weeks:
        weeks = sorted({int(w) for w in pd.to_numeric(locked['week'], errors='coerce').dropna().unique() if int(w) > 0})
    else:
        if args.week is None:
            print('Provide --week or --all-weeks')
            return
        weeks = [args.week]

    season = args.season
    locked['season'] = pd.to_numeric(locked['season'], errors='coerce')
    locked['week'] = pd.to_numeric(locked['week'], errors='coerce')

    norm_rows = []
    for wk in weeks:
        lock_sub = locked[(locked['season'] == season) & (locked['week'] == wk)].copy()
        if lock_sub.empty:
            print(f'Season {season} Week {wk}: no locked rows; skipping')
            continue
        # Select minimal columns
        subset_cols = [c for c in KEEP_COLS if c in lock_sub.columns]
        base = lock_sub[subset_cols].copy()
        if 'game_date' not in base.columns and 'date' in base.columns:
            base = base.rename(columns={'date':'game_date'})
        # Standardize required columns
        for c in ['season','week','home_team','away_team']:
            if c not in base.columns:
                base[c] = None
        # Drop potential duplicates
        base = base.drop_duplicates(subset=['game_id'] if 'game_id' in base.columns else None)
        base['source'] = 'locked'

        # Existing games rows for same keys
        if not games.empty:
            exist_sub = _filter_sw(games, season, wk)
            if not exist_sub.empty:
                exist_cols = [c for c in KEEP_COLS if c in exist_sub.columns]
                exist_base = exist_sub[exist_cols].copy()
                if 'game_date' not in exist_base.columns and 'date' in exist_base.columns:
                    exist_base = exist_base.rename(columns={'date':'game_date'})
                for c in ['season','week','home_team','away_team']:
                    if c not in exist_base.columns:
                        exist_base[c] = None
                exist_base['source'] = 'games'
                # Keep union; prefer explicit games row for overlapping game_id
                if 'game_id' in base.columns and 'game_id' in exist_base.columns:
                    overlap = set(base['game_id']) & set(exist_base['game_id'])
                    if overlap:
                        # replace locked with games for overlapping ids
                        base = pd.concat([
                            base[~base['game_id'].isin(overlap)],
                            exist_base[exist_base['game_id'].isin(overlap)]
                        ], ignore_index=True)
                    else:
                        base = pd.concat([exist_base, base], ignore_index=True)
        norm_rows.append(base)
        print(f'Season {season} Week {wk}: normalized rows {len(base)}')

    if not norm_rows:
        print('No rows produced.')
        return

    out_df = pd.concat(norm_rows, ignore_index=True)

    # Merge into existing normalized file if present (replace weeks processed)
    if NORMALIZED_FILE.exists():
        try:
            prior = pd.read_csv(NORMALIZED_FILE)
            prior['season'] = pd.to_numeric(prior['season'], errors='coerce')
            prior['week'] = pd.to_numeric(prior['week'], errors='coerce')
            prior = prior[~((prior['season'] == season) & (prior['week'].isin(out_df['week'].unique())))]
            out_df = pd.concat([prior, out_df], ignore_index=True)
        except Exception:
            pass

    # Final column ordering
    final_cols = ['season','week','game_id','game_date','home_team','away_team','source'] + [c for c in out_df.columns if c not in ['season','week','game_id','game_date','home_team','away_team','source']]
    out_df = out_df[final_cols]
    NORMALIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(NORMALIZED_FILE, index=False)
    print(f'Wrote {len(out_df)} rows to {NORMALIZED_FILE}')

if __name__ == '__main__':
    main()
