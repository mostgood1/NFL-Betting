"""
Augment games.csv with playoff games synthesized from lines/predictions.

This script is idempotent: it only adds missing playoff rows (by game_id),
preserving any existing rows in nfl_compare/data/games.csv.

Heuristics:
- Playoff weeks are treated as numeric weeks > 18 (Wild Card = 19, Divisional = 20,
  Conference = 21, Super Bowl = 22). If source already uses week numbers, we keep them.
- Sources: lines.csv, predictions.csv. We prefer rows that include game_id, season,
  week, home_team, away_team and a date (date/game_date). If date missing, we still
  write the row with an empty date.
- Neutral site is not forced here except for obvious Super Bowl identification by week 22.

Usage:
  python scripts/augment_playoffs_schedule.py [--season 2025]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR")) if os.environ.get("NFL_DATA_DIR") else (BASE_DIR / "nfl_compare" / "data")


def _read_csv_safe(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _normalize_core(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Standardize date field name
    if 'game_date' not in d.columns and 'date' in d.columns:
        d = d.rename(columns={'date': 'game_date'})
    # Coerce numeric season/week when present
    for c in ('season','week'):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def _derive_playoff_rows(source: pd.DataFrame, season: int | None) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame(columns=['season','week','game_id','game_date','home_team','away_team','home_score','away_score'])
    df = _normalize_core(source)
    # Filter to requested season if provided
    if season is not None and 'season' in df.columns:
        df = df[df['season'].eq(int(season))]
    # Consider weeks > 18 as playoff; if week missing, attempt by month (Jan/Feb)
    wk_col = 'week' if 'week' in df.columns else None
    if wk_col is None:
        # Try to infer by date month
        if 'game_date' in df.columns:
            dt = pd.to_datetime(df['game_date'], errors='coerce', utc=True)
            df['__is_playoff'] = dt.dt.month.isin([1,2])
        else:
            df['__is_playoff'] = False
    else:
        df['__is_playoff'] = pd.to_numeric(df['week'], errors='coerce').fillna(0) > 18

    keep_cols = [c for c in ['season','week','game_id','game_date','home_team','away_team','home_score','away_score'] if c in df.columns]
    if not keep_cols:
        return pd.DataFrame(columns=['season','week','game_id','game_date','home_team','away_team','home_score','away_score'])
    out = df[df['__is_playoff']][keep_cols].drop_duplicates().copy()
    # Ensure required columns exist
    for c in ['season','week','game_id','game_date','home_team','away_team','home_score','away_score']:
        if c not in out.columns:
            out[c] = pd.NA
    # If week is missing but inferred as playoff by date, map to calendar buckets (all games in same weekend share week)
    if out['week'].isna().any():
        try:
            out['__dt'] = pd.to_datetime(out['game_date'], errors='coerce', utc=True)
            # Derive a weekly bucket anchored to Sunday to group same-round games
            out['__bucket'] = out['__dt'].dt.to_period('W-SUN')
            # Sort by season, bucket, then teams for deterministic ordering
            out = out.sort_values(['season','__bucket','home_team','away_team'])
            for s in out['season'].dropna().unique():
                smask = out['season'].eq(s)
                buckets = (
                    out.loc[smask, '__bucket']
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
                # Map each bucket to playoff week starting at 19
                bmap = {b: (19 + i) for i, b in enumerate(sorted(buckets))}
                # Assign week per row based on bucket; leave NA if bucket unknown
                out.loc[smask & out['__bucket'].notna(), 'week'] = out.loc[smask & out['__bucket'].notna(), '__bucket'].astype(str).map(bmap)
            out = out.drop(columns=['__dt','__bucket'], errors='ignore')
        except Exception:
            # If date parsing fails, leave weeks as NA and upstream logic will handle
            pass
    # Coerce types
    for c in ('season','week'):
        out[c] = pd.to_numeric(out[c], errors='coerce').astype('Int64')
    return out


def augment_games_with_playoffs(season: int | None = None) -> tuple[int, int]:
    games_fp = DATA_DIR / 'games.csv'
    lines_fp = DATA_DIR / 'lines.csv'
    preds_fp = DATA_DIR / 'predictions.csv'

    games = _read_csv_safe(games_fp)
    lines = _read_csv_safe(lines_fp)
    preds = _read_csv_safe(preds_fp)

    base_cols = ['season','week','game_id','game_date','home_team','away_team','home_score','away_score']
    if games.empty:
        games = pd.DataFrame(columns=base_cols)
    games = _normalize_core(games)

    p_from_lines = _derive_playoff_rows(lines, season)
    p_from_preds = _derive_playoff_rows(preds, season)

    playoffs = pd.concat([p_from_lines, p_from_preds], ignore_index=True)
    playoffs = playoffs.drop_duplicates(subset=['game_id']).copy()
    # Filter to playoff week numbers >= 19
    if 'week' in playoffs.columns:
        playoffs = playoffs[pd.to_numeric(playoffs['week'], errors='coerce').fillna(0) >= 19]
    # Ensure columns
    for c in base_cols:
        if c not in playoffs.columns:
            playoffs[c] = pd.NA

    if playoffs.empty:
        return (0, 0)

    before = len(games)
    # Anti-dup by game_id
    if 'game_id' in games.columns and 'game_id' in playoffs.columns:
        existing = set(games['game_id'].astype(str).dropna().unique())
        add = playoffs[~playoffs['game_id'].astype(str).isin(existing)].copy()
    else:
        # If game_id missing in base, just union all (rare)
        add = playoffs.copy()

    if add.empty:
        return (0, 0)

    merged = pd.concat([games, add], ignore_index=True)
    # Reorder columns
    col_order = [c for c in base_cols if c in merged.columns] + [c for c in merged.columns if c not in base_cols]
    merged = merged[col_order]

    # Write atomically
    tmp = games_fp.with_suffix('.tmp')
    merged.to_csv(tmp, index=False)
    try:
        tmp.replace(games_fp)
    except Exception:
        import shutil
        shutil.move(str(tmp), str(games_fp))

    added = len(merged) - before
    return (added, len(merged))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, default=None, help='Limit augmentation to a single season')
    args = ap.parse_args(argv)
    added, total = augment_games_with_playoffs(args.season)
    print(f"Playoff augmentation complete: added={added}, total_games_rows={total}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
