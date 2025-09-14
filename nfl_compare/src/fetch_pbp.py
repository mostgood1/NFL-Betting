"""
Fetch and persist NFL play-by-play (PBP) data using nfl-data-py.

Features
- Historical: save one Parquet file per season (e.g., data/pbp_2021.parquet).
- Append current: fetch a season (e.g., 2025), merge with existing file, and dedupe.

Examples (run from repo root or nfl_compare folder):
  python -m nfl_compare.src.fetch_pbp --historical-last 5 --exclude-current
  python -m nfl_compare.src.fetch_pbp --append-season 2025

Notes
- Requires: pip install nfl-data-py pyarrow
- Dedupe key: ['game_id', 'play_id'] when available; falls back to ['season','game_id','play_id'] or full-row drop_duplicates as last resort.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:  # pragma: no cover
    raise SystemExit(
        f"Missing nfl_data_py. Install with: pip install nfl-data-py pyarrow\nOriginal error: {e}"
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _seasons_last_n(n: int, include_current: bool = True) -> List[int]:
    year = datetime.now().year
    if include_current:
        start = year - (n - 1)
        return list(range(start, year + 1))
    else:
        # last n seasons before current year
        start = (year - 1) - (n - 1)
        return list(range(start, year))


def _write_parquet(df: pd.DataFrame, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    # Use snappy by default; fallback to default if codec missing
    try:
        df.to_parquet(fp, index=False)
    except Exception:
        df.to_parquet(fp, index=False, engine="pyarrow")


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer unique play key
    if {"game_id", "play_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["game_id", "play_id"]).reset_index(drop=True)
    if {"season", "game_id", "play_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["season", "game_id", "play_id"]).reset_index(drop=True)
    # Fallback to full-row
    return df.drop_duplicates().reset_index(drop=True)


def fetch_historical(seasons: Iterable[int]) -> List[Path]:
    out_paths: List[Path] = []
    for s in seasons:
        print(f"Fetching PBP for season {s}...")
        df = nfl.import_pbp_data([int(s)])
        fp = DATA_DIR / f"pbp_{s}.parquet"
        _write_parquet(df, fp)
        print(f"  Wrote {len(df):,} rows -> {fp}")
        out_paths.append(fp)
    return out_paths


def append_current(season: int) -> Path:
    """Fetch the season's PBP and append into a single file with deduping."""
    fp = DATA_DIR / f"pbp_{season}.parquet"
    print(f"Fetching current-season PBP for {season}...")
    df_new = nfl.import_pbp_data([int(season)])
    if fp.exists():
        try:
            df_old = pd.read_parquet(fp)
            print(f"  Existing file has {len(df_old):,} rows; merging...")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"  Warning: failed to read existing file, overwriting. Error: {e}")
            df_all = df_new
    else:
        df_all = df_new

    df_all = _dedupe(df_all)
    _write_parquet(df_all, fp)
    print(f"  Wrote {len(df_all):,} rows -> {fp}")
    return fp


def _parse_range(expr: str) -> List[int]:
    a, b = expr.split("-")
    sa, sb = int(a), int(b)
    if sa > sb:
        sa, sb = sb, sa
    return list(range(sa, sb + 1))


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fetch NFL PBP data and save to data folder")
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--historical-last", type=int, default=None, help="Fetch last N seasons")
    grp.add_argument("--seasons", nargs="*", type=int, help="Explicit list of seasons")
    grp.add_argument("--range", dest="range_", type=str, help="Range like 2020-2024")
    p.add_argument("--exclude-current", action="store_true", help="When using --historical-last, exclude current year")
    p.add_argument("--append-season", type=int, default=None, help="Append/merge a season into a single Parquet (e.g., 2025)")

    args = p.parse_args(argv)

    # Historical fetch
    hist_seasons: List[int] = []
    if args.historical_last is not None:
        hist_seasons = _seasons_last_n(args.historical_last, include_current=not args.exclude_current)
    elif args.seasons:
        hist_seasons = sorted(set(int(s) for s in args.seasons))
    elif args.range_:
        hist_seasons = _parse_range(args.range_)

    if hist_seasons:
        print(f"Historical seasons: {hist_seasons}")
        fetch_historical(hist_seasons)

    # Append current
    if args.append_season is not None:
        append_current(int(args.append_season))


if __name__ == "__main__":
    main()
