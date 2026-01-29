"""Fetch and cache roster sources locally.

This repo needs week-accurate rosters/actives. We cache nfl_data_py tables under:
  nfl_compare/data/external/nfl_data_py/

What it fetches:
- seasonal_rosters_{season}
- weekly_rosters_{season}

These caches are then used by:
- nfl_compare/src/player_props.py (weekly actives map)
- nfl_compare/src/roster_validation.py (weekly roster validation)

Usage:
  python scripts/fetch_rosters_cache.py --season 2025
  python scripts/fetch_rosters_cache.py --season 2025 --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.roster_cache import CACHE_DIR, get_seasonal_rosters, get_weekly_rosters


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch roster caches (nfl_data_py) into nfl_compare/data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--refresh", action="store_true", help="Force refetch and overwrite cache")
    ap.add_argument("--timeout-sec", type=float, default=30.0)
    args = ap.parse_args()

    season = int(args.season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ros = get_seasonal_rosters(season, refresh=bool(args.refresh), timeout_sec=float(args.timeout_sec))
    wr = get_weekly_rosters(season, refresh=bool(args.refresh), timeout_sec=float(args.timeout_sec))

    print(f"Cache dir: {CACHE_DIR}")
    print(f"seasonal_rosters rows={0 if ros is None else len(ros)} cols={0 if ros is None else len(ros.columns)}")
    print(f"weekly_rosters rows={0 if wr is None else len(wr)} cols={0 if wr is None else len(wr.columns)}")

    # Quick sanity on weeks present
    if wr is not None and not wr.empty and "week" in wr.columns:
        w = pd.to_numeric(wr["week"], errors="coerce").dropna().astype(int)
        if not w.empty:
            print(f"weekly_rosters weeks: min={int(w.min())} max={int(w.max())} unique={int(w.nunique())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
