"""
Materialize team-level ratings CSV for a given season/week.

Usage:
  python scripts/build_team_ratings.py --season 2025 --week 11 [--alpha 0.6]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfl_compare.src.data_sources import load_games
from nfl_compare.src.team_ratings import write_team_ratings_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build team ratings artifact for a season/week.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args(argv)

    games = load_games()
    if games is None or games.empty:
        print("No games available; nothing to build.")
        return 0

    out = write_team_ratings_csv(games, int(args.season), int(args.week), alpha=float(args.alpha))
    print(f"Wrote team ratings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
