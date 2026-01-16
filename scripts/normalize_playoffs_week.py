"""
Normalize playoff week games to unique matchups and cap to a target count.

Usage:
  python scripts/normalize_playoffs_week.py --season 2025 --week 19 --max-games 6

Behavior:
- Reads nfl_compare/data/games.csv
- For the specified (season, week), dedupes rows by matchup (home_team|away_team)
  preferring earliest scheduled date/time when available.
- If more than --max-games unique matchups exist, keeps the earliest N matchups
  and drops the rest.
- Writes back to games.csv atomically.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR")) if os.environ.get("NFL_DATA_DIR") else (BASE_DIR / "nfl_compare" / "data")
GAMES_FP = DATA_DIR / "games.csv"


def _read_csv_safe(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--max-games", type=int, default=6)
    args = ap.parse_args()

    games = _read_csv_safe(GAMES_FP)
    if games.empty:
        print("games.csv is empty; nothing to normalize")
        return 0

    # Standardize date field
    df = games.copy()
    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    sw = df[(df.get("season") == args.season) & (df.get("week") == args.week)].copy()
    if sw.empty:
        print(f"No rows for season={args.season} week={args.week}; nothing to change")
        return 0

    # Build matchup key and sort by date
    sw["__match_key"] = sw["home_team"].astype(str) + "|" + sw["away_team"].astype(str)
    if "game_date" in sw.columns:
        sw["__dt"] = pd.to_datetime(sw["game_date"], errors="coerce")
    else:
        sw["__dt"] = pd.NaT
    sw = sw.sort_values(["__dt", "home_team", "away_team"])  # earliest first

    # Deduplicate by matchup, keeping earliest
    unique = sw.drop_duplicates(subset=["__match_key"], keep="first").copy()
    # Cap to max-games
    if len(unique) > int(args.max_games):
        unique = unique.iloc[: int(args.max_games)].copy()

    # Recombine with non-target rows
    keep_ids = set(unique["game_id"].astype(str)) if "game_id" in unique.columns else set()
    rest = df[~((df.get("season") == args.season) & (df.get("week") == args.week))].copy()
    out = pd.concat([rest, unique.drop(columns=["__match_key", "__dt"], errors="ignore")], ignore_index=True)

    # Write atomically
    tmp = GAMES_FP.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    try:
        tmp.replace(GAMES_FP)
    except Exception:
        import shutil
        shutil.move(str(tmp), str(GAMES_FP))

    print(f"Normalized season={args.season} week={args.week}: kept {len(unique)} matchups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
