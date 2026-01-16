"""
Check playoff schedule coverage in games.csv for the current season.

Verifies presence and expected counts for weeks 19–22 (WC, Divisional, Conference, Super Bowl).
Prints a concise summary and returns 0 always (logging aid), unless --strict is set.

Usage:
  python scripts/check_playoff_schedule.py --season 2025 [--strict]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR")) if os.environ.get("NFL_DATA_DIR") else (BASE_DIR / "nfl_compare" / "data")
GAMES_FP = DATA_DIR / "games.csv"

EXPECTED_COUNTS = {
    19: 6,  # Wild Card
    20: 4,  # Divisional
    21: 2,  # Conference Championships
    22: 1,  # Super Bowl
}


def _read_csv_safe(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "game_date" not in d.columns and "date" in d.columns:
        d = d.rename(columns={"date": "game_date"})
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def check(season: int | None, strict: bool = False) -> int:
    games = normalize_games(_read_csv_safe(GAMES_FP))
    if games.empty:
        print("[check_playoff_schedule] games.csv is empty")
        return 1 if strict else 0
    if season is not None:
        games = games[games.get("season").eq(int(season))]
    # Filter to playoff weeks
    pl = games[pd.to_numeric(games.get("week"), errors="coerce").fillna(0) >= 19]
    # Compute counts per week
    counts = pl.groupby("week").size().to_dict() if not pl.empty else {}
    # Summary
    missing = []
    wrong_counts = []
    for wk, exp in EXPECTED_COUNTS.items():
        got = int(counts.get(wk, 0))
        print(f"[check_playoff_schedule] Week {wk}: {got} rows (expected {exp})")
        if got == 0:
            missing.append(wk)
        elif exp > 0 and got != exp:
            wrong_counts.append((wk, got, exp))
    if missing:
        print(f"[check_playoff_schedule] Missing playoff weeks: {','.join(str(w) for w in missing)}")
        print("[check_playoff_schedule] Suggest: run scripts/augment_playoffs_schedule.py and seed lines/predictions for future rounds.")
    if wrong_counts:
        for wk, got, exp in wrong_counts:
            print(f"[check_playoff_schedule] Week {wk} count mismatch: got {got}, expected {exp}")
        print("[check_playoff_schedule] If duplicates exist, run scripts/normalize_playoffs_week.py with --max-games to cap.")
    ok = not missing and not wrong_counts
    print(f"[check_playoff_schedule] Coverage OK: {ok}")
    return 0 if (ok or not strict) else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    return check(args.season, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
