from __future__ import annotations

"""
Seed playoff games into games.csv from simple inputs.

Options:
  - --pairs "AWY@HOME,AWY2@HOME2"  (comma-separated)
  - --csv path/to/pairs.csv         (columns: home_team, away_team, [game_date])
  - --date YYYY-MM-DD               (applies to all pairs when per-row date missing)

This is idempotent by game_id. Game IDs are generated as
  f"{season}_{week}_{away}_{home}"

Usage:
  python scripts/seed_playoff_games.py --season 2025 --week 19 \
    --pairs "PIT@KC,GB@DAL,LAR@PHI,MIA@BUF"

Or using CSV:
  python scripts/seed_playoff_games.py --season 2025 --week 19 \
    --csv nfl_compare/data/playoffs_pairs_template.csv --date 2026-01-11
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from typing import List, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR")) if os.environ.get("NFL_DATA_DIR") else (BASE_DIR / "nfl_compare" / "data")


def _read_csv_safe(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _normalize_team(s: str) -> str:
    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm  # type: ignore
        return _norm(str(s))
    except Exception:
        return str(s).strip().upper()


def _parse_pairs_arg(arg: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not arg:
        return out
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        if "@" not in p:
            continue
        away, home = [x.strip() for x in p.split("@", 1)]
        out.append((_normalize_team(away), _normalize_team(home)))
    return out


def _pairs_from_csv(fp: Path) -> List[Tuple[str, str, str | None]]:
    df = _read_csv_safe(fp)
    if df is None or df.empty:
        return []
    cols = [c.lower() for c in df.columns]
    ren = {c: c.lower() for c in df.columns}
    d = df.rename(columns=ren)
    if not {"home_team","away_team"}.issubset(set(cols)):
        return []
    out: List[Tuple[str, str, str | None]] = []
    for _, r in d.iterrows():
        home = _normalize_team(r.get("home_team"))
        away = _normalize_team(r.get("away_team"))
        gd = r.get("game_date")
        gd_str = str(gd) if isinstance(gd, str) else None
        out.append((away, home, gd_str))
    return out


def seed(season: int, week: int, pairs: List[Tuple[str, str]], date_default: str | None = None, csv_pairs: Path | None = None) -> Tuple[int, int]:
    games_fp = DATA_DIR / "games.csv"
    games = _read_csv_safe(games_fp)
    if games is None or games.empty:
        games = pd.DataFrame(columns=["season","week","game_id","game_date","home_team","away_team","home_score","away_score"])
    # Normalize core
    if "date" in games.columns and "game_date" not in games.columns:
        games = games.rename(columns={"date":"game_date"})
    # Existing IDs
    have_ids = set()
    if "game_id" in games.columns:
        try:
            have_ids = set(games["game_id"].astype(str).dropna().unique().tolist())
        except Exception:
            have_ids = set()

    rows = []
    # From csv
    if csv_pairs is not None:
        triples = _pairs_from_csv(csv_pairs)
        for away, home, gd in triples:
            gid = f"{season}_{week}_{away}_{home}"
            if gid in have_ids:
                continue
            rows.append({
                "season": int(season),
                "week": int(week),
                "game_id": gid,
                "game_date": gd if gd else (date_default or None),
                "home_team": home,
                "away_team": away,
                "home_score": pd.NA,
                "away_score": pd.NA,
            })
    # From --pairs
    for away, home in pairs:
        gid = f"{season}_{week}_{away}_{home}"
        if gid in have_ids:
            continue
        rows.append({
            "season": int(season),
            "week": int(week),
            "game_id": gid,
            "game_date": date_default or None,
            "home_team": home,
            "away_team": away,
            "home_score": pd.NA,
            "away_score": pd.NA,
        })

    if not rows:
        return (0, len(games))

    add = pd.DataFrame(rows)
    merged = pd.concat([games, add], ignore_index=True)
    # Write atomically
    tmp = games_fp.with_suffix('.tmp')
    merged.to_csv(tmp, index=False)
    try:
        tmp.replace(games_fp)
    except Exception:
        import shutil
        shutil.move(str(tmp), str(games_fp))
    return (len(add), len(merged))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed playoff games into games.csv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--pairs", type=str, default=None, help="Comma-separated AWAY@HOME list")
    ap.add_argument("--csv", type=str, default=None, help="CSV file with home_team, away_team, [game_date]")
    ap.add_argument("--date", type=str, default=None, help="Default YYYY-MM-DD for game_date when missing")
    args = ap.parse_args(argv)

    pairs: List[Tuple[str, str]] = _parse_pairs_arg(args.pairs) if args.pairs else []
    csv_path: Path | None = Path(args.csv) if args.csv else None
    added, total = seed(int(args.season), int(args.week), pairs, args.date, csv_path)
    print(f"Seeded playoff games: added={added}, total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
