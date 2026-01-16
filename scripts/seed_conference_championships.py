"""
Seed Conference Championship (week 21) games into games.csv for a given season.

Derives matchups from Divisional Round (week 20) winners, determines host by higher seed
using playoffs_seeds_{season}.json under nfl_compare/data.

Idempotent: skips if game_id already present or if winners are not yet known.

Usage:
  python scripts/seed_conference_championships.py --season 2025 --date 2026-01-25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from typing import Dict, Tuple, Optional
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


def _normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "game_date" not in d.columns and "date" in d.columns:
        d = d.rename(columns={"date": "game_date"})
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _load_seeds(season: int) -> Optional[Dict[str, Dict[str, int]]]:
    fp = DATA_DIR / f"playoffs_seeds_{season}.json"
    if not fp.exists():
        return None
    try:
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        afc = obj.get("AFC", {})
        nfc = obj.get("NFC", {})
        return {"AFC": {str(k): int(v) for k, v in afc.items()}, "NFC": {str(k): int(v) for k, v in nfc.items()}}
    except Exception:
        return None


def _winner(row: pd.Series) -> Optional[str]:
    hs = row.get("home_score")
    as_ = row.get("away_score")
    try:
        if pd.isna(hs) or pd.isna(as_):
            return None
        hs = float(hs)
        as_ = float(as_)
    except Exception:
        return None
    if hs > as_:
        return str(row.get("home_team"))
    if as_ > hs:
        return str(row.get("away_team"))
    return None


def _conference_of(team: str, seeds: Dict[str, Dict[str, int]]) -> Optional[str]:
    t = str(team)
    for conf in ("AFC", "NFC"):
        if t in seeds.get(conf, {}):
            return conf
    return None


def _host_by_seed(team_a: str, team_b: str, seeds: Dict[str, Dict[str, int]]) -> Tuple[str, str]:
    # Higher seed (lower number) hosts
    conf = _conference_of(team_a, seeds) or _conference_of(team_b, seeds)
    if not conf:
        # Fallback: keep alphabetical host
        return sorted([team_a, team_b])[0], sorted([team_a, team_b])[1]
    sa = seeds[conf].get(team_a)
    sb = seeds[conf].get(team_b)
    if sa is None or sb is None:
        return sorted([team_a, team_b])[0], sorted([team_a, team_b])[1]
    if sa < sb:
        return team_a, team_b
    else:
        return team_b, team_a


def seed_conference_championships(season: int, date_str: Optional[str] = None) -> Tuple[int, int]:
    seeds = _load_seeds(season)
    games = _normalize_games(_read_csv_safe(GAMES_FP))
    if games.empty or seeds is None:
        print("[seed_cc] Missing games.csv or seeds file; skipping")
        return 0, len(games)
    # Existing IDs
    have_ids = set()
    if "game_id" in games.columns:
        try:
            have_ids = set(games["game_id"].astype(str).dropna().unique().tolist())
        except Exception:
            have_ids = set()
    # Gather week 20 winners
    w20 = games[(games.get("season") == season) & (games.get("week") == 20)].copy()
    if w20.empty or len(w20) < 4:
        print("[seed_cc] Week 20 games not ready; skipping")
        return 0, len(games)
    winners_by_conf: Dict[str, list] = {"AFC": [], "NFC": []}
    for _, r in w20.iterrows():
        win = _winner(r)
        if not win:
            # require final scores to determine winners
            continue
        conf = _conference_of(win, seeds)
        if conf:
            winners_by_conf[conf].append(win)
    added_rows = []
    for conf in ("AFC", "NFC"):
        wins = winners_by_conf.get(conf, [])
        if len(wins) != 2:
            print(f"[seed_cc] {conf}: winners unresolved ({len(wins)} found); skipping")
            continue
        home, away = _host_by_seed(wins[0], wins[1], seeds)
        gid = f"{season}_21_{away.replace(' ', '_')}_{home.replace(' ', '_')}"
        if gid in have_ids:
            continue
        added_rows.append({
            "season": int(season),
            "week": int(21),
            "game_id": gid,
            "game_date": date_str or None,
            "home_team": home,
            "away_team": away,
            "home_score": pd.NA,
            "away_score": pd.NA,
        })
    if not added_rows:
        return 0, len(games)
    merged = pd.concat([games, pd.DataFrame(added_rows)], ignore_index=True)
    # Write atomically
    tmp = GAMES_FP.with_suffix('.tmp')
    merged.to_csv(tmp, index=False)
    try:
        tmp.replace(GAMES_FP)
    except Exception:
        import shutil
        shutil.move(str(tmp), str(GAMES_FP))
    return len(added_rows), len(merged)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed Conference Championships (week 21)")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--date", type=str, default="2026-01-25", help="YYYY-MM-DD for game_date")
    args = ap.parse_args(argv)
    added, total = seed_conference_championships(int(args.season), args.date)
    print(f"Seeded conference championships: added={added}, total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
