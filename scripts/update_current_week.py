"""
Update current_week.json to the upcoming NFL week based on games.csv.

Heuristic:
- Parse nfl_compare/data/games.csv, coerce date columns, and find the earliest
  upcoming game (>= today). Use its (season, week) as the current week.
- If no future games found but games exist, fall back to the max week in the
  latest season present (useful late Monday before next week's schedule appears).
- If games.csv missing/invalid, attempt predictions_week.csv/predictions.csv as fallback.

This keeps the app’s defaults auto-advancing without manual edits.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass
class Result:
    season: int
    week: int
    source: str


def _data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = os.environ.get("NFL_DATA_DIR")
    return Path(d).resolve() if d else (root / "nfl_compare" / "data")


def _read_games_csv(fp: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    # Normalize core fields
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Date columns to try in order
    date_cols = [
        "game_date",
        "date",
        "start_time",
        "kickoff",
        "datetime",
        "game_time",
    ]
    dt = None
    for c in date_cols:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dt.notna().any():
                    break
            except Exception:
                dt = None
    if dt is None:
        # No usable date column
        df["__dt"] = pd.NaT
    else:
        df["__dt"] = dt
    return df


def _infer_from_games(df: pd.DataFrame) -> Optional[Tuple[int, int, str]]:
    if df is None or df.empty:
        return None
    # Prefer rows with valid season/week
    if "season" not in df.columns or "week" not in df.columns:
        return None
    now = datetime.now(timezone.utc)
    # If we have dates, pick the earliest future game
    if "__dt" in df.columns and df["__dt"].notna().any():
        fut = df[df["__dt"] >= pd.Timestamp(now)]
        if not fut.empty:
            row = fut.sort_values("__dt", ascending=True).iloc[0]
            try:
                s = int(row["season"]) if pd.notna(row["season"]) else None
                w = int(row["week"]) if pd.notna(row["week"]) else None
                if s and w:
                    return s, w, "games.csv (future-by-date)"
            except Exception:
                pass
    # Fallback: latest season’s max week
    try:
        g = (
            df.dropna(subset=["season", "week"]) [["season", "week"]]
            .astype({"season": int, "week": int})
        )
        if not g.empty:
            latest_season = g["season"].max()
            max_week = g[g["season"] == latest_season]["week"].max()
            return int(latest_season), int(max_week), "games.csv (latest-season max week)"
    except Exception:
        pass
    return None


def _infer_from_predictions(dd: Path) -> Optional[Tuple[int, int, str]]:
    # Try predictions_week.csv first
    pw = dd / "predictions_week.csv"
    if pw.exists():
        try:
            df = pd.read_csv(pw)
            for c in ("season", "week"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["season", "week"])
            if not df.empty:
                latest_season = int(df["season"].max())
                max_week = int(df[df["season"] == latest_season]["week"].max())
                return latest_season, max_week, "predictions_week.csv"
        except Exception:
            pass
    # Then predictions.csv
    p = dd / "predictions.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            for c in ("season", "week"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["season", "week"])
            if not df.empty:
                latest_season = int(df["season"].max())
                max_week = int(df[df["season"] == latest_season]["week"].max())
                return latest_season, max_week, "predictions.csv"
        except Exception:
            pass
    return None


def infer_current_week() -> Optional[Result]:
    dd = _data_dir()
    # 1) Try games.csv with date awareness
    games_fp = dd / "games.csv"
    if games_fp.exists():
        got = _infer_from_games(_read_games_csv(games_fp))
        if got:
            s, w, src = got
            return Result(s, w, src)
    # 2) Fall back to predictions files
    got = _infer_from_predictions(dd)
    if got:
        s, w, src = got
        return Result(s, w, src)
    return None


def read_current_marker(dd: Path) -> Optional[Tuple[int, int]]:
    fp = dd / "current_week.json"
    if not fp.exists():
        return None
    try:
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        s = int(obj.get("season")) if obj.get("season") is not None else None
        w = int(obj.get("week")) if obj.get("week") is not None else None
        if s and w:
            return s, w
    except Exception:
        return None
    return None


def write_current_marker(dd: Path, s: int, w: int) -> None:
    fp = dd / "current_week.json"
    obj = {"season": int(s), "week": int(w)}
    tmp = fp.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(fp)


def main() -> int:
    dd = _data_dir()
    dd.mkdir(parents=True, exist_ok=True)
    inf = infer_current_week()
    if not inf:
        print("[update_current_week] Unable to infer current week; leaving marker unchanged.")
        return 0
    existing = read_current_marker(dd)
    if existing == (inf.season, inf.week):
        print(f"[update_current_week] current_week.json already set to season={inf.season} week={inf.week} (source: {inf.source})")
        return 0
    write_current_marker(dd, inf.season, inf.week)
    print(f"[update_current_week] Updated current_week.json -> season={inf.season} week={inf.week} (source: {inf.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
