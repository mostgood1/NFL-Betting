"""Update nfl_compare/data/games.csv with final scores from nfl_data_py schedules.

Why this exists:
- Our playoff schedule rows may be seeded ahead of time with scores missing.
- Once games finalize, we want to populate home_score/away_score so backtests and UI can reconcile.

This script is defensive and only updates rows it can confidently match by:
  (season, week, game_date, home_team, away_team)
where teams are normalized via nfl_compare.src.team_normalizer.normalize_team_name.

Usage:
  python scripts/update_games_scores_from_nfl_data_py.py --season 2025 --week 20
  python scripts/update_games_scores_from_nfl_data_py.py --season 2025 --week 20 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "nfl_compare" / "data"
GAMES_FP = DATA_DIR / "games.csv"


def _coerce_date_str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        # Keep date-only string for matching
        return str(pd.to_datetime(s, errors="coerce").date())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill in games.csv scores from nfl_data_py schedules")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not GAMES_FP.exists():
        raise SystemExit(f"Missing {GAMES_FP}")

    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception as e:
        raise SystemExit(f"nfl-data-py required: pip install nfl-data-py\n{e}")

    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
    except Exception:
        def _norm_team(s: str) -> str:
            return str(s or "").strip()

    games = pd.read_csv(GAMES_FP)
    if games is None or games.empty:
        print("games.csv is empty; nothing to update")
        return

    # Filter to target week
    if "season" not in games.columns or "week" not in games.columns:
        raise SystemExit("games.csv missing season/week columns")

    g = games.copy()
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    g = g[(g["season"] == int(args.season)) & (g["week"] == int(args.week))].copy()
    if g.empty:
        print(f"No games.csv rows found for season={args.season} week={args.week}")
        return

    # Determine which date column to use
    date_col = "game_date" if "game_date" in g.columns else ("date" if "date" in g.columns else None)
    if date_col is None:
        raise SystemExit("games.csv missing game_date/date")

    for col in ("home_team", "away_team"):
        if col in g.columns:
            g[col] = g[col].astype(str).apply(_norm_team)

    for col in ("home_score", "away_score"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    g["_date"] = g[date_col].map(_coerce_date_str)

    sched = nfl.import_schedules([int(args.season)])
    if sched is None or len(sched) == 0:
        raise SystemExit("No schedule rows returned by nfl_data_py")

    s = sched.copy()
    if "season" in s.columns:
        s = s[s["season"] == int(args.season)].copy()
    if "week" not in s.columns:
        raise SystemExit("Schedule missing week column")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s = s[s["week"] == int(args.week)].copy()

    # Use gameday for robust date match
    if "gameday" in s.columns:
        s["_date"] = s["gameday"].map(_coerce_date_str)
    elif "game_date" in s.columns:
        s["_date"] = s["game_date"].map(_coerce_date_str)
    else:
        s["_date"] = None

    for col in ("home_team", "away_team"):
        if col in s.columns:
            s[col] = s[col].astype(str).apply(_norm_team)

    for col in ("home_score", "away_score"):
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce")

    need_cols = {"home_team", "away_team", "home_score", "away_score", "_date"}
    if not need_cols.issubset(set(s.columns)):
        raise SystemExit(f"Schedule missing required columns: {sorted(need_cols - set(s.columns))}")

    # Keep only finals we can actually write
    s_done = s[s["home_score"].notna() & s["away_score"].notna()].copy()
    if s_done.empty:
        print(f"No completed schedule games found for season={args.season} week={args.week}")
        return

    # Deduplicate schedule in case of provider duplicates
    s_done = s_done.sort_values(["_date", "home_team", "away_team"]).drop_duplicates(subset=["_date", "home_team", "away_team"], keep="last")

    # Join
    join_keys = ["_date", "home_team", "away_team"]
    g2 = g.merge(
        s_done[join_keys + ["home_score", "away_score"]].rename(columns={"home_score": "home_score_sched", "away_score": "away_score_sched"}),
        on=join_keys,
        how="left",
    )

    upd_mask = g2["home_score_sched"].notna() & g2["away_score_sched"].notna()
    if "home_score" in g2.columns and "away_score" in g2.columns:
        upd_mask = upd_mask & (g2["home_score"].isna() | g2["away_score"].isna() | (g2["home_score"] != g2["home_score_sched"]) | (g2["away_score"] != g2["away_score_sched"]))

    to_update = g2[upd_mask].copy()
    print(f"Matched completed schedule games: {int(upd_mask.sum())} / {len(g2)}")
    if not to_update.empty:
        show_cols = [c for c in ["game_id", date_col, "home_team", "away_team", "home_score", "away_score", "home_score_sched", "away_score_sched"] if c in to_update.columns]
        print(to_update[show_cols].to_string(index=False))

    if args.dry_run:
        print("dry-run: not writing games.csv")
        return

    if to_update.empty:
        print("No score updates needed")
        return

    # Apply updates back to original games.csv (by game_id when possible; else by join keys)
    out = games.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")

    # Build lookup from join key -> scores
    upd_map = {}
    for _, r in to_update.iterrows():
        key = (r.get("_date"), r.get("home_team"), r.get("away_team"))
        upd_map[key] = (float(r["home_score_sched"]), float(r["away_score_sched"]))

    # Apply
    date_col_out = "game_date" if "game_date" in out.columns else ("date" if "date" in out.columns else None)
    if date_col_out is None:
        raise SystemExit("games.csv missing game_date/date")

    def _row_key(row) -> tuple:
        return (_coerce_date_str(row.get(date_col_out)), _norm_team(str(row.get("home_team") or "")), _norm_team(str(row.get("away_team") or "")))

    changed = 0
    for idx, row in out.iterrows():
        if int(pd.to_numeric(row.get("season"), errors="coerce") or -1) != int(args.season):
            continue
        if int(pd.to_numeric(row.get("week"), errors="coerce") or -1) != int(args.week):
            continue
        k = _row_key(row)
        if k in upd_map:
            hs, as_ = upd_map[k]
            prev_h = row.get("home_score")
            prev_a = row.get("away_score")
            # only write when different or missing
            try:
                prev_hf = float(prev_h) if pd.notna(prev_h) else None
            except Exception:
                prev_hf = None
            try:
                prev_af = float(prev_a) if pd.notna(prev_a) else None
            except Exception:
                prev_af = None
            if prev_hf != hs or prev_af != as_:
                out.at[idx, "home_score"] = hs
                out.at[idx, "away_score"] = as_
                changed += 1

    if changed == 0:
        print("No rows changed after applying updates; aborting write")
        return

    tmp_fp = GAMES_FP.with_suffix(".tmp")
    out.to_csv(tmp_fp, index=False)
    tmp_fp.replace(GAMES_FP)
    print(f"Wrote updated scores to {GAMES_FP} (rows changed: {changed})")


if __name__ == "__main__":
    main()
