"""
Fetch schedules and team-week stats from nflfastR (via nfl_data_py) and write CSVs:
- data/games.csv (GameRow schema)
- data/team_stats.csv (TeamStatRow schema)

Usage (from nfl_compare directory):
  python -m src.fetch_nflfastr --seasons 2023 2024 2025
  python -m src.fetch_nflfastr --range 2022-2025

If no seasons specified, defaults to last 5 seasons up to current year.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from nfl_data_py import import_schedules, import_pbp_data
except Exception as e:  # pragma: no cover - only raised if package missing
    import_schedules = None  # type: ignore
    import_pbp_data = None  # type: ignore

from .team_normalizer import normalize_team_name


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _ensure_packages():
    if import_schedules is None or import_pbp_data is None:
        raise RuntimeError(
            "Missing nfl_data_py. Install with: pip install nfl-data-py pyarrow"
        )


def _parse_season_args(seasons: Optional[List[int]], rng: Optional[str]) -> List[int]:
    if seasons:
        return sorted(set(int(s) for s in seasons))
    if rng:
        a, b = _parse_range(rng)
        return list(range(a, b + 1))
    # default: last 5 seasons including current
    now = datetime.now().year
    return [now - 4, now - 3, now - 2, now - 1, now]


def _parse_range(expr: str) -> Tuple[int, int]:
    parts = expr.split("-")
    if len(parts) != 2:
        raise ValueError("Range must be like 2022-2025")
    return int(parts[0]), int(parts[1])


def _norm_team(value: str) -> str:
    return normalize_team_name(value)


def fetch_games(seasons: Iterable[int]) -> pd.DataFrame:
    _ensure_packages()
    sched = import_schedules(list(seasons))
    # Expected columns include: season, week, game_id, gameday/start_time, home_team, away_team, home_score, away_score
    # Normalize names and select schema
    df = sched.copy()
    # Date: prefer "gameday" or "start_time"
    date_col = "gameday" if "gameday" in df.columns else ("start_time" if "start_time" in df.columns else None)
    if date_col is None:
        df["date"] = pd.NaT
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)

    # Normalize team names
    df["home_team"] = df["home_team"].astype(str).apply(_norm_team)
    df["away_team"] = df["away_team"].astype(str).apply(_norm_team)

    # Ensure required numeric columns exist
    for c in ["home_score", "away_score"]:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[[
        "season",
        "week",
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]].drop_duplicates().reset_index(drop=True)

    # Attempt to enrich with quarterly scoring from PBP (season-by-season, skip if unavailable)
    try:
        pbp_list = []
        for s in seasons:
            try:
                p = import_pbp_data([int(s)])
                pbp_list.append(p)
            except Exception as e:
                print(f"Skipping PBP for season {s}: {e}")
        if pbp_list:
            pbp = pd.concat(pbp_list, ignore_index=True)
            qdf = _quarterly_scores_from_pbp(pbp)
            out = out.merge(qdf, on="game_id", how="left")
    except Exception as e:
        print(f"Quarterly scoring enrichment failed: {e}")
    return out


def _quarterly_scores_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Derive per-quarter scoring from play-by-play cumulative scores.

    Prefer cumulative score columns (total_home_score/total_away_score). Fall back to
    home_score/away_score if needed. Compute deltas between end-of-quarter cumulative
    totals to get per-quarter points. Ignore OT.
    """
    # Pick cumulative score columns robustly
    home_cols = [c for c in ["total_home_score", "home_score"] if c in pbp.columns]
    away_cols = [c for c in ["total_away_score", "away_score"] if c in pbp.columns]
    if not {"game_id", "qtr"}.issubset(pbp.columns) or not home_cols or not away_cols:
        return pd.DataFrame(columns=[
            "game_id","home_q1","home_q2","home_q3","home_q4","away_q1","away_q2","away_q3","away_q4"
        ])
    hcol, acol = home_cols[0], away_cols[0]
    df = pbp[["game_id","qtr", hcol, acol]].copy()
    # Coerce quarter to integers 1-4; ignore OT and non-numeric
    df["qtr"] = pd.to_numeric(df["qtr"], errors="coerce")
    df = df[df["qtr"].isin([1, 2, 3, 4])]
    if df.empty:
        return pd.DataFrame(columns=[
            "game_id","home_q1","home_q2","home_q3","home_q4","away_q1","away_q2","away_q3","away_q4"
        ])
    # Take the last observation within each (game, quarter) for cumulative scores
    last_q = (
        df.sort_values(["game_id", "qtr"])  # ascending within game
          .groupby(["game_id", "qtr"], as_index=False, sort=False)
          .last()
    )
    # Pivot to wide by quarter (columns will be {1,2,3,4} where present)
    wide_h = last_q.pivot(index="game_id", columns="qtr", values=hcol)
    wide_a = last_q.pivot(index="game_id", columns="qtr", values=acol)
    # Ensure columns 1..4 exist, and forward-fill cumulative within the row to allow diff
    for w in (wide_h, wide_a):
        for q in [1, 2, 3, 4]:
            if q not in w.columns:
                w[q] = np.nan
        w.sort_index(axis=1, inplace=True)
        w.ffill(axis=1, inplace=True)
        w.fillna(0, inplace=True)
    # Compute per-quarter deltas from cumulative totals
    def deltas(wide: pd.DataFrame) -> pd.DataFrame:
        q1 = wide[1]
        q2 = (wide[2] - wide[1]).clip(lower=0)
        q3 = (wide[3] - wide[2]).clip(lower=0)
        q4 = (wide[4] - wide[3]).clip(lower=0)
        return pd.concat([
            q1.rename("q1"), q2.rename("q2"), q3.rename("q3"), q4.rename("q4")
        ], axis=1)

    hd = deltas(wide_h)
    ad = deltas(wide_a)
    out = pd.DataFrame({
        "game_id": wide_h.index,
        "home_q1": hd["q1"].values,
        "home_q2": hd["q2"].values,
        "home_q3": hd["q3"].values,
        "home_q4": hd["q4"].values,
        "away_q1": ad["q1"].values,
        "away_q2": ad["q2"].values,
        "away_q3": ad["q3"].values,
        "away_q4": ad["q4"].values,
    })
    return out


def _neutral_pace(off_pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute neutral seconds/play by team-week using PBP.
    Neutral = 0.2 <= wp <= 0.8 and abs(score_differential) <= 7.
    """
    df = off_pbp.copy()
    # Sort so diff(-1) gives next play in sequence per game/team
    df = df.sort_values([
        "season", "week", "game_id", "posteam", "game_seconds_remaining"
    ], ascending=[True, True, True, True, False])

    # Per (game, team), seconds between consecutive plays
    df["secs"] = df.groupby(["game_id", "posteam"])['game_seconds_remaining'].diff(-1)

    # Neutral filter (tweaked): WP 0.15-0.85 and score diff <= 10
    if "wp" in df.columns:
        neutral_wp = (df["wp"] >= 0.15) & (df["wp"] <= 0.85)
    else:
        neutral_wp = True
    if "score_differential" in df.columns:
        neutral_score = df["score_differential"].abs() <= 10
    else:
        neutral_score = True
    # Exclude late 4th quarter hurry-up: keep 4Q if > 300s remaining
    late_ok = (df.get("qtr", 0) < 4) | ((df.get("qtr", 0) == 4) & (df["game_seconds_remaining"] > 300))

    neutral = df[neutral_wp & neutral_score & late_ok]
    pace = (
        neutral.groupby(["season", "week", "posteam"])['secs']
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "secs": "pace_secs_play"})
    )
    return pace


def fetch_team_week_stats(seasons: Iterable[int]) -> pd.DataFrame:
    _ensure_packages()
    # Import PBP season by season to gracefully skip years not available yet
    pbp_list = []
    for s in seasons:
        try:
            df_s = import_pbp_data([int(s)])
            pbp_list.append(df_s)
        except Exception as e:
            print(f"Skipping PBP for season {s}: {e}")
    if not pbp_list:
        return pd.DataFrame(columns=[
            "season","week","team","off_epa","def_epa","pace_secs_play","pass_rate","rush_rate","qb_adj","sos"
        ])
    pbp = pd.concat(pbp_list, ignore_index=True)
    # Offense aggregations by posteam/week
    off = (
        pbp[pbp["posteam"].notna()].copy()
        .assign(
            is_pass=lambda d: d["pass"].fillna(0).astype(int),
            is_rush=lambda d: d["rush"].fillna(0).astype(int),
        )
    )

    grp_off = off.groupby(["season", "week", "posteam"], dropna=True)
    off_stats = grp_off.agg(
        plays=("play_id", "count"),
        off_epa=("epa", "mean"),
        pass_plays=("is_pass", "sum"),
        rush_plays=("is_rush", "sum"),
    ).reset_index()

    off_stats["pass_rate"] = (
        off_stats["pass_plays"] / (off_stats["pass_plays"] + off_stats["rush_plays"].replace(0, pd.NA))
    )
    off_stats["rush_rate"] = 1 - off_stats["pass_rate"]

    # EPA by half for offense
    off_half = off.copy()
    off_half["half"] = np.where(off_half.get("qtr", 0) <= 2, "1h", "2h")
    off_half_agg = (
        off_half.groupby(["season","week","posteam","half"], dropna=True)
                 .agg(off_epa_half=("epa","mean"))
                 .reset_index()
    )
    off_half_piv = (
        off_half_agg.pivot_table(index=["season","week","posteam"], columns="half", values="off_epa_half")
                    .rename(columns={"1h":"off_epa_1h","2h":"off_epa_2h"})
                    .reset_index()
    )

    # Defense aggregations by defteam/week
    grp_def = pbp[pbp["defteam"].notna()].groupby(["season", "week", "defteam"], dropna=True)
    def_stats = grp_def.agg(def_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"})

    # EPA by half for defense
    def_df = pbp[pbp["defteam"].notna()].copy()
    def_df["half"] = np.where(def_df.get("qtr", 0) <= 2, "1h", "2h")
    def_half_agg = (
        def_df.groupby(["season","week","defteam","half"], dropna=True)
              .agg(def_epa_half=("epa","mean"))
              .reset_index()
    )
    def_half_piv = (
        def_half_agg.pivot_table(index=["season","week","defteam"], columns="half", values="def_epa_half")
                     .rename(columns={"1h":"def_epa_1h","2h":"def_epa_2h"})
                     .reset_index()
    )

    # Merge offense and defense
    merged = off_stats.merge(
        def_stats,
        left_on=["season", "week", "posteam"],
        right_on=["season", "week", "team"],
        how="outer",
    )

    # Merge offense half EPA
    merged = merged.merge(
        off_half_piv,
        on=["season","week","posteam"],
        how="left",
    )
    # Merge defense half EPA
    merged = merged.merge(
        def_half_piv,
        left_on=["season","week","posteam"],
        right_on=["season","week","defteam"],
        how="left",
        suffixes=("", "_defh"),
    )

    merged["team"] = merged["posteam"].fillna(merged["team"])  # prefer posteam label
    merged = merged.drop(columns=["posteam"]).reset_index(drop=True)

    # Normalize team names
    merged["team"] = merged["team"].astype(str).apply(_norm_team)

    # Select/rename fields to schema
    out = merged[[
        "season",
        "week",
        "team",
        "off_epa",
        "def_epa",
        "off_epa_1h","off_epa_2h","def_epa_1h","def_epa_2h",
        "pass_rate",
        "rush_rate",
    ]].copy()

    # Neutral pace from PBP
    pace = _neutral_pace(off)
    pace["team"] = pace["team"].astype(str).apply(_norm_team)
    out = out.merge(pace, on=["season", "week", "team"], how="left")

    # Placeholders for optional fields
    out["qb_adj"] = pd.NA
    out["sos"] = pd.NA

    # Reorder columns
    out = out[[
        "season",
        "week",
        "team",
        "off_epa",
        "def_epa",
        "off_epa_1h","off_epa_2h","def_epa_1h","def_epa_2h",
        "pace_secs_play",
        "pass_rate",
        "rush_rate",
        "qb_adj",
        "sos",
    ]]

    # Drop duplicates from overlapping sources
    out = out.drop_duplicates(subset=["season", "week", "team"]).reset_index(drop=True)
    return out


def _compute_sos(team_stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple SOS as average of opponents' rolling (off_epa - def_epa)
    up to the previous week within the same season.
    """
    ts = team_stats.copy()
    ts["net_epa"] = ts[["off_epa", "def_epa"]].mean(axis=1) - ts["def_epa"]
    # Correct net_epa: off_epa - def_epa
    ts["net_epa"] = ts["off_epa"] - ts["def_epa"]

    ts = ts.sort_values(["season", "team", "week"])\
           .assign(net_epa_roll=lambda d: d.groupby(["season", "team"])['net_epa'].expanding().mean().reset_index(level=[0,1], drop=True))
    # Shift to prior week value
    ts["net_epa_roll_prev"] = ts.groupby(["season", "team"])['net_epa_roll'].shift(1)

    # Build schedule long with opponent, week, and location
    sched = games[["season", "week", "home_team", "away_team"]].copy()
    a = sched.assign(location="H").rename(columns={"home_team": "team", "away_team": "opp", "week": "opp_week"})
    b = sched.assign(location="A").rename(columns={"away_team": "team", "home_team": "opp", "week": "opp_week"})
    opps = pd.concat([a, b], ignore_index=True)

    # Map opponent rolling net EPA at that opp_week
    opp_strength = ts[["season", "team", "week", "net_epa_roll_prev"]].rename(columns={
        "team": "opp",
        "week": "opp_week",
        "net_epa_roll_prev": "opp_strength"
    })

    opps = opps.merge(opp_strength, on=["season", "opp", "opp_week"], how="left")

    # For each (season, team, week), weighted average opp_strength over all past games
    keys = ts[["season", "team", "week"]]
    sos = keys.merge(opps, on=["season", "team"], how="left")
    sos = sos[sos["opp_week"] < sos["week"]]
    # Weights: recency (0.85^gap) and location (away 1.05, home 0.95)
    gap = (sos["week"] - sos["opp_week"]).clip(lower=1)
    w_rec = (0.85 ** gap)
    w_loc = sos["location"].map({"A": 1.05, "H": 0.95}).fillna(1.0)
    w = w_rec * w_loc
    def _wavg(d: pd.DataFrame) -> float:
        ww = w.loc[d.index]
        num = (d["opp_strength"] * ww).sum()
        den = ww.sum()
        return float(num / den) if den and pd.notna(num) else float('nan')
    # Exclude grouping columns during apply to avoid pandas deprecation; fallback for older pandas
    try:
        sos_agg = sos.groupby(["season", "team", "week"]).apply(_wavg, include_groups=False).reset_index(name="sos")
    except TypeError:
        sos_agg = sos.groupby(["season", "team", "week"], group_keys=False).apply(_wavg).reset_index(name="sos")

    # Merge back
    ts = ts.merge(sos_agg, on=["season", "team", "week"], how="left")
    # Consolidate to a single 'sos' column
    if "sos" in ts.columns and "sos_y" in ts.columns:
        ts["sos"] = ts["sos_y"].where(ts["sos_y"].notna(), ts["sos"])  # favor computed
        ts = ts.drop(columns=["sos_y"])  # drop extra
    elif "sos_y" in ts.columns and "sos" not in ts.columns:
        ts = ts.rename(columns={"sos_y": "sos"})
    # Clean working columns
    for col in ["net_epa", "net_epa_roll", "net_epa_roll_prev", "sos_x"]:
        if col in ts.columns:
            ts = ts.drop(columns=[col])
    return ts


def _overlay_qb_adjustments(team_stats: pd.DataFrame) -> pd.DataFrame:
    """If data/qb_adjustments.csv exists with columns [season, week, team, qb_adj],
    merge it to fill/override qb_adj values.
    """
    fp = DATA_DIR / "qb_adjustments.csv"
    if not fp.exists():
        return team_stats
    try:
        adj = pd.read_csv(fp)
    except Exception:
        return team_stats
    needed = {"season", "week", "team", "qb_adj"}
    if not needed.issubset(set(adj.columns)):
        return team_stats
    adj["team"] = adj["team"].astype(str).apply(_norm_team)
    out = team_stats.merge(adj[["season","week","team","qb_adj"]], on=["season","week","team"], how="left", suffixes=("", "_ovr"))
    # prefer override when present
    if "qb_adj_ovr" in out.columns:
        out["qb_adj"] = out["qb_adj_ovr"].where(out["qb_adj_ovr"].notna(), out["qb_adj"])
        out = out.drop(columns=["qb_adj_ovr"])
    return out


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / filename
    df.to_csv(fp, index=False)
    return fp


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NFL data from nflfastR (nfl_data_py)")
    parser.add_argument("--seasons", nargs="*", type=int, help="List of seasons, e.g., 2023 2024 2025")
    parser.add_argument("--range", dest="range_", type=str, help="Range like 2022-2025", default=None)
    parser.add_argument("--only-schedules", action="store_true", help="Only fetch schedules/games.csv")
    parser.add_argument("--only-stats", action="store_true", help="Only fetch team_stats.csv from PBP")
    args = parser.parse_args(argv)

    seasons = _parse_season_args(args.seasons, args.range_)

    if not args.only_stats:
        print(f"Fetching schedules for seasons: {seasons}")
        games = fetch_games(seasons)
        gfp = write_csv(games, "games.csv")
        print(f"Wrote {len(games)} rows to {gfp}")
    else:
        # Try to read existing games.csv for SOS computation
        try:
            games = pd.read_csv(DATA_DIR / "games.csv")
        except Exception:
            games = pd.DataFrame(columns=["season","week","home_team","away_team"])  # minimal

    if not args.only_schedules:
        print("Aggregating team-week stats from play-by-play (this may take a while on first run)...")
        team_stats = fetch_team_week_stats(seasons)
        # Compute SOS if we have games data
        try:
            team_stats = _compute_sos(team_stats, games)
        except Exception:
            pass
        # Overlay any manual QB adjustments
        try:
            team_stats = _overlay_qb_adjustments(team_stats)
        except Exception:
            pass
        tfp = write_csv(team_stats, "team_stats.csv")
        print(f"Wrote {len(team_stats)} rows to {tfp}")


if __name__ == "__main__":
    main()
