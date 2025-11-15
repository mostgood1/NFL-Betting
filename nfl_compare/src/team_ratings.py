"""
Lightweight team-level ratings (offense/defense/net) computed from finalized games.

Ratings are exponential moving averages (EMA) of points for/against and margin,
computed per team and aligned so that for a game in week W, we use ratings built
from weeks < W (no leakage). These can be attached to the weekly view to enrich
features and serve as priors for model predictions.

Artifacts: optional CSV caches under nfl_compare/data/team_ratings_{season}_wk{week}.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import pandas as pd
import numpy as np


# Respect NFL_DATA_DIR
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (Path(__file__).resolve().parents[1] / "data")


def _ema_update(prev: Optional[float], value: Optional[float], alpha: float) -> Optional[float]:
    try:
        v = float(value)
        if not np.isfinite(v):
            return prev
    except Exception:
        return prev
    if prev is None or not np.isfinite(prev):
        return v
    return alpha * v + (1.0 - alpha) * float(prev)


def compute_team_ratings_from_games(games: pd.DataFrame, season: int, up_to_week: int, alpha: float = 0.6) -> pd.DataFrame:
    """Compute EMA-based team ratings up to (but not including) up_to_week.

    Returns a DataFrame with columns:
      season, week, team, off_ppg, def_ppg, net_margin, games

    For week W rows, the metrics reflect games from weeks < W only.
    """
    if games is None or games.empty:
        return pd.DataFrame(columns=["season","week","team","off_ppg","def_ppg","net_margin","games"])  # empty

    df = games.copy()
    # Coerce types
    for c in ("season","week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("home_score","away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["season"].eq(int(season)))].copy()
    if df.empty:
        return pd.DataFrame(columns=["season","week","team","off_ppg","def_ppg","net_margin","games"])  # empty

    # Build per-week finalized rows only for ratings updates
    df_fin = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    if df_fin.empty:
        # If no finals yet, ratings are empty
        return pd.DataFrame(columns=["season","week","team","off_ppg","def_ppg","net_margin","games"])  # empty

    # Iterate weeks in order to build history
    weeks = sorted(pd.to_numeric(df["week"], errors="coerce").dropna().unique().tolist())
    # We'll prepare outputs for all weeks up to up_to_week
    weeks_for_output = [w for w in weeks if np.isfinite(w) and (w <= int(up_to_week))]

    # State: per team EMA stats and game counts
    teams = pd.unique(pd.concat([df_fin["home_team"], df_fin["away_team"]], ignore_index=True))
    state = {str(t): {"off_ppg": None, "def_ppg": None, "net_margin": None, "games": 0} for t in teams}

    rows = []
    for w in weeks_for_output:
        # For ratings used in week w, we want to output BEFORE updating with week w games
        # So first, emit rows for all teams for week w using current state, then update state with week w finals
        for team, vals in state.items():
            rows.append({
                "season": int(season),
                "week": int(w),
                "team": team,
                "off_ppg": vals.get("off_ppg"),
                "def_ppg": vals.get("def_ppg"),
                "net_margin": vals.get("net_margin"),
                "games": int(vals.get("games", 0)),
            })
        # Now update using week w finals
        wk = df_fin[pd.to_numeric(df_fin["week"], errors="coerce").eq(w)]
        for _, g in wk.iterrows():
            home = str(g.get("home_team"))
            away = str(g.get("away_team"))
            hs = float(g.get("home_score"))
            as_ = float(g.get("away_score"))
            # Update home
            st_h = state.get(home)
            if st_h is None:
                st_h = {"off_ppg": None, "def_ppg": None, "net_margin": None, "games": 0}
                state[home] = st_h
            st_h["off_ppg"] = _ema_update(st_h.get("off_ppg"), hs, alpha)
            st_h["def_ppg"] = _ema_update(st_h.get("def_ppg"), as_, alpha)
            st_h["net_margin"] = _ema_update(st_h.get("net_margin"), hs - as_, alpha)
            st_h["games"] = int(st_h.get("games", 0)) + 1
            # Update away
            st_a = state.get(away)
            if st_a is None:
                st_a = {"off_ppg": None, "def_ppg": None, "net_margin": None, "games": 0}
                state[away] = st_a
            st_a["off_ppg"] = _ema_update(st_a.get("off_ppg"), as_, alpha)
            st_a["def_ppg"] = _ema_update(st_a.get("def_ppg"), hs, alpha)
            st_a["net_margin"] = _ema_update(st_a.get("net_margin"), as_ - hs, alpha)
            st_a["games"] = int(st_a.get("games", 0)) + 1

    out = pd.DataFrame(rows)
    # Fill initial None with 0 for numeric stability
    for c in ("off_ppg","def_ppg","net_margin"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "games" in out.columns:
        out["games"] = pd.to_numeric(out["games"], errors="coerce").fillna(0).astype(int)
    return out


def attach_team_ratings_to_view(view: pd.DataFrame, games: Optional[pd.DataFrame] = None, alpha: Optional[float] = None) -> pd.DataFrame:
    """Attach team ratings to a weekly view.

    For each row (season, week, home_team, away_team), merges ratings computed
    from games up to that week (exclusive). Adds columns:
      home_off_ppg, home_def_ppg, home_net_margin, away_off_ppg, ..., and diffs:
      off_ppg_diff, def_ppg_diff, net_margin_diff (home - away).
    """
    if view is None or view.empty:
        return view
    v = view.copy()
    if not {"season","week","home_team","away_team"}.issubset(v.columns):
        return v
    try:
        v["season"] = pd.to_numeric(v["season"], errors="coerce").astype("Int64")
        v["week"] = pd.to_numeric(v["week"], errors="coerce").astype("Int64")
    except Exception:
        pass
    # Determine needed (season, week) pairs
    sw = v[["season","week"]].dropna().drop_duplicates()
    if sw.empty:
        return v
    # Load games lazily if not provided
    if games is None:
        try:
            from .data_sources import load_games as _load_games  # type: ignore
            games = _load_games()
        except Exception:
            games = None
    alpha_val = float(alpha) if alpha is not None else float(os.environ.get("TEAM_RATING_EMA", 0.6))

    ratings_frames = []
    for _, row in sw.iterrows():
        try:
            s = int(pd.to_numeric(row["season"], errors="coerce"))
            w = int(pd.to_numeric(row["week"], errors="coerce"))
        except Exception:
            continue
        try:
            r = compute_team_ratings_from_games(games, s, w, alpha=alpha_val)
        except Exception:
            r = pd.DataFrame()
        if r is not None and not r.empty:
            ratings_frames.append(r)
    if not ratings_frames:
        return v
    R = pd.concat(ratings_frames, ignore_index=True)
    # Merge for home and away
    for side in ("home","away"):
        key = f"{side}_team"
        r_cols = ["season","week","team","off_ppg","def_ppg","net_margin","games"]
        r = R[r_cols].rename(columns={
            "team": key,
            "off_ppg": f"{side}_off_ppg",
            "def_ppg": f"{side}_def_ppg",
            "net_margin": f"{side}_net_margin",
            "games": f"{side}_rating_games",
        })
        v = v.merge(r, on=["season","week", key], how="left")
    # Diffs (home - away)
    v["off_ppg_diff"] = v.get("home_off_ppg", 0).fillna(0) - v.get("away_off_ppg", 0).fillna(0)
    v["def_ppg_diff"] = v.get("home_def_ppg", 0).fillna(0) - v.get("away_def_ppg", 0).fillna(0)
    v["net_margin_diff"] = v.get("home_net_margin", 0).fillna(0) - v.get("away_net_margin", 0).fillna(0)
    return v


def write_team_ratings_csv(games: pd.DataFrame, season: int, week: int, out_path: Optional[Path] = None, alpha: float = 0.6) -> Path:
    """Materialize ratings CSV for convenience in pipelines.

    Returns the written Path.
    """
    out = compute_team_ratings_from_games(games, season, week, alpha=alpha)
    if out_path is None:
        out_path = DATA_DIR / f"team_ratings_{season}_wk{week}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
