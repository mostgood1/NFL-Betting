from __future__ import annotations

"""Roster validation report.

Compares our modeled weekly roster (from player props output) against external sources.

Week-accurate roster/actives are required. To keep runs deterministic and avoid flaky
network fetches, this module prefers local caches:
- nfl_compare/data/external/nfl_data_py/seasonal_rosters_{season}.*
- nfl_compare/data/external/nfl_data_py/weekly_rosters_{season}.*

For depth ordering we prefer the local ESPN snapshot:
- nfl_compare/data/depth_chart_{season}_wk{week}.csv

Produces:
- data/roster_validation_summary_{season}_wk{week}.csv: team-level metrics
- data/roster_validation_details_{season}_wk{week}.csv: per-player matching details

Usage:
  python -m nfl_compare.src.roster_validation --season 2025 --week 2
"""

import argparse
from pathlib import Path
import socket
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .team_normalizer import normalize_team_name
from .name_normalizer import normalize_name_loose
from .player_props import compute_player_props, DATA_DIR
from .depth_charts import load_depth_chart_csv
from .roster_cache import get_seasonal_rosters, get_weekly_rosters
try:
    # Use same override mapping as modeling to avoid false negatives when external feeds are wrong
    from .player_props import QB_OVERRIDE_BY_TEAM  # type: ignore
except Exception:  # pragma: no cover - fallback when constant not present
    QB_OVERRIDE_BY_TEAM = {}


def _read_props_or_compute(season: int, week: int) -> pd.DataFrame:
    """Try to read cached props CSV, else compute via compute_player_props.
    We look for common patterns: player_props_{season}_wk{week}.csv under DATA_DIR.
    """
    season = int(season); week = int(week)
    patterns = [
        DATA_DIR / f"player_props_{season}_wk{week}.csv",
        DATA_DIR / f"player_props_{season}_week{week}.csv",
        DATA_DIR / f"props_{season}_wk{week}.csv",
    ]
    for p in patterns:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if df is not None and not df.empty:
                    return df
        except Exception:
            pass
    # Compute on the fly
    try:
        df = compute_player_props(season=season, week=week)
        return df
    except Exception as e:
        raise SystemExit(f"Failed to load or compute player props for {season} wk{week}: {e}")


def _load_external_rosters(season: int, week: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load external data from nfl_data_py: seasonal rosters, depth charts, weekly rosters.
    Returns (rosters, depth_charts, weekly_rosters) with normalized team names.
    """
    # Seasonal rosters (cached locally; fetch-on-miss)
    ros = get_seasonal_rosters(int(season))
    if ros is None:
        ros = pd.DataFrame()
    ros = ros.copy() if not ros.empty else ros

    # Local weekly depth chart (ESPN) acts as our week-specific depth ordering.
    dc = load_depth_chart_csv(int(season), int(week))
    if dc is None or dc.empty:
        dch = pd.DataFrame(columns=["team", "player", "player_id", "depth_chart_position", "depth_chart_order", "_nm"])
    else:
        dch = dc.copy()
        dch["team"] = dch.get("team", pd.Series(dtype=object)).astype(str).apply(normalize_team_name)
        dch["depth_chart_position"] = dch.get("position", pd.Series(dtype=object)).astype(str)
        dch["depth_chart_order"] = pd.to_numeric(dch.get("depth_rank", pd.Series(dtype=object)), errors="coerce")
        dch["player"] = dch.get("player", pd.Series(dtype=object)).astype(str)
        dch["player_id"] = pd.NA
        dch["_nm"] = dch["player"].astype(str).map(normalize_name_loose)
        dch = dch[["team", "player", "player_id", "depth_chart_position", "depth_chart_order", "_nm"]].copy()

    # Weekly rosters for active flags (cached locally; fetch-on-miss)
    wr = get_weekly_rosters(int(season))
    if wr is None or wr.empty:
        wr = pd.DataFrame(columns=["team", "player", "player_id", "is_active", "_nm"])
    else:
        wr = wr.copy()
        # Map team
        tcol3 = None
        for c in ["team", "recent_team", "team_abbr", "club_code"]:
            if c in wr.columns:
                tcol3 = c
                break
        wr["team"] = wr[tcol3].astype(str).apply(normalize_team_name) if tcol3 else pd.NA
        # week filter
        if "week" in wr.columns:
            wr["week"] = pd.to_numeric(wr["week"], errors="coerce").fillna(0).astype(int)
            wr = wr[wr["week"] == int(week)].copy()

        # active flag
        active_col = None
        for c in ["is_active", "active", "status"]:
            if c in wr.columns:
                active_col = c
                break
        if active_col is not None:
            if active_col == "status":
                wr["is_active"] = wr["status"].astype(str).str.upper().isin(["ACT", "ACTIVE"]).astype(int)
            else:
                wr["is_active"] = pd.to_numeric(wr[active_col], errors="coerce").fillna(0).astype(int)
        else:
            wr["is_active"] = 1

        # Names/ids
        name_cols = [
            "player_display_name", "display_name", "full_name", "football_name",
            "player_name", "name", "gsis_name",
        ]
        id_cols = ["gsis_id", "player_id", "nfl_id", "pfr_id", "sportradar_id"]

        wr["player"] = pd.NA
        for c in name_cols:
            if c in wr.columns:
                wr["player"] = wr["player"].fillna(wr[c])
        wr["player_id"] = pd.NA
        for c in id_cols:
            if c in wr.columns:
                wr["player_id"] = wr["player_id"].fillna(wr[c])
        wr["player"] = wr["player"].astype(str)
        wr["_nm"] = wr["player"].astype(str).map(normalize_name_loose)
        wr = wr[["team", "player", "player_id", "is_active", "_nm"]].copy()

    # If seasonal rosters are empty (offline / fetch failure), keep the pipeline alive.
    if ros is None or ros.empty:
        ros = pd.DataFrame(columns=["team", "player", "player_id", "position", "depth_chart_position", "depth_chart_order", "_nm"])
    else:
        # Map team and derive a robust player display name
        tcol = None
        for c in ["team", "recent_team", "team_abbr", "club_code"]:
            if c in ros.columns:
                tcol = c; break
        if tcol is None:
            ros["team"] = pd.NA
        else:
            ros["team"] = ros[tcol].astype(str).apply(normalize_team_name)
        # Build display name column
        name_cols = [
            "player_display_name", "display_name", "full_name", "football_name",
            "player_name", "name", "gsis_name",
        ]
        ros["player"] = pd.NA
        for c in name_cols:
            if c in ros.columns:
                ros["player"] = ros["player"].fillna(ros[c])
        # Position and depth
        if "depth_chart_position" not in ros.columns:
            ros["depth_chart_position"] = pd.NA
        if "depth_chart_order" not in ros.columns:
            ros["depth_chart_order"] = pd.NA
        if "position" not in ros.columns:
            ros["position"] = pd.NA
        # ID
        id_cols = ["gsis_id", "player_id", "nfl_id", "pfr_id", "sportradar_id"]
        ros["player_id"] = pd.NA
        for c in id_cols:
            if c in ros.columns:
                ros["player_id"] = ros["player_id"].fillna(ros[c])
        ros["player"] = ros["player"].astype(str)
        ros["_nm"] = ros["player"].astype(str).map(normalize_name_loose)
        ros["depth_chart_order"] = pd.to_numeric(ros["depth_chart_order"], errors="coerce")

        return ros, dch, wr


def _match_props_to_external(props: pd.DataFrame, ros: pd.DataFrame, dch: pd.DataFrame, wr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (details, summary) dataframes comparing props to external sources."""
    # Normalize inputs
    props = props.copy()
    props["team"] = props["team"].astype(str).apply(normalize_team_name)
    props["player"] = props["player"].astype(str)
    props["_nm"] = props["player"].astype(str).map(normalize_name_loose)
    if "player_id" in props.columns:
        props["player_id"] = props["player_id"].astype(str)

    # Build lookups
    ros_idx = ros[["team","player","player_id","position","depth_chart_position","depth_chart_order","_nm"]].copy()
    dch_idx = dch[["team","player","player_id","depth_chart_position","depth_chart_order","_nm"]].copy()
    wr_idx = wr[["team","player","player_id","is_active","_nm"]].copy()

    details_rows: List[Dict] = []
    for _, r in props.iterrows():
        team = r.get("team"); nm = r.get("_nm"); pid = r.get("player_id"); pos = r.get("position")
        # Default
        match_type = "none"; ext_team = None; ext_pos = None; ext_order = np.nan; active = np.nan
        # Try id match first on rosters then depth charts
        m_id = pd.DataFrame()
        if pid and str(pid) and str(pid).lower() != "nan":
            m_id = ros_idx[ros_idx["player_id"].astype(str) == str(pid)]
            if m_id.empty:
                m_id = dch_idx[dch_idx["player_id"].astype(str) == str(pid)]
        if not m_id.empty:
            m = m_id.iloc[0]
            match_type = "id"
            ext_team = m.get("team"); ext_pos = m.get("depth_chart_position") or m.get("position")
            ext_order = m.get("depth_chart_order")
        else:
            # Name match on team, fallback to cross-team
            mt = ros_idx[(ros_idx["team"] == team) & (ros_idx["_nm"] == nm)]
            if mt.empty:
                mt = dch_idx[(dch_idx["team"] == team) & (dch_idx["_nm"] == nm)]
            if not mt.empty:
                m = mt.sort_values(by=["depth_chart_order"], na_position="last").iloc[0]
                match_type = "name"
                ext_team = m.get("team"); ext_pos = m.get("depth_chart_position") or m.get("position")
                ext_order = m.get("depth_chart_order")
            else:
                # Cross-team name match to flag contamination
                mt2 = ros_idx[ros_idx["_nm"] == nm]
                if mt2.empty:
                    mt2 = dch_idx[dch_idx["_nm"] == nm]
                if not mt2.empty:
                    m = mt2.sort_values(by=["depth_chart_order"], na_position="last").iloc[0]
                    match_type = "name_other_team"
                    ext_team = m.get("team"); ext_pos = m.get("depth_chart_position") or m.get("position")
                    ext_order = m.get("depth_chart_order")

        # Active flag
        if nm and not pd.isna(nm) and wr_idx is not None and not wr_idx.empty:
            wa = wr_idx[(wr_idx["team"] == team) & (wr_idx["_nm"] == nm)]
            if not wa.empty:
                active = pd.to_numeric(wa.iloc[0].get("is_active"), errors="coerce")

        pos_mismatch = False
        if pos and ext_pos:
            pos_mismatch = str(pos).upper() != str(ext_pos).upper()

        details_rows.append({
            "team": team,
            "player": r.get("player"),
            "player_id": r.get("player_id"),
            "position": pos,
            "match_type": match_type,
            "ext_team": ext_team,
            "ext_position": ext_pos,
            "ext_depth_order": ext_order,
            "ext_active": active,
            # a few model columns to aid debugging
            "targets": pd.to_numeric(r.get("targets"), errors="coerce"),
            "rush_attempts": pd.to_numeric(r.get("rush_attempts"), errors="coerce"),
            "pass_attempts": pd.to_numeric(r.get("pass_attempts"), errors="coerce"),
        })

    details = pd.DataFrame(details_rows)
    if details.empty:
        return details, pd.DataFrame()

    # Team-level summary
    def _summarize_team(g: pd.DataFrame) -> Dict:
        total = len(g)
        matched_on_team = int((((g["match_type"] == "id") | (g["match_type"] == "name")) & ((g["ext_team"] == g["team"]) | g["ext_team"].isna())).sum())
        matched_other = int((g["match_type"] == "name_other_team").sum())
        unmatched = int((g["match_type"] == "none").sum())
        pos_mm = int((g["ext_position"].notna() & g["position"].notna() & (g["ext_position"].astype(str).str.upper() != g["position"].astype(str).str.upper())).sum())
        inactive = int(pd.to_numeric(g.get("ext_active"), errors="coerce").fillna(1).astype(int).eq(0).sum())
        # Heuristic QB starter match: our QB with max pass_attempts vs external depth order 1
        qb_model = g[g["position"].astype(str).str.upper() == "QB"].copy()
        qb_model["pass_attempts"] = pd.to_numeric(qb_model.get("pass_attempts"), errors="coerce").fillna(0)
        qb_model_name = qb_model.sort_values(["pass_attempts","rush_attempts"], ascending=[False, False])["player"].iloc[0] if not qb_model.empty else None
        # Determine external QB starter more robustly: prefer depth chart order==1; use id when possible, else name
        team_key = g["team"].iloc[0]
        qb_ext = details[(details["team"] == team_key) & (details["ext_position"].astype(str).str.upper() == "QB")].copy()
        qb_ext["ext_depth_order"] = pd.to_numeric(qb_ext.get("ext_depth_order"), errors="coerce")
        qb_ext1 = qb_ext[qb_ext["ext_depth_order"] == 1]
        if not qb_ext1.empty:
            qb_ext_name = qb_ext1["player"].iloc[0]
        else:
            qb_ext_name = qb_ext.sort_values(["ext_depth_order"], na_position="last")["player"].iloc[0] if not qb_ext.empty else None
        # If we have a force override for this team, prefer it as the "external" starter when available
        try:
            forced = QB_OVERRIDE_BY_TEAM.get(team_key)
            if forced:
                qb_ext_name = forced
        except Exception:
            pass
        qb_match = int(qb_model_name is not None and qb_ext_name is not None and normalize_name_loose(qb_model_name) == normalize_name_loose(qb_ext_name))
        return {
            "team": g["team"].iloc[0],
            "total_model_players": total,
            "matched_on_team": matched_on_team,
            "matched_other_team": matched_other,
            "unmatched": unmatched,
            "pos_mismatch": pos_mm,
            "ext_inactive": inactive,
            "qb_starter_match": qb_match,
        }

    summary = pd.DataFrame([_summarize_team(g) for _, g in details.groupby("team")])
    return details, summary


def build_roster_validation(season: int, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    props = _read_props_or_compute(season, week)
    # Keep core positions
    if "position" in props.columns:
        props = props[props["position"].astype(str).str.upper().isin(["QB","RB","WR","TE"])].copy()
    ros, dch, wr = _load_external_rosters(season, week)
    details, summary = _match_props_to_external(props, ros, dch, wr)
    # Write outputs
    out1 = DATA_DIR / f"roster_validation_details_{int(season)}_wk{int(week)}.csv"
    out2 = DATA_DIR / f"roster_validation_summary_{int(season)}_wk{int(week)}.csv"
    try:
        if details is not None and not details.empty:
            out1.parent.mkdir(parents=True, exist_ok=True)
            details.to_csv(out1, index=False)
        if summary is not None and not summary.empty:
            out2.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out2, index=False)
    except Exception:
        pass
    return details, summary


def main():
    ap = argparse.ArgumentParser(description="Roster validation report vs nfl_data_py rosters/depth charts")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    details, summary = build_roster_validation(args.season, args.week)
    if summary is None or summary.empty:
        print("No validation summary produced.")
    else:
        print(summary.sort_values(["qb_starter_match","matched_on_team","unmatched"], ascending=[False, False, True]).to_string(index=False))
        print()
        print("Wrote:")
        print(f" - {DATA_DIR / f'roster_validation_summary_{int(args.season)}_wk{int(args.week)}.csv'}")
        print(f" - {DATA_DIR / f'roster_validation_details_{int(args.season)}_wk{int(args.week)}.csv'}")


if __name__ == "__main__":
    main()
