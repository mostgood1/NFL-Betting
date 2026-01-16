from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.data_sources import load_games, load_lines, load_team_stats
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games


DATA_DIR = REPO_ROOT / "nfl_compare" / "data"


def _as_bool(v) -> bool:
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"false", "0", "no", "n", "inactive", "out"}:
        return False
    if s in {"true", "1", "yes", "y", "active", "in"}:
        return True
    return True


def _get_baseline_week() -> int:
    try:
        baseline_week = int(pd.to_numeric(os.environ.get("INJURY_BASELINE_WEEK", 1), errors="coerce"))
        if not np.isfinite(baseline_week) or baseline_week < 1:
            baseline_week = 1
        return int(baseline_week)
    except Exception:
        return 1


def _load_weekly_depth_chart(season: int, week: int) -> pd.DataFrame:
    # Prefer the same loader used by merge_features
    from nfl_compare.src.player_props import _load_weekly_depth_chart as _ldc  # type: ignore

    df = _ldc(int(season), int(week))
    if df is None or df.empty:
        return pd.DataFrame()

    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team  # type: ignore
    except Exception:
        _norm_team = lambda s: str(s)

    d = df.copy()
    if "team" in d.columns:
        d["team"] = d["team"].astype(str).apply(_norm_team)
    if "active" not in d.columns:
        d["active"] = True
    d["active"] = d["active"].map(_as_bool)

    if "position" not in d.columns:
        return pd.DataFrame()

    if "depth_rank" not in d.columns:
        d["depth_rank"] = d.groupby(["team", "position"]).cumcount() + 1

    d["pos_up"] = d["position"].astype(str).str.upper().str.strip()
    d["depth_rank"] = pd.to_numeric(d["depth_rank"], errors="coerce")
    return d


def _predicted_not_active_set(season: int, week: int) -> set[tuple[str, str]]:
    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team  # type: ignore
    except Exception:
        _norm_team = lambda s: str(s)

    pred_fp = DATA_DIR / f"predicted_not_active_{int(season)}_wk{int(week)}.csv"
    if not pred_fp.exists():
        return set()

    try:
        pred = pd.read_csv(pred_fp)
    except Exception:
        return set()

    if pred is None or pred.empty:
        return set()

    tcol = "team" if "team" in pred.columns else None
    pcol = "player" if "player" in pred.columns else None
    if not tcol or not pcol:
        return set()

    tmp = pred[[tcol, pcol]].dropna().copy()
    tmp[tcol] = tmp[tcol].astype(str).apply(_norm_team)
    tmp[pcol] = tmp[pcol].astype(str)
    return set(zip(tmp[tcol].tolist(), tmp[pcol].tolist()))


def _baseline_slots(baseline_dc: pd.DataFrame) -> pd.DataFrame:
    slots = {"QB": 1, "RB": 1, "TE": 1, "WR": 2}
    b = baseline_dc.copy()
    b = b[b["pos_up"].isin(list(slots.keys()))].copy()
    if b.empty:
        return pd.DataFrame()

    b = b.sort_values(["team", "pos_up", "depth_rank", "player"])
    b["_slot"] = b.groupby(["team", "pos_up"]).cumcount() + 1
    b["_max_slot"] = b["pos_up"].map(slots).fillna(1).astype(int)
    b = b[b["_slot"] <= b["_max_slot"]].copy()
    return b[["team", "pos_up", "_slot", "player"]].reset_index(drop=True)


def drilldown_game(game_id: str) -> None:
    games = load_games()
    g = None
    if games is not None and not games.empty and "game_id" in games.columns:
        gg = games[games["game_id"].astype(str) == str(game_id)]
        if not gg.empty:
            g = gg.iloc[0].to_dict()

    if g is None:
        # Parse as fallback: YYYY_WW_AWAY_HOME
        parts = str(game_id).split("_")
        if len(parts) >= 4:
            season = int(parts[0])
            week = int(parts[1])
        else:
            raise SystemExit(f"Could not locate game_id in games.csv and couldn't parse: {game_id}")
        away_team = None
        home_team = None
    else:
        season = int(pd.to_numeric(g.get("season"), errors="coerce"))
        week = int(pd.to_numeric(g.get("week"), errors="coerce"))
        away_team = str(g.get("away_team"))
        home_team = str(g.get("home_team"))

    baseline_week = _get_baseline_week()

    baseline_dc = _load_weekly_depth_chart(season, baseline_week)
    cur_dc = _load_weekly_depth_chart(season, week)

    if baseline_dc.empty:
        raise SystemExit(f"Baseline depth chart missing/empty for season={season} week={baseline_week}")
    if cur_dc.empty:
        raise SystemExit(f"Current depth chart missing/empty for season={season} week={week}")

    bslots = _baseline_slots(baseline_dc)
    if bslots.empty:
        raise SystemExit("No baseline starter slots found (QB/RB/TE/WR)")

    cur_act = cur_dc[cur_dc["pos_up"].isin(["QB", "RB", "TE", "WR"])][["team", "pos_up", "player", "active"]].drop_duplicates()
    cur_act["active"] = cur_act["active"].map(_as_bool)

    pna_set = _predicted_not_active_set(season, week)

    m = bslots.merge(cur_act, on=["team", "pos_up", "player"], how="left", indicator=True)
    m["in_current"] = (m["_merge"] == "both")
    m["active_now"] = m["active"].map(_as_bool) if "active" in m.columns else True
    # Missing in current depth chart => out
    m["active_now"] = m["active_now"].where(m["in_current"], False)

    if pna_set:
        m["pna"] = list(zip(m["team"].astype(str), m["player"].astype(str)))
        m["pna"] = pd.Series(m["pna"]).isin(pna_set)
        m.loc[m["pna"] == True, "active_now"] = False
    else:
        m["pna"] = False

    m["out"] = (~m["active_now"].astype(bool)).astype(int)
    m["reason"] = ""
    m.loc[~m["in_current"], "reason"] = "missing_in_current"
    m.loc[(m["in_current"]) & (~m["active_now"].astype(bool)), "reason"] = "inactive_in_current"
    m.loc[m["pna"] == True, "reason"] = "predicted_not_active"

    # Pretty print
    print(f"Game: {game_id} | season={season} week={week} | baseline_week={baseline_week}")
    if away_team and home_team:
        print(f"Away: {away_team} | Home: {home_team}")
    if pna_set:
        print(f"predicted_not_active rows: {len(pna_set)}")

    # Per-team summaries
    def _team_summary(team: str) -> dict[str, int]:
        def _slot_out(pos: str, slot: int) -> int:
            mm = m[(m["team"] == team) & (m["pos_up"] == pos) & (m["_slot"] == slot)]
            if mm.empty:
                return 0
            return int(mm.iloc[0]["out"])

        qb_out = _slot_out("QB", 1)
        rb1_out = _slot_out("RB", 1)
        te1_out = _slot_out("TE", 1)
        wr1_out = _slot_out("WR", 1)
        wr2_out = _slot_out("WR", 2)
        return {
            "inj_qb_out": qb_out,
            "inj_rb1_out": rb1_out,
            "inj_te1_out": te1_out,
            "inj_wr1_out": wr1_out,
            "inj_wr_top2_out": int(wr1_out + wr2_out),
            "inj_starters_out": int(qb_out + rb1_out + te1_out + wr1_out),
        }

    # If we have team names from games.csv, print those first
    if away_team and home_team:
        for label, team in [("away", away_team), ("home", home_team)]:
            try:
                from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team  # type: ignore
            except Exception:
                _norm_team = lambda s: str(s)
            tnorm = _norm_team(team)
            if not (m["team"] == tnorm).any():
                print(f"\n{label.upper()} team not found in depth charts after normalization: {team} -> {tnorm}")
                continue
            summ = _team_summary(tnorm)
            print(f"\n{label.upper()} team summary: {tnorm} | {summ}")
            tt = m[m["team"] == tnorm].sort_values(["pos_up", "_slot"])
            print(tt[["pos_up", "_slot", "player", "in_current", "active_now", "out", "reason"]].to_string(index=False))

    # Also show any other teams present in baseline slots (for debugging)
    # Note: baseline slots include every team in the league; we only print the game teams above.

    # Cross-check with merge_features output
    try:
        lines = load_lines()
        team_stats = load_team_stats()
        wx = load_weather_for_games(games)
        feats = merge_features(games, team_stats, lines, wx)
        row = feats[feats["game_id"].astype(str) == str(game_id)].copy()
        if not row.empty:
            r0 = row.iloc[0]
            cols = [
                "home_inj_starters_out",
                "away_inj_starters_out",
                "inj_starters_out_total",
                "inj_starters_out_abs_diff",
                "inj_starters_out_diff",
            ]
            have = [c for c in cols if c in row.columns]
            if have:
                print("\nmerge_features injury columns:")
                print(r0[have].to_dict())
    except Exception as e:
        print(f"\nWARNING: merge_features cross-check failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Drill down baseline-starter injury flags for a single game_id")
    ap.add_argument("game_id", type=str, help="e.g. 2025_09_ARI_DAL")
    args = ap.parse_args()

    drilldown_game(str(args.game_id).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
