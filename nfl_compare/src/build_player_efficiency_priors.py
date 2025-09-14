from __future__ import annotations

"""
Build player-level efficiency priors from play-by-play data.

Outputs a CSV at data/player_efficiency_priors.csv with, per player:
- player_id (when available), player (display name), position
- targets, receptions, rec_yards, catch_rate, ypt, rz_targets, rz_target_rate
- rush_att, rush_yards, ypc, rz_rush_att, rz_rush_rate

We aggregate across the provided seasons (e.g., last 3 years + current).
These priors are later blended with league/position baselines in props.
"""

from pathlib import Path
from typing import Iterable, Optional, List, Dict

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_pbp_seasons(seasons: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for s in sorted(set(int(x) for x in seasons)):
        fp = DATA_DIR / f"pbp_{s}.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
            df["season"] = int(s)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    # Use minimal set of columns to reduce memory
    use_cols = [
        "season","pass","complete_pass","rush","yardline_100",
        # receiver fields (nflfastR / nfl_data_py naming variants)
        "receiver_player_name","receiver_player_id","receiver_id","receiver","receiver_name",
        # rusher fields
        "rusher_player_name","rusher_player_id","rusher_id","rusher","rusher_name",
    ]
    # Keep only columns that exist in each frame
    for i, df in enumerate(frames):
        keep = [c for c in use_cols if c in df.columns]
        frames[i] = df[keep].copy()
    out = pd.concat(frames, ignore_index=True)
    # Coerce numeric flags
    for c in ["pass","complete_pass","rush"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if "yardline_100" in out.columns:
        out["yardline_100"] = pd.to_numeric(out["yardline_100"], errors="coerce")
    return out


def _pick_name(row: pd.Series, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and pd.notna(v):
            s = str(v).strip()
            if s:
                return s
    return ""


def _receiver_df(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    df = pbp.copy()
    if "pass" not in df.columns:
        return pd.DataFrame()
    # A target occurs on a pass play with a receiver identified
    name_cols = ["receiver_player_name","receiver_name","receiver"]
    id_cols = ["receiver_player_id","receiver_id"]
    df["receiver_name_"] = df.apply(lambda r: _pick_name(r, name_cols), axis=1)
    df["receiver_id_"] = df.apply(lambda r: _pick_name(r, id_cols), axis=1)
    has_tgt = (df["pass"] == 1) & (df["receiver_name_"].astype(str).str.len() > 0)
    if "complete_pass" not in df.columns:
        df["complete_pass"] = 0
    rec = df[has_tgt].copy()
    rec["target"] = 1
    rec["comp"] = pd.to_numeric(rec["complete_pass"], errors="coerce").fillna(0).astype(int)
    # receiving_yards column name varies; try a few
    ry = None
    for c in ["receiving_yards","yards_gained","air_yards"]:
        if c in rec.columns:
            ry = c
            break
    if ry is None:
        rec["rec_yards"] = 0.0
    else:
        rec["rec_yards"] = pd.to_numeric(rec[ry], errors="coerce").fillna(0.0)
    if "yardline_100" in rec.columns:
        rec["rz"] = (pd.to_numeric(rec["yardline_100"], errors="coerce") <= 20).astype(int)
    else:
        rec["rz"] = 0
    g = (
        rec.groupby(["receiver_id_","receiver_name_"])
           .agg(
               targets=("target","sum"),
               receptions=("comp","sum"),
               rec_yards=("rec_yards","sum"),
               rz_targets=("rz","sum"),
           )
           .reset_index()
    )
    if g.empty:
        return g
    g["catch_rate"] = np.where(g["targets"] > 0, g["receptions"] / g["targets"], np.nan)
    g["ypt"] = np.where(g["targets"] > 0, g["rec_yards"] / g["targets"], np.nan)
    g["rz_target_rate"] = np.where(g["targets"] > 0, g["rz_targets"] / g["targets"], np.nan)
    g = g.rename(columns={"receiver_id_":"player_id","receiver_name_":"player"})
    return g


def _rusher_df(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    df = pbp.copy()
    if "rush" not in df.columns:
        return pd.DataFrame()
    name_cols = ["rusher_player_name","rusher_name","rusher"]
    id_cols = ["rusher_player_id","rusher_id"]
    df["rusher_name_"] = df.apply(lambda r: _pick_name(r, name_cols), axis=1)
    df["rusher_id_"] = df.apply(lambda r: _pick_name(r, id_cols), axis=1)
    run = df[(df["rush"] == 1) & (df["rusher_name_"].astype(str).str.len() > 0)].copy()
    run["rush_att"] = 1
    # rushing_yards column name varies; try a few
    ry = None
    for c in ["rushing_yards","yards_gained"]:
        if c in run.columns:
            ry = c
            break
    if ry is None:
        run["rush_yards"] = 0.0
    else:
        run["rush_yards"] = pd.to_numeric(run[ry], errors="coerce").fillna(0.0)
    if "yardline_100" in run.columns:
        run["rz"] = (pd.to_numeric(run["yardline_100"], errors="coerce") <= 20).astype(int)
    else:
        run["rz"] = 0
    g = (
        run.groupby(["rusher_id_","rusher_name_"])
           .agg(
               rush_att=("rush_att","sum"),
               rush_yards=("rush_yards","sum"),
               rz_rush_att=("rz","sum"),
           )
           .reset_index()
    )
    if g.empty:
        return g
    g["ypc"] = np.where(g["rush_att"] > 0, g["rush_yards"] / g["rush_att"], np.nan)
    g["rz_rush_rate"] = np.where(g["rush_att"] > 0, g["rz_rush_att"] / g["rush_att"], np.nan)
    g = g.rename(columns={"rusher_id_":"player_id","rusher_name_":"player"})
    return g


def _attach_positions(df: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        import nfl_data_py as nfl  # type: ignore
        ros = nfl.import_seasonal_rosters([int(s) for s in seasons])
    except Exception:
        ros = pd.DataFrame()
    if ros is None or ros.empty:
        df["position"] = pd.NA
        return df
    # Build id -> position/name map
    id_col = None
    for c in ["gsis_id","player_id","nfl_id","pfr_id"]:
        if c in ros.columns:
            id_col = c
            break
    name_col = None
    for c in ["player_display_name","player_name","display_name","full_name","football_name"]:
        if c in ros.columns:
            name_col = c
            break
    pos_col = None
    for c in ["position","depth_chart_position"]:
        if c in ros.columns:
            pos_col = c
            break
    if not id_col and not name_col:
        df["position"] = pd.NA
        return df
    m = ros[[c for c in [id_col, name_col, pos_col] if c]].copy().drop_duplicates()
    # Normalize pos
    if pos_col in m.columns:
        m[pos_col] = m[pos_col].astype(str).str.upper()
    # Prefer join by id when present
    if id_col and "player_id" in df.columns:
        out = df.merge(m.rename(columns={id_col: "player_id", name_col or "": "_name_join", pos_col or "": "position"}),
                       on="player_id", how="left")
        # Fill missing player names from roster mapping
        if "player" in out.columns and "_name_join" in out.columns:
            out["player"] = out["player"].where(out["player"].astype(str).str.len() > 0, out["_name_join"])
        if "_name_join" in out.columns:
            out = out.drop(columns=["_name_join"])
        return out
    # Fallback: join by name (lower)
    if name_col and "player" in df.columns:
        df["_nm"] = df["player"].astype(str).str.lower()
        m["_nm"] = m[name_col].astype(str).str.lower()
        out = df.merge(m[["_nm", pos_col]].rename(columns={pos_col: "position"}), on="_nm", how="left")
        out = out.drop(columns=["_nm"])
        return out
    df["position"] = pd.NA
    return df


def build_player_efficiency_priors(seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Aggregate PBP into per-player efficiency priors for the given seasons.

    If seasons is None, default to the last 3 complete seasons plus current season (if present).
    """
    if seasons is None:
        # Infer from files available in DATA_DIR
        cand = []
        for p in DATA_DIR.glob("pbp_*.parquet"):
            try:
                y = int(p.stem.split("_")[1])
                cand.append(y)
            except Exception:
                continue
        if cand:
            y_max = max(cand)
            base = sorted(set(cand))
            # last 3 before max plus max
            seasons = sorted({y for y in base if y >= y_max - 3})
        else:
            seasons = []

    pbp = _load_pbp_seasons(seasons)
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=[
            "player_id","player","position","targets","receptions","rec_yards","catch_rate","ypt","rz_targets","rz_target_rate",
            "rush_att","rush_yards","ypc","rz_rush_att","rz_rush_rate",
        ])

    recv = _receiver_df(pbp)
    rush = _rusher_df(pbp)
    # Outer join on (player_id, player)
    pri = pd.merge(recv, rush, on=["player_id","player"], how="outer")
    pri = pri.fillna({
        "targets": 0, "receptions": 0, "rec_yards": 0, "catch_rate": np.nan, "ypt": np.nan, "rz_targets": 0, "rz_target_rate": np.nan,
        "rush_att": 0, "rush_yards": 0, "ypc": np.nan, "rz_rush_att": 0, "rz_rush_rate": np.nan,
    })
    # Attach positions (best effort)
    pri = _attach_positions(pri, seasons)
    # Coerce types
    int_cols = ["targets","receptions","rz_targets","rush_att","rz_rush_att"]
    for c in int_cols:
        if c in pri.columns:
            pri[c] = pd.to_numeric(pri[c], errors="coerce").fillna(0).astype(int)
    for c in ["rec_yards","rush_yards","catch_rate","ypt","rz_target_rate","ypc","rz_rush_rate"]:
        if c in pri.columns:
            pri[c] = pd.to_numeric(pri[c], errors="coerce")

    # Keep a reasonable subset of columns and drop empty players
    keep = [
        "player_id","player","position","targets","receptions","rec_yards","catch_rate","ypt","rz_targets","rz_target_rate",
        "rush_att","rush_yards","ypc","rz_rush_att","rz_rush_rate",
    ]
    pri = pri[[c for c in keep if c in pri.columns]].copy()
    # Drop rows with no activity in either rushing or receiving
    pri = pri.loc[~((pri.get("targets", 0) == 0) & (pri.get("rush_att", 0) == 0))].copy()
    return pri


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build player efficiency priors from PBP Parquet files")
    ap.add_argument("--seasons", nargs="*", type=int, help="Seasons to aggregate (defaults to last 3 + current if available)")
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "player_efficiency_priors.csv"))
    args = ap.parse_args(argv)

    seasons = args.seasons if args.seasons else None
    df = build_player_efficiency_priors(seasons)
    if df is None or df.empty:
        print("No priors built (missing PBP files?)")
        return
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f"Wrote {len(df)} rows -> {out_fp}")


if __name__ == "__main__":
    main()
