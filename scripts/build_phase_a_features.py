import argparse
import os
from pathlib import Path
import pandas as pd

try:
    from nfl_data_py import import_pbp_data as _import_pbp
except Exception:
    _import_pbp = None


DATA_DIR = Path(os.environ.get("NFL_DATA_DIR") or (Path(__file__).resolve().parents[1] / "nfl_compare" / "data"))


def _safe_import_pbp(seasons: list[int]) -> pd.DataFrame:
    if _import_pbp is None:
        return pd.DataFrame()
    try:
        df = _import_pbp(seasons)
        return df
    except Exception:
        # Some versions use a different function name
        try:
            from nfl_data_py import import_pbp
            return import_pbp(seasons)
        except Exception:
            return pd.DataFrame()


def _team_key(df: pd.DataFrame, side: str) -> pd.Series:
    # nflfastR uses posteam/defteam
    col = f"{side}team" if f"{side}team" in df.columns else (f"{side}_team" if f"{side}_team" in df.columns else None)
    if col is None:
        col = "posteam" if side == "off" else "defteam"
    return df.get(col, pd.Series([None] * len(df)))


def build_explosive_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["season","week","team","explosive_pass_rate","explosive_run_rate"])
    df = pbp.copy()
    # Normalize basics
    for c in ["season","week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    off = _team_key(df, "off")
    df["team"] = off.astype(str)
    # Explosive thresholds (common: pass>=20, run>=15 yards)
    yds = pd.to_numeric(df.get("yards_gained"), errors="coerce")
    is_pass = df.get("pass_attempt", pd.Series([0]*len(df))).astype(bool)
    is_run = df.get("rush_attempt", pd.Series([0]*len(df))).astype(bool)
    exp_pass = is_pass & (yds >= 20)
    exp_run = is_run & (yds >= 15)
    grp = df.groupby(["season","week","team"], dropna=True)
    out = grp.apply(lambda g: pd.Series({
        "explosive_pass_rate": float((g[exp_pass].shape[0]) / max(g[is_pass].shape[0], 1)),
        "explosive_run_rate": float((g[exp_run].shape[0]) / max(g[is_run].shape[0], 1)),
    })).reset_index()
    return out


def build_drive_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["season","week","team","drives","points_per_drive","td_per_drive","fg_per_drive","avg_start_fp","yards_per_drive","seconds_per_drive"])
    df = pbp.copy()
    # Ensure needed fields
    for c in ["season","week","drive","drive_play_count","drive_ended_with_score","drive_end_result","yardline_100","game_seconds_remaining","game_seconds","qtr","quarter_seconds_remaining"]:
        if c not in df.columns:
            df[c] = pd.NA
    off = _team_key(df, "off").astype(str)
    df["team"] = off
    # Points on a play
    pts = pd.to_numeric(df.get("touchdown", pd.Series([0]*len(df))), errors="coerce").fillna(0) * 6
    # Field goals made
    fg_made = (df.get("field_goal_result", pd.Series([None]*len(df))).astype(str).str.lower() == "made")
    pts = pts + fg_made.astype(int) * 3
    df["pts_play"] = pts
    # Derive a monotonic time axis for ordering and duration.
    # nflfastR-style pbp typically has `game_seconds_remaining`.
    time_col = None
    if "game_seconds_remaining" in df.columns and df["game_seconds_remaining"].notna().any():
        time_col = "game_seconds_remaining"
        time_remaining = True
    elif "game_seconds" in df.columns and df["game_seconds"].notna().any():
        time_col = "game_seconds"
        time_remaining = False
    else:
        # Fallback: approximate game seconds remaining from quarter + quarter seconds remaining.
        # game_seconds_remaining ~= (4-qtr)*900 + quarter_seconds_remaining
        qtr = pd.to_numeric(df.get("qtr"), errors="coerce")
        qsr = pd.to_numeric(df.get("quarter_seconds_remaining"), errors="coerce")
        approx = (4 - qtr) * 900 + qsr
        df["__approx_game_seconds_remaining"] = approx
        time_col = "__approx_game_seconds_remaining"
        time_remaining = True

    t = pd.to_numeric(df.get(time_col), errors="coerce")
    df["__t"] = t

    # Drive start yardline approximated by first play in drive.
    # If using a *remaining* clock, earlier plays have larger values => sort descending.
    df = df.sort_values(
        ["season", "week", "team", "drive", "__t"],
        ascending=[True, True, True, True, (not time_remaining)],
    )
    start_fp = df.groupby(["season","week","team","drive"], dropna=True)["yardline_100"].first()
    dur = df.groupby(["season","week","team","drive"], dropna=True)["__t"].agg(lambda s: (s.max() - s.min()) if (s.notna().any()) else pd.NA)
    yards = pd.to_numeric(df.get("yards_gained"), errors="coerce")
    yards_drive = df.assign(_yards=yards).groupby(["season","week","team","drive"], dropna=True)["_yards"].sum()
    pts_drive = df.groupby(["season","week","team","drive"], dropna=True)["pts_play"].sum()
    td_drive = df.groupby(["season","week","team","drive"], dropna=True)["touchdown"].sum()
    fg_drive = df.groupby(["season","week","team","drive"], dropna=True)["field_goal_result"].apply(lambda s: int((s.astype(str).str.lower() == "made").any()))
    agg = pd.concat([start_fp.rename("start_fp"), dur.rename("seconds"), yards_drive.rename("yards"), pts_drive.rename("points"), td_drive.rename("tds"), fg_drive.rename("fgs")], axis=1).reset_index()
    # Per team-week aggregates
    grp = agg.groupby(["season","week","team"], dropna=True)
    drives = grp.size().rename("drives").reset_index()
    out = drives.copy()
    tmp = grp["points"].mean().rename("points_per_drive").reset_index()
    out = out.merge(tmp, on=["season","week","team"], how="left")
    tmp = grp["tds"].mean().rename("td_per_drive").reset_index(); out = out.merge(tmp, on=["season","week","team"], how="left")
    tmp = grp["fgs"].mean().rename("fg_per_drive").reset_index(); out = out.merge(tmp, on=["season","week","team"], how="left")
    tmp = grp["start_fp"].mean().rename("avg_start_fp").reset_index(); out = out.merge(tmp, on=["season","week","team"], how="left")
    tmp = grp["yards"].mean().rename("yards_per_drive").reset_index(); out = out.merge(tmp, on=["season","week","team"], how="left")
    tmp = grp["seconds"].mean().rename("seconds_per_drive").reset_index(); out = out.merge(tmp, on=["season","week","team"], how="left")
    return out


def build_redzone_splits(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["season","week","team","rzd_off_td_rate","rzd_def_td_rate","rzd_off_eff","rzd_def_eff"])
    df = pbp.copy()
    for c in ["season","week","yardline_100","touchdown","posteam","defteam","pass_attempt","rush_attempt"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce")
    in_rz = df["yardline_100"].le(20)
    off = _team_key(df, "off").astype(str)
    df["team"] = off
    # Offense TD rate inside RZ
    grp_off = df[in_rz].groupby(["season","week","team"], dropna=True)
    off_td_rate = grp_off["touchdown"].mean().rename("rzd_off_td_rate").reset_index()
    # Simple efficiency proxy: success via EPA>0
    epa = pd.to_numeric(df.get("epa"), errors="coerce").fillna(0.0)
    df["success"] = (epa > 0).astype(int)
    off_eff = grp_off["success"].mean().rename("rzd_off_eff").reset_index()
    out = off_td_rate.merge(off_eff, on=["season","week","team"], how="outer")
    # Defense TD rate allowed inside RZ
    df_def = df.copy(); df_def["team"] = _team_key(df_def, "def").astype(str)
    grp_def = df_def[in_rz].groupby(["season","week","team"], dropna=True)
    def_td_rate = grp_def["touchdown"].mean().rename("rzd_def_td_rate").reset_index()
    def_eff = grp_def["success"].mean().rename("rzd_def_eff").reset_index()
    out = out.merge(def_td_rate, on=["season","week","team"], how="outer")
    out = out.merge(def_eff, on=["season","week","team"], how="outer")
    return out


def build_penalties_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["season","week","team","penalty_rate","turnover_adj_rate"])
    df = pbp.copy()
    for c in ["season","week","penalty","interception","fumble_lost"]:
        if c not in df.columns:
            df[c] = 0
    off = _team_key(df, "off").astype(str)
    df["team"] = off
    grp = df.groupby(["season","week","team"], dropna=True)
    penalty_rate = grp["penalty"].mean().rename("penalty_rate").reset_index()
    to_rate = (grp["interception"].mean() + grp["fumble_lost"].mean()).rename("turnover_adj_rate").reset_index()
    out = penalty_rate.merge(to_rate, on=["season","week","team"], how="outer")
    return out


def build_special_teams(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["season","week","team","fg_acc","punt_epa","kick_return_epa","touchback_rate"])
    df = pbp.copy()
    off = _team_key(df, "off").astype(str)
    df["team"] = off
    epa = pd.to_numeric(df.get("epa"), errors="coerce")
    df["epa"] = epa
    # FG accuracy: made / attempts
    is_fg = (df.get("field_goal_attempt", pd.Series([0]*len(df))).astype(bool))
    fg_made = (df.get("field_goal_result", pd.Series([None]*len(df))).astype(str).str.lower() == "made")
    # Punt EPA: mean EPA on punt plays
    is_punt = (df.get("punt_attempt", pd.Series([0]*len(df))).astype(bool))
    # Kick return EPA: kickoff returned plays
    is_kickoff = (df.get("kickoff_attempt", pd.Series([0]*len(df))).astype(bool))
    is_return = (df.get("return_player_id", pd.Series([None]*len(df))).notna())
    is_touchback = (df.get("touchback", pd.Series([0]*len(df))).astype(bool))
    grp = df.groupby(["season","week","team"], dropna=True)
    fg_acc = (grp.apply(lambda g: float((fg_made[g.index]).sum()) / max(int(is_fg[g.index].sum()), 1))).rename("fg_acc").reset_index()
    punt_epa = grp.apply(lambda g: float(df.loc[g.index]["epa"][is_punt[g.index]].mean()) if int(is_punt[g.index].sum())>0 else 0.0).rename("punt_epa").reset_index()
    kick_return_epa = grp.apply(lambda g: float(df.loc[g.index]["epa"][is_kickoff[g.index] & is_return[g.index]].mean()) if int((is_kickoff[g.index] & is_return[g.index]).sum())>0 else 0.0).rename("kick_return_epa").reset_index()
    touchback_rate = grp.apply(lambda g: float(is_touchback[g.index].mean())).rename("touchback_rate").reset_index()
    out = fg_acc.merge(punt_epa, on=["season","week","team"], how="outer")
    out = out.merge(kick_return_epa, on=["season","week","team"], how="outer")
    out = out.merge(touchback_rate, on=["season","week","team"], how="outer")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build Phase A team-week features from nfl-data-py pbp")
    ap.add_argument("--season", type=int, required=True)
    # Include postseason by default (NFL uses weeks up through ~22 incl. Super Bowl)
    ap.add_argument("--end-week", type=int, default=22)
    args = ap.parse_args()

    seasons = [args.season]
    pbp = _safe_import_pbp(seasons)
    if pbp is None or pbp.empty:
        print("pbp import failed or empty; no Phase A features generated")
        return
    # Filter to season and weeks
    pbp["season"] = pd.to_numeric(pbp["season"], errors="coerce")
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce")
    pbp = pbp[(pbp["season"] == args.season) & (pbp["week"].between(1, args.end_week, inclusive="both"))]

    # Build and write each CSV
    outputs = [
        (build_drive_stats(pbp), DATA_DIR / "pfr_drive_stats.csv"),
        (build_redzone_splits(pbp), DATA_DIR / "redzone_splits.csv"),
        (build_explosive_rates(pbp), DATA_DIR / "explosive_rates.csv"),
        (build_penalties_stats(pbp), DATA_DIR / "penalties_stats.csv"),
        (build_special_teams(pbp), DATA_DIR / "special_teams.csv"),
    ]
    for df, fp in outputs:
        try:
            if df is not None and not df.empty:
                fp.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(fp, index=False)
                print(f"Wrote {fp} rows={len(df)}")
            else:
                print(f"Skipped {fp.name}: empty")
        except Exception as e:
            print(f"Failed to write {fp}: {e}")


if __name__ == "__main__":
    main()
