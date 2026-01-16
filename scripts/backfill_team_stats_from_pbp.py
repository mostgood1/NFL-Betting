import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "nfl_compare" / "data"


def _safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def _load_pbp(season: int) -> pd.DataFrame:
    p = DATA_DIR / f"pbp_{season}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")


def _weekly_rollups_from_pbp(pbp: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (offense_rollup, defense_rollup) with cumulative-through-prior-week rates.

        Output columns (per team, week):
            off_sack_rate, def_sack_rate, off_rz_pass_rate, def_rz_pass_rate

        The values for week w are computed from plays in weeks <= w (cumulative through that week).
        This aligns with the feature merge behavior that uses week-1 stats for games in week w+1.
    """
    if pbp is None or pbp.empty:
        return (pd.DataFrame(), pd.DataFrame())

    df = pbp.copy()

    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team  # type: ignore
    except Exception:
        _norm_team = lambda s: str(s)

    # Basic filters
    if "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce").eq(int(season))]
    if "week" not in df.columns:
        return (pd.DataFrame(), pd.DataFrame())

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df[df["week"].notna()].copy()
    df["week"] = df["week"].astype(int)

    for c in ["posteam", "defteam"]:
        if c in df.columns:
            # nflfastR pbp uses abbreviations (e.g., ARI, ATL). Normalize to full team names
            # to match the rest of this repo.
            df[c] = df[c].astype(str).apply(_norm_team)

    qb_dropback = pd.to_numeric(df.get("qb_dropback"), errors="coerce") if "qb_dropback" in df.columns else pd.Series(0, index=df.index)
    sack = pd.to_numeric(df.get("sack"), errors="coerce") if "sack" in df.columns else pd.Series(np.nan, index=df.index)
    yardline_100 = pd.to_numeric(df.get("yardline_100"), errors="coerce") if "yardline_100" in df.columns else pd.Series(np.nan, index=df.index)

    is_dropback = qb_dropback.fillna(0).eq(1)
    is_sack = sack.fillna(0).eq(1) & is_dropback

    # Red zone definition: within 20 yards of end zone.
    in_rz = yardline_100.notna() & (yardline_100 <= 20)

    # Treat any QB dropback as a pass attempt proxy; rush plays use `rush==1` if present.
    is_rz_pass = in_rz & is_dropback
    rush_flag = pd.to_numeric(df.get("rush"), errors="coerce") if "rush" in df.columns else pd.Series(np.nan, index=df.index)
    is_rz_rush = in_rz & rush_flag.fillna(0).eq(1)
    is_rz_total = is_rz_pass | is_rz_rush

    # OFFENSE weekly sums
    off = df[df.get("posteam").notna()].copy()
    off_roll = (
        off.assign(
            _dropback=is_dropback.loc[off.index].astype(int),
            _sack=is_sack.loc[off.index].astype(int),
            _rz_pass=is_rz_pass.loc[off.index].astype(int),
            _rz_total=is_rz_total.loc[off.index].astype(int),
        )
        .groupby(["posteam", "week"], as_index=False)
        .agg(
            off_dropbacks=("_dropback", "sum"),
            off_sacks=("_sack", "sum"),
            off_rz_pass_plays=("_rz_pass", "sum"),
            off_rz_total_plays=("_rz_total", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )

    # DEFENSE weekly sums (by defteam)
    deff = df[df.get("defteam").notna()].copy()
    def_roll = (
        deff.assign(
            _dropback=is_dropback.loc[deff.index].astype(int),
            _sack=is_sack.loc[deff.index].astype(int),
            _rz_pass=is_rz_pass.loc[deff.index].astype(int),
            _rz_total=is_rz_total.loc[deff.index].astype(int),
        )
        .groupby(["defteam", "week"], as_index=False)
        .agg(
            def_dropbacks_faced=("_dropback", "sum"),
            def_sacks=("_sack", "sum"),
            def_rz_pass_faced=("_rz_pass", "sum"),
            def_rz_total_faced=("_rz_total", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )

    # Build cumulative-through-current-week rates
    # (Games in week W+1 should use the stats row from week W to avoid leakage.)
    def _cum_through_week(g: pd.DataFrame, num: str, den: str) -> pd.Series:
        g = g.sort_values("week")
        num_c = g[num].cumsum()
        den_c = g[den].cumsum()
        return num_c / den_c.replace({0: np.nan})

    off_roll = off_roll.sort_values(["team", "week"])
    off_roll["off_sack_rate"] = off_roll.groupby("team", group_keys=False).apply(lambda g: _cum_through_week(g, "off_sacks", "off_dropbacks"))
    off_roll["off_rz_pass_rate"] = off_roll.groupby("team", group_keys=False).apply(lambda g: _cum_through_week(g, "off_rz_pass_plays", "off_rz_total_plays"))

    def_roll = def_roll.sort_values(["team", "week"])
    def_roll["def_sack_rate"] = def_roll.groupby("team", group_keys=False).apply(lambda g: _cum_through_week(g, "def_sacks", "def_dropbacks_faced"))
    def_roll["def_rz_pass_rate"] = def_roll.groupby("team", group_keys=False).apply(lambda g: _cum_through_week(g, "def_rz_pass_faced", "def_rz_total_faced"))

    return (
        off_roll[["team", "week", "off_sack_rate", "off_rz_pass_rate"]].copy(),
        def_roll[["team", "week", "def_sack_rate", "def_rz_pass_rate"]].copy(),
    )


def _qb_adj_from_depth_chart(season: int, week: int) -> pd.DataFrame:
    p = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["team", "week", "qb_adj"])
    try:
        dc = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["team", "week", "qb_adj"])

    if dc is None or dc.empty:
        return pd.DataFrame(columns=["team", "week", "qb_adj"])

    d = dc.copy()
    d["position"] = d.get("position", "").astype(str).str.upper()
    qb = d[d["position"] == "QB"].copy()
    if qb.empty:
        return pd.DataFrame(columns=["team", "week", "qb_adj"])

    if "active" not in qb.columns:
        qb["active"] = True
    if "depth_rank" not in qb.columns:
        qb["depth_rank"] = qb.groupby("team").cumcount() + 1

    qb = qb.sort_values(["team", "depth_rank", "player" if "player" in qb.columns else "depth_rank"])
    starter = qb.groupby("team", as_index=False).first()[["team", "active"]]
    starter["qb_adj"] = starter["active"].astype(bool).map(lambda ok: 0.0 if ok else -1.0)
    starter["week"] = int(week)
    return starter[["team", "week", "qb_adj"]]


def backfill(season: int, weeks: list[int] | None = None) -> None:
    team_stats_path = DATA_DIR / "team_stats.csv"
    if not team_stats_path.exists():
        raise FileNotFoundError(f"Missing {team_stats_path}")

    ts = pd.read_csv(team_stats_path)
    ts["season"] = pd.to_numeric(ts.get("season"), errors="coerce")
    ts["week"] = pd.to_numeric(ts.get("week"), errors="coerce")
    ts = ts[ts["season"].eq(int(season)) & ts["week"].notna()].copy()
    ts["week"] = ts["week"].astype(int)

    if weeks is not None and len(weeks) > 0:
        ts = ts[ts["week"].isin([int(w) for w in weeks])].copy()

    pbp = _load_pbp(season)
    off_roll, def_roll = _weekly_rollups_from_pbp(pbp, season)

    # Join sack / RZ rates
    out = pd.read_csv(team_stats_path)
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
    out["week"] = pd.to_numeric(out.get("week"), errors="coerce")

    # Only update selected weeks for this season (or all for season)
    if weeks is None or len(weeks) == 0:
        mask = out["season"].eq(int(season))
    else:
        mask = out["season"].eq(int(season)) & out["week"].isin([int(w) for w in weeks])

    block = out[mask].copy()
    block["week"] = pd.to_numeric(block.get("week"), errors="coerce").astype(int)

    # If these columns already exist in team_stats.csv, the merge below would suffix them
    # (e.g. off_sack_rate_x/off_sack_rate_y) and we'd never update the intended columns.
    # Drop them from the update block so the rollup merge brings them in with clean names.
    for c in ["off_sack_rate", "def_sack_rate", "off_rz_pass_rate", "def_rz_pass_rate"]:
        if c in block.columns:
            block = block.drop(columns=[c])

    block = block.merge(off_roll, on=["team", "week"], how="left")
    block = block.merge(def_roll, on=["team", "week"], how="left")

    # QB adjust per team/week from depth chart (optional)
    qb_rows = []
    wk_list = sorted(block["week"].dropna().unique().astype(int).tolist())
    for w in wk_list:
        qb_rows.append(_qb_adj_from_depth_chart(season, int(w)))
    qb = pd.concat(qb_rows, ignore_index=True) if qb_rows else pd.DataFrame(columns=["team", "week", "qb_adj"])
    if not qb.empty:
        block = block.merge(qb, on=["team", "week"], how="left", suffixes=("", "_new"))
        if "qb_adj_new" in block.columns:
            block["qb_adj"] = block["qb_adj_new"].where(block["qb_adj_new"].notna(), block.get("qb_adj"))
            block = block.drop(columns=["qb_adj_new"])

    # Write back into out
    key_cols = ["season", "week", "team"]
    update_cols = [c for c in ["off_sack_rate", "def_sack_rate", "off_rz_pass_rate", "def_rz_pass_rate", "qb_adj"] if c in block.columns]

    # Ensure columns exist
    for c in update_cols:
        if c not in out.columns:
            out[c] = np.nan

    # Build lookup for updates
    upd = block[key_cols + update_cols].copy()

    # Merge updates into original out
    out2 = out.merge(upd, on=key_cols, how="left", suffixes=("", "__upd"))
    for c in update_cols:
        uc = f"{c}__upd"
        if uc in out2.columns:
            out2[c] = out2[uc].where(out2[uc].notna(), out2[c])
            out2 = out2.drop(columns=[uc])

    # Backup + write
    backup = team_stats_path.with_suffix(".csv.backup")
    try:
        team_stats_path.replace(backup)
    except Exception:
        # If replace fails (e.g., open file), just write without backup
        backup = None

    out2.to_csv(team_stats_path, index=False)
    print(f"Updated {team_stats_path} with {len(upd)} rows of sack/rz/qb features")
    if backup is not None:
        print(f"Backup saved to {backup}")


def _parse_weeks(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if str(x).strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill team_stats.csv with sack/redzone rates from pbp parquet")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", type=str, default="", help="Week list like '17-18' or '1,2,3'. Empty => all weeks in file for season")
    args = p.parse_args()

    weeks = _parse_weeks(args.weeks) if args.weeks else None
    backfill(int(args.season), weeks=weeks)


if __name__ == "__main__":
    main()
