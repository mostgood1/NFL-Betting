import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "nfl_compare" / "data"


def infer_season_week() -> tuple[int, int]:
    cw = DATA_DIR / "current_week.json"
    if cw.exists():
        try:
            js = json.loads(cw.read_text(encoding="utf-8"))
            return int(js.get("season")), int(js.get("week"))
        except Exception:
            pass
    # Fallback: try to parse latest props file present
    cand = sorted(DATA_DIR.glob("player_props_*_wk*.csv"))
    if cand:
        last = cand[-1].name
        try:
            # player_props_2025_wk7.csv
            parts = last.replace("player_props_", "").replace(".csv", "").split("_wk")
            season = int(parts[0])
            week = int(parts[1])
            return season, week
        except Exception:
            pass
    # Default to current season guess
    return 2025, 1


def load_depth(season: int, week: int) -> pd.DataFrame:
    fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def main() -> int:
    season, week = infer_season_week()
    props_fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
    if not props_fp.exists():
        print(f"FAIL: props file missing: {props_fp}")
        return 2
    try:
        props = pd.read_csv(props_fp)
    except Exception as e:
        print(f"FAIL: could not read props csv: {e}")
        return 3

    depth = load_depth(season, week)
    if depth.empty:
        print("WARN: depth chart CSV missing; skipping starters presence check.")
        return 0

    # Filter to WR/TE starters and active per ESPN
    depth = depth.copy()
    depth["position"] = depth["position"].astype(str).str.upper()
    depth = depth[(depth["position"].isin(["WR", "TE"])) & (depth["depth_rank"] == 1)]
    if "active" in depth.columns:
        depth = depth[depth["active"] == True]
    starters = depth[["team", "player", "position"]].dropna()
    if starters.empty:
        print("WARN: no WR/TE starters found in depth chart; skipping.")
        return 0

    # Normalize props names via canonicalization if present
    # Ensure project root is importable for nfl_compare package
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from nfl_compare.src.name_normalizer import canonical_player_name  # type: ignore
    pcol = None
    for c in ("display_name", "player", "name", "player_name"):
        if c in props.columns:
            pcol = c; break
    if not pcol:
        print("FAIL: props csv missing player column")
        return 4
    try:
        props[pcol] = props[pcol].astype(str).apply(canonical_player_name)
    except Exception:
        props[pcol] = props[pcol].astype(str)

    # Join starters against props by team + player strict lower
    starters["__player_key"] = starters["player"].astype(str).apply(canonical_player_name).str.lower()
    props["__player_key"] = props[pcol].astype(str).str.lower()

    tcol = None
    for c in ("team", "team_abbr", "team_code", "posteam", "pos_team"):
        if c in props.columns:
            tcol = c; break
    # If a team column exists, restrict starters to those teams found in props (avoid partial-week false negatives)
    if tcol:
        present_teams = set(props[tcol].astype(str).str.strip().str.upper().unique())
        starters = starters[starters["team"].astype(str).str.strip().str.upper().isin(present_teams)].copy()
    # If no team in props, check presence by name only
    if tcol:
        starters["__team_k"] = starters["team"].astype(str).str.strip().str.upper()
        props["__team_k"] = props[tcol].astype(str).str.strip().str.upper()
        merged = starters.merge(
            props[["__team_k", "__player_key"]].drop_duplicates(),
            on=["__team_k", "__player_key"], how="left", indicator=True
        )
    else:
        merged = starters.merge(
            props[["__player_key"]].drop_duplicates(),
            on=["__player_key"], how="left", indicator=True
        )

    missing = merged[merged["_merge"] == "left_only"].copy()
    if not missing.empty:
        sample = missing[["team", "player", "position"]].head(10).to_dict(orient="records")
        print("FAIL: WR/TE starters missing from props:")
        for r in sample:
            print(" -", r)
        return 5

    print(f"OK: All ESPN WR/TE starters present in props for {season} wk{week}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
