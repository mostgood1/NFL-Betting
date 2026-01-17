import sys
import os
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (ROOT / "nfl_compare" / "data")


def _league_avg_rates_from_pbp(season: int, through_week: int) -> tuple[float, float]:
    """Return (penalty_rate_per_play, dpi_rate_per_play) using pbp parquet."""
    pbp_fp = DATA_DIR / f"pbp_{int(season)}.parquet"
    if not pbp_fp.exists():
        return (0.0, 0.0)

    cols = ["week", "penalty", "penalty_type"]
    pbp = pd.read_parquet(pbp_fp, columns=cols)
    pbp["week"] = pd.to_numeric(pbp.get("week"), errors="coerce")
    pbp = pbp[pbp["week"].notna()].copy()
    pbp["week"] = pbp["week"].astype(int)
    pbp = pbp[pbp["week"] <= int(through_week)].copy()
    if pbp.empty:
        return (0.0, 0.0)

    pen = pbp["penalty"].fillna(0.0)
    # In nflfastR pbp, penalty is 1.0 on penalty plays
    p_pen = float((pen == 1.0).mean())

    pen_type = pbp.get("penalty_type")
    if pen_type is None:
        return (p_pen, 0.0)

    is_dpi = (
        (pen == 1.0)
        & pen_type.notna()
        & pen_type.astype(str).str.contains("Defensive Pass Interference", case=False, na=False)
    )
    p_dpi = float(is_dpi.mean())

    return (p_pen, p_dpi)


def build(season: int, week: int, out: Path) -> pd.DataFrame:
    games_fp = DATA_DIR / "games.csv"
    if not games_fp.exists():
        raise FileNotFoundError(f"Missing games.csv at {games_fp}")

    games = pd.read_csv(games_fp)
    games["season"] = pd.to_numeric(games.get("season"), errors="coerce")
    games["week"] = pd.to_numeric(games.get("week"), errors="coerce")
    g = games[(games["season"] == int(season)) & (games["week"] == int(week))].copy()
    if g.empty:
        raise ValueError(f"No games found for season={season} week={week}")

    prev_week = max(1, int(week) - 1)
    p_pen, p_dpi = _league_avg_rates_from_pbp(int(season), through_week=prev_week)

    out_df = pd.DataFrame({
        "season": int(season),
        "week": int(week),
        "game_id": g["game_id"].astype(str),
        "crew_name": "LEAGUE_AVG",
        "crew_penalty_rate": float(p_pen),
        "crew_dpi_rate": float(p_dpi),
        # pace adj is a neutral placeholder until a real crew feed is available
        "crew_pace_adj": 0.0,
    })

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "officiating_crews.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    df = build(args.season, args.week, out)
    print(f"Wrote officiating_crews rows={len(df)} to {out}")
    print(df[["crew_name", "crew_penalty_rate", "crew_dpi_rate", "crew_pace_adj"]].head(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
