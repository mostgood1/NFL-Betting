import sys
import os
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
except Exception:
    _norm_team = lambda s: str(s)

_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (ROOT / "nfl_compare" / "data")


def _pick_col(row: pd.Series, *candidates: str):
    for c in candidates:
        if c in row.index and pd.notna(row.get(c)):
            return row.get(c)
    return pd.NA


def build(season: int, week: int, out: Path) -> pd.DataFrame:
    games_fp = DATA_DIR / "games.csv"
    if not games_fp.exists():
        raise FileNotFoundError(f"Missing games.csv at {games_fp}")

    games = pd.read_csv(games_fp)
    games["season"] = pd.to_numeric(games.get("season"), errors="coerce")
    games["week"] = pd.to_numeric(games.get("week"), errors="coerce")
    games = games[(games["season"] == int(season)) & (games["week"] == int(week))].copy()
    if games.empty:
        raise ValueError(f"No games found for season={season} week={week}")

    date_col = "game_date" if "game_date" in games.columns else ("date" if "date" in games.columns else None)
    if date_col is None:
        raise ValueError("games.csv missing game_date/date")

    out_rows = []
    for _, g in games.iterrows():
        gid = g.get("game_id")
        dt = pd.to_datetime(g.get(date_col), errors="coerce")
        if pd.isna(dt) or gid is None:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        home_team = _norm_team(str(g.get("home_team", ""))).strip()

        wfp = DATA_DIR / f"weather_{date_str}.csv"
        if not wfp.exists():
            # no row; still emit placeholder so merge doesn't miss columns
            out_rows.append({
                "game_id": gid,
                "wx_gust_mph": pd.NA,
                "wx_dew_point_f": pd.NA,
                "_date": date_str,
                "_home_team": home_team,
            })
            continue

        try:
            w = pd.read_csv(wfp)
        except Exception:
            out_rows.append({
                "game_id": gid,
                "wx_gust_mph": pd.NA,
                "wx_dew_point_f": pd.NA,
                "_date": date_str,
                "_home_team": home_team,
            })
            continue

        if "home_team" in w.columns:
            w["home_team"] = w["home_team"].astype(str).map(lambda x: _norm_team(str(x)).strip())

        m = w[w.get("home_team").astype(str) == str(home_team)] if (not w.empty and "home_team" in w.columns) else pd.DataFrame()
        if m.empty:
            out_rows.append({
                "game_id": gid,
                "wx_gust_mph": pd.NA,
                "wx_dew_point_f": pd.NA,
                "_date": date_str,
                "_home_team": home_team,
            })
            continue

        r0 = m.iloc[0]
        gust = _pick_col(r0, "wx_gust_mph", "wx_wind_mph")
        dew = _pick_col(r0, "wx_dew_point_f")

        out_rows.append({
            "game_id": gid,
            "wx_gust_mph": gust,
            "wx_dew_point_f": dew,
            "_date": date_str,
            "_home_team": home_team,
        })

    out_df = pd.DataFrame(out_rows)
    # Coerce numeric
    for c in ["wx_gust_mph", "wx_dew_point_f"]:
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df[["game_id", "wx_gust_mph", "wx_dew_point_f"]].to_csv(out, index=False)
    return out_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "weather_noaa.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    df = build(args.season, args.week, out)
    print(f"Wrote weather_noaa rows={len(df)} to {out}")
    missing = int(df["wx_gust_mph"].isna().sum()) if "wx_gust_mph" in df.columns else -1
    missing2 = int(df["wx_dew_point_f"].isna().sum()) if "wx_dew_point_f" in df.columns else -1
    print(f"Missing wx_gust_mph: {missing} / {len(df)}")
    print(f"Missing wx_dew_point_f: {missing2} / {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
