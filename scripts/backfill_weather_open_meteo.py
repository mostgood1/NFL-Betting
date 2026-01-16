import sys
import os
from pathlib import Path
import argparse
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (ROOT / "nfl_compare" / "data")


def _c_to_f(c: float) -> float:
    return (float(c) * 9.0 / 5.0) + 32.0


def _kmh_to_mph(kmh: float) -> float:
    return float(kmh) * 0.621371


def _fetch_open_meteo_daily(lat: float, lon: float, date: str, timeout: int = 30) -> dict:
    # Open-Meteo archive API (free, no key)
    # Use daily aggregates to avoid needing kickoff times.
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": date,
        "end_date": date,
        "daily": ",".join([
            "temperature_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
        ]),
        "timezone": "UTC",
    }
    url = f"{base}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "NFL-Betting/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _ensure_weather_file(date: str, games: pd.DataFrame, stadium_meta: pd.DataFrame) -> Path:
    fp = DATA_DIR / f"weather_{date}.csv"
    if fp.exists():
        return fp

    # Create a minimal file with home teams for that date, and roof/surface if known.
    gd = games[games["game_date"].astype(str) == str(date)].copy()
    if gd.empty:
        # Still create an empty placeholder
        pd.DataFrame(columns=[
            "date","home_team","wx_temp_f","wx_wind_mph","wx_precip_pct","wx_precip_type","wx_sky","roof","surface","neutral_site"
        ]).to_csv(fp, index=False)
        return fp

    home_teams = sorted(set(gd["home_team"].astype(str)))
    base = pd.DataFrame({"date": [date] * len(home_teams), "home_team": home_teams})

    # Attach roof/surface from stadium_meta
    sm = stadium_meta.copy()
    sm["team"] = sm["team"].astype(str)
    base = base.merge(sm[["team", "roof", "surface"]].rename(columns={"team": "home_team"}), on="home_team", how="left")

    # Initialize numeric columns
    for c in ["wx_temp_f", "wx_wind_mph", "wx_precip_pct", "wx_precip_type", "wx_sky", "neutral_site"]:
        if c not in base.columns:
            base[c] = np.nan

    base.to_csv(fp, index=False)
    return fp


def backfill(season: int, weeks: list[int] | None = None, sleep_s: float = 0.35) -> None:
    games = pd.read_csv(DATA_DIR / "games.csv")
    games["season"] = pd.to_numeric(games.get("season"), errors="coerce")
    games["week"] = pd.to_numeric(games.get("week"), errors="coerce")
    games = games[games["season"].eq(int(season))].copy()

    if "game_date" not in games.columns:
        raise ValueError("games.csv missing game_date")

    if weeks is not None and len(weeks) > 0:
        games = games[games["week"].isin([int(w) for w in weeks])].copy()

    stadium_meta = pd.read_csv(DATA_DIR / "stadium_meta.csv")
    stadium_meta["team"] = stadium_meta["team"].astype(str)

    needed_dates = sorted(set(games["game_date"].astype(str)))
    if not needed_dates:
        print("No game dates found")
        return

    cache: dict[tuple[str, str], dict] = {}

    updated_files = 0
    updated_rows = 0

    for date in needed_dates:
        fp = _ensure_weather_file(date, games, stadium_meta)
        try:
            w = pd.read_csv(fp)
        except Exception:
            continue

        if w.empty:
            continue

        # Ensure expected columns
        for c in ["date", "home_team", "wx_temp_f", "wx_wind_mph", "wx_precip_pct", "wx_precip_type", "wx_sky", "roof", "surface", "neutral_site"]:
            if c not in w.columns:
                w[c] = np.nan

        # Prevent dtype warnings when assigning strings into all-NaN columns.
        for c in ["wx_precip_type", "wx_sky"]:
            if c in w.columns:
                w[c] = w[c].astype("object")

        # Fill roof/surface if missing
        if w["roof"].isna().any() or w["surface"].isna().any():
            w = w.merge(stadium_meta[["team", "roof", "surface"]].rename(columns={"team": "home_team"}), on="home_team", how="left", suffixes=("", "_m"))
            for c in ["roof", "surface"]:
                mc = f"{c}_m"
                if mc in w.columns:
                    w[c] = w[c].where(w[c].notna(), w[mc])
                    w = w.drop(columns=[mc])

        changed = False

        for idx, row in w.iterrows():
            team = str(row.get("home_team"))
            # Skip if numeric already present
            if pd.notna(row.get("wx_temp_f")) and pd.notna(row.get("wx_wind_mph")) and pd.notna(row.get("wx_precip_pct")):
                continue

            sm = stadium_meta[stadium_meta["team"] == team]
            if sm.empty:
                continue

            lat = sm.iloc[0].get("lat")
            lon = sm.iloc[0].get("lon")
            if lat is None or lon is None or (not np.isfinite(float(lat))) or (not np.isfinite(float(lon))):
                continue

            key = (date, team)
            if key not in cache:
                try:
                    cache[key] = _fetch_open_meteo_daily(float(lat), float(lon), date)
                except Exception:
                    cache[key] = {}
                time.sleep(float(sleep_s))

            js = cache.get(key) or {}
            daily = js.get("daily") or {}
            if not daily:
                continue

            # Extract single-day values (arrays)
            try:
                t_c = (daily.get("temperature_2m_mean") or [None])[0]
                wind_kmh = (daily.get("wind_speed_10m_max") or [None])[0]
                precip_mm = (daily.get("precipitation_sum") or [None])[0]
            except Exception:
                t_c = wind_kmh = precip_mm = None

            if t_c is not None and pd.isna(row.get("wx_temp_f")):
                w.at[idx, "wx_temp_f"] = _c_to_f(float(t_c))
                changed = True
            if wind_kmh is not None and pd.isna(row.get("wx_wind_mph")):
                w.at[idx, "wx_wind_mph"] = _kmh_to_mph(float(wind_kmh))
                changed = True
            if precip_mm is not None and pd.isna(row.get("wx_precip_pct")):
                # We don't have probability historically; treat measurable precipitation as 100%.
                mm = float(precip_mm)
                w.at[idx, "wx_precip_pct"] = 100.0 if mm > 0.0 else 0.0
                if pd.isna(row.get("wx_precip_type")):
                    try:
                        tf = float(w.at[idx, "wx_temp_f"]) if pd.notna(w.at[idx, "wx_temp_f"]) else np.nan
                    except Exception:
                        tf = np.nan
                    if mm > 0.0:
                        if np.isfinite(tf) and tf <= 32.0:
                            w.at[idx, "wx_precip_type"] = "snow"
                        else:
                            w.at[idx, "wx_precip_type"] = "rain"
                    else:
                        w.at[idx, "wx_precip_type"] = "none"
                changed = True

        if changed:
            # Backup and write
            backup = fp.with_suffix(".backup.csv")
            try:
                fp.replace(backup)
            except Exception:
                backup = None
            w.to_csv(fp, index=False)
            updated_files += 1
            updated_rows += int(w[["wx_temp_f", "wx_wind_mph", "wx_precip_pct"]].notna().all(axis=1).sum())

    print(f"Updated {updated_files} weather files; rows with full numerics now: {updated_rows}")


def _parse_weeks(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if str(x).strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill per-date weather CSVs with Open-Meteo archive numerics")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", type=str, default="", help="Week list like '17-18' or '1,2,3'. Empty => all dates in season")
    p.add_argument("--sleep", type=float, default=0.35, help="Sleep between API calls")
    args = p.parse_args()

    weeks = _parse_weeks(args.weeks) if args.weeks else None
    backfill(int(args.season), weeks=weeks, sleep_s=float(args.sleep))


if __name__ == "__main__":
    main()
