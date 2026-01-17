from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

# Respect NFL_DATA_DIR env when present; fallback to package data folder
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (Path(__file__).resolve().parents[1] / "data")


@dataclass
class WeatherCols:
    temp_f: str = "wx_temp_f"
    wind_mph: str = "wx_wind_mph"
    precip_pct: str = "wx_precip_pct"
    precip_type: str = "wx_precip_type"  # e.g., rain, snow, none
    sky: str = "wx_sky"  # e.g., sunny, partly cloudy, overcast
    roof: str = "roof"
    surface: str = "surface"


def _ensure_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.precip_type, WeatherCols.sky, WeatherCols.roof, WeatherCols.surface]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_stadium_meta() -> pd.DataFrame:
    """Load optional stadium metadata (team, roof, surface, lat, lon, tz, altitude).
    Expected file: data/stadium_meta.csv with at least columns ['team','roof','surface'].
    Returns empty DataFrame if not present.
    """
    fp = DATA_DIR / "stadium_meta.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["team","roof","surface","lat","lon","tz","altitude_ft"])
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame(columns=["team","roof","surface","lat","lon","tz","altitude_ft"])
    return df


def load_weather_for_date(date_str: str) -> pd.DataFrame:
    """Load optional per-date weather forecast file: data/weather_<YYYY-MM-DD>.csv.
    Expected columns: ['date','home_team','wx_temp_f','wx_wind_mph','wx_precip_pct'].
    Returns empty DataFrame if not present.
    """
    candidates = [
        DATA_DIR / f"weather_{date_str}.csv",
        DATA_DIR / f"weather_{date_str.replace('-', '_')}.csv",
        DATA_DIR / "weather.csv",
    ]
    for fp in candidates:
        if fp.exists():
            try:
                return pd.read_csv(fp)
            except Exception:
                continue
    return pd.DataFrame(columns=["date","home_team", WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.precip_type, WeatherCols.sky])


def load_weather_for_games(games: pd.DataFrame) -> pd.DataFrame:
    """Join weather and stadium meta to the games DataFrame, returning per-game weather rows keyed by game_id.
    Non-destructive: if files are missing, returns a frame with expected columns and NaNs.
    """
    if games is None or games.empty:
        return pd.DataFrame(columns=["game_id","date","home_team","away_team", WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.precip_type, WeatherCols.sky, WeatherCols.roof, WeatherCols.surface])

    # games can be keyed by either `date` or `game_date` depending on pipeline.
    date_col = "date" if "date" in games.columns else ("game_date" if "game_date" in games.columns else None)

    out_rows = []
    stad = load_stadium_meta()
    # Build a quick map for stadium attributes by team
    stad_map = {}
    if not stad.empty and "team" in stad.columns:
        try:
            from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
        except Exception:
            _norm_team = lambda s: str(s)
        stad_map = stad.assign(_k=stad["team"].astype(str).map(lambda x: _norm_team(str(x)).strip())).set_index("_k").to_dict(orient="index")

    # group by date for weather file loads (pandas groupby drops NaN keys)
    if date_col is None:
        # No usable date column; emit rows with stadium meta only.
        date_s = pd.Series([pd.NA] * len(games), index=games.index)
    else:
        date_s = games[date_col]
    # Normalize date string to YYYY-MM-DD (weather files use that convention)
    date_key = pd.to_datetime(date_s, errors="coerce").dt.strftime("%Y-%m-%d")

    for date_str, gdf in games.loc[date_key.notna()].groupby(date_key[date_key.notna()]):
        wdf = load_weather_for_date(str(date_str))
        # Normalize join keys
        try:
            from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
        except Exception:
            _norm_team = lambda s: str(s)
        if not wdf.empty and "home_team" in wdf.columns:
            wdf["home_team"] = wdf["home_team"].astype(str).map(lambda x: _norm_team(str(x)).strip())
        for _, row in gdf.iterrows():
            ht = _norm_team(str(row.get("home_team", ""))).strip()
            at = _norm_team(str(row.get("away_team", ""))).strip()
            r = {
                "game_id": row.get("game_id"),
                "date": row.get(date_col) if date_col else row.get("date"),
                "home_team": ht,
                "away_team": at,
                WeatherCols.temp_f: pd.NA,
                WeatherCols.wind_mph: pd.NA,
                WeatherCols.precip_pct: pd.NA,
                WeatherCols.precip_type: pd.NA,
                WeatherCols.sky: pd.NA,
                WeatherCols.roof: pd.NA,
                WeatherCols.surface: pd.NA,
                "neutral_site": pd.NA,
            }
            if not wdf.empty:
                m = wdf[wdf["home_team"] == ht]
                if not m.empty:
                    r[WeatherCols.temp_f] = m.iloc[0].get(WeatherCols.temp_f)
                    r[WeatherCols.wind_mph] = m.iloc[0].get(WeatherCols.wind_mph)
                    r[WeatherCols.precip_pct] = m.iloc[0].get(WeatherCols.precip_pct)
                    if WeatherCols.precip_type in m.columns:
                        r[WeatherCols.precip_type] = m.iloc[0].get(WeatherCols.precip_type)
                    if WeatherCols.sky in m.columns:
                        r[WeatherCols.sky] = m.iloc[0].get(WeatherCols.sky)
                    # Neutral flag if produced by fetcher
                    if 'neutral_site' in m.columns:
                        r["neutral_site"] = m.iloc[0].get('neutral_site')
            if ht in stad_map:
                r[WeatherCols.roof] = stad_map[ht].get("roof")
                r[WeatherCols.surface] = stad_map[ht].get("surface")
            out_rows.append(r)

    # Also include any games where date_key was NaN (stadium meta only).
    missing_date_games = games.loc[date_key.isna()]
    if not missing_date_games.empty:
        try:
            from .team_normalizer import normalize_team_name as _norm_team  # type: ignore
        except Exception:
            _norm_team = lambda s: str(s)
        for _, row in missing_date_games.iterrows():
            ht = _norm_team(str(row.get("home_team", ""))).strip()
            at = _norm_team(str(row.get("away_team", ""))).strip()
            r = {
                "game_id": row.get("game_id"),
                "date": row.get(date_col) if date_col else row.get("date"),
                "home_team": ht,
                "away_team": at,
                WeatherCols.temp_f: pd.NA,
                WeatherCols.wind_mph: pd.NA,
                WeatherCols.precip_pct: pd.NA,
                WeatherCols.precip_type: pd.NA,
                WeatherCols.sky: pd.NA,
                WeatherCols.roof: pd.NA,
                WeatherCols.surface: pd.NA,
                "neutral_site": pd.NA,
            }
            if ht in stad_map:
                r[WeatherCols.roof] = stad_map[ht].get("roof")
                r[WeatherCols.surface] = stad_map[ht].get("surface")
            out_rows.append(r)

    out = pd.DataFrame(out_rows)
    return _ensure_weather_columns(out)
