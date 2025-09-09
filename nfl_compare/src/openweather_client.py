from __future__ import annotations

import os
from .config import load_env as _load_env  # ensure .env is loaded
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List
import requests


def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def _ms_to_mph(ms: float) -> float:
    return ms * 2.23693629


def _nearest_block(forecasts: list[dict], target_dt_local: datetime, tz: ZoneInfo) -> Optional[dict]:
    best = None
    best_diff = None
    for item in forecasts:
        try:
            dt_utc = datetime.fromtimestamp(int(item.get('dt')), tz=timezone.utc)
            dt_local = dt_utc.astimezone(tz)
            # Only consider same calendar date as target
            if dt_local.date() != target_dt_local.date():
                continue
            diff = abs((dt_local - target_dt_local).total_seconds())
            if best_diff is None or diff < best_diff:
                best = item
                best_diff = diff
        except Exception:
            continue
    return best


def get_openweather_forecast(lat: float, lon: float, date_str: str, *, tz_name: Optional[str] = None,
                              api_key: Optional[str] = None,
                              kickoff_hour_local: int = 16) -> Optional[Dict[str, Any]]:
    """Fetch a per-game weather snapshot using OpenWeather 5-day / 3h forecast and derive
    temp, wind, precip prob, precip type, and sky cover.
    """
    _load_env()
    api_key = api_key or os.getenv('OPENWEATHER_API_KEY') or os.getenv('OPENWEATHER_KEY')
    if not api_key:
        return None
    try:
        resp = requests.get(
            'https://api.openweathermap.org/data/2.5/forecast',
            params={'lat': lat, 'lon': lon, 'appid': api_key}, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        lst = data.get('list', []) or []
        if not lst:
            return None
        tz = ZoneInfo(tz_name) if tz_name else timezone.utc
        target_dt_local = datetime.fromisoformat(f"{date_str}T{kickoff_hour_local:02d}:00:00")
        if target_dt_local.tzinfo is None:
            target_dt_local = target_dt_local.replace(tzinfo=tz)
        chosen = _nearest_block(lst, target_dt_local, tz)
        if not chosen:
            midday = datetime.fromisoformat(f"{date_str}T12:00:00").replace(tzinfo=tz)
            chosen = _nearest_block(lst, midday, tz)
        if not chosen:
            return None
        main = chosen.get('main', {})
        wind = chosen.get('wind', {})
        pop = chosen.get('pop', 0.0)
        temp_f = _kelvin_to_f(float(main.get('temp'))) if main.get('temp') is not None else None
        wind_mph = _ms_to_mph(float(wind.get('speed'))) if wind.get('speed') is not None else None
        precip_pct = float(pop) * 100.0 if pop is not None else None
        weather_arr = chosen.get('weather') or []
        main_weather = (weather_arr[0].get('main') if weather_arr else '') or ''
        main_l = main_weather.lower()
        try:
            clouds_pct = float((chosen.get('clouds') or {}).get('all'))
        except Exception:
            clouds_pct = None
        precip_type = 'none'
        if main_l in {'rain','drizzle'}:
            precip_type = 'rain'
        elif main_l in {'thunderstorm','storm'}:
            precip_type = 'storm'
        elif main_l == 'snow':
            precip_type = 'snow'
        if precip_type != 'none' and (precip_pct is None or precip_pct < 15):
            precip_type = 'none'
        sky = None
        if clouds_pct is not None:
            if clouds_pct <= 10:
                sky = 'Clear'
            elif clouds_pct <= 30:
                sky = 'Mostly Clear'
            elif clouds_pct <= 60:
                sky = 'Partly Cloudy'
            elif clouds_pct <= 85:
                sky = 'Mostly Cloudy'
            else:
                sky = 'Overcast'
        else:
            if main_l in {'clear'}:
                sky = 'Clear'
            elif main_l in {'clouds','cloudy'}:
                sky = 'Cloudy'
        return {
            'wx_temp_f': round(temp_f, 1) if isinstance(temp_f, float) else None,
            'wx_wind_mph': round(wind_mph, 1) if isinstance(wind_mph, float) else None,
            'wx_precip_pct': round(precip_pct, 0) if isinstance(precip_pct, float) else None,
            'wx_precip_type': precip_type,
            'wx_sky': sky,
        }
    except Exception:
        return None


def geocode_city(query: str, *, api_key: Optional[str] = None, limit: int = 1) -> Optional[Dict[str, Any]]:
    """
    Use OpenWeather Geocoding API to resolve a city/team name to lat/lon.
    Returns a dict with 'lat','lon','name','country','state' or None.
    """
    _load_env()
    api_key = api_key or os.getenv('OPENWEATHER_API_KEY') or os.getenv('OPENWEATHER_KEY')
    if not api_key or not query:
        return None
    try:
        resp = requests.get(
            'https://api.openweathermap.org/geo/1.0/direct',
            params={'q': query, 'limit': limit, 'appid': api_key}, timeout=15
        )
        resp.raise_for_status()
        arr: List[Dict[str, Any]] = resp.json() or []
        if not arr:
            return None
        top = arr[0]
        return {
            'lat': top.get('lat'),
            'lon': top.get('lon'),
            'name': top.get('name'),
            'country': top.get('country'),
            'state': top.get('state'),
        }
    except Exception:
        return None
