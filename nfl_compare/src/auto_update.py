from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from .weather import DATA_DIR, load_stadium_meta
from .openweather_client import get_openweather_forecast, geocode_city
from .predict import main as predict_main


def _load_location_overrides():
    """Best-effort loader for per-game location overrides.
    Returns dict with 'by_game_id' and 'by_match' maps.
    """
    ovr_path = DATA_DIR / 'game_location_overrides.csv'
    out = {"by_game_id": {}, "by_match": {}}
    if not ovr_path.exists():
        return out
    try:
        expected = ['game_id','date','home_team','away_team','venue','city','country','tz','lat','lon','roof','surface','neutral_site','note']
        df = pd.read_csv(ovr_path, comment='#', header=None, names=expected)
        def norm(v):
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            return str(v)
        for _, r in df.iterrows():
            rec = {k: r.get(k) for k in [
                'venue','city','country','tz','lat','lon','roof','surface','neutral_site','note'
            ] if k in df.columns}
            gid = norm(r.get('game_id')) if 'game_id' in df.columns else None
            date_v = norm(r.get('date')) if 'date' in df.columns else None
            home = norm(r.get('home_team')) if 'home_team' in df.columns else None
            away = norm(r.get('away_team')) if 'away_team' in df.columns else None
            if gid:
                out['by_game_id'][gid] = rec
            if date_v and home and away:
                out['by_match'][(date_v, home, away)] = rec
    except Exception:
        return out
    return out


def _future_game_dates(games: pd.DataFrame) -> list[str]:
    if games is None or games.empty:
        return []
    today = date.today()
    # future games: no final scores or date >= today
    g = games.copy()
    g['date'] = pd.to_datetime(g['date'], errors='coerce')
    g_future = g[(g['home_score'].isna()) | (g['away_score'].isna()) | (g['date'].dt.date >= today)]
    return sorted({d.date().isoformat() for d in g_future['date'].dropna()})


def update_weather_for_date(date_str: str, kickoff_hour_local: int = 16) -> int:
    stad = load_stadium_meta()
    if stad.empty:
        print('No stadium_meta.csv found; cannot fetch weather.')
        return 0

    gfp = DATA_DIR / 'games.csv'
    if not gfp.exists():
        print('games.csv not found; cannot fetch weather.')
        return 0
    games = pd.read_csv(gfp)
    games = games[(games['date'] == date_str)].copy()
    if games.empty:
        print(f'No games for {date_str}.')
        return 0

    stad_cols = {c.lower(): c for c in stad.columns}
    overrides = _load_location_overrides()

    def _get(team: str, col: str):
        col_real = stad_cols.get(col)
        if col_real is None:
            return None
        row = stad[stad['team'].astype(str).str.strip() == str(team).strip()]
        if row.empty:
            return None
        return row.iloc[0].get(col_real)

    out_rows: list[dict] = []
    for _, g in games.iterrows():
        home = g.get('home_team')
        away = g.get('away_team')
        game_id = g.get('game_id')

        # Base from stadium meta
        lat = _get(home, 'lat')
        lon = _get(home, 'lon')
        tz = _get(home, 'tz')
        roof = _get(home, 'roof')
        surface = _get(home, 'surface')

        # Apply overrides if present
        ovr = None
        gid_s = str(game_id) if pd.notna(game_id) else None
        if gid_s and overrides['by_game_id']:
            ovr = overrides['by_game_id'].get(gid_s)
        if ovr is None and overrides['by_match']:
            try:
                date_key = pd.to_datetime(date_str, errors='coerce').date().isoformat()
            except Exception:
                date_key = str(date_str)
            ovr = overrides['by_match'].get((date_key, str(home), str(away)))
        if ovr:
            tz = ovr.get('tz') or tz
            roof = ovr.get('roof') or roof
            surface = ovr.get('surface') or surface
            ov_lat = ovr.get('lat')
            ov_lon = ovr.get('lon')
            try:
                lat = float(ov_lat) if ov_lat is not None and not (isinstance(ov_lat, float) and pd.isna(ov_lat)) else lat
                lon = float(ov_lon) if ov_lon is not None and not (isinstance(ov_lon, float) and pd.isna(ov_lon)) else lon
            except Exception:
                pass

        # If still missing coordinates, geocode using the most specific location available
        if lat is None or lon is None:
            query = None
            if ovr and (ovr.get('city') or ovr.get('venue')):
                city = ovr.get('city')
                venue = ovr.get('venue')
                country = ovr.get('country')
                query = ', '.join([v for v in [venue, city, country] if v])
            if not query:
                query = str(home)
            geo = geocode_city(query)
            if geo is None:
                continue
            lat, lon = geo.get('lat'), geo.get('lon')

        wx = get_openweather_forecast(float(lat), float(lon), date_str, tz_name=str(tz) if tz else None, kickoff_hour_local=kickoff_hour_local)
        # Build output row; include neutral flag if override specified
        row = {
            'date': date_str,
            'home_team': home,
            'wx_temp_f': None if not wx else wx.get('wx_temp_f'),
            'wx_wind_mph': None if not wx else wx.get('wx_wind_mph'),
            'wx_precip_pct': None if not wx else wx.get('wx_precip_pct'),
            'wx_precip_type': None if not wx else wx.get('wx_precip_type'),
            'wx_sky': None if not wx else wx.get('wx_sky'),
            'roof': roof,
            'surface': surface,
            'neutral_site': (ovr.get('neutral_site') if ovr and ('neutral_site' in ovr) else None),
        }
        out_rows.append(row)

    if not out_rows:
        print(f'No weather rows for {date_str}.')
        return 0

    out = pd.DataFrame(out_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"weather_{date_str}.csv"
    out.to_csv(fp, index=False)
    print(f'Wrote {len(out)} weather rows to {fp}')
    return len(out)


def main():
    gfp = DATA_DIR / 'games.csv'
    if not gfp.exists():
        print('games.csv not found; aborting.')
        return
    games = pd.read_csv(gfp)
    dates = _future_game_dates(games)
    if not dates:
        print('No future game dates found.')
        return
    total = 0
    for d in dates:
        total += update_weather_for_date(d)
    print(f'Weather update complete: {total} rows across {len(dates)} date(s).')

    # Re-run predictions to incorporate weather
    predict_main()


if __name__ == '__main__':
    main()
