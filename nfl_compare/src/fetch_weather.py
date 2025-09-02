from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .weather import DATA_DIR, load_stadium_meta
from .openweather_client import get_openweather_forecast, geocode_city


def main():
    parser = argparse.ArgumentParser(description='Fetch per-game weather snapshots using OpenWeather.')
    parser.add_argument('--date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--kickoff-hour', type=int, default=16, help='Local kickoff hour (24h)')
    args = parser.parse_args()

    stad = load_stadium_meta()
    if stad.empty:
        print('No stadium_meta.csv found in data/. Skipping weather fetch.')
        return

    # Expect games.csv to map home teams and dates
    gfp = DATA_DIR / 'games.csv'
    if not gfp.exists():
        print('games.csv not found. Skipping weather fetch.')
        return
    games = pd.read_csv(gfp)
    games = games[(games['date'] == args.date)].copy()
    if games.empty:
        print(f'No games for {args.date}.')
        return

    # Prepare stadium lookup with lat/lon/tz
    stad_cols = {c.lower(): c for c in stad.columns}
    def _get(team: str, col: str):
        col_real = stad_cols.get(col)
        if col_real is None:
            return None
        row = stad[stad['team'].astype(str).str.strip() == str(team).strip()]
        if row.empty:
            return None
        return row.iloc[0].get(col_real)

    out_rows = []
    for _, g in games.iterrows():
        home = g.get('home_team')
        lat = _get(home, 'lat')
        lon = _get(home, 'lon')
        tz = _get(home, 'tz')
        roof = _get(home, 'roof')
        surface = _get(home, 'surface')
        if lat is None or lon is None:
            # Try to geocode the home team city name
            geo = geocode_city(str(home))
            if geo is None:
                continue
            lat, lon = geo.get('lat'), geo.get('lon')
        # Try primary kickoff hour; if missing, try common alt windows to improve coverage
        wx = get_openweather_forecast(float(lat), float(lon), args.date, tz_name=str(tz) if tz else None, kickoff_hour_local=args.kickoff_hour)
        if not wx or (wx.get('wx_temp_f') is None and wx.get('wx_wind_mph') is None and wx.get('wx_precip_pct') is None):
            for h in [13, 14, 15, 16, 17, 20]:  # typical NFL Sunday windows local time
                if h == args.kickoff_hour:
                    continue
                alt = get_openweather_forecast(float(lat), float(lon), args.date, tz_name=str(tz) if tz else None, kickoff_hour_local=h)
                if alt and any(alt.get(k) is not None for k in ('wx_temp_f','wx_wind_mph','wx_precip_pct')):
                    wx = alt
                    break
        row = {
            'date': args.date,
            'home_team': home,
            'wx_temp_f': wx.get('wx_temp_f') if wx else None,
            'wx_wind_mph': wx.get('wx_wind_mph') if wx else None,
            'wx_precip_pct': wx.get('wx_precip_pct') if wx else None,
            'roof': roof,
            'surface': surface,
        }
        out_rows.append(row)

    if not out_rows:
        print('No weather rows produced.')
        return

    out = pd.DataFrame(out_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"weather_{args.date}.csv"
    out.to_csv(fp, index=False)
    print(f'Wrote {len(out)} weather rows to {fp}')


if __name__ == '__main__':
    main()
