from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .weather import DATA_DIR
from .team_normalizer import normalize_team_name, _MAP
from .openweather_client import geocode_city


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_current_National_Football_League_stadiums"


# Simple team -> timezone mapping (IANA names)
TEAM_TZ: Dict[str, str] = {
    # Eastern
    'Buffalo Bills': 'America/New_York',
    'Miami Dolphins': 'America/New_York',
    'New England Patriots': 'America/New_York',
    'New York Jets': 'America/New_York',
    'Baltimore Ravens': 'America/New_York',
    'Cincinnati Bengals': 'America/New_York',
    'Cleveland Browns': 'America/New_York',
    'Pittsburgh Steelers': 'America/New_York',
    'Indianapolis Colts': 'America/Indiana/Indianapolis',
    'Jacksonville Jaguars': 'America/New_York',
    'Tennessee Titans': 'America/Chicago',
    'Houston Texans': 'America/Chicago',
    'Kansas City Chiefs': 'America/Chicago',
    'Denver Broncos': 'America/Denver',
    'Las Vegas Raiders': 'America/Los_Angeles',
    'Los Angeles Chargers': 'America/Los_Angeles',
    'Los Angeles Rams': 'America/Los_Angeles',
    'Dallas Cowboys': 'America/Chicago',
    'New York Giants': 'America/New_York',
    'Philadelphia Eagles': 'America/New_York',
    'Washington Commanders': 'America/New_York',
    'Chicago Bears': 'America/Chicago',
    'Detroit Lions': 'America/Detroit',
    'Green Bay Packers': 'America/Chicago',
    'Minnesota Vikings': 'America/Chicago',
    'Atlanta Falcons': 'America/New_York',
    'Carolina Panthers': 'America/New_York',
    'New Orleans Saints': 'America/Chicago',
    'Tampa Bay Buccaneers': 'America/New_York',
    'Arizona Cardinals': 'America/Phoenix',
    'San Francisco 49ers': 'America/Los_Angeles',
    'Seattle Seahawks': 'America/Los_Angeles',
}

# Static fallback coordinates and attributes (approximate) if scraping/geocoding fails
STATIC_META: Dict[str, Dict[str, object]] = {
    'Arizona Cardinals': {'lat': 33.5277, 'lon': -112.2626, 'roof': 'retractable', 'surface': 'grass', 'altitude_ft': None},
    'Atlanta Falcons': {'lat': 33.7554, 'lon': -84.4010, 'roof': 'retractable', 'surface': 'artificial', 'altitude_ft': None},
    'Baltimore Ravens': {'lat': 39.2780, 'lon': -76.6227, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Buffalo Bills': {'lat': 42.7738, 'lon': -78.7869, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Carolina Panthers': {'lat': 35.2258, 'lon': -80.8528, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Chicago Bears': {'lat': 41.8623, 'lon': -87.6167, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Cincinnati Bengals': {'lat': 39.0954, 'lon': -84.5160, 'roof': 'open', 'surface': 'artificial', 'altitude_ft': None},
    'Cleveland Browns': {'lat': 41.5061, 'lon': -81.6995, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Dallas Cowboys': {'lat': 32.7473, 'lon': -97.0945, 'roof': 'retractable', 'surface': 'artificial', 'altitude_ft': None},
    'Denver Broncos': {'lat': 39.7439, 'lon': -105.0201, 'roof': 'open', 'surface': 'grass', 'altitude_ft': 5280},
    'Detroit Lions': {'lat': 42.3400, 'lon': -83.0456, 'roof': 'fixed', 'surface': 'artificial', 'altitude_ft': None},
    'Green Bay Packers': {'lat': 44.5013, 'lon': -88.0622, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Houston Texans': {'lat': 29.6847, 'lon': -95.4107, 'roof': 'retractable', 'surface': 'artificial', 'altitude_ft': None},
    'Indianapolis Colts': {'lat': 39.7601, 'lon': -86.1639, 'roof': 'retractable', 'surface': 'artificial', 'altitude_ft': None},
    'Jacksonville Jaguars': {'lat': 30.3239, 'lon': -81.6374, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Kansas City Chiefs': {'lat': 39.0490, 'lon': -94.4839, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Las Vegas Raiders': {'lat': 36.0908, 'lon': -115.1830, 'roof': 'fixed', 'surface': 'grass', 'altitude_ft': None},
    'Los Angeles Chargers': {'lat': 33.9535, 'lon': -118.3391, 'roof': 'fixed', 'surface': 'artificial', 'altitude_ft': None},
    'Los Angeles Rams': {'lat': 33.9535, 'lon': -118.3391, 'roof': 'fixed', 'surface': 'artificial', 'altitude_ft': None},
    'Miami Dolphins': {'lat': 25.9580, 'lon': -80.2389, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Minnesota Vikings': {'lat': 44.9735, 'lon': -93.2575, 'roof': 'fixed', 'surface': 'artificial', 'altitude_ft': None},
    'New England Patriots': {'lat': 42.0909, 'lon': -71.2643, 'roof': 'open', 'surface': 'artificial', 'altitude_ft': None},
    'New Orleans Saints': {'lat': 29.9509, 'lon': -90.0815, 'roof': 'fixed', 'surface': 'artificial', 'altitude_ft': None},
    'New York Giants': {'lat': 40.8135, 'lon': -74.0745, 'roof': 'open', 'surface': 'artificial', 'altitude_ft': None},
    'New York Jets': {'lat': 40.8135, 'lon': -74.0745, 'roof': 'open', 'surface': 'artificial', 'altitude_ft': None},
    'Philadelphia Eagles': {'lat': 39.9008, 'lon': -75.1675, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Pittsburgh Steelers': {'lat': 40.4468, 'lon': -80.0158, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'San Francisco 49ers': {'lat': 37.4030, 'lon': -121.9690, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Seattle Seahawks': {'lat': 47.5952, 'lon': -122.3316, 'roof': 'open', 'surface': 'artificial', 'altitude_ft': None},
    'Tampa Bay Buccaneers': {'lat': 27.9759, 'lon': -82.5033, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Tennessee Titans': {'lat': 36.1665, 'lon': -86.7713, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
    'Washington Commanders': {'lat': 38.9077, 'lon': -76.8645, 'roof': 'open', 'surface': 'grass', 'altitude_ft': None},
}


def _extract_table() -> pd.DataFrame:
    # Fetch the page and parse the first stadiums table that includes Teams / Surface / Roof
    html = requests.get(WIKI_URL, timeout=30).text
    tables = pd.read_html(html, flavor='lxml')
    # Find table with headers including 'Team' or 'Teams'
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any('team' in c for c in cols) and any('surface' in c for c in cols):
            return t
    raise RuntimeError('Could not find stadiums table on Wikipedia page')


def _clean_team(cell: str) -> List[str]:
    # Split multiple teams if listed; remove footnotes
    s = re.sub(r"\[.*?\]", "", str(cell))
    parts = re.split(r"/|,| and | & |\n", s)
    teams = []
    for p in parts:
        name = normalize_team_name(p.strip())
        if name:
            teams.append(name)
    return list(dict.fromkeys(teams))  # unique preserve order


def _clean_text(cell: str) -> str:
    return re.sub(r"\[.*?\]", "", str(cell)).strip()


def build_stadium_meta() -> pd.DataFrame:
    try:
        df = _extract_table()
        # Normalize columns
        # Build a lookup of lower-cased header -> original header and allow substring matching
        cols_map = {str(c).strip().lower(): c for c in df.columns}
        def get_col(name_opts: List[str]) -> Optional[str]:
            """Return the first column whose lower-cased header contains any of the
            provided name option substrings (e.g., 'location' matches 'location',
            'stadium location', etc.)."""
            keys = list(cols_map.keys())
            for n in name_opts:
                n_l = n.strip().lower()
                for k in keys:
                    if n_l in k:
                        return cols_map[k]
            return None

        c_stadium = get_col(['stadium', 'name'])
        c_team = get_col(['team', 'teams'])
        c_location = get_col(['location', 'city', 'municipality'])
        c_surface = get_col(['surface', 'turf'])
        c_roof = get_col(['roof', 'type'])

        rows: List[Dict[str, str]] = []
        for _, r in df.iterrows():
            teams = _clean_team(r.get(c_team)) if c_team else []
            loc = _clean_text(r.get(c_location)) if c_location else ''
            surf = _clean_text(r.get(c_surface)) if c_surface else ''
            roof = _clean_text(r.get(c_roof)) if c_roof else ''
            for team in teams:
                rows.append({
                    'team': team,
                    'stadium': _clean_text(r.get(c_stadium)) if c_stadium else '',
                    'location': loc,
                    'surface': surf,
                    'roof': roof,
                })

        meta = pd.DataFrame(rows).drop_duplicates(subset=['team']).reset_index(drop=True)
        # Geocode location for lat/lon
        lats, lons = [], []
        for _, rr in meta.iterrows():
            query = rr['location'] or rr['stadium']
            geo = geocode_city(query)
            lats.append(geo.get('lat') if geo else None)
            lons.append(geo.get('lon') if geo else None)
        meta['lat'] = lats
        meta['lon'] = lons
    except Exception:
        # Fallback: build minimal meta from canonical team list, geocoding by team name
        teams = sorted(set(_MAP.values()))
        meta = pd.DataFrame({'team': teams})
        meta['roof'] = ''
        meta['surface'] = ''
        lats, lons = [], []
        for t in teams:
            geo = geocode_city(t)
            lats.append(geo.get('lat') if geo else None)
            lons.append(geo.get('lon') if geo else None)
        meta['lat'] = lats
        meta['lon'] = lons

    # Apply static fallback for any missing pieces
    def _fill_from_static(row: pd.Series) -> pd.Series:
        team = str(row.get('team'))
        s = STATIC_META.get(team)
        if not s:
            return row
        if pd.isna(row.get('lat')) or row.get('lat') is None:
            row['lat'] = s.get('lat')
        if pd.isna(row.get('lon')) or row.get('lon') is None:
            row['lon'] = s.get('lon')
        if (not row.get('roof')) and s.get('roof'):
            row['roof'] = s.get('roof')
        if (not row.get('surface')) and s.get('surface'):
            row['surface'] = s.get('surface')
        if pd.isna(row.get('altitude_ft')) or row.get('altitude_ft') is None:
            row['altitude_ft'] = s.get('altitude_ft')
        return row

    meta = meta.apply(_fill_from_static, axis=1)

    # Timezone
    meta['tz'] = meta['team'].map(lambda t: TEAM_TZ.get(t, 'America/New_York'))
    # Altitude placeholder
    meta['altitude_ft'] = pd.NA
    # Reorder
    meta = meta[['team', 'roof', 'surface', 'lat', 'lon', 'tz', 'altitude_ft']]
    return meta


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = build_stadium_meta()
    fp = DATA_DIR / 'stadium_meta.csv'
    meta.to_csv(fp, index=False)
    print(f'Wrote stadium metadata for {len(meta)} teams to {fp}')


if __name__ == '__main__':
    main()
