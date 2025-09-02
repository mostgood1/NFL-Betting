"""
NFL team name normalizer.

Ensures consistent naming across provider feeds and local CSVs.
"""
from __future__ import annotations

from typing import Dict

_MAP: Dict[str, str] = {
    # NFC East
    "dallas": "Dallas Cowboys",
    "cowboys": "Dallas Cowboys",
    "ny giants": "New York Giants",
    "new york giants": "New York Giants",
    "giants": "New York Giants",
    "philadelphia eagles": "Philadelphia Eagles",
    "eagles": "Philadelphia Eagles",
    "washington commanders": "Washington Commanders",
    "washington": "Washington Commanders",
    "commanders": "Washington Commanders",
    # NFC North
    "green bay packers": "Green Bay Packers",
    "packers": "Green Bay Packers",
    "chicago bears": "Chicago Bears",
    "bears": "Chicago Bears",
    "minnesota vikings": "Minnesota Vikings",
    "vikings": "Minnesota Vikings",
    "detroit lions": "Detroit Lions",
    "lions": "Detroit Lions",
    # NFC South
    "tampa bay buccaneers": "Tampa Bay Buccaneers",
    "buccaneers": "Tampa Bay Buccaneers",
    "carolina panthers": "Carolina Panthers",
    "panthers": "Carolina Panthers",
    "new orleans saints": "New Orleans Saints",
    "saints": "New Orleans Saints",
    "atlanta falcons": "Atlanta Falcons",
    "falcons": "Atlanta Falcons",
    # NFC West
    "san francisco 49ers": "San Francisco 49ers",
    "49ers": "San Francisco 49ers",
    "sf": "San Francisco 49ers",
    "seattle seahawks": "Seattle Seahawks",
    "seahawks": "Seattle Seahawks",
    "los angeles rams": "Los Angeles Rams",
    "rams": "Los Angeles Rams",
    "lar": "Los Angeles Rams",
    "la": "Los Angeles Rams",
    "arizona cardinals": "Arizona Cardinals",
    "cardinals": "Arizona Cardinals",
    # AFC East
    "new england patriots": "New England Patriots",
    "patriots": "New England Patriots",
    "ne": "New England Patriots",
    "buffalo bills": "Buffalo Bills",
    "bills": "Buffalo Bills",
    "buf": "Buffalo Bills",
    "miami dolphins": "Miami Dolphins",
    "dolphins": "Miami Dolphins",
    "mia": "Miami Dolphins",
    "new york jets": "New York Jets",
    "ny jets": "New York Jets",
    "jets": "New York Jets",
    "nyj": "New York Jets",
    # AFC North
    "baltimore ravens": "Baltimore Ravens",
    "ravens": "Baltimore Ravens",
    "bal": "Baltimore Ravens",
    "pittsburgh steelers": "Pittsburgh Steelers",
    "steelers": "Pittsburgh Steelers",
    "pit": "Pittsburgh Steelers",
    "cleveland browns": "Cleveland Browns",
    "browns": "Cleveland Browns",
    "cle": "Cleveland Browns",
    "cincinnati bengals": "Cincinnati Bengals",
    "bengals": "Cincinnati Bengals",
    "cin": "Cincinnati Bengals",
    # AFC South
    "tennessee titans": "Tennessee Titans",
    "titans": "Tennessee Titans",
    "ten": "Tennessee Titans",
    "indianapolis colts": "Indianapolis Colts",
    "colts": "Indianapolis Colts",
    "ind": "Indianapolis Colts",
    "jacksonville jaguars": "Jacksonville Jaguars",
    "jaguars": "Jacksonville Jaguars",
    "jax": "Jacksonville Jaguars",
    "houston texans": "Houston Texans",
    "texans": "Houston Texans",
    "hou": "Houston Texans",
    # AFC West
    "kansas city chiefs": "Kansas City Chiefs",
    "chiefs": "Kansas City Chiefs",
    "kc": "Kansas City Chiefs",
    "los angeles chargers": "Los Angeles Chargers",
    "la chargers": "Los Angeles Chargers",
    "chargers": "Los Angeles Chargers",
    "lac": "Los Angeles Chargers",
    "las vegas raiders": "Las Vegas Raiders",
    "raiders": "Las Vegas Raiders",
    "lv": "Las Vegas Raiders",
    "denver broncos": "Denver Broncos",
    "broncos": "Denver Broncos",
    "den": "Denver Broncos",
    # Other abbreviations
    "dal": "Dallas Cowboys",
    "nyg": "New York Giants",
    "phi": "Philadelphia Eagles",
    "was": "Washington Commanders",
    "gb": "Green Bay Packers",
    "chi": "Chicago Bears",
    "min": "Minnesota Vikings",
    "det": "Detroit Lions",
    "tb": "Tampa Bay Buccaneers",
    "car": "Carolina Panthers",
    "no": "New Orleans Saints",
    "atl": "Atlanta Falcons",
    "sea": "Seattle Seahawks",
    "ari": "Arizona Cardinals",
}


def normalize_team_name(name: str) -> str:
    if not name:
        return ""
    key = name.replace("_", " ").strip().lower()
    return _MAP.get(key, name.strip())
