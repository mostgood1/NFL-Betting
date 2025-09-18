from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

# Data directory within nfl_compare
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _map_pos(p: object) -> str:
    p = str(p or "").upper()
    if p in {"HB", "FB"}:
        return "RB"
    if p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("RB"):
        return "RB"
    if p.startswith("QB"):
        return "QB"
    return p


# ESPN team abbreviations used in depth chart URLs
ESPN_TEAM_ABBR = {
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
}


def _extract_status_and_clean(txt: str) -> tuple[str, str]:
    """Return (clean_name, status_token) from a name cell such as 'T.J. Hockenson Q'.
    Status tokens recognized: Q (Questionable), D (Doubtful), O (Out), IR, PUP, NFI, SUSP, DNP.
    """
    raw = (txt or "").strip()
    # Capture trailing status token(s)
    m = re.search(r"\s+(Q|D|O|IR|PUP|NFI|SUSP|DNP)$", raw, flags=re.IGNORECASE)
    status = (m.group(1).upper() if m else "")
    # Clean markers and asterisks from the name portion
    name = raw
    if m:
        name = raw[: m.start()].strip()
    name = name.replace("*", "").strip()
    return name, status


def _status_to_active(status: str) -> bool:
    s = (status or "").upper()
    # Treat clear out designations as inactive; Q/D remain active for projections
    if s in {"O", "IR", "PUP", "NFI", "SUSP", "DNP"}:
        return False
    return True


def _scrape_espn_depth_for_team(team_name: str) -> dict[str, list[str]]:
    """Return pos -> ordered list scraped from ESPN depth chart page for a team."""
    import requests
    from bs4 import BeautifulSoup

    abbr = ESPN_TEAM_ABBR.get(team_name)
    if not abbr:
        return {}
    url = f"https://www.espn.com/nfl/team/depth/_/name/{abbr}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    # Parse formation tokens if present (e.g., "3WR 1TE 1RB")
    formation_text = ""
    try:
        txt_nodes = soup.find_all(string=True)
        cand = [t.strip() for t in txt_nodes if t and "WR" in t and ("TE" in t or "RB" in t)]
        cand = sorted(cand, key=lambda s: len(s))
        if cand:
            formation_text = cand[0]
    except Exception:
        formation_text = ""
    wr_count = 3; te_count = 1; rb_count = 1
    if formation_text:
        wr_m = re.search(r"(\d+)\s*WR", formation_text, flags=re.IGNORECASE)
        te_m = re.search(r"(\d+)\s*TE", formation_text, flags=re.IGNORECASE)
        rb_m = re.search(r"(\d+)\s*RB", formation_text, flags=re.IGNORECASE)
        if wr_m:
            wr_count = max(1, int(wr_m.group(1)))
        if te_m:
            te_count = max(1, int(te_m.group(1)))
        if rb_m:
            rb_count = max(1, int(rb_m.group(1)))

    # Extract the offense tiers table via pandas
    try:
        # Wrap literal HTML string in StringIO to avoid pandas deprecation
        tables = pd.read_html(StringIO(resp.text))
    except Exception:
        return {}
    df = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(c in {"Starter", "2nd"} for c in cols):
            df = t.copy(); break
    if df is None:
        return {}
    if df.shape[1] >= 4:
        df = df.iloc[:, :4].copy()
        df.columns = ["Starter", "2nd", "3rd", "4th"]
    else:
        return {}
    for c in ["Starter", "2nd", "3rd", "4th"]:
        df[c] = df[c].astype(str).str.strip().replace({"-": ""})
    skill_rows = min(1 + rb_count + wr_count + te_count, len(df))
    df_skill = df.iloc[:skill_rows].reset_index(drop=True)

    pos_lists: dict[str, list[dict]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    # QB row
    if skill_rows >= 1:
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            raw = df_skill.loc[0, tier]
            nm, st = _extract_status_and_clean(raw)
            if nm and all(nm != x.get('player') for x in pos_lists["QB"]):
                pos_lists["QB"].append({"player": nm, "status": st, "active": _status_to_active(st)})
    # RB rows
    idx = 1
    for _ in range(rb_count):
        if idx >= skill_rows: break
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            raw = df_skill.loc[idx, tier]
            nm, st = _extract_status_and_clean(raw)
            if nm and all(nm != x.get('player') for x in pos_lists["RB"]):
                pos_lists["RB"].append({"player": nm, "status": st, "active": _status_to_active(st)})
        idx += 1
    # WR rows aggregated by tier
    wr_rows = []
    for _ in range(wr_count):
        if idx >= skill_rows: break
        wr_rows.append(df_skill.loc[idx, ["Starter", "2nd", "3rd", "4th"]].tolist())
        idx += 1
    if wr_rows:
        for tier_idx in range(4):
            for r in wr_rows:
                raw = (r[tier_idx] or "").strip()
                nm, st = _extract_status_and_clean(raw)
                if nm and all(nm != x.get('player') for x in pos_lists["WR"]):
                    pos_lists["WR"].append({"player": nm, "status": st, "active": _status_to_active(st)})
    # TE rows
    for _ in range(te_count):
        if idx >= skill_rows: break
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            raw = df_skill.loc[idx, tier]
            nm, st = _extract_status_and_clean(raw)
            if nm and all(nm != x.get('player') for x in pos_lists["TE"]):
                pos_lists["TE"].append({"player": nm, "status": st, "active": _status_to_active(st)})
        idx += 1
    return {k: v for k, v in pos_lists.items() if v}


def build_depth_chart_from_espn(season: int, week: int) -> pd.DataFrame:
    rows = []
    for team in ESPN_TEAM_ABBR.keys():
        pos_lists = _scrape_espn_depth_for_team(team)
        for pos, players in pos_lists.items():
            depth_size = len(players)
            for i, row in enumerate(players, start=1):
                rows.append({
                    "season": season,
                    "week": week,
                    "team": team,
                    "position": pos,
                    "player": row.get("player"),
                    "depth_rank": i,
                    "depth_size": depth_size,
                    "status": row.get("status", ""),
                    "active": bool(row.get("active", True)),
                })
    return pd.DataFrame(rows)


def build_depth_chart(season: int, week: int, source: str = "espn") -> pd.DataFrame:
    """Build a weekly depth chart. Default source uses ESPN web page parsing.
    Returns columns: season, week, team, position, player, depth_rank, depth_size, status, active.
    """
    if source == "espn":
        df = build_depth_chart_from_espn(season, week)
    else:
        df = build_depth_chart_from_espn(season, week)
    return df


def save_depth_chart(season: int, week: int, source: str = "espn") -> Path:
    df = build_depth_chart(season, week, source=source)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    df.to_csv(out, index=False)
    return out


def load_depth_chart_csv(season: int, week: int) -> pd.DataFrame:
    fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
        return df
    except Exception:
        return pd.DataFrame()
