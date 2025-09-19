"""
Fetch Bovada NFL game-level markets and write a CSV for edge computation.

Markets captured (best-effort):
  - Moneyline (home/away prices)
  - Point Spread (home/away lines and prices)
  - Total Points (Over/Under line and prices)
  - Alternate Spread / Alternate Total (treated similarly with is_alternate=true)
  - Team Total Points (per-team Over/Under line and prices)

Output columns (subset may be empty depending on available markets):
    - event, game_time, home_team, away_team
    - market_key: moneyline|spread|total|alt_spread|alt_total|team_total
    - market_name: original market description from feed (e.g., "Total Points")
    - period: G|1H|2H|1Q|2Q|3Q|4Q (inferred from display group/market text)
    - team_side: home|away (for moneyline/spread/team_total context)
    - ou_side: Over|Under (for total/team_total)
    - line (numeric for spread/total/team_total)
    - price_home, price_away (for moneyline/spread)
    - over_price, under_price (for total/team_total)
    - is_alternate (bool)

Usage:
  python scripts/fetch_bovada_game_props.py --season 2025 --week 3 \
    --out nfl_compare/data/bovada_game_props_2025_wk3.csv

Notes:
  - This relies on Bovada's public event feed. Endpoint may change; pass a custom --url if needed.
  - Parsing is best-effort; schema changes are handled defensively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import re


DEFAULT_URL = (
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl"
    "?preMatchOnly=true&lang=en"
)


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def fetch_json(url: str) -> Any:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bovada.lv/",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        return json.loads(resp.text)


def iter_events(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict) and "events" in payload:
        for ev in payload.get("events", []) or []:
            yield ev
        return
    if isinstance(payload, list):
        for grp in payload:
            for ev in grp.get("events", []) or []:
                yield ev
        return
    return []


def extract_competitors(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    home = None
    away = None
    for comp in ev.get("competitors", []) or []:
        nm = comp.get("shortName") or comp.get("name") or comp.get("abbreviation") or comp.get("description")
        if comp.get("home"):
            home = nm
        else:
            away = nm
    return home, away


def parse_event_game_markets(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    event_name = ev.get("description")
    start_time = ev.get("startTime")  # epoch ms
    dgs = ev.get("displayGroups") or []
    home, away = extract_competitors(ev)

    def add_row(r: Dict[str, Any]):
        r.update({
            "event": event_name,
            "game_time": start_time,
            "home_team": home,
            "away_team": away,
        })
        rows.append(r)

    def infer_period(texts: List[str]) -> str:
        blob = " ".join([_norm(t) for t in texts if t])
        # Prioritize quarter before half to avoid "1st half" matching "1st"
        if re.search(r"1st\s*quarter|first\s*quarter|q1", blob):
            return "1Q"
        if re.search(r"2nd\s*quarter|second\s*quarter|q2", blob):
            return "2Q"
        if re.search(r"3rd\s*quarter|third\s*quarter|q3", blob):
            return "3Q"
        if re.search(r"4th\s*quarter|fourth\s*quarter|q4", blob):
            return "4Q"
        if re.search(r"1st\s*half|first\s*half|h1", blob):
            return "1H"
        if re.search(r"2nd\s*half|second\s*half|h2", blob):
            return "2H"
        return "G"

    def is_game_lines_group(desc: str) -> bool:
        dl = _norm(desc)
        return any(k in dl for k in ["game lines", "game line", "main", "featured", "full game"]) and not any(
            k in dl for k in ["props", "race", "player", "team props", "quarter", "half"]
        )

    def is_alternate_group(desc: str) -> bool:
        dl = _norm(desc)
        return "alternate" in dl or "alts" in dl

    # Try to map markets
    for dg in dgs:
        markets = dg.get("markets") or []
        dg_desc = dg.get("description") or dg.get("displayGroup") or ""
        dg_low = _norm(dg_desc)
        period = infer_period([dg_desc])
        for m in markets:
            mdesc = m.get("description") or m.get("marketType") or ""
            mlow = _norm(mdesc)
            outcomes = m.get("outcomes") or []
            is_alt = ("alternate" in mlow) or is_alternate_group(dg_desc)

            # Keep original name and period metadata
            market_name = mdesc
            m_period = infer_period([dg_desc, mdesc]) or period

            # Moneyline
            if "moneyline" in mlow and m_period == "G":
                price_home = np.nan
                price_away = np.nan
                for oc in outcomes:
                    part = _norm(oc.get("participant") or oc.get("competitor") or oc.get("name") or oc.get("displayName"))
                    amer = oc.get("price", {}).get("american")
                    try:
                        amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                    except Exception:
                        amer = np.nan
                    if home and _norm(home) in part:
                        price_home = amer
                    elif away and _norm(away) in part:
                        price_away = amer
                add_row({
                    "market_key": "moneyline",
                    "market_name": market_name,
                    "period": m_period,
                    "price_home": price_home,
                    "price_away": price_away,
                    "is_alternate": False,
                })
                continue

            # Winning Margin (game or half)
            if "winning margin" in mlow or "dynamic winning margin" in mlow:
                # Outcomes like "Bills by 1-6 points" or "Tie" or with - 1H suffix
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    team_side = None; a=None; b=None; tie=False
                    m = re.search(r"(.*?)(?:\s*-\s*1h)?\s*by\s*(\d+)(?:\s*[-–]\s*(\d+)|\s*or\s*more)?", od)
                    if m:
                        who = m.group(1).strip()
                        a = float(m.group(2)); b = float(m.group(3)) if m.group(3) else None
                        if b is None and re.search(r"or\s*more", od):
                            b = None  # open-ended upper
                        wlow = _norm(who)
                        if home and _norm(home) in wlow:
                            team_side = "home"
                        elif away and _norm(away) in wlow:
                            team_side = "away"
                    elif od.startswith("tie") or od == "tie":
                        tie = True
                    add_row({
                        "market_key": "winning_margin",
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": team_side,
                        "range_low": a,
                        "range_high": b,
                        "tie": tie,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Total Points Range (game or period)
            if "total points range" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    a=None; b=None; rtype="between"
                    m = re.search(r"between\s*(\d+)\s*and\s*(\d+)", od)
                    if m:
                        a = float(m.group(1)); b = float(m.group(2)); rtype = "between"
                    m2 = re.search(r"(\d+)\s*or\s*less", od)
                    if m2:
                        a = float(m2.group(1)); b = None; rtype = "le"
                    m3 = re.search(r"(\d+)\s*or\s*more", od)
                    if m3:
                        a = float(m3.group(1)); b = None; rtype = "ge"
                    add_row({
                        "market_key": "total_range",
                        "market_name": market_name,
                        "period": m_period,
                        "range_low": a,
                        "range_high": b,
                        "range_type": rtype,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Both teams to score N+ points (game or period)
            if "both teams to score" in mlow and "points" in mlow:
                # Extract threshold
                thr = None
                mt = re.search(r"(\d+)\s*or\s*more", mlow)
                if mt:
                    thr = float(mt.group(1))
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    side = "Yes" if od.startswith("yes") else "No" if od.startswith("no") else None
                    add_row({
                        "market_key": "btts_points",
                        "market_name": market_name,
                        "period": m_period,
                        "threshold": thr,
                        "side": side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Spread + Total combo (SGP-like)
            if ("point spread" in mlow and "o/u" in mlow) or re.search(r"point\s*spread.*o/\s*u", mlow):
                # From market name, pull spread_line and total_line
                sp = re.search(r"spread\s*([+-]?[0-9]+(?:\.[0-9])?)", mlow)
                tp = re.search(r"o/\s*u\s*([0-9]+(?:\.[0-9])?)", mlow)
                spread_line = float(sp.group(1)) if sp else None
                total_line = float(tp.group(1)) if tp else None
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    team_side=None; total_side=None
                    if home and _norm(home) in od:
                        team_side="home"
                    elif away and _norm(away) in od:
                        team_side="away"
                    if "over" in od:
                        total_side = "Over"
                    elif "under" in od:
                        total_side = "Under"
                    add_row({
                        "market_key": "spread_total_combo",
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": team_side,
                        "spread_line": spread_line,
                        "total_line": total_line,
                        "total_side": total_side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Highest scoring half
            if "highest scoring half" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    side = "1H" if od.startswith("first") else "2H" if od.startswith("second") else None
                    add_row({
                        "market_key": "highest_scoring_half",
                        "market_name": market_name,
                        "period": m_period,
                        "side": side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Odd/Even Total Points (game or period)
            if "odd/even total points" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    side = "Odd" if od.startswith("odd") else "Even" if od.startswith("even") else None
                    add_row({
                        "market_key": "odd_even",
                        "market_name": market_name,
                        "period": m_period,
                        "side": side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Overtime (Regulation Time)
            if ("overtime" in mlow and "regulation" in mlow) or mlow.startswith("will the game go to overtime"):
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    side = "Yes" if od.startswith("yes") else "No" if od.startswith("no") else None
                    add_row({
                        "market_key": "overtime",
                        "market_name": market_name,
                        "period": m_period,
                        "side": side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # First Team To Score
            if "first team to score" in mlow and m_period == "G":
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    team_side=None
                    if home and _norm(home) in od:
                        team_side="home"
                    elif away and _norm(away) in od:
                        team_side="away"
                    add_row({
                        "market_key": "first_to_score",
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": team_side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Race to N Points (first to N)
            if re.search(r"race\s*to\s*\d+", mlow):
                mnum = re.search(r"race\s*to\s*(\d+)", mlow)
                thr = float(mnum.group(1)) if mnum else None
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    team_side=None; neither=False
                    if home and _norm(home) in od:
                        team_side="home"
                    elif away and _norm(away) in od:
                        team_side="away"
                    elif od.startswith("neither"):
                        neither=True
                    add_row({
                        "market_key": "race_to_points",
                        "market_name": market_name,
                        "period": m_period,
                        "threshold": thr,
                        "team_side": team_side,
                        "neither": neither,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Both Teams to Score N+ and Winner
            if "both teams to score" in mlow and "and winner" in mlow:
                thr = None
                mt = re.search(r"(\d+)\s*or\s*more", mlow)
                if mt:
                    thr = float(mt.group(1))
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    winner=None
                    if home and _norm(home) in od:
                        winner="home"
                    elif away and _norm(away) in od:
                        winner="away"
                    add_row({
                        "market_key": "btts_and_winner",
                        "market_name": market_name,
                        "period": m_period,
                        "threshold": thr,
                        "winner": winner,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Double Chance (could be period specific)
            if "double chance" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    combo = None
                    # Normalize to HOME/AWAY/DRAW
                    parts = [p.strip() for p in od.replace("- 1h","-").replace("- reg","-").split("/")]
                    norm = []
                    for p in parts:
                        pu = _norm(p)
                        if home and _norm(home) in pu:
                            norm.append("HOME")
                        elif away and _norm(away) in pu:
                            norm.append("AWAY")
                        elif pu.startswith("draw"):
                            norm.append("DRAW")
                    if norm:
                        combo = "/".join(norm)
                    add_row({
                        "market_key": "double_chance",
                        "market_name": market_name,
                        "period": m_period,
                        "combo": combo,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # To Win and Allow 0 Points (win to nil)
            if "to win and allow 0 points" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    team_side=None
                    if home and _norm(home) in od:
                        team_side="home"
                    elif away and _norm(away) in od:
                        team_side="away"
                    add_row({
                        "market_key": "win_to_nil",
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": team_side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Half Time / Full Time (Regulation)
            if ("half time / full time" in mlow) or ("ht / ft" in mlow) or ("half time" in mlow and "full time" in mlow):
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    # Format like "Bills - Dolphins - REG" or "Bills - Draw - REG"
                    parts = [p.strip() for p in od.replace("- reg", "-").split("-") if p.strip()]
                    ht=None; ft=None
                    if len(parts) >= 2:
                        def norm_team(x):
                            xu = _norm(x)
                            if home and _norm(home) in xu:
                                return "HOME"
                            if away and _norm(away) in xu:
                                return "AWAY"
                            if xu.startswith("draw"):
                                return "DRAW"
                            return None
                        ht = norm_team(parts[0]); ft = norm_team(parts[1])
                    add_row({
                        "market_key": "ht_ft",
                        "market_name": market_name,
                        "period": m_period,
                        "ht_result": ht,
                        "ft_result": ft,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Highest Scoring Quarter
            if "highest scoring quarter" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    side = None
                    if od.startswith("first"):
                        side = "1Q"
                    elif od.startswith("second"):
                        side = "2Q"
                    elif od.startswith("third"):
                        side = "3Q"
                    elif od.startswith("fourth"):
                        side = "4Q"
                    add_row({
                        "market_key": "highest_scoring_quarter",
                        "market_name": market_name,
                        "period": m_period,
                        "side": side,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue

            # Largest Lead of the Game (ranges)
            if "largest lead" in mlow:
                for oc in outcomes:
                    od = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or "")
                    amer = (oc.get("price", {}) or {}).get("american")
                    try:
                        amer = int(str(amer).replace("+","")) if amer not in (None,"") else np.nan
                    except Exception:
                        amer = np.nan
                    a=None; b=None
                    m = re.search(r"(\d+)\s*and\s*under", od)
                    if m:
                        b = float(m.group(1))
                    m2 = re.search(r"between\s*(\d+)\s*and\s*(\d+)", od)
                    if m2:
                        a = float(m2.group(1)); b = float(m2.group(2))
                    m3 = re.search(r"(\d+)\s*and\s*over", od)
                    if m3:
                        a = float(m3.group(1))
                    add_row({
                        "market_key": "largest_lead_range",
                        "market_name": market_name,
                        "period": m_period,
                        "range_low": a,
                        "range_high": b,
                        "price": amer,
                        "is_alternate": False,
                    })
                continue
            # Point Spread - produce a row per unique handicap per side
            if ("point spread" in mlow or ("spread" in mlow and "total" not in mlow)) and m_period == "G":
                ladder = {}
                for oc in outcomes:
                    part = _norm(oc.get("participant") or oc.get("competitor") or oc.get("name") or oc.get("displayName"))
                    price = oc.get("price", {}) or {}
                    hcap = price.get("handicap")
                    try:
                        hcap = float(hcap) if hcap is not None and str(hcap) != "" else np.nan
                    except Exception:
                        hcap = np.nan
                    amer = price.get("american")
                    try:
                        amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                    except Exception:
                        amer = np.nan
                    if pd.isna(hcap):
                        continue
                    side = None
                    if home and _norm(home) in part:
                        side = "home"
                    elif away and _norm(away) in part:
                        side = "away"
                    if side is None:
                        continue
                    rec = ladder.setdefault(hcap, {"home": np.nan, "away": np.nan})
                    rec[side] = amer
                for hcap, rec in ladder.items():
                    add_row({
                        "market_key": ("alt_spread" if is_alt else "spread"),
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": "home",
                        "line": hcap,
                        "price_home": rec.get("home"),
                        "price_away": rec.get("away"),
                        "is_alternate": is_alt,
                    })
                    add_row({
                        "market_key": ("alt_spread" if is_alt else "spread"),
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": "away",
                        "line": hcap,
                        "price_home": rec.get("home"),
                        "price_away": rec.get("away"),
                        "is_alternate": is_alt,
                    })
                continue

            # Totals (game total Over/Under) - one row per line with both O/U prices
            if ("total points" in mlow or mlow == "totals") and "team total" not in mlow and m_period == "G":
                ladder = {}
                for oc in outcomes:
                    desc = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or oc.get("outcomeDescription"))
                    price = oc.get("price", {}) or {}
                    hcap = price.get("handicap")
                    try:
                        hcap = float(hcap) if hcap is not None and str(hcap) != "" else np.nan
                    except Exception:
                        hcap = np.nan
                    amer = price.get("american")
                    try:
                        amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                    except Exception:
                        amer = np.nan
                    if pd.isna(hcap):
                        continue
                    rec = ladder.setdefault(hcap, {"over": np.nan, "under": np.nan})
                    if "over" in desc:
                        rec["over"] = amer
                    elif "under" in desc:
                        rec["under"] = amer
                for hcap, rec in ladder.items():
                    add_row({
                        "market_key": ("alt_total" if is_alt else "total"),
                        "market_name": market_name,
                        "period": m_period,
                        "line": hcap,
                        "over_price": rec.get("over"),
                        "under_price": rec.get("under"),
                        "is_alternate": is_alt,
                    })
                continue

            # Team Total (per-team Over/Under)
            if "team total" in mlow and m_period == "G":
                # Identify which team this market refers to (from description)
                team_side = None
                desc_all = _norm(mdesc)
                if home and _norm(home) in desc_all:
                    team_side = "home"
                elif away and _norm(away) in desc_all:
                    team_side = "away"
                # Collect O/U
                ladder = {}
                for oc in outcomes:
                    price = oc.get("price", {}) or {}
                    hcap = price.get("handicap")
                    try:
                        hcap = float(hcap) if hcap is not None and str(hcap) != "" else np.nan
                    except Exception:
                        hcap = np.nan
                    amer = price.get("american")
                    try:
                        amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                    except Exception:
                        amer = np.nan
                    desc = _norm(oc.get("description") or oc.get("name") or oc.get("displayName") or oc.get("outcomeDescription"))
                    if pd.isna(hcap):
                        continue
                    rec = ladder.setdefault(hcap, {"over": np.nan, "under": np.nan})
                    if "over" in desc:
                        rec["over"] = amer
                    elif "under" in desc:
                        rec["under"] = amer
                for hcap, rec in ladder.items():
                    add_row({
                        "market_key": "team_total",
                        "market_name": market_name,
                        "period": m_period,
                        "team_side": team_side,
                        "line": hcap,
                        "over_price": rec.get("over"),
                        "under_price": rec.get("under"),
                        "is_alternate": False,
                    })
                continue

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Bovada NFL game-level markets to CSV")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--url", type=str, default=DEFAULT_URL, help="Bovada NFL events API URL")
    args = ap.parse_args()

    try:
        payload = fetch_json(args.url)
    except Exception as e:
        print(f"ERROR fetching Bovada data: {e}")
        return 2

    rows: List[Dict[str, Any]] = []
    for ev in iter_events(payload):
        rows.extend(parse_event_game_markets(ev))

    df = pd.DataFrame(rows)
    cols = [
        "event","game_time","home_team","away_team",
        "market_key","market_name","period","team_side","ou_side","line",
        "price_home","price_away","over_price","under_price",
        "is_alternate",
    ]
    out_df = df[[c for c in cols if c in df.columns]].copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
