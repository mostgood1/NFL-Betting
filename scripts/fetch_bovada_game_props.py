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
  - team_side: home|away (for moneyline/spread)
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

    # Try to map markets
    for dg in dgs:
        markets = dg.get("markets") or []
        for m in markets:
            mdesc = m.get("description") or m.get("marketType") or ""
            mlow = _norm(mdesc)
            outcomes = m.get("outcomes") or []
            is_alt = ("alternate" in mlow)

            # Moneyline
            if "moneyline" in mlow:
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
                    "price_home": price_home,
                    "price_away": price_away,
                    "is_alternate": False,
                })
                continue

            # Point Spread
            if "spread" in mlow and "total" not in mlow:
                # We expect two outcomes: one for home, one for away, each with handicap and price
                line_home = np.nan; price_home = np.nan
                line_away = np.nan; price_away = np.nan
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
                    if home and _norm(home) in part:
                        line_home = hcap; price_home = amer
                    elif away and _norm(away) in part:
                        line_away = hcap; price_away = amer
                add_row({
                    "market_key": ("alt_spread" if is_alt else "spread"),
                    "team_side": "home",
                    "line": line_home,
                    "price_home": price_home,
                    "price_away": price_away,
                    "is_alternate": is_alt,
                })
                add_row({
                    "market_key": ("alt_spread" if is_alt else "spread"),
                    "team_side": "away",
                    "line": line_away,
                    "price_home": price_home,
                    "price_away": price_away,
                    "is_alternate": is_alt,
                })
                continue

            # Totals (game total Over/Under)
            if "total" in mlow and "team total" not in mlow:
                line = np.nan
                over_p = np.nan; under_p = np.nan
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
                    if not pd.isna(hcap):
                        line = hcap
                    if "over" in desc:
                        over_p = amer
                    elif "under" in desc:
                        under_p = amer
                add_row({
                    "market_key": ("alt_total" if is_alt else "total"),
                    "line": line,
                    "over_price": over_p,
                    "under_price": under_p,
                    "is_alternate": is_alt,
                })
                continue

            # Team Total (per-team Over/Under)
            if "team total" in mlow:
                # Identify which team this market refers to (from description)
                team_side = None
                desc_all = _norm(mdesc)
                if home and _norm(home) in desc_all:
                    team_side = "home"
                elif away and _norm(away) in desc_all:
                    team_side = "away"
                # Collect O/U
                line = np.nan; over_p = np.nan; under_p = np.nan
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
                    if not pd.isna(hcap):
                        line = hcap
                    if "over" in desc:
                        over_p = amer
                    elif "under" in desc:
                        under_p = amer
                add_row({
                    "market_key": "team_total",
                    "team_side": team_side,
                    "line": line,
                    "over_price": over_p,
                    "under_price": under_p,
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
        "market_key","team_side","ou_side","line",
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
