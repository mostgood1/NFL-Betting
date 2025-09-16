"""
Fetch Bovada NFL player props and write a CSV compatible with props_edges_join.py.

Outputs columns:
  - player (required)
  - team (optional)
  - market (one of: Receiving Yards, Receptions, Rushing Yards, Passing Yards, Anytime TD)
  - line (numeric, where applicable)
  - over_price (American odds, optional)
  - under_price (American odds, optional)
  - book ("Bovada")
  - event (optional: event description)
  - game_time (optional ISO timestamp if available)

Usage:
  python scripts/fetch_bovada_props.py --season 2025 --week 3 \
    --out nfl_compare/data/bovada_player_props_2025_wk3.csv

Notes:
  - This relies on Bovada's public event feed. Endpoint may change; pass a custom --url if needed.
  - We avoid synthesizing odds. If a side is missing, the corresponding price is left blank.
  - Matching/market parsing is best-effort and resilient to minor schema changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import requests
import pandas as pd
import numpy as np


DEFAULT_URL = (
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl"
    "?preMatchOnly=true&lang=en"
)

# Market phrase mapping -> standardized market label used by our join script
MARKET_ALIASES: List[Tuple[str, str]] = [
    ("receiving yards", "Receiving Yards"),
    ("receptions", "Receptions"),
    ("rushing yards", "Rushing Yards"),
    ("passing yards", "Passing Yards"),
    ("anytime touchdown", "Anytime TD"),
    ("any time touchdown", "Anytime TD"),
    ("anytime td", "Anytime TD"),
]


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _std_market(name: str) -> Optional[str]:
    n = _norm(name)
    for key, std in MARKET_ALIASES:
        if key in n:
            return std
    return None


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
        # Some Bovada endpoints return JSON embedded in an array/string; try fallback
        return json.loads(resp.text)


def iter_events(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield event dicts from Bovada payload (handles v2 grouping structure)."""
    if isinstance(payload, dict) and "events" in payload:
        for ev in payload.get("events", []) or []:
            yield ev
        return
    if isinstance(payload, list):
        # v2 format commonly returns a list of groups, each with 'events'
        for grp in payload:
            for ev in grp.get("events", []) or []:
                yield ev
        return
    # Unknown shape; nothing to yield
    return []


def extract_competitors(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    home = None
    away = None
    for comp in ev.get("competitors", []) or []:
        nm = comp.get("shortName") or comp.get("name") or comp.get("abbreviation")
        nm = nm or comp.get("description")
        if comp.get("home"):  # bool
            home = nm
        else:
            away = nm
    return home, away


def parse_event_markets(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    event_name = ev.get("description")
    start_time = ev.get("startTime")  # epoch ms
    # Markets might be nested under multiple display groups
    dgs = ev.get("displayGroups") or []
    home, away = extract_competitors(ev)

    for dg in dgs:
        markets = dg.get("markets") or []
        for m in markets:
            mname = m.get("description") or m.get("marketType") or ""
            std = _std_market(mname)
            if not std:
                continue  # not a player prop we care about

            # Outcomes often have Over/Under entries with price.handicap = line and participant (player) info
            outcomes = m.get("outcomes") or []
            # Build per-player container
            bucket: Dict[str, Dict[str, Any]] = {}

            # Some Bovada markets are per-player (player name appears in market description)
            # e.g., "Receiving Yards - Dalton Kincaid (BUF)".
            # Try to extract player and optional team from the market description for fallback.
            player_from_market: Optional[str] = None
            team_from_market: Optional[str] = None
            if " - " in mname:
                # Take substring after the last ' - '
                after = mname.split(" - ")[-1].strip()
                # Remove trailing team in parentheses, capture as team if present
                m_player = re.sub(r"\s*\(.*?\)\s*$", "", after).strip()
                player_from_market = m_player if m_player else None
                m_team = re.findall(r"\(([^)]+)\)\s*$", after)
                if m_team:
                    team_from_market = m_team[0].strip()

            team_tag_re = re.compile(r"\s*\(([A-Za-z]{2,4})\)\s*$")
            for oc in outcomes:
                side = _norm(oc.get("description"))  # 'over'/'under' expected
                # Participant can be nested in outcome or referenced via 'participant'
                participant = (
                    oc.get("participant")
                    or oc.get("competitor")
                    or oc.get("name")
                    or oc.get("displayName")
                    or oc.get("description")
                )
                # Some outcomes have 'description' = Over/Under; ensure we don't use that as player
                if _norm(participant) in ("over", "under"):
                    participant = oc.get("participant") or oc.get("name") or oc.get("displayName")
                    if not participant and player_from_market:
                        participant = player_from_market

                player = str(participant or "").strip()
                # Remove trailing team tag from player name like "Tyreek Hill (MIA)"
                m_tag = team_tag_re.search(player)
                tag_team = m_tag.group(1) if m_tag else None
                if m_tag:
                    player = team_tag_re.sub("", player).strip()
                if not player:
                    # As a last resort, try market-level player
                    if player_from_market:
                        player = player_from_market
                    else:
                        # Skip if we cannot identify the player
                        continue

                price = oc.get("price", {}) or {}
                line = price.get("handicap")
                try:
                    line = float(line) if line is not None and str(line) != "" else np.nan
                except Exception:
                    line = np.nan

                amer = price.get("american")
                try:
                    amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                except Exception:
                    amer = np.nan

                key = player
                bucket.setdefault(key, {"player": player, "market": std, "line": np.nan, "over_price": np.nan, "under_price": np.nan})
                # Update line if present
                if std != "Anytime TD" and not pd.isna(line):
                    bucket[key]["line"] = line
                # Assign side price
                if side == "over":
                    bucket[key]["over_price"] = amer
                elif side == "under":
                    bucket[key]["under_price"] = amer
                else:
                    # Unknown side; emit as single-sided row by over_price
                    bucket[key]["over_price"] = amer

            for row in bucket.values():
                row.update({
                    "book": "Bovada",
                    "event": event_name,
                    "game_time": start_time,
                    "home_team": home,
                    "away_team": away,
                })
                # Try to attach a team based on substring match in event description if not clear
                # Keep 'team' optional; consumer may ignore or use home/away fields
                if team_from_market:
                    row.setdefault("team", team_from_market)
                else:
                    # If player text had a team tag like (MIA), use it if team is missing
                    if tag_team and (not row.get("team") or pd.isna(row.get("team"))):
                        row["team"] = tag_team
                    else:
                        row.setdefault("team", np.nan)
                rows.append(row)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Bovada NFL player props to CSV")
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
        rows.extend(parse_event_markets(ev))

    if not rows:
        print("WARNING: No player prop rows parsed from Bovada payload. Schema may have changed or endpoint lacks props.")

    df = pd.DataFrame(rows)
    # Reorder/select columns
    cols = [
        "player", "team", "market", "line", "over_price", "under_price",
        "book", "event", "game_time", "home_team", "away_team",
    ]
    out_df = df[[c for c in cols if c in df.columns]].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
