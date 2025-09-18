"""
Fetch Bovada NFL player props and write a CSV compatible with props_edges_join.py.

Outputs columns:
  - player (required)
  - team (optional)
    - market (one of: Receiving Yards, Receptions, Rushing Yards, Passing Yards, Passing TDs, Anytime TD)
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
    ("passing touchdowns", "Passing TDs"),
    ("pass touchdowns", "Passing TDs"),
    ("passing tds", "Passing TDs"),
    ("pass tds", "Passing TDs"),
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


def _parse_ladder_threshold(market_desc: Optional[str]) -> Optional[float]:
    """Extract a numeric threshold from ladder-style markets.
    Examples handled:
      - "To Record 100+ Receiving Yards"
      - "To Record 6+ Receptions"
      - "To Record 250+ Passing Yards"
      - "100+ Receiving Yards"
      - "6+ Receptions"
      - "250+ Passing Yards"
    Returns the numeric value as float if found; otherwise None.
    """
    if not market_desc:
        return None
    s = str(market_desc)
    # Pattern A: explicit "To Record <num>+"
    m = re.search(r"to\s+record\s+([0-9]+(?:\.[0-9]+)?)\s*\+", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Pattern B: generic "<num>+ (Receiving|Rushing|Passing|Receptions)"
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\+\s*(receiving|rushing|passing|receptions)(?:\s+yards|\s+yds|\b)", s, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
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
            mname_lower = mname.lower()
            std = _std_market(mname)
            if not std:
                continue  # not a player prop we care about
            is_alternate_market = "alternate" in mname_lower

            # Outcomes often have Over/Under entries with price.handicap = line and participant (player) info
            outcomes = m.get("outcomes") or []

            # Detect ladder threshold if this is a ladder market (e.g., "To Record 100+ Receiving Yards")
            ladder_line = _parse_ladder_threshold(mname)
            is_ladder_market = (ladder_line is not None)

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

            def _pick_outcome_player(oc: Dict[str, Any]) -> Optional[str]:
                """Choose the most likely player string from outcome fields, ignoring generic market text.
                We prefer 'participant'/'competitor' then 'name'/'displayName', and avoid strings starting with
                'To Record' (ladder descriptors) or the words 'Over'/'Under'.
                """
                for fld in ("participant", "competitor", "name", "displayName", "description"):
                    val = oc.get(fld)
                    s = str(val or "").strip()
                    if not s:
                        continue
                    s_norm = _norm(s)
                    if s_norm in ("over", "under"):
                        continue
                    if s_norm.startswith("to record "):
                        # This is the ladder market label, not a player name
                        continue
                    return s
                return None
            def _parse_outcome_threshold(oc: Dict[str, Any]) -> Optional[float]:
                texts = []
                for fld in ("description","name","displayName","market","outcomeDescription"):
                    v = oc.get(fld)
                    if v:
                        texts.append(str(v))
                combo = " ".join(texts)
                return _parse_ladder_threshold(combo)

            # Special handling for "Alternate ..." markets: emit one row per (player, line) with both O/U prices
            if is_alternate_market and std != "Anytime TD":
                # build {(player, line): {over_price, under_price, team_hint}}
                alt_bucket: Dict[Tuple[str, float], Dict[str, Any]] = {}
                for oc in outcomes:
                    side = _norm(oc.get("description"))  # 'over'/'under' expected for alt yards/rec
                    participant = _pick_outcome_player(oc) or player_from_market
                    raw_participant = str(participant or "").strip()
                    if not raw_participant:
                        continue
                    # Strip trailing team tag
                    m_tag = team_tag_re.search(raw_participant)
                    tag_team = m_tag.group(1) if m_tag else None
                    player = team_tag_re.sub("", raw_participant).strip() if m_tag else raw_participant
                    price = oc.get("price", {}) or {}
                    line = price.get("handicap")
                    try:
                        line = float(line) if line is not None and str(line) != "" else np.nan
                    except Exception:
                        line = np.nan
                    if pd.isna(line):
                        # Some alt TDs are like "1+" without handicap; try outcome text
                        oc_thresh = _parse_outcome_threshold(oc)
                        line = oc_thresh if oc_thresh is not None else np.nan
                    if pd.isna(line):
                        # Cannot form a rung without a numeric line
                        continue
                    amer = price.get("american")
                    try:
                        amer = int(str(amer).replace("+", "")) if amer is not None and str(amer) != "" else np.nan
                    except Exception:
                        amer = np.nan
                    key = (player, float(line))
                    if key not in alt_bucket:
                        alt_bucket[key] = {
                            "player": player,
                            "market": std,
                            "line": float(line),
                            "over_price": np.nan,
                            "under_price": np.nan,
                            "team_hint": tag_team or None,
                        }
                    else:
                        if not alt_bucket[key].get("team_hint") and tag_team:
                            alt_bucket[key]["team_hint"] = tag_team
                    if side == "over":
                        alt_bucket[key]["over_price"] = amer
                    elif side == "under":
                        alt_bucket[key]["under_price"] = amer
                    else:
                        # Single-sided (e.g., "1+")
                        alt_bucket[key]["over_price"] = amer

                for row in alt_bucket.values():
                    row.update({
                        "book": "Bovada",
                        "event": event_name,
                        "game_time": start_time,
                        "home_team": home,
                        "away_team": away,
                        "is_ladder": True,  # treat alternate lines as ladder rungs
                    })
                    if team_from_market:
                        row.setdefault("team", team_from_market)
                    else:
                        th = row.get("team_hint")
                        if th and (not row.get("team") or pd.isna(row.get("team"))):
                            row["team"] = th
                        else:
                            row.setdefault("team", np.nan)
                    rows.append(row)
                continue  # done with this market

            # Default handling: aggregate per player, one row per player/market
            bucket: Dict[str, Dict[str, Any]] = {}
            for oc in outcomes:
                side = _norm(oc.get("description"))  # 'over'/'under' expected
                participant = _pick_outcome_player(oc)
                if not participant and player_from_market:
                    participant = player_from_market
                raw_participant = str(participant or "").strip()
                player = raw_participant
                m_tag = team_tag_re.search(player)
                tag_team = m_tag.group(1) if m_tag else None
                if m_tag:
                    player = team_tag_re.sub("", player).strip()
                if not player:
                    if player_from_market:
                        player = player_from_market
                    else:
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
                oc_thresh = _parse_outcome_threshold(oc)
                key = player
                if key not in bucket:
                    bucket[key] = {
                        "player": player,
                        "market": std,
                        "line": np.nan,
                        "over_price": np.nan,
                        "under_price": np.nan,
                        "team_hint": tag_team or None,
                        "_is_ladder": False,
                    }
                else:
                    if not bucket[key].get("team_hint") and tag_team:
                        bucket[key]["team_hint"] = tag_team
                if std != "Anytime TD" and not pd.isna(line):
                    bucket[key]["line"] = line
                if std != "Anytime TD" and pd.isna(bucket[key]["line"]):
                    if oc_thresh is not None:
                        bucket[key]["line"] = oc_thresh
                        bucket[key]["_is_ladder"] = True
                    elif ladder_line is not None:
                        bucket[key]["line"] = ladder_line
                        bucket[key]["_is_ladder"] = True
                if side == "over":
                    bucket[key]["over_price"] = amer
                elif side == "under":
                    bucket[key]["under_price"] = amer
                else:
                    bucket[key]["over_price"] = amer

            for row in bucket.values():
                row.update({
                    "book": "Bovada",
                    "event": event_name,
                    "game_time": start_time,
                    "home_team": home,
                    "away_team": away,
                })
                row["is_ladder"] = bool(row.get("_is_ladder") or is_ladder_market)
                if team_from_market:
                    row.setdefault("team", team_from_market)
                else:
                    th = row.get("team_hint")
                    if th and (not row.get("team") or pd.isna(row.get("team"))):
                        row["team"] = th
                    else:
                        row.setdefault("team", np.nan)
                row.pop("_is_ladder", None)
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
        "book", "event", "game_time", "home_team", "away_team", "is_ladder",
    ]
    out_df = df[[c for c in cols if c in df.columns]].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
