"""
Odds API client for NFL odds.

Fetches markets (moneyline, spreads, totals) from The Odds API (or compatible)
and writes unified JSON to data/real_betting_lines_YYYY_MM_DD.json expected by
the existing loader.

Env vars:
  ODDS_API_KEY: API key token
  ODDS_API_BASE (optional): override base URL (default https://api.the-odds-api.com/v4)
  ODDS_API_REGION (optional): default 'us'
  ODDS_API_BOOKS (optional): comma-separated preferred books (e.g., 'draftkings,fanduel')

Run:
    From the repo root:
        python -m src.odds_api_client (with working dir set to nfl_compare)
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .team_normalizer import normalize_team_name
from .config import load_env as _load_env


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    _load_env()
    v = os.environ.get(name)
    return v if v is not None else default


def _get_base_url() -> str:
    return _env("ODDS_API_BASE", "https://api.the-odds-api.com/v4")


def _preferred_books() -> List[str]:
    raw = _env("ODDS_API_BOOKS", "draftkings,fanduel,betmgm,pointsbetus,caesars") or ""
    return [b.strip().lower() for b in raw.split(",") if b.strip()]


def fetch_odds(
    api_key: str,
    sport_key: str = "americanfootball_nfl",
    region: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
) -> List[Dict[str, Any]]:
    base = _get_base_url()
    url = f"{base}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _choose_bookmaker(bookmakers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not bookmakers:
        return None
    prefs = _preferred_books()
    # Try preferred list
    name_to_bm = {bm.get("key", "").lower(): bm for bm in bookmakers}
    for pref in prefs:
        if pref in name_to_bm:
            return name_to_bm[pref]
    # Fallback: first bookmaker
    return bookmakers[0]


def _extract_markets(bm: Dict[str, Any]) -> Dict[str, Any]:
    markets = []
    moneyline = None
    total_runs = None
    run_line = None

    for m in bm.get("markets", []) or []:
        key = m.get("key")
        outcomes = m.get("outcomes", []) or []
        markets.append({
            "key": key,
            "outcomes": outcomes,
        })

        if key in ("h2h", "moneyline"):
            # Structure as {'home': odds, 'away': odds} later (we keep raw for now)
            pass
        elif key in ("totals", "total"):
            # We'll also snapshot into total_runs shortcut if possible
            if outcomes:
                # try to capture the point once
                pt = next((o.get("point") for o in outcomes if o.get("point") is not None), None)
                over = next((o.get("price") for o in outcomes if str(o.get("name", "")).lower().startswith("over")), None)
                under = next((o.get("price") for o in outcomes if str(o.get("name", "")).lower().startswith("under")), None)
                total_runs = {"line": pt, "over": over, "under": under}
        elif key in ("spreads", "spread"):
            # Keep raw; we'll derive home spread via team matching
            pass

    return {
        "markets": markets,
        "moneyline": moneyline,
        "total_runs": total_runs,
        "run_line": run_line,
    }


def _compose_key(away: str, home: str) -> str:
    return f"{away} @ {home}"


def build_unified_lines(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines: Dict[str, Any] = {}

    for ev in events:
        try:
            # Normalize team names
            raw_away = ev.get("away_team") or ev.get("awayTeam") or ev.get("teams", [None, None])[0]
            raw_home = ev.get("home_team") or ev.get("homeTeam") or ev.get("teams", [None, None])[1]
            if not raw_away or not raw_home:
                # Some feeds use 'teams' and 'home_team' separately; ensure both exist
                continue
            away = normalize_team_name(raw_away)
            home = normalize_team_name(raw_home)

            bm = _choose_bookmaker(ev.get("bookmakers") or [])
            if not bm:
                # no odds for this event
                continue
            mk = _extract_markets(bm)

            # Attempt to fill convenience fields for moneyline and spread
            ml_home = None
            ml_away = None
            spread_home = None

            for market in (bm.get("markets") or []):
                key = market.get("key")
                outcomes = market.get("outcomes", []) or []
                if key in ("h2h", "moneyline"):
                    for o in outcomes:
                        name = normalize_team_name(o.get("name", ""))
                        if name.lower() == home.lower():
                            ml_home = o.get("price")
                        elif name.lower() == away.lower():
                            ml_away = o.get("price")
                elif key in ("spreads", "spread"):
                    # find the home team outcome to get point
                    for o in outcomes:
                        name = normalize_team_name(o.get("name", ""))
                        if name.lower() == home.lower():
                            spread_home = o.get("point")

            unified = {
                # convenience fields
                "moneyline": {"home": ml_home, "away": ml_away} if (ml_home is not None or ml_away is not None) else None,
                "total_runs": mk.get("total_runs"),
                "run_line": mk.get("run_line"),
                # always include full markets for downstream parsing
                "markets": mk.get("markets", []),
            }

            # If we extracted a spread, add run_line with a simple structure
            if spread_home is not None:
                unified["run_line"] = {"home": spread_home}

            key = _compose_key(away, home)
            lines[key] = unified
        except Exception:
            continue

    return {"lines": lines, "source": "odds_api", "fetched_at": datetime.now(timezone.utc).isoformat()}


def write_daily_lines(payload: Dict[str, Any]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today_us = datetime.now().strftime("%Y_%m_%d")
    fp = DATA_DIR / f"real_betting_lines_{today_us}.json"
    fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return fp


def main() -> None:
    api_key = _env("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY environment variable")

    region = _env("ODDS_API_REGION", "us") or "us"
    events = fetch_odds(api_key=api_key, region=region)
    unified = build_unified_lines(events)
    out = write_daily_lines(unified)
    print(f"Saved real odds to {out}")


if __name__ == "__main__":
    main()
