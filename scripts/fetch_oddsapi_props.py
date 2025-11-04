"""
Fetch NFL player props from The Odds API and write a CSV compatible with props_edges_join.py.

Outputs columns:
  - player (required)
  - team (optional)
  - market (one of: Receiving Yards, Receptions, Rushing Yards, Passing Yards, Passing TDs, Anytime TD,
            Passing Attempts, Rushing Attempts, Interceptions)
  - line (numeric, where applicable)
  - over_price (American odds; 'Yes' for Anytime TD)
  - under_price (American odds; 'No' for Anytime TD)
  - book (bookmaker key)
  - event (optional: Away @ Home)
  - game_time (optional ISO timestamp)
  - home_team, away_team
  - is_ladder (False; we don't synthesize ladders here)

Usage:
  python scripts/fetch_oddsapi_props.py --season 2025 --week 3 \
    --out nfl_compare/data/oddsapi_player_props_2025_wk3.csv

Notes:
  - Requires ODDS_API_KEY in environment; .env auto-loaded via nfl_compare.src.config
  - Markets requested can be overridden with ODDS_API_PLAYER_MARKETS env var (comma-separated)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timezone

import requests
from requests.exceptions import HTTPError
import pandas as pd
import numpy as np

# Reuse shared helpers for env loading and team normalization
try:
    from nfl_compare.src.config import load_env as _load_env
    from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
except Exception:  # pragma: no cover
    def _load_env() -> None:
        return None
    def _norm_team(s: Optional[str]) -> str:
        return str(s or "").strip()

# Borrow bookmaker preference logic from odds_api_client
try:
    from nfl_compare.src.odds_api_client import _get_base_url as _get_base_url, _preferred_books as _preferred_books
except Exception:
    def _get_base_url() -> str:
        return os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
    def _preferred_books() -> List[str]:
        raw = os.environ.get("ODDS_API_BOOKS", "draftkings,fanduel,betmgm,pointsbetus,caesars") or ""
        return [b.strip().lower() for b in raw.split(",") if b.strip()]


DEFAULT_PLAYER_MARKETS = [
    # Receiving
    "player_rec_yds",       # Receiving Yards
    "player_receptions",    # Receptions
    # Rushing
    "player_rush_yds",      # Rushing Yards
    "player_rush_attempts", # Rushing Attempts
    # Passing
    "player_pass_yds",      # Passing Yards
    "player_pass_tds",      # Passing TDs
    "player_pass_attempts", # Passing Attempts
    "player_interceptions", # Interceptions thrown
    # Touchdowns
    "player_anytime_td",    # Anytime TD (Yes/No)
]

MARKET_STD_MAP: Dict[str, str] = {
    # OddsAPI market key -> standardized market string used by join script
    "player_rec_yds": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_rush_yds": "Rushing Yards",
    "player_rush_attempts": "Rushing Attempts",
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_attempts": "Passing Attempts",
    "player_interceptions": "Interceptions",
    "player_anytime_td": "Anytime TD",
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    _load_env()
    v = os.environ.get(name)
    return v if v is not None else default


def _player_markets() -> List[str]:
    raw = _env("ODDS_API_PLAYER_MARKETS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_PLAYER_MARKETS)


def _choose_bookmaker(bookmakers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not bookmakers:
        return None
    prefs = _preferred_books()
    name_to_bm = {str(bm.get("key", "")).lower(): bm for bm in bookmakers}
    for pref in prefs:
        if pref in name_to_bm:
            return name_to_bm[pref]
    return bookmakers[0]


def fetch_player_props(api_key: str, sport_key: str = "americanfootball_nfl", region: str = "us", markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    base = _get_base_url()
    url = f"{base}/sports/{sport_key}/odds"
    mkts = ",".join(markets or _player_markets())
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": mkts,
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_player_props_chunked(api_key: str, sport_key: str = "americanfootball_nfl", region: str = "us", markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch events by requesting markets individually to bypass 422s for specific markets.
    Returns concatenated events lists (may contain duplicate events across markets which is okay for row parsing).
    """
    agg_events: List[Dict[str, Any]] = []
    markets_list = markets or _player_markets()
    for m in markets_list:
        try:
            ev = fetch_player_props(api_key=api_key, sport_key=sport_key, region=region, markets=[m])
            agg_events.extend(ev)
        except HTTPError as he:
            code = getattr(getattr(he, 'response', None), 'status_code', None)
            if code == 422:
                print(f"WARNING: OddsAPI returned 422 for market '{m}'. Skipping this market.")
                continue
            raise
        except Exception as e:
            print(f"WARNING: Failed fetching market '{m}': {e}")
            continue
    return agg_events


def _is_side_str(s: str) -> bool:
    x = s.strip().lower()
    return x in ("over", "under", "yes", "no")


def _side_key(s: str) -> Optional[str]:
    x = s.strip().lower()
    if x in ("over", "yes"): return "over"
    if x in ("under", "no"): return "under"
    return None


def _pick_player_from_outcome(oc: Dict[str, Any]) -> Optional[str]:
    # Try typical fields that may contain the player name
    for fld in ("name", "description", "participant", "competitor"):
        val = oc.get(fld)
        if not val:
            continue
        s = str(val).strip()
        if not s:
            continue
        if _is_side_str(s):
            continue
        return s
    # If the outcome only has a side (Over/Under), The Odds API usually provides player name in counterpart fields;
    # without it we cannot attribute the row—skip in that case.
    return None


def parse_events_to_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in events:
        try:
            raw_away = ev.get("away_team") or ev.get("awayTeam") or (ev.get("teams") or [None, None])[0]
            raw_home = ev.get("home_team") or ev.get("homeTeam") or (ev.get("teams") or [None, None])[1]
            if not raw_away or not raw_home:
                continue
            away = _norm_team(raw_away)
            home = _norm_team(raw_home)
            start_time = ev.get("commence_time") or ev.get("commenceTime")
            if isinstance(start_time, str):
                game_time = start_time
            else:
                try:
                    # milliseconds or seconds epoch
                    ts = float(start_time)
                    if ts > 1e12: ts = ts / 1000.0
                    game_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                except Exception:
                    game_time = None
            event_desc = f"{away} @ {home}"

            bm = _choose_bookmaker(ev.get("bookmakers") or [])
            if not bm:
                continue
            book_key = str(bm.get("key") or "").strip() or "oddsapi"

            markets = bm.get("markets") or []
            for m in markets:
                mkey = str(m.get("key") or "").strip()
                std_market = MARKET_STD_MAP.get(mkey)
                if not std_market:
                    continue
                outcomes = m.get("outcomes") or []
                # Aggregate per player: line, over/under prices
                agg: Dict[str, Dict[str, Any]] = {}
                for oc in outcomes:
                    side = _side_key(str(oc.get("name") or oc.get("description") or ""))
                    # Some feeds invert fields; try to detect side from either
                    if side is None:
                        side = _side_key(str(oc.get("description") or oc.get("name") or ""))
                    player = _pick_player_from_outcome(oc)
                    if not player:
                        # As a fallback, if 'outcomes' contain a pair (Over/Under) with a 'description' for player at market level,
                        # we cannot reliably assign—skip this leg.
                        continue
                    line = oc.get("point")
                    try:
                        line_f = float(line) if line is not None and str(line) != "" else np.nan
                    except Exception:
                        line_f = np.nan
                    price = oc.get("price")
                    try:
                        amer = int(str(price).replace("+", "")) if price is not None and str(price) != "" else np.nan
                    except Exception:
                        amer = np.nan

                    rec = agg.get(player)
                    if rec is None:
                        rec = {
                            "player": player,
                            "market": std_market,
                            "line": np.nan,
                            "over_price": np.nan,
                            "under_price": np.nan,
                        }
                        agg[player] = rec
                    # Keep a representative line (prefer not-NaN)
                    if std_market != "Anytime TD":
                        if not pd.notna(rec.get("line")) and pd.notna(line_f):
                            rec["line"] = float(line_f)
                        elif pd.notna(line_f):
                            # choose the line closest to current (rarely differs between O/U)
                            cur = rec.get("line")
                            if pd.isna(cur):
                                rec["line"] = float(line_f)
                            else:
                                # keep the smaller absolute distance to median of two values
                                if abs(float(line_f) - float(cur)) < 1e-9:
                                    rec["line"] = float(line_f)
                    # Assign prices
                    if side == "over":
                        rec["over_price"] = amer
                    elif side == "under":
                        rec["under_price"] = amer
                    else:
                        # Single-sided markets: map as Over (e.g., Yes for Anytime TD)
                        rec["over_price"] = amer

                # Emit rows
                for rec in agg.values():
                    row = {
                        **rec,
                        "book": book_key,
                        "event": event_desc,
                        "game_time": game_time,
                        "home_team": home,
                        "away_team": away,
                        "is_ladder": False,
                    }
                    rows.append(row)
        except Exception:
            # Be defensive; skip problematic event
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch player props from The Odds API to CSV")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    _load_env()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ERROR: Missing ODDS_API_KEY; set in environment or .env")
        return 2
    # Masked key fingerprint for confirmation (avoid printing full secret)
    masked = f"***{api_key[-6:]}" if len(api_key) >= 6 else "(set)"
    print(f"Using OddsAPI key: {masked}")
    region = os.environ.get("ODDS_API_REGION", "us")
    try:
        events = fetch_player_props(api_key=api_key, region=region)
    except HTTPError as he:
        code = getattr(getattr(he, 'response', None), 'status_code', None)
        if code == 422:
            # Fall back to chunked market requests to salvage available markets
            print("INFO: OddsAPI 422 on combined request; retrying markets individually…")
            try:
                events = fetch_player_props_chunked(api_key=api_key, region=region)
            except Exception as e2:
                print(f"ERROR fetching OddsAPI player props (chunked) after 422: {e2}")
                return 2
        else:
            print(f"ERROR fetching OddsAPI player props: HTTP {code} {he}")
            return 2
    except Exception as e:
        print(f"ERROR fetching OddsAPI player props: {e}")
        return 2

    rows = parse_events_to_rows(events)
    if not rows:
        print("WARNING: No player prop rows parsed from OddsAPI payload.")

    df = pd.DataFrame(rows)
    # Column order compatible with props_edges_join expectations
    cols = [
        "player", "team", "market", "line", "over_price", "under_price",
        "book", "event", "game_time", "home_team", "away_team", "is_ladder",
    ]
    # Add missing optional columns
    if "team" not in df.columns:
        df["team"] = np.nan
    out_df = df[[c for c in cols if c in df.columns]].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
