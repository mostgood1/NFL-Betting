"""
Quick OddsAPI health check per market.

- Loads ODDS_API_KEY from .env via nfl_compare.src.config.load_env
- Probes each market individually to identify 422s or empty responses
- Prints a compact summary with counts

Usage:
  python scripts/oddsapi_markets_health.py
  python scripts/oddsapi_markets_health.py --markets player_pass_yds,player_rush_yds
"""
from __future__ import annotations

import argparse
import os
from typing import List

from requests.exceptions import HTTPError

# Ensure repo root is on sys.path for local package imports
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfl_compare.src.config import load_env as _load_env

# Reuse helpers from the fetch script so market keys stay in one place
from scripts.fetch_oddsapi_props import (
    _player_markets,
    fetch_player_props,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="OddsAPI per-market health check")
    ap.add_argument("--markets", type=str, default=None, help="Comma-separated OddsAPI market keys to probe")
    ap.add_argument("--region", type=str, default="us", help="OddsAPI region (default: us)")
    args = ap.parse_args()

    _load_env()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ERROR: Missing ODDS_API_KEY; set it in .env or environment.")
        return 2
    masked = f"***{api_key[-6:]}" if len(api_key) >= 6 else "(set)"
    print(f"Using OddsAPI key: {masked}")

    if args.markets:
        markets: List[str] = [m.strip() for m in args.markets.split(",") if m.strip()]
    else:
        markets = _player_markets()

    ok = 0
    bad = 0
    for m in markets:
        try:
            events = fetch_player_props(api_key=api_key, region=args.region, markets=[m])
            cnt = len(events) if events else 0
            status = "OK" if cnt > 0 else "EMPTY"
            print(f"- {m:22s} -> {status:6s} events={cnt}")
            ok += 1
        except HTTPError as he:
            code = getattr(getattr(he, 'response', None), 'status_code', None)
            print(f"- {m:22s} -> HTTP {code}")
            bad += 1
        except Exception as e:
            print(f"- {m:22s} -> ERROR {e}")
            bad += 1

    print(f"Summary: {ok} probed, {bad} with errors or HTTP failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
