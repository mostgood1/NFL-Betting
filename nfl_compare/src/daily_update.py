from __future__ import annotations

"""
Daily update runner:
- Loads .env
- Fetches real NFL odds (moneyline/spreads/totals) and writes today's JSON
- Updates weather snapshots for all future game dates
- Re-runs predictions to incorporate fresh odds + weather

Usage (from nfl_compare):
  python -m src.daily_update
"""

from pathlib import Path

from .config import load_env
from .odds_api_client import main as fetch_odds
from .auto_update import main as update_weather_and_predict


def main() -> None:
    load_env()
    # 1) Fetch odds first so predictions include latest markets
    try:
        fetch_odds()
    except Exception as e:
        print(f"Odds update failed: {e}")

    # 2) Update weather and re-run predictions (auto_update calls predict at end)
    try:
        update_weather_and_predict()
    except Exception as e:
        print(f"Weather/predict failed: {e}")

    out_fp = Path(__file__).resolve().parents[1] / 'data' / 'predictions.csv'
    if out_fp.exists():
        print(f"Daily update complete. Predictions at {out_fp}")
    else:
        print("Daily update complete, but predictions file was not found.")


if __name__ == "__main__":
    main()
