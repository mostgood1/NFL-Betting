from __future__ import annotations

"""
Update OpenWeather snapshots for September game dates only.
- Detects game dates in September from data/games.csv
- Limits to dates on/after today
- Backs up existing weather_YYYY-MM-DD.csv to .backup.csv before overwriting

Usage (from repo root or nfl_compare):
  python -m nfl_compare.src.update_weather_september
"""

from datetime import date as _date
from pathlib import Path
import shutil
import pandas as pd

from .weather import DATA_DIR
from .auto_update import update_weather_for_date


def _september_dates() -> list[str]:
    gfp = DATA_DIR / 'games.csv'
    if not gfp.exists():
        print('games.csv not found; aborting weather update.')
        return []
    games = pd.read_csv(gfp)
    games['date'] = pd.to_datetime(games.get('date'), errors='coerce')
    today = _date.today()
    # Only September (month == 9) and on/after today
    dates = sorted({
        d.date().isoformat()
        for d in games['date'].dropna()
        if d.month == 9 and d.date() >= today
    })
    return dates


def main() -> None:
    dates = _september_dates()
    if not dates:
        print('No September dates to update.')
        return

    total_rows = 0
    for ds in dates:
        # Backup existing file if present
        fp = DATA_DIR / f"weather_{ds}.csv"
        if fp.exists():
            backup = DATA_DIR / f"weather_{ds}.backup.csv"
            try:
                shutil.copy2(fp, backup)
                print(f"Backed up existing {fp.name} -> {backup.name}")
            except Exception as e:
                print(f"Backup failed for {fp.name}: {e}")

        # Update snapshot for this date
        try:
            cnt = update_weather_for_date(ds)
            total_rows += int(cnt or 0)
        except Exception as e:
            print(f"Weather update failed for {ds}: {e}")

    print(f"September weather update complete: {total_rows} rows across {len(dates)} date(s): {dates}")


if __name__ == '__main__':
    main()
