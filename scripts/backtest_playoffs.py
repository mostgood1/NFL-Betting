from __future__ import annotations

"""
Playoff-focused backtest wrapper for winners/ATS/totals and player props.

Runs game-level and props backtests across playoff weeks (>=19) for a season,
producing standardized outputs under nfl_compare/data/backtests/<season>_wk<end_week>/.

Usage:
  python scripts/backtest_playoffs.py --season 2025 --end-week 22 \
    [--blend-margin 0.10] [--blend-total 0.20]

Notes:
  - Requires playoff schedule rows present in games.csv.
  - Props backtest requires reconciliation data availability for the selected weeks.
"""

import argparse
from pathlib import Path
import pandas as pd
import json
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR")) if os.environ.get("NFL_DATA_DIR") else (BASE_DIR / "nfl_compare" / "data")


def _run_games(season: int, start_week: int, end_week: int, blend_margin: float, blend_total: float) -> Path | None:
    from scripts.backtest_games import backtest as backtest_games
    summ, details = backtest_games(season, start_week, end_week, include_same=True, blend_margin=blend_margin, blend_total=blend_total)
    if summ is None or summ.empty:
        print("[playoffs] No game results; skipping write.")
        return None
    out_dir = DATA_DIR / "backtests" / f"{season}_wk{end_week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_fp = out_dir / "games_summary.csv"
    summ.to_csv(summ_fp, index=False)
    print(f"[playoffs] Wrote {summ_fp}")
    if details is not None and not details.empty:
        det_fp = out_dir / "games_details.csv"
        details.to_csv(det_fp, index=False)
        print(f"[playoffs] Wrote {det_fp}")
    # Update metrics.json
    metrics_fp = out_dir / "metrics.json"
    metrics = {}
    if metrics_fp.exists():
        try:
            metrics = json.loads(metrics_fp.read_text(encoding='utf-8'))
        except Exception:
            metrics = {}
    # Flatten single-row metrics
    try:
        row = summ.iloc[0].to_dict()
        metrics['games'] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}
    except Exception:
        pass
    with open(metrics_fp, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return out_dir


def _run_props(season: int, start_week: int, end_week: int) -> Path | None:
    from scripts.backtest_props import backtest as backtest_props
    summ, weekly = backtest_props(season, start_week, end_week)
    if summ is None or summ.empty:
        print("[playoffs] No props results; skipping write.")
        return None
    out_dir = DATA_DIR / "backtests" / f"{season}_wk{end_week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_fp = out_dir / "props_summary.csv"
    summ.to_csv(summ_fp, index=False)
    print(f"[playoffs] Wrote {summ_fp}")
    if weekly is not None and not weekly.empty:
        wk_fp = out_dir / "props_weekly.csv"
        weekly.to_csv(wk_fp, index=False)
        print(f"[playoffs] Wrote {wk_fp}")
    # Update metrics.json
    metrics_fp = out_dir / "metrics.json"
    metrics = {}
    if metrics_fp.exists():
        try:
            metrics = json.loads(metrics_fp.read_text(encoding='utf-8'))
        except Exception:
            metrics = {}
    try:
        metrics['props'] = summ.mean(numeric_only=True).to_dict()
    except Exception:
        pass
    with open(metrics_fp, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Playoff backtests for games and props")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=19)
    ap.add_argument("--blend-margin", type=float, default=0.10)
    ap.add_argument("--blend-total", type=float, default=0.20)
    args = ap.parse_args(argv)

    s = int(args.season)
    w0 = int(args.start_week)
    w1 = int(args.end_week)
    out_g = _run_games(s, w0, w1, float(args.blend_margin), float(args.blend_total))
    out_p = _run_props(s, w0, w1)
    if out_g is None and out_p is None:
        print("No playoff results produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
