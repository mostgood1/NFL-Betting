from __future__ import annotations
"""
Backtest EV-based recommendations (Moneyline, ATS, Total) over a recent window.

- Builds weekly views via app helpers, attaches predictions and totals calibration
- Generates recommendations per game using app._compute_recommendations_for_row
- Filters using RECS_MIN_EV_PCT and includes finals for grading
- Summarizes win rates, pushes, and EV stats per market and confidence tier

Outputs under nfl_compare/data/backtests/<season>_wk<end_week>/:
- recs_backtest_details.csv
- recs_backtest_summary.json
- recs_backtest_summary.md

Usage:
  python scripts/backtest_recommendations.py --season 2025 --end-week 17 --lookback 6 --min-ev 2.0 --one-per-game false
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"

# Ensure repo root is on sys.path when running from scripts/
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import app helpers lazily
from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _apply_totals_calibration_to_view,
    _compute_recommendations_for_row,
)


def _build_window(season: int, end_week: int, lookback: int) -> pd.DataFrame:
    pred = _load_predictions()
    games = _load_games()
    start_week = max(1, int(end_week) - int(lookback) + 1)
    frames: List[pd.DataFrame] = []
    for wk in range(start_week, end_week + 1):
        try:
            v = _build_week_view(pred, games, int(season), int(wk))
            v = _attach_model_predictions(v)
            try:
                v = _apply_totals_calibration_to_view(v)
            except Exception:
                pass
            if v is not None and not v.empty:
                frames.append(v)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Restrict to season and finals when available
    try:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
        m = (out["season"].eq(int(season))) & out["week"].notna() & out["week"].between(start_week, end_week)
        out = out[m].copy()
    except Exception:
        pass
    return out


def _summarize_recs(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0}
    # Standardize columns
    work = df.copy()
    work["type"] = work["type"].astype(str)
    work["confidence"] = work.get("confidence", pd.Series([None]*len(work))).astype(str)
    # Outcome flags
    res = work.get("result")
    win = res.astype(str).str.upper().eq("WIN")
    loss = res.astype(str).str.upper().eq("LOSS")
    push = res.astype(str).str.upper().eq("PUSH")
    out: Dict[str, Any] = {"rows": int(len(work))}
    for key in ("MONEYLINE", "SPREAD", "TOTAL"):
        sub = work[work["type"].astype(str).str.upper().eq(key)]
        if sub.empty:
            out[key] = {"rows": 0}
            continue
        s_win = sub["result"].astype(str).str.upper().eq("WIN").sum()
        s_loss = sub["result"].astype(str).str.upper().eq("LOSS").sum()
        s_push = sub["result"].astype(str).str.upper().eq("PUSH").sum()
        denom = max(1, (s_win + s_loss))
        out[key] = {
            "rows": int(len(sub)),
            "wins": int(s_win),
            "losses": int(s_loss),
            "pushes": int(s_push),
            "win_rate": float(s_win) / float(denom),
            "ev_mean": float(pd.to_numeric(sub.get("ev_units"), errors="coerce").mean() or 0.0),
            "ev_median": float(pd.to_numeric(sub.get("ev_units"), errors="coerce").median() or 0.0),
        }
        # Confidence-tier breakdown
        tiers = ["High", "Medium", "Low"]
        tier_stats: Dict[str, Any] = {}
        for t in tiers:
            tsub = sub[sub["confidence"].astype(str).str.upper().eq(t.upper())]
            if tsub.empty:
                tier_stats[t] = {"rows": 0}
                continue
            tw = tsub["result"].astype(str).str.upper().eq("WIN").sum()
            tl = tsub["result"].astype(str).str.upper().eq("LOSS").sum()
            td = max(1, (tw + tl))
            tier_stats[t] = {
                "rows": int(len(tsub)),
                "wins": int(tw),
                "losses": int(tl),
                "win_rate": float(tw) / float(td),
                "ev_mean": float(pd.to_numeric(tsub.get("ev_units"), errors="coerce").mean() or 0.0),
            }
        out[key]["tiers"] = tier_stats
    return out


def main():
    ap = argparse.ArgumentParser(description="Backtest EV-based recommendations across weeks")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--min-ev", type=float, default=None, help="Override RECS_MIN_EV_PCT (percent)")
    ap.add_argument("--one-per-game", type=str, default=None, help="Override RECS_ONE_PER_GAME (true/false)")
    args = ap.parse_args()

    # Optional env overrides to match production recs filtering
    if args.min_ev is not None:
        os.environ["RECS_MIN_EV_PCT"] = str(float(args.min_ev))
    if args.one_per_game is not None:
        os.environ["RECS_ONE_PER_GAME"] = str(args.one_per_game)

    season = int(args.season)
    end_week = int(args.end_week)
    lookback = max(1, int(args.lookback))
    v = _build_window(season, end_week, lookback)
    if v is None or v.empty:
        print("No window data; nothing to backtest.")
        return 0

    # Generate recommendations per game row
    rows: List[Dict[str, Any]] = []
    for _, row in v.iterrows():
        try:
            picks = _compute_recommendations_for_row(row)
        except Exception:
            picks = []
        if not picks:
            continue
        for r in picks:
            rec = dict(r)
            # Add identifiers to facilitate joins/analysis
            rec["game_id"] = row.get("game_id")
            rec["season"] = row.get("season")
            rec["week"] = row.get("week")
            rec["home_team"] = row.get("home_team")
            rec["away_team"] = row.get("away_team")
            rows.append(rec)

    if not rows:
        print("No recommendations produced from window.")
        return 0

    df = pd.DataFrame(rows)
    out_dir = DATA_DIR / "backtests" / f"{season}_wk{end_week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write details
    det_fp = out_dir / "recs_backtest_details.csv"
    try:
        df.to_csv(det_fp, index=False)
    except Exception:
        pass
    # Summarize
    summ = _summarize_recs(df)
    summ["meta"] = {"season": season, "end_week": end_week, "lookback": lookback, "rows": int(len(df))}
    try:
        with open(out_dir / "recs_backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
    except Exception:
        pass
    # Markdown summary
    lines: List[str] = []
    lines.append(f"# Recommendations Backtest (Season {season}, Weeks {max(1, end_week - lookback + 1)}–{end_week})\n")
    for key in ("MONEYLINE", "SPREAD", "TOTAL"):
        stats = summ.get(key) or {}
        lines.append(f"## {key}\n")
        lines.append(f"- Rows: {stats.get('rows', 0)}\n")
        lines.append(f"- Win Rate: {stats.get('win_rate', float('nan')):.3f}\n")
        lines.append(f"- EV Mean: {stats.get('ev_mean', float('nan')):.4f}\n")
        tiers = (stats.get("tiers") or {})
        if tiers:
            lines.append("- Tiers:")
            for t, st in tiers.items():
                lines.append(f"  - {t}: rows={st.get('rows',0)}, win_rate={(st.get('win_rate') if st.get('win_rate') is not None else float('nan')):.3f}, ev_mean={(st.get('ev_mean') if st.get('ev_mean') is not None else float('nan')):.4f}")
        lines.append("")
    try:
        (out_dir / "recs_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote recs backtest: details={det_fp}, summary={out_dir / 'recs_backtest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
