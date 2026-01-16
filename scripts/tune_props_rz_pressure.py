from __future__ import annotations

"""
Sweep tuning for props red-zone/pressure knobs using the existing props backtest.

Knobs:
- LEAGUE_RZ_PASS_RATE (baseline)
- PROPS_RZ_SHARE_W (tilt WR/TE vs RB target shares)
- PROPS_RZ_TD_W (tilt team/QB pass TDs)
- PROPS_PRESSURE_YPT_W (pressure effect on YPT)

Outputs:
- CSV grid with composite score and individual MAEs per position/metric
- Console summary with best configuration

Usage:
  python scripts/tune_props_rz_pressure.py --season 2025 --start-week 1 --end-week 12 \
    --out nfl_compare/data/tuning/props_rz_pressure_sweep.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from scripts.backtest_props import backtest as props_backtest


def _composite_score(summary: pd.DataFrame) -> float:
    """Lower is better. Focus on receiving and QB TDs primarily.

    Components (averaged across positions present):
    - rec_yards_MAE (WR, TE, RB)
    - rec_tds_MAE (WR, TE, RB)
    - pass_tds_MAE (QB)
    - pass_yards_MAE (QB) small weight to avoid YPT over-suppression
    """
    if summary is None or summary.empty:
        return float("inf")
    df = summary.copy()
    df['position'] = df['position'].astype(str)
    # Helper to get average across listed positions ignoring missing
    def avg(col: str, positions: Iterable[str]) -> float:
        pts = pd.to_numeric(df[df['position'].isin(list(positions))][col], errors='coerce').dropna()
        return float(pts.mean()) if not pts.empty else np.nan

    rec_yards = avg('rec_yards_MAE', ['WR','TE','RB'])
    rec_tds = avg('rec_tds_MAE', ['WR','TE','RB'])
    qb_ptd = avg('pass_tds_MAE', ['QB'])
    qb_py = avg('pass_yards_MAE', ['QB'])

    # Weights: emphasize rec_yards and rec_tds; TDs important; yards modest
    parts: List[float] = []
    if np.isfinite(rec_yards): parts.append(1.0 * rec_yards)
    if np.isfinite(rec_tds): parts.append(1.0 * rec_tds)
    if np.isfinite(qb_ptd): parts.append(0.8 * qb_ptd)
    if np.isfinite(qb_py): parts.append(0.2 * qb_py)
    if not parts:
        return float("inf")
    return float(np.mean(parts))


def run_sweep(
    season: int,
    start_week: int,
    end_week: int,
    rz_share_vals: List[float],
    rz_td_vals: List[float],
    pressure_vals: List[float],
    base_rz_vals: List[float],
) -> pd.DataFrame:
    rows: List[Dict] = []
    total = len(rz_share_vals) * len(rz_td_vals) * len(pressure_vals) * len(base_rz_vals)
    i = 0
    for base_rz in base_rz_vals:
        os.environ['LEAGUE_RZ_PASS_RATE'] = str(base_rz)
        for rz_share in rz_share_vals:
            os.environ['PROPS_RZ_SHARE_W'] = str(rz_share)
            for rz_td in rz_td_vals:
                os.environ['PROPS_RZ_TD_W'] = str(rz_td)
                for press_w in pressure_vals:
                    os.environ['PROPS_PRESSURE_YPT_W'] = str(press_w)
                    i += 1
                    print(f"[{i}/{total}] RZ_BASE={base_rz} RZ_SHARE_W={rz_share} RZ_TD_W={rz_td} PRESS_YPT_W={press_w}")
                    summ_df, _weekly_df = props_backtest(int(season), int(start_week), int(end_week))
                    score = _composite_score(summ_df)
                    rec: Dict = {
                        'LEAGUE_RZ_PASS_RATE': base_rz,
                        'PROPS_RZ_SHARE_W': rz_share,
                        'PROPS_RZ_TD_W': rz_td,
                        'PROPS_PRESSURE_YPT_W': press_w,
                        'score': score,
                    }
                    # Attach selected MAEs for visibility
                    if summ_df is not None and not summ_df.empty:
                        for pos in ['QB','RB','WR','TE']:
                            sub = summ_df[summ_df['position'] == pos]
                            if not sub.empty:
                                for col in [
                                    'pass_yards_MAE','pass_tds_MAE','interceptions_MAE',
                                    'rush_yards_MAE','rush_tds_MAE',
                                    'rec_yards_MAE','rec_tds_MAE','receptions_MAE'
                                ]:
                                    if col in sub.columns:
                                        rec[f"{pos}_{col}"] = float(pd.to_numeric(sub[col], errors='coerce').iloc[0])
                    rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Tune props red-zone/pressure knobs via grid sweep")
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--start-week', type=int, default=1)
    ap.add_argument('--end-week', type=int, required=True)
    ap.add_argument('--out', type=str, default='nfl_compare/data/tuning/props_rz_pressure_sweep.csv')
    # Grid params (comma-separated floats)
    ap.add_argument('--grid-rz-share', type=str, default='0.10,0.20,0.30')
    ap.add_argument('--grid-rz-td', type=str, default='0.20,0.30,0.40')
    ap.add_argument('--grid-press', type=str, default='0.08,0.12,0.16')
    ap.add_argument('--grid-base-rz', type=str, default='0.50,0.52,0.55')
    args = ap.parse_args()

    def parse_list(s: str) -> List[float]:
        vals: List[float] = []
        for p in str(s).split(','):
            p = p.strip()
            if not p:
                continue
            try:
                vals.append(float(p))
            except Exception:
                pass
        return vals

    df = run_sweep(
        season=args.season,
        start_week=args.start_week,
        end_week=args.end_week,
        rz_share_vals=parse_list(args.grid_rz_share),
        rz_td_vals=parse_list(args.grid_rz_td),
        pressure_vals=parse_list(args.grid_press),
        base_rz_vals=parse_list(args.grid_base_rz),
    )
    if df is None or df.empty:
        print("No results produced")
        return 1
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values('score', ascending=True)
    df.to_csv(out_fp, index=False)
    print(f"Wrote sweep results -> {out_fp}")
    best = df.iloc[0]
    print("\nBest configuration:")
    print(best.to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
