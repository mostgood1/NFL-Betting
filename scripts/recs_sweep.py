from __future__ import annotations
"""
Sweep environment tuning for EV-based recommendations and summarize performance.

- Imports helpers from scripts/backtest_recommendations to build a window and summarize
- Iterates over a small grid of env settings: min EV, one-per-game, WP/ATS/TOTAL bands, shrink
- Generates recommendations for each config and records per-market win rates and row counts

Outputs under nfl_compare/data/backtests/<season>_wk<end_week>/:
- recs_sweep.csv
- recs_sweep_best.md

Usage:
  python scripts/recs_sweep.py --season 2025 --end-week 17 --lookback 6
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"

import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.backtest_recommendations import _build_window, _summarize_recs
from app import _compute_recommendations_for_row


def _run_config(v: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # Apply envs
    for k, val in cfg.items():
        if val is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = str(val)
    # Generate recs
    rows: List[Dict[str, Any]] = []
    for _, row in v.iterrows():
        try:
            picks = _compute_recommendations_for_row(row)
        except Exception:
            picks = []
        for r in picks:
            rec = dict(r)
            rec["game_id"] = row.get("game_id")
            rec["season"] = row.get("season")
            rec["week"] = row.get("week")
            rec["home_team"] = row.get("home_team")
            rec["away_team"] = row.get("away_team")
            rows.append(rec)
    df = pd.DataFrame(rows)
    summ = _summarize_recs(df)
    return summ, df


def main():
    ap = argparse.ArgumentParser(description="Sweep recs tuning and summarize performance")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=6)
    args = ap.parse_args()

    season = int(args.season)
    end_week = int(args.end_week)
    lookback = max(1, int(args.lookback))

    v = _build_window(season, end_week, lookback)
    if v is None or v.empty:
        print("No window data; sweep aborted.")
        return 0

    # Config grid (keep modest for runtime)
    min_ev_opts = [2.0, 5.0]
    one_per_game_opts = ["false", "true"]
    wp_band_opts = [0.12, 0.0]
    ats_band_opts = [0.10, 0.0]
    total_band_opts = [0.10, 0.0]
    shrink_opts = [0.35]
    # Per-market gates (new): deltas from 0.5 and EV percent thresholds
    wp_delta_opts = [0.08, 0.12]
    ats_delta_opts = [0.10, 0.15]
    total_delta_opts = [0.10, 0.15]
    ev_ml_opts = [4.0, 5.0]
    ev_ats_opts = [4.0, 6.0]
    ev_total_opts = [4.0, 6.0]

    configs: List[Dict[str, Any]] = []
    for min_ev in min_ev_opts:
        for opg in one_per_game_opts:
            for wpb in wp_band_opts:
                for atsb in ats_band_opts:
                    for tb in total_band_opts:
                        for shr in shrink_opts:
                            for wpd in wp_delta_opts:
                                for atsd in ats_delta_opts:
                                    for td in total_delta_opts:
                                        for evml in ev_ml_opts:
                                            for evats in ev_ats_opts:
                                                for evtot in ev_total_opts:
                                                    cfg = {
                                                        "RECS_MIN_EV_PCT": min_ev,
                                                        "RECS_ONE_PER_GAME": opg,
                                                        "RECS_WP_MARKET_BAND": wpb,
                                                        "RECS_ATS_BAND": atsb,
                                                        "RECS_TOTAL_BAND": tb,
                                                        "RECS_PROB_SHRINK": shr,
                                                        # New per-market gates
                                                        "RECS_MIN_WP_DELTA": wpd,
                                                        "RECS_MIN_ATS_DELTA": atsd,
                                                        "RECS_MIN_TOTAL_DELTA": td,
                                                        "RECS_MIN_EV_PCT_ML": evml,
                                                        "RECS_MIN_EV_PCT_ATS": evats,
                                                        "RECS_MIN_EV_PCT_TOTAL": evtot,
                                                        "RECS_ALLOWED_MARKETS": "MONEYLINE,SPREAD,TOTAL",
                                                    }
                                                    configs.append(cfg)

    rows: List[Dict[str, Any]] = []
    out_dir = DATA_DIR / "backtests" / f"{season}_wk{end_week}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        summ, df = _run_config(v, cfg)
        # Flatten summary
        rec: Dict[str, Any] = {
            "season": season,
            "end_week": end_week,
            "lookback": lookback,
            **cfg,
            "rows": summ.get("rows", 0),
            "ml_rows": (summ.get("MONEYLINE") or {}).get("rows", 0),
            "ml_win_rate": (summ.get("MONEYLINE") or {}).get("win_rate", float("nan")),
            "sp_rows": (summ.get("SPREAD") or {}).get("rows", 0),
            "sp_win_rate": (summ.get("SPREAD") or {}).get("win_rate", float("nan")),
            "tot_rows": (summ.get("TOTAL") or {}).get("rows", 0),
            "tot_win_rate": (summ.get("TOTAL") or {}).get("win_rate", float("nan")),
        }
        rows.append(rec)
        # Write details per config (optional)
        tag = (
            f"min{cfg['RECS_MIN_EV_PCT']}_opg{cfg['RECS_ONE_PER_GAME']}"
            f"_wpb{cfg['RECS_WP_MARKET_BAND']}_atsb{cfg['RECS_ATS_BAND']}_tb{cfg['RECS_TOTAL_BAND']}"
            f"_shr{cfg['RECS_PROB_SHRINK']}_wpd{cfg['RECS_MIN_WP_DELTA']}_atsd{cfg['RECS_MIN_ATS_DELTA']}"
            f"_td{cfg['RECS_MIN_TOTAL_DELTA']}_evml{cfg['RECS_MIN_EV_PCT_ML']}_evats{cfg['RECS_MIN_EV_PCT_ATS']}"
            f"_evtot{cfg['RECS_MIN_EV_PCT_TOTAL']}"
        )
        try:
            df.to_csv(out_dir / f"recs_details_{tag}.csv", index=False)
        except Exception:
            pass
        print(f"[{i}/{len(configs)}] rows={len(df)} tag={tag} ml={rec['ml_win_rate']:.3f} sp={rec['sp_win_rate']:.3f} tot={rec['tot_win_rate']:.3f}")

    if not rows:
        print("Sweep produced no rows.")
        return 0

    sweep_df = pd.DataFrame(rows)
    sweep_fp = out_dir / "recs_sweep.csv"
    try:
        sweep_df.to_csv(sweep_fp, index=False)
    except Exception:
        pass

    # Pick best configs by SPREAD and TOTAL win rates (break ties by rows)
    def _pick_best(col_rate: str, col_rows: str) -> Tuple[pd.Series, str]:
        df2 = sweep_df.copy()
        df2 = df2.sort_values([col_rate, col_rows], ascending=[False, False])
        best = df2.iloc[0]
        return best, col_rate

    best_sp, _ = _pick_best("sp_win_rate", "sp_rows")
    best_tot, _ = _pick_best("tot_win_rate", "tot_rows")

    lines: List[str] = []
    lines.append(f"# Recs Sweep Best (Season {season}, Weeks {max(1, end_week - lookback + 1)}–{end_week})\n")
    lines.append("## Best by Spread Win Rate\n")
    for k in ["RECS_MIN_EV_PCT","RECS_ONE_PER_GAME","RECS_WP_MARKET_BAND","RECS_ATS_BAND","RECS_TOTAL_BAND","RECS_PROB_SHRINK","sp_rows","sp_win_rate"]:
        lines.append(f"- {k}: {best_sp.get(k)}")
    lines.append("")
    lines.append("## Best by Total Win Rate\n")
    for k in ["RECS_MIN_EV_PCT","RECS_ONE_PER_GAME","RECS_WP_MARKET_BAND","RECS_ATS_BAND","RECS_TOTAL_BAND","RECS_PROB_SHRINK","tot_rows","tot_win_rate"]:
        lines.append(f"- {k}: {best_tot.get(k)}")
    try:
        (out_dir / "recs_sweep_best.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote sweep: {sweep_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
