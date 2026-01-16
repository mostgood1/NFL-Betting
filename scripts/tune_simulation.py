from __future__ import annotations

"""
Sweep a small grid of simulation parameters and backtest over weeks 1–18.

Optimizes a weighted score built from Brier (ATS/Total) and MAE (Total) to
select a parameter set that improves realism and predictive performance.

Outputs:
- sim_tuning_summary.csv: per-config metrics and score
- best_config.json: chosen parameters and metrics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from scripts.backtest_simulation import backtest_sim
except ImportError:
    from .backtest_simulation import backtest_sim


def _score(metrics: Dict[str, float]) -> float:
    # Weighted score: lower is better
    b_ats = float(metrics.get("brier_ats", float("nan")))
    b_tot = float(metrics.get("brier_total", float("nan")))
    mae_tot = float(metrics.get("mae_total", float("nan")))
    # Normalize MAE Total to roughly match Brier scale
    mae_norm = mae_tot / 14.0 if pd.notna(mae_tot) else float("nan")
    return 0.4 * b_ats + 0.4 * b_tot + 0.2 * mae_norm


def _set_env(params: Dict[str, str | float]) -> None:
    for k, v in params.items():
        os.environ[str(k)] = str(v)


def main():
    out_dir = Path("nfl_compare/data/backtests/2025_wk18")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expanded grid including feature coefficients; keep size modest to avoid long runtimes
    grid: List[Dict[str, float]] = []
    for blend_m in [0.0, 0.10]:
        for blend_t in [0.10, 0.20]:
            for ats_sigma in [11.0, 12.0]:
                for total_sigma in [11.0, 12.0]:
                    for k_wind in [-0.8, -0.5, -0.2]:  # per 10mph
                        for k_rest in [0.10, 0.18, 0.25]:  # per 7-day diff
                            for k_elo in [0.05, 0.08, 0.12]:  # per 100 Elo
                                for k_inj in [-0.6, -0.4, -0.2]:  # per starter out
                                    for k_def in [-0.6, -0.4, -0.2]:  # per 5 PPG
                                        for k_press in [-1.0, -0.8, -0.4]:  # per 0.05 above baseline
                                            for press_base in [0.055, 0.065, 0.075]:
                                                for k_rate in [0.05, 0.08, 0.12]:  # per 1.0 rating diff
                                                    for rho in [-0.05, 0.10, 0.20]:
                                                        grid.append({
                                                            "SIM_MARKET_BLEND_MARGIN": blend_m,
                                                            "SIM_MARKET_BLEND_TOTAL": blend_t,
                                                            # Base sigma bounds (per-game scaling uses these)
                                                            "SIM_ATS_SIGMA_MIN": ats_sigma - 4.0,
                                                            "SIM_ATS_SIGMA_MAX": ats_sigma + 8.0,
                                                            "SIM_TOTAL_SIGMA_MIN": total_sigma - 5.0,
                                                            "SIM_TOTAL_SIGMA_MAX": total_sigma + 8.0,
                                                            # Feature mean knobs
                                                            "SIM_MEAN_TOTAL_K_WIND": k_wind,
                                                            "SIM_MEAN_MARGIN_K_REST": k_rest,
                                                            "SIM_MEAN_MARGIN_K_ELO": k_elo,
                                                            "SIM_MEAN_TOTAL_K_INJ": k_inj,
                                                            "SIM_MEAN_TOTAL_K_DEFPPG": k_def,
                                                            "SIM_MEAN_TOTAL_K_PRESSURE": k_press,
                                                            "SIM_PRESSURE_BASELINE": press_base,
                                                            "SIM_MEAN_MARGIN_K_RATING": k_rate,
                                                            "SIM_CORR_MARGIN_TOTAL": rho,
                                                            # Realism clamps
                                                            "SIM_TOTAL_DELTA_MAX": 10.0,
                                                            "SIM_TOTAL_MEAN_MIN": 30.0,
                                                            "SIM_TOTAL_MEAN_MAX": 62.0,
                                                            "SIM_MARGIN_DELTA_MAX": 10.0,
                                                            "SIM_MARGIN_MEAN_ABS_MAX": 20.0,
                                                        })

    rows: List[Dict[str, float | int | str]] = []
    best: Tuple[float, Dict[str, float], Dict[str, float]] | None = None
    # Attempt to use cached predictions if present in out_dir
    pred_cache_fp = None
    try:
        std_fp = out_dir / "games_details.csv"
        if std_fp.exists():
            pred_cache_fp = str(std_fp)
    except Exception:
        pred_cache_fp = None

    for params in grid:
        _set_env(params)
        metrics = backtest_sim(season=2025, start_week=1, end_week=18, n_sims=1200, out_dir=out_dir, pred_cache_fp=pred_cache_fp)
        s = _score(metrics)
        row = {**{k: float(v) for k, v in params.items()}, **metrics, "score": float(s)}
        rows.append(row)
        if best is None or s < best[0]:
            best = (s, params, metrics)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sim_tuning_summary.csv", index=False)
    if best is not None:
        with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump({"score": best[0], "params": best[1], "metrics": best[2]}, f, indent=2)
        print(f"Best score: {best[0]:.4f}")
        print("Best params:")
        for k, v in best[1].items():
            print(f"  {k}={v}")
        print("Metrics:")
        for k, v in best[2].items():
            print(f"  {k}={v}")
    else:
        print("No tuning results produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
