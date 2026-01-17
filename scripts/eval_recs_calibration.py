"""Evaluate calibration of backtest recommendation probabilities.

Reads a `recs_backtest_details.csv` produced by `scripts/backtest_recommendations.py` and computes:
- Brier score and log loss for `prob_selected` vs realized outcome (Win=1, Loss=0; Push excluded)
- A simple reliability table by probability bins
- A market-implied baseline using the bet's odds (implied probability = 1/decimal)

Usage:
  python scripts/eval_recs_calibration.py --details nfl_compare/data/backtests/.../recs_backtest_details.csv
  python scripts/eval_recs_calibration.py --details A.csv --details B.csv

Notes:
- For spread/total, odds are usually -110; baseline implied will be ~0.524.
- For moneyline, odds vary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _american_to_decimal(american: float) -> float:
    if american is None or pd.isna(american) or float(american) == 0.0:
        return float("nan")
    a = float(american)
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def _odds_to_decimal(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    # Heuristic: treat values in [1.01, 15] as decimal; others as American
    is_decimal = o.between(1.01, 15.0)
    dec = pd.Series(np.nan, index=o.index, dtype="float")
    dec.loc[is_decimal] = o.loc[is_decimal].astype(float)
    for idx, val in o.loc[~is_decimal].items():
        dec.loc[idx] = _american_to_decimal(val)
    return pd.to_numeric(dec, errors="coerce")


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "type" not in out.columns:
        raise ValueError("details CSV missing 'type' column")

    out["market"] = out["type"].astype(str).str.upper().str.strip()
    out["result_u"] = out.get("result", pd.Series([None] * len(out))).astype(str).str.upper().str.strip()

    # Outcome: WIN=1, LOSS=0; push/other excluded
    out["y"] = np.where(out["result_u"].eq("WIN"), 1.0, np.where(out["result_u"].eq("LOSS"), 0.0, np.nan))

    out["p"] = pd.to_numeric(out.get("prob_selected"), errors="coerce")
    out["dec_odds"] = _odds_to_decimal(out.get("odds", pd.Series([np.nan] * len(out))))
    out["p_implied"] = 1.0 / out["dec_odds"]
    return out


def _reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # include right edge in last bin
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, bins - 1)

    rows: List[Dict[str, float]] = []
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": int(np.sum(m)),
                "p_mean": float(np.mean(p[m])),
                "win_rate": float(np.mean(y[m])),
                "gap": float(np.mean(p[m]) - np.mean(y[m])),
            }
        )
    return pd.DataFrame(rows)


def _summarize_market(df: pd.DataFrame, market: str) -> Dict[str, object]:
    sub = df[df["market"].eq(market)].copy()
    sub = sub[sub["y"].notna() & sub["p"].notna()]
    if sub.empty:
        return {"market": market, "n": 0}

    y = sub["y"].to_numpy(dtype=float)
    p = sub["p"].to_numpy(dtype=float)

    # baseline: market implied from odds
    p0 = pd.to_numeric(sub.get("p_implied"), errors="coerce").to_numpy(dtype=float)
    # For any missing implied values, fallback to -110 break-even (1/1.909..)
    p0 = np.where(np.isfinite(p0), p0, 1.0 / _american_to_decimal(-110.0))

    out: Dict[str, object] = {
        "market": market,
        "n": int(len(sub)),
        "win_rate": float(np.mean(y)),
        "p_mean": float(np.mean(p)),
        "brier": _brier(y, p),
        "logloss": _logloss(y, p),
        "brier_implied": _brier(y, p0),
        "logloss_implied": _logloss(y, p0),
    }
    out["reliability"] = _reliability_table(y, p, bins=10)
    return out


def _print_report(path: Path, df: pd.DataFrame) -> None:
    print(f"\n=== {path} ===")
    print(f"rows={len(df)} cols={len(df.columns)}")

    markets = ["MONEYLINE", "SPREAD", "TOTAL"]
    work = _prep(df)

    for m in markets:
        s = _summarize_market(work, m)
        if int(s.get("n", 0)) == 0:
            print(f"\n{m}: n=0")
            continue
        print(
            f"\n{m}: n={s['n']} win_rate={s['win_rate']:.3f} p_mean={s['p_mean']:.3f} "
            f"brier={s['brier']:.4f} (implied {s['brier_implied']:.4f}) "
            f"logloss={s['logloss']:.4f} (implied {s['logloss_implied']:.4f})"
        )
        rel = s.get("reliability")
        if isinstance(rel, pd.DataFrame) and not rel.empty:
            # Print a compact reliability table
            show = rel.copy()
            show["bin"] = show.apply(lambda r: f"[{r['bin_lo']:.1f},{r['bin_hi']:.1f})", axis=1)
            show = show[["bin", "n", "p_mean", "win_rate", "gap"]]
            print(show.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="append", required=True, help="Path to recs_backtest_details.csv")
    args = ap.parse_args()

    for fp in args.details:
        path = Path(fp)
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        df = pd.read_csv(path)
        _print_report(path, df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
