"""Fit probability calibration curves from *all games* in a built week view.

This avoids selection bias from using only recommended bets.

We:
- Build a window DataFrame (like backtests) with either baseline predictions or walk-forward predictions.
- For final games, compute:
  - ATS: p_home_cover from model margin vs spread
  - TOTAL: p_over from model total vs market total
- Fit bin-based calibration curves (piecewise linear) for:
  - 'ats' (home cover)
  - 'total' (over)

Output format is compatible with `app._apply_prob_calibration()`.

Usage:
  python scripts/fit_prob_calibration_from_view.py --season 2025 --end-week 17 --lookback 17 --walk-forward --line-mode open \
    --out nfl_compare/data/prob_calibration_walkfwd_view_2025.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import _cover_prob_from_edge, _load_sigma_calibration

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore[assignment]


def _coalesce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    out = None
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if out is None:
            out = s
        else:
            out = out.where(out.notna(), s)
    if out is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float")
    return pd.to_numeric(out, errors="coerce")


def _fit_bin_linear(p: pd.Series, y: pd.Series, bins: int) -> Optional[Dict[str, Any]]:
    p = pd.to_numeric(p, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = p.notna() & y.notna()
    n = int(m.sum())
    # We often only have ~90-110 final games in a typical window, so allow
    # fitting with smaller samples; qcut will drop bins if needed.
    if n < max(50, bins * 6):
        return None

    pp = p.loc[m].clip(0.0, 1.0)
    yy = y.loc[m].clip(0.0, 1.0)

    try:
        cats = pd.qcut(pp, q=bins, duplicates="drop")
    except Exception:
        return None

    g = pd.DataFrame({"p": pp, "y": yy, "bin": cats}).groupby("bin", observed=True)
    centers = g["p"].mean()
    rates = g["y"].mean()

    xs = [0.0] + [float(v) for v in centers.tolist()] + [1.0]
    ys = [0.0] + [float(v) for v in rates.tolist()] + [1.0]

    # Sort and enforce weak monotonicity (expected by app)
    pairs = sorted(zip(xs, ys), key=lambda t: float(t[0]))
    xs_s: List[float] = []
    ys_s: List[float] = []
    for x, yv in pairs:
        xs_s.append(float(max(0.0, min(1.0, x))))
        ys_s.append(float(max(0.0, min(1.0, yv))))
    for i in range(1, len(ys_s)):
        if ys_s[i] < ys_s[i - 1]:
            ys_s[i] = ys_s[i - 1]

    return {"xs": xs_s, "ys": ys_s, "method": "bin-linear", "n_bins": int(len(centers)), "n": int(n)}


def _fit_isotonic(p: pd.Series, y: pd.Series, points: int) -> Optional[Dict[str, Any]]:
    if IsotonicRegression is None:
        return None

    p = pd.to_numeric(p, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = p.notna() & y.notna()
    n = int(m.sum())
    if n < max(80, points * 3):
        return None

    pp = p.loc[m].clip(0.0, 1.0).to_numpy(dtype=float)
    yy = y.loc[m].clip(0.0, 1.0).to_numpy(dtype=float)

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(pp, yy)

    # Use quantile-spaced x-grid to represent the curve compactly.
    q = np.linspace(0.0, 1.0, int(max(5, points)))
    xs = np.quantile(pp, q)
    xs[0] = 0.0
    xs[-1] = 1.0
    xs = np.unique(np.clip(xs, 0.0, 1.0))
    ys = np.clip(ir.predict(xs), 0.0, 1.0)

    return {"xs": [float(x) for x in xs.tolist()], "ys": [float(v) for v in ys.tolist()], "method": "isotonic", "n": int(n), "n_points": int(len(xs))}


def _load_build_window_func() -> Any:
    mod_path = Path("scripts/backtest_recommendations.py").resolve()
    spec = importlib.util.spec_from_file_location("btr", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod._build_window


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit prob calibration curves from full week view")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=17)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--line-mode", choices=["auto", "open", "close"], default="open")
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--method", choices=["isotonic", "bin-linear"], default="isotonic")
    ap.add_argument("--points", type=int, default=25, help="Number of curve points to store when using isotonic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    build_window = _load_build_window_func()
    v: pd.DataFrame = build_window(int(args.season), int(args.end_week), int(args.lookback), compute_mc=False, walk_forward=bool(args.walk_forward))
    if v is None or v.empty:
        raise SystemExit("No window data")

    # Finals only
    hs = pd.to_numeric(v.get("home_score"), errors="coerce")
    aas = pd.to_numeric(v.get("away_score"), errors="coerce")
    is_final = hs.notna() & aas.notna()
    work = v[is_final].copy()
    if work.empty:
        raise SystemExit("No final games in window")

    # Sigma + shrink
    sig = _load_sigma_calibration() or {}
    ats_sigma = float(sig.get("ats_sigma") or 9.0)
    tot_sigma = float(sig.get("total_sigma") or 10.0)
    try:
        shrink = float((__import__("os").environ.get("RECS_PROB_SHRINK", "0.35")))
    except Exception:
        shrink = 0.35

    # Line selection matching _compute_recommendations_for_row
    lm = str(args.line_mode).strip().lower()
    if lm == "close":
        spread = _coalesce_numeric(work, ["close_spread_home", "market_spread_home", "spread_home", "open_spread_home"])
        total_line = _coalesce_numeric(work, ["close_total", "market_total", "total", "open_total"])
    elif lm == "open":
        spread = _coalesce_numeric(work, ["open_spread_home", "spread_home", "market_spread_home", "close_spread_home"])
        total_line = _coalesce_numeric(work, ["open_total", "total", "market_total", "close_total"])
    else:
        spread = _coalesce_numeric(work, ["market_spread_home", "spread_home", "open_spread_home", "close_spread_home"])
        total_line = _coalesce_numeric(work, ["market_total", "total", "open_total", "close_total"])

    # Predictions
    ph = _coalesce_numeric(work, ["pred_home_points", "pred_home_score"])
    pa = _coalesce_numeric(work, ["pred_away_points", "pred_away_score"])
    margin = ph - pa

    total_pred = _coalesce_numeric(work, ["pred_total_cal", "pred_total"])

    # Targets (exclude pushes)
    actual_margin = hs - aas
    cover_val = actual_margin + spread
    y_home_cover = pd.Series(np.nan, index=work.index, dtype="float")
    y_home_cover.loc[cover_val > 0] = 1.0
    y_home_cover.loc[cover_val < 0] = 0.0

    actual_total = hs + aas
    over_val = actual_total - total_line
    y_over = pd.Series(np.nan, index=work.index, dtype="float")
    y_over.loc[over_val > 0] = 1.0
    y_over.loc[over_val < 0] = 0.0

    # Moneyline target (exclude ties)
    y_home_win = pd.Series(np.nan, index=work.index, dtype="float")
    y_home_win.loc[actual_margin > 0] = 1.0
    y_home_win.loc[actual_margin < 0] = 0.0

    # Model probabilities (pre-calibration)
    edge_pts = margin + spread
    p_home_cover = pd.to_numeric(edge_pts, errors="coerce").apply(lambda e: _cover_prob_from_edge(float(e), ats_sigma) if pd.notna(e) else np.nan)
    p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)

    edge_total = total_pred - total_line
    p_over = pd.to_numeric(edge_total, errors="coerce").apply(lambda e: _cover_prob_from_edge(float(e), tot_sigma) if pd.notna(e) else np.nan)
    p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)

    # Moneyline home-win probability from predicted margin (matches the fallback mapping used in recs)
    try:
        import math
        scale_ml = float(__import__("os").environ.get("RECS_ML_MARGIN_SCALE", __import__("os").environ.get("ML_MARGIN_SCALE", "6.5")))
    except Exception:
        scale_ml = 6.5
    if not (scale_ml > 0):
        scale_ml = 6.5
    def _p_home_from_margin(m: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(m) / float(scale_ml)))
        except Exception:
            return float("nan")
    p_home_win = pd.to_numeric(margin, errors="coerce").apply(lambda m: _p_home_from_margin(float(m)) if pd.notna(m) else np.nan)
    p_home_win = pd.to_numeric(p_home_win, errors="coerce").clip(0.01, 0.99)

    method = str(args.method).strip().lower()
    if method == "isotonic":
        ml_curve = _fit_isotonic(p_home_win, y_home_win, points=int(args.points))
        ats_curve = _fit_isotonic(p_home_cover, y_home_cover, points=int(args.points))
        tot_curve = _fit_isotonic(p_over, y_over, points=int(args.points))
    else:
        ml_curve = _fit_bin_linear(p_home_win, y_home_win, bins=int(args.bins))
        ats_curve = _fit_bin_linear(p_home_cover, y_home_cover, bins=int(args.bins))
        tot_curve = _fit_bin_linear(p_over, y_over, bins=int(args.bins))

    out: Dict[str, Any] = {
        "moneyline": ml_curve,
        "ats": ats_curve,
        "total": tot_curve,
        "meta": {
            "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
            "season": int(args.season),
            "end_week": int(args.end_week),
            "lookback": int(args.lookback),
            "walk_forward": bool(args.walk_forward),
            "line_mode": str(args.line_mode),
            "bins": int(args.bins),
            "method": str(args.method),
            "points": int(args.points),
            "rows_final": int(len(work)),
            "ats_sigma": float(ats_sigma),
            "total_sigma": float(tot_sigma),
            "ml_margin_scale": float(scale_ml),
            "prob_shrink": float(shrink),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
