"""Fit probability calibration curves from recommendation backtest details.

This produces a calibration file compatible with `app._apply_prob_calibration()`:
  nfl_compare/data/prob_calibration*.json

We fit simple bin-based reliability curves (piecewise linear):
- MONEYLINE: calibrates `prob_home_win`
- SPREAD: calibrates `prob_home_cover`
- TOTAL: calibrates `prob_over_total`

Targets are inferred from `selection`, `home_team`, `away_team`, and `result`:
- For ML: selection is home/away ML; Win/Loss => home win = 1/0
- For ATS: selection is home/away spread; Win/Loss => home cover = 1/0
- For total: selection is Over/Under; Win/Loss => over = 1/0
Pushes and unknowns are excluded.

Usage:
  python scripts/fit_prob_calibration_from_recs.py \
    --details nfl_compare/data/backtests/.../recs_backtest_details.csv \
    --out nfl_compare/data/prob_calibration_walkfwd_2025.json \
    --bins 12

Notes:
- This is a *calibration* step, not a model improvement.
- Ideally, calibration should be fit on past data only and applied forward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CalCurve:
    xs: List[float]
    ys: List[float]
    method: str
    n_bins: int
    n: int


def _infer_side(selection: str, home_team: str, away_team: str) -> Optional[str]:
    s = (selection or "").strip()
    ht = (home_team or "").strip()
    at = (away_team or "").strip()
    if not s or not ht or not at:
        return None
    # Prefer prefix match (selection format is typically "<Team> ...")
    if s.startswith(ht):
        return "HOME"
    if s.startswith(at):
        return "AWAY"
    # Fallback contains
    if ht in s:
        return "HOME"
    if at in s:
        return "AWAY"
    return None


def _infer_total_side(selection: str) -> Optional[str]:
    s = (selection or "").strip().lower()
    if s.startswith("over"):
        return "OVER"
    if s.startswith("under"):
        return "UNDER"
    return None


def _extract_base_probs(df: pd.DataFrame, market: str) -> Tuple[pd.Series, pd.Series]:
    """Return (p_base, y_base) for market.

    p_base is always from HOME/OVER perspective (not selected side).
    y_base is 1 when HOME/OVER wins.
    """
    m = df.copy()

    res_u = m.get("result", pd.Series([None] * len(m))).astype(str).str.upper().str.strip()
    is_win = res_u.eq("WIN")
    is_loss = res_u.eq("LOSS")

    # y_selected: win=1 loss=0 else nan
    y_sel = pd.Series(np.nan, index=m.index, dtype="float")
    y_sel.loc[is_win] = 1.0
    y_sel.loc[is_loss] = 0.0

    sel = m.get("selection", pd.Series([""] * len(m))).astype(str)

    if market == "MONEYLINE":
        side = pd.Series([None] * len(m), index=m.index, dtype="object")
        for i in m.index:
            side.loc[i] = _infer_side(sel.loc[i], str(m.get("home_team").loc[i]), str(m.get("away_team").loc[i]))
        p_home = pd.to_numeric(m.get("prob_home_win"), errors="coerce")
        p_sel = pd.to_numeric(m.get("prob_selected"), errors="coerce")
        # Reconstruct home prob if missing
        miss = p_home.isna() & p_sel.notna() & side.notna()
        p_home.loc[miss & (side == "HOME")] = p_sel.loc[miss & (side == "HOME")]
        p_home.loc[miss & (side == "AWAY")] = 1.0 - p_sel.loc[miss & (side == "AWAY")]
        # Target: home win
        y_home = pd.Series(np.nan, index=m.index, dtype="float")
        ok = y_sel.notna() & side.notna()
        y_home.loc[ok & (side == "HOME")] = y_sel.loc[ok & (side == "HOME")]
        y_home.loc[ok & (side == "AWAY")] = 1.0 - y_sel.loc[ok & (side == "AWAY")]
        return p_home, y_home

    if market == "SPREAD":
        side = pd.Series([None] * len(m), index=m.index, dtype="object")
        for i in m.index:
            side.loc[i] = _infer_side(sel.loc[i], str(m.get("home_team").loc[i]), str(m.get("away_team").loc[i]))
        p_home = pd.to_numeric(m.get("prob_home_cover"), errors="coerce")
        p_sel = pd.to_numeric(m.get("prob_selected"), errors="coerce")
        miss = p_home.isna() & p_sel.notna() & side.notna()
        p_home.loc[miss & (side == "HOME")] = p_sel.loc[miss & (side == "HOME")]
        p_home.loc[miss & (side == "AWAY")] = 1.0 - p_sel.loc[miss & (side == "AWAY")]
        y_home = pd.Series(np.nan, index=m.index, dtype="float")
        ok = y_sel.notna() & side.notna()
        y_home.loc[ok & (side == "HOME")] = y_sel.loc[ok & (side == "HOME")]
        y_home.loc[ok & (side == "AWAY")] = 1.0 - y_sel.loc[ok & (side == "AWAY")]
        return p_home, y_home

    if market == "TOTAL":
        side = sel.map(_infer_total_side)
        p_over = pd.to_numeric(m.get("prob_over_total"), errors="coerce")
        p_sel = pd.to_numeric(m.get("prob_selected"), errors="coerce")
        miss = p_over.isna() & p_sel.notna() & side.notna()
        p_over.loc[miss & (side == "OVER")] = p_sel.loc[miss & (side == "OVER")]
        p_over.loc[miss & (side == "UNDER")] = 1.0 - p_sel.loc[miss & (side == "UNDER")]
        y_over = pd.Series(np.nan, index=m.index, dtype="float")
        ok = y_sel.notna() & side.notna()
        y_over.loc[ok & (side == "OVER")] = y_sel.loc[ok & (side == "OVER")]
        y_over.loc[ok & (side == "UNDER")] = 1.0 - y_sel.loc[ok & (side == "UNDER")]
        return p_over, y_over

    raise ValueError(f"Unknown market: {market}")


def _fit_bin_linear(p: pd.Series, y: pd.Series, bins: int) -> Optional[CalCurve]:
    p = pd.to_numeric(p, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = p.notna() & y.notna()
    if int(m.sum()) < max(20, bins * 3):
        return None

    pp = p.loc[m].clip(lower=0.0, upper=1.0)
    yy = y.loc[m].clip(lower=0.0, upper=1.0)

    # Equal-count bins; duplicates='drop' handles low variance p.
    try:
        cats = pd.qcut(pp, q=bins, duplicates="drop")
    except Exception:
        return None

    g = pd.DataFrame({"p": pp, "y": yy, "bin": cats}).groupby("bin", observed=True)
    centers = g["p"].mean()
    rates = g["y"].mean()
    ns = g.size()

    xs = [0.0] + [float(v) for v in centers.tolist()] + [1.0]
    ys = [0.0] + [float(v) for v in rates.tolist()] + [1.0]

    # Sort and enforce weak monotonicity
    pairs = sorted(zip(xs, ys), key=lambda t: float(t[0]))
    xs_s: List[float] = []
    ys_s: List[float] = []
    for x, yv in pairs:
        x = float(max(0.0, min(1.0, x)))
        yv = float(max(0.0, min(1.0, yv)))
        xs_s.append(x)
        ys_s.append(yv)
    for i in range(1, len(ys_s)):
        if ys_s[i] < ys_s[i - 1]:
            ys_s[i] = ys_s[i - 1]

    return CalCurve(xs=xs_s, ys=ys_s, method="bin-linear", n_bins=int(len(centers)), n=int(m.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit probability calibration curves from recs backtest details")
    ap.add_argument("--details", action="append", required=True, help="Path(s) to recs_backtest_details.csv")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. nfl_compare/data/prob_calibration_walkfwd.json)")
    ap.add_argument("--bins", type=int, default=12, help="Target number of bins (qcut); may drop if duplicates")
    ap.add_argument("--season", type=int, default=None, help="Optional season label for meta")
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    for fp in args.details:
        p = Path(fp)
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise SystemExit("No rows loaded")

    out: Dict[str, Any] = {
        "moneyline": None,
        "ats": None,
        "total": None,
        "meta": {
            "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
            "source_details": [str(Path(fp)) for fp in args.details],
            "bins": int(args.bins),
            "season": int(args.season) if args.season is not None else None,
            "rows": int(len(df)),
        },
    }

    for key, market in [("moneyline", "MONEYLINE"), ("ats", "SPREAD"), ("total", "TOTAL")]:
        sub = df[df.get("type").astype(str).str.upper().eq(market)].copy()
        if sub.empty:
            continue
        p_base, y_base = _extract_base_probs(sub, market)
        curve = _fit_bin_linear(p_base, y_base, bins=int(args.bins))
        if curve is None:
            continue
        out[key] = {
            "xs": curve.xs,
            "ys": curve.ys,
            "method": curve.method,
            "n_bins": int(curve.n_bins),
            "n": int(curve.n),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
