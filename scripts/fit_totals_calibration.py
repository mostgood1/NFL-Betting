"""
Fit simple totals calibration parameters from recent completed weeks and write nfl_compare/data/totals_calibration.json.

Model:
  adjusted = scale * pred_total + shift
Optional market blend:
  adjusted_blend = (1 - market_blend) * adjusted + market_blend * market_total

We fit (scale, shift) by least squares on actual_total ~ pred_total.
Then we grid-search market_blend in [0,1] to minimize MSE using rows that have market_total.

Inputs are read from nfl_compare/data via existing loaders when possible.
Defensive fallbacks are used when files/columns are missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get_data_dir() -> Path:
    # Mirror app/data_sources logic
    env = os.environ.get("NFL_DATA_DIR")
    if env:
        return Path(env)
    # default to package data dir
    return Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _safe_read_csv(fp: Path) -> pd.DataFrame:
    try:
        if not fp.exists():
            return pd.DataFrame()
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _load_games(data_dir: Path) -> pd.DataFrame:
    # Prefer through local file to avoid import-time dependency issues
    return _safe_read_csv(data_dir / "games.csv")


def _load_lines(data_dir: Path) -> pd.DataFrame:
    return _safe_read_csv(data_dir / "lines.csv")


def _load_predictions_any(data_dir: Path) -> pd.DataFrame:
    # Try multiple locations/files for flexibility and combine them
    candidates: List[Path] = []
    # Primary: package data dir
    for name in ["predictions_week.csv", "predictions.csv", "predictions_locked.csv"]:
        candidates.append(data_dir / name)
    # Secondary: repo-level data folder
    repo_data = Path(__file__).resolve().parents[2] / "data"
    for name in ["predictions_week.csv", "predictions.csv", "predictions_locked.csv"]:
        candidates.append(repo_data / name)

    frames: List[pd.DataFrame] = []
    for fp in candidates:
        df = _safe_read_csv(fp)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    # Deduplicate by keys if present
    keys = [c for c in ["season","week","home_team","away_team"] if c in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep="last")
    # Ensure pred_total exists if possible
    if "pred_total" not in out.columns:
        if {"pred_home_points","pred_away_points"}.issubset(out.columns):
            out["pred_total"] = pd.to_numeric(out["pred_home_points"], errors="coerce") + pd.to_numeric(out["pred_away_points"], errors="coerce")
        elif {"pred_home_score","pred_away_score"}.issubset(out.columns):
            out["pred_total"] = pd.to_numeric(out["pred_home_score"], errors="coerce") + pd.to_numeric(out["pred_away_score"], errors="coerce")
    return out


def _pick_market_total_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["close_total", "market_total", "total", "open_total"]:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _infer_recent_weeks(games: pd.DataFrame, k: int) -> List[Tuple[int, int]]:
    if games is None or games.empty:
        return []
    # Completed games only
    g = games.copy()
    for c in ["season", "week"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    # Completed: both scores present and non-null
    if "home_score" in g.columns and "away_score" in g.columns:
        g = g[g["home_score"].notna() & g["away_score"].notna()]
    # Sort by season, week
    g = g.sort_values(["season", "week"]).dropna(subset=["season", "week"]) if {"season","week"}.issubset(g.columns) else g
    if g.empty or not {"season","week"}.issubset(g.columns):
        return []
    uniq = g[["season","week"]].drop_duplicates().sort_values(["season","week"], ascending=[True, True]).values.tolist()
    uniq = [(int(s), int(w)) for s, w in uniq]
    # take last k weeks across seasons
    return uniq[-k:]


def _join_frames(
    games: pd.DataFrame,
    preds: pd.DataFrame,
    lines: pd.DataFrame,
    weeks: List[Tuple[int, int]],
) -> pd.DataFrame:
    if not weeks:
        return pd.DataFrame()
    # Filter by weeks
    gg = games.copy()
    for c in ["season","week"]:
        if c in gg.columns:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
    mask = pd.Series([False] * len(gg))
    for (s, w) in weeks:
        mask = mask | ((gg.get("season") == s) & (gg.get("week") == w))
    gg = gg[mask]

    # Actual total
    if {"home_score","away_score"}.issubset(gg.columns):
        gg["actual_total"] = pd.to_numeric(gg["home_score"], errors="coerce") + pd.to_numeric(gg["away_score"], errors="coerce")
    else:
        gg["actual_total"] = np.nan

    # Build join keys
    keys = ["season","week","home_team","away_team"]
    # Normalize key types
    for df in (gg, preds, lines):
        if df is None or df.empty:
            continue
        for c in keys:
            if c in df.columns:
                if c in ("season","week"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    df[c] = df[c].astype(str)

    base = gg[keys + ["actual_total"]].copy() if set(keys).issubset(gg.columns) else gg.copy()

    # Merge predictions
    if preds is not None and not preds.empty:
        cols_pred = [c for c in preds.columns if c in keys or c in ("pred_total","pred_home_points","pred_away_points")]
        p = preds[cols_pred].copy()
        base = pd.merge(base, p, on=keys, how="left")

    # Merge lines and compute market_total
    if lines is not None and not lines.empty:
        mcol = _pick_market_total_col(lines)
        cols_line = [c for c in lines.columns if c in keys or c == mcol]
        l = lines[cols_line].copy()
        if mcol and mcol != "market_total":
            l = l.rename(columns={mcol: "market_total"})
        base = pd.merge(base, l, on=keys, how="left")

    # Final filter for rows with pred_total and actual_total
    if "pred_total" not in base.columns:
        # derive pred_total if pred_home/away present
        if {"pred_home_points","pred_away_points"}.issubset(base.columns):
            base["pred_total"] = pd.to_numeric(base["pred_home_points"], errors="coerce") + pd.to_numeric(base["pred_away_points"], errors="coerce")
        elif {"pred_home_score","pred_away_score"}.issubset(base.columns):
            base["pred_total"] = pd.to_numeric(base["pred_home_score"], errors="coerce") + pd.to_numeric(base["pred_away_score"], errors="coerce")
    return base


@dataclass
class FitResult:
    scale: float
    shift: float
    market_blend: float
    mse: float
    mae: float
    n: int
    weeks_used: List[Tuple[int, int]]


def _fit_scale_shift(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # Fit y ~ a*x + b via least squares; handle degenerate cases
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < 2 or np.allclose(x, x.mean()):
        # fallback: identity scale, bias toward mean diff
        a = 1.0
        b = float(np.nanmean(y - x)) if len(x) else 0.0
        return a, b
    try:
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b)
    except Exception:
        return 1.0, float(np.nanmean(y - x))


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    err = y_pred - y_true
    mse = float(np.nanmean(err ** 2)) if len(err) else float("nan")
    mae = float(np.nanmean(np.abs(err))) if len(err) else float("nan")
    return mse, mae


def fit_totals_calibration(df: pd.DataFrame) -> Optional[FitResult]:
    if df is None or df.empty:
        return None
    work = df.copy()
    # Coerce numeric
    for c in ["pred_total","actual_total","market_total"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["pred_total","actual_total"])  # require both
    if work.empty:
        return None
    x = work["pred_total"].to_numpy()
    y = work["actual_total"].to_numpy()
    a_raw, b_raw = _fit_scale_shift(x, y)

    # Safety clamp: prevent extreme calibrations from destabilizing sims/UI.
    # Note: we always write clamped (safe) values to disk; raw fit params are
    # returned separately for diagnostics.
    try:
        scale_min = float(os.environ.get('TOTALS_CAL_SCALE_MIN', '0.85'))
        scale_max = float(os.environ.get('TOTALS_CAL_SCALE_MAX', '1.15'))
        shift_min = float(os.environ.get('TOTALS_CAL_SHIFT_MIN', '-7.0'))
        shift_max = float(os.environ.get('TOTALS_CAL_SHIFT_MAX', '7.0'))
    except Exception:
        scale_min, scale_max = 0.85, 1.15
        shift_min, shift_max = -7.0, 7.0
    try:
        a = float(np.clip(float(a_raw), float(scale_min), float(scale_max)))
        b = float(np.clip(float(b_raw), float(shift_min), float(shift_max)))
    except Exception:
        a, b = 1.0, 0.0
    y_adj = a * x + b

    # Grid-search market_blend using only rows with market_total
    mbest = 0.0
    mse_best, mae_best = _eval_metrics(y, y_adj)
    if "market_total" in work.columns and work["market_total"].notna().any():
        has_m = work["market_total"].notna().to_numpy()
        xm = x[has_m]; ym = y[has_m]; mm = work.loc[has_m, "market_total"].to_numpy().astype(float)
        y_adj_m = a * xm + b
        # Search alpha in [0,1]
        cand = np.linspace(0.0, 1.0, 21)  # 0.05 steps
        for alpha in cand:
            y_hat = (1.0 - alpha) * y_adj_m + alpha * mm
            mse, mae = _eval_metrics(ym, y_hat)
            if math.isnan(mse):
                continue
            if mse < mse_best:
                mse_best, mae_best, mbest = mse, mae, float(alpha)

    res = FitResult(
        scale=float(a),
        shift=float(b),
        market_blend=float(mbest),
        mse=float(mse_best),
        mae=float(mae_best),
        n=int(len(work)),
        weeks_used=sorted(
            set(
                (int(s), int(w))
                for s, w in zip(work.get("season", []), work.get("week", []))
            )
            if {"season", "week"}.issubset(work.columns)
            else []
        ),
    )
    # Attach raw fit values for diagnostics (safe values are what get used).
    setattr(res, "raw_scale", float(a_raw))
    setattr(res, "raw_shift", float(b_raw))
    return res


def main():
    p = argparse.ArgumentParser(description="Fit totals calibration from recent completed weeks.")
    p.add_argument("--weeks", type=int, default=4, help="How many most recent completed weeks to use")
    p.add_argument("--out", type=str, default=None, help="Optional override output path for totals_calibration.json")
    args = p.parse_args()

    data_dir = _get_data_dir()
    games = _load_games(data_dir)
    preds = _load_predictions_any(data_dir)
    lines = _load_lines(data_dir)

    weeks = _infer_recent_weeks(games, max(1, args.weeks))
    if not weeks:
        print(json.dumps({"error": "No recent completed weeks inferred", "data_dir": str(data_dir)}))
        return

    df = _join_frames(games, preds, lines, weeks)
    if df.empty:
        print(json.dumps({"error": "No joined rows for selected weeks", "weeks": weeks}))
        return

    fit = fit_totals_calibration(df)
    if not fit:
        print(json.dumps({"error": "Unable to fit calibration", "rows": int(len(df))}))
        return

    out_path = Path(args.out) if args.out else (data_dir / "totals_calibration.json")
    payload = {
        "scale": fit.scale,
        "shift": fit.shift,
        "raw_scale": getattr(fit, "raw_scale", None),
        "raw_shift": getattr(fit, "raw_shift", None),
        "market_blend": fit.market_blend,
        "metrics": {
            "mse": fit.mse,
            "mae": fit.mae,
            "n": fit.n,
        },
        "weeks_used": fit.weeks_used,
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "out": str(out_path), **payload}))
    except Exception as e:
        print(json.dumps({"error": f"Failed to write {out_path}: {e}", **payload}))


if __name__ == "__main__":
    main()
