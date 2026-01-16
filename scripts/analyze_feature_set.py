from __future__ import annotations

"""
Analyze enhanced feature set:
- Correlation and mutual information with targets/residuals
- Outputs ranked CSVs to inform modeling focus

Usage:
  python scripts/analyze_feature_set.py --season 2025 --lookback 10 --out-dir nfl_compare/data/backtests/2025_wk17
"""

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def analyze(season: int, lookback: int) -> dict:
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)
    df = merge_features(games, stats, lines, wx).copy()
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    # Completed games only, target season window (last N weeks)
    comp = df.dropna(subset=["home_score", "away_score"]).copy()
    comp = comp[comp["season"] == int(season)]
    if lookback and "week" in comp.columns:
        max_w = int(_safe_numeric(comp["week"]).max()) if len(comp) else int(lookback)
        comp = comp[_safe_numeric(comp["week"]) >= max(1, max_w - int(lookback) + 1)]

    # Targets
    margin_act = _safe_numeric(comp.get("home_score")) - _safe_numeric(comp.get("away_score"))
    total_act = _safe_numeric(comp.get("home_score")) + _safe_numeric(comp.get("away_score"))
    spread_home = _safe_numeric(comp.get("spread_home")) if "spread_home" in comp.columns else pd.Series(index=comp.index, dtype=float)
    close_spread = _safe_numeric(comp.get("close_spread_home")) if "close_spread_home" in comp.columns else pd.Series(index=comp.index, dtype=float)
    spread_ref = spread_home.fillna(close_spread)
    market_total = _safe_numeric(comp.get("total")) if "total" in comp.columns else pd.Series(index=comp.index, dtype=float)
    close_total = _safe_numeric(comp.get("close_total")) if "close_total" in comp.columns else pd.Series(index=comp.index, dtype=float)
    total_ref = market_total.fillna(close_total)

    resid_margin = margin_act - (-spread_ref)
    resid_total = total_act - total_ref
    home_win = (margin_act > 0).astype(int)
    home_cover = (margin_act + spread_ref > 0).astype(int)
    over_total = (total_act > total_ref).astype(int)

    # Feature candidates: exclude IDs/scores/targets to avoid leakage
    exclude = {"home_score", "away_score", "game_id", "date", "game_date", "season", "week",
               "spread_home", "close_spread_home", "total", "close_total"}
    X = comp[[c for c in comp.columns if c not in exclude]].copy()
    # Coerce numerics and drop all-null columns
    for c in X.columns:
        X[c] = _safe_numeric(X[c])
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    # Correlations
    def corr_rank(target: pd.Series) -> pd.DataFrame:
        t = _safe_numeric(target)
        vals = []
        for c in X.columns:
            try:
                r = float(pd.Series(X[c]).corr(t))
            except Exception:
                r = np.nan
            vals.append({"feature": c, "corr": r})
        out = pd.DataFrame(vals).sort_values("corr", ascending=False)
        return out

    corr_margin = corr_rank(resid_margin)
    corr_total = corr_rank(resid_total)

    # Mutual information
    def mi_reg(target: pd.Series) -> pd.DataFrame:
        try:
            scores = mutual_info_regression(X.values, _safe_numeric(target).values)
            return pd.DataFrame({"feature": X.columns, "mi": scores}).sort_values("mi", ascending=False)
        except Exception:
            return pd.DataFrame(columns=["feature", "mi"])  # empty

    def mi_clf(target: pd.Series) -> pd.DataFrame:
        try:
            scores = mutual_info_classif(X.values, target.astype(int).values, discrete_features=False)
            return pd.DataFrame({"feature": X.columns, "mi": scores}).sort_values("mi", ascending=False)
        except Exception:
            return pd.DataFrame(columns=["feature", "mi"])  # empty

    mi_margin = mi_reg(resid_margin)
    mi_total = mi_reg(resid_total)
    mi_home_win = mi_clf(home_win)
    mi_home_cover = mi_clf(home_cover)
    mi_over_total = mi_clf(over_total)

    return {
        "corr_margin": corr_margin,
        "corr_total": corr_total,
        "mi_margin": mi_margin,
        "mi_total": mi_total,
        "mi_home_win": mi_home_win,
        "mi_home_cover": mi_home_cover,
        "mi_over_total": mi_over_total,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze feature correlations and mutual information with targets")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    res = analyze(args.season, args.lookback)
    out_dir: Optional[Path]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = DATA_DIR / "backtests" / f"{args.season}_wk{args.season}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        for key, df in res.items():
            fp = out_dir / f"feature_{key}.csv"
            df.to_csv(fp, index=False)
        print(f"Wrote feature analysis CSVs to {out_dir}")
    except Exception as e:
        print(f"Failed writing feature analysis: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
