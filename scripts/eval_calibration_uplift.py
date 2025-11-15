"""
Evaluate probability calibration uplift for moneyline, ATS, and totals over a recent window.

This script builds the merged game view using app helpers, computes raw vs calibrated
probabilities, and reports Brier scores and reliability buckets (deciles) for each market.

Outputs JSON and Markdown under nfl_compare/data/backtests/<season>_wk<end_week>/.

Usage:
  python scripts/eval_calibration_uplift.py --season 2025 --end-week 10 --lookback 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'nfl_compare' / 'data'

# Import app helpers lazily to avoid heavy import side-effects elsewhere
from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _derive_predictions_from_market,
    _apply_totals_calibration_to_view,
    _apply_prob_calibration,
    _load_sigma_calibration,
)


def brier_score(p: pd.Series, y: pd.Series) -> float:
    p = pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)
    y = pd.to_numeric(y, errors='coerce')
    m = p.notna() & y.notna()
    if not m.any():
        return float('nan')
    return float(((p[m] - y[m]) ** 2).mean())


def reliability_table(p: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    p = pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)
    y = pd.to_numeric(y, errors='coerce')
    m = p.notna() & y.notna()
    if not m.any():
        return pd.DataFrame(columns=['bin_low','bin_high','count','pred_mean','obs_rate'])
    # Use quantile bins for balanced counts
    try:
        edges = np.unique(np.nanquantile(p[m], q=np.linspace(0, 1, n_bins + 1)))
        # Ensure at least 2 edges
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    except Exception:
        edges = np.array([0.0, 1.0])
    bins = pd.cut(p[m], bins=edges, include_lowest=True, duplicates='drop')
    df = pd.DataFrame({'p': p[m], 'y': y[m], 'bin': bins})
    # Explicit observed=False for compatibility across pandas versions
    try:
        g = df.groupby('bin', observed=False)
    except TypeError:
        g = df.groupby('bin')
    out = g.agg(count=('y','size'), pred_mean=('p','mean'), obs_rate=('y','mean')).reset_index(drop=False)
    # Expand interval bounds
    out['bin_low'] = out['bin'].apply(lambda x: float(x.left) if hasattr(x, 'left') else np.nan)
    out['bin_high'] = out['bin'].apply(lambda x: float(x.right) if hasattr(x, 'right') else np.nan)
    out = out.drop(columns=['bin'])
    cols = ['bin_low','bin_high','count','pred_mean','obs_rate']
    return out[cols]


def logistic_prob_from_edge(edge: pd.Series, scale: float) -> pd.Series:
    scale = float(scale) if scale and scale > 0 else 1.0
    e = pd.to_numeric(edge, errors='coerce')
    with np.errstate(over='ignore'):
        p = 1.0 / (1.0 + np.exp(-e / scale))
    return pd.Series(p, index=edge.index)


def compute_probs(v: pd.DataFrame, ats_sigma: float, total_sigma: float, prob_shrink: float) -> Dict[str, Dict[str, pd.Series]]:
    out: Dict[str, Dict[str, pd.Series]] = {}

    # Moneyline: use prob_home_win directly
    p_ml_raw = pd.to_numeric(v.get('prob_home_win'), errors='coerce').clip(0.0, 1.0)
    p_ml_cal = p_ml_raw.apply(lambda x: _apply_prob_calibration(float(x), which='moneyline') if pd.notna(x) else np.nan)
    out['moneyline'] = {'raw': p_ml_raw, 'cal': pd.to_numeric(p_ml_cal, errors='coerce')}

    # ATS: edge = model margin + close_spread_home; map to prob via logistic, shrink then calibrate
    cs = pd.to_numeric(v.get('close_spread_home'), errors='coerce')
    if cs.isna().all():
        # Fallback to consensus/seeded markets then generic spread_home if close not available
        cs = pd.to_numeric(v.get('market_spread_home'), errors='coerce')
        if cs.isna().all():
            cs = pd.to_numeric(v.get('spread_home'), errors='coerce')
    ph = pd.to_numeric(v.get('pred_home_points'), errors='coerce')
    pa = pd.to_numeric(v.get('pred_away_points'), errors='coerce')
    margin_pred = ph - pa
    edge_ats = margin_pred + cs
    p_ats_raw = logistic_prob_from_edge(edge_ats, ats_sigma)
    p_ats_raw = 0.5 + (p_ats_raw - 0.5) * (1.0 - float(prob_shrink))
    p_ats_cal = p_ats_raw.apply(lambda x: _apply_prob_calibration(float(x), which='ats') if pd.notna(x) else np.nan)
    out['ats'] = {'raw': p_ats_raw, 'cal': pd.to_numeric(p_ats_cal, errors='coerce')}

    # Totals Over: edge = total_pred(_cal) - close_total
    ct = pd.to_numeric(v.get('close_total'), errors='coerce')
    if ct.isna().all():
        ct = pd.to_numeric(v.get('market_total'), errors='coerce')
        if ct.isna().all():
            ct = pd.to_numeric(v.get('total'), errors='coerce')
    total_pred = pd.to_numeric(v.get('pred_total_cal'), errors='coerce')
    if total_pred.isna().all():
        total_pred = pd.to_numeric(v.get('pred_total'), errors='coerce')
    edge_tot = total_pred - ct
    p_tot_raw = logistic_prob_from_edge(edge_tot, total_sigma)
    p_tot_raw = 0.5 + (p_tot_raw - 0.5) * (1.0 - float(prob_shrink))
    p_tot_cal = p_tot_raw.apply(lambda x: _apply_prob_calibration(float(x), which='total') if pd.notna(x) else np.nan)
    out['total'] = {'raw': p_tot_raw, 'cal': pd.to_numeric(p_tot_cal, errors='coerce')}

    return out


def build_window(season: int, end_week: int, lookback: int) -> pd.DataFrame:
    """Build a multi-week evaluation window for the given season.

    We concatenate the week views for weeks [end_week - lookback + 1 .. end_week]
    so calibration has enough rows, instead of evaluating a single week only.
    """
    pred = _load_predictions()
    games = _load_games()
    start_week = max(1, int(end_week) - int(lookback) + 1)

    frames = []
    for wk in range(start_week, end_week + 1):
        try:
            v_w = _build_week_view(pred, games, int(season), int(wk))
            v_w = _attach_model_predictions(v_w)
            # If predictions are disabled (odds-only), derive minimal preds from markets
            try:
                v_w = _derive_predictions_from_market(v_w)
            except Exception:
                pass
            try:
                v_w = _apply_totals_calibration_to_view(v_w)
            except Exception:
                pass
            if v_w is not None and not v_w.empty:
                frames.append(v_w)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    v = pd.concat(frames, ignore_index=True)

    # Restrict to target season and finals within the window
    try:
        v['season'] = pd.to_numeric(v['season'], errors='coerce').astype('Int64')
        v['week'] = pd.to_numeric(v['week'], errors='coerce').astype('Int64')
        m = (v['season'].eq(int(season))) & v['week'].notna() & v['week'].between(start_week, end_week)
        v = v[m].copy()
    except Exception:
        pass
    try:
        hs = pd.to_numeric(v.get('home_score'), errors='coerce')
        as_ = pd.to_numeric(v.get('away_score'), errors='coerce')
        finals = hs.notna() & as_.notna()
        v = v[finals].copy()
        v['home_win'] = (hs > as_).astype(int)
        # ATS cover outcome (exclude pushes -> NaN)
        cs = pd.to_numeric(v.get('close_spread_home'), errors='coerce')
        if cs.isna().all():
            cs = pd.to_numeric(v.get('market_spread_home'), errors='coerce')
            if cs.isna().all():
                cs = pd.to_numeric(v.get('spread_home'), errors='coerce')
        cover_val = (hs - as_) + cs
        v['home_cover'] = cover_val.apply(lambda x: (1 if x > 0 else (0 if x < 0 else np.nan)))
        # Totals Over outcome (exclude pushes -> NaN)
        ct = pd.to_numeric(v.get('close_total'), errors='coerce')
        if ct.isna().all():
            ct = pd.to_numeric(v.get('market_total'), errors='coerce')
            if ct.isna().all():
                ct = pd.to_numeric(v.get('total'), errors='coerce')
        total_val = (hs + as_) - ct
        v['over'] = total_val.apply(lambda x: (1 if x > 0 else (0 if x < 0 else np.nan)))
    except Exception:
        pass
    return v


def main():
    ap = argparse.ArgumentParser(description='Evaluate prob calibration uplift for ML/ATS/Total.')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--end-week', type=int, required=True)
    ap.add_argument('--lookback', type=int, default=6)
    ap.add_argument('--ats-sigma', type=float, default=None)
    ap.add_argument('--total-sigma', type=float, default=None)
    ap.add_argument('--prob-shrink', type=float, default=0.35)
    args = ap.parse_args()

    season = int(args.season)
    end_week = int(args.end_week)
    lookback = max(1, int(args.lookback))

    v = build_window(season, end_week, lookback)
    if v is None or v.empty:
        print('No window data to evaluate; nothing written.')
        return 0

    # Prefer calibrated sigma from file/env if not provided
    ats_sigma = args.ats_sigma
    total_sigma = args.total_sigma
    if ats_sigma is None or total_sigma is None:
        try:
            sigma = _load_sigma_calibration()
            if ats_sigma is None:
                ats_sigma = float(sigma.get('ats_sigma', 9.0))
            if total_sigma is None:
                total_sigma = float(sigma.get('total_sigma', 10.0))
        except Exception:
            if ats_sigma is None:
                ats_sigma = 9.0
            if total_sigma is None:
                total_sigma = 10.0

    probs = compute_probs(v, float(ats_sigma), float(total_sigma), float(args.prob_shrink))

    results: Dict[str, Any] = {'meta': {'season': season, 'end_week': end_week, 'lookback': lookback, 'rows': int(len(v))}}
    md_lines = []
    md_lines.append(f"# Calibration Uplift (Season {season}, Weeks {max(1, end_week - lookback + 1)}–{end_week})\n")

    # Evaluate each market
    eval_specs = [
        ('moneyline', 'home_win'),
        ('ats', 'home_cover'),
        ('total', 'over'),
    ]
    out_dir = DATA_DIR / 'backtests' / f'{season}_wk{end_week}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, target_col in eval_specs:
        p_raw = probs[key]['raw']
        p_cal = probs[key]['cal']
        y = pd.to_numeric(v.get(target_col), errors='coerce')
        # Exclude pushes for ats/total (NaN in y)
        m_raw = p_raw.notna() & y.notna()
        m_cal = p_cal.notna() & y.notna()
        br_raw = brier_score(p_raw[m_raw], y[m_raw])
        br_cal = brier_score(p_cal[m_cal], y[m_cal])
        rel_raw = reliability_table(p_raw[m_raw], y[m_raw], n_bins=10)
        rel_cal = reliability_table(p_cal[m_cal], y[m_cal], n_bins=10)

        results[key] = {
            'brier_raw': br_raw,
            'brier_cal': br_cal,
            'brier_improvement': (br_raw - br_cal) if (pd.notna(br_raw) and pd.notna(br_cal)) else None,
            'counts_raw': int(m_raw.sum()),
            'counts_cal': int(m_cal.sum()),
            'reliability_raw': rel_raw.to_dict(orient='records'),
            'reliability_cal': rel_cal.to_dict(orient='records'),
        }

        # Write per-market CSVs for convenience
        try:
            rel_raw.to_csv(out_dir / f'{key}_reliability_raw.csv', index=False)
            rel_cal.to_csv(out_dir / f'{key}_reliability_cal.csv', index=False)
        except Exception:
            pass

        # Markdown section
        md_lines.append(f"## {key.title()}\n")
        md_lines.append(f"- Brier (raw): {br_raw:.5f}  |  Brier (cal): {br_cal:.5f}  |  Δ: {(br_raw - br_cal) if (pd.notna(br_raw) and pd.notna(br_cal)) else float('nan'):.5f}\n")
        md_lines.append("")

    # Write JSON and Markdown
    try:
        with open(out_dir / 'calibration_eval.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass
    try:
        (out_dir / 'calibration_eval.md').write_text("\n".join(md_lines), encoding='utf-8')
    except Exception:
        pass

    # Console summary
    try:
        print(json.dumps(results, indent=2))
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
