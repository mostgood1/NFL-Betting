import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_predictions, _load_games, _build_week_view, _attach_model_predictions, _apply_totals_calibration_to_view

DATA_DIR = ROOT / 'nfl_compare' / 'data'


def _binwise_calibration(probs: pd.Series, y: pd.Series, n_bins: int = 20):
    p = pd.to_numeric(probs, errors='coerce')
    t = pd.to_numeric(y, errors='coerce')
    m = p.notna() & t.notna()
    p = p[m].clip(0.0, 1.0)
    t = t[m].astype(int)
    if p.empty:
        return None
    # Avoid degenerate small sample
    n_bins = int(max(5, min(n_bins, 50)))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p.values, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=t.values, minlength=n_bins)
    # Bin centers and rates
    xs = (bins[:-1] + bins[1:]) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(counts > 0, sums / counts, np.nan)
    # Fill gaps by interpolation
    # Forward/backward fill then linear interpolate remaining
    ser = pd.Series(rates)
    if ser.isna().all():
        return None
    ser = ser.fillna(method='ffill').fillna(method='bfill')
    ser = ser.interpolate(limit_direction='both')
    ys = ser.clip(0.0, 1.0).values.tolist()
    return {
        'xs': xs.tolist(),
        'ys': ys,
        'method': 'bin-linear',
        'n_bins': n_bins,
    }


def _cover_prob_from_edge(edge_pts: float, scale: float) -> float:
    import math
    if scale <= 0:
        scale = 1.0
    try:
        return 1.0 / (1.0 + math.exp(-edge_pts / scale))
    except OverflowError:
        return 1.0 if edge_pts > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description='Fit probability calibration for moneyline (prob_home_win), ATS cover, and Totals over.')
    ap.add_argument('--season', type=int, required=True, help='Season for slicing the window')
    ap.add_argument('--end-week', type=int, required=True, help='End week inclusive for the calibration window')
    ap.add_argument('--lookback', type=int, default=6, help='Number of recent weeks to include (walk-back)')
    ap.add_argument('--out', type=str, default=str(DATA_DIR / 'prob_calibration.json'), help='Output JSON file path')
    ap.add_argument('--ats-sigma', type=float, default=float((os.environ.get('NFL_ATS_SIGMA') or 9.0)), help='Sigma for ATS logistic mapping (edge/scale)')
    ap.add_argument('--total-sigma', type=float, default=float((os.environ.get('NFL_TOTAL_SIGMA') or 10.0)), help='Sigma for Totals logistic mapping (edge/scale)')
    ap.add_argument('--prob-shrink', type=float, default=float((os.environ.get('RECS_PROB_SHRINK') or 0.35)), help='Shrinkage toward 0.5 applied before calibration')
    args = ap.parse_args()

    pred = _load_predictions()
    games = _load_games()
    if pred is None and games is None:
        print('No data available for calibration.', file=sys.stderr)
        return 0

    season = int(args.season)
    end_wk = int(args.end_week)
    look = max(1, int(args.lookback))

    view = _build_week_view(pred, games, season, end_wk)
    view = _attach_model_predictions(view)
    try:
        view = _apply_totals_calibration_to_view(view)
    except Exception:
        pass

    if view is None or view.empty:
        print('Empty merged view; nothing to calibrate.', file=sys.stderr)
        return 0

    v = view.copy()
    # Restrict to training window: (season, week) pairs within lookback
    # If earlier weeks not present in this season (e.g., playoffs), we'll use what exists
    if 'week' in v.columns and 'season' in v.columns:
        try:
            mask_season = pd.to_numeric(v['season'], errors='coerce').eq(season)
            w = pd.to_numeric(v['week'], errors='coerce')
            mask_week = w.notna() & (w <= end_wk) & (w >= max(1, end_wk - look + 1))
            v = v[mask_season & mask_week].copy()
        except Exception:
            pass

    # Final games only with outcomes
    try:
        hs = pd.to_numeric(v.get('home_score'), errors='coerce')
        as_ = pd.to_numeric(v.get('away_score'), errors='coerce')
        finals = hs.notna() & as_.notna()
        v = v[finals].copy()
        v['home_win'] = (hs > as_).astype(int)
    except Exception:
        pass

    if v.empty or 'prob_home_win' not in v.columns:
        print('No prob_home_win or no finals in window; nothing to calibrate.', file=sys.stderr)
        return 0

    # Moneyline calibration
    cal_ml = _binwise_calibration(v['prob_home_win'], v['home_win'], n_bins=20)

    # ATS cover calibration
    cal_ats = None
    try:
        # Close spread and model margin
        cs = pd.to_numeric(v.get('close_spread_home'), errors='coerce')
        if cs.isna().all():
            # Fallbacks when closers missing
            cs = pd.to_numeric(v.get('market_spread_home'), errors='coerce')
            if cs.isna().all():
                cs = pd.to_numeric(v.get('spread_home'), errors='coerce')
        ph = pd.to_numeric(v.get('pred_home_points'), errors='coerce')
        pa = pd.to_numeric(v.get('pred_away_points'), errors='coerce')
        margin_pred = (ph - pa)
        valid = cs.notna() & margin_pred.notna()
        if valid.any():
            edge = (margin_pred + cs).where(valid)
            p_raw = _cover_prob_from_edge(edge, float(args.ats_sigma))
            # Shrink toward 0.5 before calibration like in app
            shrink = float(args.prob_shrink)
            p_raw = 0.5 + (p_raw - 0.5) * (1.0 - shrink)
            # Observed cover outcome (drop pushes)
            hs = pd.to_numeric(v.get('home_score'), errors='coerce')
            as_ = pd.to_numeric(v.get('away_score'), errors='coerce')
            actual_margin = (hs - as_)
            cover_val = (actual_margin + cs)
            # Drop pushes
            obs = cover_val.apply(lambda x: 1 if x > 0 else (0 if x < 0 else np.nan))
            cal_ats = _binwise_calibration(pd.Series(p_raw), obs, n_bins=20)
    except Exception:
        cal_ats = None

    # Totals over calibration
    cal_tot = None
    try:
        ct = pd.to_numeric(v.get('close_total'), errors='coerce')
        if ct.isna().all():
            ct = pd.to_numeric(v.get('market_total'), errors='coerce')
            if ct.isna().all():
                ct = pd.to_numeric(v.get('total'), errors='coerce')
        # Prefer calibrated total if present
        total_pred = pd.to_numeric(v.get('pred_total_cal'), errors='coerce')
        if total_pred.isna().all():
            total_pred = pd.to_numeric(v.get('pred_total'), errors='coerce')
        valid = ct.notna() & total_pred.notna()
        if valid.any():
            edge_t = (total_pred - ct).where(valid)
            p_raw = _cover_prob_from_edge(edge_t, float(args.total_sigma))
            shrink = float(args.prob_shrink)
            p_raw = 0.5 + (p_raw - 0.5) * (1.0 - shrink)
            hs = pd.to_numeric(v.get('home_score'), errors='coerce')
            as_ = pd.to_numeric(v.get('away_score'), errors='coerce')
            actual_total = (hs + as_)
            obs = pd.Series(np.where(actual_total > ct, 1, np.where(actual_total < ct, 0, np.nan)))
            cal_tot = _binwise_calibration(pd.Series(p_raw), obs, n_bins=20)
    except Exception:
        cal_tot = None

    out = {
        'moneyline': cal_ml,
        'ats': cal_ats,
        'total': cal_tot,
        'meta': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'season': season,
            'end_week': end_wk,
            'lookback': look,
            'rows': int(len(v)),
        }
    }

    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(out, indent=2))
    print(f'Wrote probability calibration -> {out_fp}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
