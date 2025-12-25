import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from typing import Optional, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _load_predictions, _load_games, _build_week_view, _attach_model_predictions, _apply_totals_calibration_to_view

DATA_DIR = ROOT / 'nfl_compare' / 'data'


def _apply_mapping(xs: list[float], ys: list[float], p: pd.Series) -> pd.Series:
    """Piecewise-linear apply of calibration mapping defined by xs->ys onto probability series p.
    Clamps to [0,1] and enforces weak monotonicity on ys.
    """
    try:
        if not isinstance(xs, (list, tuple)) or not isinstance(ys, (list, tuple)) or len(xs) < 2 or len(xs) != len(ys):
            return pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)
        # Sort and clamp
        pairs = sorted([(float(x), float(y)) for x, y in zip(xs, ys)], key=lambda t: t[0])
        xs_s = [a for a, _ in pairs]
        ys_s = [max(0.0, min(1.0, b)) for _, b in pairs]
        # Enforce weak monotonicity to avoid pathological mappings
        for i in range(1, len(ys_s)):
            if ys_s[i] < ys_s[i-1]:
                ys_s[i] = ys_s[i-1]
        x0 = xs_s[0]; xN = xs_s[-1]
        s = pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)
        # Vectorized apply via numpy interpolation
        import numpy as _np
        vals = _np.interp(s.astype(float).values, _np.array(xs_s, dtype=float), _np.array(ys_s, dtype=float))
        return pd.Series(vals, index=s.index)
    except Exception:
        return pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)


def _brier_score(p: pd.Series, y: pd.Series) -> float:
    p = pd.to_numeric(p, errors='coerce').clip(0.0, 1.0)
    y = pd.to_numeric(y, errors='coerce')
    m = p.notna() & y.notna()
    if not m.any():
        return float('nan')
    return float(((p[m] - y[m]) ** 2).mean())


def _non_degrading_or_identity(probs: pd.Series, y: pd.Series, cal_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return cal_obj if it improves Brier on the provided data; otherwise identity mapping.
    Uses PROB_CAL_REQUIRE_UPLIFT epsilon to require a minimum improvement.
    """
    ident = {'xs': [0.0, 1.0], 'ys': [0.0, 1.0], 'method': 'identity'}
    try:
        p = pd.to_numeric(probs, errors='coerce').clip(0.0, 1.0)
        t = pd.to_numeric(y, errors='coerce')
        m = p.notna() & t.notna()
        if not m.any():
            return ident
        raw = _brier_score(p[m], t[m])
        if not cal_obj:
            return ident
        p_cal = _apply_mapping(cal_obj.get('xs', []), cal_obj.get('ys', []), p[m])
        cal = _brier_score(p_cal, t[m])
        require = float(os.environ.get('PROB_CAL_REQUIRE_UPLIFT', '0.0'))
        if pd.isna(raw) or pd.isna(cal) or (cal >= (raw - max(0.0, require))):
            return ident
        return cal_obj
    except Exception:
        return ident


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
    ser = ser.ffill().bfill()
    ser = ser.interpolate(limit_direction='both')
    ys = ser.clip(0.0, 1.0).values.tolist()
    return {
        'xs': xs.tolist(),
        'ys': ys,
        'method': 'bin-linear',
        'n_bins': n_bins,
    }


def _isotonic_calibration(probs: pd.Series, y: pd.Series, n_points: int = 51):
    """Fit an isotonic regression mapping p->calibrated_p and return as sampled xs/ys pairs.
    Uses sklearn IsotonicRegression with clipping; returns None if insufficient data.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return None
    p = pd.to_numeric(probs, errors='coerce').clip(0.0, 1.0)
    t = pd.to_numeric(y, errors='coerce')
    m = p.notna() & t.notna()
    p = p[m]
    t = t[m].astype(int)
    if len(p) < 20:
        return None
    try:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(p.values.astype(float), t.values.astype(float))
        xs = np.linspace(0.0, 1.0, max(10, int(n_points)))
        ys = iso.predict(xs)
        ys = np.clip(ys, 0.0, 1.0)
        # Enforce weak monotonicity (should already hold)
        for i in range(1, len(ys)):
            if ys[i] < ys[i-1]:
                ys[i] = ys[i-1]
        return {
            'xs': xs.tolist(),
            'ys': ys.tolist(),
            'method': 'isotonic',
            'n_points': int(n_points),
        }
    except Exception:
        return None


def _cv_select_mapping(probs: pd.Series, y: pd.Series, candidates: list[str], n_splits: int = 3, random_state: int = 42) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    """Cross-validate candidate calibration methods and return best mapping plus scores.
    candidates: list of method names in {'bin', 'isotonic'}
    Returns (best_mapping_obj or None, {'raw': br_raw, 'cal': br_cal}).
    If no candidate improves over raw, returns None to indicate identity is preferred.
    """
    try:
        from sklearn.model_selection import KFold
    except Exception:
        # Fallback: no sklearn CV; just pick isotonic if available else bin; no guarantee of uplift
        best = None
        scores = {'raw': float('nan'), 'cal': float('nan')}
        for m in candidates:
            if m == 'isotonic':
                best = _isotonic_calibration(probs, y)
                break
            if m == 'bin':
                best = _binwise_calibration(probs, y, n_bins=20)
                break
        return best, scores
    p = pd.to_numeric(probs, errors='coerce').clip(0.0, 1.0)
    t = pd.to_numeric(y, errors='coerce')
    m = p.notna() & t.notna()
    p = p[m]
    t = t[m].astype(int)
    if len(p) < 40:
        # Too few to CV; fit isotonic if possible else bin
        for mth in candidates:
            if mth == 'isotonic':
                cal = _isotonic_calibration(p, t)
                if cal:
                    return cal, {'raw': float('nan'), 'cal': float('nan')}
            if mth == 'bin':
                cal = _binwise_calibration(p, t, n_bins=20)
                if cal:
                    return cal, {'raw': float('nan'), 'cal': float('nan')}
        return None, {'raw': float('nan'), 'cal': float('nan')}
    kf = KFold(n_splits=max(2, int(n_splits)), shuffle=True, random_state=random_state)
    def _brier(a: pd.Series, b: pd.Series) -> float:
        a = pd.to_numeric(a, errors='coerce').clip(0.0, 1.0)
        b = pd.to_numeric(b, errors='coerce')
        m2 = a.notna() & b.notna()
        if not m2.any():
            return float('nan')
        return float(((a[m2] - b[m2]) ** 2).mean())
    best_m = None
    best_score = float('inf')
    raw_scores = []
    cal_scores = []
    for train_idx, val_idx in kf.split(p.values):
        p_tr = pd.Series(p.values[train_idx])
        y_tr = pd.Series(t.values[train_idx])
        p_va = pd.Series(p.values[val_idx])
        y_va = pd.Series(t.values[val_idx])
        raw_scores.append(_brier(p_va, y_va))
        # Evaluate each candidate on this split
        local_best = None
        local_best_score = float('inf')
        for mth in candidates:
            if mth == 'isotonic':
                cal_obj = _isotonic_calibration(p_tr, y_tr)
            elif mth == 'bin':
                cal_obj = _binwise_calibration(p_tr, y_tr, n_bins=20)
            else:
                cal_obj = None
            if not cal_obj:
                continue
            p_cal_va = _apply_mapping(cal_obj['xs'], cal_obj['ys'], p_va)
            sc = _brier(p_cal_va, y_va)
            if pd.notna(sc) and sc < local_best_score:
                local_best = cal_obj
                local_best_score = sc
        cal_scores.append(local_best_score)
        # Track global best by average performance; we'll refit after CV on all data using the method that most often won
        if local_best is not None and local_best_score < best_score:
            best_m = local_best.get('method')
            best_score = local_best_score
    # Aggregate
    avg_raw = float(np.nanmean(raw_scores)) if raw_scores else float('nan')
    avg_cal = float(np.nanmean(cal_scores)) if cal_scores else float('nan')
    scores = {'raw': avg_raw, 'cal': avg_cal}
    # Require non-degradation; allow tiny epsilon
    require = float(os.environ.get('PROB_CAL_REQUIRE_UPLIFT', '0.0'))
    if pd.isna(avg_raw) or pd.isna(avg_cal) or (avg_cal >= (avg_raw - max(0.0, require))):
        return None, scores
    # Refit the chosen method on full data
    if best_m == 'isotonic':
        final = _isotonic_calibration(p, t)
    elif best_m == 'bin':
        final = _binwise_calibration(p, t, n_bins=20)
    else:
        final = None
    return final, scores


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
    ap.add_argument('--merge', action='store_true', help='Merge into existing file under by_season for the given season (adds/updates season key).')
    ap.add_argument('--ats-sigma', type=float, default=float((os.environ.get('NFL_ATS_SIGMA') or 9.0)), help='Sigma for ATS logistic mapping (edge/scale)')
    ap.add_argument('--total-sigma', type=float, default=float((os.environ.get('NFL_TOTAL_SIGMA') or 10.0)), help='Sigma for Totals logistic mapping (edge/scale)')
    ap.add_argument('--prob-shrink', type=float, default=float((os.environ.get('RECS_PROB_SHRINK') or 0.35)), help='Shrinkage toward 0.5 applied before calibration')
    ap.add_argument('--method', type=str, default=str(os.environ.get('PROB_CAL_METHOD') or 'auto'), choices=['auto','bin','isotonic'], help='Calibration method to fit (bin, isotonic) or auto-select via CV')
    ap.add_argument('--force-identity-ml', action='store_true', help='Force identity mapping for moneyline for this season')
    ap.add_argument('--force-identity-ats', action='store_true', help='Force identity mapping for ATS for this season')
    ap.add_argument('--force-identity-total', action='store_true', help='Force identity mapping for Totals for this season')
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
    cal_ml: Optional[Dict[str, Any]] = None
    if args.force_identity_ml:
        cal_ml = {'xs': [0.0, 1.0], 'ys': [0.0, 1.0], 'method': 'identity'}
    elif args.method == 'bin':
        cal_ml = _binwise_calibration(v['prob_home_win'], v['home_win'], n_bins=20)
    elif args.method == 'isotonic':
        cal_ml = _isotonic_calibration(v['prob_home_win'], v['home_win'], n_points=51)
    else:
        cal_ml, _ = _cv_select_mapping(v['prob_home_win'], v['home_win'], candidates=['isotonic','bin'])
    # Final guard: require uplift on full window (unless forced identity)
    if not args.force_identity_ml:
        cal_ml = _non_degrading_or_identity(v['prob_home_win'], v['home_win'], cal_ml)

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
            if args.force_identity_ats:
                cal_ats = {'xs': [0.0, 1.0], 'ys': [0.0, 1.0], 'method': 'identity'}
            elif args.method == 'bin':
                cal_ats = _binwise_calibration(pd.Series(p_raw), obs, n_bins=20)
            elif args.method == 'isotonic':
                cal_ats = _isotonic_calibration(pd.Series(p_raw), obs, n_points=51)
            else:
                cal_ats, _ = _cv_select_mapping(pd.Series(p_raw), obs, candidates=['isotonic','bin'])
            if not args.force_identity_ats:
                cal_ats = _non_degrading_or_identity(pd.Series(p_raw), obs, cal_ats)
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
            if args.force_identity_total:
                cal_tot = {'xs': [0.0, 1.0], 'ys': [0.0, 1.0], 'method': 'identity'}
            elif args.method == 'bin':
                cal_tot = _binwise_calibration(pd.Series(p_raw), obs, n_bins=20)
            elif args.method == 'isotonic':
                cal_tot = _isotonic_calibration(pd.Series(p_raw), obs, n_points=51)
            else:
                cal_tot, _ = _cv_select_mapping(pd.Series(p_raw), obs, candidates=['isotonic','bin'])
            if not args.force_identity_total:
                cal_tot = _non_degrading_or_identity(pd.Series(p_raw), obs, cal_tot)
    except Exception:
        cal_tot = None

    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    if args.merge:
        # Merge season-specific mappings into existing file using by_season
        try:
            if out_fp.exists():
                existing = json.loads(out_fp.read_text(encoding='utf-8'))
            else:
                existing = {}
        except Exception:
            existing = {}

        def _merge_market(key: str, cal_obj: dict | None):
            # If no improvement, write an explicit identity mapping to avoid season using a harmful prior curve
            if not cal_obj:
                cal_obj = {'xs': [0.0, 1.0], 'ys': [0.0, 1.0], 'method': 'identity'}
            node = existing.get(key)
            if node is None:
                node = {'by_season': {}}
            elif isinstance(node, dict) and ('xs' in node and 'ys' in node):
                # Convert direct map into by_season with default
                node = {'by_season': {'default': node}}
            elif not isinstance(node, dict):
                node = {'by_season': {}}
            bys = node.get('by_season', {})
            bys[str(season)] = cal_obj
            node['by_season'] = bys
            existing[key] = node

        _merge_market('moneyline', cal_ml)
        _merge_market('ats', cal_ats)
        _merge_market('total', cal_tot)
        meta = existing.get('meta', {}) if isinstance(existing.get('meta'), dict) else {}
        meta.update({'last_generated_at': datetime.utcnow().isoformat() + 'Z'})
        existing['meta'] = meta
        out_fp.write_text(json.dumps(existing, indent=2))
        print(f'Updated probability calibration (by_season) -> {out_fp} [season={season}]')
    else:
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
        out_fp.write_text(json.dumps(out, indent=2))
        print(f'Wrote probability calibration -> {out_fp}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
