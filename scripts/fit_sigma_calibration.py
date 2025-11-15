import sys
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _apply_totals_calibration_to_view,
)

OUT_FP = ROOT / 'nfl_compare' / 'data' / 'sigma_calibration.json'

def fit_sigma(season: int, through_week: int, lookback_weeks: int = 4) -> dict:
    # Gather residuals over [through_week - lookback_weeks, through_week - 1]
    wk_start = max(1, int(through_week) - int(lookback_weeks))
    pred_df = _load_predictions()
    games_df = _load_games()
    margin_resid = []
    total_resid = []
    for wk in range(wk_start, max(1, through_week)):
        view = _build_week_view(pred_df, games_df, season, wk)
        if view is None or view.empty:
            continue
        view = _attach_model_predictions(view)
        view = _apply_totals_calibration_to_view(view)
        # Final games only
        fin = view.copy()
        if 'home_score' not in fin.columns or 'away_score' not in fin.columns:
            continue
        fin = fin[pd.to_numeric(fin['home_score'], errors='coerce').notna() & pd.to_numeric(fin['away_score'], errors='coerce').notna()]
        if fin.empty:
            continue
        # Predictions
        # prefer pred_home_points/away_points; total uses pred_total_cal if present
        if 'pred_home_points' not in fin.columns and 'pred_home_score' in fin.columns:
            fin['pred_home_points'] = fin['pred_home_score']
        if 'pred_away_points' not in fin.columns and 'pred_away_score' in fin.columns:
            fin['pred_away_points'] = fin['pred_away_score']
        # Margin residuals
        try:
            ph = pd.to_numeric(fin.get('pred_home_points'), errors='coerce')
            pa = pd.to_numeric(fin.get('pred_away_points'), errors='coerce')
            if ph is not None and pa is not None:
                pred_margin = ph - pa
                act_margin = pd.to_numeric(fin['home_score'], errors='coerce') - pd.to_numeric(fin['away_score'], errors='coerce')
                resid = (act_margin - pred_margin).dropna().astype(float)
                margin_resid.extend(resid.tolist())
        except Exception:
            pass
        # Total residuals
        try:
            # use calibrated total when present; else raw pred_total
            pt = pd.to_numeric(fin.get('pred_total_cal', fin.get('pred_total')), errors='coerce')
            if pt is not None:
                act_total = pd.to_numeric(fin['home_score'], errors='coerce') + pd.to_numeric(fin['away_score'], errors='coerce')
                resid_t = (act_total - pt).dropna().astype(float)
                total_resid.extend(resid_t.tolist())
        except Exception:
            pass
    # Compute RMS as sigma estimate; fall back to defaults if insufficient data
    def rms(arr, default):
        if not arr:
            return default
        s = sum(x*x for x in arr)
        return math.sqrt(s / len(arr))
    ats_sigma = rms(margin_resid, 9.0)
    total_sigma = rms(total_resid, 10.0)
    return {"ats_sigma": float(ats_sigma), "total_sigma": float(total_sigma), "weeks": int(lookback_weeks)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--week', type=int, required=True)
    p.add_argument('--lookback', type=int, default=4)
    p.add_argument('--out', type=str, default=str(OUT_FP))
    args = p.parse_args()
    res = fit_sigma(args.season, args.week, args.lookback)
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    print(f"Wrote sigma calibration to {out_fp} -> {res}")

if __name__ == '__main__':
    main()
