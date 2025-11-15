from __future__ import annotations

"""
Train light supervised adjusters using recent reconciliation data, then apply to this week's props:

- WR receiving yards (rec_yards)
- WR receptions (receptions)
- RB rushing yards (rush_yards)

Model: Ridge regression per target with simple, robust features.
Outputs: nfl_compare/data/player_props_ml_<season>_wk<week>.csv with blended columns when available:
    - rec_yards_ml, rec_yards_blend
    - receptions_ml, receptions_blend
    - rush_yards_ml, rush_yards_blend

Usage:
    python scripts/props_supervised_adjust.py --season 2025 --week 10

Notes:
- Degrades gracefully if reconciliation or features are missing.
- Blends use PROPS_ML_BLEND (0..1, default 0.5).
"""

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "nfl_compare" / "data"


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1","true","yes","y","on"}


def _load_weekly_props(season: int, week: int) -> Optional[pd.DataFrame]:
    fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
    if not fp.exists():
        return None
    try:
        return pd.read_csv(fp)
    except Exception:
        return None


def _reconcile_weeks(season: int, weeks: List[int]) -> pd.DataFrame:
    try:
        from nfl_compare.src.reconciliation import reconcile_props  # lazy import
    except Exception:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for wk in weeks:
        try:
            df = reconcile_props(int(season), int(wk))
        except Exception:
            df = None
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def _fit_wr_model_rec_yards(train_df: pd.DataFrame) -> Optional[Ridge]:
    if train_df is None or train_df.empty:
        return None
    df = train_df.copy()
    # WR only
    if 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == 'WR']
    # target variable
    y = pd.to_numeric(df.get('rec_yards_act'), errors='coerce')
    # features
    rec = pd.to_numeric(df.get('rec_yards'), errors='coerce')
    tgt = pd.to_numeric(df.get('targets'), errors='coerce')
    rcp = pd.to_numeric(df.get('receptions'), errors='coerce')
    X = pd.DataFrame({
        'rec_yards': rec,
        'targets': tgt,
        'receptions': rcp,
        'tgt_x_cr': tgt * (rcp / (tgt.replace(0, np.nan))).fillna(0.0),
    })
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < 50:
        return None
    try:
        model = Ridge(alpha=1.0, random_state=42)
    except TypeError:
        model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def _apply_wr_rec_yards_model(model: Ridge, df: pd.DataFrame, blend: float) -> pd.DataFrame:
    out = df.copy()
    pos = out.get('position', pd.Series(''))
    wr = pos.astype(str).str.upper() == 'WR'
    if not wr.any():
        return out
    rec = pd.to_numeric(out.get('rec_yards'), errors='coerce')
    tgt = pd.to_numeric(out.get('targets'), errors='coerce')
    rcp = pd.to_numeric(out.get('receptions'), errors='coerce')
    X = pd.DataFrame({
        'rec_yards': rec,
        'targets': tgt,
        'receptions': rcp,
        'tgt_x_cr': tgt * (rcp / (tgt.replace(0, np.nan))).fillna(0.0),
    })
    try:
        pred = model.predict(X.fillna(0.0))
    except Exception:
        pred = rec.fillna(0.0).values
    out['rec_yards_ml'] = pred
    try:
        b = float(blend)
    except Exception:
        b = 0.5
    b = float(np.clip(b, 0.0, 1.0))
    out['rec_yards_blend'] = (1.0 - b) * rec.fillna(0.0) + b * out['rec_yards_ml'].astype(float)
    return out


def _fit_wr_model_receptions(train_df: pd.DataFrame) -> Optional[Ridge]:
    if train_df is None or train_df.empty:
        return None
    df = train_df.copy()
    if 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == 'WR']
    y = pd.to_numeric(df.get('receptions_act'), errors='coerce')
    tgt = pd.to_numeric(df.get('targets'), errors='coerce')
    rec = pd.to_numeric(df.get('rec_yards'), errors='coerce')
    rcp = pd.to_numeric(df.get('receptions'), errors='coerce')
    X = pd.DataFrame({
        'receptions': rcp,
        'targets': tgt,
        'catch_rate': (rcp / (tgt.replace(0, np.nan))).fillna(0.0),
        'rec_yards': rec,
    })
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < 50:
        return None
    try:
        model = Ridge(alpha=1.0, random_state=42)
    except TypeError:
        model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def _apply_wr_receptions_model(model: Ridge, df: pd.DataFrame, blend: float) -> pd.DataFrame:
    out = df.copy()
    pos = out.get('position', pd.Series(''))
    wr = pos.astype(str).str.upper() == 'WR'
    if not wr.any():
        return out
    rcp = pd.to_numeric(out.get('receptions'), errors='coerce')
    tgt = pd.to_numeric(out.get('targets'), errors='coerce')
    rec = pd.to_numeric(out.get('rec_yards'), errors='coerce')
    X = pd.DataFrame({
        'receptions': rcp,
        'targets': tgt,
        'catch_rate': (rcp / (tgt.replace(0, np.nan))).fillna(0.0),
        'rec_yards': rec,
    })
    try:
        pred = model.predict(X.fillna(0.0))
    except Exception:
        pred = rcp.fillna(0.0).values
    out['receptions_ml'] = pred
    try:
        b = float(blend)
    except Exception:
        b = 0.5
    b = float(np.clip(b, 0.0, 1.0))
    out['receptions_blend'] = (1.0 - b) * rcp.fillna(0.0) + b * out['receptions_ml'].astype(float)
    return out


def _fit_rb_model_rush_yards(train_df: pd.DataFrame) -> Optional[Ridge]:
    if train_df is None or train_df.empty:
        return None
    df = train_df.copy()
    if 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == 'RB']
    y = pd.to_numeric(df.get('rush_yards_act'), errors='coerce')
    ry = pd.to_numeric(df.get('rush_yards'), errors='coerce')
    ra = pd.to_numeric(df.get('rush_attempts'), errors='coerce')
    X = pd.DataFrame({
        'rush_yards': ry,
        'rush_attempts': ra,
        'ypc': (ry / (ra.replace(0, np.nan))).fillna(0.0),
    })
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < 50:
        return None
    try:
        model = Ridge(alpha=1.0, random_state=42)
    except TypeError:
        model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def _apply_rb_rush_yards_model(model: Ridge, df: pd.DataFrame, blend: float) -> pd.DataFrame:
    out = df.copy()
    pos = out.get('position', pd.Series(''))
    rb = pos.astype(str).str.upper() == 'RB'
    if not rb.any():
        return out
    ry = pd.to_numeric(out.get('rush_yards'), errors='coerce')
    ra = pd.to_numeric(out.get('rush_attempts'), errors='coerce')
    X = pd.DataFrame({
        'rush_yards': ry,
        'rush_attempts': ra,
        'ypc': (ry / (ra.replace(0, np.nan))).fillna(0.0),
    })
    try:
        pred = model.predict(X.fillna(0.0))
    except Exception:
        pred = ry.fillna(0.0).values
    out['rush_yards_ml'] = pred
    try:
        b = float(blend)
    except Exception:
        b = 0.5
    b = float(np.clip(b, 0.0, 1.0))
    out['rush_yards_blend'] = (1.0 - b) * ry.fillna(0.0) + b * out['rush_yards_ml'].astype(float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervised adjustment for WR receiving yards")
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--lookback', type=int, default=4, help='Completed weeks to use for training')
    ap.add_argument('--blend', type=float, default=None, help='Blend weight for ML adjustment (0..1)')
    ap.add_argument('--out', type=str, default=None, help='Output CSV (default player_props_ml_<season>_wk<week>.csv)')
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)
    lookback = max(1, int(args.lookback))
    blend = args.blend
    if blend is None:
        import os
        try:
            blend = float(os.environ.get('PROPS_ML_BLEND', '0.5'))
        except Exception:
            blend = 0.5

    props_df = _load_weekly_props(season, week)
    if props_df is None or props_df.empty:
        print('No props found for this week; aborting ML adjust.')
        return 0

    weeks = list(range(max(1, week - lookback), week))
    train_df = _reconcile_weeks(season, weeks)
    # Initialize output
    out_df = props_df.copy()

    # WR receiving yards
    model_wr_yds = _fit_wr_model_rec_yards(train_df)
    if model_wr_yds is not None:
        out_df = _apply_wr_rec_yards_model(model_wr_yds, out_df, float(blend))
    else:
        out_df['rec_yards_ml'] = np.nan
        out_df['rec_yards_blend'] = out_df.get('rec_yards')

    # WR receptions
    model_wr_rec = _fit_wr_model_receptions(train_df)
    if model_wr_rec is not None:
        out_df = _apply_wr_receptions_model(model_wr_rec, out_df, float(blend))
    else:
        out_df['receptions_ml'] = np.nan
        out_df['receptions_blend'] = out_df.get('receptions')

    # RB rushing yards
    model_rb_rush = _fit_rb_model_rush_yards(train_df)
    if model_rb_rush is not None:
        out_df = _apply_rb_rush_yards_model(model_rb_rush, out_df, float(blend))
    else:
        out_df['rush_yards_ml'] = np.nan
        out_df['rush_yards_blend'] = out_df.get('rush_yards')

    out_fp = Path(args.out) if args.out else (DATA_DIR / f"player_props_ml_{season}_wk{week}.csv")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_fp, index=False)
    print(f"Wrote ML-adjusted props -> {out_fp}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
