import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
# Prefer XGBoost; gracefully fall back to scikit-learn's HistGradientBoosting if unavailable
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    XGBRegressor = None  # type: ignore
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False

def _make_regressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42):
    if _HAS_XGB:
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state)
    else:
        # Map to HistGradientBoostingRegressor params
        return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate,
                                             max_depth=max_depth, random_state=random_state)

def _make_classifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                     random_state=42):
    if _HAS_XGB:
        return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state,
                             objective='binary:logistic', eval_metric='logloss')
    else:
        return HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=learning_rate,
                                              max_depth=max_depth, random_state=random_state)

TARGETS = {
    'home_margin': 'reg',
    'total_points': 'reg',
    'home_win': 'clf',
    'q1_total': 'reg',
    'q2_total': 'reg',
    'q3_total': 'reg',
    'q4_total': 'reg',
    'h1_total': 'reg',
    'h2_total': 'reg',
}

FEATURES = [
    'elo_diff',
    # use blended diffs for early-season stability
    'off_epa_diff', 'def_epa_diff', 'pace_secs_play_diff', 'pass_rate_diff', 'rush_rate_diff', 'sos_diff',
    # EPA by half (optional, filled when available)
    'off_epa_1h_diff','off_epa_2h_diff','def_epa_1h_diff','def_epa_2h_diff',
    # QB and continuity priors
    'qb_prior_diff','continuity_diff',
    # Rolling half/quarter diffs
    'h1_scored_diff','h1_allowed_diff','h2_scored_diff','h2_allowed_diff',
    'q1_scored_diff','q1_allowed_diff','q2_scored_diff','q2_allowed_diff','q3_scored_diff','q3_allowed_diff','q4_scored_diff','q4_allowed_diff',
    'h2_minus_h1_diff',
    # market + weather
    'spread_home', 'total', 'market_home_prob',
    # Weather/stadium effects
    'is_dome', 'is_turf', 'wx_temp_f', 'wx_wind_mph', 'wx_precip_pct'
]

@dataclass
class TrainedModels:
    regressors: Dict[str, Any]
    classifiers: Dict[str, Optional[Any]]


def _build_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    df = df.copy()
    df['home_win'] = (df['home_margin'] > 0).astype(int)
    # Reindex to ensure optional feature columns exist; fill missing with 0
    X = df.reindex(columns=FEATURES, fill_value=0).fillna(0)
    y_margin = df['home_margin']
    y_total = df['total_points']
    y_homewin = df['home_win']
    # Labels for half winners where available
    if {'home_first_half','away_first_half'}.issubset(df.columns):
        df['home_1h_win'] = (df['home_first_half'] >= df['away_first_half']).astype(int)
    elif {'home_q1','home_q2','away_q1','away_q2'}.issubset(df.columns):
        h1_home = (df['home_q1'].fillna(0) + df['home_q2'].fillna(0)).astype(float)
        h1_away = (df['away_q1'].fillna(0) + df['away_q2'].fillna(0)).astype(float)
        df['home_1h_win'] = (h1_home >= h1_away).astype(int)
    else:
        df['home_1h_win'] = np.nan
    if {'home_second_half','away_second_half'}.issubset(df.columns):
        df['home_2h_win'] = (df['home_second_half'] >= df['away_second_half']).astype(int)
    elif {'home_q3','home_q4','away_q3','away_q4'}.issubset(df.columns):
        h2_home = (df['home_q3'].fillna(0) + df['home_q4'].fillna(0)).astype(float)
        h2_away = (df['away_q3'].fillna(0) + df['away_q4'].fillna(0)).astype(float)
        df['home_2h_win'] = (h2_home >= h2_away).astype(int)
    else:
        df['home_2h_win'] = np.nan
    # quarter/half totals where present
    qh = pd.DataFrame(index=df.index)
    if {'home_q1','home_q2','home_q3','home_q4','away_q1','away_q2','away_q3','away_q4'}.issubset(df.columns):
        qh['q1_total'] = (df['home_q1'] + df['away_q1']).astype(float)
        qh['q2_total'] = (df['home_q2'] + df['away_q2']).astype(float)
        qh['q3_total'] = (df['home_q3'] + df['away_q3']).astype(float)
        qh['q4_total'] = (df['home_q4'] + df['away_q4']).astype(float)
        qh['h1_total'] = qh['q1_total'] + qh['q2_total']
        qh['h2_total'] = qh['q3_total'] + qh['q4_total']
    return X, y_margin, y_total, y_homewin, qh


def train_models(df: pd.DataFrame) -> TrainedModels:
    X, y_margin, y_total, y_homewin, qh = _build_frame(df)
    # Single split shared across targets to keep dimensions aligned
    X_train, X_val, ym_train, ym_val, yt_train, yt_val, yh_train, yh_val = train_test_split(
        X, y_margin, y_total, y_homewin, test_size=0.2, random_state=42
    )

    reg_margin = _make_regressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42)
    reg_total = _make_regressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42)

    reg_margin.fit(X_train, ym_train)
    reg_total.fit(X_train, yt_train)

    clf_home: Optional[Any] = None
    clf_1h: Optional[Any] = None
    clf_2h: Optional[Any] = None
    if len(np.unique(yh_train)) > 1:
        clf_home = _make_classifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                                    random_state=42)
        clf_home.fit(X_train, yh_train)
    # First-half and second-half winner classifiers: derive labels from halves or quarters
    def _derive_half_wins(d: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        # 1H
        if {'home_first_half','away_first_half'}.issubset(d.columns):
            y1_all = (d['home_first_half'] >= d['away_first_half']).astype(int)
        elif {'home_q1','home_q2','away_q1','away_q2'}.issubset(d.columns):
            h1_home = (d['home_q1'].fillna(0) + d['home_q2'].fillna(0)).astype(float)
            h1_away = (d['away_q1'].fillna(0) + d['away_q2'].fillna(0)).astype(float)
            y1_all = (h1_home >= h1_away).astype(int)
        else:
            y1_all = pd.Series(np.nan, index=d.index)
        # 2H
        if {'home_second_half','away_second_half'}.issubset(d.columns):
            y2_all = (d['home_second_half'] >= d['away_second_half']).astype(int)
        elif {'home_q3','home_q4','away_q3','away_q4'}.issubset(d.columns):
            h2_home = (d['home_q3'].fillna(0) + d['home_q4'].fillna(0)).astype(float)
            h2_away = (d['away_q3'].fillna(0) + d['away_q4'].fillna(0)).astype(float)
            y2_all = (h2_home >= h2_away).astype(int)
        else:
            y2_all = pd.Series(np.nan, index=d.index)
        return y1_all, y2_all

    y1_all, y2_all = _derive_half_wins(df)
    y1 = y1_all.reindex(X_train.index)
    y2 = y2_all.reindex(X_train.index)
    if not y1.isna().all() and len(np.unique(y1.dropna())) > 1:
        clf_1h = _make_classifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                                  random_state=42)
        clf_1h.fit(X_train.loc[y1.dropna().index], y1.dropna())
    if not y2.isna().all() and len(np.unique(y2.dropna())) > 1:
        clf_2h = _make_classifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                                  random_state=42)
        clf_2h.fit(X_train.loc[y2.dropna().index], y2.dropna())

    # Optional quarter/half regressors if labels exist
    extra_regs: Dict[str, Any] = {}
    if not qh.empty:
        # Align labels to X_train indices
        qh_aligned = qh.reindex(X_train.index)
        for key in ['q1_total','q2_total','q3_total','q4_total','h1_total','h2_total']:
            if key in qh_aligned and qh_aligned[key].notna().any():
                yk = qh_aligned[key].dropna()
                Xk = X_train.loc[yk.index]
                if not Xk.empty:
                    reg = _make_regressor(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
                    reg.fit(Xk, yk)
                    extra_regs[key] = reg

    # basic metrics (safe)
    ym_pred = reg_margin.predict(X_val)
    yt_pred = reg_total.predict(X_val)
    try:
        mae_margin = float(mean_absolute_error(ym_val, ym_pred))
    except Exception:
        mae_margin = None
    try:
        mae_total = float(mean_absolute_error(yt_val, yt_pred))
    except Exception:
        mae_total = None
    if clf_home is not None and len(np.unique(yh_val)) > 1:
        try:
            yh_pred = clf_home.predict_proba(X_val)[:, 1]
            auc_home = float(roc_auc_score(yh_val, yh_pred))
        except Exception:
            auc_home = None
    else:
        auc_home = None

    # You could persist metrics somewhere if desired
    _ = {'MAE_margin': mae_margin, 'MAE_total': mae_total, 'AUC_home_win': auc_home}

    regs = {'home_margin': reg_margin, 'total_points': reg_total}
    regs.update(extra_regs)
    models = TrainedModels(
        regressors=regs,
        classifiers={'home_win': clf_home, 'home_1h_win': clf_1h, 'home_2h_win': clf_2h}
    )
    return models


def predict(models: TrainedModels, df_future: pd.DataFrame) -> pd.DataFrame:
    df = df_future.copy()
    X = df.reindex(columns=FEATURES, fill_value=0).fillna(0)
    df['pred_margin'] = models.regressors['home_margin'].predict(X)
    df['pred_total'] = models.regressors['total_points'].predict(X)
    # Optional calibration of game totals toward market with scale/shift and band clamp
    try:
        # Defaults tuned to avoid systematic unders while preserving model signal
        total_scale = float(os.getenv('NFL_TOTAL_SCALE', '1.03'))
        total_shift = float(os.getenv('NFL_TOTAL_SHIFT', '1.8'))
        total_blend = float(os.getenv('NFL_MARKET_TOTAL_BLEND', '0.70'))  # 0=no market, 1=all market
        total_band = float(os.getenv('NFL_TOTAL_MARKET_BAND', '6.5'))     # clamp within +/- points of market (0 to disable)
    except Exception:
        total_scale, total_shift, total_blend, total_band = 1.03, 1.8, 0.70, 6.5
    # Apply scale/shift first
    df['pred_total'] = (df['pred_total'] * total_scale) + total_shift
    # Blend toward market total when available
    if total_blend > 0 and 'total' in df.columns:
        mt = df['total'].astype(float)
        base = df['pred_total'].astype(float)
        df['pred_total'] = ((1 - total_blend) * base) + (total_blend * mt)
        # Optional clamp to stay within a band of the market
        if total_band and total_band > 0:
            df['pred_total'] = np.clip(df['pred_total'], mt - total_band, mt + total_band)
    if models.classifiers.get('home_win') is not None:
        df['prob_home_win'] = models.classifiers['home_win'].predict_proba(X)[:, 1]
    else:
        # Fallback: convert predicted margin to probability via logistic link
        # tuned slope roughly for NFL margins
        df['prob_home_win'] = 1 / (1 + np.exp(-df['pred_margin'] / 7.5))

    # implied team scores using spread/total-like system
    # Initial team targets from game-level models (prior to period allocation)
    df['pred_home_score'] = np.maximum(0, (df['pred_total'] + df['pred_margin']) / 2)
    df['pred_away_score'] = np.maximum(0, df['pred_total'] - df['pred_home_score'])
    df['_target_home_score'] = df['pred_home_score'].copy()
    df['_target_away_score'] = df['pred_away_score'].copy()

    # Quarter/half totals: prefer dedicated regressors if present; avoid hard 50/50 splits
    has_q_regs = all(k in models.regressors for k in ['q1_total','q2_total','q3_total','q4_total'])
    has_h_regs = all(k in models.regressors for k in ['h1_total','h2_total'])

    if has_q_regs or has_h_regs:
        # Predict raw quarters/halves where available
        if has_q_regs:
            df['pred_q1_total'] = models.regressors['q1_total'].predict(X)
            df['pred_q2_total'] = models.regressors['q2_total'].predict(X)
            df['pred_q3_total'] = models.regressors['q3_total'].predict(X)
            df['pred_q4_total'] = models.regressors['q4_total'].predict(X)
        else:
            # Initialize with zeros; will be filled after half totals decided
            df['pred_q1_total'] = 0.0
            df['pred_q2_total'] = 0.0
            df['pred_q3_total'] = 0.0
            df['pred_q4_total'] = 0.0

        if has_h_regs:
            h1_raw = models.regressors['h1_total'].predict(X)
            h2_raw = models.regressors['h2_total'].predict(X)
            # Scale halves to match game total and clamp shares
            hsum = (h1_raw + h2_raw)
            scale = np.where(hsum > 0, df['pred_total'] / hsum, 1.0)
            h1_target = np.maximum(0, h1_raw * scale)
            h2_target = np.maximum(0, h2_raw * scale)
            # Optional blend weight between quarter-sum halves and half regressors
            try:
                half_blend = float(os.getenv('NFL_HALF_BLEND', '0.65'))
            except Exception:
                half_blend = 0.65
        else:
            # Derive half targets from team tendencies when half regressors missing
            # Positive tendency -> more 2H scoring
            h_tend = (
                df.get('home_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) +
                df.get('away_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            ) / 2.0
            try:
                half_scale = float(os.getenv('NFL_HALF_TOTAL_SCALE', '6.0'))
                half_strength = float(os.getenv('NFL_HALF_TOTAL_STRENGTH', '0.08'))
            except Exception:
                half_scale, half_strength = 6.0, 0.08
            adj = -np.tanh(h_tend / max(half_scale, 1e-6)) * half_strength
            h1_share = (0.5 + adj).clip(0.42, 0.58)
            h1_target = h1_share * df['pred_total']
            h2_target = df['pred_total'] - h1_target
            half_blend = 1.0  # no quarter-sum info yet

        # Clean quarter predictions and check validity
        if has_q_regs:
            _q = df[['pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total']].copy().clip(lower=0)
            qsum = _q.sum(axis=1)
            invalid = (
                (_q[['pred_q1_total','pred_q2_total']] <= 0).any(axis=1)
                | (_q[['pred_q3_total','pred_q4_total']] <= 0).any(axis=1)
                | (qsum <= 0)
                | (_q['pred_q1_total'] > (df['pred_total'] * 0.8))
            )
            # Normalize quarters to game total first
            scale_q = np.where(qsum > 0, (df['pred_total'] / qsum), 1.0)
            _q = _q.mul(scale_q, axis=0)

            # Compute half sums from normalized quarters
            q_h1 = _q['pred_q1_total'] + _q['pred_q2_total']
            q_h2 = _q['pred_q3_total'] + _q['pred_q4_total']
            # Blend quarter-derived halves with targets
            df['pred_1h_total'] = (1 - half_blend) * q_h1 + half_blend * h1_target
            df['pred_2h_total'] = df['pred_total'] - df['pred_1h_total']
            # Rescale quarters within each half to match blended half totals
            h1_scale = np.where(q_h1 > 0, (df['pred_1h_total'] / q_h1), 0.5)
            h2_scale = np.where(q_h2 > 0, (df['pred_2h_total'] / q_h2), 0.5)
            _q['pred_q1_total'] = _q['pred_q1_total'] * h1_scale
            _q['pred_q2_total'] = _q['pred_q2_total'] * h1_scale
            _q['pred_q3_total'] = _q['pred_q3_total'] * h2_scale
            _q['pred_q4_total'] = _q['pred_q4_total'] * h2_scale

            # If invalid quarters, rebuild via heuristic within-half shares but keep half totals
            if invalid.any():
                pace = df.get('pace_secs_play_diff', pd.Series(0, index=df.index)).fillna(0)
                fast = (pace < 0).astype(float)
                # Within halves, Q2 and Q4 slightly higher; tilt by pace
                q1h_share = (0.48 + (-0.02 * fast)).clip(0.35, 0.60)
                q2h_share = 1.0 - q1h_share
                q3h_share = (0.49 + (0.01 * fast)).clip(0.35, 0.60)
                q4h_share = 1.0 - q3h_share
                _q.loc[invalid, 'pred_q1_total'] = df.loc[invalid, 'pred_1h_total'] * q1h_share.loc[invalid]
                _q.loc[invalid, 'pred_q2_total'] = df.loc[invalid, 'pred_1h_total'] * q2h_share.loc[invalid]
                _q.loc[invalid, 'pred_q3_total'] = df.loc[invalid, 'pred_2h_total'] * q3h_share.loc[invalid]
                _q.loc[invalid, 'pred_q4_total'] = df.loc[invalid, 'pred_2h_total'] * q4h_share.loc[invalid]

            df[['pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total']] = _q
        else:
            # No quarter regs: build quarters from half totals with reasonable within-half splits
            pace = df.get('pace_secs_play_diff', pd.Series(0, index=df.index)).fillna(0)
            fast = (pace < 0).astype(float)
            q1h_share = (0.48 + (-0.02 * fast)).clip(0.35, 0.60)
            q2h_share = 1.0 - q1h_share
            q3h_share = (0.49 + (0.01 * fast)).clip(0.35, 0.60)
            q4h_share = 1.0 - q3h_share
            df['pred_1h_total'] = h1_target
            df['pred_2h_total'] = h2_target
            df['pred_q1_total'] = df['pred_1h_total'] * q1h_share
            df['pred_q2_total'] = df['pred_1h_total'] * q2h_share
            df['pred_q3_total'] = df['pred_2h_total'] * q3h_share
            df['pred_q4_total'] = df['pred_2h_total'] * q4h_share

    else:
        # No extra regs: Team-specific half allocation with dynamic half totals (not 50/50)
        eps = 1e-6
        h1_home_int = (df.get('home_1h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) + df.get('away_1h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
        h1_away_int = (df.get('away_1h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) + df.get('home_1h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
        h2_home_int = (df.get('home_2h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) + df.get('away_2h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
        h2_away_int = (df.get('away_2h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) + df.get('home_2h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
        h1_home_share = (h1_home_int) / ((h1_home_int) + (h1_away_int) + eps)
        h2_home_share = (h2_home_int) / ((h2_home_int) + (h2_away_int) + eps)
        # Env-configurable second-half bias adjustment
        h2_bias = (df.get('home_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) - df.get('away_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0))
        bias_scale = float(os.getenv('NFL_H2_BIAS_SCALE', '5.0'))
        bias_strength = float(os.getenv('NFL_H2_BIAS_STRENGTH', '0.05'))
        h2_adj = np.tanh(h2_bias / max(bias_scale, 1e-6)) * bias_strength
        h2_home_share = (h2_home_share + h2_adj).clip(0.05, 0.95)
    # Determine half totals from tendencies (break 50/50 symmetry)
        h_tend = (
            df.get('home_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) +
            df.get('away_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
        ) / 2.0
        try:
            half_scale = float(os.getenv('NFL_HALF_TOTAL_SCALE', '6.0'))
            half_strength = float(os.getenv('NFL_HALF_TOTAL_STRENGTH', '0.08'))
        except Exception:
            half_scale, half_strength = 6.0, 0.08
        adj = -np.tanh(h_tend / max(half_scale, 1e-6)) * half_strength
        h1_share_total = (0.5 + adj).clip(0.42, 0.58)
        df['pred_1h_total'] = h1_share_total * df['pred_total']
        df['pred_2h_total'] = df['pred_total'] - df['pred_1h_total']

        # Build quarter totals within each half with modest variation
        pace = df.get('pace_secs_play_diff', pd.Series(0, index=df.index)).fillna(0)
        fast = (pace < 0).astype(float)
        q1h_share = (0.48 + (-0.02 * fast)).clip(0.35, 0.60)
        q2h_share = 1.0 - q1h_share
        q3h_share = (0.49 + (0.01 * fast)).clip(0.35, 0.60)
        q4h_share = 1.0 - q3h_share
        df['pred_q1_total'] = df['pred_1h_total'] * q1h_share
        df['pred_q2_total'] = df['pred_1h_total'] * q2h_share
        df['pred_q3_total'] = df['pred_2h_total'] * q3h_share
        df['pred_q4_total'] = df['pred_2h_total'] * q4h_share

    # Allocate team halves/quarters using rolling intensity shares and calibrate
    eps = 1e-6
    h1_home_int = (df.get('home_1h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) +
         df.get('away_1h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
    h1_away_int = (df.get('away_1h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) +
         df.get('home_1h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
    h2_home_int = (df.get('home_2h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) +
         df.get('away_2h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
    h2_away_int = (df.get('away_2h_scored_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0) +
         df.get('home_2h_allowed_avg', pd.Series(1.0, index=df.index)).astype(float).fillna(1.0))
    h1_home_share = (h1_home_int) / ((h1_home_int) + (h1_away_int) + eps)
    h2_home_share = (h2_home_int) / ((h2_home_int) + (h2_away_int) + eps)
    # Env-configurable second-half bias adjustment
    h2_bias = (df.get('home_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) -
        df.get('away_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0))
    bias_scale = float(os.getenv('NFL_H2_BIAS_SCALE', '5.0'))
    bias_strength = float(os.getenv('NFL_H2_BIAS_STRENGTH', '0.05'))
    h2_adj = np.tanh(h2_bias / max(bias_scale, 1e-6)) * bias_strength
    h2_home_share = (h2_home_share + h2_adj).clip(0.05, 0.95)

    # Halves (pre-calibration)
    df['pred_home_1h'] = df['pred_1h_total'] * h1_home_share
    df['pred_away_1h'] = df['pred_1h_total'] - df['pred_home_1h']
    df['pred_home_2h'] = df['pred_2h_total'] * h2_home_share
    df['pred_away_2h'] = df['pred_2h_total'] - df['pred_home_2h']

    # Calibrate half shares so that final team totals match the initial game-level targets
    tsum = (df['pred_1h_total'] + df['pred_2h_total']).replace(0, np.nan)
    # Current implied home total from halves
    home_half_sum = (df['pred_home_1h'] + df['pred_home_2h'])
    # Required delta to reach target
    delta = (df['_target_home_score'] - home_half_sum) / tsum
    # Adjust shares and recompute halves; clamp to [0.05, 0.95]
    h1_home_share_adj = (h1_home_share + delta).clip(0.05, 0.95)
    h2_home_share_adj = (h2_home_share + delta).clip(0.05, 0.95)
    df['pred_home_1h'] = df['pred_1h_total'] * h1_home_share_adj
    df['pred_away_1h'] = df['pred_1h_total'] - df['pred_home_1h']
    df['pred_home_2h'] = df['pred_2h_total'] * h2_home_share_adj
    df['pred_away_2h'] = df['pred_2h_total'] - df['pred_home_2h']
    # Quarters: apply calibrated quarter-specific shares if available to prevent same-team sweeps
    # Build quarter-specific intensity-derived shares; fallback to half shares if missing
    for q, half_share in [(1, h1_home_share_adj), (2, h1_home_share_adj), (3, h2_home_share_adj), (4, h2_home_share_adj)]:
        hs = df.get(f'home_q{q}_scored_avg', pd.Series(np.nan, index=df.index)).astype(float)
        aa = df.get(f'away_q{q}_allowed_avg', pd.Series(np.nan, index=df.index)).astype(float)
        as_ = df.get(f'away_q{q}_scored_avg', pd.Series(np.nan, index=df.index)).astype(float)
        ha = df.get(f'home_q{q}_allowed_avg', pd.Series(np.nan, index=df.index)).astype(float)
        inten_home = (hs.fillna(1.0) + aa.fillna(1.0))
        inten_away = (as_.fillna(1.0) + ha.fillna(1.0))
        share_q = (inten_home / (inten_home + inten_away + eps)).clip(0.05, 0.95)
        # blend quarter share lightly with half-level calibrated share
        try:
            qshare_blend = float(os.getenv('NFL_Q_SHARE_BLEND', '0.7'))
        except Exception:
            qshare_blend = 0.7
        share_final = (qshare_blend * half_share) + ((1 - qshare_blend) * share_q)
        df[f'pred_home_q{q}'] = df[f'pred_q{q}_total'] * share_final
    df['pred_away_q1'] = df['pred_q1_total'] - df['pred_home_q1']
    df['pred_away_q2'] = df['pred_q2_total'] - df['pred_home_q2']
    df['pred_away_q3'] = df['pred_q3_total'] - df['pred_home_q3']
    df['pred_away_q4'] = df['pred_q4_total'] - df['pred_home_q4']

    # (removed duplicated allocation/calibration block)

    # Optional anti-sweep adjustment: if one team is projected to win all 4 quarters but one is very close,
    # flip the closest quarter by a small within-half reallocation that preserves:
    # - each quarter total
    # - each half's team totals
    # - full-game team totals
    try:
        anti_sweep_on = os.getenv('NFL_ANTI_SWEEP_ENABLE', '1') not in ('0', 'false', 'False')
        close_thr = float(os.getenv('NFL_ANTI_SWEEP_CLOSE_THR', '1.2'))  # only flip if closest quarter margin < 1.2
        blowout_thr = float(os.getenv('NFL_ANTI_SWEEP_BLOWOUT_THR', '3.0'))  # skip if all quarter margins are big
        buffer = float(os.getenv('NFL_ANTI_SWEEP_BUFFER', '0.1'))  # extra to ensure flip
    except Exception:
        anti_sweep_on, close_thr, blowout_thr, buffer = True, 1.2, 3.0, 0.1

    if anti_sweep_on:
        for i in df.index:
            dq = np.array([
                float(df.at[i, 'pred_home_q1'] - df.at[i, 'pred_away_q1']),
                float(df.at[i, 'pred_home_q2'] - df.at[i, 'pred_away_q2']),
                float(df.at[i, 'pred_home_q3'] - df.at[i, 'pred_away_q3']),
                float(df.at[i, 'pred_home_q4'] - df.at[i, 'pred_away_q4']),
            ])
            signs = np.sign(dq)
            if not (np.all(signs > 0) or np.all(signs < 0)):
                continue  # not a sweep
            # If all margins are convincingly large, keep the sweep
            if np.all(np.abs(dq) >= blowout_thr):
                continue
            # Pick the quarter with the smallest absolute margin to flip
            q_idx = int(np.argmin(np.abs(dq)))  # 0..3
            if np.abs(dq[q_idx]) >= close_thr:
                continue  # closest is not close enough
            # Choose a donor quarter in the same half to counterbalance
            same_half = [0, 1] if q_idx in (0, 1) else [2, 3]
            donor_candidates = [j for j in same_half if j != q_idx]
            if not donor_candidates:
                continue
            # Prefer the quarter with largest margin in same direction for safe buffer
            donor_idx = max(donor_candidates, key=lambda j: np.abs(dq[j]))
            # Required nudge to flip target quarter: new_delta = old_delta -/+ 2*n > 0 for sign flip
            required = (abs(dq[q_idx]) / 2.0) + buffer
            # Cap nudge so we don't push a team below zero in either target or donor quarter
            # For target quarter we will move from the leading team to the trailing team
            if dq[q_idx] > 0:
                max_n_target = float(df.at[i, 'pred_home_q{}'.format(q_idx + 1)])
                max_n_donor = float(df.at[i, 'pred_away_q{}'.format(donor_idx + 1)])
            else:
                max_n_target = float(df.at[i, 'pred_away_q{}'.format(q_idx + 1)])
                max_n_donor = float(df.at[i, 'pred_home_q{}'.format(donor_idx + 1)])
            nudge = min(required, max(0.0, max_n_target - 1e-6), max(0.0, max_n_donor - 1e-6))
            if nudge <= 0:
                continue
            # Apply reallocation: flip target quarter by shifting nudge from leader to trailer,
            # and reverse the shift in the donor quarter to keep half team totals unchanged.
            tq = q_idx + 1
            dq_idx = donor_idx + 1
            if dq[q_idx] > 0:
                # Home leads in target quarter; move to away in target, and from away to home in donor
                df.at[i, f'pred_home_q{tq}'] = df.at[i, f'pred_home_q{tq}'] - nudge
                df.at[i, f'pred_away_q{tq}'] = df.at[i, f'pred_away_q{tq}'] + nudge
                df.at[i, f'pred_home_q{dq_idx}'] = df.at[i, f'pred_home_q{dq_idx}'] + nudge
                df.at[i, f'pred_away_q{dq_idx}'] = df.at[i, f'pred_away_q{dq_idx}'] - nudge
            else:
                # Away leads in target quarter; move to home in target, and from home to away in donor
                df.at[i, f'pred_home_q{tq}'] = df.at[i, f'pred_home_q{tq}'] + nudge
                df.at[i, f'pred_away_q{tq}'] = df.at[i, f'pred_away_q{tq}'] - nudge
                df.at[i, f'pred_home_q{dq_idx}'] = df.at[i, f'pred_home_q{dq_idx}'] - nudge
                df.at[i, f'pred_away_q{dq_idx}'] = df.at[i, f'pred_away_q{dq_idx}'] + nudge
            # No need to rescale totals: quarter totals and half team totals remain unchanged by construction

    # Margin calibration: optionally blend toward market spread and sharpen dispersion, then
    # rescale team halves/quarters proportionally to preserve pred_total while widening separation.
    try:
        market_blend = float(os.getenv('NFL_MARKET_MARGIN_BLEND', '0.15'))  # 0=no market, 1=all market
        margin_sharpen = float(os.getenv('NFL_MARGIN_SHARPEN', '1.15'))     # base scalar >1 widens margins
        margin_clip = float(os.getenv('NFL_MARGIN_CLIP', '24.0'))           # cap absolute margin
        target_std = float(os.getenv('NFL_MARGIN_TARGET_STD', '8.5'))       # auto-scale target std (0 disables)
        min_scale = float(os.getenv('NFL_MARGIN_MIN_SCALE', '0.75'))        # lower bound for auto-scale
        max_scale = float(os.getenv('NFL_MARGIN_MAX_SCALE', '3.0'))         # upper bound for auto-scale
    except Exception:
        market_blend, margin_sharpen, margin_clip = 0.15, 1.15, 24.0
        target_std, min_scale, max_scale = 8.5, 0.75, 3.0
    if market_blend > 0 or margin_sharpen != 1.0:
        # Base/model margin from current halves
        base_margin = (df['pred_home_1h'] + df['pred_home_2h']) - (df['pred_away_1h'] + df['pred_away_2h'])
        # Market spread to home-margin (spread_home negative when home favored)
        spread = df.get('spread_home', pd.Series(np.nan, index=df.index)).astype(float)
        market_margin = np.where(spread.notna(), -spread.values, base_margin.values)
        blended_margin = (1 - market_blend) * base_margin.values + market_blend * market_margin
        # Auto-scale toward a target std if configured
        scale_auto = 1.0
        if target_std and target_std > 0:
            cur_std = float(np.nanstd(blended_margin)) if np.isfinite(blended_margin).all() else float(np.nanstd(base_margin.values))
            if cur_std > 1e-6:
                scale_auto = np.clip(target_std / cur_std, min_scale, max_scale)
        scale_total = margin_sharpen * scale_auto
        scaled_margin = np.clip(blended_margin * scale_total, -margin_clip, margin_clip)
        # Compute new team totals while keeping game total fixed
        new_home_tot = np.maximum(0.0, (df['pred_total'].values + scaled_margin) / 2.0)
        new_away_tot = np.maximum(0.0, df['pred_total'].values - new_home_tot)
        # Current team totals
        cur_home_tot = (df['pred_home_1h'] + df['pred_home_2h']).values + 1e-9
        cur_away_tot = (df['pred_away_1h'] + df['pred_away_2h']).values + 1e-9
        f_home = np.clip(new_home_tot / cur_home_tot, 0.25, 4.0)
        f_away = np.clip(new_away_tot / cur_away_tot, 0.25, 4.0)
        # Scale halves
        df['pred_home_1h'] = df['pred_home_1h'] * f_home
        df['pred_home_2h'] = df['pred_home_2h'] * f_home
        df['pred_away_1h'] = df['pred_away_1h'] * f_away
        df['pred_away_2h'] = df['pred_away_2h'] * f_away
        # Scale quarters per team
        for q in [1,2,3,4]:
            df[f'pred_home_q{q}'] = df[f'pred_home_q{q}'] * f_home
            df[f'pred_away_q{q}'] = df[f'pred_away_q{q}'] * f_away
        # Recompute quarter/half totals from team values
        df['pred_q1_total'] = df['pred_home_q1'] + df['pred_away_q1']
        df['pred_q2_total'] = df['pred_home_q2'] + df['pred_away_q2']
        df['pred_q3_total'] = df['pred_home_q3'] + df['pred_away_q3']
        df['pred_q4_total'] = df['pred_home_q4'] + df['pred_away_q4']
        df['pred_1h_total'] = df['pred_q1_total'] + df['pred_q2_total']
        df['pred_2h_total'] = df['pred_q3_total'] + df['pred_q4_total']

        # Re-apply anti-sweep after rescaling, as global home/away scaling can change quarter winners
        if anti_sweep_on:
            for i in df.index:
                dq = np.array([
                    float(df.at[i, 'pred_home_q1'] - df.at[i, 'pred_away_q1']),
                    float(df.at[i, 'pred_home_q2'] - df.at[i, 'pred_away_q2']),
                    float(df.at[i, 'pred_home_q3'] - df.at[i, 'pred_away_q3']),
                    float(df.at[i, 'pred_home_q4'] - df.at[i, 'pred_away_q4']),
                ])
                signs = np.sign(dq)
                if not (np.all(signs > 0) or np.all(signs < 0)):
                    continue  # not a sweep
                if np.all(np.abs(dq) >= blowout_thr):
                    continue
                q_idx = int(np.argmin(np.abs(dq)))
                if np.abs(dq[q_idx]) >= close_thr:
                    continue
                same_half = [0, 1] if q_idx in (0, 1) else [2, 3]
                donor_candidates = [j for j in same_half if j != q_idx]
                if not donor_candidates:
                    continue
                donor_idx = max(donor_candidates, key=lambda j: np.abs(dq[j]))
                required = (abs(dq[q_idx]) / 2.0) + buffer
                if dq[q_idx] > 0:
                    max_n_target = float(df.at[i, f'pred_home_q{q_idx + 1}'])
                    max_n_donor = float(df.at[i, f'pred_away_q{donor_idx + 1}'])
                else:
                    max_n_target = float(df.at[i, f'pred_away_q{q_idx + 1}'])
                    max_n_donor = float(df.at[i, f'pred_home_q{donor_idx + 1}'])
                nudge = min(required, max(0.0, max_n_target - 1e-6), max(0.0, max_n_donor - 1e-6))
                if nudge <= 0:
                    continue
                tq = q_idx + 1
                dq_idx = donor_idx + 1
                if dq[q_idx] > 0:
                    df.at[i, f'pred_home_q{tq}'] -= nudge
                    df.at[i, f'pred_away_q{tq}'] += nudge
                    df.at[i, f'pred_home_q{dq_idx}'] += nudge
                    df.at[i, f'pred_away_q{dq_idx}'] -= nudge
                else:
                    df.at[i, f'pred_home_q{tq}'] += nudge
                    df.at[i, f'pred_away_q{tq}'] -= nudge
                    df.at[i, f'pred_home_q{dq_idx}'] -= nudge
                    df.at[i, f'pred_away_q{dq_idx}'] += nudge
            # Quarter/half totals remain consistent by construction

    # tiny epsilon to avoid exact zeros or equalities that cause labeling ties; preserve totals by symmetric nudges
    mask_pos = df['pred_total'] > 0
    for col in ['pred_home_1h','pred_away_1h','pred_home_2h','pred_away_2h']:
        df.loc[mask_pos & (df[col] <= 0), col] = 1e-6
    # add a minuscule random-ish nudge based on hash to break exact equality at period level without drift
    rng = np.sin((df.index.to_series().astype(float, errors='ignore').fillna(0).values if hasattr(df.index, 'to_series') else np.arange(len(df))) + 1) * 1e-6
    df['pred_home_q1'] = df['pred_home_q1'] + rng
    df['pred_away_q1'] = df['pred_q1_total'] - df['pred_home_q1']
    df['pred_home_q2'] = df['pred_home_q2'] + rng
    df['pred_away_q2'] = df['pred_q2_total'] - df['pred_home_q2']
    df['pred_home_q3'] = df['pred_home_q3'] + rng
    df['pred_away_q3'] = df['pred_q3_total'] - df['pred_home_q3']
    df['pred_home_q4'] = df['pred_home_q4'] + rng
    df['pred_away_q4'] = df['pred_q4_total'] - df['pred_home_q4']

    # winners by quarter with stricter tie tolerance + market-based tie-break for very close deltas
    eps_strict = 0.01
    eps_label = 0.5  # if still effectively tied within 0.5 pts, lean to market favorite for labeling only
    spread = df.get('spread_home', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    mhp = df.get('market_home_prob', pd.Series(0.5, index=df.index)).astype(float).fillna(0.5)
    market_pm = np.where(spread != 0, -spread, (mhp - 0.5) * 12.0)
    favored_team = np.where(market_pm >= 0, df['home_team'], df['away_team'])
    for q in [1,2,3,4]:
        h = df[f'pred_home_q{q}']
        a = df[f'pred_away_q{q}']
        delta = h - a
        base = np.where(delta > eps_strict, df['home_team'], np.where(delta < -eps_strict, df['away_team'], 'Tie'))
        # Break labeling ties softly using market favorite when within 0.5 points
        tie_mask = (base == 'Tie') & (delta.abs() <= eps_label)
        out = np.where(tie_mask, favored_team, base)
        df[f'pred_q{q}_winner'] = out
    # Half-winner probabilities (if classifiers exist) are kept for display, but winners are
    # enforced to match predicted team half scores for consistency.
    if models.classifiers.get('home_1h_win') is not None:
        p1 = models.classifiers['home_1h_win'].predict_proba(X)[:, 1]
        df['prob_home_1h_win'] = p1
    else:
        df['prob_home_1h_win'] = np.nan
    if models.classifiers.get('home_2h_win') is not None:
        p2 = models.classifiers['home_2h_win'].predict_proba(X)[:, 1]
        # Optional calibration toggle
        try:
            cal_scale = float(os.getenv('NFL_H2_PROB_SCALE', '5.0'))
            cal_strength = float(os.getenv('NFL_H2_PROB_STRENGTH', '0.0'))
        except Exception:
            cal_scale, cal_strength = 5.0, 0.0
        if abs(cal_strength) > 0:
            h2_bias = (df.get('home_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) -
                       df.get('away_h2_minus_h1_avg', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0))
            adj = np.tanh(h2_bias / max(cal_scale, 1e-6)) * cal_strength
            p = np.clip(p2, 1e-6, 1-1e-6)
            logit = np.log(p/(1-p))
            p2 = 1/(1+np.exp(-(logit + adj)))
        df['prob_home_2h_win'] = p2
    else:
        df['prob_home_2h_win'] = np.nan

    # Winners by half are consistent with the numeric predictions
    df['pred_1h_winner'] = np.where(df['pred_home_1h'] >= df['pred_away_1h'], df['home_team'], df['away_team'])
    df['pred_2h_winner'] = np.where(df['pred_home_2h'] >= df['pred_away_2h'], df['home_team'], df['away_team'])

    # half/quarter totals (ensure present if not using dedicated models)
    if 'pred_q1_total' not in df.columns:
        df['pred_q1_total'] = df['pred_home_q1'] + df['pred_away_q1']
        df['pred_q2_total'] = df['pred_home_q2'] + df['pred_away_q2']
        df['pred_q3_total'] = df['pred_home_q3'] + df['pred_away_q3']
        df['pred_q4_total'] = df['pred_home_q4'] + df['pred_away_q4']
        df['pred_1h_total'] = df['pred_home_1h'] + df['pred_away_1h']
        df['pred_2h_total'] = df['pred_home_2h'] + df['pred_away_2h']

    # Final consistency: recompute scores from calibrated halves but preserve the game-level target totals
    df['pred_home_score'] = df['pred_home_1h'] + df['pred_home_2h']
    df['pred_away_score'] = df['pred_away_1h'] + df['pred_away_2h']
    # Keep total equal to original pred_total to avoid drift; nudge by small epsilon if off due to clamping
    tot_err = (df['pred_home_score'] + df['pred_away_score']) - df['pred_total']
    df['pred_home_score'] = df['pred_home_score'] - tot_err/2.0
    df['pred_away_score'] = df['pred_away_score'] - tot_err/2.0
    # Update margin from calibrated scores
    df['pred_margin'] = df['pred_home_score'] - df['pred_away_score']

    # Break exact ties softly using market info when available (directional, small nudge)
    tie_mask = df['pred_margin'].abs() < 0.1
    if tie_mask.any():
        # Convert home spread (negative when favored) to expected home-margin sign
        spread = df.get('spread_home', pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
        market_pm = -spread
        # Fallback from moneylines if spread missing
        mhp = df.get('market_home_prob', pd.Series(0.5, index=df.index)).astype(float).fillna(0.5)
        alt_pm = (mhp - 0.5) * 12.0
        use_pm = np.where(spread != 0, market_pm, alt_pm)
        nudge = np.clip(use_pm * 0.1, -1.5, 1.5)  # cap nudge to +/-1.5 points
        df.loc[tie_mask, 'pred_home_score'] = df.loc[tie_mask, 'pred_home_score'] + (nudge[tie_mask] / 2.0)
        df.loc[tie_mask, 'pred_away_score'] = df.loc[tie_mask, 'pred_away_score'] - (nudge[tie_mask] / 2.0)
        df['pred_margin'] = df['pred_home_score'] - df['pred_away_score']

    # Recompute win prob from calibrated margin if classifier was absent
    if models.classifiers.get('home_win') is None:
        df['prob_home_win'] = 1 / (1 + np.exp(-df['pred_margin'] / 7.5))

    # Cleanup
    df = df.drop(columns=[c for c in ['_target_home_score','_target_away_score'] if c in df.columns])
    return df
