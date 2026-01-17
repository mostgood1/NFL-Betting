from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Union, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import json
import os
from pathlib import Path

# Try to import XGBoost; gracefully fall back to sklearn if unavailable or broken
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    HAVE_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore
    XGBClassifier = None  # type: ignore
    HAVE_XGB = False

TARGETS = {
    'home_margin': 'reg',
    'total_points': 'reg',
    'home_win': 'clf',
    'home_cover': 'clf',
    'over_total': 'clf',
}

FEATURES = [
    # Core diffs
    'elo_diff', 'off_epa_diff', 'def_epa_diff', 'pace_secs_play_diff',
    'pass_rate_diff', 'rush_rate_diff', 'qb_adj_diff', 'sos_diff',
    # EMA diffs (season-to-date; included when available)
    'off_epa_ema_diff','def_epa_ema_diff','pace_secs_play_ema_diff','pass_rate_ema_diff','rush_rate_ema_diff',
    'off_sack_rate_ema_diff','def_sack_rate_ema_diff','off_rz_pass_rate_ema_diff','def_rz_pass_rate_ema_diff',
    # Deeper team stats
    'off_sack_rate_diff', 'def_sack_rate_diff',
    'off_rz_pass_rate_diff', 'def_rz_pass_rate_diff',
    'def_wr_share_mult_diff', 'def_te_share_mult_diff', 'def_rb_share_mult_diff',
    'def_wr_ypt_mult_diff', 'def_te_ypt_mult_diff', 'def_rb_ypt_mult_diff',
    # Team ratings diffs (EMA-based priors; attached when available)
    'off_ppg_diff', 'def_ppg_diff', 'net_margin_diff',
    # Pressure aggregates (avg of defensive sack rates; optional)
    'def_pressure_avg_ema','def_pressure_avg',
    # Market anchors
    'spread_home', 'total',
    # Injury diffs (new; safe to ignore when models expect old shape)
    'inj_qb_out_diff', 'inj_wr1_out_diff', 'inj_te1_out_diff', 'inj_rb1_out_diff',
    'inj_wr_top2_out_diff', 'inj_starters_out_diff',
    # Playoff/context features
    'is_postseason', 'neutral_site_flag', 'rest_days_diff',
    # Weather features (numeric)
    'wx_temp_f','wx_wind_mph','wx_precip_pct','roof_closed_flag','roof_open_flag','wind_open'
    ,
    # Phase A diffs (optional; used when present)
    'ppd_diff','td_per_drive_diff','fg_per_drive_diff','avg_start_fp_diff','yards_per_drive_diff','seconds_per_drive_diff','drives_diff',
    'rzd_off_eff_diff','rzd_def_eff_diff','rzd_off_td_rate_diff','rzd_def_td_rate_diff',
    'explosive_pass_rate_diff','explosive_run_rate_diff',
    'penalty_rate_diff','turnover_adj_rate_diff',
    'fg_acc_diff','punt_epa_diff','kick_return_epa_diff','touchback_rate_diff',
    # Officiating crew and NOAA additions
    'crew_penalty_rate','crew_dpi_rate','crew_pace_adj',
    'wx_gust_mph','wx_dew_point_f',
    # Phase A totals heuristic delta
    'phase_a_total_delta'
]


def _select_X(df: pd.DataFrame, model: object, fallback_features: list[str]) -> pd.DataFrame:
    """Select feature matrix aligned to the model.
    - Prefer model.feature_names_in_ if available.
    - Else use fallback_features present in df.
    - If model.n_features_in_ exists and count mismatches, trim or zero-pad as needed.
    """
    Xdf = df.copy()
    # Prefer named features stored in the estimator
    names = getattr(model, 'feature_names_in_', None)
    if names is not None and len(names) > 0:
        # Ensure all required columns exist; fill missing with 0
        for c in names:
            if c not in Xdf.columns:
                Xdf[c] = 0.0
        return Xdf[names].fillna(0)
    # Fallback: use provided feature order
    cols = [f for f in fallback_features if f in Xdf.columns]
    n_expected = getattr(model, 'n_features_in_', None)
    if isinstance(n_expected, int):
        if len(cols) >= n_expected:
            cols = cols[:n_expected]
        else:
            # pad with synthetic zero columns
            need = n_expected - len(cols)
            for i in range(need):
                pad = f'__pad_{i}'
                Xdf[pad] = 0.0
                cols.append(pad)
    return Xdf[cols].fillna(0)

@dataclass
class TrainedModels:
    """Container for trained estimators.
    Values can be a single estimator or a list of estimators (ensemble bagging).
    """
    regressors: Dict[str, Union[object, List[object]]]
    classifiers: Dict[str, Optional[Union[object, List[object]]]]
    calibrators: Dict[str, Optional[Any]]
    use_residuals_margin: bool = False
    use_residuals_total: bool = False


_SIGMA_CACHE: Optional[Dict[str, float]] = None

def _load_sigma_calibration() -> Dict[str, float]:
    global _SIGMA_CACHE
    if _SIGMA_CACHE is not None:
        return _SIGMA_CACHE
    # Default fallbacks if file not present
    out = {"ats_sigma": 9.0, "total_sigma": 10.0}
    try:
        base = Path(__file__).resolve().parents[1] / 'data' / 'sigma_calibration.json'
        if base.exists():
            with open(base, 'r', encoding='utf-8') as f:
                js = json.load(f)
            for k in ("ats_sigma","total_sigma"):
                v = js.get(k)
                if v is not None:
                    out[k] = float(v)
    except Exception:
        pass
    _SIGMA_CACHE = out
    return out

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # standard normal CDF via error function
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _build_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df['home_win'] = (df['home_margin'] > 0).astype(int)
    # Use all FEATURES (DataFrame to capture names for feature_names_in_)
    X = df[[c for c in FEATURES if c in df.columns]].fillna(0)
    # Market anchors can be partially missing historically. Dropping them entirely makes the
    # model blind to market information (which you explicitly want as a guide).
    # Instead, keep them and impute missing values to robust medians, plus missingness flags.
    try:
        if 'total' in X.columns:
            total_raw = pd.to_numeric(df.get('total'), errors='coerce')
            total_ok = total_raw.between(20, 70)
            total_med = float(total_raw[total_ok].median()) if total_ok.any() else 45.0
            X['total_missing'] = (~total_ok).astype(int)
            X['total'] = total_raw.where(total_ok, total_med)
        if 'spread_home' in X.columns:
            spread_raw = pd.to_numeric(df.get('spread_home'), errors='coerce')
            spread_ok = spread_raw.notna() & spread_raw.abs().le(30)
            spread_med = float(spread_raw[spread_ok].median()) if spread_ok.any() else 0.0
            X['spread_home_missing'] = (~spread_ok).astype(int)
            X['spread_home'] = spread_raw.where(spread_ok, spread_med)
    except Exception:
        pass
    y_margin = df['home_margin']
    y_total = df['total_points']
    y_homewin = df['home_win']
    return X, y_margin, y_total, y_homewin


def train_models(df: pd.DataFrame) -> TrainedModels:
    X, y_margin, y_total, y_homewin = _build_frame(df)
    # Raw market anchors (do NOT fill missing with 0). Used for residual modeling.
    spread_line_raw = pd.to_numeric(df.get('spread_home'), errors='coerce') if 'spread_home' in df.columns else pd.Series(index=df.index, dtype=float)
    total_line_raw = pd.to_numeric(df.get('total'), errors='coerce') if 'total' in df.columns else pd.Series(index=df.index, dtype=float)
    # Single split shared across targets to keep dimensions aligned
    X_train, X_val, ym_train, ym_val, yt_train, yt_val, yh_train, yh_val, spread_train, spread_val, total_train, total_val = train_test_split(
        X, y_margin, y_total, y_homewin, spread_line_raw, total_line_raw, test_size=0.2, random_state=42
    )
    # Optional ATS / Over targets computed from original df
    try:
        spread_full = pd.to_numeric(df.get('spread_home'), errors='coerce') if 'spread_home' in df.columns else None
        y_cover_full = ((df['home_margin'] + spread_full) > 0).astype(int) if spread_full is not None else None
    except Exception:
        y_cover_full = None
    try:
        total_full = pd.to_numeric(df.get('total'), errors='coerce') if 'total' in df.columns else None
        y_over_full = (df['total_points'] > total_full).astype(int) if total_full is not None else None
    except Exception:
        y_over_full = None
    if y_cover_full is not None and y_over_full is not None:
        # Use the same split indices as the main train/val by re-running with same random_state and shapes
        X_train_idx, X_val_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        yc_train = y_cover_full.iloc[X_train_idx]
        yc_val = y_cover_full.iloc[X_val_idx]
        yo_train = y_over_full.iloc[X_train_idx]
        yo_val = y_over_full.iloc[X_val_idx]
        X_val_clf = X.iloc[X_val_idx].copy()
    else:
        yc_train = yc_val = yo_train = yo_val = None
        X_val_clf = None
    # Residual modeling vs market lines when available.
    # Guard: if lines are missing historically, treating missing as 0 yields residuals ~+45,
    # and then adding back a real market total at inference doubles totals.
    use_residuals_margin = False
    use_residuals_total = False
    ym_train_res, ym_val_res = ym_train, ym_val
    yt_train_res, yt_val_res = yt_train, yt_val
    try:
        spread_ok_train = spread_train.notna() & spread_train.abs().le(30)
        spread_ok_val = spread_val.notna() & spread_val.abs().le(30)
        use_residuals_margin = bool(spread_ok_train.mean() >= 0.80 and spread_ok_val.mean() >= 0.80)
    except Exception:
        use_residuals_margin = False
    try:
        total_ok_train = total_train.between(20, 70)
        total_ok_val = total_val.between(20, 70)
        use_residuals_total = bool(total_ok_train.mean() >= 0.80 and total_ok_val.mean() >= 0.80)
    except Exception:
        use_residuals_total = False

    if use_residuals_margin:
        try:
            spread_med = float(pd.to_numeric(spread_train, errors='coerce').dropna().median()) if spread_train.notna().any() else 0.0
            ym_train_res = ym_train - pd.to_numeric(spread_train, errors='coerce').fillna(spread_med)
            ym_val_res = ym_val - pd.to_numeric(spread_val, errors='coerce').fillna(spread_med)
        except Exception:
            ym_train_res, ym_val_res = ym_train, ym_val
            use_residuals_margin = False

    if use_residuals_total:
        try:
            total_med = float(pd.to_numeric(total_train, errors='coerce').dropna().median()) if total_train.notna().any() else 0.0
            yt_train_res = yt_train - pd.to_numeric(total_train, errors='coerce').fillna(total_med)
            yt_val_res = yt_val - pd.to_numeric(total_val, errors='coerce').fillna(total_med)
        except Exception:
            yt_train_res, yt_val_res = yt_train, yt_val
            use_residuals_total = False

    # Optional simple bagging: train multiple seeds and average at inference
    try:
        n_ens = int(float(__import__('os').environ.get('GAME_MODEL_N_ENSEMBLE', '1')))
    except Exception:
        n_ens = 1
    n_ens = max(1, min(8, n_ens))

    # Choose estimators based on availability
    if HAVE_XGB:
        regs_margin: List[object] = []
        regs_total: List[object] = []
        clfs_home: List[object] = []
        clfs_cover: List[object] = []
        clfs_over: List[object] = []
        cal_cover: Optional[Any] = None
        cal_over: Optional[Any] = None
        seeds = [42 + i*17 for i in range(n_ens)]
        # Class balance helpers for ATS and Over
        def _spw(y: Optional[pd.Series]) -> float:
            try:
                if y is None:
                    return 1.0
                y_clean = pd.to_numeric(y, errors='coerce').dropna().astype(int)
                pos = float(y_clean.sum())
                neg = float(len(y_clean) - y_clean.sum())
                return neg / max(pos, 1.0)
            except Exception:
                return 1.0
        spw_cover = _spw(yc_train)
        spw_over = _spw(yo_train)
        for rs in seeds:
            rm = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=rs)  # type: ignore
            rt = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=rs)  # type: ignore
            rm.fit(X_train, ym_train_res)
            rt.fit(X_train, yt_train_res)
            regs_margin.append(rm)
            regs_total.append(rt)
            if len(np.unique(yh_train)) > 1:
                ch = XGBClassifier(  # type: ignore
                    n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                    random_state=rs, objective='binary:logistic', eval_metric='logloss'
                )
                ch.fit(X_train, yh_train)
                clfs_home.append(ch)
            # Train ATS/Over classifiers if targets present
            if yc_train is not None and len(np.unique(yc_train.dropna())) > 1:
                cc = XGBClassifier(  # type: ignore
                    n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                    random_state=rs, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=float(spw_cover)
                )
                cc.fit(X_train, yc_train)
                clfs_cover.append(cc)
            if yo_train is not None and len(np.unique(yo_train.dropna())) > 1:
                co = XGBClassifier(  # type: ignore
                    n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                    random_state=rs, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=float(spw_over)
                )
                co.fit(X_train, yo_train)
                clfs_over.append(co)
        # Fit isotonic calibrators on validation if we have targets
        try:
            if X_val_clf is not None and yc_val is not None and clfs_cover:
                # Use first estimator to produce validation probs for calibration
                raw = clfs_cover[0].predict_proba(X_val_clf)[:, 1]
                cal_cover = IsotonicRegression(out_of_bounds='clip').fit(raw, yc_val)
            else:
                cal_cover = None
        except Exception:
            cal_cover = None
        try:
            if X_val_clf is not None and yo_val is not None and clfs_over:
                raw = clfs_over[0].predict_proba(X_val_clf)[:, 1]
                cal_over = IsotonicRegression(out_of_bounds='clip').fit(raw, yo_val)
            else:
                cal_over = None
        except Exception:
            cal_over = None
        reg_margin = regs_margin if n_ens > 1 else regs_margin[0]
        reg_total = regs_total if n_ens > 1 else regs_total[0]
        clf_home: Optional[Union[object, List[object]]] = (clfs_home if (n_ens > 1 and clfs_home) else (clfs_home[0] if clfs_home else None))
        clf_cover: Optional[Union[object, List[object]]] = (clfs_cover if (n_ens > 1 and clfs_cover) else (clfs_cover[0] if clfs_cover else None)) if clfs_cover else None
        clf_over: Optional[Union[object, List[object]]] = (clfs_over if (n_ens > 1 and clfs_over) else (clfs_over[0] if clfs_over else None)) if clfs_over else None
    else:
        # Sklearn fallbacks: relatively light and available across platforms
        regs_margin = []
        regs_total = []
        clfs_home: List[object] = []
        clfs_cover: List[object] = []
        clfs_over: List[object] = []
        seeds = [42 + i*17 for i in range(n_ens)]
        cal_cover: Optional[Any] = None
        cal_over: Optional[Any] = None
        for rs in seeds:
            rm = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=rs, n_jobs=-1)
            rt = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=rs, n_jobs=-1)
            rm.fit(X_train, ym_train_res)
            rt.fit(X_train, yt_train_res)
            regs_margin.append(rm)
            regs_total.append(rt)
            if len(np.unique(yh_train)) > 1:
                ch = LogisticRegression(max_iter=1000, solver='lbfgs')
                ch.fit(X_train, yh_train)
                clfs_home.append(ch)
            if yc_train is not None and len(np.unique(yc_train.dropna())) > 1:
                cc = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
                cc.fit(X_train, yc_train)
                clfs_cover.append(cc)
            if yo_train is not None and len(np.unique(yo_train.dropna())) > 1:
                co = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
                co.fit(X_train, yo_train)
                clfs_over.append(co)
        # Fit isotonic calibrators on validation if we have targets
        try:
            if X_val_clf is not None and yc_val is not None and clfs_cover:
                raw = clfs_cover[0].predict_proba(X_val_clf)[:, 1]
                cal_cover = IsotonicRegression(out_of_bounds='clip').fit(raw, yc_val)
            else:
                cal_cover = None
        except Exception:
            cal_cover = None
        try:
            if X_val_clf is not None and yo_val is not None and clfs_over:
                raw = clfs_over[0].predict_proba(X_val_clf)[:, 1]
                cal_over = IsotonicRegression(out_of_bounds='clip').fit(raw, yo_val)
            else:
                cal_over = None
        except Exception:
            cal_over = None
        reg_margin = regs_margin if n_ens > 1 else regs_margin[0]
        reg_total = regs_total if n_ens > 1 else regs_total[0]
        clf_home = (clfs_home if (n_ens > 1 and clfs_home) else (clfs_home[0] if clfs_home else None))
        clf_cover = (clfs_cover if (n_ens > 1 and clfs_cover) else (clfs_cover[0] if clfs_cover else None)) if clfs_cover else None
        clf_over = (clfs_over if (n_ens > 1 and clfs_over) else (clfs_over[0] if clfs_over else None)) if clfs_over else None

    # basic metrics (safe)
    def _avg_pred(est_or_list, Xn):
        try:
            if isinstance(est_or_list, list):
                preds = [e.predict(Xn) for e in est_or_list]
                return np.mean(preds, axis=0)
            return est_or_list.predict(Xn)
        except Exception:
            return np.zeros(len(Xn))

    ym_pred_res = _avg_pred(reg_margin, X_val)
    yt_pred_res = _avg_pred(reg_total, X_val)
    # Reconstruct absolute predictions if residuals were used
    ym_pred = ym_pred_res
    yt_pred = yt_pred_res
    if use_residuals_margin:
        try:
            spread_med = float(pd.to_numeric(spread_train, errors='coerce').dropna().median()) if spread_train.notna().any() else 0.0
            ym_pred = ym_pred_res + pd.to_numeric(spread_val, errors='coerce').fillna(spread_med)
        except Exception:
            pass
    if use_residuals_total:
        try:
            total_med = float(pd.to_numeric(total_train, errors='coerce').dropna().median()) if total_train.notna().any() else 0.0
            yt_pred = yt_pred_res + pd.to_numeric(total_val, errors='coerce').fillna(total_med)
        except Exception:
            pass
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
            if isinstance(clf_home, list):
                probs = [c.predict_proba(X_val)[:, 1] for c in clf_home]
                yh_pred = np.mean(probs, axis=0)
            else:
                yh_pred = clf_home.predict_proba(X_val)[:, 1]
            auc_home = float(roc_auc_score(yh_val, yh_pred))
        except Exception:
            auc_home = None
    else:
        auc_home = None

    # You could persist metrics somewhere if desired
    _ = {'MAE_margin': mae_margin, 'MAE_total': mae_total, 'AUC_home_win': auc_home}

    models = TrainedModels(
        regressors={'home_margin': reg_margin, 'total_points': reg_total},
        classifiers={'home_win': clf_home, 'home_cover': clf_cover, 'over_total': clf_over},
        calibrators={'home_cover': cal_cover, 'over_total': cal_over},
        use_residuals_margin=bool(use_residuals_margin),
        use_residuals_total=bool(use_residuals_total),
    )
    return models


def predict(models: TrainedModels, df_future: pd.DataFrame) -> pd.DataFrame:
    df = df_future.copy()

    # If the model was trained with market-anchor imputation / missingness flags,
    # reproduce the same transformations at inference time.
    try:
        req_cols: set[str] = set()
        for key in ('home_margin', 'total_points'):
            est = models.regressors.get(key)
            est0 = est[0] if isinstance(est, list) else est
            names = getattr(est0, 'feature_names_in_', None)
            if names is not None:
                req_cols |= set([str(c) for c in names])
        for key in ('home_win', 'home_cover', 'over_total'):
            est = models.classifiers.get(key) if hasattr(models, 'classifiers') else None
            est0 = est[0] if isinstance(est, list) else est
            names = getattr(est0, 'feature_names_in_', None) if est0 is not None else None
            if names is not None:
                req_cols |= set([str(c) for c in names])

        if ('total' in req_cols) or ('total_missing' in req_cols):
            total_raw = pd.to_numeric(df.get('total'), errors='coerce')
            # Prefer close_total if it exists and total is missing
            try:
                if 'close_total' in df.columns:
                    total_raw = total_raw.fillna(pd.to_numeric(df.get('close_total'), errors='coerce'))
            except Exception:
                pass
            total_ok = total_raw.between(20, 70)
            total_med = float(total_raw[total_ok].median()) if total_ok.any() else 45.0
            df['total_missing'] = (~total_ok).astype(int)
            df['total'] = total_raw.where(total_ok, total_med)

        if ('spread_home' in req_cols) or ('spread_home_missing' in req_cols):
            spread_raw = pd.to_numeric(df.get('spread_home'), errors='coerce')
            # Prefer close_spread_home if it exists and spread_home is missing
            try:
                if 'close_spread_home' in df.columns:
                    spread_raw = spread_raw.fillna(pd.to_numeric(df.get('close_spread_home'), errors='coerce'))
            except Exception:
                pass
            spread_ok = spread_raw.notna() & spread_raw.abs().le(30)
            spread_med = float(spread_raw[spread_ok].median()) if spread_ok.any() else 0.0
            df['spread_home_missing'] = (~spread_ok).astype(int)
            df['spread_home'] = spread_raw.where(spread_ok, spread_med)
    except Exception:
        pass
    # Build X aligned to the margin regressor (assumes all models trained with same features)
    reg_m = models.regressors['home_margin']
    Xm = _select_X(df, reg_m[0] if isinstance(reg_m, list) else reg_m, FEATURES)
    if isinstance(reg_m, list):
        preds = [m.predict(Xm) for m in reg_m]
        df['pred_margin'] = np.mean(preds, axis=0)
    else:
        df['pred_margin'] = reg_m.predict(Xm)
    # Total regressor
    reg_t = models.regressors['total_points']
    Xt = _select_X(df, reg_t[0] if isinstance(reg_t, list) else reg_t, FEATURES)
    if isinstance(reg_t, list):
        preds_t = [m.predict(Xt) for m in reg_t]
        df['pred_total'] = np.mean(preds_t, axis=0)
    else:
        df['pred_total'] = reg_t.predict(Xt)

    # If residual modeling was used in training, reconstruct absolute predictions by adding
    # back market anchors. Do NOT do this unconditionally: if the regressor was trained on
    # absolute targets, adding anchors here will double-count market lines.
    if getattr(models, 'use_residuals_margin', False):
        try:
            anchor = pd.to_numeric(df.get('spread_home'), errors='coerce')
            if 'close_spread_home' in df.columns:
                anchor = anchor.fillna(pd.to_numeric(df.get('close_spread_home'), errors='coerce'))
            df['pred_margin'] = pd.to_numeric(df.get('pred_margin'), errors='coerce') + anchor.fillna(0)
        except Exception:
            pass
    if getattr(models, 'use_residuals_total', False):
        try:
            anchor = pd.to_numeric(df.get('total'), errors='coerce')
            if 'close_total' in df.columns:
                anchor = anchor.fillna(pd.to_numeric(df.get('close_total'), errors='coerce'))
            df['pred_total'] = pd.to_numeric(df.get('pred_total'), errors='coerce') + anchor.fillna(0)
        except Exception:
            pass
    # Apply Phase A totals delta if present
    try:
        if 'phase_a_total_delta' in df.columns:
            df['pred_total'] = pd.to_numeric(df.get('pred_total'), errors='coerce') + pd.to_numeric(df.get('phase_a_total_delta'), errors='coerce').fillna(0)
    except Exception:
        pass
    # Classifier
    clf = models.classifiers.get('home_win')
    if clf is not None:
        Xc = _select_X(df, clf[0] if isinstance(clf, list) else clf, FEATURES)
        if isinstance(clf, list):
            probs = [c.predict_proba(Xc)[:, 1] for c in clf]
            df['prob_home_win'] = np.mean(probs, axis=0)
        else:
            df['prob_home_win'] = clf.predict_proba(Xc)[:, 1]
    else:
        # Fallback: convert predicted margin to probability via logistic link
        # tuned slope roughly for NFL margins
        df['prob_home_win'] = 1 / (1 + np.exp(-df['pred_margin'] / 7.5))

    # ATS and Over classifiers (optional)
    clf_cov = models.classifiers.get('home_cover')
    if clf_cov is not None:
        Xcov = _select_X(df, clf_cov[0] if isinstance(clf_cov, list) else clf_cov, FEATURES)
        try:
            if isinstance(clf_cov, list):
                probs = [c.predict_proba(Xcov)[:, 1] for c in clf_cov]
                df['prob_home_cover'] = np.mean(probs, axis=0)
            else:
                df['prob_home_cover'] = clf_cov.predict_proba(Xcov)[:, 1]
        except Exception:
            pass
        # Apply isotonic calibration if available
        try:
            cal = models.calibrators.get('home_cover') if hasattr(models, 'calibrators') else None
            if cal is not None and 'prob_home_cover' in df.columns:
                phc = pd.to_numeric(df['prob_home_cover'], errors='coerce').astype(float)
                df['prob_home_cover'] = np.clip(cal.predict(phc.values), 0.0, 1.0)
        except Exception:
            pass
    clf_over = models.classifiers.get('over_total')
    if clf_over is not None:
        Xovr = _select_X(df, clf_over[0] if isinstance(clf_over, list) else clf_over, FEATURES)
        try:
            if isinstance(clf_over, list):
                probs = [c.predict_proba(Xovr)[:, 1] for c in clf_over]
                df['prob_over_total'] = np.mean(probs, axis=0)
            else:
                df['prob_over_total'] = clf_over.predict_proba(Xovr)[:, 1]
        except Exception:
            pass
        # Apply isotonic calibration if available
        try:
            cal = models.calibrators.get('over_total') if hasattr(models, 'calibrators') else None
            if cal is not None and 'prob_over_total' in df.columns:
                pov = pd.to_numeric(df['prob_over_total'], errors='coerce').astype(float)
                df['prob_over_total'] = np.clip(cal.predict(pov.values), 0.0, 1.0)
        except Exception:
            pass

    # Optional parametric probability using calibrated sigma, blended with classifier
    try:
        blend_p_ats = float(os.environ.get('GAME_PROB_BLEND_PARAM_ATS', '0.20'))
    except Exception:
        blend_p_ats = 0.20
    try:
        blend_p_tot = float(os.environ.get('GAME_PROB_BLEND_PARAM_TOTAL', '0.20'))
    except Exception:
        blend_p_tot = 0.20
    try:
        sig = _load_sigma_calibration()
    except Exception:
        sig = {"ats_sigma": 9.0, "total_sigma": 10.0}
    # ATS parametric prob: P(margin + spread_home > 0)
    if blend_p_ats and ('spread_home' in df.columns):
        try:
            m_pred = pd.to_numeric(df.get('pred_margin'), errors='coerce')
            spr = pd.to_numeric(df.get('spread_home'), errors='coerce')
            z = (m_pred + spr) / max(1e-6, float(sig.get('ats_sigma', 9.0)))
            p_param = _norm_cdf(z.values)
            if 'prob_home_cover' in df.columns:
                df['prob_home_cover'] = (1.0 - blend_p_ats) * pd.to_numeric(df['prob_home_cover'], errors='coerce').astype(float) + blend_p_ats * p_param
            else:
                df['prob_home_cover'] = p_param
        except Exception:
            pass
    # Totals parametric prob: P(total_points > market_total)
    if blend_p_tot and ('total' in df.columns):
        try:
            t_pred = pd.to_numeric(df.get('pred_total'), errors='coerce')
            t_mkt = pd.to_numeric(df.get('total'), errors='coerce')
            zt = (t_pred - t_mkt) / max(1e-6, float(sig.get('total_sigma', 10.0)))
            p_param_t = _norm_cdf(zt.values)
            if 'prob_over_total' in df.columns:
                df['prob_over_total'] = (1.0 - blend_p_tot) * pd.to_numeric(df['prob_over_total'], errors='coerce').astype(float) + blend_p_tot * p_param_t
            else:
                df['prob_over_total'] = p_param_t
        except Exception:
            pass

    # implied team scores using spread/total-like system
    df['pred_home_score'] = np.maximum(0, (df['pred_total'] + df['pred_margin']) / 2)
    df['pred_away_score'] = np.maximum(0, df['pred_total'] - df['pred_home_score'])
    # Backward-compatible aliases expected by card logic
    df['pred_home_points'] = df['pred_home_score']
    df['pred_away_points'] = df['pred_away_score']

    # quarters/halves heuristic split (pace + pass rate influence)
    pace = df.get('pace_secs_play_diff', pd.Series(0, index=df.index)).fillna(0)
    fast = (pace < 0).astype(float)  # faster home than away
    q1_share = 0.235 + 0.02*fast
    q2_share = 0.265
    q3_share = 0.245
    q4_share = 0.255 - 0.02*fast

    for side in ['home','away']:
        tot = df[f'pred_{side}_score']
        df[f'pred_{side}_q1'] = tot * q1_share
        df[f'pred_{side}_q2'] = tot * q2_share
        df[f'pred_{side}_q3'] = tot * q3_share
        df[f'pred_{side}_q4'] = tot * q4_share
        df[f'pred_{side}_1h'] = df[f'pred_{side}_q1'] + df[f'pred_{side}_q2']
        df[f'pred_{side}_2h'] = df[f'pred_{side}_q3'] + df[f'pred_{side}_q4']

    return df
