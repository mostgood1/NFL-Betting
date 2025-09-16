from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

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
}

FEATURES = [
    'elo_diff', 'off_epa_diff', 'def_epa_diff', 'pace_secs_play_diff',
    'pass_rate_diff', 'rush_rate_diff', 'qb_adj_diff', 'sos_diff',
    'spread_home', 'total'
]

@dataclass
class TrainedModels:
    # Use loose typing so either xgboost or sklearn estimators can be stored
    regressors: Dict[str, object]
    classifiers: Dict[str, Optional[object]]


def _build_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df['home_win'] = (df['home_margin'] > 0).astype(int)
    X = df[FEATURES].fillna(0)
    y_margin = df['home_margin']
    y_total = df['total_points']
    y_homewin = df['home_win']
    return X, y_margin, y_total, y_homewin


def train_models(df: pd.DataFrame) -> TrainedModels:
    X, y_margin, y_total, y_homewin = _build_frame(df)
    # Single split shared across targets to keep dimensions aligned
    X_train, X_val, ym_train, ym_val, yt_train, yt_val, yh_train, yh_val = train_test_split(
        X, y_margin, y_total, y_homewin, test_size=0.2, random_state=42
    )

    # Choose estimators based on availability
    if HAVE_XGB:
        reg_margin = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42)  # type: ignore
        reg_total = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42)  # type: ignore
        reg_margin.fit(X_train, ym_train)
        reg_total.fit(X_train, yt_train)
        clf_home: Optional[object] = None
        if len(np.unique(yh_train)) > 1:
            clf_home = XGBClassifier(  # type: ignore
                n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                random_state=42, objective='binary:logistic', eval_metric='logloss'
            )
            clf_home.fit(X_train, yh_train)
    else:
        # Sklearn fallbacks: relatively light and available across platforms
        reg_margin = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        reg_total = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        reg_margin.fit(X_train, ym_train)
        reg_total.fit(X_train, yt_train)
        clf_home = None
        if len(np.unique(yh_train)) > 1:
            # Use liblinear for small-ish datasets; ensure probability output
            clf_home = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf_home.fit(X_train, yh_train)

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

    models = TrainedModels(
        regressors={'home_margin': reg_margin, 'total_points': reg_total},
        classifiers={'home_win': clf_home}
    )
    return models


def predict(models: TrainedModels, df_future: pd.DataFrame) -> pd.DataFrame:
    df = df_future.copy()
    X = df[FEATURES].fillna(0)
    df['pred_margin'] = models.regressors['home_margin'].predict(X)
    df['pred_total'] = models.regressors['total_points'].predict(X)
    if models.classifiers.get('home_win') is not None:
        df['prob_home_win'] = models.classifiers['home_win'].predict_proba(X)[:, 1]
    else:
        # Fallback: convert predicted margin to probability via logistic link
        # tuned slope roughly for NFL margins
        df['prob_home_win'] = 1 / (1 + np.exp(-df['pred_margin'] / 7.5))

    # implied team scores using spread/total-like system
    df['pred_home_score'] = np.maximum(0, (df['pred_total'] + df['pred_margin']) / 2)
    df['pred_away_score'] = np.maximum(0, df['pred_total'] - df['pred_home_score'])

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
