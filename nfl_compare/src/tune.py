import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .data_sources import load_games, load_team_stats, load_lines
from .features import merge_features
from .weather import load_weather_for_games
from .models import train_models, predict as model_predict


def _half_totals_actual(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    q_ok = {'home_q1','home_q2','home_q3','home_q4','away_q1','away_q2','away_q3','away_q4'}.issubset(df.columns)
    if not q_ok:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    h1 = (df['home_q1'].fillna(0)+df['home_q2'].fillna(0)+df['away_q1'].fillna(0)+df['away_q2'].fillna(0)).astype(float)
    h2 = (df['home_q3'].fillna(0)+df['home_q4'].fillna(0)+df['away_q3'].fillna(0)+df['away_q4'].fillna(0)).astype(float)
    return h1, h2


def _quarter_totals_actual(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    for q in [1,2,3,4]:
        hk = f'home_q{q}'; ak = f'away_q{q}'
        if hk in df.columns and ak in df.columns:
            out[f'q{q}_total'] = (df[hk].fillna(0) + df[ak].fillna(0)).astype(float)
        else:
            out[f'q{q}_total'] = pd.Series(np.nan, index=df.index)
    return out


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_pred = pd.to_numeric(y_pred, errors='coerce')
    m = (y_true - y_pred).abs().dropna()
    return float(m.mean()) if not m.empty else float('nan')


def _acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    ok = y_true.notna() & y_pred.notna()
    if not ok.any():
        return float('nan')
    return float((y_true[ok] == y_pred[ok]).mean())


def run_tuning() -> pd.DataFrame:
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    if games.empty:
        raise SystemExit('games.csv is empty; cannot tune')
    # split seasons: train <=2023, validate ==2024
    if 'season' not in games.columns:
        raise SystemExit('games.csv missing season column')
    seasons = sorted(pd.to_numeric(games['season'], errors='coerce').dropna().unique().tolist())
    if 2024 not in seasons:
        raise SystemExit('No 2024 season in games.csv to tune against')

    wx = load_weather_for_games(games)

    def build_feats() -> pd.DataFrame:
        df = merge_features(games, team_stats, lines, wx)
        # completed-only for evaluation targets
        return df

    # parameter grid
    win_grid = [4, 6, 8]
    bias_grid: List[Tuple[float,float]] = [(5.0,0.05),(4.0,0.08),(3.0,0.10)]
    prob_grid: List[Tuple[float,float]] = [(5.0,0.0),(3.0,0.1)]

    results: List[Dict[str, Any]] = []
    for win in win_grid:
        os.environ['NFL_HALF_ROLLING_WINDOW'] = str(win)
        for bscale, bstr in bias_grid:
            os.environ['NFL_H2_BIAS_SCALE'] = str(bscale)
            os.environ['NFL_H2_BIAS_STRENGTH'] = str(bstr)
            for pscale, pstr in prob_grid:
                os.environ['NFL_H2_PROB_SCALE'] = str(pscale)
                os.environ['NFL_H2_PROB_STRENGTH'] = str(pstr)

                df_all = build_feats()
                df_hist = df_all.dropna(subset=['home_score','away_score']).copy()
                df_tr = df_hist[pd.to_numeric(df_hist['season'], errors='coerce') <= 2023]
                df_val = df_hist[pd.to_numeric(df_hist['season'], errors='coerce') == 2024]
                if df_tr.empty or df_val.empty:
                    continue
                models = train_models(df_tr)
                pred = model_predict(models, df_val)

                # Actuals
                pred['home_margin_actual'] = df_val['home_score'] - df_val['away_score']
                pred['total_points_actual'] = df_val['home_score'] + df_val['away_score']
                h1a, h2a = _half_totals_actual(df_val)
                qtrue = _quarter_totals_actual(df_val)

                row = {
                    'win': win,
                    'bias_scale': bscale,
                    'bias_strength': bstr,
                    'prob_scale': pscale,
                    'prob_strength': pstr,
                    'n_val': len(df_val),
                    'mae_margin': _metrics(pred['home_margin_actual'], pred['pred_margin']),
                    'mae_total': _metrics(pred['total_points_actual'], pred['pred_total']),
                    'mae_h1': _metrics(h1a, pred.get('pred_1h_total', np.nan)),
                    'mae_h2': _metrics(h2a, pred.get('pred_2h_total', np.nan)),
                    'mae_q1': _metrics(qtrue['q1_total'], pred.get('pred_q1_total', np.nan)),
                    'mae_q2': _metrics(qtrue['q2_total'], pred.get('pred_q2_total', np.nan)),
                    'mae_q3': _metrics(qtrue['q3_total'], pred.get('pred_q3_total', np.nan)),
                    'mae_q4': _metrics(qtrue['q4_total'], pred.get('pred_q4_total', np.nan)),
                }
                # Winner accuracy
                pred['home_win_actual'] = (pred['home_margin_actual'] > 0).map({True:1, False:0})
                pred['home_win_pred'] = (pred['pred_margin'] > 0).astype(int)
                row['acc_home_win'] = _acc(pred['home_win_actual'], pred['home_win_pred'])

                # 1H/2H predicted winners (from totals if classifier missing)
                def _half_winner_pred(col_home: str, col_away: str) -> pd.Series:
                    return (pred[col_home] >= pred[col_away]).astype(int)
                h1_pred = _half_winner_pred('pred_home_1h', 'pred_away_1h')
                h2_pred = _half_winner_pred('pred_home_2h', 'pred_away_2h')
                # Actual half winners from quarters
                if {'home_q1','home_q2','away_q1','away_q2'}.issubset(df_val.columns):
                    h1_act = ((df_val['home_q1']+df_val['home_q2']) >= (df_val['away_q1']+df_val['away_q2'])).astype(int)
                else:
                    h1_act = pd.Series(np.nan, index=df_val.index)
                if {'home_q3','home_q4','away_q3','away_q4'}.issubset(df_val.columns):
                    h2_act = ((df_val['home_q3']+df_val['home_q4']) >= (df_val['away_q3']+df_val['away_q4'])).astype(int)
                else:
                    h2_act = pd.Series(np.nan, index=df_val.index)
                row['acc_1h'] = _acc(h1_act, h1_pred)
                row['acc_2h'] = _acc(h2_act, h2_pred)

                results.append(row)
                print(f"win={win} bias=({bscale},{bstr}) prob=({pscale},{pstr}) -> mae_total={row['mae_total']:.3f} mae_h2={row['mae_h2']:.3f} acc_2h={row['acc_2h']:.3f}")

    out = pd.DataFrame(results)
    out_fp = Path(__file__).resolve().parents[1] / 'data' / 'tuning_2024.csv'
    out.to_csv(out_fp, index=False)
    print(f"Tuning report written to {out_fp} with {len(out)} rows")
    if not out.empty:
        # pick best by weighted score (example: prioritize game total + 2H MAE + 2H acc)
        score = (
            out['mae_total'].rank(ascending=True) +
            out['mae_h2'].rank(ascending=True) +
            (1 - out['acc_2h'].fillna(0)).rank(ascending=True)
        )
        best_idx = int(score.idxmin())
        best = out.loc[best_idx]
        print('Suggested params ->', {
            'NFL_HALF_ROLLING_WINDOW': int(best['win']),
            'NFL_H2_BIAS_SCALE': float(best['bias_scale']),
            'NFL_H2_BIAS_STRENGTH': float(best['bias_strength']),
            'NFL_H2_PROB_SCALE': float(best['prob_scale']),
            'NFL_H2_PROB_STRENGTH': float(best['prob_strength']),
        })
    return out


def main():
    run_tuning()


if __name__ == '__main__':
    main()
