import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models, predict as model_predict


def walkforward_eval(min_weeks_train: int = 4) -> Dict[str, float]:
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    if games.empty:
        raise SystemExit("No historical games found (games.csv).")

    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = None
    df = merge_features(games, team_stats, lines, wx)
    # Only rows with actual results for evaluation
    df_hist = df.dropna(subset=["home_score", "away_score"]).copy()
    if df_hist.empty:
        raise SystemExit("No rows with actual scores; can't evaluate.")

    mae_m_list: List[float] = []
    mae_t_list: List[float] = []
    rmse_m_list: List[float] = []
    rmse_t_list: List[float] = []
    # baselines (market)
    mae_m_mkt: List[float] = []
    mae_t_mkt: List[float] = []
    rmse_m_mkt: List[float] = []
    rmse_t_mkt: List[float] = []
    auc_mkt_list: List[float] = []
    brier_mkt_list: List[float] = []
    prob_and_y: List[Tuple[float, int]] = []

    # Evaluate per season to respect chronology
    seasons = sorted([int(s) for s in df_hist["season"].dropna().unique().tolist()])
    total_preds = 0
    for season in seasons:
        ds = df_hist[df_hist["season"] == season].copy()
        # Weeks as integers
        weeks = sorted([int(w) for w in ds["week"].dropna().unique().tolist()])
        for w in weeks:
            # Require some history before testing week w
            prior = ds[ds["week"].astype(int) < int(w)]
            test = ds[ds["week"].astype(int) == int(w)]
            if len(prior) < max(min_weeks_train, 8):
                continue
            if test.empty:
                continue
            models = train_models(prior)
            pred = model_predict(models, test)
            # Align with actuals using distinct column names to avoid suffix conflicts
            actuals = (
                test[["game_id", "home_score", "away_score"]]
                .rename(columns={"home_score": "actual_home", "away_score": "actual_away"})
            )
            pred = pred.merge(actuals, on="game_id", how="left")
            pred["actual_margin"] = pred["actual_home"] - pred["actual_away"]
            pred["actual_total"] = pred[["actual_home", "actual_away"]].sum(axis=1)
            # Drop rows lacking predictions or actuals
            keep = pred[["actual_margin","actual_total","pred_margin","pred_total"]].replace([np.inf,-np.inf], np.nan).dropna().index
            if len(keep) == 0:
                continue
            pred = pred.loc[keep].copy()
            y = (pred["actual_margin"] > 0).astype(int)
            # Metrics this week (model)
            mae_m_list.append(mean_absolute_error(pred["actual_margin"], pred["pred_margin"]))
            mae_t_list.append(mean_absolute_error(pred["actual_total"], pred["pred_total"]))
            rmse_m_list.append(mean_squared_error(pred["actual_margin"], pred["pred_margin"]) ** 0.5)
            rmse_t_list.append(mean_squared_error(pred["actual_total"], pred["pred_total"]) ** 0.5)
            if "prob_home_win" in pred.columns:
                probs = pd.to_numeric(pred["prob_home_win"], errors="coerce")
                for p_i, y_i in zip(probs.dropna().tolist(), y.loc[probs.dropna().index].tolist()):
                    prob_and_y.append((float(p_i), int(y_i)))
            # Baseline metrics using market
            # Margin baseline: market-implied home margin = -spread_home; prefer close_spread_home
            spread_use = pred.get("close_spread_home", pred.get("spread_home"))
            spread_use = pd.to_numeric(spread_use, errors="coerce")
            idx_m = spread_use.dropna().index.intersection(pred.index)
            if len(idx_m) > 0:
                mkt_margin = -spread_use.loc[idx_m]
                mae_m_mkt.append(mean_absolute_error(pred.loc[idx_m, "actual_margin"], mkt_margin))
                rmse_m_mkt.append(mean_squared_error(pred.loc[idx_m, "actual_margin"], mkt_margin) ** 0.5)
            # Total baseline: market total; prefer close_total
            total_use = pred.get("close_total", pred.get("total"))
            total_use = pd.to_numeric(total_use, errors="coerce")
            idx_t = total_use.dropna().index.intersection(pred.index)
            if len(idx_t) > 0:
                mae_t_mkt.append(mean_absolute_error(pred.loc[idx_t, "actual_total"], total_use.loc[idx_t]))
                rmse_t_mkt.append(mean_squared_error(pred.loc[idx_t, "actual_total"], total_use.loc[idx_t]) ** 0.5)
            # Home-win probability baseline: market_home_prob if available
            if "market_home_prob" in pred.columns:
                mkt_probs = pd.to_numeric(pred["market_home_prob"], errors="coerce").clip(1e-6, 1-1e-6)
                mask = mkt_probs.notna()
                if mask.any() and y.loc[mask].nunique() > 1:
                    try:
                        auc_mkt_list.append(float(roc_auc_score(y.loc[mask], mkt_probs.loc[mask])))
                    except Exception:
                        pass
                    try:
                        brier_mkt_list.append(float(brier_score_loss(y.loc[mask], mkt_probs.loc[mask])))
                    except Exception:
                        pass
            total_preds += len(pred)

    summary: Dict[str, float] = {}
    if mae_m_list:
        summary["MAE_margin"] = float(np.mean(mae_m_list))
        summary["RMSE_margin"] = float(np.mean(rmse_m_list))
    if mae_m_mkt:
        summary["MAE_margin_market"] = float(np.mean(mae_m_mkt))
        summary["RMSE_margin_market"] = float(np.mean(rmse_m_mkt))
        if "MAE_margin" in summary:
            summary["MAE_margin_delta_vs_market"] = summary["MAE_margin"] - summary["MAE_margin_market"]
        if "RMSE_margin" in summary:
            summary["RMSE_margin_delta_vs_market"] = summary["RMSE_margin"] - summary["RMSE_margin_market"]
    if mae_t_list:
        summary["MAE_total"] = float(np.mean(mae_t_list))
        summary["RMSE_total"] = float(np.mean(rmse_t_list))
    if mae_t_mkt:
        summary["MAE_total_market"] = float(np.mean(mae_t_mkt))
        summary["RMSE_total_market"] = float(np.mean(rmse_t_mkt))
        if "MAE_total" in summary:
            summary["MAE_total_delta_vs_market"] = summary["MAE_total"] - summary["MAE_total_market"]
        if "RMSE_total" in summary:
            summary["RMSE_total_delta_vs_market"] = summary["RMSE_total"] - summary["RMSE_total_market"]
    if prob_and_y:
        probs, ys = zip(*prob_and_y)
        try:
            summary["AUC_home_win"] = float(roc_auc_score(ys, probs))
        except Exception:
            pass
        try:
            summary["Brier_home_win"] = float(brier_score_loss(ys, probs))
        except Exception:
            pass
    if auc_mkt_list:
        summary["AUC_home_win_market"] = float(np.mean(auc_mkt_list))
    if brier_mkt_list:
        summary["Brier_home_win_market"] = float(np.mean(brier_mkt_list))
    summary["predictions_scored"] = int(total_preds)
    summary["seasons_evaluated"] = len(seasons)
    return summary


if __name__ == "__main__":
    # Optional speed knob: allow reducing training size per season via env
    min_weeks = int(os.getenv("NFL_EVAL_MIN_WEEKS_TRAIN", "4"))
    out = walkforward_eval(min_weeks_train=min_weeks)
    print(out)
