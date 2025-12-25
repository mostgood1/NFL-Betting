from __future__ import annotations

"""
Backtest game-level models on a target season up to a given week.

We train on historical seasons (< target season) and optionally on same-season weeks prior to the cutoff,
then evaluate on the target season weeks [start_week..end_week]. Metrics include MAE for margin and total,
and accuracy for home team wins.

Usage:
  python scripts/backtest_games.py --season 2025 --start-week 1 --end-week 9 \
    --include-same-season --out nfl_compare/data/backtest_games_2025_wk9.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models, predict as model_predict
from nfl_compare.src.weather import load_weather_for_games

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    e = (yt - yp).abs().dropna()
    return float(e.mean()) if len(e) else float("nan")


def _acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    ok = y_true.notna() & y_pred.notna()
    return float((y_true[ok] == y_pred[ok]).mean()) if ok.any() else float("nan")


def backtest(season: int, start_week: int, end_week: int, include_same: bool = True, blend_margin: float = 0.0, blend_total: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx)
    # Completed games only for target season slice
    feat = feat.copy()
    for c in ("season","week"):
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
    # Build train/eval splits
    hist = feat.dropna(subset=["home_score","away_score"]).copy()
    train_mask = hist["season"] < int(season)
    if include_same:
        train_mask = train_mask | ((hist["season"] == int(season)) & (hist["week"] < int(start_week)))
    df_train = hist[train_mask].copy()

    eval_mask = (hist["season"] == int(season)) & (hist["week"] >= int(start_week)) & (hist["week"] <= int(end_week))
    df_eval = hist[eval_mask].copy()

    # Deduplicate by game_id to avoid explosion from joins (e.g., multiple lines rows)
    try:
        if 'game_id' in df_train.columns:
            df_train = df_train.sort_values(['season','week','game_id']).drop_duplicates(subset=['game_id'], keep='first')
    except Exception:
        pass
    try:
        if 'game_id' in df_eval.columns:
            df_eval = df_eval.sort_values(['season','week','game_id']).drop_duplicates(subset=['game_id'], keep='first')
    except Exception:
        pass

    if df_train.empty or df_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    models = train_models(df_train)
    pred = model_predict(models, df_eval)
    # Optional blending of predictions toward market
    try:
        if blend_margin and blend_margin > 0:
            mh = pd.to_numeric(df_eval.get('spread_home'), errors='coerce') if 'spread_home' in df_eval.columns else pd.Series(index=df_eval.index, dtype=float)
            market_margin = -mh
            pm = pd.to_numeric(pred.get('pred_margin'), errors='coerce')
            pred['pred_margin'] = (1.0 - float(blend_margin)) * pm + float(blend_margin) * market_margin
    except Exception:
        pass
    try:
        if blend_total and blend_total > 0:
            mt = pd.to_numeric(df_eval.get('total'), errors='coerce') if 'total' in df_eval.columns else pd.Series(index=df_eval.index, dtype=float)
            pt = pd.to_numeric(pred.get('pred_total'), errors='coerce')
            pred['pred_total'] = (1.0 - float(blend_total)) * pt + float(blend_total) * mt
    except Exception:
        pass

    # Actual targets
    pred["home_margin_actual"] = df_eval["home_score"] - df_eval["away_score"]
    pred["total_points_actual"] = df_eval["home_score"] + df_eval["away_score"]
    pred["home_win_actual"] = (pred["home_margin_actual"] > 0).astype(int)
    pred["home_win_pred"] = (pred["pred_margin"] > 0).astype(int)

    # Metrics
    # Prepare market lines for ATS and totals classification
    spread_home = pd.to_numeric(df_eval.get("spread_home"), errors="coerce") if "spread_home" in df_eval.columns else pd.Series(index=df_eval.index, dtype=float)
    market_total = pd.to_numeric(df_eval.get("total"), errors="coerce") if "total" in df_eval.columns else pd.Series(index=df_eval.index, dtype=float)
    margin_actual = pd.to_numeric(pred["home_margin_actual"], errors="coerce")
    margin_pred = pd.to_numeric(pred["pred_margin"], errors="coerce")
    total_actual = pd.to_numeric(pred["total_points_actual"], errors="coerce")
    total_pred = pd.to_numeric(pred["pred_total"], errors="coerce")

    # ATS: home covers if margin + spread_home > 0
    ats_act = (margin_actual + spread_home) > 0
    ats_pred = (margin_pred + spread_home) > 0
    mask_spread = spread_home.notna()
    # Over/Under: total vs market_total
    ou_act = total_actual > market_total
    ou_pred = total_pred > market_total
    mask_total = market_total.notna()

    # Build summary row
    row: Dict[str, float | int] = {
        "season": int(season),
        "start_week": int(start_week),
        "end_week": int(end_week),
        "n_games": int(len(df_eval)),
        "mae_margin": _mae(pred["home_margin_actual"], pred["pred_margin"]),
        "mae_total": _mae(pred["total_points_actual"], pred["pred_total"]),
        "acc_home_win": _acc(pred["home_win_actual"], pred["home_win_pred"]),
        # Accuracy computed only where market lines exist
        "acc_spread_cover": float(((ats_act[mask_spread] == ats_pred[mask_spread]).astype(float)).mean()) if mask_spread.any() else float('nan'),
        "acc_over_under": float(((ou_act[mask_total] == ou_pred[mask_total]).astype(float)).mean()) if mask_total.any() else float('nan'),
    }
    # Build per-game details for standardized reporting
    details_cols: List[str] = []
    base = df_eval.copy()
    keep_base = [c for c in ["season","week","game_id","game_date","date","home_team","away_team","home_score","away_score","spread_home","total"] if c in base.columns]
    if keep_base:
        base = base[keep_base].copy()
    else:
        base = pd.DataFrame()
    det = pred.copy()
    # Attach market lines if present in eval base
    try:
        if not base.empty:
            det = det.merge(base, left_index=True, right_index=True, how="left", suffixes=("",""))
    except Exception:
        pass
    # Compute residuals and normalize column names
    try:
        det["resid_margin"] = pd.to_numeric(det["home_margin_actual"], errors="coerce") - pd.to_numeric(det["pred_margin"], errors="coerce")
    except Exception:
        det["resid_margin"] = pd.NA
    try:
        det["resid_total"] = pd.to_numeric(det["total_points_actual"], errors="coerce") - pd.to_numeric(det["pred_total"], errors="coerce")
    except Exception:
        det["resid_total"] = pd.NA
    # Rename common columns for clarity
    det = det.rename(columns={
        "home_margin_actual": "margin_actual",
        "total_points_actual": "total_actual",
        "pred_margin": "margin_pred",
        "pred_total": "total_pred",
        "spread_home": "market_spread_home",
        "total": "market_total",
        "date": "game_date",
    })
    # ATS and O/U flags
    try:
        det["act_home_cover"] = (pd.to_numeric(det.get("margin_actual"), errors="coerce") + pd.to_numeric(det.get("market_spread_home"), errors="coerce")) > 0
    except Exception:
        det["act_home_cover"] = pd.NA
    try:
        det["pred_home_cover"] = (pd.to_numeric(det.get("margin_pred"), errors="coerce") + pd.to_numeric(det.get("market_spread_home"), errors="coerce")) > 0
    except Exception:
        det["pred_home_cover"] = pd.NA
    try:
        det["act_over"] = pd.to_numeric(det.get("total_actual"), errors="coerce") > pd.to_numeric(det.get("market_total"), errors="coerce")
    except Exception:
        det["act_over"] = pd.NA
    try:
        det["pred_over"] = pd.to_numeric(det.get("total_pred"), errors="coerce") > pd.to_numeric(det.get("market_total"), errors="coerce")
    except Exception:
        det["pred_over"] = pd.NA
    # Re-order detail columns when available
    prefer = [
        "season","week","game_id","game_date","home_team","away_team",
        "margin_pred","total_pred","market_spread_home","market_total",
        "margin_actual","total_actual","resid_margin","resid_total"
    ]
    # Extend preferred order with ATS and O/U flags
    prefer = prefer + [c for c in ["pred_home_cover","act_home_cover","pred_over","act_over"] if c in det.columns]
    cols = [c for c in prefer if c in det.columns] + [c for c in det.columns if c not in prefer]
    det = det[cols]
    return pd.DataFrame([row]), det


def main():
    ap = argparse.ArgumentParser(description="Backtest game models on a target season/week range")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--include-same-season", action="store_true", help="Include same-season weeks < start-week in training")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None, help="Standardized output directory (backtests/<season>_wk<end_week>/)")
    ap.add_argument("--blend-margin", type=float, default=0.0, help="Blend model margin toward market-implied margin (-spread_home)")
    ap.add_argument("--blend-total", type=float, default=0.0, help="Blend model total toward market total")
    args = ap.parse_args()

    summ_df, det_df = backtest(
        args.season,
        args.start_week,
        args.end_week,
        include_same=args.include_same_season,
        blend_margin=args.blend_margin,
        blend_total=args.blend_total,
    )
    if summ_df is None or summ_df.empty:
        print("No results; nothing written.")
        return 0
    # Standardized output dir
    std_dir: Optional[Path] = None
    if args.out_dir:
        std_dir = Path(args.out_dir)
    else:
        std_dir = DATA_DIR / "backtests" / f"{args.season}_wk{args.end_week}"
    try:
        std_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        std_dir = None
    # Write legacy single CSV if --out provided
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(out_fp, index=False)
        print(f"Wrote games backtest summary -> {out_fp}")
    # Write standardized files
    if std_dir is not None:
        try:
            summ_fp = std_dir / "games_summary.csv"
            summ_df.to_csv(summ_fp, index=False)
            print(f"Wrote {summ_fp}")
        except Exception as e:
            print(f"Failed writing games_summary.csv: {e}")
        try:
            if det_df is not None and not det_df.empty:
                det_fp = std_dir / "games_details.csv"
                det_df.to_csv(det_fp, index=False)
                print(f"Wrote {det_fp}")
                # Derive weekly accuracy metrics for winners, ATS, and O/U
                try:
                    import numpy as _np
                    df = det_df.copy()
                    wk = pd.to_numeric(df.get('week'), errors='coerce')
                    margin_pred = pd.to_numeric(df.get('margin_pred'), errors='coerce')
                    margin_act = pd.to_numeric(df.get('margin_actual'), errors='coerce')
                    total_pred = pd.to_numeric(df.get('total_pred'), errors='coerce')
                    total_act = pd.to_numeric(df.get('total_actual'), errors='coerce')
                    spread_home = pd.to_numeric(df.get('market_spread_home'), errors='coerce')
                    market_total = pd.to_numeric(df.get('market_total'), errors='coerce')

                    home_pred_flag = margin_pred > 0
                    away_pred_flag = ~home_pred_flag
                    home_act_flag = margin_act > 0
                    away_act_flag = ~home_act_flag
                    home_pred = home_pred_flag.astype('float')
                    home_act = home_act_flag.astype('float')
                    ats_pred = (margin_pred + spread_home) > 0
                    ats_act = (margin_act + spread_home) > 0
                    mask_spread = spread_home.notna()
                    ou_pred = total_pred > market_total
                    ou_act = total_act > market_total
                    mask_total = market_total.notna()
                    mae_margin = (margin_act - margin_pred).abs()
                    mae_total = (total_act - total_pred).abs()

                    rows = []
                    for w in sorted([int(x) for x in wk.dropna().unique()]):
                        m = wk == w
                        n = int(m.sum())
                        if n == 0:
                            continue
                        # Accuracy splits
                        try:
                            acc_pred_home = float(_np.nanmean((home_act_flag[m & home_pred_flag]).astype('float'))) if (m & home_pred_flag).any() else _np.nan
                        except Exception:
                            acc_pred_home = _np.nan
                        try:
                            acc_pred_away = float(_np.nanmean((away_act_flag[m & away_pred_flag]).astype('float'))) if (m & away_pred_flag).any() else _np.nan
                        except Exception:
                            acc_pred_away = _np.nan
                        try:
                            acc_actual_home = float(_np.nanmean((home_pred_flag[m & home_act_flag]).astype('float'))) if (m & home_act_flag).any() else _np.nan
                        except Exception:
                            acc_actual_home = _np.nan
                        try:
                            acc_actual_away = float(_np.nanmean((away_pred_flag[m & away_act_flag]).astype('float'))) if (m & away_act_flag).any() else _np.nan
                        except Exception:
                            acc_actual_away = _np.nan

                        rows.append({
                            'week': w,
                            'n_games': n,
                            'acc_home_win': float(_np.nanmean((home_pred[m] == home_act[m]).astype('float'))),
                            'acc_spread_cover': float(_np.nanmean((ats_pred[m & mask_spread] == ats_act[m & mask_spread]).astype('float'))) if (m & mask_spread).any() else _np.nan,
                            'acc_over_under': float(_np.nanmean((ou_pred[m & mask_total] == ou_act[m & mask_total]).astype('float'))) if (m & mask_total).any() else _np.nan,
                            'mae_margin': float(_np.nanmean(mae_margin[m])),
                            'mae_total': float(_np.nanmean(mae_total[m])),
                            'home_pred_rate': float(_np.nanmean(home_pred[m])),
                            'acc_pred_home': acc_pred_home,
                            'acc_pred_away': acc_pred_away,
                            'acc_actual_home': acc_actual_home,
                            'acc_actual_away': acc_actual_away,
                        })
                    weekly_df = pd.DataFrame(rows)
                    wk_fp = std_dir / 'games_weekly.csv'
                    weekly_df.to_csv(wk_fp, index=False)
                    print(f"Wrote {wk_fp}")
                except Exception as e:
                    print(f"Failed writing weekly metrics: {e}")
        except Exception as e:
            print(f"Failed writing games_details.csv: {e}")
        # Minimal metrics.json for quick consumption
        try:
            import json as _json
            metrics = summ_df.iloc[0].to_dict()
            # Try augment with weekly summary if available
            try:
                wk_fp = std_dir / 'games_weekly.csv'
                if wk_fp.exists():
                    wk_df = pd.read_csv(wk_fp)
                    # split H1/H2 for quick glance
                    h1 = wk_df[wk_df['week'] <= 8]
                    h2 = wk_df[wk_df['week'] > 8]
                    def _mean(df, col):
                        try:
                            return float(df[col].astype(float).mean())
                        except Exception:
                            return None
                    metrics.update({
                        'acc_spread_cover': metrics.get('acc_spread_cover'),
                        'acc_over_under': metrics.get('acc_over_under'),
                        'weekly_h1_acc_spread_cover': _mean(h1, 'acc_spread_cover'),
                        'weekly_h2_acc_spread_cover': _mean(h2, 'acc_spread_cover'),
                        'weekly_h1_acc_over_under': _mean(h1, 'acc_over_under'),
                        'weekly_h2_acc_over_under': _mean(h2, 'acc_over_under'),
                    })
            except Exception:
                pass
            with open(std_dir / "metrics.json", "w", encoding="utf-8") as f:
                _json.dump({"games": metrics}, f, indent=2)
            print(f"Wrote {std_dir / 'metrics.json'}")
        except Exception as e:
            print(f"Failed writing metrics.json: {e}")
    # Console print
    try:
        print(summ_df.to_string(index=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
