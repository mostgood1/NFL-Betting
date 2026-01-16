from __future__ import annotations

"""
Monte Carlo simulations for game outcomes using model predictions and calibrated sigmas.

Outputs per-game probabilities for MONEYLINE, ATS, and TOTAL based on simulated distributions.

Usage:
  python scripts/simulate_games.py --season 2025 --start-week 1 --end-week 17 --n-sims 2000 \
    --out-dir nfl_compare/data/backtests/2025_wk17
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.features import merge_features
from nfl_compare.src.models import train_models, predict as model_predict
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.sim_engine import compute_margin_total_draws, simulate_mc_probs, simulate_quarter_means, simulate_drive_timeline

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _load_sigma_calibration() -> Dict[str, float]:
    fp = DATA_DIR / "sigma_calibration.json"
    try:
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
            # support both {"ats_sigma": x, "total_sigma": y} and nested
            if isinstance(j, dict) and "ats_sigma" in j and "total_sigma" in j:
                return {"ats_sigma": float(j["ats_sigma"]), "total_sigma": float(j["total_sigma"])}
            if isinstance(j, dict) and "sigma" in j and isinstance(j["sigma"], dict):
                s = j["sigma"]
                return {"ats_sigma": float(s.get("ats_sigma", np.nan)), "total_sigma": float(s.get("total_sigma", np.nan))}
    except Exception:
        pass
    # Fallback reasonable scales if calibration missing
    return {"ats_sigma": 12.0, "total_sigma": 11.0}


def simulate(
    season: int,
    start_week: int,
    end_week: int,
    n_sims: int = 2000,
    include_same: bool = True,
    ats_sigma_override: float | None = None,
    total_sigma_override: float | None = None,
    seed: int | None = None,
    pred_cache_fp: str | None = None,
    skip_features: bool | None = None,
    compute_quarters: bool = False,
    compute_drives: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)

    # Decide if we should skip heavy feature merge (fast mode for sweeps)
    try:
        import os as _os
        if skip_features is None:
            skip_features = bool(int(_os.environ.get('SIM_SKIP_FEATURES', '0')))
    except Exception:
        pass

    if skip_features:
        # Build a minimal eval frame: games + lines (no weather/stats)
        feat = games.copy()
        try:
            feat = feat.merge(lines[[c for c in lines.columns if c in ['game_id','spread_home','total','close_spread_home','close_total']]], on='game_id', how='left')
        except Exception:
            pass
    else:
        feat = merge_features(games, stats, lines, wx).copy()
    for c in ("season", "week"):
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # Completed games subset for training
    hist = feat.dropna(subset=["home_score", "away_score"]).copy()
    train_mask = hist["season"] < int(season)
    if include_same:
        train_mask = train_mask | ((hist["season"] == int(season)) & (hist["week"] < int(start_week)))
    df_train = hist[train_mask].copy()

    # Evaluation slice for target season weeks
    eval_mask = (feat["season"] == int(season)) & (feat["week"] >= int(start_week)) & (feat["week"] <= int(end_week))
    df_eval = feat[eval_mask].copy()

    # Deduplicate by game_id to avoid multiple lines rows inflating eval
    try:
        if "game_id" in df_train.columns:
            df_train = df_train.sort_values(["season", "week", "game_id"]).drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass
    try:
        if "game_id" in df_eval.columns:
            df_eval = df_eval.sort_values(["season", "week", "game_id"]).drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass

    if df_train.empty or df_eval.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Optional: use cached predictions to avoid retraining
    pred: pd.DataFrame
    _use_cache = False
    try:
        import os as _os
        if pred_cache_fp is None:
            pred_cache_fp = _os.environ.get('SIM_PRED_CACHE_FP')
        _use_cache = bool(_os.environ.get('SIM_USE_PRED_CACHE')) or bool(pred_cache_fp)
    except Exception:
        _use_cache = bool(pred_cache_fp)
    if _use_cache and pred_cache_fp:
        try:
            cache = pd.read_csv(Path(pred_cache_fp))
            # Accept either margin_pred/total_pred or pred_margin/pred_total
            pm_col = 'margin_pred' if 'margin_pred' in cache.columns else ('pred_margin' if 'pred_margin' in cache.columns else None)
            pt_col = 'total_pred' if 'total_pred' in cache.columns else ('pred_total' if 'pred_total' in cache.columns else None)
            key_cols = [c for c in ['season','week','game_id'] if c in cache.columns]
            if pm_col and pt_col and key_cols:
                pred = df_eval[key_cols].copy()
                pred = pred.merge(cache[key_cols + [pm_col, pt_col]], on=key_cols, how='left')
                # Normalize names
                pred = pred.rename(columns={pm_col: 'pred_margin', pt_col: 'pred_total'})
            else:
                # Fallback: no viable cache match, train models
                models = train_models(df_train)
                pred = model_predict(models, df_eval).copy()
        except Exception:
            models = train_models(df_train)
            pred = model_predict(models, df_eval).copy()
    else:
        models = train_models(df_train)
        pred = model_predict(models, df_eval).copy()

    # Merge predicted means into the eval frame by keys (avoid index alignment issues)
    try:
        pred_means = pred.copy()
        key_cols = [c for c in ["season", "week", "game_id"] if c in df_eval.columns and c in pred_means.columns]
        if not key_cols:
            key_cols = [c for c in ["season", "week", "home_team", "away_team"] if c in df_eval.columns and c in pred_means.columns]
        mean_cols = [c for c in ["pred_margin", "pred_total"] if c in pred_means.columns]
        if key_cols and mean_cols:
            pred_means = pred_means[key_cols + mean_cols].drop_duplicates()
            merged = df_eval.merge(pred_means, on=key_cols, how="left", suffixes=("", "_p"))
            for c in mean_cols:
                cp = f"{c}_p"
                if cp in merged.columns:
                    if c in merged.columns:
                        merged[c] = merged[c].where(merged[c].notna(), merged[cp])
                    else:
                        merged[c] = merged[cp]
            drop_p = [c for c in merged.columns if c.endswith("_p")]
            if drop_p:
                merged = merged.drop(columns=drop_p)
            df_eval = merged
    except Exception:
        pass

    draws_by_game = None
    if compute_quarters or compute_drives:
        draws_by_game = compute_margin_total_draws(
            df_eval,
            n_sims=int(n_sims),
            ats_sigma_override=ats_sigma_override,
            total_sigma_override=total_sigma_override,
            seed=seed,
            data_dir=DATA_DIR,
        )

    probs_df = simulate_mc_probs(
        df_eval,
        n_sims=int(n_sims),
        ats_sigma_override=ats_sigma_override,
        total_sigma_override=total_sigma_override,
        seed=seed,
        data_dir=DATA_DIR,
        draws_by_game=draws_by_game,
    )

    # Optional: quarter-by-quarter means artifact
    if compute_quarters:
        try:
            q_df = simulate_quarter_means(
                df_eval,
                n_sims=int(n_sims),
                ats_sigma_override=ats_sigma_override,
                total_sigma_override=total_sigma_override,
                seed=seed,
                data_dir=DATA_DIR,
                draws_by_game=draws_by_game,
            )
            if q_df is not None and not q_df.empty:
                # Attach to probs_df for convenience if desired by callers
                pass
        except Exception:
            q_df = pd.DataFrame()
    else:
        q_df = pd.DataFrame()

    # Optional: drive-by-drive timeline artifact
    if compute_drives:
        try:
            # Concatenate per-week player props caches when present; improves drive count estimation.
            props_frames: list[pd.DataFrame] = []
            for wk in range(int(start_week), int(end_week) + 1):
                fp = DATA_DIR / f"player_props_{int(season)}_wk{int(wk)}.csv"
                if fp.exists():
                    try:
                        props_frames.append(pd.read_csv(fp))
                    except Exception:
                        continue
            props_df = pd.concat(props_frames, ignore_index=True) if props_frames else None

            d_df = simulate_drive_timeline(
                view_df=df_eval,
                props_df=props_df,
                n_sims=int(n_sims),
                ats_sigma_override=ats_sigma_override,
                total_sigma_override=total_sigma_override,
                seed=seed,
                data_dir=DATA_DIR,
                draws_by_game=draws_by_game,
            )
        except Exception:
            d_df = pd.DataFrame()
    else:
        d_df = pd.DataFrame()
    # Attach classifier probabilities if available for comparison
    try:
        if "game_id" in df_eval.columns and "game_id" in probs_df.columns:
            if "prob_home_cover" in pred.columns and "game_id" in pred.columns:
                probs_df = probs_df.merge(pred[["game_id", "prob_home_cover"]].rename(columns={"prob_home_cover": "prob_home_cover_clf"}), on="game_id", how="left")
            if "prob_over_total" in pred.columns and "game_id" in pred.columns:
                probs_df = probs_df.merge(pred[["game_id", "prob_over_total"]].rename(columns={"prob_over_total": "prob_over_total_clf"}), on="game_id", how="left")
    except Exception:
        pass

    # Minimal summary (counts of finite probabilities)
    try:
        sig = _load_sigma_calibration()
        ats_sigma = float(sig.get("ats_sigma", 12.0))
        total_sigma = float(sig.get("total_sigma", 11.0))
        if isinstance(ats_sigma_override, (int, float)) and float(ats_sigma_override) > 0:
            ats_sigma = float(ats_sigma_override)
        if isinstance(total_sigma_override, (int, float)) and float(total_sigma_override) > 0:
            total_sigma = float(total_sigma_override)
        summary = {
            "season": season,
            "start_week": start_week,
            "end_week": end_week,
            "n_games": int(len(probs_df)),
            "n_ml_probs": int(np.isfinite(probs_df.get("prob_home_win_mc")).sum()),
            "n_ats_probs": int(np.isfinite(probs_df.get("prob_home_cover_mc")).sum()),
            "n_total_probs": int(np.isfinite(probs_df.get("prob_over_total_mc")).sum()),
            "ats_sigma": ats_sigma,
            "total_sigma": total_sigma,
            "n_sims": int(n_sims),
            "quarters_enabled": bool(compute_quarters),
            "drives_enabled": bool(compute_drives),
        }
    except Exception:
        summary = {}

    # Stash optional artifacts in summary df attrs so callers can fetch.
    try:
        summ = pd.DataFrame([summary])
        summ.attrs["quarters_df"] = q_df
        summ.attrs["drives_df"] = d_df
        return probs_df, summ
    except Exception:
        return probs_df, pd.DataFrame([summary])


def main():
    ap = argparse.ArgumentParser(description="Simulate game outcomes to derive ML/ATS/Total probabilities")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--include-same-season", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--ats-sigma", type=float, default=None, help="Override ATS sigma for simulations")
    ap.add_argument("--total-sigma", type=float, default=None, help="Override Total sigma for simulations")
    ap.add_argument("--pred-cache", type=str, default=None, help="Path to cached predictions CSV (games_details.csv)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--quarters", action="store_true", help="Also compute quarter-by-quarter expected scores")
    ap.add_argument("--drives", action="store_true", help="Also compute a drive-by-drive expected timeline")
    args = ap.parse_args()

    probs_df, summ_df = simulate(
        args.season,
        args.start_week,
        args.end_week,
        n_sims=args.n_sims,
        include_same=args.include_same_season,
        ats_sigma_override=args.ats_sigma,
        total_sigma_override=args.total_sigma,
        seed=args.seed,
        pred_cache_fp=args.pred_cache,
        compute_quarters=bool(args.quarters),
        compute_drives=bool(args.drives),
    )
    if probs_df is None or probs_df.empty:
        print("No simulation results; nothing written.")
        return 0

    out_dir: Optional[Path]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = DATA_DIR / "backtests" / f"{args.season}_wk{args.end_week}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        probs_fp = out_dir / "sim_probs.csv"
        probs_df.to_csv(probs_fp, index=False)
        print(f"Wrote {probs_fp}")
    except Exception as e:
        print(f"Failed writing sim_probs.csv: {e}")

    # Write quarter artifact if requested and present
    try:
        q_df = getattr(summ_df, "attrs", {}).get("quarters_df") if summ_df is not None else None
        if args.quarters and q_df is not None and hasattr(q_df, "empty") and not q_df.empty:
            q_fp = out_dir / "sim_quarters.csv"
            q_df.to_csv(q_fp, index=False)
            print(f"Wrote {q_fp}")
    except Exception as e:
        print(f"Failed writing sim_quarters.csv: {e}")

    # Write drives artifact if requested and present
    try:
        d_df = getattr(summ_df, "attrs", {}).get("drives_df") if summ_df is not None else None
        if args.drives and d_df is not None and hasattr(d_df, "empty") and not d_df.empty:
            d_fp = out_dir / "sim_drives.csv"
            d_df.to_csv(d_fp, index=False)
            print(f"Wrote {d_fp}")
    except Exception as e:
        print(f"Failed writing sim_drives.csv: {e}")
    try:
        summ_fp = out_dir / "sim_summary.json"
        with open(summ_fp, "w", encoding="utf-8") as f:
            json.dump(summ_df.iloc[0].to_dict(), f, indent=2)
        print(f"Wrote {summ_fp}")
    except Exception as e:
        print(f"Failed writing sim_summary.json: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
