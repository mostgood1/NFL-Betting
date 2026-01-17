from __future__ import annotations
"""
Backtest EV-based recommendations (Moneyline, ATS, Total) over a recent window.

- Builds weekly views via app helpers, attaches predictions and totals calibration
- Generates recommendations per game using app._compute_recommendations_for_row
- Filters using RECS_MIN_EV_PCT and includes finals for grading
- Summarizes win rates, pushes, and EV stats per market and confidence tier

Outputs under nfl_compare/data/backtests/<season>_wk<end_week>/:
- recs_backtest_details.csv
- recs_backtest_summary.json
- recs_backtest_summary.md

Usage:
  python scripts/backtest_recommendations.py --season 2025 --end-week 17 --lookback 6 --min-ev 2.0 --one-per-game false
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"

# Ensure repo root is on sys.path when running from scripts/
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import app helpers lazily
from app import (
    _load_predictions,
    _load_games,
    _build_week_view,
    _attach_model_predictions,
    _apply_totals_calibration_to_view,
    _compute_recommendations_for_row,
    _get_sim_probs_df,
)


def _attach_walk_forward_predictions(
    view_df: pd.DataFrame,
    games_df: pd.DataFrame,
    season: int,
    week: int,
    stats_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    wx_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Train models on prior weeks only and predict the given week.

    This is intended for trustworthy backtests: for week N, train on weeks < N.
    It also blanks out scores for weeks >= N when building features to prevent
    Elo updates (and similar) from using same-week/future results.
    """
    if view_df is None or view_df.empty:
        return view_df

    # Build as-of games: hide same-week and future scores
    g = games_df.copy()
    try:
        for c in ("season", "week"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
        m_hide = (g.get("season") == int(season)) & (g.get("week") >= int(week))
        for c in ("home_score", "away_score"):
            if c in g.columns:
                g.loc[m_hide, c] = pd.NA
    except Exception:
        pass

    from nfl_compare.src.features import merge_features
    from nfl_compare.src.models import train_models, predict as model_predict

    feat = merge_features(g, stats_df, lines_df, wx_df)
    if feat is None or feat.empty:
        return view_df
    try:
        feat["season"] = pd.to_numeric(feat.get("season"), errors="coerce")
        feat["week"] = pd.to_numeric(feat.get("week"), errors="coerce")
    except Exception:
        pass

    # Training set: all prior seasons + prior weeks of the target season (walk-forward).
    # This avoids training on the same week's outcomes while still leveraging historical data.
    try:
        s = pd.to_numeric(feat.get("season"), errors="coerce")
        w = pd.to_numeric(feat.get("week"), errors="coerce")
        m_train = (s < int(season)) | ((s == int(season)) & (w < int(week)))
        train = feat[m_train].copy()
    except Exception:
        train = feat[(feat.get("season") == int(season)) & (feat.get("week") < int(week))].copy()
    if train.empty:
        return view_df
    try:
        hs = pd.to_numeric(train.get("home_score"), errors="coerce")
        aas = pd.to_numeric(train.get("away_score"), errors="coerce")
        train = train[hs.notna() & aas.notna()].copy()
        train["home_margin"] = hs - aas
        train["total_points"] = hs + aas
    except Exception:
        return view_df
    if train.empty:
        return view_df

    # Predict set: current week
    fut = feat[(feat.get("season") == int(season)) & (feat.get("week") == int(week))].copy()
    if fut.empty:
        return view_df

    # Defensive: ensure one row per game before predicting/merging.
    try:
        if "game_id" in fut.columns:
            fut = fut.drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass

    models = train_models(train)
    pred = model_predict(models, fut)

    try:
        if "game_id" in pred.columns:
            pred = pred.drop_duplicates(subset=["game_id"], keep="first")
    except Exception:
        pass

    # Convert (margin,total) into team points for downstream card/recs logic
    try:
        m = pd.to_numeric(pred.get("pred_margin"), errors="coerce")
        t = pd.to_numeric(pred.get("pred_total"), errors="coerce")
        ph = 0.5 * (t + m)
        pa = t - ph
        pred["pred_home_points"] = ph
        pred["pred_away_points"] = pa
    except Exception:
        pass
    # Align naming with app logic
    try:
        if "prob_home_win" in pred.columns and "pred_home_win_prob" not in pred.columns:
            pred["pred_home_win_prob"] = pred["prob_home_win"]
    except Exception:
        pass

    # Merge into the provided view
    out = view_df.copy()
    join_cols = [c for c in [
        "pred_home_points",
        "pred_away_points",
        "pred_total",
        "pred_margin",
        "prob_home_win",
        "pred_home_win_prob",
        "prob_home_cover",
        "prob_over_total",
    ] if c in pred.columns]

    if "game_id" in out.columns and "game_id" in pred.columns:
        # Prefer overwrite-by-key so downstream logic reads the intended columns.
        try:
            oidx = out.set_index("game_id", drop=False)
            pidx = pred.set_index("game_id", drop=False)
            for c in join_cols:
                if c in pidx.columns:
                    oidx[c] = pidx[c]
            out = oidx.reset_index(drop=True)
        except Exception:
            keep = ["game_id"] + join_cols
            out = out.merge(pred[keep], on="game_id", how="left", suffixes=("", "_wf"))
            for c in join_cols:
                wf = f"{c}_wf"
                if wf in out.columns:
                    out[c] = out[wf]
                    out = out.drop(columns=[wf])
    else:
        # Fallback join on season/week/home/away
        keys = [k for k in ["season", "week", "home_team", "away_team"] if k in out.columns and k in pred.columns]
        keep = keys + join_cols
        if keys:
            out = out.merge(pred[keep], on=keys, how="left", suffixes=("", "_wf"))
            for c in join_cols:
                wf = f"{c}_wf"
                if wf in out.columns:
                    out[c] = out[wf]
                    out = out.drop(columns=[wf])

    try:
        out["pred_source"] = "walkfwd"
    except Exception:
        pass
    return out


def _build_window(
    season: int,
    end_week: int,
    lookback: int,
    compute_mc: bool = False,
    walk_forward: bool = False,
) -> pd.DataFrame:
    pred = _load_predictions()
    games = _load_games()
    start_week = max(1, int(end_week) - int(lookback) + 1)
    frames: List[pd.DataFrame] = []

    stats_df = None
    lines_df = None
    wx_df = None
    if walk_forward:
        try:
            from nfl_compare.src.data_sources import load_team_stats, load_lines
            from nfl_compare.src.weather import load_weather_for_games

            stats_df = load_team_stats()
            lines_df = load_lines()
            wx_df = load_weather_for_games(games)
        except Exception:
            stats_df = pd.DataFrame()
            lines_df = pd.DataFrame()
            wx_df = None

    for wk in range(start_week, end_week + 1):
        try:
            v = _build_week_view(pred, games, int(season), int(wk))
            if walk_forward:
                # Ensure market lines/odds enrichment happens even in walk-forward mode.
                # We'll overwrite prediction columns afterwards.
                v = _attach_model_predictions(v)
                v = _attach_walk_forward_predictions(
                    v,
                    games_df=games,
                    season=int(season),
                    week=int(wk),
                    stats_df=stats_df if stats_df is not None else pd.DataFrame(),
                    lines_df=lines_df if lines_df is not None else pd.DataFrame(),
                    wx_df=wx_df,
                )
            else:
                v = _attach_model_predictions(v)
            try:
                v = _apply_totals_calibration_to_view(v)
            except Exception:
                pass
            # Optional: precompute Monte Carlo probabilities for this week (cached in-process).
            # This enables RECS_* MC gates/conf methods to actually use per-game MC probs.
            if compute_mc:
                try:
                    _get_sim_probs_df(int(season), int(wk), view_df=v)
                except Exception:
                    pass
            if v is not None and not v.empty:
                frames.append(v)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Restrict to season and finals when available
    try:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
        m = (out["season"].eq(int(season))) & out["week"].notna() & out["week"].between(start_week, end_week)
        out = out[m].copy()
    except Exception:
        pass
    return out


def _summarize_recs(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0}

    def _american_to_decimal(american: float) -> float:
        # 1u stake: +150 => 2.50 dec, -110 => 1.9091 dec
        if american is None or pd.isna(american) or american == 0:
            return float("nan")
        a = float(american)
        if a > 0:
            return 1.0 + (a / 100.0)
        return 1.0 + (100.0 / abs(a))

    def _odds_to_decimal(s: pd.Series) -> pd.Series:
        o = pd.to_numeric(s, errors="coerce")
        # Heuristic: values in [1.01, 10] are already decimal; others treat as American
        is_decimal = o.between(1.01, 10.0)
        dec = pd.Series(float("nan"), index=o.index, dtype="float")
        dec.loc[is_decimal] = o.loc[is_decimal].astype(float)
        for idx, val in o.loc[~is_decimal].items():
            dec.loc[idx] = _american_to_decimal(val)
        return pd.to_numeric(dec, errors="coerce")

    # Standardize columns
    work = df.copy()
    work["type"] = work["type"].astype(str)
    work["confidence"] = work.get("confidence", pd.Series([None]*len(work))).astype(str)

    # Realized P/L (1u stake) when odds + result are available
    dec_odds = _odds_to_decimal(work.get("odds", pd.Series([pd.NA] * len(work))))
    res_u = work.get("result", pd.Series([None] * len(work))).astype(str).str.upper()
    profit = pd.Series(float("nan"), index=work.index, dtype="float")
    profit.loc[res_u.eq("WIN")] = dec_odds.loc[res_u.eq("WIN")] - 1.0
    profit.loc[res_u.eq("LOSS")] = -1.0
    profit.loc[res_u.eq("PUSH")] = 0.0
    profit = profit.where(dec_odds.notna())
    work["profit_units"] = profit
    # Outcome flags
    res = work.get("result")
    win = res.astype(str).str.upper().eq("WIN")
    loss = res.astype(str).str.upper().eq("LOSS")
    push = res.astype(str).str.upper().eq("PUSH")
    out: Dict[str, Any] = {"rows": int(len(work))}
    for key in ("MONEYLINE", "SPREAD", "TOTAL"):
        sub = work[work["type"].astype(str).str.upper().eq(key)]
        if sub.empty:
            out[key] = {"rows": 0}
            continue
        s_win = sub["result"].astype(str).str.upper().eq("WIN").sum()
        s_loss = sub["result"].astype(str).str.upper().eq("LOSS").sum()
        s_push = sub["result"].astype(str).str.upper().eq("PUSH").sum()
        denom = max(1, (s_win + s_loss))
        out[key] = {
            "rows": int(len(sub)),
            "wins": int(s_win),
            "losses": int(s_loss),
            "pushes": int(s_push),
            "win_rate": float(s_win) / float(denom),
            "ev_mean": float(pd.to_numeric(sub.get("ev_units"), errors="coerce").mean() or 0.0),
            "ev_median": float(pd.to_numeric(sub.get("ev_units"), errors="coerce").median() or 0.0),
            "profit_sum_units": float(pd.to_numeric(sub.get("profit_units"), errors="coerce").sum() or 0.0),
            "profit_mean_units": float(pd.to_numeric(sub.get("profit_units"), errors="coerce").mean() or 0.0),
        }
        try:
            n_pl = int(pd.to_numeric(sub.get("profit_units"), errors="coerce").count())
            out[key]["profit_n"] = n_pl
            out[key]["roi_units_per_bet"] = float(out[key]["profit_sum_units"]) / float(max(1, n_pl))
        except Exception:
            pass
        # Confidence-tier breakdown
        tiers = ["High", "Medium", "Low"]
        tier_stats: Dict[str, Any] = {}
        for t in tiers:
            tsub = sub[sub["confidence"].astype(str).str.upper().eq(t.upper())]
            if tsub.empty:
                tier_stats[t] = {"rows": 0}
                continue
            tw = tsub["result"].astype(str).str.upper().eq("WIN").sum()
            tl = tsub["result"].astype(str).str.upper().eq("LOSS").sum()
            td = max(1, (tw + tl))
            tier_stats[t] = {
                "rows": int(len(tsub)),
                "wins": int(tw),
                "losses": int(tl),
                "win_rate": float(tw) / float(td),
                "ev_mean": float(pd.to_numeric(tsub.get("ev_units"), errors="coerce").mean() or 0.0),
                "profit_sum_units": float(pd.to_numeric(tsub.get("profit_units"), errors="coerce").sum() or 0.0),
            }
            try:
                n_pl = int(pd.to_numeric(tsub.get("profit_units"), errors="coerce").count())
                tier_stats[t]["profit_n"] = n_pl
                tier_stats[t]["roi_units_per_bet"] = float(tier_stats[t]["profit_sum_units"]) / float(max(1, n_pl))
            except Exception:
                pass
        out[key]["tiers"] = tier_stats
    return out


def main():
    ap = argparse.ArgumentParser(description="Backtest EV-based recommendations across weeks")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--min-ev", type=float, default=None, help="Override RECS_MIN_EV_PCT (percent)")
    ap.add_argument("--one-per-game", type=str, default=None, help="Override RECS_ONE_PER_GAME (true/false)")
    ap.add_argument("--out-tag", type=str, default=None, help="Optional suffix to avoid overwriting outputs")
    ap.add_argument("--compute-mc", action="store_true", help="Compute MC probs per week (requires SIM_COMPUTE_ON_REQUEST=1)")
    ap.add_argument("--mc-sims", type=int, default=None, help="Override SIM_N_SIMS_ON_REQUEST when --compute-mc is enabled")
    ap.add_argument("--strict-gates", action="store_true", help="Apply gates even on final games (recommended for true backtests)")
    ap.add_argument(
        "--walk-forward",
        action="store_true",
        help="Train models on weeks < wk and predict wk (walk-forward). Recommended for trustworthy backtests.",
    )
    ap.add_argument(
        "--line-mode",
        type=str,
        default="auto",
        choices=["auto", "open", "close"],
        help="Which market line to use for ATS/TOTAL EV + grading: auto (default), open, or close.",
    )
    args = ap.parse_args()

    # Optional env overrides to match production recs filtering
    if args.min_ev is not None:
        os.environ["RECS_MIN_EV_PCT"] = str(float(args.min_ev))
    if args.one_per_game is not None:
        os.environ["RECS_ONE_PER_GAME"] = str(args.one_per_game)
    if args.mc_sims is not None:
        os.environ["SIM_N_SIMS_ON_REQUEST"] = str(int(args.mc_sims))

    season = int(args.season)
    end_week = int(args.end_week)
    lookback = max(1, int(args.lookback))
    v = _build_window(
        season,
        end_week,
        lookback,
        compute_mc=bool(args.compute_mc),
        walk_forward=bool(args.walk_forward),
    )
    if v is None or v.empty:
        print("No window data; nothing to backtest.")
        return 0

    # Diagnostics: what prediction source are we actually using?
    try:
        if 'pred_source' in v.columns:
            vc = v['pred_source'].astype(str).fillna('')
            counts = vc.value_counts(dropna=False).to_dict()
            ignore_locked = str(os.environ.get('PRED_IGNORE_LOCKED', '0')).strip().lower() in {'1','true','yes','on'}
            locked_n = int(counts.get('locked', 0))
            total_n = int(len(v))
            print(f"Prediction sources in window: {counts}")
            if (not ignore_locked) and total_n > 0 and (locked_n / total_n) >= 0.50:
                print("WARNING: Backtest window is dominated by pred_source='locked'. Consider setting PRED_IGNORE_LOCKED=1 for historical backtests, or materialize predictions_week.csv for the window.")
            if ignore_locked and locked_n > 0:
                print("NOTE: PRED_IGNORE_LOCKED=1 is set but some locked rows are still present in the view; investigate upstream merges.")
    except Exception:
        pass

    # Generate recommendations per game row
    rows: List[Dict[str, Any]] = []
    for _, row in v.iterrows():
        try:
            picks = _compute_recommendations_for_row(
                row,
                strict_gates=bool(args.strict_gates),
                line_mode=str(args.line_mode),
            )
        except Exception:
            picks = []
        if not picks:
            continue
        for r in picks:
            rec = dict(r)
            # Add identifiers to facilitate joins/analysis
            rec["game_id"] = row.get("game_id")
            rec["season"] = row.get("season")
            rec["week"] = row.get("week")
            rec["home_team"] = row.get("home_team")
            rec["away_team"] = row.get("away_team")
            rows.append(rec)

    if not rows:
        print("No recommendations produced from window.")
        return 0

    df = pd.DataFrame(rows)
    out_name = f"{season}_wk{end_week}"
    if args.out_tag:
        out_name = f"{out_name}_{str(args.out_tag).strip()}"
    out_dir = DATA_DIR / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write details
    det_fp = out_dir / "recs_backtest_details.csv"
    try:
        df.to_csv(det_fp, index=False)
    except Exception:
        pass
    # Summarize
    summ = _summarize_recs(df)
    summ["meta"] = {
        "season": season,
        "end_week": end_week,
        "lookback": lookback,
        "rows": int(len(df)),
        "strict_gates": bool(args.strict_gates),
        "line_mode": str(args.line_mode),
    }
    try:
        with open(out_dir / "recs_backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
    except Exception:
        pass
    # Markdown summary
    lines: List[str] = []
    lines.append(f"# Recommendations Backtest (Season {season}, Weeks {max(1, end_week - lookback + 1)}–{end_week})\n")
    for key in ("MONEYLINE", "SPREAD", "TOTAL"):
        stats = summ.get(key) or {}
        lines.append(f"## {key}\n")
        lines.append(f"- Rows: {stats.get('rows', 0)}\n")
        lines.append(f"- Win Rate: {stats.get('win_rate', float('nan')):.3f}\n")
        lines.append(f"- EV Mean: {stats.get('ev_mean', float('nan')):.4f}\n")
        if stats.get("profit_n"):
            lines.append(f"- Realized ROI (units/bet): {stats.get('roi_units_per_bet', float('nan')):.4f} (n={stats.get('profit_n')}, sum={stats.get('profit_sum_units', float('nan')):.2f})\n")
        tiers = (stats.get("tiers") or {})
        if tiers:
            lines.append("- Tiers:")
            for t, st in tiers.items():
                tier_bits = [
                    f"rows={st.get('rows',0)}",
                    f"win_rate={(st.get('win_rate') if st.get('win_rate') is not None else float('nan')):.3f}",
                    f"ev_mean={(st.get('ev_mean') if st.get('ev_mean') is not None else float('nan')):.4f}",
                ]
                if st.get("profit_n"):
                    tier_bits.append(f"roi={st.get('roi_units_per_bet', float('nan')):.4f} (n={st.get('profit_n')}, sum={st.get('profit_sum_units', float('nan')):.2f})")
                lines.append(f"  - {t}: " + ", ".join(tier_bits))
        lines.append("")
    try:
        (out_dir / "recs_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote recs backtest: details={det_fp}, summary={out_dir / 'recs_backtest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
