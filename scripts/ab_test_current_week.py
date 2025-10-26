import os
from pathlib import Path
import json
import pandas as pd

# Use the same data directory rules as the app
BASE_DIR = Path(__file__).resolve().parents[1]
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (BASE_DIR / "nfl_compare" / "data")


def _load_current_week() -> tuple[int, int] | None:
    """Return (season, week) from nfl_compare/data/current_week.json if present,
    else infer from games.csv by picking the latest season and the earliest week with any game not finalized.
    """
    cfg = DATA_DIR / "current_week.json"
    if cfg.exists():
        try:
            j = json.loads(cfg.read_text(encoding="utf-8"))
            s = int(j.get("season"))
            w = int(j.get("week"))
            return (s, w)
        except Exception:
            pass
    # Fallback: infer from games.csv
    try:
        games_fp = DATA_DIR / "games.csv"
        if not games_fp.exists():
            return None
        g = pd.read_csv(games_fp)
        if g.empty:
            return None
        for c in ("season", "week"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
        latest_season = int(pd.to_numeric(g.get("season"), errors="coerce").dropna().max())
        gs = g[g["season"] == latest_season].copy()
        # Prefer earliest week with any non-final games
        has_scores = {"home_score", "away_score"}.issubset(gs.columns)
        if has_scores:
            hs = pd.to_numeric(gs["home_score"], errors="coerce")
            as_ = pd.to_numeric(gs["away_score"], errors="coerce")
            gs["_done"] = hs.notna() & as_.notna()
            comp = gs.groupby("week")["_done"].agg(["sum", "count"]).reset_index()
            active = comp[comp["sum"] < comp["count"]]
            if not active.empty:
                w = int(pd.to_numeric(active["week"], errors="coerce").dropna().min())
                return latest_season, w
        # Fallback: pick min week whose max game datetime is in the future
        dt_col = "game_date" if "game_date" in gs.columns else ("date" if "date" in gs.columns else None)
        if dt_col:
            gs[dt_col] = pd.to_datetime(gs[dt_col], errors="coerce")
            grp = gs.groupby("week")[dt_col].agg(["min", "max"]).reset_index().rename(columns={"min": "min_dt", "max": "max_dt"})
            now = pd.Timestamp.now()
            pending = grp[grp["max_dt"] >= now]
            if not pending.empty:
                w = int(pd.to_numeric(pending["week"], errors="coerce").dropna().min())
                return latest_season, w
            # Else latest past week
            last = grp.sort_values("max_dt").iloc[-1]
            return latest_season, int(last["week"])
        # Last resort: max week in latest season
        wmax = int(pd.to_numeric(gs["week"], errors="coerce").dropna().max())
        return latest_season, wmax
    except Exception:
        return None


def _load_baseline_predictions() -> pd.DataFrame:
    """Load baseline predictions from CSVs.
    Priority: predictions_locked.csv > predictions_week.csv > predictions.csv.
    Also checks top-level ./data as fallback.
    """
    paths = [
        DATA_DIR / "predictions_locked.csv",
        DATA_DIR / "predictions_week.csv",
        DATA_DIR / "predictions.csv",
        BASE_DIR / "data" / "predictions_locked.csv",
        BASE_DIR / "data" / "predictions_week.csv",
        BASE_DIR / "data" / "predictions.csv",
    ]
    dfs: list[pd.DataFrame] = []
    for p in paths:
        if p.exists():
            try:
                d = pd.read_csv(p)
                d["_src"] = p.name
                dfs.append(d)
                # First existing file in priority order is enough
                break
            except Exception:
                continue
    if not dfs:
        return pd.DataFrame()
    df = dfs[0]
    # Normalize types
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure presence of pred_total, pred_margin, prob_home_win, pred_home_points, pred_away_points.
    Try to derive missing ones when possible.
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    # Normalize alt names
    if "pred_home_win_prob" in d.columns and "prob_home_win" not in d.columns:
        d["prob_home_win"] = d["pred_home_win_prob"]
    # If only scores present, compute
    if {"pred_home_points", "pred_away_points"}.issubset(d.columns):
        if "pred_total" not in d.columns:
            try:
                d["pred_total"] = pd.to_numeric(d["pred_home_points"], errors="coerce") + pd.to_numeric(d["pred_away_points"], errors="coerce")
            except Exception:
                pass
        if "pred_margin" not in d.columns:
            try:
                d["pred_margin"] = pd.to_numeric(d["pred_home_points"], errors="coerce") - pd.to_numeric(d["pred_away_points"], errors="coerce")
            except Exception:
                pass
    # If margin and total present, derive points
    if {"pred_total", "pred_margin"}.issubset(d.columns):
        try:
            t = pd.to_numeric(d["pred_total"], errors="coerce")
            m = pd.to_numeric(d["pred_margin"], errors="coerce")
            if "pred_home_points" not in d.columns:
                d["pred_home_points"] = (t + m) / 2.0
            if "pred_away_points" not in d.columns:
                d["pred_away_points"] = t - d["pred_home_points"]
        except Exception:
            pass
    return d


def main():
    cw = _load_current_week()
    if not cw:
        print("Could not determine current season/week; aborting")
        return 2
    season, week = cw
    print(f"A/B test for season={season} week={week}")

    # Load data + features
    from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
    from nfl_compare.src.weather import load_weather_for_games
    from nfl_compare.src.features import merge_features
    from nfl_compare.src.models import predict as model_predict
    import joblib

    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)
    feat = merge_features(games, stats, lines, wx)
    # Filter features to current week
    feat_sw = feat[(pd.to_numeric(feat.get("season"), errors="coerce") == season) & (pd.to_numeric(feat.get("week"), errors="coerce") == week)].copy()
    if feat_sw.empty:
        print("No feature rows for current week; nothing to compare.")
        return 0

    # New model predictions (A)
    model_path = BASE_DIR / "nfl_compare" / "models" / "nfl_models.joblib"
    models = joblib.load(model_path)
    pred_new = model_predict(models, feat_sw)
    pred_new = _ensure_pred_columns(pred_new)
    keep_new = [c for c in ["game_id", "season", "week", "home_team", "away_team", "pred_total", "pred_margin", "prob_home_win", "pred_home_points", "pred_away_points"] if c in pred_new.columns]
    pred_new = pred_new[keep_new].copy()
    pred_new = pred_new.rename(columns={
        "pred_total": "pred_total_new",
        "pred_margin": "pred_margin_new",
        "prob_home_win": "prob_home_win_new",
        "pred_home_points": "pred_home_points_new",
        "pred_away_points": "pred_away_points_new",
    })

    # Baseline predictions (B)
    base = _load_baseline_predictions()
    base = base[(pd.to_numeric(base.get("season"), errors="coerce") == season) & (pd.to_numeric(base.get("week"), errors="coerce") == week)].copy()
    base = _ensure_pred_columns(base)
    keep_base = [c for c in ["game_id", "season", "week", "home_team", "away_team", "pred_total", "pred_margin", "prob_home_win", "pred_home_points", "pred_away_points", "_src"] if c in base.columns]
    base = base[keep_base].copy()
    base = base.rename(columns={
        "pred_total": "pred_total_base",
        "pred_margin": "pred_margin_base",
        "prob_home_win": "prob_home_win_base",
        "pred_home_points": "pred_home_points_base",
        "pred_away_points": "pred_away_points_base",
    })

    # Join on game_id if available, else on match keys
    on_cols = ["game_id"] if ("game_id" in pred_new.columns and "game_id" in base.columns) else [c for c in ["season","week","home_team","away_team"] if c in pred_new.columns and c in base.columns]
    if not on_cols:
        print("Could not align new predictions with baseline (missing keys).")
        return 3
    cmp_df = pd.merge(base, pred_new, on=on_cols, how="outer", suffixes=("_base", "_new"))

    # Compute deltas
    for col in ["pred_total", "pred_margin", "prob_home_win", "pred_home_points", "pred_away_points"]:
        b = f"{col}_base"; n = f"{col}_new"; d = f"d_{col}"
        if b in cmp_df.columns and n in cmp_df.columns:
            cmp_df[d] = pd.to_numeric(cmp_df[n], errors="coerce") - pd.to_numeric(cmp_df[b], errors="coerce")

    # Winner change flag
    def _winner_from_margin(m):
        try:
            m = float(m)
        except Exception:
            return None
        return "home" if m > 0 else ("away" if m < 0 else "push")

    if {"pred_margin_base", "pred_margin_new"}.issubset(cmp_df.columns):
        cmp_df["winner_base"] = cmp_df["pred_margin_base"].map(_winner_from_margin)
        cmp_df["winner_new"] = cmp_df["pred_margin_new"].map(_winner_from_margin)
        cmp_df["winner_changed"] = (cmp_df["winner_base"] != cmp_df["winner_new"])

    # Summary
    def _mad(s):
        s = pd.to_numeric(s, errors="coerce")
        return float(s.abs().mean()) if s.notna().any() else 0.0

    n_games = len(cmp_df)
    mad_total = _mad(cmp_df.get("d_pred_total"))
    mad_margin = _mad(cmp_df.get("d_pred_margin"))
    wins_changed = int(cmp_df.get("winner_changed", pd.Series(dtype=bool)).fillna(False).sum()) if "winner_changed" in cmp_df.columns else 0
    print(f"Games compared: {n_games}")
    print(f"Mean |Δ total|: {mad_total:.3f}")
    print(f"Mean |Δ margin|: {mad_margin:.3f}")
    print(f"Winner changes: {wins_changed}")

    # Save detailed diff
    out_fp = DATA_DIR / f"ab_diff_{season}_wk{week}.csv"
    try:
        cmp_df.to_csv(out_fp, index=False)
        print(f"Wrote detailed diff to {out_fp}")
    except Exception as e:
        print(f"Failed writing diff CSV: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
