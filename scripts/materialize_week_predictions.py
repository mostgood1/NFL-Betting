import os
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (BASE_DIR / "nfl_compare" / "data")


def _load_current_week() -> tuple[int, int] | None:
    cfg = DATA_DIR / "current_week.json"
    if cfg.exists():
        try:
            j = json.loads(cfg.read_text(encoding="utf-8"))
            s = int(j.get("season"))
            w = int(j.get("week"))
            return (s, w)
        except Exception:
            pass
    # Fallback inference from games.csv
    try:
        g = pd.read_csv(DATA_DIR / "games.csv")
        if g.empty:
            return None
        for c in ("season", "week"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
        latest_s = int(pd.to_numeric(g.get("season"), errors="coerce").dropna().max())
        gs = g[g["season"] == latest_s].copy()
        has_scores = {"home_score", "away_score"}.issubset(gs.columns)
        if has_scores:
            hs = pd.to_numeric(gs["home_score"], errors="coerce")
            as_ = pd.to_numeric(gs["away_score"], errors="coerce")
            gs["_done"] = hs.notna() & as_.notna()
            comp = gs.groupby("week")["_done"].agg(["sum", "count"]).reset_index()
            active = comp[comp["sum"] < comp["count"]]
            if not active.empty:
                w = int(pd.to_numeric(active["week"], errors="coerce").dropna().min())
                return latest_s, w
        # Fallback by datetime
        dt_col = "game_date" if "game_date" in gs.columns else ("date" if "date" in gs.columns else None)
        if dt_col:
            gs[dt_col] = pd.to_datetime(gs[dt_col], errors="coerce")
            grp = gs.groupby("week")[dt_col].agg(["min", "max"]).reset_index().rename(columns={"min": "min_dt", "max": "max_dt"})
            now = pd.Timestamp.now()
            pending = grp[grp["max_dt"] >= now]
            if not pending.empty:
                w = int(pd.to_numeric(pending["week"], errors="coerce").dropna().min())
                return latest_s, w
            last = grp.sort_values("max_dt").iloc[-1]
            return latest_s, int(last["week"])
        wmax = int(pd.to_numeric(gs["week"], errors="coerce").dropna().max())
        return latest_s, wmax
    except Exception:
        return None


def _ensure_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    if "pred_home_win_prob" in d.columns and "prob_home_win" not in d.columns:
        d["prob_home_win"] = d["pred_home_win_prob"]
    if {"pred_home_points", "pred_away_points"}.issubset(d.columns):
        if "pred_total" not in d.columns:
            d["pred_total"] = pd.to_numeric(d["pred_home_points"], errors="coerce") + pd.to_numeric(d["pred_away_points"], errors="coerce")
        if "pred_margin" not in d.columns:
            d["pred_margin"] = pd.to_numeric(d["pred_home_points"], errors="coerce") - pd.to_numeric(d["pred_away_points"], errors="coerce")
    return d


def main(season: int | None = None, week: int | None = None) -> int:
    cw = (season, week) if (season and week) else _load_current_week()
    if not cw:
        print("Could not determine current season/week; aborting")
        return 2
    season, week = cw
    print(f"Materializing predictions for season={season} week={week}")

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
    if feat is None or feat.empty:
        print("No features available; aborting")
        return 3

    sel = feat[(pd.to_numeric(feat.get("season"), errors="coerce") == season) & (pd.to_numeric(feat.get("week"), errors="coerce") == week)].copy()
    if sel.empty:
        print("No feature rows for selected week; nothing to write")
        return 0

    model_path = BASE_DIR / "nfl_compare" / "models" / "nfl_models.joblib"
    models = joblib.load(model_path)
    pred = model_predict(models, sel)
    pred = _ensure_pred_columns(pred)
    keep = [
        c for c in [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "pred_home_points",
            "pred_away_points",
            "pred_total",
            "pred_margin",
            "prob_home_win",
        ]
        if c in pred.columns
    ]
    out_df = pred[keep].copy()

    # Merge into predictions_week.csv (preserve other weeks; replace only this season/week)
    out_fp = DATA_DIR / "predictions_week.csv"
    existing = None
    if out_fp.exists():
        try:
            existing = pd.read_csv(out_fp)
            for c in ("season", "week"):
                if c in existing.columns:
                    existing[c] = pd.to_numeric(existing[c], errors="coerce")
        except Exception:
            existing = None
    if existing is not None and not existing.empty:
        rem = existing[~((pd.to_numeric(existing.get("season"), errors="coerce") == season) & (pd.to_numeric(existing.get("week"), errors="coerce") == week))]
        combined = pd.concat([rem, out_df], ignore_index=True)
    else:
        combined = out_df

    combined.to_csv(out_fp, index=False)
    print(f"Wrote {len(out_df)} rows to {out_fp} (total {len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
