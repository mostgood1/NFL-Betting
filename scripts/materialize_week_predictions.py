import os
import sys
from pathlib import Path
import json
import pandas as pd
    
# Ensure repo root is on sys.path so `nfl_compare` is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def main(
    season: int | None = None,
    week: int | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    out_path: str | None = None,
) -> int:
    cw = (season, week) if (season and week) else _load_current_week()
    if not cw and season is None:
        print("Could not determine current season/week; aborting")
        return 2
    if season is None:
        season = cw[0]

    if week is not None:
        weeks = [int(week)]
    else:
        if start_week is None and end_week is None and cw:
            weeks = [int(cw[1])]
        else:
            if start_week is None or end_week is None:
                print("Need --week or both --start-week and --end-week")
                return 2
            weeks = list(range(int(start_week), int(end_week) + 1))

    out_fp = Path(out_path) if out_path else (DATA_DIR / "predictions_week.csv")
    print(f"Materializing predictions for season={season} weeks={weeks[0]}..{weeks[-1]} -> {out_fp}")

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

    model_path = BASE_DIR / "nfl_compare" / "models" / "nfl_models.joblib"
    models = joblib.load(model_path)

    out_week_dfs: list[pd.DataFrame] = []
    for wk in weeks:
        sel = feat[(pd.to_numeric(feat.get("season"), errors="coerce") == season) & (pd.to_numeric(feat.get("week"), errors="coerce") == wk)].copy()
        # Restrict to matchups present in lines.csv for authoritative subset (reduces playoff week to actual scheduled games)
        try:
            applied_lines_subset = False
            if lines is not None and not getattr(lines, 'empty', True) and {'home_team','away_team'}.issubset(lines.columns):
                lsw = lines.copy()
                for c in ('season','week'):
                    if c in lsw.columns:
                        lsw[c] = pd.to_numeric(lsw[c], errors='coerce')
                lsw = lsw[(lsw.get('season') == season) & (lsw.get('week') == wk)]
                if not lsw.empty:
                    # Normalize team names to match features
                    try:
                        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                        for col in ('home_team','away_team'):
                            if col in lsw.columns:
                                lsw[col] = lsw[col].astype(str).apply(_norm_team)
                            if col in sel.columns:
                                sel[col] = sel[col].astype(str).apply(_norm_team)
                    except Exception:
                        pass
                    keys = set((lsw['home_team'].astype(str) + '|' + lsw['away_team'].astype(str)).tolist()) | set((lsw['away_team'].astype(str) + '|' + lsw['home_team'].astype(str)).tolist())
                    if {'home_team','away_team'}.issubset(sel.columns) and len(keys) > 0:
                        sel['__mk'] = sel['home_team'].astype(str) + '|' + sel['away_team'].astype(str)
                        sel = sel[sel['__mk'].isin(keys)].copy()
                        sel = sel.drop(columns=['__mk'])
                        applied_lines_subset = True
            # If lines filter didn't apply (empty subset), fall back to scheduled games with valid dates for this week
            if (not applied_lines_subset) and games is not None and not getattr(games, 'empty', True):
                gsw = games.copy()
                for c in ('season','week'):
                    if c in gsw.columns:
                        gsw[c] = pd.to_numeric(gsw[c], errors='coerce')
                # prefer 'game_date' else 'date'
                date_col = 'game_date' if 'game_date' in gsw.columns else ('date' if 'date' in gsw.columns else None)
                if date_col is not None:
                    gsw = gsw[(gsw.get('season') == season) & (gsw.get('week') == wk)]
                    gsw = gsw[gsw[date_col].notna() & (gsw[date_col].astype(str).str.len() > 0)]
                    # Prefer strict filter by game_id if present to avoid synthetic duplicates
                    if not gsw.empty and 'game_id' in gsw.columns and 'game_id' in sel.columns:
                        gids = set(gsw['game_id'].astype(str).tolist())
                        sel = sel[sel['game_id'].astype(str).isin(gids)].copy()
                    elif not gsw.empty and {'home_team','away_team'}.issubset(gsw.columns) and {'home_team','away_team'}.issubset(sel.columns):
                        try:
                            from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                            for col in ('home_team','away_team'):
                                if col in gsw.columns:
                                    gsw[col] = gsw[col].astype(str).apply(_norm_team)
                                if col in sel.columns:
                                    sel[col] = sel[col].astype(str).apply(_norm_team)
                        except Exception:
                            pass
                        keys = set((gsw['home_team'].astype(str) + '|' + gsw['away_team'].astype(str)).tolist()) | set((gsw['away_team'].astype(str) + '|' + gsw['home_team'].astype(str)).tolist())
                        sel['__mk'] = sel['home_team'].astype(str) + '|' + sel['away_team'].astype(str)
                        sel = sel[sel['__mk'].isin(keys)].copy()
                        sel = sel.drop(columns=['__mk'])
        except Exception:
            pass
        if sel.empty:
            print(f"Week {wk}: no feature rows; skipping")
            continue
        # Deduplicate by game_id (lines merge can create multiple rows per game).
        # Prefer rows where close_* columns are present.
        if 'game_id' in sel.columns:
            try:
                has_close = sel[[c for c in ['close_spread_home','close_total'] if c in sel.columns]].notna().any(axis=1)
                sel['__has_close'] = has_close.astype(int)
                sel = sel.sort_values(['__has_close'], ascending=False)
                sel = sel.drop_duplicates(subset=['game_id'], keep='first')
                sel = sel.drop(columns=['__has_close'])
            except Exception:
                sel = sel.drop_duplicates(subset=['game_id'], keep='first')
        # If game_id uniqueness isn't sufficient (e.g., synthetic IDs), dedupe by matchup.
        if {'home_team','away_team'}.issubset(sel.columns):
            try:
                sel['__match_key'] = sel['home_team'].astype(str) + '|' + sel['away_team'].astype(str)
                # Prefer rows where close_* are present
                has_close2 = sel[[c for c in ['close_spread_home','close_total'] if c in sel.columns]].notna().any(axis=1)
                sel['__has_close2'] = has_close2.astype(int)
                # If date present, prefer earliest scheduled
                if 'game_date' in sel.columns:
                    sel['__dt'] = pd.to_datetime(sel['game_date'], errors='coerce')
                elif 'date' in sel.columns:
                    sel['__dt'] = pd.to_datetime(sel['date'], errors='coerce')
                else:
                    sel['__dt'] = pd.NaT
                sel = sel.sort_values(['__match_key','__has_close2','__dt'], ascending=[True, False, True])
                sel = sel.drop_duplicates(subset=['__match_key'], keep='first')
                sel = sel.drop(columns=['__match_key','__has_close2','__dt'])
            except Exception:
                sel = sel.drop_duplicates(subset=['home_team','away_team'], keep='first')

        pred = model_predict(models, sel)
        pred = _ensure_pred_columns(pred)

        # Optional: blend model margin/total toward market values and apply to team points for this week
        try:
            bm_raw = os.environ.get("RECS_MARKET_BLEND_MARGIN", "0")
            bt_raw = os.environ.get("RECS_MARKET_BLEND_TOTAL", "0")
            apply_pts_flag = os.environ.get("RECS_MARKET_BLEND_APPLY_POINTS", "0")
            bm = float(bm_raw) if bm_raw not in (None, "") else 0.0
            bt = float(bt_raw) if bt_raw not in (None, "") else 0.0
            bm = max(0.0, min(1.0, bm))
            bt = max(0.0, min(1.0, bt))
            apply_pts = str(apply_pts_flag).strip().lower() in {"1", "true", "yes", "on"}
            if apply_pts and (bm > 0.0 or bt > 0.0):
                def _pick(row: pd.Series, *cols: str):
                    for c in cols:
                        if c in row.index:
                            v = row.get(c)
                            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                                return v
                    return None

                rows = []
                for _, r in pred.iterrows():
                    margin = r.get("pred_margin")
                    total = r.get("pred_total")
                    # Prefer market/open values for upcoming games
                    m_spread = _pick(r, "market_spread_home", "spread_home", "open_spread_home", "close_spread_home")
                    m_total = _pick(r, "market_total", "total", "open_total", "close_total")
                    try:
                        if bm > 0.0 and margin is not None and m_spread is not None:
                            mmkt = -float(m_spread)
                            margin = (1.0 - bm) * float(margin) + bm * mmkt
                        if bt > 0.0 and total is not None and m_total is not None:
                            total = (1.0 - bt) * float(total) + bt * float(m_total)
                        if (margin is not None) and (total is not None):
                            th = 0.5 * (float(total) + float(margin))
                            ta = float(total) - th
                            r["pred_home_points"] = th
                            r["pred_away_points"] = ta
                            r["pred_total"] = float(total)
                            r["pred_margin"] = float(margin)
                    except Exception:
                        pass
                    rows.append(r)
                pred = pd.DataFrame(rows)
        except Exception:
            pass

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

        # Final safety filter: enforce authoritative lines subset when enabled, else restrict to scheduled games
        try:
            # First, apply authoritative lines subset if requested
            auth_flag = os.environ.get("RECS_AUTHORITATIVE_LINES_SUBSET", "0")
            use_auth = str(auth_flag).strip().lower() in {"1", "true", "yes", "on"}
            if use_auth and lines is not None and not getattr(lines, 'empty', True) and {'home_team','away_team'}.issubset(out_df.columns):
                lsw = lines.copy()
                for c in ('season','week'):
                    if c in lsw.columns:
                        lsw[c] = pd.to_numeric(lsw[c], errors='coerce')
                lsw = lsw[(lsw.get('season') == season) & (lsw.get('week') == wk)]
                if not lsw.empty and {'home_team','away_team'}.issubset(lsw.columns):
                    try:
                        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                        for col in ('home_team','away_team'):
                            if col in lsw.columns:
                                lsw[col] = lsw[col].astype(str).apply(_norm_team)
                            if col in out_df.columns:
                                out_df[col] = out_df[col].astype(str).apply(_norm_team)
                    except Exception:
                        pass
                    keys = set((lsw['home_team'].astype(str) + '|' + lsw['away_team'].astype(str)).tolist()) | set((lsw['away_team'].astype(str) + '|' + lsw['home_team'].astype(str)).tolist())
                    out_df['__mk'] = out_df['home_team'].astype(str) + '|' + out_df['away_team'].astype(str)
                    out_df = out_df[out_df['__mk'].isin(keys)].copy()
                    out_df = out_df.drop(columns=['__mk'])
            # If lines-gating not applied or yielded empty, fall back to games with valid dates using game_id
            if games is not None and not getattr(games, 'empty', True) and 'game_id' in out_df.columns:
                gsw = games.copy()
                for c in ('season','week'):
                    if c in gsw.columns:
                        gsw[c] = pd.to_numeric(gsw[c], errors='coerce')
                date_col = 'game_date' if 'game_date' in gsw.columns else ('date' if 'date' in gsw.columns else None)
                if date_col is not None:
                    gsw = gsw[(gsw.get('season') == season) & (gsw.get('week') == wk)]
                    gsw = gsw[gsw[date_col].notna() & (gsw[date_col].astype(str).str.len() > 0)]
                    if not gsw.empty and 'game_id' in gsw.columns:
                        gids = set(gsw['game_id'].astype(str).tolist())
                        out_df = out_df[out_df['game_id'].astype(str).isin(gids)].copy()
        except Exception:
            pass

        out_week_dfs.append(out_df)
        print(f"Week {wk}: staged {len(out_df)} rows")

    if not out_week_dfs:
        print("No weeks materialized; nothing to write")
        return 0

    out_all = pd.concat(out_week_dfs, ignore_index=True)
    # Merge into predictions_week.csv (preserve other weeks; replace only this season/week range)
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
        ex = existing.copy()
        for c in ("season", "week"):
            if c in ex.columns:
                ex[c] = pd.to_numeric(ex[c], errors="coerce")
        rem = ex[~(
            (pd.to_numeric(ex.get("season"), errors="coerce") == season)
            & (pd.to_numeric(ex.get("week"), errors="coerce").isin(weeks))
        )]
        combined = pd.concat([rem, out_all], ignore_index=True)
    else:
        combined = out_all

    # Final de-dupe
    try:
        if 'game_id' in combined.columns:
            combined = combined.drop_duplicates(subset=['season','week','game_id'], keep='last')
        elif {'season','week','home_team','away_team'}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=['season','week','home_team','away_team'], keep='last')
    except Exception:
        pass

    combined.to_csv(out_fp, index=False)
    print(f"Wrote {len(out_all)} rows to {out_fp} (total {len(combined)} rows)")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Materialize model predictions for a specific season/week")
    ap.add_argument("--season", type=int, default=None, help="Season to materialize")
    ap.add_argument("--week", type=int, default=None, help="Week to materialize")
    ap.add_argument("--start-week", type=int, default=None, help="Start week (inclusive) to materialize")
    ap.add_argument("--end-week", type=int, default=None, help="End week (inclusive) to materialize")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default nfl_compare/data/predictions_week.csv)")
    args = ap.parse_args()
    raise SystemExit(
        main(
            season=args.season,
            week=args.week,
            start_week=args.start_week,
            end_week=args.end_week,
            out_path=args.out,
        )
    )
