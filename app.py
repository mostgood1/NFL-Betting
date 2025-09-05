from __future__ import annotations

import os
from pathlib import Path
import hashlib
from typing import Optional, Dict, Any, List

import pandas as pd
from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import json
import math
import traceback
from joblib import load as joblib_load


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "nfl_compare" / "data"
PRED_FILE = DATA_DIR / "predictions.csv"
PRED_WEEK_FILE = DATA_DIR / "predictions_week.csv"
ASSETS_FILE = DATA_DIR / "nfl_team_assets.json"
STADIUM_META_FILE = DATA_DIR / "stadium_meta.csv"
LOCATION_OVERRIDES_FILE = DATA_DIR / "game_location_overrides.csv"

app = Flask(__name__)


def _load_predictions() -> pd.DataFrame:
    """Load predictions.csv if present; return empty DataFrame if missing."""
    try:
        dfs = []
        if PRED_FILE.exists():
            dfs.append(pd.read_csv(PRED_FILE))
        # Optionally merge week-level predictions that include completed games
        if PRED_WEEK_FILE.exists():
            dfs.append(pd.read_csv(PRED_WEEK_FILE))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            # Drop duplicate game_ids favoring the last occurrence (week file overrides)
            if 'game_id' in df.columns:
                df = df.drop_duplicates(subset=['game_id'], keep='last')
            # Normalize typical columns if present
            for c in ("week", "season"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Sort by season/week/date if available
            sort_cols = [c for c in ["season", "week", "game_date"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            return df
    except Exception:
        # Fall through to empty frame
        pass
    return pd.DataFrame()


def _load_games() -> pd.DataFrame:
    """Load games.csv if present; return empty DataFrame if missing."""
    try:
        fp = DATA_DIR / "games.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for c in ("week", "season"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Sort for stability
            sort_cols = [c for c in ["season", "week", "game_date", "date"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _build_week_view(pred_df: pd.DataFrame, games_df: pd.DataFrame, season: Optional[int], week: Optional[int]) -> pd.DataFrame:
    """Combine games (for finals) with predictions (for upcoming) for a given season/week.
    - Always start from games for complete coverage of the week.
    - Left-merge prediction columns onto games by game_id when possible; fallback to team/week match.
    - Avoid overwriting core game fields from games.csv.
    """
    if games_df is None or games_df.empty:
        # No games file; fall back to predictions filtered by season/week
        out = pred_df.copy()
        if not out.empty:
            if season is not None and "season" in out.columns:
                out = out[out["season"] == season]
            if week is not None and "week" in out.columns:
                out = out[out["week"] == week]
        return out

    # Filter games to the requested group
    view = games_df.copy()
    if season is not None and "season" in view.columns:
        view = view[view["season"] == season]
    if week is not None and "week" in view.columns:
        view = view[view["week"] == week]
    if view.empty:
        # Try to synthesize rows from lines if schedule is missing
        try:
            from nfl_compare.src.data_sources import load_lines as _load_lines_for_view
            lines_all = _load_lines_for_view()
        except Exception:
            lines_all = None
        if lines_all is not None and not getattr(lines_all, 'empty', True):
            l = lines_all.copy()
            if season is not None and 'season' in l.columns:
                l = l[l['season'] == season]
            if week is not None and 'week' in l.columns:
                l = l[l['week'] == week]
            # Build minimal rows
            keep_cols = [c for c in ['season','week','game_id','game_date','date','home_team','away_team'] if c in l.columns]
            if not l.empty and keep_cols:
                extras = l[keep_cols].drop_duplicates()
                # Standardize datetime column
                if 'game_date' not in extras.columns and 'date' in extras.columns:
                    extras = extras.rename(columns={'date': 'game_date'})
                # Ensure required columns exist
                for c in ['season','week','home_team','away_team']:
                    if c not in extras.columns:
                        extras[c] = None
                view = extras
                # Continue merging predictions below
            else:
                return view
        else:
            return view

    # Prepare predictions to merge
    p = pred_df.copy() if pred_df is not None else pd.DataFrame()
    # Core keys we do NOT want to overwrite from games
    core_keys = {"season", "week", "game_id", "game_date", "date", "home_team", "away_team", "home_score", "away_score"}
    if not p.empty:
        drop_cols = [c for c in p.columns if c in core_keys]
        p_nokeys = p.drop(columns=drop_cols, errors="ignore")
        # Merge preference: by game_id if present in both
        if "game_id" in games_df.columns and "game_id" in pred_df.columns and not pred_df["game_id"].isna().all():
            merged = view.merge(pred_df[[c for c in pred_df.columns if c not in {"season","week","game_date","date","home_team","away_team","home_score","away_score"}]],
                                on="game_id", how="left")
        else:
            # Fallback join by team pairing within the same season/week slice
            key_cols = [c for c in ["home_team", "away_team"] if c in view.columns]
            if key_cols:
                merged = view.merge(p_nokeys, left_on=key_cols, right_on=[c for c in key_cols if c in p_nokeys.columns], how="left")
            else:
                merged = view
        return merged
    else:
        return view


def _attach_model_predictions(view_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: fill missing prediction columns for the given rows using trained models.
    - Skips when running on Render (RENDER env true).
    - Uses game_id to align where possible; falls back to (home_team, away_team, game_date/date).
    """
    try:
        if view_df is None or view_df.empty:
            return view_df
        # Always start with a shallow enrichment of lines/weather so odds display works even on Render
        out_base = view_df.copy()
        try:
            from nfl_compare.src.data_sources import load_games as ds_load_games, load_lines
            from nfl_compare.src.weather import load_weather_for_games
            line_cols = ['spread_home','total','moneyline_home','moneyline_away',
                         'spread_home_price','spread_away_price','total_over_price','total_under_price',
                         'close_spread_home','close_total']
            # Merge lines by game_id (preferred), fallback to team-based
            try:
                lines = load_lines()
            except Exception:
                lines = None
            if lines is not None and not getattr(lines, 'empty', True):
                cols_present = [c for c in line_cols if c in lines.columns]
                if 'game_id' in out_base.columns and 'game_id' in lines.columns:
                    out_base = out_base.merge(lines[['game_id'] + cols_present], on='game_id', how='left', suffixes=('', '_ln'))
                # If many missing, try supplement by (home_team, away_team)
                need_sup = False
                try:
                    need_sup = (('spread_home' in out_base.columns and out_base['spread_home'].isna().mean() > 0.5) or
                                ('total' in out_base.columns and out_base['total'].isna().mean() > 0.5))
                except Exception:
                    need_sup = True
                if need_sup and {'home_team','away_team'}.issubset(set(lines.columns)):
                    sup = lines[['home_team','away_team'] + cols_present].copy()
                    out_base = out_base.merge(sup, on=['home_team','away_team'], how='left', suffixes=('', '_ln2'))
                # Fill from any suffix variants (and create columns if missing)
                for c in line_cols:
                    c1, c2 = f"{c}_ln", f"{c}_ln2"
                    has_base = c in out_base.columns
                    v1 = out_base[c1] if c1 in out_base.columns else None
                    v2 = out_base[c2] if c2 in out_base.columns else None
                    if has_base:
                        if v1 is not None:
                            out_base[c] = out_base[c].where(out_base[c].notna(), v1)
                        if v2 is not None:
                            out_base[c] = out_base[c].where(out_base[c].notna(), v2)
                    else:
                        # Create the base column from right-hand values when not present
                        if v1 is not None:
                            out_base[c] = v1
                            has_base = True
                        if (not has_base) and v2 is not None:
                            out_base[c] = v2
                # Drop helper suffix columns
                drop_cols = [c for c in out_base.columns if c.endswith('_ln') or c.endswith('_ln2')]
                if drop_cols:
                    out_base = out_base.drop(columns=drop_cols)
                # If still no odds across the board, try a raw CSV fallback merge
                try:
                    need_fallback = True
                    for cc in ['moneyline_home','moneyline_away','spread_home','total','close_spread_home','close_total']:
                        if cc in out_base.columns and out_base[cc].notna().any():
                            need_fallback = False
                            break
                except Exception:
                    need_fallback = True
                if need_fallback:
                    import pandas as _pd
                    from pathlib import Path as _Path
                    csv_fp = BASE_DIR / 'nfl_compare' / 'data' / 'lines.csv'
                    if csv_fp.exists():
                        try:
                            df_csv_fb = _pd.read_csv(csv_fp)
                            # Try merge by game_id first
                            cols_present_fb = [c for c in line_cols if c in df_csv_fb.columns]
                            if 'game_id' in out_base.columns and 'game_id' in df_csv_fb.columns:
                                out_base = out_base.merge(df_csv_fb[['game_id'] + cols_present_fb], on='game_id', how='left', suffixes=('', '_lnfb'))
                            else:
                                if {'home_team','away_team'}.issubset(df_csv_fb.columns):
                                    out_base = out_base.merge(df_csv_fb[['home_team','away_team'] + cols_present_fb], on=['home_team','away_team'], how='left', suffixes=('', '_lnfb'))
                            for c in line_cols:
                                cfb = f"{c}_lnfb"
                                if c in out_base.columns and cfb in out_base.columns:
                                    out_base[c] = out_base[c].where(out_base[c].notna(), out_base[cfb])
                                elif cfb in out_base.columns and c not in out_base.columns:
                                    out_base[c] = out_base[cfb]
                            drop_fb = [c for c in out_base.columns if c.endswith('_lnfb')]
                            if drop_fb:
                                out_base = out_base.drop(columns=drop_fb)
                        except Exception:
                            pass
            # Weather/stadium (optional)
            try:
                games_all = ds_load_games()
                wx = load_weather_for_games(games_all)
            except Exception:
                wx = None
            if wx is not None and not getattr(wx, 'empty', True):
                wx_cols = ['game_id','date','home_team','away_team','wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']
                keep = [c for c in wx_cols if c in wx.columns]
                if keep:
                    out_base = out_base.merge(wx[keep], on=[c for c in ['game_id','date','home_team','away_team'] if c in out_base.columns and c in wx.columns], how='left', suffixes=('', '_wx'))
                    # Prefer non-null base, then fill from wx
                    for c in ['wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']:
                        cwx = f"{c}_wx"
                        if c in out_base.columns and cwx in out_base.columns:
                            out_base[c] = out_base[c].where(out_base[c].notna(), out_base[cwx])
                    drop_wx = [c for c in out_base.columns if c.endswith('_wx')]
                    if drop_wx:
                        out_base = out_base.drop(columns=drop_wx)
        except Exception:
            pass

        # If running on Render/minimal, skip heavy model predictions but keep enriched odds/weather
        if str(os.environ.get("RENDER", "").lower()) in {"1","true","yes"}:
            # Fallback: if finals have close lines but missing moneylines, derive approximate MLs using spread heuristic
            try:
                import numpy as np
                if {'home_score','away_score','moneyline_home','moneyline_away','close_spread_home'}.issubset(out_base.columns):
                    def _approx_ml_from_spread(sp):
                        # Simple mapping: convert spread to prob via logistic, then to American odds
                        try:
                            sp = float(sp)
                        except Exception:
                            return (None, None)
                        # scale tuned loosely for NFL; avoids extreme numbers
                        sigma = float(os.environ.get('NFL_SPREAD_PROB_SIGMA', '7.0'))
                        import math
                        p_home = 1.0/(1.0+math.exp(-( -sp)/sigma))  # negative spread favors home
                        p_home = min(max(p_home, 0.05), 0.95)
                        # Convert prob to fair American odds
                        def _prob_to_american(p):
                            if p <= 0 or p >= 1:
                                return None
                            if p >= 0.5:
                                return int(round(-p/(1-p)*100))
                            else:
                                return int(round((1-p)/p*100))
                        return (_prob_to_american(1-p_home), _prob_to_american(p_home))
                    finals = out_base[(out_base['home_score'].notna()) & (out_base['away_score'].notna())]
                    for idx, rr in finals.iterrows():
                        if (pd.isna(rr.get('moneyline_home')) or pd.isna(rr.get('moneyline_away'))):
                            sp = rr.get('close_spread_home') if pd.notna(rr.get('close_spread_home')) else rr.get('spread_home')
                            if pd.notna(sp):
                                ml_away, ml_home = _approx_ml_from_spread(sp)
                                if pd.isna(rr.get('moneyline_home')) and ml_home is not None:
                                    out_base.at[idx, 'moneyline_home'] = ml_home
                                if pd.isna(rr.get('moneyline_away')) and ml_away is not None:
                                    out_base.at[idx, 'moneyline_away'] = ml_away
            except Exception:
                pass
            return out_base

        # Lazy imports from package
        from nfl_compare.src.data_sources import load_games as ds_load_games, load_team_stats, load_lines
        from nfl_compare.src.features import merge_features
        from nfl_compare.src.weather import load_weather_for_games
        from nfl_compare.src.models import predict as model_predict

        # Load base data and models
        games = ds_load_games()
        stats = load_team_stats()
        lines = load_lines()
        try:
            models = joblib_load(BASE_DIR / 'nfl_compare' / 'models' / 'nfl_models.joblib')
        except Exception:
            return view_df  # models not available
        try:
            wx = load_weather_for_games(games)
        except Exception:
            wx = None
        feat = merge_features(games, stats, lines, wx)
        if feat is None or feat.empty:
            return out_base
        # Filter features to the rows in view_df (exclude completed/final games to keep historical predictions locked)
        vf = out_base.copy()
        # Determine which rows actually need predictions (any key pred_* missing)
        need_mask = pd.Series(False, index=vf.index)
        for col in ("pred_home_points","pred_away_points","pred_total","pred_home_win_prob"):
            if col in vf.columns:
                need_mask = need_mask | vf[col].isna()
            else:
                need_mask = True  # if a key column missing entirely, we need to predict
        vf_pred = vf.loc[need_mask].copy()
        if vf_pred.empty:
            return out_base

        if 'game_id' in vf_pred.columns and 'game_id' in feat.columns:
            keys = vf_pred['game_id'].dropna().astype(str).unique().tolist()
            sub = feat[feat['game_id'].astype(str).isin(keys)].copy()
        else:
            # fallback by match
            key_cols = [c for c in ['season','week','home_team','away_team'] if c in vf_pred.columns and c in feat.columns]
            if key_cols:
                sub = feat.merge(vf_pred[key_cols].drop_duplicates(), on=key_cols, how='inner')
            else:
                sub = feat
        if sub.empty:
            return out_base
        # Run model predictions
        pred = model_predict(models, sub)
        if pred is None or pred.empty:
            return out_base
        # Select columns to merge back
        keep_cols = [c for c in pred.columns if c.startswith('pred_') or c.startswith('prob_')] + ['game_id','home_team','away_team']
        pred_keep = pred[[c for c in keep_cols if c in pred.columns]].copy()
        # Merge back into view_df without overwriting existing non-null prediction values
        if 'game_id' in vf_pred.columns and 'game_id' in pred_keep.columns and pred_keep['game_id'].notna().any():
            merged_partial = vf_pred.merge(pred_keep, on='game_id', how='left', suffixes=('', '_m'))
        else:
            merged_partial = vf_pred.merge(pred_keep, on=[c for c in ['home_team','away_team'] if c in vf_pred.columns and c in pred_keep.columns], how='left', suffixes=('', '_m'))
        # Fill nulls from _m columns (do not overwrite existing non-null values)
        for col in ['pred_home_points','pred_away_points','pred_total','pred_margin','pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total','pred_1h_total','pred_2h_total','prob_home_win','pred_home_win_prob']:
            base = col
            alt = f"{col}_m"
            if base in merged_partial.columns and alt in merged_partial.columns:
                merged_partial[base] = merged_partial[base].fillna(merged_partial[alt])
        # Also fill odds/lines/weather fields from features frame for completeness
        line_cols = ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price','close_spread_home','close_total',
                     'wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']
        feat_keep = feat[['game_id','home_team','away_team'] + [c for c in line_cols if c in feat.columns]].copy()
        if 'game_id' in merged_partial.columns and 'game_id' in feat_keep.columns and feat_keep['game_id'].notna().any():
            merged2_partial = merged_partial.merge(feat_keep, on='game_id', how='left', suffixes=('', '_f'))
        else:
            merged2_partial = merged_partial.merge(feat_keep, on=[c for c in ['home_team','away_team'] if c in merged_partial.columns and c in feat_keep.columns], how='left', suffixes=('', '_f'))
        for col in [c for c in line_cols if c in merged2_partial.columns and f"{c}_f" in merged2_partial.columns]:
            merged2_partial[col] = merged2_partial[col].where(merged2_partial[col].notna(), merged2_partial[f"{col}_f"]) 
        # drop helper cols on partial
        drop_cols2 = [c for c in merged2_partial.columns if c.endswith('_m') or c.endswith('_f')]
        merged2_partial = merged2_partial.drop(columns=drop_cols2)
    # Stitch partial predictions back with untouched finals
        out = vf.copy()
        if 'game_id' in out.columns and 'game_id' in merged2_partial.columns and merged2_partial['game_id'].notna().any():
            out = out.merge(merged2_partial, on=[c for c in out.columns if c in merged2_partial.columns and c in ['game_id']], how='left', suffixes=('', '_new'))
        else:
            join_keys = [c for c in ['season','week','home_team','away_team'] if c in out.columns and c in merged2_partial.columns]
            out = out.merge(merged2_partial, on=join_keys, how='left', suffixes=('', '_new'))
        # For prediction fields, fill missing values from newly computed values (keep existing non-null values intact)
        for col in ['pred_home_points','pred_away_points','pred_total','pred_margin','pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total','pred_1h_total','pred_2h_total','prob_home_win','pred_home_win_prob']:
            if f"{col}_new" in out.columns:
                out[col] = out[col].fillna(out[f"{col}_new"])  # fill only where missing
        # Clean helper columns
        drop_cols3 = [c for c in out.columns if c.endswith('_new')]
        out = out.drop(columns=drop_cols3)
        # Finally, enrich odds/lines/weather for ALL rows (including finals) from features
        try:
            feat_keep_all = feat[['game_id','home_team','away_team'] + [c for c in line_cols if c in feat.columns]].drop_duplicates()
            if 'game_id' in out.columns and 'game_id' in feat_keep_all.columns and feat_keep_all['game_id'].notna().any():
                out2 = out.merge(feat_keep_all, on='game_id', how='left', suffixes=('', '_f2'))
            else:
                join_keys2 = [c for c in ['home_team','away_team'] if c in out.columns and c in feat_keep_all.columns]
                out2 = out.merge(feat_keep_all, on=join_keys2, how='left', suffixes=('', '_f2'))
            for col in [c for c in line_cols if c in out2.columns and f"{c}_f2" in out2.columns]:
                out2[col] = out2[col].where(out2[col].notna(), out2[f"{col}_f2"])  # fill only missing
            drop_cols4 = [c for c in out2.columns if c.endswith('_f2')]
            out = out2.drop(columns=drop_cols4)
        except Exception:
            pass
        return out
    except Exception:
        try:
            # If enrichment partially succeeded, prefer returning that
            return out_base  # type: ignore[name-defined]
        except Exception:
            return view_df


def _load_team_assets() -> Dict[str, Dict[str, str]]:
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_stadium_meta_map() -> Dict[str, Dict[str, Any]]:
    """Return mapping of team -> {'stadium': str|None, 'tz': str|None, 'roof': str|None, 'surface': str|None} if available."""
    try:
        import pandas as pd  # already imported at top
        if STADIUM_META_FILE.exists():
            df = pd.read_csv(STADIUM_META_FILE)
            # Normalize team key
            if 'team' in df.columns:
                df['team'] = df['team'].astype(str)
                cols = {c: c for c in ['stadium','tz','roof','surface'] if c in df.columns}
                if cols:
                    return df.set_index('team')[list(cols.keys())].to_dict(orient='index')
    except Exception:
        pass
    return {}


def _file_status(fp: Path) -> Dict[str, Any]:
    try:
        exists = fp.exists()
        size = int(fp.stat().st_size) if exists else 0
        mtime = int(fp.stat().st_mtime) if exists else None
        sha = None
        if exists and size <= 5_000_000:  # avoid hashing very large files
            h = hashlib.sha256()
            with open(fp, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            sha = h.hexdigest()[:16]
        return {"path": str(fp), "exists": exists, "size": size, "mtime": mtime, "sha256_16": sha}
    except Exception:
        return {"path": str(fp), "exists": False}


@app.route('/api/data-status')
def api_data_status():
    """Report presence and small hashes of key data files to debug parity between local and Render."""
    files = {
        "predictions": PRED_FILE,
        "lines": DATA_DIR / 'lines.csv',
        "games": DATA_DIR / 'games.csv',
        "team_stats": DATA_DIR / 'team_stats.csv',
        "eval_summary": DATA_DIR / 'eval_summary.json',
        "assets": ASSETS_FILE,
        "stadium_meta": STADIUM_META_FILE,
        "overrides": LOCATION_OVERRIDES_FILE,
    }
    status = {k: _file_status(v) for k, v in files.items()}
    # also list a couple of JSON odds snapshots if present
    try:
        candidates = []
        for pat in ('real_betting_lines_*.json','real_betting_lines.json'):
            candidates.extend([p for p in DATA_DIR.glob(pat)])
        status['json_odds'] = [ _file_status(p) for p in sorted(candidates)[:5] ]
    except Exception:
        status['json_odds'] = []
    return jsonify({
        "env": {"RENDER": os.getenv('RENDER'), "DISABLE_JSON_ODDS": os.getenv('DISABLE_JSON_ODDS')},
        "data": status
    })


def _load_location_overrides() -> Dict[str, Dict[str, Any]]:
    """Load optional per-game location overrides.
    CSV columns (any subset): game_id, date, home_team, away_team, venue, city, country, tz, lat, lon, roof, surface, neutral_site
    Returns mapping with two keys:
      - 'by_game_id': {game_id: {...}}
      - 'by_match': {(date, home_team, away_team): {...}}
    """
    out = {"by_game_id": {}, "by_match": {}}
    try:
        if LOCATION_OVERRIDES_FILE.exists():
            expected = ['game_id','date','home_team','away_team','venue','city','country','tz','lat','lon','roof','surface','neutral_site','note']
            # Support commented headers and missing header row
            df = pd.read_csv(LOCATION_OVERRIDES_FILE, comment='#', header=None, names=expected)
            norm = lambda s: None if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
            for _, r in df.iterrows():
                rec = {k: r.get(k) for k in [
                    'venue','city','country','tz','lat','lon','roof','surface','neutral_site','note'
                ] if k in df.columns}
                gid = norm(r.get('game_id')) if 'game_id' in df.columns else None
                date = norm(r.get('date')) if 'date' in df.columns else None
                home = norm(r.get('home_team')) if 'home_team' in df.columns else None
                away = norm(r.get('away_team')) if 'away_team' in df.columns else None
                if gid:
                    out['by_game_id'][gid] = rec
                if date and home and away:
                    out['by_match'][(date, home, away)] = rec
    except Exception:
        pass
    return out


def _infer_current_season_week(df: pd.DataFrame) -> Optional[tuple[int, int]]:
    """Infer the current (season, week) to display based on game dates.
    Strategy: group by season/week and take the group's earliest game datetime.
    Choose the first group with min_dt >= now; if none, choose the latest past group.
    """
    try:
        if df is None or df.empty:
            return None
        if not {'season','week'}.issubset(df.columns):
            return None
        dt_col = 'game_date' if 'game_date' in df.columns else ('date' if 'date' in df.columns else None)
        if not dt_col:
            return None
        tmp = df[['season','week', dt_col]].copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors='coerce')
        tmp = tmp.dropna(subset=[dt_col])
        if tmp.empty:
            return None
        grp = tmp.groupby(['season','week'])[dt_col].min().reset_index().rename(columns={dt_col: 'min_dt'})
        now = pd.Timestamp.now()
        future = grp[grp['min_dt'] >= now]
        if not future.empty:
            row = future.sort_values('min_dt', ascending=True).iloc[0]
        else:
            row = grp.sort_values('min_dt', ascending=True).iloc[-1]
        season_i = int(row['season']) if pd.notna(row['season']) else None
        week_i = int(row['week']) if pd.notna(row['week']) else None
        if season_i is None or week_i is None:
            return None
        return season_i, week_i
    except Exception:
        return None


# --- Betting EV helpers ---
def _american_to_decimal(ml: Optional[float]) -> Optional[float]:
    try:
        if ml is None or (isinstance(ml, float) and pd.isna(ml)):
            return None
        v = float(ml)
        if v > 0:
            return 1.0 + v / 100.0
        else:
            return 1.0 + 100.0 / abs(v)
    except Exception:
        return None


def _ev_from_prob_and_decimal(p: float, dec_odds: float) -> float:
    """Risk 1 unit. EV in units: p * (dec-1) - (1-p) * 1."""
    win = max(dec_odds - 1.0, 0.0)
    return p * win - (1.0 - p) * 1.0


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def _cover_prob_from_edge(edge_pts: float, scale: float) -> float:
    """Map point edge to win probability via logistic with scale parameter."""
    if scale <= 0:
        scale = 1.0
    return _sigmoid(edge_pts / scale)


def _conf_from_ev(ev_units: float) -> Optional[str]:
    """Map EV (in units per 1 risked) to Low/Medium/High; None if not positive.
    Thresholds are percent and can be tuned via env:
      RECS_EV_THRESH_LOW (default 4), RECS_EV_THRESH_MED (8), RECS_EV_THRESH_HIGH (15)
    """
    if ev_units is None or not isinstance(ev_units, (int, float)):
        return None
    if ev_units <= 0:
        return None
    # Read thresholds in percent with safe fallbacks
    try:
        th_low = float(os.environ.get('RECS_EV_THRESH_LOW', '3.5'))
    except Exception:
        th_low = 4.0
    try:
        th_med = float(os.environ.get('RECS_EV_THRESH_MED', '7.5'))
    except Exception:
        th_med = 8.0
    try:
        th_high = float(os.environ.get('RECS_EV_THRESH_HIGH', '12.5'))
    except Exception:
        th_high = 15.0
    ev_pct = ev_units * 100.0
    if ev_pct >= th_high:
        return "High"
    if ev_pct >= th_med:
        return "Medium"
    if ev_pct >= th_low:
        return "Low"
    return None


def _tier_to_num(t: Optional[str]) -> int:
    m = {None: 0, "": 0, "Low": 1, "Medium": 2, "High": 3}
    return m.get(t, 0)


def _num_to_tier(n: int) -> Optional[str]:
    n = max(0, min(3, int(n)))
    r = {0: None, 1: "Low", 2: "Medium", 3: "High"}
    return r[n]


def _combine_confs(*tiers: Optional[str]) -> Optional[str]:
    """Conservatively combine tiers by taking the weakest non-null tier.
    Order of strength: None < Low < Medium < High.
    """
    values = [_tier_to_num(t) for t in tiers if t]
    if not values:
        return None
    return _num_to_tier(min(values))


def _clamp_prob_to_band(p: float, anchor: Optional[float], band: float) -> float:
    try:
        if anchor is None or not isinstance(anchor, (int, float)) or not isinstance(p, (int, float)):
            return p
        lo = max(0.0, float(anchor) - float(band))
        hi = min(1.0, float(anchor) + float(band))
        return max(lo, min(hi, float(p)))
    except Exception:
        return p


def _implied_probs_from_moneylines(ml_home: Optional[float], ml_away: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Return de-vig implied win probs from American odds for home and away."""
    try:
        dh = _american_to_decimal(ml_home)
        da = _american_to_decimal(ml_away)
        if dh is None or da is None:
            return (None, None)
        # Convert decimal odds back to implied probs with vig
        # dec = 1 + b/a => implied prob ~ 1/dec
        ph = 1.0 / dh
        pa = 1.0 / da
        s = ph + pa
        if s <= 0:
            return (None, None)
        return (ph / s, pa / s)
    except Exception:
        return (None, None)


def _format_game_datetime(date_like: Any, tz: Optional[str]) -> Optional[str]:
    """Format a date or datetime value to a friendly string like 'Sat, Aug 23, 2025, 11:00 AM TZ'.
    Falls back gracefully if parsing fails or time missing."""
    try:
        if date_like is None or (isinstance(date_like, float) and pd.isna(date_like)):
            return None
        ts = pd.to_datetime(date_like, errors='coerce')
        if pd.isna(ts):
            return str(date_like)
        # If no time component, omit time
        has_time = not (ts.hour == 0 and ts.minute == 0 and ts.second == 0 and getattr(ts, 'nanosecond', 0) == 0)
        if has_time:
            s = ts.strftime('%a, %b %d, %Y, %I:%M %p')
        else:
            s = ts.strftime('%a, %b %d, %Y')
        if tz:
            s = f"{s} {tz}"
        return s
    except Exception:
        return str(date_like) if date_like is not None else None


def _compute_recommendations_for_row(row: pd.Series) -> List[Dict[str, Any]]:
    """Compute EV-based recommendations (ML, Spread, Total) for a single game row.
    Returns a list of recommendation dicts with keys: type, selection, odds, ev_units, ev_pct, confidence, sort_weight, and some game metadata.
    """
    recs: List[Dict[str, Any]] = []

    def g(key: str, *alts: str, default=None):
        for k in (key, *alts):
            if k in row.index:
                v = row.get(k)
                return v
        return default

    home = g("home_team")
    away = g("away_team")
    season = g("season")
    week = g("week")
    game_date = g("game_date", "date")
    # Actuals for grading
    actual_home = g("home_score")
    actual_away = g("away_score")
    actual_total = None
    actual_margin = None
    is_final = False
    if actual_home is not None and actual_away is not None and not pd.isna(actual_home) and not pd.isna(actual_away):
        try:
            ah = float(actual_home); aa = float(actual_away)
            actual_total = ah + aa
            actual_margin = ah - aa
            is_final = True
        except Exception:
            pass

    # Winner (moneyline)
    wp_home = g("pred_home_win_prob", "prob_home_win")
    ml_home = g("moneyline_home")
    ml_away = g("moneyline_away")
    dec_home = _american_to_decimal(ml_home) if ml_home is not None else None
    dec_away = _american_to_decimal(ml_away) if ml_away is not None else None
    ev_home_ml = ev_away_ml = None
    p_home_eff = None
    try:
        p_home = float(wp_home) if (wp_home is not None and not pd.isna(wp_home)) else None
    except Exception:
        p_home = None
    if p_home is not None:
        mkt_ph, _ = _implied_probs_from_moneylines(ml_home, ml_away)
        try:
            beta = float(os.environ.get('RECS_MARKET_BLEND', '0.35'))
        except Exception:
            beta = 0.35
        p_home_eff = (1.0 - beta) * p_home + beta * mkt_ph if mkt_ph is not None else p_home
        # Optional clamp to a band around market (disabled by default)
        try:
            band = float(os.environ.get('RECS_MARKET_BAND', '0.0'))
        except Exception:
            band = 0.0
        if mkt_ph is not None and band and band > 0:
            p_home_eff = _clamp_prob_to_band(p_home_eff, mkt_ph, band)
        if dec_home:
            ev_home_ml = _ev_from_prob_and_decimal(p_home_eff, dec_home)
        if dec_away:
            ev_away_ml = _ev_from_prob_and_decimal(1.0 - p_home_eff, dec_away)
    # Select best ML side
    if ev_home_ml is not None or ev_away_ml is not None:
        cand = [(home or "Home", ev_home_ml, ml_home), (away or "Away", ev_away_ml, ml_away)]
        cand = [(s, e, o) for s, e, o in cand if e is not None and o is not None]
        if cand:
            s, e, o = max(cand, key=lambda t: t[1])
            # Per-market tier by EV
            # Confidence is based on this market's EV only
            conf = _conf_from_ev(e)
            # Grade if final
            result = None
            if is_final and actual_margin is not None:
                actual_side = "HOME" if actual_margin > 0 else ("AWAY" if actual_margin < 0 else "PUSH")
                picked_side = "HOME" if str(s) == str(home) else ("AWAY" if str(s) == str(away) else None)
                if picked_side:
                    if actual_side == "PUSH":
                        result = "Push"
                    else:
                        result = "Win" if picked_side == actual_side else "Loss"
            recs.append({
                "type": "MONEYLINE",
                "selection": f"{s} ML",
                "odds": int(o) if isinstance(o, (int, float)) and not pd.isna(o) else o,
                "ev_units": e,
                "ev_pct": e * 100.0 if e is not None else None,
                "confidence": conf,
                # Weight: confidence first, then EV
                "sort_weight": (_tier_to_num(conf), e or -999),
                "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                "result": result,
            })

    # Spread (ATS) at -110
    margin = None
    # Prefer close spread for finals; fallback otherwise
    _hs = g("home_score"); _as = g("away_score")
    _is_final = (_hs is not None and not pd.isna(_hs)) and (_as is not None and not pd.isna(_as))
    spread = g("close_spread_home") if _is_final else g("market_spread_home", "spread_home", "open_spread_home")
    if spread is None or (isinstance(spread, float) and pd.isna(spread)):
        spread = g("market_spread_home", "spread_home", "open_spread_home")
    try:
        ph = g("pred_home_points", "pred_home_score")
        pa = g("pred_away_points", "pred_away_score")
        if ph is not None and pa is not None and not pd.isna(ph) and not pd.isna(pa):
            margin = float(ph) - float(pa)
    except Exception:
        margin = None
    if margin is not None and spread is not None and not pd.isna(spread):
        try:
            edge_pts = float(margin) + float(spread)
            scale_margin = float(os.environ.get('NFL_ATS_SIGMA', '9.0'))
        except Exception:
            edge_pts, scale_margin = None, 9.0
        if edge_pts is not None:
            p_home_cover = _cover_prob_from_edge(edge_pts, scale_margin)
            try:
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
            except Exception:
                shrink = 0.35
            p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)
            # Use market prices if available; fallback to -110
            sh_price = g("spread_home_price")
            sa_price = g("spread_away_price")
            dec_home_sp = _american_to_decimal(sh_price) if sh_price is not None and not pd.isna(sh_price) else 1.0 + 100.0/110.0
            dec_away_sp = _american_to_decimal(sa_price) if sa_price is not None and not pd.isna(sa_price) else 1.0 + 100.0/110.0
            ev_home = _ev_from_prob_and_decimal(p_home_cover, dec_home_sp)
            ev_away = _ev_from_prob_and_decimal(1.0 - p_home_cover, dec_away_sp)
            # Build selections
            try:
                sp = float(spread)
            except Exception:
                sp = None
            # Normalize price display as signed ints if present
            sh_disp = (int(sh_price) if (sh_price is not None and not pd.isna(sh_price)) else None)
            sa_disp = (int(sa_price) if (sa_price is not None and not pd.isna(sa_price)) else None)
            if sp is not None:
                home_sel = f"{home} {sp:+.1f}{(' ('+('%+d' % sh_disp)+')') if (sh_disp is not None) else ''}"
                away_sel = f"{away} {(-sp):+.1f}{(' ('+('%+d' % sa_disp)+')') if (sa_disp is not None) else ''}"
            else:
                home_sel = f"{home} ATS{(' ('+('%+d' % sh_disp)+')') if (sh_disp is not None) else ''}"
                away_sel = f"{away} ATS{(' ('+('%+d' % sa_disp)+')') if (sa_disp is not None) else ''}"
            # Choose best
            cand = [(home_sel, ev_home), (away_sel, ev_away)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                # Confidence is based on this market's EV only
                conf = _conf_from_ev(e)
                # Grade if final
                result = None
                if is_final and actual_margin is not None and sp is not None:
                    cover_val = actual_margin + float(spread)
                    actual_side = "HOME" if cover_val > 0 else ("AWAY" if cover_val < 0 else "PUSH")
                    picked_side = "HOME" if str(s).startswith(str(home)) else ("AWAY" if str(s).startswith(str(away)) else None)
                    if picked_side:
                        if actual_side == "PUSH":
                            result = "Push"
                        else:
                            result = "Win" if picked_side == actual_side else "Loss"
                recs.append({
                    "type": "SPREAD",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                    "result": result,
                })

    # Total at -110
    total_pred = g("pred_total")
    m_total = g("close_total") if _is_final else g("market_total", "total", "open_total")
    if m_total is None or (isinstance(m_total, float) and pd.isna(m_total)):
        m_total = g("market_total", "total", "open_total")
    if total_pred is not None and not pd.isna(total_pred) and m_total is not None and not pd.isna(m_total):
        try:
            edge_t = float(total_pred) - float(m_total)
            scale_total = float(os.environ.get('NFL_TOTAL_SIGMA', '10.0'))
        except Exception:
            edge_t, scale_total = None, 10.0
        if edge_t is not None:
            p_over = _cover_prob_from_edge(edge_t, scale_total)
            try:
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
            except Exception:
                shrink = 0.35
            p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)
            # Use market prices if available; fallback to -110
            to_price = g("total_over_price")
            tu_price = g("total_under_price")
            dec_over = _american_to_decimal(to_price) if to_price is not None and not pd.isna(to_price) else 1.0 + 100.0/110.0
            dec_under = _american_to_decimal(tu_price) if tu_price is not None and not pd.isna(tu_price) else 1.0 + 100.0/110.0
            ev_over = _ev_from_prob_and_decimal(p_over, dec_over)
            ev_under = _ev_from_prob_and_decimal(1.0 - p_over, dec_under)
            # Choose best
            try:
                tot = float(m_total)
            except Exception:
                tot = None
            to_disp = (int(to_price) if (to_price is not None and not pd.isna(to_price)) else None)
            tu_disp = (int(tu_price) if (tu_price is not None and not pd.isna(tu_price)) else None)
            over_sel = (f"Over {tot:.1f}" if tot is not None else "Over") + (f" ({to_disp:+d})" if (to_disp is not None) else "")
            under_sel = (f"Under {tot:.1f}" if tot is not None else "Under") + (f" ({tu_disp:+d})" if (tu_disp is not None) else "")
            cand = [(over_sel, ev_over), (under_sel, ev_under)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                # Confidence is based on this market's EV only
                conf = _conf_from_ev(e)
                # Grade if final
                result = None
                if is_final and actual_total is not None and m_total is not None and not pd.isna(m_total):
                    try:
                        mt = float(m_total)
                        if actual_total > mt:
                            actual_ou = "OVER"
                        elif actual_total < mt:
                            actual_ou = "UNDER"
                        else:
                            actual_ou = "PUSH"
                        picked_ou = "OVER" if str(s).startswith("Over") else ("UNDER" if str(s).startswith("Under") else None)
                        if picked_ou:
                            if actual_ou == "PUSH":
                                result = "Push"
                            else:
                                result = "Win" if picked_ou == actual_ou else "Loss"
                    except Exception:
                        result = None
                recs.append({
                    "type": "TOTAL",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                    "result": result,
                })

    # Apply global filtering to reduce noise
    try:
        # Default minimum EV (2%) so Week 1 isn't empty; can be overridden via query or env
        min_ev_pct = float(os.environ.get('RECS_MIN_EV_PCT', '2.0'))
    except Exception:
        min_ev_pct = 2.0
    include_completed = str(os.environ.get('RECS_INCLUDE_COMPLETED', 'true')).strip().lower() in {'1','true','yes','y'}
    filtered: List[Dict[str, Any]] = []
    for r in recs:
        evp = r.get('ev_pct')
        if evp is not None and evp >= min_ev_pct:
            filtered.append(r)
        elif is_final and include_completed:
            filtered.append(r)
    # Ensure every passing pick has a visible confidence tier; only floor when none computed
    for r in filtered:
        if (r.get('confidence') is None or r.get('confidence') == '') and r.get('ev_pct') is not None and r.get('ev_pct') >= min_ev_pct:
            r['confidence'] = 'Low'
            w_ev = r.get('ev_units') or -999
            r['sort_weight'] = (_tier_to_num('Low'), w_ev)
    # Optional: keep only the single highest-EV recommendation per game
    one_per_game = str(os.environ.get('RECS_ONE_PER_GAME', 'false')).strip().lower() in {'1','true','yes','y'}
    if one_per_game and filtered:
        best = max(filtered, key=lambda r: r.get('ev_units') if r.get('ev_units') is not None else -999)
        return [best]
    return filtered


@app.route("/health")
def health():
    return {"status": "ok", "have_predictions": PRED_FILE.exists()}, 200


@app.route("/api/predictions")
def api_predictions():
    df = _load_predictions()
    if df.empty:
        return {"rows": 0, "data": []}, 200

    # Optional filters
    season = request.args.get("season")
    week = request.args.get("week")
    # Default: latest season, week 1 when no filters provided
    if not season and not week:
        try:
            if "season" in df.columns and not df["season"].isna().all():
                latest_season = int(df["season"].max())
                df = df[df["season"] == latest_season]
        except Exception:
            pass
        if "week" in df.columns:
            df = df[df["week"].astype(str) == "1"]
    if season:
        try:
            season_i = int(season)
            if "season" in df.columns:
                df = df[df["season"] == season_i]
        except ValueError:
            pass
    if week:
        try:
            week_i = int(week)
            if "week" in df.columns:
                df = df[df["week"] == week_i]
        except ValueError:
            pass

    # Limit columns for API clarity if present
    prefer_cols = [
        "season", "week", "game_id", "game_date", "home_team", "away_team",
        "pred_home_points", "pred_away_points", "pred_total", "pred_home_win_prob",
        "market_spread_home", "market_total",
    ]
    cols = [c for c in prefer_cols if c in df.columns]
    out = df[cols].to_dict(orient="records") if cols else df.to_dict(orient="records")
    return {"rows": len(out), "data": out}, 200


@app.route("/api/recommendations")
def api_recommendations():
    """Return EV-based betting recommendations aggregated across games.
    Optional query params: season, week, date (YYYY-MM-DD)
    """
    pred_df = _load_predictions()
    games_df = _load_games()

    # Parse filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    season_i = None
    week_i = None
    if season:
        try:
            season_i = int(season)
        except Exception:
            season_i = None
    if week:
        try:
            week_i = int(week)
        except Exception:
            week_i = None
    # If week is provided but season missing, infer latest season from games/predictions
    if week_i is not None and season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            pass
    # Default to the current (season, week) inferred by date when no explicit week/date
    if week_i is None and not date:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_i, week_i = int(inferred[0]), int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    if season_i is None:
                        season_i = int(src['season'].max())
                week_i = 1
        except Exception:
            # Last-resort fallback
            week_i = 1

    # Build combined view and enrich with model preds + odds/weather
    view_df = _build_week_view(pred_df, games_df, season_i, week_i)
    view_df = _attach_model_predictions(view_df)
    if view_df is None or view_df.empty:
        return {"rows": 0, "data": []}, 200
    # Optional date filter against combined view
    if date:
        try:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        except Exception:
            pass

    # Global filter overrides
    min_ev = request.args.get("min_ev")
    one = request.args.get("one_per_game")
    if min_ev:
        os.environ['RECS_MIN_EV_PCT'] = str(min_ev)
    if one is not None:
        os.environ['RECS_ONE_PER_GAME'] = str(one)
    # Build recs
    all_recs: List[Dict[str, Any]] = []
    for _, row in view_df.iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
    # If nothing qualifies and no explicit min_ev was provided, relax threshold once
    if not all_recs and not request.args.get("min_ev"):
        try:
            os.environ['RECS_MIN_EV_PCT'] = '1.0'
            for _, row in view_df.iterrows():
                try:
                    recs = _compute_recommendations_for_row(row)
                    all_recs.extend(recs)
                except Exception:
                    continue
        except Exception:
            pass
    # Optional sorting: time | type | odds | ev | level
    sort_param = (request.args.get("sort") or "").lower()
    type_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    def sort_key_time(r: Dict[str, Any]):
        try:
            return pd.to_datetime(r.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    def sort_key_type(r: Dict[str, Any]):
        return type_order.get(str(r.get("type")).upper(), 99)
    def sort_key_odds(r: Dict[str, Any]):
        o = r.get("odds")
        try:
            return float(o) if o is not None and not pd.isna(o) else -9999
        except Exception:
            return -9999
    def sort_key_ev(r: Dict[str, Any]):
        v = r.get("ev_pct")
        try:
            return float(v) if v is not None else -9999
        except Exception:
            return -9999
    def sort_key_level(r: Dict[str, Any]):
        return _tier_to_num(r.get("confidence"))
    if sort_param == "time":
        all_recs.sort(key=sort_key_time)
    elif sort_param == "type":
        all_recs.sort(key=sort_key_type)
    elif sort_param == "odds":
        all_recs.sort(key=sort_key_odds, reverse=True)
    elif sort_param == "ev":
        all_recs.sort(key=sort_key_ev, reverse=True)
    elif sort_param == "level":
        all_recs.sort(key=sort_key_level, reverse=True)
    else:
        # Default: by confidence then EV desc
        def sort_key(r: Dict[str, Any]):
            w = r.get("sort_weight") or (0, -999)
            return (w[0], w[1])
        all_recs.sort(key=sort_key, reverse=True)
    for r in all_recs:
        r.pop("sort_weight", None)
    return {"rows": len(all_recs), "data": all_recs}, 200


@app.route("/recommendations")
def recommendations_page():
    """HTML page for recommendations, sorted and grouped by confidence.
    Build from combined games + predictions view to ensure lines/finals are available.
    """
    pred_df = _load_predictions()
    games_df = _load_games()

    # Parse filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    active_week = None

    # Determine default season/week: default to latest season, week 1 when no explicit filters
    season_i = None
    week_i = None
    if season:
        try:
            season_i = int(season)
        except Exception:
            season_i = None
    if week:
        try:
            week_i = int(week)
        except Exception:
            week_i = None
    # If week provided but season missing, infer latest season
    if week_i is not None and season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            season_i = None
    if season_i is None and week_i is None and not date:
        # Latest season from games (fallback to predictions)
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            season_i = None
        week_i = 1
        active_week = 1
    else:
        active_week = week_i

    # Build combined view and optionally filter by date
    view_df = _build_week_view(pred_df, games_df, season_i, week_i)
    view_df = _attach_model_predictions(view_df)
    if view_df is None:
        view_df = pd.DataFrame()
    if date and not view_df.empty:
        try:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        except Exception:
            pass

    # Global filter overrides (query)
    min_ev = request.args.get("min_ev")
    one = request.args.get("one_per_game")
    if min_ev:
        os.environ['RECS_MIN_EV_PCT'] = str(min_ev)
    if one is not None:
        os.environ['RECS_ONE_PER_GAME'] = str(one)
    all_recs: List[Dict[str, Any]] = []
    for _, row in (view_df if view_df is not None else pd.DataFrame()).iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
    # If still empty and no explicit min_ev in query, relax threshold once
    if not all_recs and not request.args.get("min_ev"):
        try:
            os.environ['RECS_MIN_EV_PCT'] = '0.5'
            for _, row in (view_df if view_df is not None else pd.DataFrame()).iterrows():
                try:
                    recs = _compute_recommendations_for_row(row)
                    all_recs.extend(recs)
                except Exception:
                    continue
        except Exception:
            pass
    # Sort per query: time | type | odds | ev | level; default confidence->EV
    sort_param = (request.args.get("sort") or "").lower()
    type_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    def sort_key_time(r: Dict[str, Any]):
        try:
            return pd.to_datetime(r.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    def sort_key_type(r: Dict[str, Any]):
        return type_order.get(str(r.get("type")).upper(), 99)
    def sort_key_odds(r: Dict[str, Any]):
        o = r.get("odds")
        try:
            return float(o) if o is not None and not pd.isna(o) else -9999
        except Exception:
            return -9999
    def sort_key_ev(r: Dict[str, Any]):
        v = r.get("ev_pct")
        try:
            return float(v) if v is not None else -9999
        except Exception:
            return -9999
    def sort_key_level(r: Dict[str, Any]):
        return _tier_to_num(r.get("confidence"))
    if sort_param == "time":
        all_recs.sort(key=sort_key_time)
    elif sort_param == "type":
        all_recs.sort(key=sort_key_type)
    elif sort_param == "odds":
        all_recs.sort(key=sort_key_odds, reverse=True)
    elif sort_param == "ev":
        all_recs.sort(key=sort_key_ev, reverse=True)
    elif sort_param == "level":
        all_recs.sort(key=sort_key_level, reverse=True)
    else:
        def sort_key(r: Dict[str, Any]):
            w = r.get("sort_weight") or (0, -999)
            return (w[0], w[1])
        all_recs.sort(key=sort_key, reverse=True)
    for r in all_recs:
        r.pop("sort_weight", None)

    groups: Dict[str, List[Dict[str, Any]]] = {"High": [], "Medium": [], "Low": [], "": []}
    for r in all_recs:
        c = r.get("confidence") or ""
        if c not in groups:
            groups[c] = []
        groups[c].append(r)

    return render_template("recommendations.html", recs=all_recs, groups=groups, have_data=len(all_recs) > 0, week=active_week, sort=sort_param)


@app.route("/")
def index():
    df = _load_predictions()
    games_df = _load_games()
    # Filters
    season_param: Optional[int] = None
    week_param: Optional[int] = None
    sort_param: str = request.args.get("sort") or "date"
    try:
        if request.args.get("season"):
            season_param = int(request.args.get("season"))
        if request.args.get("week"):
            week_param = int(request.args.get("week"))
    except Exception:
        pass

    # Default to the current (season, week) inferred by date when no explicit filters
    if season_param is None and week_param is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_param, week_param = int(inferred[0]), int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_param = int(src['season'].max())
                week_param = 1
        except Exception:
            week_param = 1

    # Build combined view from games + predictions for the target week
    view_df = _build_week_view(df, games_df, season_param, week_param)
    # Attach predictions for completed games if missing (local only)
    view_df = _attach_model_predictions(view_df)
    if view_df is None:
        view_df = pd.DataFrame()

    # Build card-friendly rows
    cards: List[Dict[str, Any]] = []
    assets = _load_team_assets()
    stad_map = _load_stadium_meta_map()
    if not view_df.empty:
        for _, r in view_df.iterrows():
            def g(key: str, *alts: str, default=None):
                for k in (key, *alts):
                    if k in view_df.columns:
                        v = r.get(k)
                        # prefer first non-null value
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            return v
                return default

            # Compute assessments
            # Normalize model team scores
            ph = g("pred_home_points", "pred_home_score")
            pa = g("pred_away_points", "pred_away_score")
            margin = None
            winner = None
            total_pred = g("pred_total")
            # Weather-aware tweak for upcoming outdoor games: small downward adjustment for high precip/wind
            try:
                # Only apply to non-final games with a model total
                is_final_now = False
                _st = g("status")
                if _st is not None and str(_st).upper() == 'FINAL':
                    is_final_now = True
                if (g("home_score") is not None and not (isinstance(g("home_score"), float) and pd.isna(g("home_score")))) \
                   and (g("away_score") is not None and not (isinstance(g("away_score"), float) and pd.isna(g("away_score")))):
                    is_final_now = True
                if (total_pred is not None) and (not is_final_now):
                    roof_ctx = g("stadium_roof", "roof")
                    is_dome_like = False
                    if roof_ctx is not None and not (isinstance(roof_ctx, float) and pd.isna(roof_ctx)):
                        srf = str(roof_ctx).strip().lower()
                        is_dome_like = srf in {"dome","indoor","closed","retractable-closed"}
                    if not is_dome_like:
                        precip_ctx = g("wx_precip_pct", "precip_pct")
                        wind_ctx = g("wx_wind_mph", "wind_mph")
                        adj = 0.0
                        try:
                            if precip_ctx is not None and not (isinstance(precip_ctx, float) and pd.isna(precip_ctx)):
                                p = float(precip_ctx)
                                adj += -2.5 * max(0.0, min(p, 100.0)) / 100.0
                        except Exception:
                            pass
                        try:
                            if wind_ctx is not None and not (isinstance(wind_ctx, float) and pd.isna(wind_ctx)):
                                w = float(wind_ctx)
                                over = max(0.0, w - 10.0)
                                adj += -0.10 * over
                        except Exception:
                            pass
                        try:
                            tp = float(total_pred)
                            total_pred = max(0.0, tp + adj)
                        except Exception:
                            pass
            except Exception:
                pass
            if ph is not None and pa is not None and pd.notna(ph) and pd.notna(pa):
                try:
                    margin = float(ph) - float(pa)
                    if margin > 0:
                        winner = g("home_team")
                    elif margin < 0:
                        winner = g("away_team")
                    else:
                        winner = "Tie"
                except Exception:
                    pass

            # Normalize market lines
            # Prefer closing lines for completed games; else use market/open values
            _hs = g("home_score"); _as = g("away_score")
            _is_final = (_hs is not None and not (isinstance(_hs, float) and pd.isna(_hs))) and (_as is not None and not (isinstance(_as, float) and pd.isna(_as)))
            if _is_final:
                m_spread = g("close_spread_home")
                if m_spread is None or (isinstance(m_spread, float) and pd.isna(m_spread)):
                    m_spread = g("market_spread_home", "spread_home", "open_spread_home")
                m_total = g("close_total")
                if m_total is None or (isinstance(m_total, float) and pd.isna(m_total)):
                    m_total = g("market_total", "total", "open_total")
            else:
                m_spread = g("market_spread_home", "spread_home", "open_spread_home")
                m_total = g("market_total", "total", "open_total")
            edge_spread = None
            edge_total = None
            try:
                if margin is not None and m_spread is not None and pd.notna(m_spread):
                    # Positive edge means model likes home side vs market (home covers if margin + spread > 0)
                    edge_spread = float(margin) + float(m_spread)
                if total_pred is not None and m_total is not None and pd.notna(m_total):
                    edge_total = float(total_pred) - float(m_total)
            except Exception:
                pass

            # Picks summary
            pick_spread = None
            pick_total = None
            try:
                if edge_spread is not None:
                    if edge_spread > 0.5:
                        pick_spread = f"{g('home_team')} covers by {edge_spread:+.1f}"
                    elif edge_spread < -0.5:
                        pick_spread = f"{g('away_team')} covers by {(-edge_spread):+.1f}"
                    else:
                        pick_spread = "No ATS edge"
                if edge_total is not None:
                    if edge_total > 0.5:
                        pick_total = f"Over by {edge_total:+.1f}"
                    elif edge_total < -0.5:
                        pick_total = f"Under by {(-edge_total):+.1f}"
                    else:
                        pick_total = "No total edge"
            except Exception:
                pass

            # Quarter/half breakdown (optional)
            quarters: List[Dict[str, Any]] = []
            for i in (1, 2, 3, 4):
                hq = g(f"pred_home_q{i}")
                aq = g(f"pred_away_q{i}")
                tq = g(f"pred_q{i}_total")
                wq = g(f"pred_q{i}_winner")
                if hq is not None or aq is not None or tq is not None:
                    quarters.append({
                        "label": f"Q{i}",
                        "home": hq,
                        "away": aq,
                        "total": tq,
                        "winner": wq,
                    })
            half1 = g("pred_h1_total")
            half2 = g("pred_h2_total")

            home = g("home_team")
            away = g("away_team")
            a_home = assets.get(str(home), {}) if home else {}
            a_away = assets.get(str(away), {}) if away else {}

            def logo_url(asset: Dict[str, Any]) -> Optional[str]:
                # If a custom logo provided, prefer it; else fallback to a generic placeholder
                if asset.get("logo"):
                    return asset.get("logo")
                abbr = asset.get("abbr")
                if abbr:
                    # Placeholder pattern; replace with your CDN if desired
                    espn_map = {"WAS": "wsh"}
                    code = espn_map.get(abbr.upper(), abbr.lower())
                    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
                return None

            # Actuals if present (for past games)
            actual_home = g("home_score")
            actual_away = g("away_score")
            actual_total = None
            actual_margin = None
            if actual_home is not None and actual_away is not None and pd.notna(actual_home) and pd.notna(actual_away):
                try:
                    actual_home_f = float(actual_home)
                    actual_away_f = float(actual_away)
                    actual_total = actual_home_f + actual_away_f
                    actual_margin = actual_home_f - actual_away_f
                except Exception:
                    pass

            # Status text
            status_text = "FINAL" if actual_total is not None else "Scheduled"

            # Winner correctness
            winner_correct = None
            if winner and actual_margin is not None:
                actual_winner = home if actual_margin > 0 else (away if actual_margin < 0 else "Tie")
                winner_correct = (winner == actual_winner)

            # ATS correctness
            ats_text = None
            ats_correct = None
            if m_spread is not None and pd.notna(m_spread):
                # Model side
                model_side = "Home" if (edge_spread is not None and edge_spread >= 0) else ("Away" if edge_spread is not None else None)
                model_team = home if model_side == "Home" else (away if model_side == "Away" else None)
                # Actual cover
                if actual_margin is not None:
                    # Home covers if actual margin + spread > 0
                    cover_val = actual_margin + float(m_spread)
                    actual_side = "Home" if cover_val > 0 else ("Away" if cover_val < 0 else "Push")
                    if model_side:
                        ats_correct = (model_side == actual_side) if actual_side != "Push" else None
                        if actual_side == "Push":
                            ats_text = "ATS: Push"
                        else:
                            actual_team = home if actual_side == "Home" else away
                            ats_text = f"ATS: {actual_team}"
                # Default text if no actuals
                if ats_text is None and model_side is not None:
                    if model_team:
                        ats_text = f"ATS: {model_team} (model)"

            # Totals correctness (O/U)
            ou_text = None
            totals_text = None
            totals_correct = None
            if m_total is not None and pd.notna(m_total):
                model_ou = "Over" if (edge_total is not None and edge_total > 0) else ("Under" if edge_total is not None else None)
                ou_text = f"O/U: {float(m_total):.2f} • Model: {model_ou or '—'} (Edge {edge_total:+.2f})" if edge_total is not None else f"O/U: {float(m_total):.2f}"
                if actual_total is not None:
                    if actual_total > float(m_total):
                        actual_ou = "Over"
                    elif actual_total < float(m_total):
                        actual_ou = "Under"
                    else:
                        actual_ou = "Push"
                    if model_ou and actual_ou != "Push":
                        totals_correct = (model_ou == actual_ou)
                        totals_text = f"Totals: {model_ou}"
                    elif actual_ou == "Push":
                        totals_text = "Totals: Push"

            # Win prob text (market-blended + shrink toward 0.5 for display)
            wp_home = g("pred_home_win_prob", "prob_home_win", default=None)
            try:
                if wp_home is not None and pd.notna(wp_home):
                    wp_home = float(wp_home)
                    # Blend with implied market probability if odds are present
                    ml_home_val = g("moneyline_home")
                    ml_away_val = g("moneyline_away")
                    mkt_ph, _ = _implied_probs_from_moneylines(ml_home_val, ml_away_val)
                    try:
                        beta_wp = float(os.environ.get('WP_MARKET_BLEND', '0.50'))  # 50% market weight by default
                    except Exception:
                        beta_wp = 0.50
                    p_home_eff = ((1.0 - beta_wp) * wp_home + beta_wp * mkt_ph) if mkt_ph is not None else wp_home
                    # Shrink toward 0.5 to avoid overconfident display
                    try:
                        shrink_wp = float(os.environ.get('WP_SHRINK', '0.35'))
                    except Exception:
                        shrink_wp = 0.35
                    p_home_disp = 0.5 + (p_home_eff - 0.5) * (1.0 - shrink_wp)
                    # Constrain within a band around market implied prob, if available
                    if mkt_ph is not None:
                        try:
                            band = float(os.environ.get('WP_MARKET_BAND', '0.12'))
                        except Exception:
                            band = 0.12
                        lower = max(0.0, mkt_ph - band)
                        upper = min(1.0, mkt_ph + band)
                        p_home_disp = min(max(p_home_disp, lower), upper)
                    # Hard clamp to avoid extreme display
                    try:
                        hard_min = float(os.environ.get('WP_HARD_MIN', '0.15'))
                        hard_max = float(os.environ.get('WP_HARD_MAX', '0.85'))
                    except Exception:
                        hard_min, hard_max = 0.15, 0.85
                    p_home_disp = max(hard_min, min(hard_max, p_home_disp))
                    wp_text = f"Win Prob: Away {((1.0 - p_home_disp)*100):.1f}% / Home {(p_home_disp*100):.1f}%"
                else:
                    wp_text = None
            except Exception:
                wp_text = None

            # Venue + datetime (best-effort)
            game_date = g("game_date", "date")
            # Stadium/TZ from meta with per-game override support
            smeta = stad_map.get(str(home), {}) if home else {}
            # load overrides (reload each request to reflect file changes)
            loc_ovr = _load_location_overrides()
            ovr = None
            gid = str(g("game_id")) if g("game_id") is not None else None
            if gid and loc_ovr['by_game_id']:
                ovr = loc_ovr['by_game_id'].get(gid)
            if ovr is None and game_date and home and away:
                # Normalize date to YYYY-MM-DD to match overrides file
                try:
                    date_key = pd.to_datetime(game_date, errors='coerce').date().isoformat()
                except Exception:
                    date_key = str(game_date)
                ovr = loc_ovr['by_match'].get((date_key, str(home), str(away))) if loc_ovr['by_match'] else None

            stadium = (ovr.get('venue') if (ovr and ovr.get('venue')) else (smeta.get('stadium') or home))
            tz = (ovr.get('tz') if (ovr and ovr.get('tz')) else (smeta.get('tz') or g("tz")))  # prefer override, then meta
            date_str = _format_game_datetime(game_date, tz)
            # location suffix if provided (city/country)
            loc_suffix = None
            if ovr:
                city = ovr.get('city')
                country = ovr.get('country')
                if city or country:
                    if city and country:
                        loc_suffix = f"{city}, {country}"
                    else:
                        loc_suffix = city or country
            venue_text = f"Venue: {stadium} • {date_str}" if date_str else f"Venue: {stadium}"
            if loc_suffix:
                venue_text = f"{venue_text} ({loc_suffix})"
            if ovr and ovr.get('neutral_site') not in (None, '', False, 0, '0', 'False', 'false', 'FALSE'):
                venue_text = f"{venue_text} • Neutral site"

            # Weather line
            wt_parts: List[str] = []
            temp_v = g("wx_temp_f", "temp_f", "temperature_f")
            wind_v = g("wx_wind_mph", "wind_mph")
            precip_v = g("wx_precip_pct", "precip_pct")
            if temp_v is not None and pd.notna(temp_v):
                try:
                    wt_parts.append(f"{float(temp_v):.0f}°F")
                except Exception:
                    pass
            if wind_v is not None and pd.notna(wind_v):
                try:
                    wt_parts.append(f"{float(wind_v):.0f} mph wind")
                except Exception:
                    pass
            if precip_v is not None and pd.notna(precip_v):
                try:
                    wt_parts.append(f"Precip {float(precip_v):.0f}%")
                except Exception:
                    pass
            # Placeholder for weather delta if available in future
            weather_text = f"Weather: {' • '.join(wt_parts)}" if wt_parts else None

            # Total diff (model vs actual) if both present
            total_diff = None
            try:
                if total_pred is not None and actual_total is not None:
                    total_diff = abs(float(total_pred) - float(actual_total))
            except Exception:
                total_diff = None

            # Roof/surface override for weather context
            roof_val = (ovr.get('roof') if (ovr and ovr.get('roof')) else g("stadium_roof", "roof"))
            surface_val = (ovr.get('surface') if (ovr and ovr.get('surface')) else g("surface"))

            cards.append({
                "season": g("season"),
                "week": g("week"),
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "pred_home_points": ph,
                "pred_away_points": pa,
                "pred_total": total_pred,
                "pred_home_win_prob": g("pred_home_win_prob", "prob_home_win", default=None),
                "market_spread_home": m_spread,
                "market_total": m_total,
                "pred_margin": margin,
                "pred_winner": winner,
                # Confidence (overall per-game)
                "game_confidence": g("game_confidence"),
                "edge_spread": edge_spread,
                "edge_total": edge_total,
                "pick_spread": pick_spread,
                "pick_total": pick_total,
                # Weather
                "stadium_roof": roof_val,
                "stadium_surface": surface_val,
                "wx_temp_f": g("wx_temp_f", "temp_f", "temperature_f"),
                "wx_wind_mph": g("wx_wind_mph", "wind_mph"),
                "wx_precip_pct": g("wx_precip_pct", "precip_pct"),
                # Colors
                "home_color": a_home.get("primary"),
                "home_color2": a_home.get("secondary"),
                "away_color": a_away.get("primary"),
                "away_color2": a_away.get("secondary"),
                "home_logo": logo_url(a_home),
                "away_logo": logo_url(a_away),
                # Periods
                "quarters": quarters,
                "half1_total": half1,
                "half2_total": half2,
                # Actuals & status
                "home_score": actual_home,
                "away_score": actual_away,
                "actual_total": actual_total,
                "status_text": status_text,
                # Assessments strings
                "wp_text": wp_text,
                "wp_blended": p_home_disp if 'p_home_disp' in locals() else None,
                "winner_correct": winner_correct,
                "ats_text": ats_text,
                "ats_correct": ats_correct,
                "ou_text": ou_text,
                "totals_text": totals_text,
                "totals_correct": totals_correct,
                # Venue text
                "venue_text": venue_text,
                "weather_text": weather_text,
                "total_diff": total_diff,
                # Odds (sanitize NaN -> None; cast to int for display)
                "moneyline_home": (int(g("moneyline_home")) if (g("moneyline_home") is not None and not pd.isna(g("moneyline_home"))) else None),
                "moneyline_away": (int(g("moneyline_away")) if (g("moneyline_away") is not None and not pd.isna(g("moneyline_away"))) else None),
                "close_spread_home": g("close_spread_home"),
                "close_total": g("close_total"),
                # Implied probabilities (computed below when possible)
                "implied_home_prob": None,
                "implied_away_prob": None,
                # Recommendations (EV-based)
                # Winner (moneyline): compute EV for Home/Away with available moneylines
                # Spread (win margin): EV at -110 using logistic prob from margin edge
                # Total: EV at -110 using logistic prob from total edge
            })

            # Compute recommendations and attach to last card
            c = cards[-1]
            # Winner EV
            try:
                p_home = float(wp_home) if (wp_home is not None and pd.notna(wp_home)) else None
            except Exception:
                p_home = None
            ml_home = g("moneyline_home")
            ml_away = g("moneyline_away")
            dec_home = _american_to_decimal(ml_home) if ml_home is not None else None
            dec_away = _american_to_decimal(ml_away) if ml_away is not None else None
            ev_home_ml = ev_away_ml = None
            # Implied probabilities for display
            iph, ipa = _implied_probs_from_moneylines(ml_home, ml_away)
            c["implied_home_prob"] = float(iph) if iph is not None else None
            c["implied_away_prob"] = float(ipa) if ipa is not None else None
            if p_home is not None:
                # Blend model prob with market implied prob to temper confidence
                mkt_ph, _ = _implied_probs_from_moneylines(ml_home, ml_away)
                try:
                    beta = float(os.environ.get('RECS_MARKET_BLEND', '0.35'))
                except Exception:
                    beta = 0.35
                if mkt_ph is not None:
                    p_home_eff = (1.0 - beta) * p_home + beta * mkt_ph
                else:
                    p_home_eff = p_home
                if dec_home:
                    ev_home_ml = _ev_from_prob_and_decimal(p_home_eff, dec_home)
                if dec_away:
                    ev_away_ml = _ev_from_prob_and_decimal(1.0 - p_home_eff, dec_away)
            # Choose best side
            winner_side = None
            winner_ev = None
            if ev_home_ml is not None or ev_away_ml is not None:
                # Prefer higher EV (even if negative) but only recommend if positive
                cand = [(home or "Home", ev_home_ml), (away or "Away", ev_away_ml)]
                cand = [(s, e) for s, e in cand if e is not None]
                if cand:
                    s, e = max(cand, key=lambda t: t[1])
                    winner_side, winner_ev = s, e
            c["rec_winner_side"] = winner_side
            c["rec_winner_ev"] = winner_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_winner_conf"] = _conf_from_ev(winner_ev) if winner_ev is not None else None

            # Spread (win margin) EV using actual prices when available (fallback to -110)
            ev_spread_home = ev_spread_away = None
            spread = m_spread
            if margin is not None and spread is not None and pd.notna(spread):
                # edge_pts = predicted margin + spread for home side (home covers if margin + spread > 0)
                try:
                    edge_pts = float(margin) + float(spread)
                    scale_margin = float(os.environ.get('NFL_ATS_SIGMA', '9.0'))
                except Exception:
                    edge_pts, scale_margin = None, 9.0
                if edge_pts is not None:
                    p_home_cover = _cover_prob_from_edge(edge_pts, scale_margin)
                    # Shrink toward 0.5 to reduce overconfidence
                    try:
                        shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
                    except Exception:
                        shrink = 0.35
                    p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)
                    sh_price = g("spread_home_price")
                    sa_price = g("spread_away_price")
                    dec_home_sp = _american_to_decimal(sh_price) if sh_price is not None and not pd.isna(sh_price) else (1.0 + 100.0/110.0)
                    dec_away_sp = _american_to_decimal(sa_price) if sa_price is not None and not pd.isna(sa_price) else (1.0 + 100.0/110.0)
                    ev_spread_home = _ev_from_prob_and_decimal(p_home_cover, dec_home_sp)
                    ev_spread_away = _ev_from_prob_and_decimal(1.0 - p_home_cover, dec_away_sp)
            spread_side = None
            spread_ev = None
            if ev_spread_home is not None or ev_spread_away is not None:
                cand = [(home or "Home", ev_spread_home), (away or "Away", ev_spread_away)]
                cand = [(s, e) for s, e in cand if e is not None]
                if cand:
                    s, e = max(cand, key=lambda t: t[1])
                    spread_side, spread_ev = s, e
            c["rec_spread_side"] = spread_side
            c["rec_spread_ev"] = spread_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_spread_conf"] = _conf_from_ev(spread_ev) if spread_ev is not None else None

            # Total EV using actual prices when available (fallback to -110)
            ev_over = ev_under = None
            if total_pred is not None and m_total is not None and pd.notna(m_total):
                try:
                    edge_t = float(total_pred) - float(m_total)
                    scale_total = float(os.environ.get('NFL_TOTAL_SIGMA', '10.0'))
                except Exception:
                    edge_t, scale_total = None, 10.0
                if edge_t is not None:
                    p_over = _cover_prob_from_edge(edge_t, scale_total)
                    try:
                        shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
                    except Exception:
                        shrink = 0.35
                    p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)
                    to_price = g("total_over_price")
                    tu_price = g("total_under_price")
                    dec_over = _american_to_decimal(to_price) if to_price is not None and not pd.isna(to_price) else (1.0 + 100.0/110.0)
                    dec_under = _american_to_decimal(tu_price) if tu_price is not None and not pd.isna(tu_price) else (1.0 + 100.0/110.0)
                    ev_over = _ev_from_prob_and_decimal(p_over, dec_over)
                    ev_under = _ev_from_prob_and_decimal(1.0 - p_over, dec_under)
            total_side = None
            total_ev = None
            if ev_over is not None or ev_under is not None:
                cand = [("Over", ev_over), ("Under", ev_under)]
                cand = [(s, e) for s, e in cand if e is not None]
                if cand:
                    s, e = max(cand, key=lambda t: t[1])
                    total_side, total_ev = s, e
            c["rec_total_side"] = total_side
            c["rec_total_ev"] = total_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_total_conf"] = _conf_from_ev(total_ev) if total_ev is not None else None

    # Apply sorting
    def _dt_key(card: Dict[str, Any]):
        try:
            return pd.to_datetime(card.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    if sort_param == "date":
        cards.sort(key=_dt_key)
    elif sort_param == "winner":
        cards.sort(key=lambda c: (c.get("rec_winner_ev") if c.get("rec_winner_ev") is not None else float('-inf')), reverse=True)
    elif sort_param == "ats":
        cards.sort(key=lambda c: (abs(c.get("edge_spread")) if c.get("edge_spread") is not None else float('-inf')), reverse=True)
    elif sort_param == "total":
        cards.sort(key=lambda c: (abs(c.get("edge_total")) if c.get("edge_total") is not None else float('-inf')), reverse=True)

    return render_template(
        "index.html",
        have_data=len(cards) > 0,
        cards=cards,
        season=season_param,
        week=week_param,
        sort=sort_param,
        total_rows=len(cards),
    )


@app.route("/table")
def table_view():
    df = _load_predictions()
    season_param = request.args.get("season")
    week_param = request.args.get("week")
    try:
        season_i = int(season_param) if season_param else None
        week_i = int(week_param) if week_param else None
    except Exception:
        season_i, week_i = None, None

    view_df = df.copy()
    if not view_df.empty:
        # Default to current week if no filters given
        if season_i is None and week_i is None:
            inferred = _infer_current_season_week(view_df)
            if inferred is not None:
                season_i, week_i = inferred
        if season_i is not None and "season" in view_df.columns:
            view_df = view_df[view_df["season"] == season_i]
        if week_i is not None and "week" in view_df.columns:
            view_df = view_df[view_df["week"] == week_i]

    show_cols = [
        c for c in [
            "season", "week", "game_date", "away_team", "home_team",
            "pred_away_points", "pred_home_points", "pred_total", "pred_home_win_prob",
            "market_spread_home", "market_total", "game_confidence",
        ] if c in view_df.columns
    ]
    rows = view_df[show_cols].to_dict(orient="records") if not view_df.empty else []

    return render_template(
        "table.html",
        have_data=not view_df.empty,
        total_rows=len(rows),
        rows=rows,
        show_cols=show_cols,
    season=season_i,
    week=week_i,
    )


@app.route("/api/refresh-data", methods=["POST", "GET"])
def refresh_data():
    """Synchronous refresh of predictions by invoking the pipeline.

    Query params:
      train=true  -> also run training before predicting
    """
    train = request.args.get("train", "false").lower() == "true"
    # On minimal web deploys (Render), heavy training libs may be absent.
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "Refresh disabled on Render minimal deploy. Run locally or add full requirements."}, 200
    py = sys.executable or "python"
    env = os.environ.copy()
    # Ensure repo root is on module path
    env["PYTHONPATH"] = str(BASE_DIR)
    cmds = []
    if train:
        cmds.append([py, "-m", "nfl_compare.src.train"])
    cmds.append([py, "-m", "nfl_compare.src.predict"])

    details = []
    rc_total = 0
    for c in cmds:
        try:
            res = subprocess.run(c, cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=600)
            details.append({
                "cmd": " ".join(c),
                "returncode": res.returncode,
                "stdout_tail": res.stdout[-1000:],
                "stderr_tail": res.stderr[-1000:],
            })
            rc_total += res.returncode
            if res.returncode != 0:
                break
        except Exception as e:
            return {"status": "error", "error": str(e), "details": details}, 500

    ok = (rc_total == 0)
    return {"status": "ok" if ok else "error", "details": details, "predictions_path": str(PRED_FILE)}, (200 if ok else 500)


@app.route("/api/refresh-odds", methods=["POST", "GET"])
def refresh_odds():
    """Fetch fresh NFL odds (moneyline/spreads/totals) and re-run predictions.

    Requires ODDS_API_KEY in environment. This will write data/real_betting_lines_YYYY_MM_DD.json
    and then execute the prediction pipeline so UI reflects updated lines.
    """
    # On minimal web deploys (Render), odds/client deps may be absent.
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "Odds refresh disabled on Render minimal deploy. Run locally or enable full requirements."}, 200
    py = sys.executable or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    details = []
    cmds = [
        [py, "-m", "nfl_compare.src.odds_api_client"],
        [py, "-m", "nfl_compare.src.predict"],
    ]
    rc_total = 0
    for c in cmds:
        try:
            res = subprocess.run(c, cwd=str(BASE_DIR / "nfl_compare"), env=env, capture_output=True, text=True, timeout=600)
            details.append({
                "cmd": " ".join(c),
                "returncode": res.returncode,
                "stdout_tail": res.stdout[-1000:],
                "stderr_tail": res.stderr[-1000:],
            })
            rc_total += res.returncode
            if res.returncode != 0:
                break
        except Exception as e:
            return {"status": "error", "error": str(e), "details": details}, 500
    ok = (rc_total == 0)
    return {"status": "ok" if ok else "error", "details": details}, (200 if ok else 500)


@app.route("/api/odds-coverage")
def odds_coverage():
    """Report coverage of odds fields across current lines and predictions."""
    from nfl_compare.src.data_sources import load_lines
    try:
        df = load_lines()
    except Exception:
        df = pd.DataFrame()
    total = int(len(df))
    def _cnt(col):
        return int((df[col].notna()).sum()) if (col in df.columns and not df.empty) else 0
    out = {
        "rows": total,
        "moneyline_home": _cnt("moneyline_home"),
        "moneyline_away": _cnt("moneyline_away"),
        "spread_home": _cnt("spread_home"),
    "spread_home_price": _cnt("spread_home_price"),
    "spread_away_price": _cnt("spread_away_price"),
    "total": _cnt("total"),
    "total_over_price": _cnt("total_over_price"),
    "total_under_price": _cnt("total_under_price"),
    }
    return jsonify(out)


@app.route("/api/eval")
def api_eval():
    """Return walk-forward evaluation metrics.

    Behavior:
    - If running on Render (RENDER env true), try to read a cached JSON at nfl_compare/data/eval_summary.json.
      If not present, return a 'skipped' status for safety.
    - Otherwise, execute the evaluator in-process and optionally write/update the cache when write_cache=true.
    Query params:
      - min_weeks (int): minimum weeks of prior data to train per season (default env NFL_EVAL_MIN_WEEKS_TRAIN=4)
      - write_cache (bool): if true, write results to cache file.
    """
    cache_path = DATA_DIR / "eval_summary.json"
    is_render = str(os.environ.get("RENDER", "")).lower() in {"1", "true", "yes"}
    min_weeks = os.environ.get("NFL_EVAL_MIN_WEEKS_TRAIN", "4")
    try:
        if request.args.get("min_weeks"):
            min_weeks = str(int(request.args.get("min_weeks")))
    except Exception:
        pass
    write_cache = str(request.args.get("write_cache", "false")).lower() in {"1","true","yes","y"}

    if is_render:
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return jsonify({"status": "ok", "from_cache": True, "cache_path": str(cache_path), "data": data})
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500
        return jsonify({"status": "skipped", "reason": "Evaluation disabled on Render; no cache present.", "cache_path": str(cache_path)}), 200

    # Local/full env: run the evaluation in-process
    try:
        from nfl_compare.scripts.evaluate_walkforward import walkforward_eval
        res = walkforward_eval(min_weeks_train=int(min_weeks))
        if write_cache:
            try:
                cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            except Exception:
                # Non-fatal
                pass
        return jsonify({"status": "ok", "from_cache": False, "data": res})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/venue-info")
def venue_info():
    gid = request.args.get('game_id')
    df = _load_predictions()
    if df is None or df.empty:
        return jsonify({"error": "no data"}), 404
    if gid:
        df = df[df['game_id'] == gid]
    if df.empty:
        return jsonify({"error": "game not found"}), 404
    stad_map = _load_stadium_meta_map()
    loc_ovr = _load_location_overrides()
    out = []
    for _, row in df.iterrows():
        home = str(row.get('home_team'))
        away = str(row.get('away_team'))
        game_date = row.get('game_date') if 'game_date' in df.columns else row.get('date')
        smeta = stad_map.get(home, {}) if home else {}
        # override lookup
        ovr = None
        gid_v = str(row.get('game_id')) if row.get('game_id') is not None else None
        if gid_v and loc_ovr['by_game_id']:
            ovr = loc_ovr['by_game_id'].get(gid_v)
        if ovr is None and game_date and home and away:
            try:
                date_key = pd.to_datetime(game_date, errors='coerce').date().isoformat()
            except Exception:
                date_key = str(game_date)
            ovr = loc_ovr['by_match'].get((date_key, home, away)) if loc_ovr['by_match'] else None
        stadium = (ovr.get('venue') if (ovr and ovr.get('venue')) else (smeta.get('stadium') or home))
        tz = (ovr.get('tz') if (ovr and ovr.get('tz')) else (smeta.get('tz') or row.get('tz')))
        date_str = None
        if game_date is not None:
            try:
                date_str = _format_game_datetime(game_date, tz)
            except Exception:
                date_str = None
        venue_text = f"Venue: {stadium} • {date_str}" if date_str else f"Venue: {stadium}"
        city = ovr.get('city') if ovr else None
        country = ovr.get('country') if ovr else None
        if city or country:
            suffix = f"{city}, {country}" if city and country else (city or country)
            venue_text = f"{venue_text} ({suffix})"
        if ovr and ovr.get('neutral_site') not in (None, '', False, 0, '0', 'False', 'false', 'FALSE'):
            venue_text = f"{venue_text} • Neutral site"
        out.append({
            'game_id': row.get('game_id'),
            'date': str(game_date),
            'home_team': home,
            'away_team': away,
            'stadium': stadium,
            'tz': tz,
            'venue_text': venue_text,
            'override_found': bool(ovr),
            'override': ovr or {},
        })
    return jsonify({'data': out})


@app.route("/api/backfill-close-lines", methods=["POST", "GET"])
def api_backfill_close_lines():
    """Run the backfill script to populate close_spread_home/close_total where missing.

    On Render, this is skipped to avoid heavy operations; run locally instead.
    Returns a brief report including counts updated and the output file path.
    """
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return jsonify({"status": "skipped", "reason": "Backfill disabled on Render minimal deploy. Run locally."}), 200
    py = sys.executable or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    cmd = [py, "-m", "nfl_compare.scripts.backfill_close_lines"]
    try:
        res = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=600)
        stdout_tail = res.stdout[-2000:]
        stderr_tail = res.stderr[-2000:]
        ok = (res.returncode == 0)
        return jsonify({
            "status": "ok" if ok else "error",
            "returncode": res.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }), (200 if ok else 500)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/debug-week-view")
def api_debug_week_view():
    """Debug endpoint: show per-game recommendation counts for a given season/week/date.
    Query: season, week, date, min_ev
    """
    try:
        pred_df = _load_predictions()
        games_df = _load_games()
        season = request.args.get("season")
        week = request.args.get("week")
        date = request.args.get("date")
        season_i = int(season) if season else None
        week_i = int(week) if week else None
        view_df = _build_week_view(pred_df, games_df, season_i, week_i)
        view_df = _attach_model_predictions(view_df)
        if date and not view_df.empty:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        # Optional min_ev override
        if request.args.get("min_ev"):
            os.environ['RECS_MIN_EV_PCT'] = str(request.args.get("min_ev"))
        out = []
        for _, row in view_df.iterrows():
            try:
                recs = _compute_recommendations_for_row(row)
            except Exception:
                recs = []
            out.append({
                'game_id': row.get('game_id'),
                'date': row.get('game_date') or row.get('date'),
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'recs': len(recs),
                'has_ml': bool(pd.notna(row.get('moneyline_home')) or pd.notna(row.get('moneyline_away'))),
                'has_spread': bool(pd.notna(row.get('close_spread_home')) or pd.notna(row.get('market_spread_home')) or pd.notna(row.get('spread_home'))),
                'has_total': bool(pd.notna(row.get('close_total')) or pd.notna(row.get('market_total')) or pd.notna(row.get('total'))),
                'pred_total': row.get('pred_total'),
                'pred_home_win_prob': row.get('pred_home_win_prob') or row.get('prob_home_win'),
            })
        return jsonify({'rows': len(out), 'data': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/inspect-game")
def api_inspect_game():
    """Inspect a single game's merged fields to debug odds missing issues.
    Query: game_id or (season, week, home_team, away_team)
    """
    try:
        pred_df = _load_predictions()
        games_df = _load_games()
        gid = request.args.get('game_id')
        season = request.args.get('season')
        week = request.args.get('week')
        home = request.args.get('home_team')
        away = request.args.get('away_team')
        season_i = int(season) if season else None
        week_i = int(week) if week else None
        view_df = _build_week_view(pred_df, games_df, season_i, week_i)
        view_df = _attach_model_predictions(view_df)
        if view_df is None or view_df.empty:
            return jsonify({'rows': 0, 'data': []})
        df = view_df.copy()
        if gid and 'game_id' in df.columns:
            df = df[df['game_id'].astype(str) == str(gid)]
        if not gid and home and away and {'home_team','away_team'}.issubset(df.columns):
            df = df[(df['home_team'].astype(str) == str(home)) & (df['away_team'].astype(str) == str(away))]
        # Return a compact set of fields
        keep = [c for c in [
            'game_id','season','week','game_date','date','home_team','away_team',
            'moneyline_home','moneyline_away','spread_home','total','spread_home_price','spread_away_price','total_over_price','total_under_price',
            'close_spread_home','close_total',
            'home_score','away_score'
        ] if c in df.columns]
        data = df[keep].to_dict(orient='records') if keep else df.to_dict(orient='records')
        return jsonify({'rows': len(data), 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # Local dev: python app.py
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port, debug=True)
