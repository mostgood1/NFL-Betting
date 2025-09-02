from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import json
import math
import traceback


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "nfl_compare" / "data"
PRED_FILE = DATA_DIR / "predictions.csv"
ASSETS_FILE = DATA_DIR / "nfl_team_assets.json"
STADIUM_META_FILE = DATA_DIR / "stadium_meta.csv"
LOCATION_OVERRIDES_FILE = DATA_DIR / "game_location_overrides.csv"

app = Flask(__name__)


def _load_predictions() -> pd.DataFrame:
    """Load predictions.csv if present; return empty DataFrame if missing."""
    try:
        if PRED_FILE.exists():
            df = pd.read_csv(PRED_FILE)
            # Normalize typical columns if present
            # Ensure week/season numeric for filtering/sorting
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
        th_low = float(os.environ.get('RECS_EV_THRESH_LOW', '4'))
    except Exception:
        th_low = 4.0
    try:
        th_med = float(os.environ.get('RECS_EV_THRESH_MED', '8'))
    except Exception:
        th_med = 8.0
    try:
        th_high = float(os.environ.get('RECS_EV_THRESH_HIGH', '15'))
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
            beta = float(os.environ.get('RECS_MARKET_BLEND', '0.50'))
        except Exception:
            beta = 0.50
        p_home_eff = (1.0 - beta) * p_home + beta * mkt_ph if mkt_ph is not None else p_home
        # Clamp to a band around market to avoid extreme EVs from model noise
        try:
            band = float(os.environ.get('RECS_MARKET_BAND', '0.10'))
        except Exception:
            band = 0.10
        if mkt_ph is not None:
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
            ev_conf = _conf_from_ev(e)
            # Game-level tier if provided
            game_conf = g("game_confidence")
            conf = _combine_confs(ev_conf, game_conf)
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
            })

    # Spread (ATS) at -110
    margin = None
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
            edge_pts = float(margin) - float(spread)
            scale_margin = float(os.environ.get('NFL_ATS_SIGMA', '9.0'))
        except Exception:
            edge_pts, scale_margin = None, 9.0
        if edge_pts is not None:
            p_home_cover = _cover_prob_from_edge(edge_pts, scale_margin)
            try:
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.50'))
            except Exception:
                shrink = 0.50
            p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)
            dec_110 = 1.909090909  # ~ -110
            ev_home = _ev_from_prob_and_decimal(p_home_cover, dec_110)
            ev_away = _ev_from_prob_and_decimal(1.0 - p_home_cover, dec_110)
            # Build selections
            try:
                sp = float(spread)
            except Exception:
                sp = None
            sh_price = g("spread_home_price")
            sa_price = g("spread_away_price")
            if sp is not None:
                home_sel = f"{home} {sp:+.1f}{(' ('+('%+d' % sh_price)+')') if (sh_price is not None and not pd.isna(sh_price)) else ''}"
                away_sel = f"{away} {(-sp):+.1f}{(' ('+('%+d' % sa_price)+')') if (sa_price is not None and not pd.isna(sa_price)) else ''}"
            else:
                home_sel = f"{home} ATS{(' ('+('%+d' % sh_price)+')') if (sh_price is not None and not pd.isna(sh_price)) else ''}"
                away_sel = f"{away} ATS{(' ('+('%+d' % sa_price)+')') if (sa_price is not None and not pd.isna(sa_price)) else ''}"
            # Choose best
            cand = [(home_sel, ev_home), (away_sel, ev_away)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                ev_conf = _conf_from_ev(e)
                game_conf = g("game_confidence")
                conf = _combine_confs(ev_conf, game_conf)
                recs.append({
                    "type": "SPREAD",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                })

    # Total at -110
    total_pred = g("pred_total")
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
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.50'))
            except Exception:
                shrink = 0.50
            p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)
            dec_110 = 1.909090909
            ev_over = _ev_from_prob_and_decimal(p_over, dec_110)
            ev_under = _ev_from_prob_and_decimal(1.0 - p_over, dec_110)
            # Choose best
            try:
                tot = float(m_total)
            except Exception:
                tot = None
            to_price = g("total_over_price")
            tu_price = g("total_under_price")
            over_sel = (f"Over {tot:.1f}" if tot is not None else "Over") + (f" ({to_price:+d})" if (to_price is not None and not pd.isna(to_price)) else "")
            under_sel = (f"Under {tot:.1f}" if tot is not None else "Under") + (f" ({tu_price:+d})" if (tu_price is not None and not pd.isna(tu_price)) else "")
            cand = [(over_sel, ev_over), (under_sel, ev_under)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                ev_conf = _conf_from_ev(e)
                game_conf = g("game_confidence")
                conf = _combine_confs(ev_conf, game_conf)
                recs.append({
                    "type": "TOTAL",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                })

    # Apply global filtering to reduce noise
    try:
        # Default minimum EV raised to 5% to avoid weak edges
        min_ev_pct = float(os.environ.get('RECS_MIN_EV_PCT', '5.0'))
    except Exception:
        min_ev_pct = 5.0
    filtered = [r for r in recs if (r.get('ev_pct') is not None and r.get('ev_pct') >= min_ev_pct)]
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
    df = _load_predictions()
    if df is None or df.empty:
        return {"rows": 0, "data": []}, 200

    # Filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    # Default: only show Week 1 of latest season if no explicit week/date is provided
    if not week and not date:
        try:
            if "season" in df.columns and not df["season"].isna().all():
                latest_season = int(df["season"].max())
                df = df[df["season"] == latest_season]
        except Exception:
            pass
        if "week" in df.columns:
            df = df[df["week"].astype(str) == "1"]
    try:
        if season and "season" in df.columns:
            df = df[df["season"].astype(str) == str(season)]
    except Exception:
        pass
    try:
        if week and "week" in df.columns:
            df = df[df["week"].astype(str) == str(week)]
    except Exception:
        pass
    if date:
        # Support either game_date or date field
        try:
            if "game_date" in df.columns:
                df = df[df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in df.columns:
                df = df[df["date"].astype(str).str[:10] == str(date)]
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
    for _, row in df.iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
    # Sort by confidence then EV desc
    def sort_key(r: Dict[str, Any]):
        w = r.get("sort_weight") or (0, -999)
        return (w[0], w[1])
    all_recs.sort(key=sort_key, reverse=True)
    for r in all_recs:
        r.pop("sort_weight", None)
    return {"rows": len(all_recs), "data": all_recs}, 200


@app.route("/recommendations")
def recommendations_page():
    """HTML page for recommendations, sorted and grouped by confidence."""
    df = _load_predictions()
    if df is None or df.empty:
        return render_template("recommendations.html", recs=[], groups={}, have_data=False)

    # Optional filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    active_week = None
    try:
        if season and "season" in df.columns:
            df = df[df["season"].astype(str) == str(season)]
    except Exception:
        pass
    try:
        if week and "week" in df.columns:
            df = df[df["week"].astype(str) == str(week)]
    except Exception:
        pass
    if date:
        try:
            if "game_date" in df.columns:
                df = df[df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in df.columns:
                df = df[df["date"].astype(str).str[:10] == str(date)]
        except Exception:
            pass
    # Default: Week 1 of latest season when no explicit week/date provided
    if not week and not date:
        try:
            if "season" in df.columns and not df["season"].isna().all():
                latest_season = int(df["season"].max())
                df = df[df["season"] == latest_season]
        except Exception:
            pass
        if "week" in df.columns:
            df = df[df["week"].astype(str) == "1"]
            active_week = 1
    else:
        try:
            active_week = int(week) if week else None
        except Exception:
            active_week = None

    # Global filter overrides (query)
    min_ev = request.args.get("min_ev")
    one = request.args.get("one_per_game")
    if min_ev:
        os.environ['RECS_MIN_EV_PCT'] = str(min_ev)
    if one is not None:
        os.environ['RECS_ONE_PER_GAME'] = str(one)
    all_recs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
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

    return render_template("recommendations.html", recs=all_recs, groups=groups, have_data=len(all_recs) > 0, week=active_week)


@app.route("/")
def index():
    df = _load_predictions()
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

    view_df = df.copy()
    if not view_df.empty:
        # If no explicit filters provided, default to "current" week inferred from dates
        if season_param is None and week_param is None:
            inferred = _infer_current_season_week(view_df)
            if inferred is not None:
                season_param, week_param = inferred
        if season_param is not None and "season" in view_df.columns:
            view_df = view_df[view_df["season"] == season_param]
        if week_param is not None and "week" in view_df.columns:
            view_df = view_df[view_df["week"] == week_param]

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
                        return v
                return default

            # Compute assessments
            # Normalize model team scores
            ph = g("pred_home_points", "pred_home_score")
            pa = g("pred_away_points", "pred_away_score")
            margin = None
            winner = None
            total_pred = g("pred_total")
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
            m_spread = g("market_spread_home", "spread_home", "open_spread_home")
            m_total = g("market_total", "total", "open_total")
            edge_spread = None
            edge_total = None
            try:
                if margin is not None and m_spread is not None and pd.notna(m_spread):
                    # Positive edge means model likes home side vs market
                    edge_spread = float(margin) - float(m_spread)
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
                    cover_val = actual_margin - float(m_spread)
                    actual_side = "Home" if cover_val > 0 else ("Away" if cover_val < 0 else "Push")
                    if model_side:
                        ats_correct = (model_side == actual_side) if actual_side != "Push" else None
                        if actual_side == "Push":
                            ats_text = "ATS: Push"
                        else:
                            actual_team = home if actual_side == "Home" else away
                            ats_text = f"ATS: {actual_team} {'Correct' if ats_correct else 'Wrong'}"
                # Default text if no actuals
                if ats_text is None and model_side is not None:
                    if model_team:
                        ats_text = f"ATS: {model_team} (model)"

            # Totals correctness (O/U)
            ou_text = None
            totals_text = None
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
                        totals_text = f"Totals: {model_ou} {'Correct' if model_ou == actual_ou else 'Wrong'}"
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
                "ou_text": ou_text,
                "totals_text": totals_text,
                # Venue text
                "venue_text": venue_text,
                "weather_text": weather_text,
                "total_diff": total_diff,
                # Odds
                "moneyline_home": g("moneyline_home"),
                "moneyline_away": g("moneyline_away"),
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
                # edge_pts = predicted margin - spread for home side
                try:
                    edge_pts = float(margin) - float(spread)
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


if __name__ == "__main__":
    # Local dev: python app.py
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port, debug=True)
