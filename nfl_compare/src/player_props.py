from __future__ import annotations

"""
Lightweight weekly player prop projections.

Inputs:
- Team-level context from td_likelihood (implied points, expected_tds, pace/pass-rush priors).
- Offensive player usage priors from data/player_usage_priors.csv (depth-chart based shares).

Outputs (per player):
- QB: pass_attempts, pass_yards, pass_tds, interceptions, rush_yards.
- RB/WR/TE: rush_attempts, rush_yards, targets, receptions, rec_yards, rec_tds.
- Defense (very coarse): expected tackles and sacks allocated to front-7/DBs using roster positions if available.
- Anytime TD probability via Poisson approx from expected TDs allocated by shares.

Heuristics are conservative and rely on league-average efficiency with modest adjustments from priors when available.
"""

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from functools import lru_cache

import numpy as np
import pandas as pd
import os

from .td_likelihood import compute_td_likelihood
from .team_normalizer import normalize_team_name
from .reconciliation import reconcile_props, summarize_errors  # calibration inputs
from .name_normalizer import normalize_name_loose

# Data dir shared with rest of project
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEPTH_OVERRIDES_FP = DATA_DIR / "depth_overrides.csv"  # columns: season, week(optional), team, player, position, rush_share, target_share, rz_rush_share, rz_target_share
EFF_PRIORS_FP = DATA_DIR / "player_efficiency_priors.csv"
QB_STARTERS_FP_TMPL = DATA_DIR / "qb_starters_{season}_wk1.csv"
PRED_NOT_ACTIVE_TMPL = DATA_DIR / "predicted_not_active_{season}_wk{week}.csv"


# --- League-average baselines (tunable via env later if needed) ---
LEAGUE_PLAYS_PER_TEAM = 64.0
LEAGUE_PASS_RATE = 0.58
LEAGUE_YPA = 6.6
LEAGUE_INT_RATE = 0.025  # per attempt
LEAGUE_SACK_RATE = 0.064  # per dropback
LEAGUE_YPC = 4.3
LEAGUE_CATCH_RATE = 0.64
LEAGUE_YPT = 7.8

# Position-specific receiving YPT heuristics
POS_YPT = {
    'WR': 8.3,
    'TE': 7.6,
    'RB': 6.4,
}

# Position-specific catch rate heuristics
POS_CATCH_RATE = {
    'WR': 0.62,
    'TE': 0.66,
    'RB': 0.72,
}

# Prior blending weights depending on sample size
def _blend_weight(n: float, k: float = 25.0) -> float:
    """Return weight in [0,1] to trust player-specific rate given sample size n.
    k is the pseudo-count; higher k means slower to trust the player's history.
    """
    try:
        n = float(n)
        return float(np.clip(n / (n + k), 0.0, 1.0))
    except Exception:
        return 0.0


def _jitter_scale() -> float:
    """Allow overriding jitter scale via env var for more predictable outputs.
    PROPS_JITTER_SCALE in [0..1]; default ~1.0 of coded values.
    """
    try:
        v = float(os.environ.get('PROPS_JITTER_SCALE', '1.0'))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return 1.0

def _obs_blend_weight(default: float = 0.45) -> float:
    """Observed SoD share blend weight for week>1; set via PROPS_OBS_BLEND (0..1)."""
    try:
        v = float(os.environ.get('PROPS_OBS_BLEND', str(default)))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return default


def _ema_blend_weight(default: float = 0.5) -> float:
    """EMA blend weight for team pass rate and plays; set via PROPS_EMA_BLEND (0..1)."""
    try:
        v = float(os.environ.get('PROPS_EMA_BLEND', str(default)))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return default


# --- Reconciliation-driven calibration helpers ---
def _calib_weight(env_key: str, default: float) -> float:
    try:
        v = float(os.environ.get(env_key, str(default)))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return default


def _load_recon_summary(season: int, prev_week: int) -> pd.DataFrame:
    """Fetch prior week reconciliation and summarize signed bias/MAE by position.
    Returns empty frame if unavailable.
    """
    if prev_week is None or prev_week <= 0:
        return pd.DataFrame()
    try:
        df = reconcile_props(int(season), int(prev_week))
        if df is None or df.empty:
            return pd.DataFrame()
        return summarize_errors(df)
    except Exception:
        return pd.DataFrame()


def _bias_frac(row: pd.Series, metric: str) -> float:
    """Convert signed bias to fractional error relative to actual mean for stability.
    Falls back to predicted mean if actual mean is tiny. Clamps to [-0.5, 0.5].
    """
    try:
        b = float(row.get(f"{metric}_bias", 0.0))
        act_mean = float(row.get(f"{metric}_act_mean", 0.0))
        pred_mean = float(row.get(f"{metric}_pred_mean", 0.0))
        denom = act_mean if act_mean > 1e-6 else (pred_mean if pred_mean > 1e-6 else 1.0)
        frac = b / denom
        return float(np.clip(frac, -0.5, 0.5))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def _load_efficiency_priors() -> pd.DataFrame:
    if not EFF_PRIORS_FP.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(EFF_PRIORS_FP)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize position
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper()
    # Clean numeric columns
    for c in [
        'targets','receptions','rec_yards','catch_rate','ypt','rz_targets','rz_target_rate',
        'rush_att','rush_yards','ypc','rz_rush_att','rz_rush_rate'
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Standardize player_id to string for joins
    if 'player_id' in df.columns:
        df['_pid'] = df['player_id'].astype(str)
    else:
        df['_pid'] = None
    # Build simple index by player name (normalized, loose) as fallback when no id linking is present
    df['_nm'] = df.get('player', pd.Series(dtype=str)).astype(str).map(normalize_name_loose)
    return df


@lru_cache(maxsize=4)
def _qb_rush_rate_priors(season: int) -> pd.DataFrame:
    """Build simple QB rushing attempts per game priors from last season weekly data.
    Returns columns: _pid (str), _nm (lower name), rush_att_pg (float).
    Falls back to empty on failure.
    """
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return pd.DataFrame(columns=['_pid','_nm','rush_att_pg'])
    try:
        seasons = [max(2002, int(season) - 1)]
        wk = nfl.import_weekly_data(seasons)
    except Exception:
        return pd.DataFrame(columns=['_pid','_nm','rush_att_pg'])
    if wk is None or wk.empty:
        return pd.DataFrame(columns=['_pid','_nm','rush_att_pg'])
    df = wk.copy()
    # Regular season only
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    # Identify carries
    ca_col = None
    for c in ['carries','rushing_attempts','rush_att','rushing_att']:
        if c in df.columns:
            ca_col = c; break
    if ca_col is None:
        return pd.DataFrame(columns=['_pid','_nm','rush_att_pg'])
    # Player id/name
    id_col = None
    for c in ['player_id','gsis_id','nfl_id']:
        if c in df.columns:
            id_col = c; break
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in df.columns:
            name_col = c; break
    # Position
    pos_col = None
    for c in ['position','player_position','pos']:
        if c in df.columns:
            pos_col = c; break
    if pos_col is None:
        return pd.DataFrame(columns=['_pid','_nm','rush_att_pg'])
    df = df.copy()
    df[ca_col] = pd.to_numeric(df[ca_col], errors='coerce').fillna(0.0)
    df['is_qb'] = df[pos_col].astype(str).str.upper().eq('QB')
    df = df[df['is_qb']]
    # Games proxy: count unique game_id rows per player
    gcol = 'game_id' if 'game_id' in df.columns else None
    if gcol is None:
        # fallback: approximate games by counting weeks
        gcol = 'week'
    grp = df.groupby([id_col or name_col])
    att = grp[ca_col].sum(min_count=1)
    games = grp[gcol].nunique() if gcol in df.columns else grp.size()
    out = pd.DataFrame({'_key': att.index, 'rush_att_pg': (att / games).astype(float)})
    # Map keys to pid and name for robust lookup
    out['_pid'] = out['_key'].astype(str)
    if name_col and name_col in df.columns and id_col:
        # Build a name lookup by latest seen name for each id
        nm_map = df[[id_col, name_col]].dropna().copy()
        nm_map[id_col] = nm_map[id_col].astype(str)
        nm_map = nm_map.drop_duplicates(subset=[id_col], keep='last')
        out = out.merge(nm_map.rename(columns={id_col: '_pid', name_col: '_nm_src'}), on='_pid', how='left')
        out['_nm'] = out['_nm_src'].astype(str).map(normalize_name_loose)
        out = out.drop(columns=['_key','_nm_src'])
    else:
        out['_nm'] = out['_key'].astype(str).map(normalize_name_loose)
        out = out.drop(columns=['_key'])
    # Clean
    out['rush_att_pg'] = pd.to_numeric(out['rush_att_pg'], errors='coerce').fillna(0.0)
    return out[['__dummy' if False else '_pid','_nm','rush_att_pg']]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _load_usage_priors() -> pd.DataFrame:
    fp = DATA_DIR / "player_usage_priors.csv"
    if not fp.exists():
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])
    try:
        df = pd.read_csv(fp)
        # Normalize team
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).apply(normalize_team_name)
        # Ensure key columns exist
        for c in ["season","team","player","position"]:
            if c not in df.columns:
                df[c] = None
        # Coerce shares to floats and clamp
        for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
            else:
                df[c] = 0.0
        # Deduplicate by summing shares across identical player rows
        df = (
            df
            .groupby(["season","team","player","position"], as_index=False)[
                ["rush_share","target_share","rz_rush_share","rz_target_share"]
            ]
            .sum()
        )
        # Drop rows with all-zero shares
        share_cols = ["rush_share","target_share","rz_rush_share","rz_target_share"]
        df = df.loc[(df[share_cols].sum(axis=1) > 0.0)].copy()
        # Per team renormalize by column to 1.0
        def _renorm(g: pd.DataFrame) -> pd.DataFrame:
            out = g.copy()
            for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
                if c in out.columns:
                    s = out[c].sum()
                    if s > 0:
                        out[c] = out[c] / s
            return out
        if not df.empty:
            # Silence pandas deprecation by excluding grouping columns during apply
            try:
                df = df.groupby(["season","team"], as_index=False).apply(_renorm, include_groups=False).reset_index(drop=True)
            except TypeError:
                # Fallback for older pandas without include_groups
                df = df.groupby(["season","team"], as_index=False, group_keys=False).apply(_renorm).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])


def _apply_depth_overrides(base: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    """Apply optional manual depth overrides from depth_overrides.csv for a given team.
    Overrides can be scoped by season and optionally week; per-team rows replace shares and are renormalized.
    """
    if base is None:
        base = pd.DataFrame(columns=["player","position","rush_share","target_share","rz_rush_share","rz_target_share"])
    if not DEPTH_OVERRIDES_FP.exists():
        return base
    try:
        od = pd.read_csv(DEPTH_OVERRIDES_FP)
    except Exception:
        return base
    if od is None or od.empty:
        return base
    # Normalize types
    for c in ["season","week"]:
        if c in od.columns:
            od[c] = pd.to_numeric(od[c], errors="coerce")
    if 'team' in od.columns:
        od['team'] = od['team'].astype(str).apply(normalize_team_name)
    # Filter rows for this team and season, and matching week if provided in overrides
    sub = od[(od.get('season').astype('Int64') == int(season)) & (od.get('team').astype(str) == team)].copy()
    if 'week' in sub.columns and not sub['week'].isna().all():
        # Take rows where week is null (applies to all weeks) or equals requested week
        sub = sub[(sub['week'].isna()) | (sub['week'].astype('Int64') == int(week))]
    if sub is None or sub.empty:
        return base
    keep_cols = [c for c in ["player","position","rush_share","target_share","rz_rush_share","rz_target_share"] if c in sub.columns]
    over = sub[keep_cols].copy()
    # Coerce shares and fill missing with 0; then renormalize by column
    for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
        if c not in over.columns:
            over[c] = 0.0
        over[c] = pd.to_numeric(over[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    # If position missing, try to infer from base
    if 'position' not in over.columns or over['position'].isna().all():
        try:
            over = over.merge(base[['player','position']], on='player', how='left')
        except Exception:
            pass
    # Renormalize shares to 1.0 per column if any non-zero
    for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
        s = float(over[c].sum())
        if s > 0:
            over[c] = over[c] / s
    # Replace base rows by player match; then add any new players from overrides
    try:
        base_no = base[~base['player'].astype(str).isin(over['player'].astype(str))].copy()
        out = pd.concat([base_no, over], ignore_index=True)
    except Exception:
        out = base
    return out


def _default_team_depth(team: str) -> pd.DataFrame:
    rows = [
        {"player": f"{team} QB1", "position": "QB", "rush_share": 0.10, "target_share": 0.00, "rz_rush_share": 0.10, "rz_target_share": 0.00},
        {"player": f"{team} RB1", "position": "RB", "rush_share": 0.45, "target_share": 0.10, "rz_rush_share": 0.50, "rz_target_share": 0.08},
        {"player": f"{team} RB2", "position": "RB", "rush_share": 0.25, "target_share": 0.05, "rz_rush_share": 0.25, "rz_target_share": 0.05},
        {"player": f"{team} WR1", "position": "WR", "rush_share": 0.03, "target_share": 0.25, "rz_rush_share": 0.02, "rz_target_share": 0.25},
        {"player": f"{team} WR2", "position": "WR", "rush_share": 0.02, "target_share": 0.20, "rz_rush_share": 0.01, "rz_target_share": 0.20},
        {"player": f"{team} WR3", "position": "WR", "rush_share": 0.01, "target_share": 0.12, "rz_rush_share": 0.01, "rz_target_share": 0.12},
        {"player": f"{team} TE1", "position": "TE", "rush_share": 0.00, "target_share": 0.15, "rz_rush_share": 0.00, "rz_target_share": 0.20},
        {"player": f"{team} TE2", "position": "TE", "rush_share": 0.00, "target_share": 0.05, "rz_rush_share": 0.00, "rz_target_share": 0.10},
    ]
    df = pd.DataFrame(rows)
    for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
        s = df[c].sum()
        if s > 0:
            df[c] = df[c] / s
    df["team"] = team
    return df


def _team_depth(usage: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    if usage is None or usage.empty:
        # Try roster-based fallback to get real names; else default
        rb = _roster_based_depth(season, team)
        return rb if rb is not None and not rb.empty else _default_team_depth(team)
    try:
        cols = list(usage.columns)
    except Exception:
        return _default_team_depth(team)
    if ("season" not in usage.columns) or ("team" not in usage.columns):
        rb = _roster_based_depth(season, team)
        return rb if rb is not None and not rb.empty else _default_team_depth(team)
    try:
        u = usage.copy()
        u["season"] = pd.to_numeric(u["season"], errors="coerce")
        u["team"] = u["team"].astype(str)
        u = u[(u["season"] == season) & (u["team"] == team)].copy()
    except Exception:
        rb = _roster_based_depth(season, team)
        return rb if rb is not None and not rb.empty else _default_team_depth(team)
    if u is None or u.empty:
        rb = _roster_based_depth(season, team)
        return rb if rb is not None and not rb.empty else _default_team_depth(team)
    # Ensure shares normalized
    for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
        if c not in u.columns:
            u[c] = 0.0
        u[c] = pd.to_numeric(u[c], errors="coerce").fillna(0.0)
        s = u[c].sum()
        if s > 0:
            u[c] = u[c] / s
    return u


@lru_cache(maxsize=4)
def _season_rosters(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
        ros = nfl.import_seasonal_rosters([int(season)])
        return ros if ros is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=4)
def _team_context_ema(season: int, until_week: int, alpha: float = 0.5) -> pd.DataFrame:
    """Compute EMA of team plays and pass rate through a given week (REG season).
    Returns columns: team, plays_ema, pass_rate_ema. Persists a snapshot CSV for visibility.
    """
    if until_week is None or int(until_week) <= 0:
        return pd.DataFrame(columns=['team','plays_ema','pass_rate_ema'])
    try:
        import nfl_data_py as nfl  # type: ignore
        wk = nfl.import_weekly_data([int(season)])
    except Exception:
        return pd.DataFrame(columns=['team','plays_ema','pass_rate_ema'])
    if wk is None or wk.empty or 'week' not in wk.columns:
        return pd.DataFrame(columns=['team','plays_ema','pass_rate_ema'])
    df = wk.copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    elif 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    df = df[df['week'] <= int(until_week)].copy()
    if df.empty:
        return pd.DataFrame(columns=['team','plays_ema','pass_rate_ema'])
    # Columns
    team_col = None
    for c in ['team','recent_team','team_abbr']:
        if c in df.columns:
            team_col = c; break
    pa_col = None
    for c in ['attempts','pass_attempts']:
        if c in df.columns:
            pa_col = c; break
    ca_col = None
    for c in ['carries','rushing_attempts','rush_att','rushing_att']:
        if c in df.columns:
            ca_col = c; break
    if not (team_col and pa_col and ca_col):
        return pd.DataFrame(columns=['team','plays_ema','pass_rate_ema'])
    sub = df[[team_col,'week',pa_col,ca_col]].copy().rename(columns={team_col:'team_src', pa_col:'pa', ca_col:'ca'})
    sub['team'] = sub['team_src'].astype(str).apply(normalize_team_name)
    sub['pa'] = pd.to_numeric(sub['pa'], errors='coerce').fillna(0.0)
    sub['ca'] = pd.to_numeric(sub['ca'], errors='coerce').fillna(0.0)
    sub['plays'] = sub['pa'] + sub['ca']
    sub['pass_rate'] = np.where(sub['plays']>0, sub['pa']/sub['plays'], np.nan)
    # EMA per team in week order
    out_rows = []
    for team, g in sub.sort_values(['team','week']).groupby('team'):
        ema_pr = None; ema_pl = None
        for _, r in g.iterrows():
            pr = r['pass_rate'] if np.isfinite(r['pass_rate']) else np.nan
            pl = r['plays']
            if ema_pr is None:
                ema_pr = pr if np.isfinite(pr) else 0.55
                ema_pl = pl
            else:
                ema_pr = alpha * (pr if np.isfinite(pr) else ema_pr) + (1-alpha) * ema_pr
                ema_pl = alpha * pl + (1-alpha) * ema_pl
        out_rows.append({'team': team, 'plays_ema': float(ema_pl), 'pass_rate_ema': float(ema_pr)})
    out = pd.DataFrame(out_rows)
    # Persist snapshot for transparency
    try:
        snap = DATA_DIR / f"team_context_ema_{season}_wk{until_week}.csv"
        out.to_csv(snap, index=False)
    except Exception:
        pass
    return out


@lru_cache(maxsize=4)
def _def_pos_tendencies(season: int, until_week: int) -> pd.DataFrame:
    """Compute defense position-vs-defense tendencies through until_week (REG):
    share and YPT allowed to WR/TE/RB. Returns columns: team, pos, share_mult, ypt_mult.
    Multipliers are small (≈0.9..1.1 for shares, 0.95..1.05 for YPT).
    """
    fp = DATA_DIR / f"pbp_{int(season)}.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    if df is None or df.empty:
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    # Filter REG and week <= until_week when available
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    if 'week' in df.columns and until_week and int(until_week) > 0:
        df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
        df = df[df['week'] <= int(until_week)]
    # Identify pass plays with receivers and defense team
    if 'pass' not in df.columns:
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    name_cols = ['receiver_player_name','receiver_name','receiver']
    id_cols = ['receiver_player_id','receiver_id']
    def_col = 'defteam' if 'defteam' in df.columns else ('defteam' if 'defteam' in df.columns else None)
    if def_col is None:
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    rec = df[(pd.to_numeric(df['pass'], errors='coerce').fillna(0).astype(int)==1)].copy()
    # Receiver identity
    def pick(row, keys):
        for k in keys:
            v = row.get(k)
            if v is not None and pd.notna(v):
                s = str(v).strip()
                if s:
                    return s
        return ''
    rec['rid'] = rec.apply(lambda r: pick(r, id_cols), axis=1)
    rec['rname'] = rec.apply(lambda r: pick(r, name_cols), axis=1)
    rec = rec[(rec['rid'].astype(str).str.len()>0) | (rec['rname'].astype(str).str.len()>0)].copy()
    # Yards gained
    if 'yards_gained' in rec.columns:
        rec['yg'] = pd.to_numeric(rec['yards_gained'], errors='coerce').fillna(0.0)
    elif 'receiving_yards' in rec.columns:
        rec['yg'] = pd.to_numeric(rec['receiving_yards'], errors='coerce').fillna(0.0)
    else:
        rec['yg'] = 0.0
    rec['def_team'] = rec[def_col].astype(str)
    # Map receiver to position via season rosters
    ros = _season_rosters(int(season))
    pos_map = None
    if ros is not None and not ros.empty:
        name_col = None
        for c in ['player_display_name','player_name','display_name','full_name','football_name']:
            if c in ros.columns:
                name_col = c; break
        id_map_col = None
        for c in ['gsis_id','player_id','nfl_id','pfr_id']:
            if c in ros.columns:
                id_map_col = c; break
        pos_col = 'position' if 'position' in ros.columns else ('depth_chart_position' if 'depth_chart_position' in ros.columns else None)
        m = None
        if id_map_col:
            m = ros[[id_map_col, (pos_col or 'position')]].copy().rename(columns={id_map_col:'rid', pos_col or 'position':'pos'})
            m['rid'] = m['rid'].astype(str)
        elif name_col:
            m = ros[[name_col, (pos_col or 'position')]].copy().rename(columns={name_col:'rname', pos_col or 'position':'pos'})
            m['rname'] = m['rname'].astype(str)
        pos_map = m
    if pos_map is not None and not pos_map.empty:
        rec = rec.merge(pos_map, on=('rid' if 'rid' in pos_map.columns else 'rname'), how='left')
    else:
        rec['pos'] = pd.NA
    rec['pos'] = rec['pos'].astype(str).str.upper()
    rec['group'] = np.where(rec['pos'].str.startswith('WR'),'WR', np.where(rec['pos'].str.startswith('TE'),'TE', np.where(rec['pos'].str.startswith('RB')|rec['pos'].str.startswith('HB'),'RB', 'OTHER')))
    rec = rec[rec['group'].isin(['WR','TE','RB'])].copy()
    if rec.empty:
        return pd.DataFrame(columns=['team','pos','share_mult','ypt_mult'])
    # Aggregates
    g = rec.groupby(['def_team','group'], as_index=False).agg(tg=('group','count'), yg=('yg','sum'))
    # League averages
    lg = g.groupby('group', as_index=False).agg(tg=('tg','sum'), yg=('yg','sum'))
    lg['ypt'] = np.where(lg['tg']>0, lg['yg']/lg['tg'], np.nan)
    tot = float(lg['tg'].sum())
    lg['share'] = np.where(tot>0, lg['tg']/tot, 0.0)
    # Defense shares
    d = g.merge(g.groupby('def_team', as_index=False)['tg'].sum().rename(columns={'tg':'tot'}), on='def_team', how='left')
    d['share'] = np.where(d['tot']>0, d['tg']/d['tot'], 0.0)
    d['ypt'] = np.where(d['tg']>0, d['yg']/d['tg'], np.nan)
    # Join against league to build multipliers
    m = d.merge(lg[['group','share','ypt']].rename(columns={'share':'lg_share','ypt':'lg_ypt'}), on='group', how='left')
    # Small weights and clamps
    share_w = 0.5
    ypt_w = 0.3
    m['share_mult'] = 1.0 + share_w * (m['share'] - m['lg_share'])
    m['ypt_mult'] = 1.0 + ypt_w * np.where(np.isfinite(m['lg_ypt']) & (m['lg_ypt']>0), (m['ypt'] - m['lg_ypt'])/m['lg_ypt'], 0.0)
    m['share_mult'] = m['share_mult'].clip(lower=0.90, upper=1.10)
    m['ypt_mult'] = m['ypt_mult'].clip(lower=0.95, upper=1.05)
    m['team'] = m['def_team'].astype(str).apply(normalize_team_name)
    m['pos'] = m['group']
    return m[['team','pos','share_mult','ypt_mult']]


@lru_cache(maxsize=4)
def _active_roster(season: int, week: int) -> pd.DataFrame:
    """Return weekly active roster flags per team and player.
    Columns: team (normalized), _pid (str, optional), _nm (normalized name), is_active (bool/int), status (str).
    Heuristics handle varying column names across nfl_data_py versions.
    """
    try:
        import nfl_data_py as nfl  # type: ignore
        wr = nfl.import_weekly_rosters([int(season)])
    except Exception:
        return pd.DataFrame(columns=['team','_pid','_nm','is_active','status'])
    if wr is None or wr.empty:
        return pd.DataFrame(columns=['team','_pid','_nm','is_active','status'])
    df = wr.copy()
    # Filter to requested week
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
        df = df[df['week'] == int(week)].copy()
    # Team
    tcol = None
    for c in ['team','recent_team','team_abbr']:
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        return pd.DataFrame(columns=['team','_pid','_nm','is_active','status'])
    # ID and name
    id_col = None
    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
        if c in df.columns:
            id_col = c; break
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in df.columns:
            name_col = c; break
    out = pd.DataFrame()
    keep = [tcol] + ([id_col] if id_col else []) + ([name_col] if name_col else [])
    out = df[keep].copy()
    out['team'] = out[tcol].astype(str).apply(normalize_team_name)
    out['_pid'] = out[id_col].astype(str) if id_col else None
    out['_nm'] = out[name_col].astype(str).map(normalize_name_loose) if name_col else None
    # Status columns vary; pick best available
    status_cols = [
        'game_status', 'gameday_status', 'status', 'injury_status', 'game_status_desc', 'status_desc'
    ]
    st = None
    for c in status_cols:
        if c in df.columns:
            st = c; break
    if st:
        s = df[[tcol] + ([id_col] if id_col else []) + ([name_col] if name_col else []) + [st]].copy()
        s = s.rename(columns={tcol:'team_src', (id_col or 'player'):'pid_src', (name_col or 'player'):'name_src', st:'status'})
        s['team'] = s['team_src'].astype(str).apply(normalize_team_name)
        if id_col:
            s['_pid'] = s['pid_src'].astype(str)
        if name_col:
            s['_nm'] = s['name_src'].astype(str).map(normalize_name_loose)
        out = out.merge(s[['team','_pid','_nm','status']], on=['team','_pid','_nm'], how='left')
    else:
        out['status'] = None
    # Heuristic is_active
    def _is_active(txt: Optional[str]) -> int:
        if txt is None or (isinstance(txt, float) and np.isnan(txt)):
            return 1  # assume active if unknown
        t = str(txt).strip().upper()
        if any(k in t for k in ['RESERVE', 'PRACTICE', 'IR', 'INACTIVE', 'PUP', 'NFI']):
            return 0
        if 'ACT' in t or 'ACTIVE' in t:
            return 1
        if t in {'QUESTIONABLE','PROBABLE','DOUBTFUL'}:
            return 1
        return 1
    out['is_active'] = out['status'].apply(_is_active).astype(int)
    # Dedup: keep last occurrence per team+pid or team+nm
    if id_col:
        out = out.sort_values(['team','_pid']).drop_duplicates(subset=['team','_pid'], keep='last')
    elif name_col:
        out = out.sort_values(['team','_nm']).drop_duplicates(subset=['team','_nm'], keep='last')
    return out[['team','_pid','_nm','is_active','status']]


def _roster_based_depth(season: int, team: str) -> pd.DataFrame:
    """Build a minimal usage table with real player names from rosters/depth charts.
    Produces typical depth allocations (QB1, RB1/RB2, WR1/WR2/WR3, TE1/TE2).
    """
    ros = _season_rosters(int(season))
    if ros is None or ros.empty:
        return pd.DataFrame()
    # Normalize team
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if not team_src:
        return pd.DataFrame()
    rdf = ros.copy()
    rdf['team_norm'] = rdf[team_src].astype(str).apply(normalize_team_name)
    rdf = rdf[rdf['team_norm'] == team].copy()
    if rdf.empty:
        return pd.DataFrame()
    # Depth info
    pos_col = 'depth_chart_position' if 'depth_chart_position' in rdf.columns else ('position' if 'position' in rdf.columns else None)
    if not pos_col:
        return pd.DataFrame()
    rdf[pos_col] = rdf[pos_col].astype(str).str.upper()
    # Helper to sort by depth_chart_order
    ord_col = 'depth_chart_order' if 'depth_chart_order' in rdf.columns else None
    if ord_col in rdf.columns:
        rdf['_ord'] = pd.to_numeric(rdf[ord_col], errors='coerce').fillna(99).astype(int)
    else:
        rdf['_ord'] = 99
    name_cols = ['first_name','last_name','player_name','player_display_name','display_name','full_name','football_name']
    def pick_name(row: pd.Series) -> str:
        fn = str(row.get('first_name') or '').strip()
        ln = str(row.get('last_name') or '').strip()
        if fn and ln:
            return f"{fn} {ln}".strip()
        for k in name_cols[2:]:
            v = row.get(k)
            if pd.notna(v):
                s = str(v).strip()
                if s:
                    return s
        return f"{team}"
    def top_group(keys: List[str], take: int) -> pd.DataFrame:
        g = rdf[rdf[pos_col].isin(keys)].copy()
        if g.empty:
            return g
        g = g.sort_values(['_ord']).head(take).copy()
        g['player'] = g.apply(pick_name, axis=1)
        return g
    qb = top_group(['QB'], 1)
    rb = top_group(['RB','HB'], 2)
    wr = top_group(['WR'], 3)
    te = top_group(['TE'], 2)
    # Weights
    def weights(base: List[float], n: int) -> List[float]:
        arr = (base[:n] + [0.0]*max(0, n-len(base))) if n>0 else []
        s = sum(arr)
        return [w/s for w in arr] if s>0 else ([1.0/n]*n if n>0 else [])
    qb_w = weights([0.10], len(qb))
    rb_w = weights([0.60,0.40], len(rb))
    wr_w = weights([0.40,0.35,0.25], len(wr))
    te_w = weights([0.65,0.35], len(te))
    rb_rz = weights([0.70,0.30], len(rb))
    wr_rz = weights([0.50,0.35,0.15], len(wr))
    te_rz = weights([0.70,0.30], len(te))
    rows = []
    def push(df: pd.DataFrame, pos: str, w: List[float], rz: List[float]):
        for i, (_, r) in enumerate(df.iterrows()):
            rows.append({
                'season': int(season), 'team': team, 'player': r.get('player'), 'position': pos,
                'rush_share': (w[i] if pos in {'RB','QB'} else 0.0),
                'target_share': (w[i] if pos in {'WR','TE','RB'} and pos != 'QB' else 0.0),
                'rz_rush_share': (rz[i] if pos in {'RB','QB'} else 0.0),
                'rz_target_share': (rz[i] if pos in {'WR','TE','RB'} and pos != 'QB' else 0.0),
            })
    push(qb,'QB',qb_w,qb_w)
    push(rb,'RB',rb_w,rb_rz)
    push(wr,'WR',wr_w,wr_rz)
    push(te,'TE',te_w,te_rz)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Renormalize by column
    for c in ['rush_share','target_share','rz_rush_share','rz_target_share']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        s = out[c].sum()
        if s>0:
            out[c] = out[c] / s
    return out


@lru_cache(maxsize=4)
def _season_to_date_usage(season: int, until_week: int) -> pd.DataFrame:
    """Aggregate season-to-date (through until_week) rush attempts and targets per team-player.
    Returns columns: team, player_id, rush_share_obs, target_share_obs.
    """
    if until_week is None or int(until_week) <= 0:
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    try:
        import nfl_data_py as nfl  # type: ignore
        wk = nfl.import_weekly_data([int(season)])
    except Exception:
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    if wk is None or wk.empty:
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    df = wk.copy()
    # Regular season only
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    elif 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    if 'week' not in df.columns:
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    df = df[df['week'] <= int(until_week)].copy()
    if df.empty:
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    # Identify cols
    team_col = None
    for c in ['team','recent_team','team_abbr']:
        if c in df.columns:
            team_col = c; break
    id_col = None
    for c in ['player_id','gsis_id','nfl_id','pfr_id']:
        if c in df.columns:
            id_col = c; break
    # Targets and rush attempts columns (nfl_data_py typically has 'targets' and 'attempts' for passing; rushing attempts 'carries' or 'rushing_attempts')
    tgt_col = 'targets' if 'targets' in df.columns else None
    ra_col = None
    for c in ['carries','rushing_attempts','rush_att','rushing_att']:
        if c in df.columns:
            ra_col = c; break
    if not (team_col and id_col and (tgt_col or ra_col)):
        return pd.DataFrame(columns=['team','player_id','rush_share_obs','target_share_obs'])
    keep = [team_col, id_col] + ([tgt_col] if tgt_col else []) + ([ra_col] if ra_col else [])
    sub = df[keep].copy()
    sub = sub.rename(columns={team_col: 'team_src', id_col: 'player_id', (tgt_col or 'targets'): 'targets', (ra_col or 'carries'): 'carries'})
    sub['team'] = sub['team_src'].astype(str).apply(normalize_team_name)
    sub['targets'] = pd.to_numeric(sub['targets'], errors='coerce').fillna(0.0)
    sub['carries'] = pd.to_numeric(sub['carries'], errors='coerce').fillna(0.0)
    agg = (
        sub.groupby(['team','player_id'], as_index=False)
           .agg(targets=('targets','sum'), carries=('carries','sum'))
    )
    # Compute team totals and shares
    team_tot = agg.groupby('team', as_index=False).agg(t_tg=('targets','sum'), t_ca=('carries','sum'))
    out = agg.merge(team_tot, on='team', how='left')
    out['target_share_obs'] = np.where(out['t_tg'] > 0, out['targets'] / out['t_tg'], 0.0)
    out['rush_share_obs'] = np.where(out['t_ca'] > 0, out['carries'] / out['t_ca'], 0.0)
    return out[['team','player_id','rush_share_obs','target_share_obs']]


def _team_roster_ids(season: int, team: str) -> pd.DataFrame:
    """Return a small dataframe mapping player display names to ids and positions for a team-season."""
    ros = _season_rosters(int(season))
    if ros is None or ros.empty:
        return pd.DataFrame()
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if not team_src:
        return pd.DataFrame()
    df = ros.copy()
    df['team_norm'] = df[team_src].astype(str).apply(normalize_team_name)
    df = df[df['team_norm'] == team].copy()
    if df.empty:
        return pd.DataFrame()
    # Columns for names and IDs
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in df.columns:
            name_col = c; break
    id_col = None
    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
        if c in df.columns:
            id_col = c; break
    pos_col = 'depth_chart_position' if 'depth_chart_position' in df.columns else ('position' if 'position' in df.columns else None)
    ord_col = 'depth_chart_order' if 'depth_chart_order' in df.columns else None
    keep = {}
    if name_col: keep[name_col] = 'player'
    if id_col: keep[id_col] = 'player_id'
    if pos_col: keep[pos_col] = 'position'
    if ord_col: keep[ord_col] = 'depth_chart_order'
    if not keep:
        return pd.DataFrame()
    out = df[list(keep.keys())].rename(columns=keep).copy()
    out['player'] = out.get('player', pd.Series(dtype=str)).astype(str)
    if 'position' in out.columns:
        out['position'] = out['position'].astype(str).str.upper()
    return out


@lru_cache(maxsize=2)
def _week1_qb_starters(season: int) -> pd.DataFrame:
    """Return mapping of team -> (player, optional player_id) using Week 1 PBP passers per team.
    Falls back to weekly stats only if PBP unavailable. Cached and persisted as CSV.
    """
    fp = QB_STARTERS_FP_TMPL.with_name(QB_STARTERS_FP_TMPL.name.format(season=int(season)))
    # Try cached file first
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                if 'team' in df.columns:
                    df['team'] = df['team'].astype(str).apply(normalize_team_name)
                return df
        except Exception:
            pass
    # Preferred: derive from PBP week 1
    starters = pd.DataFrame(columns=['team','player','player_id'])
    try:
        import nfl_data_py as nfl  # type: ignore
        pbp = nfl.import_pbp_data([int(season)])
        if pbp is not None and not pbp.empty and 'week' in pbp.columns:
            df = pbp.copy()
            # Filter week 1 regular season passes
            if 'season_type' in df.columns:
                df = df[df['season_type'].astype(str).str.upper() == 'REG']
            df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
            df = df[(df['week'] == 1) & (df.get('pass', 0) == 1)].copy()
            # Identify team and passer
            team_col = 'posteam' if 'posteam' in df.columns else None
            name_col = None
            for c in ['passer_player_name','passer','passer_name']:
                if c in df.columns:
                    name_col = c; break
            id_col = None
            for c in ['passer_player_id','passer_id']:
                if c in df.columns:
                    id_col = c; break
            if team_col and name_col:
                sub = df[[team_col, name_col] + ([id_col] if id_col else [])].copy()
                sub = sub.rename(columns={team_col:'team_abbr', name_col:'player', (id_col or 'player'):'player_id'})
                # Count pass attempts per passer per team
                sub['_pa'] = 1
                g = sub.groupby(['team_abbr','player'], as_index=False)['_pa'].sum()
                # Pick max per team
                g = g.sort_values(['team_abbr','_pa'], ascending=[True, False])
                g = g.groupby('team_abbr', as_index=False).first()
                g['team'] = g['team_abbr'].astype(str).apply(normalize_team_name)
                starters = g[['team','player']].copy()
                # Try to attach ids via rosters for season
                ros = _season_rosters(int(season))
                if ros is not None and not ros.empty:
                    name_map_col = None
                    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
                        if c in ros.columns:
                            name_map_col = c; break
                    id_map_col = None
                    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
                        if c in ros.columns:
                            id_map_col = c; break
                    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
                    if name_map_col and id_map_col and team_src:
                        m = ros[[name_map_col, id_map_col, team_src]].copy()
                        m['team'] = m[team_src].astype(str).apply(normalize_team_name)
                        m['_nm'] = m[name_map_col].astype(str).map(normalize_name_loose)
                        starters['_nm'] = starters['player'].astype(str).map(normalize_name_loose)
                        starters = starters.merge(m[['team','_nm', id_map_col]].rename(columns={id_map_col:'player_id'}), on=['team','_nm'], how='left').drop(columns=['_nm'])
    except Exception:
        pass
    # Fallback: weekly stats attempts (REG only)
    if starters is None or starters.empty:
        try:
            import nfl_data_py as nfl  # type: ignore
            wk = nfl.import_weekly_data([int(season)])
            if wk is not None and not wk.empty and 'week' in wk.columns:
                df = wk.copy()
                if 'season_type' in df.columns:
                    df = df[df['season_type'].astype(str).str.upper() == 'REG']
                elif 'game_type' in df.columns:
                    df = df[df['game_type'].astype(str).str.upper() == 'REG']
                df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
                df = df[df['week'] == 1].copy()
                team_col = None
                for c in ['team','recent_team','team_abbr']:
                    if c in df.columns:
                        team_col = c; break
                name_col = None
                for c in ['player_display_name','player_name','display_name','full_name','football_name']:
                    if c in df.columns:
                        name_col = c; break
                id_col = None
                for c in ['player_id','gsis_id','nfl_id','pfr_id']:
                    if c in df.columns:
                        id_col = c; break
                pa_col = None
                for c in ['attempts','pass_attempts']:
                    if c in df.columns:
                        pa_col = c; break
                if team_col and name_col and pa_col:
                    sub = df[[team_col, name_col] + ([id_col] if id_col else []) + ([pa_col])].copy()
                    sub = sub.rename(columns={team_col:'team_src', name_col:'player', (id_col or 'player'):'player_id', pa_col:'pass_attempts'})
                    sub['team'] = sub['team_src'].astype(str).apply(normalize_team_name)
                    sub['pass_attempts'] = pd.to_numeric(sub['pass_attempts'], errors='coerce').fillna(0)
                    starters = (
                        sub.sort_values(['team','pass_attempts'], ascending=[True, False])
                           .groupby('team', as_index=False)
                           .first()[['team','player','player_id']]
                    )
        except Exception:
            pass
    # Cache
    try:
        if starters is not None and not starters.empty:
            fp.parent.mkdir(parents=True, exist_ok=True)
            starters.to_csv(fp, index=False)
    except Exception:
        pass
    return starters if starters is not None else pd.DataFrame(columns=['team','player','player_id'])


def _split_tds(team_row: pd.Series) -> Dict[str, float]:
    exp_tds = _safe_float(team_row.get("expected_tds"), 0.0)
    # Use prior pass/rush if available, else league split
    is_home = int(team_row.get("is_home") or 0)
    if is_home:
        pr = team_row.get("home_pass_rate_prior")
        rr = team_row.get("home_rush_rate_prior")
    else:
        pr = team_row.get("away_pass_rate_prior")
        rr = team_row.get("away_rush_rate_prior")
    try:
        pr = float(pr)
    except Exception:
        pr = np.nan
    try:
        rr = float(rr)
    except Exception:
        rr = np.nan
    if not np.isfinite(pr) and not np.isfinite(rr):
        pr, rr = LEAGUE_PASS_RATE, 1.0 - LEAGUE_PASS_RATE
    elif not np.isfinite(pr):
        pr = max(0.0, min(1.0, 1.0 - rr))
    elif not np.isfinite(rr):
        rr = max(0.0, min(1.0, 1.0 - pr))
    # Favor pass TDs slightly
    w_rush = 0.35 * rr
    w_pass = 0.65 * pr
    s = w_rush + w_pass
    if s <= 0:
        w_rush, w_pass = 0.42, 0.58
        s = 1.0
    return {
        "rush_tds": exp_tds * (w_rush / s),
        "pass_tds": exp_tds * (w_pass / s),
    }


def _expected_plays(team_row: pd.Series) -> float:
    # Use pace prior if present; map seconds/play to plays via baseline 60 minutes * ~ (60 sec/min) / secs_per_play ~= 3600 / spp
    spp = team_row.get("pace_prior") or team_row.get("home_pace_prior") or team_row.get("away_pace_prior")
    try:
        spp = float(spp)
        if spp > 0:
            plays = 3600.0 / spp
            return float(np.clip(plays, 55.0, 72.0))
    except Exception:
        pass
    # Fallback: derive a small adjustment from offensive minus defensive EPA priors
    try:
        o = float(team_row.get("off_epa_prior") or 0.0)
    except Exception:
        o = 0.0
    try:
        d = float(team_row.get("opp_def_epa_prior") or 0.0)
    except Exception:
        d = 0.0
    diff = o - d
    adj = float(np.tanh(3.0 * diff))  # ~[-0.76, 0.76]
    return float(np.clip(LEAGUE_PLAYS_PER_TEAM * (1.0 + 0.05 * adj), 58.0, 70.0))


def _efficiency_scaler(team_row: pd.Series) -> float:
    # Convert (off_epa_prior - opp_def_epa_prior) to small multiplier
    try:
        o = float(team_row.get("off_epa_prior") or 0.0)
    except Exception:
        o = 0.0
    try:
        d = float(team_row.get("opp_def_epa_prior") or 0.0)
    except Exception:
        d = 0.0
    diff = o - d
    val = float(np.exp(2.0 * diff))
    return float(np.clip(val, 0.85, 1.15))


def compute_player_props(season: int, week: int) -> pd.DataFrame:
    teams = compute_td_likelihood(season=season, week=week)
    if teams is None or teams.empty:
        return pd.DataFrame(columns=["season","week","team","opponent","player","position"])

    usage = _load_usage_priors()
    # Load efficiency priors once
    eff_priors = _load_efficiency_priors()
    # Preload EMA context and defensive tendencies through previous week
    ema_all = _team_context_ema(int(season), max(0, int(week) - 1))
    def_tend_all = _def_pos_tendencies(int(season), max(0, int(week) - 1))
    # Calibration: prior week reconciliation summary by position
    recon_sum = _load_recon_summary(int(season), int(week) - 1)
    # Env-driven calibration strengths (0..1)
    alpha_vol = _calib_weight('PROPS_CALIB_ALPHA', 0.25)   # targets / rush attempts volume
    beta_yards = _calib_weight('PROPS_CALIB_BETA', 0.20)   # yards per volume (ypt/ypc proxy)
    gamma_rec = _calib_weight('PROPS_CALIB_GAMMA', 0.25)   # receptions via catch rate
    qb_pass = _calib_weight('PROPS_CALIB_QB', 0.20)        # QB pass attempts/yards/TD/INT scale

    rows: List[Dict] = []
    for _, tr in teams.iterrows():
        team = str(tr.get("team"))
        if not team:
            continue
        opp = str(tr.get("opponent"))
        is_home = int(tr.get("is_home") or 0)
        exp_tds = _safe_float(tr.get("expected_tds"), 0.0)
        split = _split_tds(tr)
        rush_tds = split["rush_tds"]
        pass_tds = split["pass_tds"]
        # Team context: base from priors/EPA with EMA smoothing if available
        plays_base = _expected_plays(tr)
        plays = plays_base
        if ema_all is not None and not ema_all.empty:
            em = ema_all[ema_all['team'] == team]
            if not em.empty and pd.notna(em.iloc[0].get('plays_ema')):
                w_ema = _ema_blend_weight(0.5)
                try:
                    ema_pl = float(em.iloc[0]['plays_ema'])
                    plays = float(np.clip((1.0 - w_ema) * plays_base + w_ema * ema_pl, 58.0, 72.0))
                except Exception:
                    plays = plays_base
        # Pass/rush rates for volume
        pr = None
        if is_home:
            pr = tr.get("home_pass_rate_prior")
        else:
            pr = tr.get("away_pass_rate_prior")
        # Prefer provided prior; else derive a pass rate from EPA differential for variety
        pr_val = _safe_float(pr, float('nan'))
        if not np.isfinite(pr_val):
            try:
                o = float(tr.get("off_epa_prior") or 0.0)
            except Exception:
                o = 0.0
            try:
                d = float(tr.get("opp_def_epa_prior") or 0.0)
            except Exception:
                d = 0.0
            # Map EPA diff to pass tendency around 0.55 ± 0.1
            pr_val = 0.55 + 0.10 * float(np.tanh(4.0 * (o - d)))
        # Blend pass rate with EMA if available
        pass_rate = float(np.clip(pr_val, 0.45, 0.70))
        if ema_all is not None and not ema_all.empty:
            em = ema_all[ema_all['team'] == team]
            if not em.empty and pd.notna(em.iloc[0].get('pass_rate_ema')):
                w_ema = _ema_blend_weight(0.5)
                try:
                    ema_pr = float(em.iloc[0]['pass_rate_ema'])
                    base_pr = pass_rate
                    pass_rate = float(np.clip((1.0 - w_ema) * base_pr + w_ema * ema_pr, 0.45, 0.70))
                except Exception:
                    pass
        rush_rate = max(0.0, 1.0 - pass_rate)
        dropbacks = plays * pass_rate
        attempts = max(0.0, dropbacks * (1.0 - LEAGUE_SACK_RATE))
        rush_att = plays * rush_rate
        eff = _efficiency_scaler(tr)
        team_pass_yards = attempts * LEAGUE_YPA * eff
        team_rush_yards = rush_att * LEAGUE_YPC * eff
        team_int = attempts * LEAGUE_INT_RATE

        # Apply reconciliation-based calibration at team/position level (week > 1)
        pos_bias = {}
        if recon_sum is not None and not recon_sum.empty and int(week) > 1:
            try:
                # Build a tiny dict: position -> bias fractions for key metrics
                for _, rr in recon_sum.iterrows():
                    posk = str(rr.get('position') or '').upper()
                    if not posk:
                        continue
                    pos_bias[posk] = {
                        't_frac': _bias_frac(rr, 'targets'),
                        'r_frac': _bias_frac(rr, 'rush_attempts'),
                        'rec_frac': _bias_frac(rr, 'receptions'),
                        'ryds_frac': _bias_frac(rr, 'rec_yards'),
                        'r_yds_frac': _bias_frac(rr, 'rush_yards'),
                        'patt_frac': _bias_frac(rr, 'pass_attempts'),
                        'pyds_frac': _bias_frac(rr, 'pass_yards'),
                        'ptd_frac': _bias_frac(rr, 'pass_tds'),
                        'int_frac': _bias_frac(rr, 'interceptions'),
                    }
            except Exception:
                pos_bias = {}

        depth = _team_depth(usage, int(tr.get("season")), team)
        # Apply manual overrides if present
        try:
            depth = _apply_depth_overrides(depth, int(season), int(week), team)
        except Exception:
            pass
        # Ensure share columns present
        for c in ["rush_share","target_share","rz_rush_share","rz_target_share"]:
            if c not in depth.columns:
                depth[c] = 0.0
            depth[c] = pd.to_numeric(depth[c], errors="coerce").fillna(0.0)
        # Prepare roster map early (used by blending and QB id)
        roster_map = _team_roster_ids(int(tr.get("season")), team)
        # Normalize pass target shares to typical positional mix
        desired_group = {"WR": 0.65, "TE": 0.25, "RB": 0.10}
        desired_group_rz = {"WR": 0.55, "TE": 0.35, "RB": 0.10}
        pcol = "rz_target_share" if depth["rz_target_share"].sum() > 0 else "target_share"
        # Softly nudge within groups toward desired totals (blend factor), not hard rescale
        def _group_sum(pos: str) -> float:
            m = depth["position"].astype(str).str.upper() == pos
            return float(depth.loc[m, pcol].sum())
        alpha = 0.5  # 0=no change, 1=hard rescale
        for pos in ("WR","TE","RB"):
            cur = _group_sum(pos)
            tgt = (desired_group_rz if pcol == "rz_target_share" else desired_group).get(pos, 0.0)
            if cur > 0 and tgt > 0:
                hard = tgt / cur
                fac = (1.0 - alpha) + alpha * hard
                m = depth["position"].astype(str).str.upper() == pos
                depth.loc[m, pcol] = depth.loc[m, pcol] * fac
        # Renormalize to 1
        s = depth[pcol].sum()
        if s > 0:
            depth[pcol] = depth[pcol] / s

        # Compute per-position ranks to vary efficiency within groups
        depth['pos_up'] = depth['position'].astype(str).str.upper()
        depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0), 0.0)
        depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0), 0.0)
        try:
            depth['recv_rank'] = depth.groupby('pos_up')['recv_strength'].rank(ascending=False, method='first')
            depth['rush_rank'] = depth.groupby('pos_up')['rush_strength'].rank(ascending=False, method='first')
        except Exception:
            depth['recv_rank'] = 1
            depth['rush_rank'] = 1

        def _rank_mult(pos: str, rank_val) -> float:
            try:
                r = int(rank_val)
            except Exception:
                r = 1
            arr = [1.0]
            if pos == 'WR':
                arr = [1.06, 1.02, 0.98, 0.95]
            elif pos == 'TE':
                arr = [1.03, 0.97, 0.95]
            elif pos == 'RB':
                arr = [1.02, 0.98, 0.96, 0.94]
            # default if rank exceeds array
            idx = max(1, r) - 1
            return arr[idx] if idx < len(arr) else arr[-1]

        def _rank_mult_rush(pos: str, rank_val) -> float:
            try:
                r = int(rank_val)
            except Exception:
                r = 1
            arr = [1.0]
            if pos == 'RB':
                arr = [1.04, 0.99, 0.96, 0.94]
            elif pos == 'QB':
                arr = [1.02]
            idx = max(1, r) - 1
            return arr[idx] if idx < len(arr) else arr[-1]

        # New: rank-weighted effective shares to diversify volumes
        def _rank_mult_targets(pos: str, rank_val) -> float:
            try:
                r = int(rank_val)
            except Exception:
                r = 1
            # Stronger separation for top options
            if pos == 'WR':
                arr = [1.25, 1.05, 0.92, 0.88, 0.85]
            elif pos == 'TE':
                arr = [1.15, 0.95, 0.90]
            elif pos == 'RB':
                arr = [1.10, 0.97, 0.92]
            else:
                arr = [1.0]
            idx = max(1, r) - 1
            return arr[idx] if idx < len(arr) else arr[-1]

        def _rank_mult_carries(pos: str, rank_val) -> float:
            try:
                r = int(rank_val)
            except Exception:
                r = 1
            if pos == 'RB':
                arr = [1.20, 1.00, 0.92, 0.88]
            elif pos == 'QB':
                arr = [1.00]
            else:
                arr = [1.0]
            idx = max(1, r) - 1
            return arr[idx] if idx < len(arr) else arr[-1]

        # If week > 1, blend in season-to-date observed shares (through week-1)
        obs = _season_to_date_usage(int(tr.get('season')), max(0, int(tr.get('week')) - 1))
        if obs is not None and not obs.empty:
            # Attach player_id via roster for better matching
            if roster_map is not None and not roster_map.empty and 'player_id' in roster_map.columns:
                depth = depth.merge(roster_map[['player','player_id','position']], on='player', how='left')
                depth['player_id'] = depth['player_id'].astype(str)
                obs['player_id'] = obs['player_id'].astype(str)
                obs_team = obs[obs['team'] == team][['player_id','rush_share_obs','target_share_obs']]
                depth = depth.merge(obs_team, on='player_id', how='left')
                # Blend observed shares into base shares (lightly, to improve stability)
                beta = _obs_blend_weight(0.45)
                depth['t_base'] = pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0)
                depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                depth['t_blend'] = np.where(depth['target_share_obs'].notna(), (1-beta)*depth['t_base'] + beta*depth['target_share_obs'], depth['t_base'])
                depth['r_blend'] = np.where(depth['rush_share_obs'].notna(), (1-beta)*depth['r_base'] + beta*depth['rush_share_obs'], depth['r_base'])
                # Replace base shares used in ranking strength
                depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth['t_blend'], 0.0)
                depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth['r_blend'], 0.0)

                # Coverage boost: inject prior-week observed players missing from depth with small floors (exclude QBs)
                try:
                    # Figure out who is missing
                    present_ids = set(depth['player_id'].dropna().astype(str).tolist())
                    cand = obs_team.copy()
                    # Thresholds for injection based on observed share
                    R_SHARE_THR = 0.05  # >=5% of team rush attempts in prev week
                    T_SHARE_THR = 0.08  # >=8% of team targets in prev week
                    cand = cand[(pd.to_numeric(cand['rush_share_obs'], errors='coerce').fillna(0.0) >= R_SHARE_THR) |
                                (pd.to_numeric(cand['target_share_obs'], errors='coerce').fillna(0.0) >= T_SHARE_THR)]
                    cand = cand[~cand['player_id'].isin(present_ids)]
                    if not cand.empty:
                        # Map ids to names and positions from roster
                        rm = roster_map.copy()
                        rm['player_id'] = rm['player_id'].astype(str)
                        add = cand.merge(rm[['player_id','player','position']], on='player_id', how='left')
                        add = add.dropna(subset=['player'])
                        if not add.empty:
                            rows_to_add = []
                            for _, rr in add.iterrows():
                                pos_up = str(rr.get('position') or '').upper()
                                if pos_up == 'QB':
                                    # QB handled separately via starter enforcement and priors
                                    continue
                                nm = rr.get('player')
                                r_obs = float(pd.to_numeric(rr.get('rush_share_obs'), errors='coerce') or 0.0)
                                t_obs = float(pd.to_numeric(rr.get('target_share_obs'), errors='coerce') or 0.0)
                                # Floors scaled by observed share with caps by position
                                r_floor = 0.0
                                t_floor = 0.0
                                if r_obs > 0:
                                    base = float(np.clip(0.6 * r_obs, 0.015, 0.15))
                                    if pos_up == 'WR':
                                        r_floor = float(min(base, 0.04))
                                    elif pos_up == 'TE':
                                        r_floor = float(min(base, 0.02))
                                    else:  # RB, FB, etc.
                                        r_floor = base
                                if t_obs > 0:
                                    base_t = float(np.clip(0.5 * t_obs, 0.025, 0.22))
                                    if pos_up == 'RB':
                                        t_floor = float(min(base_t, 0.12))
                                    elif pos_up == 'TE':
                                        t_floor = float(min(base_t, 0.18))
                                    else:  # WR default
                                        t_floor = float(min(base_t, 0.22))
                                # Skip if both floors are tiny
                                if (r_floor <= 0.0) and (t_floor <= 0.0):
                                    continue
                                rows_to_add.append({
                                    'season': int(tr.get('season')),
                                    'team': team,
                                    'player': nm,
                                    'position': pos_up if pos_up else 'WR',
                                    'rush_share': r_floor,
                                    'target_share': t_floor,
                                    'rz_rush_share': r_floor * 0.8,
                                    'rz_target_share': 0.0  # keep RZ targets conservative by default
                                })
                            if rows_to_add:
                                depth = pd.concat([depth, pd.DataFrame(rows_to_add)], ignore_index=True)
                                # Recompute strengths including injected players
                                depth['pos_up'] = depth['position'].astype(str).str.upper()
                                depth['t_base'] = pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0)
                                depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                                depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth.get('t_blend', depth['t_base']), 0.0)
                                depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth.get('r_blend', depth['r_base']), 0.0)
                except Exception:
                    # Fail-safe: ignore injection if anything goes wrong
                    pass

        # Build effective receiving target shares with rank multipliers and renormalize across receiving positions
        recv_mask = depth['pos_up'].isin(['WR','TE','RB'])
        depth['t_mult'] = depth.apply(lambda r: _rank_mult_targets(str(r['pos_up']), r.get('recv_rank')), axis=1)
        base_t_share = depth['t_blend'] if 't_blend' in depth.columns else pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0)
        depth['t_eff'] = np.where(recv_mask, base_t_share * depth['t_mult'], 0.0)
        # Calibrate receiving volume by position using prior-week target bias
        if pos_bias and int(week) > 1:
            for posk, key in [('WR','targets'),('TE','targets'),('RB','targets')]:
                if posk in pos_bias:
                    frac = pos_bias[posk].get('t_frac', 0.0)
                    if abs(frac) > 1e-6:
                        scale = float(np.clip(1.0 - alpha_vol * frac, 0.80, 1.20))
                        m = (depth['pos_up'] == posk) & recv_mask
                        if m.any():
                            depth.loc[m, 't_eff'] = depth.loc[m, 't_eff'] * scale
        t_sum = float(depth.loc[recv_mask, 't_eff'].sum())
        if t_sum > 0:
            depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'] / t_sum

        # Apply opponent defense position-vs-defense multipliers to receiving target shares, then renormalize
        def_share_mult = {'WR': 1.0, 'TE': 1.0, 'RB': 1.0}
        def_ypt_mult = {'WR': 1.0, 'TE': 1.0, 'RB': 1.0}
        if def_tend_all is not None and not def_tend_all.empty and opp:
            dt = def_tend_all[def_tend_all['team'] == opp]
            if not dt.empty:
                for pos_k in ['WR','TE','RB']:
                    row = dt[dt['pos'] == pos_k]
                    if not row.empty:
                        try:
                            def_share_mult[pos_k] = float(row.iloc[0]['share_mult'])
                        except Exception:
                            pass
                        try:
                            def_ypt_mult[pos_k] = float(row.iloc[0]['ypt_mult'])
                        except Exception:
                            pass
        for pos_k, mult in def_share_mult.items():
            m = (depth['pos_up'] == pos_k) & recv_mask
            if m.any():
                depth.loc[m, 't_eff'] = depth.loc[m, 't_eff'] * mult
        # Final renormalize to sum 1 within receivers
        t_sum = float(depth.loc[recv_mask, 't_eff'].sum())
        if t_sum > 0:
            depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'] / t_sum

        # Build effective rushing shares with rank multipliers and renormalize across RB+QB
        rush_mask = depth['pos_up'].isin(['RB','QB'])
        depth['r_mult'] = depth.apply(lambda r: _rank_mult_carries(str(r['pos_up']), r.get('rush_rank')), axis=1)
        base_r_share = depth['r_blend'] if 'r_blend' in depth.columns else pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
        depth['r_eff'] = np.where(rush_mask, base_r_share * depth['r_mult'], 0.0)
        # Calibrate rushing volume by RB using prior-week rush_attempts bias (QB rush left untouched here)
        if pos_bias and int(week) > 1 and 'RB' in pos_bias:
            frac = pos_bias['RB'].get('r_frac', 0.0)
            if abs(frac) > 1e-6:
                scale = float(np.clip(1.0 - alpha_vol * frac, 0.80, 1.20))
                m = (depth['pos_up'] == 'RB') & rush_mask
                if m.any():
                    depth.loc[m, 'r_eff'] = depth.loc[m, 'r_eff'] * scale
        r_sum = float(depth.loc[rush_mask, 'r_eff'].sum())
        if r_sum > 0:
            depth.loc[rush_mask, 'r_eff'] = depth.loc[rush_mask, 'r_eff'] / r_sum
        # Effective RZ rush shares scaled similarly
        depth['rrz_eff'] = np.where(rush_mask, pd.to_numeric(depth['rz_rush_share'], errors='coerce').fillna(0.0) * depth['r_mult'], 0.0)
        rrz_sum = float(depth.loc[rush_mask, 'rrz_eff'].sum())
        if rrz_sum > 0:
            depth.loc[rush_mask, 'rrz_eff'] = depth.loc[rush_mask, 'rrz_eff'] / rrz_sum

        def _stable_jitter(name: str, key: str, scale: float = 0.03) -> float:
            try:
                h = hashlib.sha256(f"{name}|{key}".encode('utf-8')).hexdigest()
                v = int(h[:8], 16) / float(16**8)
                return 1.0 + (v - 0.5) * 2.0 * scale  # 1±scale
            except Exception:
                return 1.0

        # Identify a primary QB to receive passing stats
        # Preference order:
        # 1) Week 1 starter (most pass attempts in Week 1 via weekly stats)
        # 2) Roster depth_chart_order (lowest number)
        # 3) Highest combined shares proxy from usage depth
        qb_name = None
        try:
            wk1 = _week1_qb_starters(int(tr.get("season")))
            if wk1 is not None and not wk1.empty:
                qb_row = wk1[wk1['team'] == team]
                if not qb_row.empty:
                    qb_name = str(qb_row.iloc[0]['player'])
        except Exception:
            pass
        if roster_map is not None and not roster_map.empty:
            qbs = roster_map[roster_map.get('position').astype(str).str.upper() == 'QB'].copy()
            if not qbs.empty:
                if not qb_name and 'depth_chart_order' in qbs.columns:
                    qbs['_ord'] = pd.to_numeric(qbs['depth_chart_order'], errors='coerce').fillna(99).astype(int)
                    qb_name = str(qbs.sort_values('_ord').iloc[0].get('player'))
                elif not qb_name:
                    qb_name = str(qbs.iloc[0].get('player'))
        if not qb_name:
            qb_mask = depth["position"].astype(str).str.upper() == "QB"
            qb_candidates = depth[qb_mask].copy()
            if not qb_candidates.empty:
                qb_candidates = qb_candidates.assign(
                    _score=(qb_candidates[["rush_share","target_share","rz_rush_share","rz_target_share"]].sum(axis=1))
                )
                qb_name = str(qb_candidates.sort_values("_score", ascending=False).iloc[0].get("player"))

        # Enforce Week 1 starter as the sole QB row in depth to avoid backups getting passing stats
        try:
            if qb_name:
                qb_mask = depth["position"].astype(str).str.upper() == "QB"
                share_cols = ["rush_share","target_share","rz_rush_share","rz_target_share"]
                # Ensure share columns exist
                for c in share_cols:
                    if c not in depth.columns:
                        depth[c] = 0.0
                    depth[c] = pd.to_numeric(depth[c], errors="coerce").fillna(0.0)
                if qb_mask.any():
                    qb_rows = depth[qb_mask].copy()
                    sums = {c: float(qb_rows[c].sum()) for c in share_cols}
                    # If named starter not present, insert a row with combined QB shares; else combine shares into that row
                    qb_present = qb_rows['player'].astype(str).eq(qb_name).any()
                    # Drop all QB rows
                    depth = depth[~qb_mask].copy()
                    # Insert the starter QB with full QB shares
                    new_row = {"season": int(tr.get("season")), "team": team, "player": qb_name, "position": "QB"}
                    new_row.update({c: sums.get(c, 0.0) for c in share_cols})
                    # QB shouldn't get target share in our model; set target shares to 0
                    new_row["target_share"] = 0.0
                    new_row["rz_target_share"] = 0.0
                    depth = pd.concat([depth, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    # No QB rows existed; add one minimal QB row for starter
                    depth = pd.concat([depth, pd.DataFrame([{ "season": int(tr.get("season")), "team": team, "player": qb_name, "position": "QB", "rush_share": 0.10, "target_share": 0.0, "rz_rush_share": 0.10, "rz_target_share": 0.0 }])], ignore_index=True)
        except Exception:
            pass
        # Anchor QB rushing share using last-season priors to avoid unrealistic volumes
        try:
            qb_mask = depth['position'].astype(str).str.upper() == 'QB'
            if qb_mask.any():
                pri = _qb_rush_rate_priors(int(tr.get('season')))
                qb_share_est = None
                if pri is not None and not pri.empty:
                    att_pg = np.nan
                    if roster_map is not None and 'player_id' in roster_map.columns:
                        pid = roster_map.loc[roster_map['player'] == qb_name, 'player_id']
                        if not pid.empty:
                            row = pri[pri['_pid'] == str(pid.iloc[0])]
                            if not row.empty:
                                att_pg = float(row.iloc[0]['rush_att_pg'])
                    if not np.isfinite(att_pg):
                        nm = normalize_name_loose(str(qb_name or ''))
                        row = pri[pri['_nm'] == nm]
                        if not row.empty:
                            att_pg = float(row.iloc[0]['rush_att_pg'])
                    if np.isfinite(att_pg):
                        # Convert to share of team rush attempts per game (assume ~26 team rush attempts)
                        qb_share_est = float(np.clip(att_pg / 26.0, 0.01, 0.22))
                if qb_share_est is None:
                    # Fallback using efficiency priors rush_att if available (~17 games)
                    if eff_priors is not None and not eff_priors.empty:
                        att = np.nan
                        match = pd.DataFrame()
                        if roster_map is not None and 'player_id' in roster_map.columns:
                            pid = roster_map.loc[roster_map['player'] == qb_name, 'player_id']
                            if not pid.empty:
                                match = eff_priors[eff_priors['_pid'] == str(pid.iloc[0])]
                        if match is None or match.empty:
                            nm = normalize_name_loose(str(qb_name or ''))
                            match = eff_priors[eff_priors['_nm'] == nm]
                        if not match.empty and 'rush_att' in match.columns:
                            att = float(match['rush_att'].iloc[0])
                        if np.isfinite(att):
                            qb_share_est = float(np.clip((att / 17.0) / 26.0, 0.01, 0.22))
                if qb_share_est is None:
                    # Conservative default
                    qb_share_est = 0.06
                # Assign estimated QB rush share; other shares will be renormalized downstream
                depth.loc[qb_mask, 'rush_share'] = float(qb_share_est)
                # Slightly lower QB RZ share baseline relative to non-RZ to reflect scrambles
                if 'rz_rush_share' in depth.columns:
                    depth.loc[qb_mask, 'rz_rush_share'] = float(np.clip(qb_share_est * 0.8, 0.005, 0.18))
        except Exception:
            pass
        # After enforcing QB starter, recompute rushing effective shares so RB+QB sum to 1
        try:
            depth['pos_up'] = depth['position'].astype(str).str.upper()
            rush_mask = depth['pos_up'].isin(['RB','QB'])
            base_r_share = depth['r_blend'] if 'r_blend' in depth.columns else pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
            # Recompute rush strength/ranks and multipliers including the (new) QB row
            depth['rush_strength'] = np.where(rush_mask, base_r_share, 0.0)
            try:
                depth['rush_rank'] = depth.groupby('pos_up')['rush_strength'].rank(ascending=False, method='first')
            except Exception:
                depth['rush_rank'] = 1
            depth['r_mult'] = depth.apply(lambda r: _rank_mult_carries(str(r['pos_up']), r.get('rush_rank')), axis=1)
            depth['r_eff'] = np.where(rush_mask, base_r_share * depth['r_mult'], 0.0)
            r_sum = float(depth.loc[rush_mask, 'r_eff'].sum())
            if r_sum > 0:
                depth.loc[rush_mask, 'r_eff'] = depth.loc[rush_mask, 'r_eff'] / r_sum
            # Recompute effective red-zone rush share, then renormalize
            depth['rrz_eff'] = np.where(rush_mask, pd.to_numeric(depth['rz_rush_share'], errors='coerce').fillna(0.0) * depth['r_mult'], 0.0)
            rrz_sum = float(depth.loc[rush_mask, 'rrz_eff'].sum())
            if rrz_sum > 0:
                depth.loc[rush_mask, 'rrz_eff'] = depth.loc[rush_mask, 'rrz_eff'] / rrz_sum
        except Exception:
            pass
        # Compute per-player projections for this team
        for _, prw in depth.iterrows():
            pos = str(prw.get("position") or "")
            pl_name = prw.get("player")
            # Attach player_id if available on roster_map for later active-roster join
            pl_pid = None
            try:
                if roster_map is not None and not roster_map.empty and 'player_id' in roster_map.columns:
                    m = roster_map.loc[roster_map['player'] == pl_name, 'player_id']
                    if not m.empty:
                        pl_pid = str(m.iloc[0])
            except Exception:
                pl_pid = None
            r_share = float(prw.get('r_eff') if prw.get('r_eff') is not None else prw.get('rush_share') or 0.0)
            t_share = float(prw.get('t_eff') if prw.get('t_eff') is not None else prw.get(pcol) or 0.0)
            # Volumes
            pl_rush_att = rush_att * r_share
            pl_targets = dropbacks * t_share
            # Yards
            # Receiving efficiency varies by position and rank; add small stable jitter and blend with player priors when available
            pos_up = pos.upper()
            cr = POS_CATCH_RATE.get(pos_up, LEAGUE_CATCH_RATE)
            ypt = POS_YPT.get(pos_up, LEAGUE_YPT)
            # Blend with player priors for catch_rate and ypt using targets as sample size (prefer ID join via roster map)
            pri_cr = np.nan; pri_ypt = np.nan; pri_tgts = 0.0
            if eff_priors is not None and not eff_priors.empty:
                try:
                    # Try ID-based link
                    match = pd.DataFrame()
                    if roster_map is not None and 'player_id' in roster_map.columns:
                        pid = roster_map.loc[roster_map['player'] == pl_name, 'player_id']
                        if not pid.empty:
                            match = eff_priors[eff_priors['_pid'] == str(pid.iloc[0])]
                    if match is None or match.empty:
                        nm = str(pl_name or '').lower()
                        match = eff_priors[eff_priors['_nm'] == nm]
                    if not match.empty:
                        pri_cr = float(match['catch_rate'].iloc[0]) if 'catch_rate' in match.columns else np.nan
                        pri_ypt = float(match['ypt'].iloc[0]) if 'ypt' in match.columns else np.nan
                        pri_tgts = float(match['targets'].iloc[0]) if 'targets' in match.columns else 0.0
                except Exception:
                    pass
            w_cr = _blend_weight(pri_tgts, k=30.0)
            if np.isfinite(pri_cr):
                cr = (1 - w_cr) * cr + w_cr * float(np.clip(pri_cr, 0.35, 0.90))
            if np.isfinite(pri_ypt):
                ypt = (1 - w_cr) * ypt + w_cr * float(np.clip(pri_ypt, 4.0, 12.0))
            rr = prw.get('recv_rank')
            rrm = _rank_mult(pos_up, rr)
            jitter_rec = _stable_jitter(pl_name, 'rec', scale=0.035 * _jitter_scale())
            cr_eff = cr * jitter_rec
            ypt_eff = ypt * rrm * jitter_rec
            # Calibrate WR/TE/RB reception efficiency using prior-week biases
            if pos_bias and int(week) > 1 and pos_up in pos_bias:
                pb = pos_bias[pos_up]
                # Catch rate from receptions bias; yards per target from rec_yards bias
                rec_scale = float(np.clip(1.0 - gamma_rec * pb.get('rec_frac', 0.0), 0.85, 1.15))
                ypt_scale = float(np.clip(1.0 - beta_yards * pb.get('ryds_frac', 0.0), 0.85, 1.15))
                cr_eff *= rec_scale
                ypt_eff *= ypt_scale
            # Apply opponent YPT defense multiplier
            if pos_up in def_ypt_mult:
                try:
                    ypt_eff = ypt_eff * float(def_ypt_mult[pos_up])
                except Exception:
                    pass
            pl_rec = pl_targets * cr_eff
            pl_rec_yards = pl_targets * ypt_eff * eff

            # Rushing efficiency: vary YPC by rank for RB/QB and jitter
            rrank = prw.get('rush_rank')
            # Blend player YPC priors using rush_att as sample size
            ypc_mult = _rank_mult_rush(pos_up, rrank)
            pri_ypc = np.nan; pri_rush = 0.0
            if eff_priors is not None and not eff_priors.empty:
                try:
                    match = pd.DataFrame()
                    if roster_map is not None and 'player_id' in roster_map.columns:
                        pid = roster_map.loc[roster_map['player'] == pl_name, 'player_id']
                        if not pid.empty:
                            match = eff_priors[eff_priors['_pid'] == str(pid.iloc[0])]
                    if match is None or match.empty:
                        nm = str(pl_name or '').lower()
                        match = eff_priors[eff_priors['_nm'] == nm]
                    if not match.empty and 'ypc' in match.columns:
                        pri_ypc = float(match['ypc'].iloc[0])
                        pri_rush = float(match['rush_att'].iloc[0]) if 'rush_att' in match.columns else 0.0
                except Exception:
                    pass
            w_ypc = _blend_weight(pri_rush, k=40.0)
            base_ypc = LEAGUE_YPC * ypc_mult
            if np.isfinite(pri_ypc):
                base_ypc = (1 - w_ypc) * base_ypc + w_ypc * float(np.clip(pri_ypc, 3.4, 6.5))
            jitter_rush = _stable_jitter(pl_name, 'rush', scale=0.03 * _jitter_scale())
            # Calibrate RB rushing efficiency using prior-week bias
            if pos_bias and int(week) > 1 and pos_up in ('RB','QB') and pos_up in pos_bias:
                r_yds_scale = float(np.clip(1.0 - beta_yards * pos_bias[pos_up].get('r_yds_frac', 0.0), 0.85, 1.15))
                base_ypc *= r_yds_scale
            pl_rush_yards = pl_rush_att * base_ypc * jitter_rush * eff
            # TDs: rushing allocated by rz_rush_share; receiving allocated by pcol (targets)
            # Bias red-zone shares using player priors when available
            rz_t_bias = 1.0; rz_r_bias = 1.0
            if eff_priors is not None and not eff_priors.empty:
                try:
                    match = pd.DataFrame()
                    if roster_map is not None and 'player_id' in roster_map.columns:
                        pid = roster_map.loc[roster_map['player'] == pl_name, 'player_id']
                        if not pid.empty:
                            match = eff_priors[eff_priors['_pid'] == str(pid.iloc[0])]
                    if match is None or match.empty:
                        nm = str(pl_name or '').lower()
                        match = eff_priors[eff_priors['_nm'] == nm]
                    if not match.empty:
                        # Receiving RZ target rate bias involves targets sample
                        if pos_up in {'WR','TE','RB'} and 'rz_target_rate' in match.columns and 'targets' in match.columns:
                            rt = float(match['rz_target_rate'].iloc[0])
                            tg = float(match['targets'].iloc[0])
                            w = _blend_weight(tg, k=50.0)
                            # Compare to baseline group RZ share expectations; use soft bias 0.8..1.2
                            base_rz_rate = 0.20 if pos_up == 'WR' else (0.25 if pos_up == 'TE' else 0.12)
                            rz_t_bias = float(np.clip(1.0 + w * (rt - base_rz_rate), 0.8, 1.2))
                        # Rushing RZ rate bias with rush attempts sample
                        if pos_up in {'RB','QB'} and 'rz_rush_rate' in match.columns and 'rush_att' in match.columns:
                            rr_ = float(match['rz_rush_rate'].iloc[0])
                            ra = float(match['rush_att'].iloc[0])
                            w2 = _blend_weight(ra, k=40.0)
                            base_rr = 0.12 if pos_up == 'RB' else 0.08
                            rz_r_bias = float(np.clip(1.0 + w2 * (rr_ - base_rr), 0.8, 1.25))
                except Exception:
                    pass
            # Apply biases multiplicatively to within-team shares (will still sum to ~1 across team if biases are modest)
            # Use effective RZ rush share for TDs; receiving uses t_eff share
            r_share_rz = float(prw.get('rrz_eff') if prw.get('rrz_eff') is not None else prw.get('rz_rush_share') or 0.0)
            pl_exp_rush_td = rush_tds * r_share_rz * rz_r_bias
            pl_exp_rec_td = 0.0 if pos.upper() == "QB" else pass_tds * t_share * rz_t_bias
            pl_any_td = pl_exp_rush_td + pl_exp_rec_td
            pl_any_prob = float(1.0 - np.exp(-max(0.0, pl_any_td)))

            row = {
                "season": int(tr.get("season")),
                "week": int(tr.get("week")),
                "date": tr.get("date"),
                "game_id": tr.get("game_id"),
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "player": pl_name,
                "position": pos,
                "player_id": pl_pid,
                # Team context
                "team_plays": plays,
                "team_pass_attempts": attempts,
                "team_rush_attempts": rush_att,
                "team_pass_yards": team_pass_yards,
                "team_rush_yards": team_rush_yards,
                "team_exp_pass_tds": pass_tds,
                "team_exp_rush_tds": rush_tds,
                "team_exp_int": team_int,
                # Player props
                "rush_attempts": pl_rush_att,
                "rush_yards": pl_rush_yards,
                "targets": pl_targets,
                "receptions": pl_rec,
                "rec_yards": pl_rec_yards,
                "rec_tds": pl_exp_rec_td,
                "rush_tds": pl_exp_rush_td,
                "any_td_prob": pl_any_prob,
            }
            # QB passing line (assign to the first/primary QB by target/rush shares)
            if pos.upper() == "QB" and qb_name and (str(pl_name) == qb_name):
                # Simple: all team passing assigned to QB1 (highest rush_share acts as QB1 proxy)
                patt = attempts
                py = team_pass_yards
                ptd = pass_tds
                pint = team_int
                # Calibrate QB passing toward prior-week QB bias
                if pos_bias and int(week) > 1 and 'QB' in pos_bias:
                    qb = pos_bias['QB']
                    patt *= float(np.clip(1.0 - qb_pass * qb.get('patt_frac', 0.0), 0.85, 1.15))
                    py *= float(np.clip(1.0 - qb_pass * qb.get('pyds_frac', 0.0), 0.85, 1.15))
                    ptd *= float(np.clip(1.0 - qb_pass * qb.get('ptd_frac', 0.0), 0.85, 1.15))
                    pint *= float(np.clip(1.0 - qb_pass * qb.get('int_frac', 0.0), 0.85, 1.15))
                row.update({
                    "pass_attempts": patt,
                    "pass_yards": py,
                    "pass_tds": ptd,
                    "interceptions": pint,
                })
            rows.append(row)

    # (Defense projections omitted to keep focus on offensive props)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Attach is_active from weekly rosters
    try:
        act = _active_roster(int(season), int(week))
        if act is not None and not act.empty:
            out['_nm'] = out['player'].astype(str).map(normalize_name_loose)
            out['player_id'] = out['player_id'].astype(str)
            # Prefer id join, fallback to name+team
            out = out.merge(act.rename(columns={'_pid':'player_id'})[['team','player_id','is_active','status']], on=['team','player_id'], how='left')
            # Fill where id missing via name key
            if 'is_active' not in out.columns or out['is_active'].isna().any():
                act2 = act.copy()
                out = out.merge(act2.rename(columns={'_nm':'_nm_act'})[['team','_nm_act','is_active','status']].rename(columns={'is_active':'is_active_nm','status':'status_nm'}), left_on=['team','_nm'], right_on=['team','_nm_act'], how='left')
                out['is_active'] = out['is_active'].fillna(out['is_active_nm'])
                out['status'] = out['status'].fillna(out['status_nm'])
                out = out.drop(columns=['_nm_act','is_active_nm','status_nm'])
            out['is_active'] = pd.to_numeric(out['is_active'], errors='coerce').fillna(1).astype(int)
            # Persist a quick report of predicted but not active
            try:
                miss = out[(out['is_active'] == 0)][['team','player','position','player_id','status']].copy()
                if not miss.empty:
                    fp = PRED_NOT_ACTIVE_TMPL.with_name(PRED_NOT_ACTIVE_TMPL.name.format(season=int(season), week=int(week)))
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    miss.to_csv(fp, index=False)
            except Exception:
                pass
            out = out.drop(columns=['_nm'])
    except Exception:
        pass
    # Round some quantities for presentation
    for c in [
        "team_plays","team_pass_attempts","team_rush_attempts","team_pass_yards","team_rush_yards",
        "team_exp_pass_tds","team_exp_rush_tds","team_exp_int","rush_attempts","rush_yards","targets",
        "receptions","rec_yards","rec_tds","rush_tds","pass_attempts","pass_yards","pass_tds","interceptions",
        "tackles","sacks",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Sensible rounding
    def _rnd(col: str, nd: int):
        if col in out.columns:
            out[col] = out[col].round(nd)
    for col in ["team_pass_yards","team_rush_yards","rush_yards","rec_yards","pass_yards"]:
        _rnd(col, 1)
    for col in ["team_plays","team_pass_attempts","team_rush_attempts","rush_attempts","targets","receptions","pass_attempts","interceptions","pass_tds","rush_tds","rec_tds","tackles","sacks"]:
        _rnd(col, 2)
    if "any_td_prob" in out.columns:
        out["any_td_prob"] = pd.to_numeric(out["any_td_prob"], errors="coerce").clip(lower=0.0, upper=1.0).round(4)
    # Order columns
    pref = [
        "season","week","date","game_id","team","opponent","is_home","player","position",
    "player_id","is_active","status",
        "pass_attempts","pass_yards","pass_tds","interceptions",
        "rush_attempts","rush_yards","rush_tds",
        "targets","receptions","rec_yards","rec_tds",
        "tackles","sacks",
        "any_td_prob",
    ]
    cols = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref]
    out = out[cols]
    # Sort for readability
    out = out.sort_values(["season","week","game_id","team","position","any_td_prob"], ascending=[True, True, True, True, True, False])
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compute weekly player prop projections.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    df = compute_player_props(args.season, args.week)
    if df is None or df.empty:
        print("No player props computed.")
        return
    out_fp = Path(args.out) if args.out else (DATA_DIR / f"player_props_{args.season}_wk{args.week}.csv")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f"Wrote {len(df)} rows to {out_fp}")


if __name__ == "__main__":
    main()
