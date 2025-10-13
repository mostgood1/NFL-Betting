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
import re
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
from .name_normalizer import normalize_name_loose, normalize_alias_init_last

# Data dir shared with rest of project
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEPTH_OVERRIDES_FP = DATA_DIR / "depth_overrides.csv"  # columns: season, week(optional), team, player, position, rush_share, target_share, rz_rush_share, rz_target_share
EFF_PRIORS_FP = DATA_DIR / "player_efficiency_priors.csv"
QB_STARTERS_FP_TMPL = DATA_DIR / "qb_starters_{season}_wk1.csv"
PRED_NOT_ACTIVE_TMPL = DATA_DIR / "predicted_not_active_{season}_wk{week}.csv"
USAGE_WK1_PBP_TMPL = DATA_DIR / "week1_usage_pbp_{season}.csv"
USAGE_WK1_CENTRAL_TMPL = DATA_DIR / "week1_central_stats_{season}.csv"

# Optional hard-coded safety net for Week 1 starters by team (normalized names)
WEEK1_QB_OVERRIDES = {
    "Minnesota Vikings": "JJ McCarthy",
    "Chicago Bears": "Caleb Williams",
}

# Optional explicit TE1 overrides by team (normalized team names)
# Used to force a specific TE to the top receiving rank when data sources conflict.
# Matching is done via alias (initial.last) and loose last-name fallback.
TE1_OVERRIDES = {
    "Minnesota Vikings": ["T.J. Hockenson", "Thomas Hockenson"],
}

# Optional explicit WR1 overrides by team (normalized team names)
# Used to force a specific WR to be the top receiving option.
WR1_OVERRIDES = {
    "Minnesota Vikings": ["Justin Jefferson"],
}

# Optional hard override for primary QB by team (applies to all weeks)
# Use sparingly when external feeds have alias collisions or misordered depth.
QB_OVERRIDE_BY_TEAM = {
    "Chicago Bears": "Caleb Williams",
}

# Optional explicit RB1 overrides by team (normalized team names)
# Used to force a specific RB to be the top rushing option when sources conflict.
RB1_OVERRIDES = {
    "Kansas City Chiefs": ["Isiah Pacheco"],
}


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

def _cfg_float(env_key: str, default: float, lo: float, hi: float) -> float:
    """Generic float config reader with clipping."""
    try:
        v = float(os.environ.get(env_key, str(default)))
        return float(np.clip(v, lo, hi))
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


def _qb_passing_priors(season: int, seasons_back: int = 2, min_games: int = 6) -> pd.DataFrame:
    """Build QB passing priors from previous seasons' play-by-play.

    Returns columns: player_id, player, attempts_pg, ypa, td_rate, games, _nm
    Degrades gracefully to empty DataFrame when data is unavailable.
    """
    cols = ["player_id","player","attempts_pg","ypa","td_rate","games","_nm"]
    try:
        seasons: list[int] = []
        for d in range(seasons_back, 0, -1):
            s = int(season) - d
            if s >= 1999:
                seasons.append(s)
        if not seasons:
            return pd.DataFrame(columns=cols)

        cache_fp = DATA_DIR / f"qb_passing_priors_{int(season)}_back{int(seasons_back)}.csv"
        if cache_fp.exists():
            try:
                cached = pd.read_csv(cache_fp)
                # Ensure expected columns
                for c in cols:
                    if c not in cached.columns:
                        cached[c] = None
                return cached[cols]
            except Exception:
                pass

        frames: List[pd.DataFrame] = []
        for s in seasons:
            df = None
            pbp_fp = DATA_DIR / f"pbp_{int(s)}.parquet"
            if pbp_fp.exists():
                try:
                    df = pd.read_parquet(pbp_fp)
                except Exception:
                    df = None
            if df is None:
                try:
                    import nfl_data_py as nfl  # type: ignore
                    df = nfl.import_pbp_data([int(s)])
                except Exception:
                    df = None
            if df is None or df.empty:
                continue

            d = df.copy()
            # Identify columns
            gid_col = None
            for c in ("game_id","gameId","gameid"):
                if c in d.columns:
                    gid_col = c; break
            if gid_col is None:
                d["game_id"] = pd.RangeIndex(start=0, stop=len(d)).astype(str)
                gid_col = "game_id"

            name_cols = ["passer_player_name","passer","passer_name","qb_player_name","qb_name"]
            id_cols = ["passer_player_id","passer_id","qb_player_id","qb_id"]
            def _pick(row: pd.Series, keys: List[str]) -> str:
                for k in keys:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        s = str(v).strip()
                        if s:
                            return s
                return ""
            d["_passer"] = d.apply(lambda r: _pick(r, name_cols), axis=1)
            d["_pid"] = d.apply(lambda r: _pick(r, id_cols), axis=1)

            # Attempt flag
            if "pass_attempt" in d.columns:
                att = pd.to_numeric(d["pass_attempt"], errors="coerce").fillna(0).astype(int)
            else:
                base = pd.to_numeric(d.get("pass"), errors="coerce").fillna(0).astype(int)
                scramble = pd.to_numeric(d.get("qb_scramble"), errors="coerce").fillna(0).astype(int)
                sack = pd.to_numeric(d.get("sack"), errors="coerce").fillna(0).astype(int)
                att = (base & (1 - scramble) & (1 - sack)).astype(int)

            # Yards and TDs
            yds = pd.to_numeric(d.get("passing_yards", d.get("yards_gained")), errors="coerce").fillna(0.0)
            if "pass_touchdown" in d.columns:
                td = pd.to_numeric(d["pass_touchdown"], errors="coerce").fillna(0).astype(int)
            else:
                td = (pd.to_numeric(d.get("touchdown"), errors="coerce").fillna(0).astype(int) & att).astype(int)

            sub = pd.DataFrame({
                "game_id": d[gid_col],
                "passer": d["_passer"],
                "passer_id": d["_pid"],
                "att": att,
                "yds": yds.where(att == 1, 0.0),
                "ptd": td,
            })
            sub = sub[sub["passer"].astype(str).str.len() > 0]

            g = (
                sub.groupby(["game_id","passer_id","passer"], as_index=False)
                   .agg(att=("att","sum"), yds=("yds","sum"), ptd=("ptd","sum"))
            )
            if not g.empty:
                g["season"] = int(s)
                frames.append(g)

        if not frames:
            return pd.DataFrame(columns=cols)

        allg = pd.concat(frames, ignore_index=True)
        allg = allg[pd.to_numeric(allg["att"], errors="coerce").fillna(0) > 0]
        if allg.empty:
            return pd.DataFrame(columns=cols)

        agg = (
            allg.groupby(["passer_id","passer"], as_index=False)
                .agg(games=("game_id","nunique"), att=("att","sum"), yds=("yds","sum"), ptd=("ptd","sum"))
        )
        agg["attempts_pg"] = np.where(agg["games"].gt(0), agg["att"] / agg["games"], np.nan)
        agg["ypa"] = np.where(agg["att"].gt(0), agg["yds"] / agg["att"], np.nan)
        agg["td_rate"] = np.where(agg["att"].gt(0), agg["ptd"] / agg["att"], np.nan)
        agg = agg[pd.to_numeric(agg["games"], errors="coerce").fillna(0) >= int(min_games)]
        agg = agg.rename(columns={"passer_id":"player_id","passer":"player"})
        agg["_nm"] = agg["player"].astype(str).map(normalize_name_loose)

        try:
            cache_fp.parent.mkdir(parents=True, exist_ok=True)
            agg.to_csv(cache_fp, index=False)
        except Exception:
            pass

        return agg[["player_id","player","attempts_pg","ypa","td_rate","games","_nm"]]
    except Exception:
        return pd.DataFrame(columns=cols)


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

        # Final override: force QB by team when specified (handles alias collisions like "C. Williams")
        try:
            forced = QB_OVERRIDE_BY_TEAM.get(team)
            if forced:
                qb_name = str(forced)
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
    # Heuristic quality check: if usage names barely match team roster, prefer roster-based depth.
    # This guards against placeholder first-name-only priors corrupting team rosters (e.g., ATL).
    try:
        rm = _team_roster_ids(int(season), team)
        if rm is not None and not rm.empty and ('player' in u.columns):
            # Build alias keys for robust matching
            try:
                rm_alias = rm.copy()
                rm_alias['player_alias'] = rm_alias['player'].astype(str).map(normalize_alias_init_last)
                u_alias = u.copy()
                u_alias['player_alias'] = u_alias['player'].astype(str).map(normalize_alias_init_last)
            except Exception:
                rm_alias = rm.copy()
                rm_alias['player_alias'] = rm_alias['player'].astype(str).map(normalize_name_loose)
                u_alias = u.copy()
                u_alias['player_alias'] = u_alias['player'].astype(str).map(normalize_name_loose)

            matched = u_alias.merge(rm_alias[['player_alias']].drop_duplicates(), on='player_alias', how='inner')
            match_frac = 0.0 if len(u_alias) == 0 else (len(matched) / float(len(u_alias)))
            # Additional signal: many one-token names tend to be placeholders
            names = u_alias['player'].astype(str).fillna('')
            one_token_frac = 0.0 if len(names) == 0 else float((names.str.split().map(len) <= 1).sum()) / float(len(names))
            if (match_frac < 0.5) or (one_token_frac > 0.6 and match_frac < 0.8):
                rb = _roster_based_depth(season, team)
                if rb is not None and not rb.empty:
                    return rb
    except Exception:
        # On any failure, keep using provided usage priors (will get corrected later by observed data injection)
        pass
    return u


@lru_cache(maxsize=4)
def _season_rosters(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
        ros = nfl.import_seasonal_rosters([int(season)])
        return ros if ros is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=2)
def _league_roster_map(season: int) -> pd.DataFrame:
    """Return league-wide roster mapping: team, player_id, player, position, depth_chart_order.
    Useful as a fallback when team-scoped roster lookups are incomplete.
    """
    ros = _season_rosters(int(season))
    if ros is None or ros.empty:
        return pd.DataFrame(columns=['team','player_id','player','position','depth_chart_order'])
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if not team_src:
        return pd.DataFrame(columns=['team','player_id','player','position','depth_chart_order'])
    df = ros.copy()
    df['team'] = df[team_src].astype(str).apply(normalize_team_name)
    # Build name column
    name_col = None
    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
        if c in df.columns:
            name_col = c; break
    if name_col:
        df['player'] = df[name_col].astype(str)
    else:
        fn = df.get('first_name', pd.Series(['']*len(df))).astype(str)
        ln = df.get('last_name', pd.Series(['']*len(df))).astype(str)
        df['player'] = (fn + ' ' + ln).str.strip()
    # IDs and positions
    id_col = None
    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
        if c in df.columns:
            id_col = c; break
    pos_col = 'depth_chart_position' if 'depth_chart_position' in df.columns else ('position' if 'position' in df.columns else None)
    if pos_col:
        df['position'] = df[pos_col].astype(str).str.upper()
    else:
        df['position'] = None
    dco = df['depth_chart_order'] if 'depth_chart_order' in df.columns else pd.Series([None]*len(df))
    out = pd.DataFrame({
        'team': df['team'],
        'player_id': df[id_col].astype(str) if id_col else pd.Series([None]*len(df), dtype=object),
        'player': df['player'],
        'position': df['position'],
        'depth_chart_order': pd.to_numeric(dco, errors='coerce'),
    })
    return out.dropna(subset=['team','player'])


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
        """Heuristic mapping from status text to active flag.
        Treat common inactive-like tokens as inactive (0), otherwise default to active (1).
        """
        if txt is None or (isinstance(txt, float) and np.isnan(txt)):
            return 1  # assume active if unknown
        t = str(txt).strip().upper()
        # Common inactive indicators from various feeds (abbreviations included)
        inactive_tokens = [
            'RESERVE', 'RES', 'IR', 'INACTIVE', 'INA', 'OUT', 'PUP', 'NFI', 'SUSP',
            'EXEMPT', 'EXE', 'PRACTICE', 'PRAC', 'PS', 'PRACTICE SQUAD', 'WAIVED', 'WAIV',
            'RELEASED', 'REL', 'CUT', 'COVID', 'DEV'
        ]
        if any(tok in t for tok in inactive_tokens):
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
    # Final safeguard: drop any nameless or placeholder rows before attaching actives/rounding
    try:
        # Identify rows with blank/null player names or placeholder strings
        pl = out.get('player')
        if pl is not None:
            s = pl.astype(str)
            mask_noname = s.str.strip().eq('') | s.str.lower().isin({'nan', 'none'})
            if mask_noname.any():
                # Optional debug snapshot of dropped rows
                try:
                    if str(os.environ.get('PROPS_DEBUG_SNAPSHOTS', '0')).strip().lower() in {'1','true','yes'}:
                        try:
                            season_i = int(pd.to_numeric(out.get('season'), errors='coerce').dropna().iloc[0]) if 'season' in out.columns else 0
                        except Exception:
                            season_i = 0
                        try:
                            week_i = int(pd.to_numeric(out.get('week'), errors='coerce').dropna().iloc[0]) if 'week' in out.columns else 0
                        except Exception:
                            week_i = 0
                        name = f"debug_noname_dropped_{season_i}_wk{week_i}.csv" if season_i and week_i else "debug_noname_dropped_latest.csv"
                        fp = DATA_DIR / name
                        out.loc[mask_noname].to_csv(fp, index=False)
                except Exception:
                    pass
                out = out[~mask_noname].copy()
        # Also ensure position is present
        pos = out.get('position')
        if pos is not None:
            s2 = pos.astype(str)
            mask_bad_pos = s2.str.strip().eq('') | s2.str.lower().isin({'nan', 'none'})
            if mask_bad_pos.any():
                out = out[~mask_bad_pos].copy()
    except Exception:
        pass
    # Renormalize by column
    for c in ['rush_share','target_share','rz_rush_share','rz_target_share']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        s = out[c].sum()
        if s>0:
            out[c] = out[c] / s
    return out


# --- External depth chart integration (ESPN) ---
@lru_cache(maxsize=4)
def _load_weekly_depth_chart(season: int, week: int) -> pd.DataFrame:
    """Load a saved weekly depth chart CSV (built via depth_charts.py) if available.
    Returns columns at least: season, week, team, position, player, depth_rank, depth_size, status, active
    """
    try:
        from .depth_charts import load_depth_chart_csv  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        df = load_depth_chart_csv(int(season), int(week))
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    # Normalize team and position
    if 'team' in d.columns:
        d['team'] = d['team'].astype(str).apply(normalize_team_name)
    if 'position' in d.columns:
        d['position'] = d['position'].astype(str).str.upper()
    if 'player' in d.columns:
        d['player'] = d['player'].astype(str)
    # Ensure depth_rank numeric
    if 'depth_rank' in d.columns:
        d['depth_rank'] = pd.to_numeric(d['depth_rank'], errors='coerce').fillna(99).astype(int)
    return d


def _espn_depth_usage(season: int, week: int, team: str) -> pd.DataFrame:
    """Construct usage shares from external ESPN depth chart for a given team.
    Mirrors _roster_based_depth weights but uses weekly saved depth ordering.
    """
    dc = _load_weekly_depth_chart(int(season), int(week))
    if dc is None or dc.empty:
        return pd.DataFrame()
    d = dc[dc['team'] == team].copy()
    if d is None or d.empty or 'position' not in d.columns:
        return pd.DataFrame()
    # Prefer active players but keep inactives available for backfill if counts are short
    d_all = d.copy()
    d_all['__act__'] = d_all.get('active', pd.Series([1]*len(d_all))).astype(bool).astype(int)
    # Helper to pick top-N by depth_rank for a given position
    def top_pos(pos: str, take: int) -> pd.DataFrame:
        sub = d_all[d_all['position'].astype(str).str.upper() == pos].copy()
        if sub.empty:
            return pd.DataFrame(columns=d_all.columns)
        # Sort by active desc, rank asc, player
        order_cols = ['__act__'] + ([
            c for c in ['depth_rank'] if c in sub.columns
        ]) + ['player']
        sub = sub.sort_values(order_cols, ascending=[False, True, True]).head(take).copy()
        return sub
    qb = top_pos('QB', 1)
    rb = top_pos('RB', 2)
    # Some teams list FB/HB; include them as RB if present and still short
    if len(rb) < 2:
        alt = d_all[d_all['position'].isin(['HB','FB'])].copy()
        if not alt.empty:
            alt = alt.sort_values(['__act__','depth_rank','player'], ascending=[False, True, True]).head(2-len(rb))
            if not alt.empty:
                alt = alt.assign(position='RB')
                rb = pd.concat([rb, alt], ignore_index=True)
    wr = top_pos('WR', 3)
    te = top_pos('TE', 2)

    # Backfill from roster if ESPN depth provides fewer than required entries
    try:
        rm = _team_roster_ids(int(season), team)
    except Exception:
        rm = pd.DataFrame()

    # Validate ESPN groups against roster positions to prevent misclassified defenders from receiving offensive shares
    def _allowed_for(pos: str) -> set[str]:
        p = pos.upper()
        if p == 'QB':
            return {'QB'}
        if p == 'RB':
            return {'RB','HB','FB'}
        if p == 'WR':
            return {'WR'}
        if p == 'TE':
            return {'TE'}
        return set()

    def _validate_group(group_df: pd.DataFrame, expected_pos: str) -> pd.DataFrame:
        if group_df is None or group_df.empty:
            return group_df
        if rm is None or rm.empty or 'player' not in rm.columns or 'position' not in rm.columns:
            return group_df
        try:
            df = group_df.copy()
            df['_nm'] = df['player'].astype(str).map(normalize_name_loose)
            rmm = rm.copy()
            rmm['_nm'] = rmm['player'].astype(str).map(normalize_name_loose)
            rmm['pos_up'] = rmm['position'].astype(str).str.upper()
            df = df.merge(rmm[['_nm','pos_up']].drop_duplicates(), on='_nm', how='left')
            allowed = _allowed_for(expected_pos)
            # Keep rows where roster position is missing (unknown) or in allowed set; drop clear mismatches
            keep = df['pos_up'].isna() | df['pos_up'].isin(allowed)
            df = df[keep].copy()
            return df.drop(columns=['_nm','pos_up'], errors='ignore')
        except Exception:
            return group_df

    qb = _validate_group(qb, 'QB')
    rb = _validate_group(rb, 'RB')
    wr = _validate_group(wr, 'WR')
    te = _validate_group(te, 'TE')

    # If any critical group is empty or still clearly invalid, fallback to roster-based depth for safety
    def _group_ok(gdf: pd.DataFrame, pos: str) -> bool:
        if gdf is None or gdf.empty:
            return False
        if rm is None or rm.empty:
            # Without roster, trust ESPN group non-empty
            return True
        try:
            gg = gdf.copy()
            gg['_nm'] = gg['player'].astype(str).map(normalize_name_loose)
            rmm = rm.copy(); rmm['_nm'] = rmm['player'].astype(str).map(normalize_name_loose)
            rmm['pos_up'] = rmm['position'].astype(str).str.upper()
            gg = gg.merge(rmm[['_nm','pos_up']].drop_duplicates(), on='_nm', how='left')
            allowed = _allowed_for(pos)
            # ok if at least one row has allowed position or unknown (NaN) that we'll backfill later
            has_allowed = gg['pos_up'].isna().any() or gg['pos_up'].isin(allowed).any()
            return bool(has_allowed)
        except Exception:
            return len(gdf) > 0

    if (not _group_ok(rb, 'RB')) or (not _group_ok(wr, 'WR')) or (not _group_ok(te, 'TE')):
        try:
            return _roster_based_depth(int(season), team)
        except Exception:
            # If fallback fails, continue with current and let backfill/weights handle
            pass
    def backfill(group_df: pd.DataFrame, pos: str, need: int) -> pd.DataFrame:
        cur = group_df.copy()
        if len(cur) >= need:
            return cur
        if rm is None or rm.empty or 'position' not in rm.columns:
            return cur
        have = set(cur['player'].astype(str)) if not cur.empty else set()
        cand = rm[rm['position'].astype(str).str.upper() == pos].copy()
        if cand.empty:
            return cur
        # Prefer ordered roster entries not already selected
        cand = cand[~cand['player'].astype(str).isin(have)].copy()
        ord_col = 'depth_chart_order' if 'depth_chart_order' in cand.columns else None
        if ord_col:
            cand = cand.sort_values([ord_col,'player'])
        else:
            cand = cand.sort_values(['player'])
        take = max(0, need - len(cur))
        if take > 0:
            add = cand.head(take).copy()
            if not add.empty:
                # Align columns similar to ESPN depth rows
                add = add.assign(position=pos)
                add['__act__'] = 1
                cur = pd.concat([cur, add[['player','position']].merge(cur.head(0), how='left')], ignore_index=True) if False else pd.concat([cur, add[['player','position']]], ignore_index=True)
        return cur
    qb = backfill(qb, 'QB', 1)
    rb = backfill(rb, 'RB', 2)
    wr = backfill(wr, 'WR', 3)
    te = backfill(te, 'TE', 2)
    # Weight templates (same as _roster_based_depth)
    def weights(base, n):
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
    def push(df: pd.DataFrame, pos: str, w, rz):
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
    # Renormalize shares per column to 1.0 to keep consistency
    for c in ['rush_share','target_share','rz_rush_share','rz_target_share']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        s = float(out[c].sum())
        if s > 0:
            out[c] = out[c] / s
    return out


@lru_cache(maxsize=32)
def _espn_team_active_map(season: int, week: int, team: str) -> pd.DataFrame:
    """Return a small map of player -> is_active flag from the ESPN depth chart for a team-week.
    Columns: player, _nm, is_active_espn
    """
    try:
        dc = _load_weekly_depth_chart(int(season), int(week))
    except Exception:
        return pd.DataFrame(columns=['player','_nm','is_active_espn'])
    if dc is None or dc.empty:
        return pd.DataFrame(columns=['player','_nm','is_active_espn'])
    d = dc[dc['team'].astype(str).apply(normalize_team_name) == team].copy()
    if d.empty or 'player' not in d.columns:
        return pd.DataFrame(columns=['player','_nm','is_active_espn'])
    d['player'] = d['player'].astype(str)
    try:
        d['_nm'] = d['player'].map(normalize_name_loose)
    except Exception:
        d['_nm'] = d['player'].astype(str).str.lower()
    if 'active' in d.columns:
        d['is_active_espn'] = d['active'].astype(bool).astype(int)
    else:
        d['is_active_espn'] = 1
    return d[['player','_nm','is_active_espn']].drop_duplicates()


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
                # If cached names are abbreviated (e.g., "K.Murray"), try to canonicalize to roster display names
                needs_fix = False
                try:
                    if 'player' in df.columns:
                        s = df['player'].astype(str)
                        # Heuristic: contains a dot or single-letter first token
                        needs_fix = bool((s.str.contains('\.').sum() > 0) or (s.str.match(r'^[A-Za-z]\.').sum() > 0))
                except Exception:
                    needs_fix = False
                if needs_fix:
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
                        if team_src and name_map_col:
                            # Build a name/id map scoped to team, and when possible prefer QB rows to avoid alias collisions (e.g., "C. Williams")
                            keep_cols = [name_map_col, team_src]
                            if id_map_col:
                                keep_cols.append(id_map_col)
                            # Include position columns if present to filter to QBs
                            pos_cols = []
                            for cpos in ['depth_chart_position','position']:
                                if cpos in ros.columns:
                                    pos_cols.append(cpos)
                            keep_cols += pos_cols
                            m_full = ros[keep_cols].copy()
                            m_full['team'] = m_full[team_src].astype(str).apply(normalize_team_name)
                            # Prefer rows where roster indicates QB
                            m_qb = m_full.copy()
                            if pos_cols:
                                qb_mask = None
                                for cpos in pos_cols:
                                    col = m_qb[cpos].astype(str).str.upper()
                                    qb_mask = col.eq('QB') if qb_mask is None else (qb_mask | col.eq('QB'))
                                m_qb = m_qb[qb_mask.fillna(False)].copy()
                            # Choose QB-filtered map when available; else fall back to all positions
                            m = m_qb if (m_qb is not None and not m_qb.empty) else m_full
                            m['_nm'] = m[name_map_col].astype(str).map(normalize_name_loose)
                            m['_alias'] = m[name_map_col].astype(str).map(normalize_alias_init_last)
                            s = df.copy()
                            s['_nm'] = s['player'].astype(str).map(normalize_name_loose)
                            s['_alias'] = s['player'].astype(str).map(normalize_alias_init_last)
                            # Try alias match first (handles K.Murray -> kmurray), then loose
                            s = s.merge(m[['team','_alias', name_map_col] + ([id_map_col] if id_map_col else [])].drop_duplicates(['team','_alias']).rename(columns={name_map_col:'player_full', id_map_col:'player_id'}), on=['team','_alias'], how='left')
                            missing = s['player_full'].isna()
                            if missing.any():
                                s = s.merge(m[['team','_nm', name_map_col] + ([id_map_col] if id_map_col else [])].drop_duplicates(['team','_nm']).rename(columns={name_map_col:'player_full', id_map_col:'player_id'}), on=['team','_nm'], how='left', suffixes=(None,'_nmfix'))
                                s['player_full'] = s['player_full'].fillna(s.get('player_full_nmfix'))
                                if 'player_full_nmfix' in s.columns:
                                    s = s.drop(columns=['player_full_nmfix'])
                            # Replace when we found a full name
                            if 'player_full' in s.columns:
                                s['player'] = s['player_full'].fillna(s['player'])
                                s = s.drop(columns=['player_full'])
                            # Clean helper cols
                            for c in ['_nm','_alias']:
                                if c in s.columns:
                                    s = s.drop(columns=[c])
                            df = s
                            # Persist fixed cache
                            try:
                                df.to_csv(fp, index=False)
                            except Exception:
                                pass
                # Additional safety: if cached player is not a QB in team roster, try to resolve by alias to a QB
                try:
                    ros = _season_rosters(int(season))
                except Exception:
                    ros = pd.DataFrame()
                if ros is not None and not ros.empty and 'player' in df.columns:
                    # Build team-scoped roster with alias and position
                    team_src = None
                    for c in ['team','recent_team','team_abbr']:
                        if c in ros.columns:
                            team_src = c; break
                    name_map_col = None
                    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
                        if c in ros.columns:
                            name_map_col = c; break
                    id_map_col = None
                    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
                        if c in ros.columns:
                            id_map_col = c; break
                    pos_col = 'depth_chart_position' if 'depth_chart_position' in ros.columns else ('position' if 'position' in ros.columns else None)
                    if team_src and name_map_col:
                        m = ros[[team_src, name_map_col] + ([id_map_col] if id_map_col else []) + ([pos_col] if pos_col else [])].copy()
                        m['team'] = m[team_src].astype(str).apply(normalize_team_name)
                        from .name_normalizer import normalize_alias_init_last
                        m['_alias'] = m[name_map_col].astype(str).map(normalize_alias_init_last)
                        # For each df row, if mapped position isn't QB, try to swap to team QB with same alias
                        s = df.copy()
                        s['team'] = s['team'].astype(str).apply(normalize_team_name)
                        s['_alias'] = s['player'].astype(str).map(normalize_alias_init_last)
                        # Attach roster position for current mapping
                        cur = s.merge(m[['team', '_alias', pos_col] if pos_col else ['team','_alias']], on=['team','_alias'], how='left')
                        if pos_col and (cur[pos_col].astype(str).str.upper() != 'QB').any():
                            not_qb = cur[pos_col].astype(str).str.upper() != 'QB'
                            to_fix = cur[not_qb].copy()
                            if not to_fix.empty:
                                # Find a QB in roster with same alias per team
                                m_qb = m.copy()
                                if pos_col:
                                    m_qb = m_qb[m_qb[pos_col].astype(str).str.upper() == 'QB']
                                rep = to_fix.merge(m_qb[['team','_alias', name_map_col] + ([id_map_col] if id_map_col else [])].rename(columns={name_map_col:'player_qb', id_map_col:'player_id_qb'}), on=['team','_alias'], how='left')
                                # Apply replacements where found
                                for idx, rr in rep.iterrows():
                                    new_nm = rr.get('player_qb')
                                    if pd.notna(new_nm) and str(new_nm).strip():
                                        s.loc[s.index == rr.name, 'player'] = new_nm
                                        if id_map_col and 'player_id_qb' in rep.columns and pd.notna(rr.get('player_id_qb')):
                                            s.loc[s.index == rr.name, 'player_id'] = rr.get('player_id_qb')
                                df = s.drop(columns=['_alias'], errors='ignore')
                                # Persist corrected cache
                                try:
                                    df.to_csv(fp, index=False)
                                except Exception:
                                    pass
                return df
        except Exception:
            pass
    # Preferred: derive from PBP week 1 using dropbacks (pass, sack, scramble)
    starters = pd.DataFrame(columns=['team','player','player_id'])
    df = pd.DataFrame()
    # Try local parquet first (faster, consistent with other funcs)
    try:
        pbp_fp = DATA_DIR / f"pbp_{int(season)}.parquet"
        if pbp_fp.exists():
            df = pd.read_parquet(pbp_fp)
    except Exception:
        df = pd.DataFrame()
    # Fallback to importing via nfl_data_py
    if df is None or df.empty:
        try:
            import nfl_data_py as nfl  # type: ignore
            df = nfl.import_pbp_data([int(season)])
        except Exception:
            df = pd.DataFrame()
    try:
        if df is not None and not df.empty and 'week' in df.columns:
            pb = df.copy()
            # Filter week 1 regular season
            if 'season_type' in pb.columns:
                pb = pb[pb['season_type'].astype(str).str.upper() == 'REG']
            elif 'game_type' in pb.columns:
                pb = pb[pb['game_type'].astype(str).str.upper() == 'REG']
            pb['week'] = pd.to_numeric(pb['week'], errors='coerce').fillna(0).astype(int)
            pb = pb[pb['week'] == 1].copy()
            if not pb.empty:
                # Dropback mask
                def _to_int(s):
                    return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
                qb_drop = _to_int(pb['qb_dropback']) if 'qb_dropback' in pb.columns else None
                pass_flag = _to_int(pb['pass']) if 'pass' in pb.columns else (_to_int(pb['pass_attempt']) if 'pass_attempt' in pb.columns else None)
                sack = _to_int(pb['sack']) if 'sack' in pb.columns else None
                scramble = _to_int(pb['qb_scramble']) if 'qb_scramble' in pb.columns else None
                mask = None
                for m in [qb_drop, pass_flag, sack, scramble]:
                    if m is not None:
                        mask = m if mask is None else (mask | (m == 1))
                if mask is None:
                    mask = pd.Series([False] * len(pb), index=pb.index)
                pb = pb[mask].copy()
                # Team and QB identity
                team_col = 'posteam' if 'posteam' in pb.columns else ( 'pos_team' if 'pos_team' in pb.columns else None )
                qb_name_col = None
                for c in ['qb_player_name','qb_name','passer_player_name','passer','passer_name']:
                    if c in pb.columns:
                        qb_name_col = c; break
                qb_id_col = None
                for c in ['qb_player_id','qb_id','passer_player_id','passer_id']:
                    if c in pb.columns:
                        qb_id_col = c; break
                if team_col and qb_name_col:
                    keep = [team_col, qb_name_col]
                    if qb_id_col:
                        keep.append(qb_id_col)
                    keep_extra = []
                    # tie-breaker columns if available
                    for c in ['game_id','play_id','qtr','quarter_seconds_remaining','time']:
                        if c in pb.columns:
                            keep_extra.append(c)
                    sub = pb[keep + keep_extra].copy()
                    sub = sub.rename(columns={team_col:'team_abbr', qb_name_col:'player', (qb_id_col or 'player'):'player_id'})
                    sub['team'] = sub['team_abbr'].astype(str).apply(normalize_team_name)
                    # Count dropbacks per QB per team
                    sub['_db'] = 1
                    grp_cols = ['team','player'] + (['player_id'] if 'player_id' in sub.columns else [])
                    g = sub.groupby(grp_cols, as_index=False)['_db'].sum()
                    # Pick max dropbacks per team; stable sort by dropbacks desc then name to be deterministic
                    g = g.sort_values(['team','_db','player'], ascending=[True, False, True])
                    # For ties, prefer the QB who appeared earlier in the game if we have play ordering
                    if {'team','player'}.issubset(set(sub.columns)) and any(c in sub.columns for c in ['play_id','qtr','quarter_seconds_remaining']):
                        # earliest play index proxy
                        order = sub.copy()
                        if 'play_id' in order.columns:
                            order['_ord'] = pd.to_numeric(order['play_id'], errors='coerce')
                        elif 'quarter_seconds_remaining' in order.columns:
                            # lower remaining seconds means later in quarter; invert to get early
                            order['_ord'] = -pd.to_numeric(order['quarter_seconds_remaining'], errors='coerce')
                        else:
                            order['_ord'] = 0
                        firsts = order.sort_values(['team','_ord']).groupby(['team','player'], as_index=False).first()[['team','player','_ord']]
                        g = g.merge(firsts, on=['team','player'], how='left')
                        g = g.sort_values(['team','_db','_ord'], ascending=[True, False, True])
                    top = g.groupby('team', as_index=False).first()
                    starters = top[['team','player']].copy()
                    if 'player_id' in top.columns:
                        starters['player_id'] = top['player_id']
                    # Attach canonical ids from roster if missing
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
                        if team_src and name_map_col and id_map_col:
                            m = ros[[name_map_col, id_map_col, team_src]].copy()
                            m['team'] = m[team_src].astype(str).apply(normalize_team_name)
                            m['_nm'] = m[name_map_col].astype(str).map(normalize_name_loose)
                            starters['_nm'] = starters['player'].astype(str).map(normalize_name_loose)
                            starters = starters.merge(
                                m[['team','_nm', name_map_col, id_map_col]].rename(columns={id_map_col:'player_id', name_map_col:'player_full'}),
                                on=['team','_nm'], how='left'
                            ).drop(columns=['_nm'])
                            # Prefer full roster name when available to avoid abbrev like "K.Murray"
                            if 'player_full' in starters.columns:
                                starters['player'] = starters['player_full'].fillna(starters['player'])
                                starters = starters.drop(columns=['player_full'])
    except Exception:
        pass
    # Week 1 hygiene: remove players flagged inactive (DEV/RES/INA/etc.) from final output to avoid nonsense entries
    try:
        wcol = 'week' if 'week' in out.columns else None
        cur_wk = int(out['week'].iloc[0]) if wcol else None
        if (cur_wk == 1) and 'is_active' in out.columns:
            out = out[out['is_active'].astype('Int64').fillna(1) == 1].copy()
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
                    # No additional filtering; rely on attempts ordering
                    starters = (
                        sub.sort_values(['team','pass_attempts'], ascending=[True, False])
                           .groupby('team', as_index=False)
                           .first()[['team','player','player_id']]
                    )
        except Exception:
            pass
    # Safety net: apply manual overrides if provided
    try:
        if WEEK1_QB_OVERRIDES is not None and len(WEEK1_QB_OVERRIDES) > 0:
            s = starters.copy() if (starters is not None) else pd.DataFrame(columns=['team','player','player_id'])
            if s.empty:
                s = pd.DataFrame(columns=['team','player','player_id'])
            if 'team' not in s.columns:
                s['team'] = []
            if 'player' not in s.columns:
                s['player'] = []
            s['team'] = s['team'].astype(str)
            for t, nm in WEEK1_QB_OVERRIDES.items():
                t_norm = normalize_team_name(t)
                if not s[s['team']==t_norm].empty:
                    s.loc[s['team']==t_norm, 'player'] = nm
                else:
                    s = pd.concat([s, pd.DataFrame([{'team': t_norm, 'player': nm}])], ignore_index=True)
            starters = s
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
    # Select the first finite pace value among possible columns
    spp = None
    for key in ("pace_prior", "home_pace_prior", "away_pace_prior"):
        if key in team_row.index:
            try:
                v = float(team_row.get(key))
                if np.isfinite(v) and v > 0:
                    spp = v
                    break
            except Exception:
                continue
    if spp is not None:
        plays = 3600.0 / spp
        return float(np.clip(plays, 55.0, 72.0))
    # Fallback: derive a small adjustment from offensive minus defensive EPA priors
    def _safe0(x) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0
    o = _safe0(team_row.get("off_epa_prior"))
    d = _safe0(team_row.get("opp_def_epa_prior"))
    diff = o - d
    adj = float(np.tanh(3.0 * diff))  # ~[-0.76, 0.76]
    return float(np.clip(LEAGUE_PLAYS_PER_TEAM * (1.0 + 0.05 * adj), 58.0, 70.0))


def _efficiency_scaler(team_row: pd.Series) -> float:
    # Convert (off_epa_prior - opp_def_epa_prior) to small multiplier
    def _safe0(x) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0
    o = _safe0(team_row.get("off_epa_prior"))
    d = _safe0(team_row.get("opp_def_epa_prior"))
    diff = o - d
    try:
        val = float(np.exp(2.0 * diff))
        if not np.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.85, 1.15))
    except Exception:
        return 1.0


def compute_player_props(season: int, week: int) -> pd.DataFrame:
    teams = compute_td_likelihood(season=season, week=week)
    if teams is None or teams.empty:
        return pd.DataFrame(columns=["season","week","team","opponent","player","position"])

    # Optional: write per-team depth debug snapshots when investigating issues
    def _dump_depth(team_name: str, stage: str, df: pd.DataFrame):
        try:
            if str(os.environ.get('PROPS_DEBUG_SNAPSHOTS','0')).strip() not in {'1','true','TRUE','yes','YES'}:
                return
            slug = re.sub(r"[^a-z0-9]+","_", normalize_team_name(team_name).lower()).strip("_") if team_name else "team"
            fp = DATA_DIR / f"debug_depth_{int(season)}_wk{int(week)}_{slug}_{stage}.csv"
            cols = [c for c in ['player','position','player_id','rush_share','target_share','rz_rush_share','rz_target_share','t_blend','r_blend','t_eff','r_eff','recv_rank','rush_rank','is_active'] if c in (df.columns if df is not None else [])]
            if df is not None and not df.empty and cols:
                try:
                    d = df[cols].copy()
                except Exception:
                    d = df.copy()
            else:
                d = df.copy() if df is not None else pd.DataFrame()
            fp.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(fp, index=False)
        except Exception:
            pass

    usage = _load_usage_priors()
    # Load efficiency priors once
    eff_priors = _load_efficiency_priors()
    # Preload EMA context and defensive tendencies through previous week
    ema_all = _team_context_ema(int(season), max(0, int(week) - 1))
    def_tend_all = _def_pos_tendencies(int(season), max(0, int(week) - 1))
    # Calibration: prior week reconciliation summary by position
    recon_sum = _load_recon_summary(int(season), int(week) - 1)
    # Env-driven calibration strengths (0..1) - tuned defaults
    alpha_vol = _calib_weight('PROPS_CALIB_ALPHA', 0.35)   # targets / rush attempts volume
    beta_yards = _calib_weight('PROPS_CALIB_BETA', 0.30)   # yards per volume (ypt/ypc proxy)
    gamma_rec = _calib_weight('PROPS_CALIB_GAMMA', 0.25)   # receptions via catch rate
    qb_pass = _calib_weight('PROPS_CALIB_QB', 0.40)        # QB pass attempts/yards/TD/INT scale

    # Position-level small multipliers (env-tunable) to address residual bias
    wr_rec_yards_mult = _cfg_float('PROPS_POS_WR_REC_YDS', 1.02, 0.80, 1.20)
    te_rec_yards_mult = _cfg_float('PROPS_POS_TE_REC_YDS', 0.98, 0.80, 1.20)
    te_rec_mult = _cfg_float('PROPS_POS_TE_REC', 0.98, 0.80, 1.20)
    qb_py_mult = _cfg_float('PROPS_POS_QB_PASS_YDS', 0.97, 0.70, 1.20)
    # Slightly higher default multiplier for QB pass TDs; can be tuned via env
    qb_ptd_mult = _cfg_float('PROPS_POS_QB_PASS_TDS', 1.02, 0.70, 1.20)
    # Allow an env-configurable upper clamp for elite QB TD rate priors (per-attempt)
    qb_td_rate_hi = _cfg_float('PROPS_QB_TD_RATE_HI', 0.075, 0.05, 0.10)
    # QB red-zone rushing tuning
    qb_rz_base = _cfg_float('PROPS_QB_RZ_BASE', 0.10, 0.05, 0.20)          # baseline RZ rush rate for QB priors biasing
    qb_rz_cap = _cfg_float('PROPS_QB_RZ_CAP', 1.30, 1.00, 1.60)             # max multiplicative bias for QB RZ rush bias
    qb_rz_share_scale = _cfg_float('PROPS_QB_RZ_SHARE_SCALE', 0.95, 0.70, 1.30)  # scale applied to QB non-RZ rush share to derive RZ rush share
    qb_rz_share_min = _cfg_float('PROPS_QB_RZ_SHARE_MIN', 0.005, 0.0, 0.10)
    qb_rz_share_max = _cfg_float('PROPS_QB_RZ_SHARE_MAX', 0.20, 0.05, 0.35)
    # New: QB rushing share clamp + blending knobs
    qb_share_min = _cfg_float('PROPS_QB_SHARE_MIN', 0.015, 0.0, 0.40)
    qb_share_max = _cfg_float('PROPS_QB_SHARE_MAX', 0.28, 0.05, 0.50)
    qb_share_default = _cfg_float('PROPS_QB_SHARE_DEFAULT', 0.07, 0.0, 0.30)
    qb_obs_blend = _cfg_float('PROPS_QB_OBS_BLEND', 0.60, 0.0, 1.0)
    # New: QB passing priors blend weight
    qb_prior_w = _cfg_float('PROPS_QB_PRIOR_WEIGHT', 0.35, 0.0, 0.8)
    # Load QB passing priors once (for current season context)
    qb_priors = _qb_passing_priors(int(season))
    # Load QB rush priors once for Week 1 rush share blending
    qb_rush_priors = _qb_rush_rate_priors(int(season))

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
        # Week 1 top WR/TE from central stats (for dynamic overrides)
        wk1_wr1_aliases: set = set()
        wk1_te1_aliases: set = set()
        wk1_wr1_lnames: set = set()
        wk1_te1_lnames: set = set()
        if int(tr.get('week')) == 1:
            try:
                top_pos = _week1_top_pos_from_central(int(tr.get('season')))
                if top_pos is not None and not top_pos.empty:
                    top_team = top_pos[top_pos['team'] == team]
                    if not top_team.empty:
                        wr_row = top_team[top_team['pos_up'] == 'WR']
                        te_row = top_team[top_team['pos_up'] == 'TE']
                        if not wr_row.empty:
                            for _, rr in wr_row.iterrows():
                                al = str(rr.get('player_alias') or '')
                                if al:
                                    wk1_wr1_aliases.add(al)
                                nm = str(rr.get('player') or '')
                                parts = nm.split()
                                if parts:
                                    wk1_wr1_lnames.add(parts[-1].lower())
                        if not te_row.empty:
                            for _, rr in te_row.iterrows():
                                al = str(rr.get('player_alias') or '')
                                if al:
                                    wk1_te1_aliases.add(al)
                                nm = str(rr.get('player') or '')
                                parts = nm.split()
                                if parts:
                                    wk1_te1_lnames.add(parts[-1].lower())
            except Exception:
                pass
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
        # Pass/rush rates for volume (robust to NaNs)
        pr_raw = tr.get("home_pass_rate_prior") if is_home else tr.get("away_pass_rate_prior")
        pr_val = _safe_float(pr_raw, float('nan'))
        def _safe0_num(x) -> float:
            try:
                v = float(x)
                return v if np.isfinite(v) else 0.0
            except Exception:
                return 0.0
        if not np.isfinite(pr_val):
            o = _safe0_num(tr.get("off_epa_prior"))
            d = _safe0_num(tr.get("opp_def_epa_prior"))
            # Map EPA diff to pass tendency around 0.55 ± 0.1
            val = 0.55 + 0.10 * float(np.tanh(4.0 * (o - d)))
            pr_val = val if np.isfinite(val) else LEAGUE_PASS_RATE
        # Blend pass rate with EMA if available
        base_pass_rate = float(np.clip(pr_val, 0.45, 0.70)) if np.isfinite(pr_val) else LEAGUE_PASS_RATE
        pass_rate = base_pass_rate
        if ema_all is not None and not ema_all.empty:
            em = ema_all[ema_all['team'] == team]
            if not em.empty and pd.notna(em.iloc[0].get('pass_rate_ema')):
                w_ema = _ema_blend_weight(0.5)
                try:
                    ema_pr = float(em.iloc[0]['pass_rate_ema'])
                    if not np.isfinite(ema_pr):
                        ema_pr = base_pass_rate
                    base_pr = base_pass_rate
                    mixed = (1.0 - w_ema) * base_pr + w_ema * ema_pr
                    pass_rate = float(np.clip(mixed, 0.45, 0.70))
                except Exception:
                    pass
        if not np.isfinite(pass_rate):
            pass_rate = LEAGUE_PASS_RATE
        rush_rate = 1.0 - pass_rate
        if not np.isfinite(rush_rate):
            rush_rate = max(0.0, 1.0 - LEAGUE_PASS_RATE)
        dropbacks = plays * pass_rate
        if not np.isfinite(dropbacks):
            dropbacks = plays * LEAGUE_PASS_RATE
        attempts = dropbacks * (1.0 - LEAGUE_SACK_RATE)
        if not np.isfinite(attempts) or attempts < 0.0:
            attempts = max(0.0, plays * LEAGUE_PASS_RATE * (1.0 - LEAGUE_SACK_RATE))
        rush_att = plays * rush_rate
        if not np.isfinite(rush_att) or rush_att < 0.0:
            rush_att = max(0.0, plays * (1.0 - LEAGUE_PASS_RATE))
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

        # Depth source: ESPN depth chart is primary for all weeks; fallback to priors/roster-based
        espn_depth = _espn_depth_usage(int(tr.get('season')), int(tr.get('week')), team)
        if espn_depth is not None and not espn_depth.empty:
            depth = espn_depth
        else:
            depth = _team_depth(usage, int(tr.get("season")), team)
        _dump_depth(team, '01_base', depth)
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
        # Prepare roster map early (used by blending, TE/RB ordering, and QB id)
        roster_map = _team_roster_ids(int(tr.get("season")), team)
        if (roster_map is None) or roster_map.empty:
            # Fallback: use league map filtered to team
            try:
                lm = _league_roster_map(int(tr.get('season')))
                if lm is not None and not lm.empty:
                    roster_map = lm[lm['team'] == team].copy()
            except Exception:
                pass
        # Normalize pass target shares to typical positional mix
        # Apply for both Week 1 and Week > 1; stronger alpha in Week 1 to avoid TE-heavy leaders
        if int(tr.get('week')) == 1:
            desired_group = {"WR": 0.75, "TE": 0.15, "RB": 0.10}
            desired_group_rz = {"WR": 0.60, "TE": 0.30, "RB": 0.10}
            alpha = 0.7
        else:
            desired_group = {"WR": 0.65, "TE": 0.25, "RB": 0.10}
            desired_group_rz = {"WR": 0.55, "TE": 0.35, "RB": 0.10}
            alpha = 0.5
        for pcol, dist in [("target_share", desired_group), ("rz_target_share", desired_group_rz)]:
            if pcol not in depth.columns:
                continue
            col_sum = float(pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0).sum())
            if col_sum <= 0:
                continue
            for pos in ("WR","TE","RB"):
                m = depth["position"].astype(str).str.upper() == pos
                cur = float(pd.to_numeric(depth.loc[m, pcol], errors='coerce').fillna(0.0).sum()) / col_sum
                tgt = dist.get(pos, 0.0)
                if cur > 0 and tgt > 0:
                    hard = tgt / cur
                    fac = (1.0 - alpha) + alpha * hard
                    depth.loc[m, pcol] = pd.to_numeric(depth.loc[m, pcol], errors='coerce').fillna(0.0) * fac
            s = float(pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0).sum())
            if s > 0:
                depth[pcol] = pd.to_numeric(depth[pcol], errors='coerce').fillna(0.0) / s
        # Compute per-position ranks to vary efficiency within groups
        depth['pos_up'] = depth['position'].astype(str).str.upper()
        # Use target_share (not RZ) to drive receiving volume strength
        depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0), 0.0)
        depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0), 0.0)
        try:
            depth['recv_rank'] = depth.groupby('pos_up')['recv_strength'].rank(ascending=False, method='first')
            depth['rush_rank'] = depth.groupby('pos_up')['rush_strength'].rank(ascending=False, method='first')
        except Exception:
            depth['recv_rank'] = 1
            depth['rush_rank'] = 1

        # TE ranking refinement: prefer pass-catching TE (higher prior targets) as TE1 when available
        try:
            te_mask = depth['pos_up'].eq('TE')
            if te_mask.any() and eff_priors is not None and not eff_priors.empty:
                dep = depth.copy()
                if '_nm' not in dep.columns:
                    dep['_nm'] = dep['player'].astype(str).map(normalize_name_loose)
                pri = eff_priors[['__dummy' if False else '_pid','_nm','targets']].copy()
                pri = pri.rename(columns={'_pid':'player_id','targets':'prior_targets'})
                # Use id join when present, else name join
                if 'player_id' in dep.columns and dep['player_id'].notna().any():
                    dep['player_id'] = dep['player_id'].astype(str)
                    dep = dep.merge(pri[['player_id','prior_targets']], on='player_id', how='left')
                else:
                    dep = dep.merge(pri, on='_nm', how='left')
                # Rank TEs by prior targets if any present
                pt = pd.to_numeric(dep.get('prior_targets'), errors='coerce').fillna(-1.0)
                if (pt[te_mask].max() > -1.0):
                    # Higher prior targets => better recv_rank (1 is best)
                    order = pt[te_mask].rank(ascending=False, method='first')
                    depth.loc[te_mask, 'recv_rank'] = order.values
        except Exception:
            pass

        # Week 1: Blend player efficiency priors into initial target/rush shares for all positions
        # This sets t_blend/r_blend that downstream steps use as base volume signals.
        if int(tr.get('week')) == 1:
            try:
                # Ensure id/name keys for joining priors
                if 'player_id' not in depth.columns:
                    depth['player_id'] = ''
                if '_nm' not in depth.columns:
                    depth['_nm'] = depth['player'].astype(str).map(normalize_name_loose)
                # Prepare priors slices
                pri = eff_priors.copy() if eff_priors is not None else pd.DataFrame()
                if pri is None:
                    pri = pd.DataFrame()
                # Columns we care about
                tg_col = 'targets' if (not pri.empty and 'targets' in pri.columns) else None
                ru_col = 'rush_att' if (not pri.empty and 'rush_att' in pri.columns) else None
                # Attach player-level priors (prefer id join)
                dep = depth.copy()
                if not pri.empty:
                    if '_pid' not in pri.columns and 'player_id' in pri.columns:
                        pri['_pid'] = pri['player_id'].astype(str)
                    if 'player_id' in dep.columns and dep['player_id'].astype(str).notna().any():
                        dep['player_id'] = dep['player_id'].astype(str)
                        dep = dep.merge(pri[[c for c in ['_pid','_nm', tg_col, ru_col] if c in pri.columns]].rename(columns={'_pid':'player_id'}),
                                        on='player_id', how='left', suffixes=(None, '_pri'))
                    # Fallback: name join
                    if tg_col and f'{tg_col}_pri' not in dep.columns:
                        dep = dep.merge(pri[[c for c in ['_nm', tg_col, ru_col] if c in pri.columns]], on='_nm', how='left', suffixes=(None, '_pri'))
                else:
                    dep[f'{tg_col}_pri' if tg_col else 'targets_pri'] = np.nan
                    dep[f'{ru_col}_pri' if ru_col else 'rush_att_pri'] = np.nan

                # Derive per-team prior shares
                dep['pos_up'] = dep['position'].astype(str).str.upper()
                # Receiving prior shares among WR/TE/RB only
                if tg_col and f'{tg_col}_pri' in dep.columns:
                    rec_mask = dep['pos_up'].isin(['WR','TE','RB'])
                    tg_vals = pd.to_numeric(dep.loc[rec_mask, f'{tg_col}_pri'], errors='coerce').fillna(0.0)
                    tg_sum = float(tg_vals.sum())
                    if tg_sum > 0:
                        pr_t_share = pd.Series(0.0, index=dep.index)
                        pr_t_share.loc[rec_mask] = tg_vals / tg_sum
                    else:
                        pr_t_share = pd.Series(0.0, index=dep.index)
                else:
                    pr_t_share = pd.Series(0.0, index=dep.index)

                # Rushing prior shares among RB and QB
                if ru_col and f'{ru_col}_pri' in dep.columns:
                    ru_mask = dep['pos_up'].isin(['RB','QB'])
                    ru_vals = pd.to_numeric(dep.loc[ru_mask, f'{ru_col}_pri'], errors='coerce').fillna(0.0)
                else:
                    ru_mask = dep['pos_up'].isin(['RB','QB'])
                    ru_vals = pd.Series(0.0, index=dep.index)
                # Fill QB rush priors when missing
                try:
                    if qb_rush_priors is not None and not qb_rush_priors.empty:
                        # Map by id first
                        qmap_id = {str(r['_pid']): float(r['rush_att_pg']) for _, r in qb_rush_priors.iterrows() if pd.notna(r.get('_pid'))}
                        qmap_nm = {str(r['_nm']): float(r['rush_att_pg']) for _, r in qb_rush_priors.iterrows() if pd.notna(r.get('_nm'))}
                        for idx in dep.index[dep['pos_up'] == 'QB']:
                            cur = float(pd.to_numeric(ru_vals.get(idx), errors='coerce') if idx in ru_vals.index else 0.0)
                            if cur <= 0:
                                pid = str(dep.at[idx, 'player_id']) if 'player_id' in dep.columns else ''
                                nm = str(dep.at[idx, '_nm']) if '_nm' in dep.columns else ''
                                val = None
                                if pid and pid in qmap_id:
                                    val = qmap_id[pid]
                                elif nm and nm in qmap_nm:
                                    val = qmap_nm[nm]
                                if val is not None:
                                    ru_vals.at[idx] = float(max(0.0, val))
                except Exception:
                    pass
                ru_sum = float(pd.to_numeric(ru_vals, errors='coerce').fillna(0.0).sum())
                if ru_sum > 0:
                    pr_r_share = pd.Series(0.0, index=dep.index)
                    pr_r_share.loc[ru_mask] = pd.to_numeric(ru_vals, errors='coerce').fillna(0.0) / ru_sum
                else:
                    pr_r_share = pd.Series(0.0, index=dep.index)

                # Blend weights (env-tunable)
                w_t = _cfg_float('PROPS_WK1_PRIOR_T_WEIGHT', 0.50, 0.0, 1.0)
                w_r = _cfg_float('PROPS_WK1_PRIOR_R_WEIGHT', 0.45, 0.0, 1.0)
                # Base shares
                dep['t_base'] = pd.to_numeric(dep.get('target_share'), errors='coerce').fillna(0.0)
                dep['r_base'] = pd.to_numeric(dep.get('rush_share'), errors='coerce').fillna(0.0)
                # Apply blend only to the relevant groups
                t_blend = dep['t_base']
                r_blend = dep['r_base']
                try:
                    t_blend = np.where(dep['pos_up'].isin(['WR','TE','RB']), (1 - w_t) * dep['t_base'] + w_t * pr_t_share, dep['t_base'])
                except Exception:
                    t_blend = dep['t_base']
                try:
                    r_blend = np.where(dep['pos_up'].isin(['RB','QB']), (1 - w_r) * dep['r_base'] + w_r * pr_r_share, dep['r_base'])
                except Exception:
                    r_blend = dep['r_base']
                depth['t_blend'] = pd.to_numeric(t_blend, errors='coerce').fillna(dep['t_base'])
                depth['r_blend'] = pd.to_numeric(r_blend, errors='coerce').fillna(dep['r_base'])
                # Use blended volumes to drive strengths
                depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth['t_blend'], 0.0)
                depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth['r_blend'], 0.0)
            except Exception:
                # If anything fails, proceed without Week 1 prior-share blending
                pass

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
            # Softer separation to avoid extreme top-heavy distributions
            if pos == 'WR':
                arr = [1.22, 1.05, 0.98, 0.94, 0.90]
            elif pos == 'TE':
                arr = [1.00, 0.96, 0.90]
            elif pos == 'RB':
                arr = [1.05, 0.98, 0.95]
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

        # Harmonize names with roster to enrich depth with ids/positions/depth order
        try:
            if roster_map is not None and not roster_map.empty:
                rm = roster_map.copy()
                rm['player_nm'] = rm['player'].astype(str).map(normalize_name_loose)
                try:
                    from .name_normalizer import normalize_alias_init_last
                    rm['player_alias'] = rm['player'].astype(str).map(normalize_alias_init_last)
                except Exception:
                    rm['player_alias'] = None
                depth['player_nm'] = depth['player'].astype(str).map(normalize_name_loose)
                try:
                    from .name_normalizer import normalize_alias_init_last
                    depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                except Exception:
                    depth['player_alias'] = None
                # Bring over id/position/depth order when missing via normalized name
                for cols in (
                    ['player','player_id','position','depth_chart_order'],
                    ['player','player_id','position']
                ):
                    try:
                        add = rm[[c for c in cols if c in rm.columns] + ['player_nm','player_alias']].copy()
                        # only merge if we don't already have player_id
                        if 'player_id' not in depth.columns or depth['player_id'].isna().any():
                            # merge by alias first
                            if 'player_alias' in depth.columns and 'player_alias' in add.columns:
                                depth = depth.merge(add.drop_duplicates(subset=['player_alias']), on='player_alias', how='left', suffixes=(None,'_rm'))
                            # then by loose name for any remaining
                            if 'player_id_rm' not in depth.columns or depth['player_id'].isna().any():
                                depth = depth.merge(add.drop_duplicates(subset=['player_nm']), on='player_nm', how='left', suffixes=(None,'_rm2'))
                            # Prefer existing values, fill missing from _rm/_rm2
                            for suf in ('_rm','_rm2'):
                                pidc = f'player_id{suf}'
                                posc = f'position{suf}'
                                dcc = f'depth_chart_order{suf}'
                                if pidc in depth.columns:
                                    depth['player_id'] = depth.get('player_id').fillna(depth[pidc])
                                if posc in depth.columns:
                                    depth['position'] = depth.get('position').fillna(depth[posc])
                                if ('depth_chart_order' not in depth.columns) and (dcc in depth.columns):
                                    depth['depth_chart_order'] = depth[dcc]
                            # cleanup
                            depth = depth.drop(columns=[c for c in depth.columns if c.endswith('_rm') or c.endswith('_rm2')], errors='ignore')
                    except Exception:
                        pass
                # Coerce dco numeric if present
                if 'depth_chart_order' in depth.columns:
                    depth['depth_chart_order'] = pd.to_numeric(depth['depth_chart_order'], errors='coerce')
        except Exception:
            pass

        # Week-specific: remove or down-weight inactive players before allocation by zeroing their shares
        try:
            act_map = _active_roster(int(tr.get('season')), int(tr.get('week')))
            if act_map is not None and not act_map.empty:
                depth['_nm'] = depth['player'].astype(str).map(normalize_name_loose)
                # Attach is_active via player_id first, then by normalized name
                if 'player_id' in depth.columns and 'player_id' in act_map.columns:
                    depth['player_id'] = depth['player_id'].astype(str)
                    act_pid = act_map.rename(columns={'_pid':'player_id'})[['team','player_id','is_active']].copy()
                    act_pid['player_id'] = act_pid['player_id'].astype(str)
                    depth = depth.merge(act_pid[act_pid['team'] == team][['player_id','is_active']].rename(columns={'is_active':'is_active_pid'}), on='player_id', how='left')
                if 'is_active_pid' not in depth.columns or depth['is_active_pid'].isna().any():
                    act_nm = act_map.copy()
                    act_nm['_nm'] = act_nm['_nm'].astype(str)
                    depth = depth.merge(act_nm[act_nm['team'] == team][['_nm','is_active']].rename(columns={'is_active':'is_active_nm'}), on='_nm', how='left')
                # Combine id/name flags conservatively: if either indicates inactive (0), treat as inactive
                a = pd.to_numeric(depth.get('is_active_pid'), errors='coerce')
                b = pd.to_numeric(depth.get('is_active_nm'), errors='coerce')
                # default active (1) when both missing
                depth['is_active'] = 1
                if a is not None:
                    depth['is_active'] = depth['is_active'].where(a.isna(), a)
                if b is not None:
                    # take min across available
                    depth['is_active'] = np.minimum(pd.to_numeric(depth['is_active'], errors='coerce').fillna(1), b.fillna(1)).astype(int)
                # If ESPN depth chart present, also apply ESPN active flag by name
                try:
                    espn_act = _espn_team_active_map(int(tr.get('season')), int(tr.get('week')), team)
                    if espn_act is not None and not espn_act.empty:
                        depth = depth.merge(espn_act[['player','is_active_espn','_nm']].rename(columns={'_nm':'_nm_espn'}), on='player', how='left')
                        # if no direct match by player, try normalized name
                        need = depth['is_active_espn'].isna()
                        if need.any():
                            # Note: the espn_act frame uses column '_nm' (no leading space)
                            depth = depth.merge(espn_act[['_nm','is_active_espn']].rename(columns={'_nm':'_nm'}), on='_nm', how='left', suffixes=(None,'_nm2'))
                            depth['is_active_espn'] = depth['is_active_espn'].fillna(depth.get('is_active_espn_nm2'))
                        depth = depth.drop(columns=[c for c in depth.columns if c.endswith('_nm2') or c=='_nm_espn'], errors='ignore')
                        c = pd.to_numeric(depth.get('is_active_espn'), errors='coerce')
                        if c is not None:
                            depth['is_active'] = np.minimum(pd.to_numeric(depth['is_active'], errors='coerce').fillna(1), c.fillna(1)).astype(int)
                except Exception:
                    pass
                # Week 1 stricter rule: if player not found on this team's weekly roster (both maps NaN), mark inactive
                try:
                    if int(tr.get('week')) == 1:
                        missing_both = a.isna() & b.isna()
                        if missing_both.any():
                            depth.loc[missing_both.fillna(False), 'is_active'] = 0
                except Exception:
                    pass
                depth = depth.drop(columns=['is_active_pid','is_active_nm'], errors='ignore')
                # Zero out shares for inactive rows
                inactive_mask = depth['is_active'].eq(0)
                if inactive_mask.any():
                    for c in ['rush_share','target_share','rz_rush_share','rz_target_share']:
                        if c in depth.columns:
                            depth.loc[inactive_mask, c] = 0.0
                    # Renormalize each share column across the remaining active players
                    for c in ['rush_share','target_share','rz_rush_share','rz_target_share']:
                        if c in depth.columns:
                            s = float(pd.to_numeric(depth[c], errors='coerce').fillna(0.0).sum())
                            if s > 1e-9:
                                depth[c] = pd.to_numeric(depth[c], errors='coerce').fillna(0.0) / s
            _dump_depth(team, '02_after_actives', depth)
        except Exception:
            pass
        # Emit snapshot even when actives not available
        _dump_depth(team, '02_after_actives', depth)

        # Week 1 coverage injection: if week == 1, inject Week 1 observed players missing from depth (exclude QBs)
        if int(tr.get('week')) == 1:
            try:
                wk1_obs = _week1_usage_from_central(int(tr.get('season')))
            except Exception:
                wk1_obs = pd.DataFrame()
            if (wk1_obs is None) or wk1_obs.empty:
                try:
                    wk1_obs = _week1_usage_from_pbp(int(tr.get('season')))
                except Exception:
                    wk1_obs = pd.DataFrame()
            if wk1_obs is not None and not wk1_obs.empty:
                try:
                    # Attach roster ids to depth to compare by player_id
                    if roster_map is None or roster_map.empty:
                        try:
                            lm = _league_roster_map(int(tr.get('season')))
                            roster_map = lm[lm['team'] == team].copy() if lm is not None and not lm.empty else roster_map
                        except Exception:
                            pass
                    if roster_map is not None and not roster_map.empty:
                        roster_map['player_id'] = roster_map['player_id'].astype(str)
                    if 'player_id' not in depth.columns:
                        depth['player_id'] = ''
                    depth['player_id'] = depth['player_id'].astype(str)
                    # Consider only this team
                    obs_team = wk1_obs[wk1_obs['team'] == team][['player_id','player','rush_share_obs','target_share_obs']].copy()
                    obs_team['player_id'] = obs_team['player_id'].astype(str)
                    # Identify missing ids
                    present_ids = set(depth['player_id'].dropna().astype(str).tolist())
                    cand = obs_team[~obs_team['player_id'].isin(present_ids)].copy()
                    # Apply small thresholds to avoid clutter: any receiver with a Week 1 target, or RB with modest carries
                    try:
                        R_SHARE_THR = float(os.environ.get('PROPS_INJECT_THR_R', '0.05'))
                    except Exception:
                        R_SHARE_THR = 0.05
                    try:
                        T_SHARE_THR = float(os.environ.get('PROPS_INJECT_THR_T', '0.00'))
                    except Exception:
                        T_SHARE_THR = 0.00
                    R_SHARE_THR = float(np.clip(R_SHARE_THR, 0.0, 1.0))
                    T_SHARE_THR = float(np.clip(T_SHARE_THR, 0.0, 1.0))
                    cand = cand[(pd.to_numeric(cand['rush_share_obs'], errors='coerce').fillna(0.0) >= R_SHARE_THR) |
                                (pd.to_numeric(cand['target_share_obs'], errors='coerce').fillna(0.0) >= T_SHARE_THR)]
                    if not cand.empty:
                        rm = roster_map.copy() if roster_map is not None else pd.DataFrame()
                        if rm is not None and not rm.empty:
                            rm['player_id'] = rm['player_id'].astype(str)
                        cand = cand.copy()
                        cand = cand.rename(columns={'player': 'player_obs'})  # keep observed name
                        add = cand.merge(
                            rm[['player_id','player','position']] if rm is not None and not rm.empty else pd.DataFrame(columns=['player_id','player','position']),
                            on='player_id', how='left', suffixes=('_obs','')
                        )
                        # Coalesce player name: roster name > player_y > player_x > observed
                        name_cols = [c for c in ['player', 'player_y', 'player_x', 'player_obs'] if c in add.columns]
                        if 'player' not in add.columns and name_cols:
                            add['player'] = None
                        if name_cols:
                            def _pick_name(row):
                                for cc in name_cols:
                                    v = row.get(cc)
                                    if pd.notna(v) and str(v).strip():
                                        return str(v)
                                return None
                            add['player'] = add.apply(_pick_name, axis=1)
                        # Cleanup possible merge suffix residuals
                        add = add.drop(columns=[c for c in add.columns if c.endswith('_x') or c.endswith('_y')], errors='ignore')
                        # If position missing, infer from shares
                        if 'position' not in add.columns:
                            add['position'] = None
                        mpos = add['position'].isna()
                        if mpos.any():
                            rsh = pd.to_numeric(add.get('rush_share_obs'), errors='coerce').fillna(0.0)
                            tsh = pd.to_numeric(add.get('target_share_obs'), errors='coerce').fillna(0.0)
                            guess = np.where(rsh >= tsh, 'RB', 'WR')
                            add.loc[mpos, 'position'] = guess[mpos]
                        # Compute floors based on observed shares and MERGE into existing depth
                        # to avoid creating duplicate player rows (seen in Week 3).
                        # Prepare present keys for fast membership checks
                        if '_nm' not in depth.columns:
                            try:
                                depth['_nm'] = depth['player'].astype(str).map(normalize_name_loose)
                            except Exception:
                                depth['_nm'] = depth['player'].astype(str).str.lower()
                        present_nm = set(depth['_nm'].astype(str)) if '_nm' in depth.columns else set()
                        present_ids = set()
                        if 'player_id' in depth.columns:
                            present_ids = set(depth['player_id'].dropna().astype(str))
                        rows_to_add = []
                        for _, rr in add.iterrows():
                            pos_up = str(rr.get('position') or '').upper()
                            if pos_up == 'QB':
                                continue
                            nm = rr.get('player')
                            if not nm or pd.isna(nm):
                                continue
                            nm_key = normalize_name_loose(str(nm)) if nm else ''
                            pid_key = str(rr.get('player_id') or '')
                            r_obs = float(pd.to_numeric(rr.get('rush_share_obs'), errors='coerce') or 0.0)
                            t_obs = float(pd.to_numeric(rr.get('target_share_obs'), errors='coerce') or 0.0)
                            r_floor = 0.0; t_floor = 0.0
                            if r_obs > 0:
                                base = float(np.clip(0.6 * r_obs, 0.015, 0.15))
                                if pos_up == 'WR':
                                    r_floor = float(min(base, 0.04))
                                elif pos_up == 'TE':
                                    r_floor = float(min(base, 0.02))
                                else:  # RB
                                    r_floor = base
                            if t_obs > 0:
                                base_t = float(np.clip(0.45 * t_obs, 0.015, 0.20))
                                if pos_up == 'RB':
                                    t_floor = float(min(base_t, 0.10))
                                elif pos_up == 'TE':
                                    t_floor = float(min(base_t, 0.16))
                                else:  # WR
                                    t_floor = float(min(base_t, 0.20))
                            if (r_floor <= 0.0) and (t_floor <= 0.0):
                                continue
                            # If already present, merge floors into existing row via max()
                            merged = False
                            try:
                                if (nm_key in present_nm) or (pid_key and (pid_key in present_ids)):
                                    m = (depth['_nm'] == nm_key)
                                    if pid_key and ('player_id' in depth.columns):
                                        m = m | (depth['player_id'].astype(str) == pid_key)
                                    if m.any():
                                        # Update existing shares to be at least the floor values
                                        for col, val in [('rush_share', r_floor), ('target_share', t_floor), ('rz_rush_share', r_floor * 0.8)]:
                                            if col in depth.columns:
                                                cur = pd.to_numeric(depth.loc[m, col], errors='coerce').fillna(0.0)
                                                depth.loc[m, col] = np.maximum(cur, float(val))
                                        merged = True
                            except Exception:
                                merged = False
                            if not merged:
                                rows_to_add.append({
                                    'season': int(tr.get('season')),
                                    'team': team,
                                    'player': nm,
                                    'position': pos_up if pos_up else 'WR',
                                    'rush_share': r_floor,
                                    'target_share': t_floor,
                                    'rz_rush_share': r_floor * 0.8,
                                    'rz_target_share': 0.0
                                })
                        if rows_to_add:
                            depth = pd.concat([depth, pd.DataFrame(rows_to_add)], ignore_index=True)
                        # Recompute helper fields for later steps
                        depth['pos_up'] = depth['position'].astype(str).str.upper()
                        depth['t_base'] = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
                        depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                        depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth['t_base'], 0.0)
                        depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth['r_base'], 0.0)
                        _dump_depth(team, '02b_after_wk1_inject', depth)
                except Exception:
                    pass

        # Include all active players: append any active roster players not in depth with zero shares (QB/RB/WR/TE)
        try:
            act_map = _active_roster(int(tr.get('season')), int(tr.get('week')))
        except Exception:
            act_map = pd.DataFrame()
        try:
            if act_map is not None and not act_map.empty:
                team_act = act_map[(act_map['team'] == team) & (pd.to_numeric(act_map.get('is_active'), errors='coerce').fillna(0).astype(int) == 1)].copy()
                if not team_act.empty:
                    # Present keys
                    present_ids = set()
                    if 'player_id' in depth.columns:
                        present_ids = set(depth['player_id'].dropna().astype(str).tolist())
                    present_nm = set(depth['player'].astype(str).map(normalize_name_loose))
                    # Ensure roster_map available with position
                    if roster_map is None or roster_map.empty:
                        try:
                            lm = _league_roster_map(int(tr.get('season')))
                            roster_map = lm[lm['team'] == team].copy() if lm is not None and not lm.empty else roster_map
                        except Exception:
                            roster_map = pd.DataFrame()
                    rm = roster_map.copy() if roster_map is not None else pd.DataFrame()
                    if rm is not None and not rm.empty:
                        rm['player_id'] = rm['player_id'].astype(str)
                        rm['_nm'] = rm['player'].astype(str).map(normalize_name_loose)
                        rm['pos_up'] = rm['position'].astype(str).str.upper()
                    # Merge actives to roster map to get display names and positions
                    team_act['_pid'] = team_act.get('_pid').astype(str)
                    cand = team_act.merge(rm[['player_id','player','pos_up']].rename(columns={'player_id':'_pid'}), on='_pid', how='left')
                    # Fallback by name if id missing
                    if 'player' not in cand.columns or cand['player'].isna().any():
                        tmp = team_act.merge(rm[['_nm','player','pos_up']], on='_nm', how='left', suffixes=(None,'_nm'))
                        cand['player'] = cand.get('player').fillna(tmp.get('player'))
                        if 'pos_up' in cand.columns:
                            cand['pos_up'] = cand['pos_up'].fillna(tmp.get('pos_up'))
                        else:
                            cand['pos_up'] = tmp.get('pos_up')
                    # Keep only skill positions
                    cand['pos_up'] = cand['pos_up'].astype(str).str.upper()
                    cand = cand[cand['pos_up'].isin(['QB','RB','WR','TE'])].copy()
                    # Identify those missing from current depth (by id when available else name)
                    to_add = []
                    for _, rr in cand.iterrows():
                        pid = str(rr.get('_pid') or '')
                        nm = str(rr.get('player') or '').strip()
                        if not nm:
                            continue
                        nm_key = normalize_name_loose(nm)
                        id_missing = (pid == '') or (pid not in present_ids)
                        nm_missing = nm_key not in present_nm
                        if id_missing and nm_missing:
                            to_add.append({
                                'season': int(tr.get('season')),
                                'team': team,
                                'player': nm,
                                'position': rr.get('pos_up') or 'WR',
                                'rush_share': 0.0,
                                'target_share': 0.0,
                                'rz_rush_share': 0.0,
                                'rz_target_share': 0.0,
                            })
                    if to_add:
                        depth = pd.concat([depth, pd.DataFrame(to_add)], ignore_index=True)
                        # Recompute helper fields for later steps
                        depth['pos_up'] = depth['position'].astype(str).str.upper()
                        depth['t_base'] = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
                        depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                        depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth['t_base'], 0.0)
                        depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth['r_base'], 0.0)
                        _dump_depth(team, '02c_after_add_all_actives', depth)
            else:
                # Fallback: when weekly actives are unavailable (e.g., upcoming weeks), optionally append roster fringe
                try:
                    use_fallback = str(os.environ.get('PROPS_INCLUDE_ROSTER_FRINGE', '1')).strip().lower() in {'1','true','yes'}
                except Exception:
                    use_fallback = True
                # Only use roster fringe when base depth is empty; otherwise, keep top-of-depth only
                if use_fallback and (depth is not None) and depth.empty:
                    # Ensure roster_map is available for this team
                    if roster_map is None or roster_map.empty:
                        try:
                            lm = _league_roster_map(int(tr.get('season')))
                            roster_map = lm[lm['team'] == team].copy() if lm is not None and not lm.empty else roster_map
                        except Exception:
                            roster_map = pd.DataFrame()
                    rm = roster_map.copy() if roster_map is not None else pd.DataFrame()
                    if rm is not None and not rm.empty:
                        rm['player_id'] = rm['player_id'].astype(str)
                        rm['_nm'] = rm['player'].astype(str).map(normalize_name_loose)
                        rm['pos_up'] = rm['position'].astype(str).str.upper()
                        rm = rm[rm['pos_up'].isin(['QB','RB','WR','TE'])].copy()
                        present_ids = set(depth['player_id'].dropna().astype(str).tolist()) if 'player_id' in depth.columns else set()
                        present_nm = set(depth['player'].astype(str).map(normalize_name_loose)) if 'player' in depth.columns else set()
                        add_rows = []
                        for _, rr in rm.iterrows():
                            pid = str(rr.get('player_id') or '')
                            nm = str(rr.get('player') or '').strip()
                            if not nm:
                                continue
                            nm_key = normalize_name_loose(nm)
                            if (pid in present_ids) or (nm_key in present_nm):
                                continue
                            add_rows.append({
                                'season': int(tr.get('season')),
                                'team': team,
                                'player': nm,
                                'player_id': pid,
                                'position': rr.get('pos_up') or 'WR',
                                'rush_share': 0.0,
                                'target_share': 0.0,
                                'rz_rush_share': 0.0,
                                'rz_target_share': 0.0,
                            })
                        if add_rows:
                            depth = pd.concat([depth, pd.DataFrame(add_rows)], ignore_index=True)
                            depth['pos_up'] = depth['position'].astype(str).str.upper()
                            depth['t_base'] = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
                            depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                            depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth['t_base'], 0.0)
                            depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth['r_base'], 0.0)
                            _dump_depth(team, '02c_after_add_roster_fringe', depth)
        except Exception:
            pass

        # If week > 1, blend in season-to-date observed shares (through week-1)
        # Only compute observed blending when week > 1
        obs = _season_to_date_usage(int(tr.get('season')), max(0, int(tr.get('week')) - 1)) if int(tr.get('week')) > 1 else pd.DataFrame()
        # Prefer Week 1 central stats (targets/carries) for week 2; fallback to PBP usage
        wk1_obs = None
        try:
            if int(tr.get('week')) == 2:
                wk1_obs = _week1_usage_from_central(int(tr.get('season')))
                if wk1_obs is None or wk1_obs.empty:
                    wk1_obs = _week1_usage_from_pbp(int(tr.get('season')))
        except Exception:
            wk1_obs = None
        # Proceed if we have any observed usage source (season-to-date weekly or Week 1 PBP)
        if int(tr.get('week')) > 1 and ((obs is not None and not obs.empty) or (wk1_obs is not None and not wk1_obs.empty)):
            # Attach player_id via roster for better matching
            if roster_map is not None and not roster_map.empty and 'player_id' in roster_map.columns:
                if roster_map is None or roster_map.empty:
                    # fallback to league map
                    try:
                        lm = _league_roster_map(int(tr.get('season')))
                        roster_map = lm[lm['team'] == team].copy() if lm is not None and not lm.empty else roster_map
                    except Exception:
                        pass
                # Merge roster info, carefully handling existing columns to avoid suffix pitfalls
                # Avoid joining the roster 'player' column here to prevent accidental overwrite of names
                rm_cols = [c for c in ['player_id','position'] if c in roster_map.columns]
                if 'player' in roster_map.columns and len(rm_cols) > 0:
                    depth = depth.merge(roster_map[['player'] + rm_cols].copy(), on='player', how='left', suffixes=(None, '_rm'))
                else:
                    # Fall back to name-based merge using a normalized name key when roster_map lacks 'player'
                    depth['_nm'] = depth.get('_nm', depth['player'].astype(str).map(normalize_name_loose))
                    rm = roster_map.copy()
                    if 'player' in rm.columns:
                        rm['_nm'] = rm['player'].astype(str).map(normalize_name_loose)
                    elif 'player_id' in rm.columns:
                        # If only ids exist, we'll attach by id later; skip now
                        rm['_nm'] = None
                    if '_nm' in rm.columns and len(rm_cols) > 0:
                        depth = depth.merge(rm[['_nm'] + rm_cols].dropna(subset=['_nm']).drop_duplicates('_nm'), on='_nm', how='left', suffixes=(None, '_rm'))
                # Consolidate player_id
                if 'player_id' in depth.columns and 'player_id_rm' in depth.columns:
                    depth['player_id'] = depth['player_id'].fillna(depth['player_id_rm'])
                elif 'player_id' not in depth.columns and 'player_id_rm' in depth.columns:
                    depth['player_id'] = depth['player_id_rm']
                # Consolidate position
                if 'position' in depth.columns and 'position_rm' in depth.columns:
                    depth['position'] = depth['position'].fillna(depth['position_rm'])
                elif 'position' not in depth.columns and 'position_rm' in depth.columns:
                    depth['position'] = depth['position_rm']
                # Cleanup suffix columns
                depth = depth.drop(columns=[c for c in ['player_id_rm','position_rm'] if c in depth.columns], errors='ignore')
                # Ensure types
                if 'player_id' in depth.columns:
                    depth['player_id'] = depth['player_id'].astype(str)
                else:
                    # create empty id column to allow downstream merges without KeyError
                    depth['player_id'] = ''
                obs['player_id'] = obs['player_id'].astype(str)
                if wk1_obs is not None and not wk1_obs.empty:
                    # Keep Week 1 names to use as a fallback if roster id join fails
                    cols = ['player_id','rush_share_obs','target_share_obs']
                    if 'player' in wk1_obs.columns:
                        cols = ['player_id','player','rush_share_obs','target_share_obs']
                    obs_team = wk1_obs[wk1_obs['team'] == team][cols]
                else:
                    obs_team = obs[obs['team'] == team][['player_id','rush_share_obs','target_share_obs']]
                depth = depth.merge(obs_team, on='player_id', how='left')
                # Blend observed shares into base shares (lightly, to improve stability)
                beta = _obs_blend_weight(0.45)
                # Use target_share as the base for receiving volume blending (not RZ target share)
                depth['t_base'] = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
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
                    # Thresholds for injection based on observed share (env-tunable)
                    try:
                        R_SHARE_THR = float(os.environ.get('PROPS_INJECT_THR_R', '0.05'))
                    except Exception:
                        R_SHARE_THR = 0.05
                    try:
                        # Default 0.00 means inject any receiver with a Week 1 target
                        T_SHARE_THR = float(os.environ.get('PROPS_INJECT_THR_T', '0.00'))
                    except Exception:
                        T_SHARE_THR = 0.00
                    R_SHARE_THR = float(np.clip(R_SHARE_THR, 0.0, 1.0))
                    T_SHARE_THR = float(np.clip(T_SHARE_THR, 0.0, 1.0))
                    cand = cand[(pd.to_numeric(cand['rush_share_obs'], errors='coerce').fillna(0.0) >= R_SHARE_THR) |
                                (pd.to_numeric(cand['target_share_obs'], errors='coerce').fillna(0.0) >= T_SHARE_THR)]
                    cand = cand[~cand['player_id'].isin(present_ids)]
                    if not cand.empty:
                        # Map ids to names and positions from roster; keep Week1 name as fallback
                        rm = roster_map.copy()
                        if rm is None or rm.empty:
                            try:
                                lm = _league_roster_map(int(tr.get('season')))
                                rm = lm[lm['team'] == team].copy() if lm is not None and not lm.empty else rm
                            except Exception:
                                pass
                        if rm is None:
                            rm = pd.DataFrame()
                        if not rm.empty:
                            rm['player_id'] = rm['player_id'].astype(str)
                        # preserve wk1 player name if present
                        has_wk1_name = 'player' in cand.columns
                        if has_wk1_name:
                            cand = cand.rename(columns={'player':'player_wk1'})
                        add = cand.merge(rm[['player_id','player','position']] if not rm.empty else pd.DataFrame(columns=['player_id','player','position']), on='player_id', how='left')

                        # If player missing after id join, try league-wide roster map then name-alias match; else use wk1 name
                        try:
                            missing_name = add['player'].isna() if 'player' in add.columns else pd.Series([True]*len(add))
                            if missing_name.any():
                                lm = _league_roster_map(int(tr.get('season')))
                                if lm is not None and not lm.empty:
                                    lm['player_id'] = lm['player_id'].astype(str)
                                    add = add.merge(lm[['player_id','player','position']].rename(columns={'player':'player_lm','position':'position_lm'}), on='player_id', how='left')
                                    # fill from league map
                                    if 'player_lm' in add.columns:
                                        add['player'] = add.get('player').fillna(add['player_lm'])
                                    if 'position_lm' in add.columns:
                                        add['position'] = add.get('position').fillna(add['position_lm'])
                                    add = add.drop(columns=[c for c in ['player_lm','position_lm'] if c in add.columns])
                            # If still missing name, use Week 1 name if available
                            if has_wk1_name:
                                add['player'] = add.get('player').fillna(add.get('player_wk1'))
                        except Exception:
                            if has_wk1_name:
                                add['player'] = add.get('player').fillna(add.get('player_wk1'))

                        # If position still missing, try to infer from season rosters by name alias or by shares
                        try:
                            need_pos = add['position'].isna() if 'position' in add.columns else pd.Series([True]*len(add))
                            if need_pos.any():
                                ros = _season_rosters(int(tr.get('season')))
                                name_col = None; id_col = None; pos_col = None; team_src = None
                                if ros is not None and not ros.empty:
                                    for c in ['player_display_name','player_name','display_name','full_name','football_name']:
                                        if c in ros.columns:
                                            name_col = c; break
                                    for c in ['gsis_id','player_id','nfl_id','pfr_id']:
                                        if c in ros.columns:
                                            id_col = c; break
                                    for c in ['depth_chart_position','position']:
                                        if c in ros.columns:
                                            pos_col = c; break
                                    for c in ['team','recent_team','team_abbr']:
                                        if c in ros.columns:
                                            team_src = c; break
                                if name_col or id_col:
                                    m = ros[[col for col in [team_src, id_col, name_col, pos_col] if col]].copy()
                                    if team_src:
                                        m['team'] = m[team_src].astype(str).apply(normalize_team_name)
                                        m = m[m['team'] == team]
                                    if id_col:
                                        m['_pid'] = m[id_col].astype(str)
                                    if name_col:
                                        try:
                                            from .name_normalizer import normalize_alias_init_last
                                            m['_alias'] = m[name_col].astype(str).map(normalize_alias_init_last)
                                        except Exception:
                                            m['_alias'] = m[name_col].astype(str).map(normalize_name_loose)
                                    # fill by id where possible
                                    if 'player_id' in add.columns and id_col:
                                        add = add.merge(m[['_pid', pos_col]].rename(columns={'_pid':'player_id', pos_col:'pos_from_ros'}), on='player_id', how='left')
                                    # fill by name alias
                                    if name_col and 'player' in add.columns:
                                        try:
                                            from .name_normalizer import normalize_alias_init_last
                                            add['_alias'] = add['player'].astype(str).map(normalize_alias_init_last)
                                        except Exception:
                                            add['_alias'] = add['player'].astype(str).map(normalize_name_loose)
                                        add = add.merge(m[['_alias', pos_col]].rename(columns={pos_col:'pos_from_alias'}), on='_alias', how='left')
                                    # finalize position
                                    add['position'] = add.get('position').fillna(add.get('pos_from_ros')).fillna(add.get('pos_from_alias'))
                                    add = add.drop(columns=[c for c in ['_alias','pos_from_ros','pos_from_alias'] if c in add.columns])
                            # Infer from shares if still missing
                            if 'position' not in add.columns:
                                add['position'] = None
                            mask_missing = add['position'].isna()
                            if mask_missing.any():
                                # RB if rush share >= target share, else WR; leave TE to be corrected by overrides later
                                rsh = pd.to_numeric(add.get('rush_share_obs'), errors='coerce').fillna(0.0)
                                tsh = pd.to_numeric(add.get('target_share_obs'), errors='coerce').fillna(0.0)
                                guess = np.where(rsh >= tsh, 'RB', 'WR')
                                add.loc[mask_missing, 'position'] = guess[mask_missing]
                        except Exception:
                            pass

                        # Drop any rows still lacking a player name
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
                                    # Use a conservative floor from observed share; cap by position
                                    base_t = float(np.clip(0.45 * t_obs, 0.015, 0.20))
                                    if pos_up == 'RB':
                                        t_floor = float(min(base_t, 0.10))
                                    elif pos_up == 'TE':
                                        t_floor = float(min(base_t, 0.16))
                                    else:  # WR default
                                        t_floor = float(min(base_t, 0.20))
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
                                depth['t_base'] = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
                                depth['r_base'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                                # Ensure blended shares exist for new rows
                                if 't_blend' in depth.columns:
                                    depth['t_blend'] = pd.to_numeric(depth['t_blend'], errors='coerce')
                                    depth['t_blend'] = depth['t_blend'].fillna(depth['t_base'])
                                if 'r_blend' in depth.columns:
                                    depth['r_blend'] = pd.to_numeric(depth['r_blend'], errors='coerce')
                                    depth['r_blend'] = depth['r_blend'].fillna(depth['r_base'])
                                depth['recv_strength'] = np.where(depth['pos_up'].isin(['WR','TE','RB']), depth.get('t_blend', depth['t_base']), 0.0)
                                depth['rush_strength'] = np.where(depth['pos_up'].isin(['RB','QB']), depth.get('r_blend', depth['r_base']), 0.0)
                except Exception:
                    # Fail-safe: ignore injection if anything goes wrong
                    pass

                # Position correction: move known pass-catching TEs (from efficiency priors) into TE group if mis-labeled as WR
                try:
                    if eff_priors is not None and not eff_priors.empty:
                        pri = eff_priors.copy()
                        pri['_nm'] = pri['_nm'].astype(str)
                        te_names = set(pri[pri.get('position', '').astype(str).str.upper().eq('TE')]['_nm'].dropna().unique().tolist())
                        depth['_nm'] = depth['_nm'].astype(str)
                        wrong_mask = depth['pos_up'].eq('WR') & depth['_nm'].isin(te_names)
                        if wrong_mask.any():
                            depth.loc[wrong_mask, 'position'] = 'TE'
                            depth.loc[wrong_mask, 'pos_up'] = 'TE'
                except Exception:
                    pass

                # Backfill missing player names from roster maps (by player_id) or Week 1 name fallback
                try:
                    if 'player' in depth.columns and depth['player'].isna().any():
                        # Prefer team roster map first
                        name_map = pd.DataFrame()
                        if roster_map is not None and not roster_map.empty and 'player_id' in roster_map.columns and 'player' in roster_map.columns:
                            nm = roster_map[['player_id','player']].copy()
                            nm['player_id'] = nm['player_id'].astype(str)
                            name_map = nm.dropna(subset=['player'])
                        # Fallback to league map
                        if (name_map is None or name_map.empty):
                            lm = _league_roster_map(int(tr.get('season')))
                            if lm is not None and not lm.empty:
                                nm = lm[['player_id','player']].copy()
                                nm['player_id'] = nm['player_id'].astype(str)
                                name_map = nm.dropna(subset=['player'])
                        if name_map is not None and not name_map.empty and 'player_id' in depth.columns:
                            depth = depth.merge(name_map.rename(columns={'player':'player_from_id'}), on='player_id', how='left')
                            depth['player'] = depth['player'].fillna(depth.get('player_from_id'))
                            depth = depth.drop(columns=['player_from_id'], errors='ignore')
                        # Fallback: if injection preserved a Week 1 name column
                        if 'player' in depth.columns and 'player_wk1' in depth.columns:
                            depth['player'] = depth['player'].fillna(depth['player_wk1'])
                except Exception:
                    pass

                # Reorder ranks by Week 1 leaders: RB by carries, WR/TE by targets
                try:
                    # WR/TE: use target_share_obs to set recv_rank (higher targets => rank 1)
                    m_rt = depth['pos_up'].isin(['WR','TE'])
                    if m_rt.any():
                        sub = depth.loc[m_rt, ['target_share_obs']].copy()
                        rnk = sub['target_share_obs'].rank(ascending=False, method='first')
                        # Keep existing rank where no observed data
                        depth.loc[m_rt, 'recv_rank'] = np.where(rnk.notna(), rnk, depth.loc[m_rt, 'recv_rank'])
                    # Week 1: dynamically force WR1/TE1 from master CSV to have recv_rank=1 and bump share slightly
                    if int(tr.get('week')) == 1 and ('player_alias' in depth.columns):
                        # TE first
                        te_mask = depth['pos_up'].eq('TE')
                        if te_mask.any() and len(wk1_te1_aliases) > 0:
                            m_alias = depth['player_alias'].astype(str).isin(wk1_te1_aliases)
                            pl = depth['player'].astype(str).str.lower()
                            m_lname = pl.apply(lambda x: any(ln in x for ln in wk1_te1_lnames)) if wk1_te1_lnames else pd.Series([False]*len(depth), index=depth.index)
                            m_dyn = te_mask & (m_alias | m_lname)
                            if m_dyn.any():
                                depth.loc[m_dyn, 'recv_rank'] = 1
                                dup_ones = te_mask & (depth['recv_rank'] == 1) & (~m_dyn)
                                if dup_ones.any():
                                    depth.loc[dup_ones, 'recv_rank'] = 2
                                # bump t_blend/base for override slightly above current max in group
                                base_col = 't_blend' if 't_blend' in depth.columns else 'target_share'
                                if base_col in depth.columns:
                                    te_max = float(pd.to_numeric(depth.loc[te_mask, base_col], errors='coerce').fillna(0.0).max())
                                    cur = pd.to_numeric(depth.loc[m_dyn, base_col], errors='coerce').fillna(0.0)
                                    bump = max(te_max, float(cur.max()))
                                    depth.loc[m_dyn, base_col] = float(min(1.0, bump * 1.01))
                                    if 'recv_strength' in depth.columns:
                                        if base_col == 't_blend':
                                            depth.loc[m_dyn, 'recv_strength'] = depth.loc[m_dyn, 't_blend']
                                        else:
                                            depth.loc[m_dyn, 'recv_strength'] = pd.to_numeric(depth.loc[m_dyn, base_col], errors='coerce').fillna(0.0)
                        # WR next
                        wr_mask = depth['pos_up'].eq('WR')
                        if wr_mask.any() and len(wk1_wr1_aliases) > 0:
                            m_alias = depth['player_alias'].astype(str).isin(wk1_wr1_aliases)
                            pl = depth['player'].astype(str).str.lower()
                            m_lname = pl.apply(lambda x: any(ln in x for ln in wk1_wr1_lnames)) if wk1_wr1_lnames else pd.Series([False]*len(depth), index=depth.index)
                            m_dyn = wr_mask & (m_alias | m_lname)
                            if m_dyn.any():
                                depth.loc[m_dyn, 'recv_rank'] = 1
                                dup_ones = wr_mask & (depth['recv_rank'] == 1) & (~m_dyn)
                                if dup_ones.any():
                                    depth.loc[dup_ones, 'recv_rank'] = 2
                                base_col = 't_blend' if 't_blend' in depth.columns else 'target_share'
                                if base_col in depth.columns:
                                    wr_max = float(pd.to_numeric(depth.loc[wr_mask, base_col], errors='coerce').fillna(0.0).max())
                                    cur = pd.to_numeric(depth.loc[m_dyn, base_col], errors='coerce').fillna(0.0)
                                    bump = max(wr_max, float(cur.max()))
                                    depth.loc[m_dyn, base_col] = float(min(1.0, bump * 1.01))
                                    if 'recv_strength' in depth.columns:
                                        if base_col == 't_blend':
                                            depth.loc[m_dyn, 'recv_strength'] = depth.loc[m_dyn, 't_blend']
                                        else:
                                            depth.loc[m_dyn, 'recv_strength'] = pd.to_numeric(depth.loc[m_dyn, base_col], errors='coerce').fillna(0.0)
                    # RB: use rush_share_obs to set rush_rank (higher carries => rank 1)
                    m_rb = depth['pos_up'].eq('RB')
                    if m_rb.any():
                        sub2 = depth.loc[m_rb, ['rush_share_obs']].copy()
                        rnk2 = sub2['rush_share_obs'].rank(ascending=False, method='first')
                        depth.loc[m_rb, 'rush_rank'] = np.where(rnk2.notna(), rnk2, depth.loc[m_rb, 'rush_rank'])
                    # TE: enforce TE1 by depth_chart_order (if available) to have top recv rank when active
                    te_mask = depth['pos_up'].eq('TE')
                    if te_mask.any() and 'depth_chart_order' in depth.columns:
                        # find minimal depth order among TEs
                        te_df = depth.loc[te_mask, ['depth_chart_order']].copy()
                        if not te_df.empty:
                            # Select rows with smallest depth_chart_order
                            min_ord = pd.to_numeric(te_df['depth_chart_order'], errors='coerce').min()
                            if pd.notna(min_ord):
                                idxs = depth.index[te_mask & (pd.to_numeric(depth['depth_chart_order'], errors='coerce') == min_ord)]
                                if len(idxs) > 0:
                                    # set their recv_rank to 1
                                    depth.loc[idxs, 'recv_rank'] = 1
                    # WR: tie-break by depth_chart_order when available
                    wr_mask = depth['pos_up'].eq('WR')
                    if wr_mask.any() and 'depth_chart_order' in depth.columns:
                        wr_df = depth.loc[wr_mask, ['depth_chart_order']].copy()
                        if not wr_df.empty:
                            min_ord = pd.to_numeric(wr_df['depth_chart_order'], errors='coerce').min()
                            if pd.notna(min_ord):
                                idxs = depth.index[wr_mask & (pd.to_numeric(depth['depth_chart_order'], errors='coerce') == min_ord)]
                                if len(idxs) > 0:
                                    depth.loc[idxs, 'recv_rank'] = 1

                    # Team-specific TE1 overrides (e.g., Vikings -> T.J. Hockenson)
                    try:
                        if TE1_OVERRIDES and team in TE1_OVERRIDES and te_mask.any():
                            # Build alias helpers if not present
                            if 'player_alias' not in depth.columns:
                                try:
                                    depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                                except Exception:
                                    depth['player_alias'] = depth['player'].astype(str)
                            # Candidate name aliases and last-name fallback
                            cand_names = TE1_OVERRIDES.get(team) or []
                            if isinstance(cand_names, str):
                                cand_names = [cand_names]
                            cand_alias = set()
                            cand_lnames = set()
                            for nm in cand_names:
                                try:
                                    cand_alias.add(normalize_alias_init_last(str(nm)))
                                except Exception:
                                    pass
                                parts = str(nm).split()
                                if parts:
                                    cand_lnames.add(parts[-1].lower())
                            m_alias = depth['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(depth), index=depth.index)
                            # Last-name contains fallback (case-insensitive)
                            pl = depth['player'].astype(str).str.lower()
                            m_lname = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(depth), index=depth.index)
                            m_override = te_mask & (m_alias | m_lname)
                            if m_override.any():
                                # Force recv_rank=1 for the override row(s)
                                depth.loc[m_override, 'recv_rank'] = 1
                                # If multiple TEs have rank 1 now, demote non-override ones to 2
                                dup_ones = te_mask & (depth['recv_rank'] == 1) & (~m_override)
                                if dup_ones.any():
                                    depth.loc[dup_ones, 'recv_rank'] = 2
                                # Also ensure t_blend (or base) for override is at least the top TE value to avoid being under-allocated
                                if 't_blend' in depth.columns:
                                    te_max = float(pd.to_numeric(depth.loc[te_mask, 't_blend'], errors='coerce').fillna(0.0).max())
                                    cur = pd.to_numeric(depth.loc[m_override, 't_blend'], errors='coerce').fillna(0.0)
                                    bump = max(te_max, float(cur.max()))
                                    # small nudge above max to prefer override if tied
                                    depth.loc[m_override, 't_blend'] = float(min(1.0, bump * 1.01))
                                    # Keep recv_strength in sync for ranking consumers
                                    depth.loc[m_override, 'recv_strength'] = depth.loc[m_override, 't_blend']
                                else:
                                    # No blended obs; bump base target share column used for receivers
                                    base_col = pcol if 'pcol' in locals() else 'target_share'
                                    if base_col in depth.columns:
                                        te_max = float(pd.to_numeric(depth.loc[te_mask, base_col], errors='coerce').fillna(0.0).max())
                                        cur = pd.to_numeric(depth.loc[m_override, base_col], errors='coerce').fillna(0.0)
                                        bump = max(te_max, float(cur.max()))
                                        depth.loc[m_override, base_col] = float(min(1.0, bump * 1.01))
                                        # Keep recv_strength aligned
                                        if 'recv_strength' in depth.columns:
                                            depth.loc[m_override, 'recv_strength'] = pd.to_numeric(depth.loc[m_override, base_col], errors='coerce').fillna(0.0)
                    except Exception:
                        pass

                    # Team-specific WR1 overrides (e.g., Vikings -> Justin Jefferson)
                    try:
                        wr_mask = depth['pos_up'].eq('WR')
                        if WR1_OVERRIDES and team in WR1_OVERRIDES and wr_mask.any():
                            if 'player_alias' not in depth.columns:
                                try:
                                    depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                                except Exception:
                                    depth['player_alias'] = depth['player'].astype(str)
                            cand_names = WR1_OVERRIDES.get(team) or []
                            if isinstance(cand_names, str):
                                cand_names = [cand_names]
                            cand_alias = set(); cand_lnames = set()
                            for nm in cand_names:
                                try:
                                    cand_alias.add(normalize_alias_init_last(str(nm)))
                                except Exception:
                                    pass
                                parts = str(nm).split()
                                if parts:
                                    cand_lnames.add(parts[-1].lower())
                            m_alias = depth['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(depth), index=depth.index)
                            pl = depth['player'].astype(str).str.lower()
                            m_lname = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(depth), index=depth.index)
                            m_override = wr_mask & (m_alias | m_lname)
                            if m_override.any():
                                depth.loc[m_override, 'recv_rank'] = 1
                                dup_ones = wr_mask & (depth['recv_rank'] == 1) & (~m_override)
                                if dup_ones.any():
                                    depth.loc[dup_ones, 'recv_rank'] = 2
                                # Ensure the override has at least the top WR t_blend/base
                                if 't_blend' in depth.columns:
                                    wr_max = float(pd.to_numeric(depth.loc[wr_mask, 't_blend'], errors='coerce').fillna(0.0).max())
                                    cur = pd.to_numeric(depth.loc[m_override, 't_blend'], errors='coerce').fillna(0.0)
                                    bump = max(wr_max, float(cur.max()))
                                    depth.loc[m_override, 't_blend'] = float(min(1.0, bump * 1.01))
                                    depth.loc[m_override, 'recv_strength'] = depth.loc[m_override, 't_blend']
                                else:
                                    base_col = pcol if 'pcol' in locals() else 'target_share'
                                    if base_col in depth.columns:
                                        wr_max = float(pd.to_numeric(depth.loc[wr_mask, base_col], errors='coerce').fillna(0.0).max())
                                        cur = pd.to_numeric(depth.loc[m_override, base_col], errors='coerce').fillna(0.0)
                                        bump = max(wr_max, float(cur.max()))
                                        depth.loc[m_override, base_col] = float(min(1.0, bump * 1.01))
                                        if 'recv_strength' in depth.columns:
                                            depth.loc[m_override, 'recv_strength'] = pd.to_numeric(depth.loc[m_override, base_col], errors='coerce').fillna(0.0)
                    except Exception:
                        pass

                    # TE group enforcement: ensure the top TE (override, then central-derived, then priors/depth/base score) gets TE1 share
                    try:
                        te_mask = depth['pos_up'].eq('TE')
                        if te_mask.any():
                            te_df = depth.loc[te_mask].copy()
                            te_df['_t_base'] = pd.to_numeric(te_df.get('target_share'), errors='coerce').fillna(0.0)
                            # Check team override
                            m_override = pd.Series([False]*len(te_df), index=te_df.index)
                            try:
                                if TE1_OVERRIDES and team in TE1_OVERRIDES:
                                    if 'player_alias' not in te_df.columns:
                                        try:
                                            te_df['player_alias'] = te_df['player'].astype(str).map(normalize_alias_init_last)
                                        except Exception:
                                            te_df['player_alias'] = te_df['player'].astype(str)
                                    cand_names = TE1_OVERRIDES.get(team) or []
                                    if isinstance(cand_names, str):
                                        cand_names = [cand_names]
                                    cand_alias = set(); cand_lnames = set()
                                    for nm in cand_names:
                                        try:
                                            cand_alias.add(normalize_alias_init_last(str(nm)))
                                        except Exception:
                                            pass
                                        parts = str(nm).split()
                                        if parts:
                                            cand_lnames.add(parts[-1].lower())
                                    m_a = te_df['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(te_df), index=te_df.index)
                                    pl = te_df['player'].astype(str).str.lower()
                                    m_l = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(te_df), index=te_df.index)
                                    m_override = (m_a | m_l)
                                # Add Week 1 dynamic TE1 from central stats
                                if int(tr.get('week')) == 1 and len(wk1_te1_aliases) > 0:
                                    try:
                                        if 'player_alias' not in te_df.columns:
                                            te_df['player_alias'] = te_df['player'].astype(str).map(normalize_alias_init_last)
                                    except Exception:
                                        te_df['player_alias'] = te_df['player'].astype(str)
                                    m_a2 = te_df['player_alias'].astype(str).isin(wk1_te1_aliases)
                                    pll = te_df['player'].astype(str).str.lower()
                                    m_l2 = pll.apply(lambda x: any(ln in x for ln in wk1_te1_lnames)) if wk1_te1_lnames else pd.Series([False]*len(te_df), index=te_df.index)
                                    m_override = m_override | m_a2 | m_l2
                            except Exception:
                                pass
                            te1_idx = None
                            if m_override.any():
                                cand = te_df[m_override]
                                if 'depth_chart_order' in te_df.columns:
                                    dco = pd.to_numeric(cand['depth_chart_order'], errors='coerce')
                                    mind = dco.min()
                                    cand = cand[dco == mind]
                                cand = cand.sort_values('_t_base', ascending=False)
                                te1_idx = cand.index[0]
                            if te1_idx is None:
                                # Score by priors/depth/base
                                try:
                                    pri = eff_priors[['__dummy' if False else '_pid','_nm','targets']].copy().rename(columns={'_pid':'player_id','targets':'prior_targets'})
                                    te_df = te_df.copy(); te_df['_nm'] = te_df['player'].astype(str).map(normalize_name_loose)
                                    if 'player_id' in te_df.columns and te_df['player_id'].notna().any():
                                        te_df['player_id'] = te_df['player_id'].astype(str)
                                        te_df = te_df.merge(pri[['player_id','prior_targets']], on='player_id', how='left')
                                    else:
                                        te_df = te_df.merge(pri, on='_nm', how='left')
                                except Exception:
                                    te_df['prior_targets'] = np.nan
                                pt = pd.to_numeric(te_df.get('prior_targets'), errors='coerce').fillna(0.0)
                                pt_n = pt / (pt.max() if float(pt.max()) > 0 else 1.0)
                                dco = pd.to_numeric(te_df.get('depth_chart_order'), errors='coerce')
                                if dco.notna().any():
                                    d_norm = (dco.max() - dco) / (dco.max() - dco.min() + 1e-6)
                                else:
                                    d_norm = pd.Series([0.5]*len(te_df), index=te_df.index)
                                b = te_df['_t_base']; b_n = b / (b.max() if float(b.max()) > 0 else 1.0)
                                score = 0.6*pt_n + 0.3*d_norm + 0.1*b_n
                                te1_idx = score.sort_values(ascending=False).index[0]
                            rest = te_df.drop(index=[te1_idx]).sort_values('_t_base', ascending=False)
                            # Apply 80/20 split to both target_share and rz_target_share (when present)
                            for col in ['target_share', 'rz_target_share']:
                                if col in depth.columns:
                                    te_total_col = float(pd.to_numeric(depth.loc[te_mask, col], errors='coerce').fillna(0.0).sum())
                                    if te_total_col > 0:
                                        te1_share = 0.80 * te_total_col
                                        te2_share = 0.20 * te_total_col if len(rest) > 0 else 0.0
                                        depth.loc[te1_idx, col] = te1_share
                                        if len(rest) > 0:
                                            depth.loc[rest.index[0], col] = te2_share
                                        if len(rest) > 1:
                                            depth.loc[rest.index[1:], col] = 0.0
                                        # Renormalize to 1.0 across all receivers for this column
                                        tot_all = float(pd.to_numeric(depth[col], errors='coerce').fillna(0.0).sum())
                                        if tot_all > 0:
                                            depth[col] = pd.to_numeric(depth[col], errors='coerce').fillna(0.0) / tot_all
                            # Keep recv_strength aligned with enforced base target_share
                            if 'recv_strength' in depth.columns and 'target_share' in depth.columns:
                                depth['recv_strength'] = np.where(depth['pos_up'].eq('TE'), pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0), depth['recv_strength'])
                    except Exception:
                        pass
                except Exception:
                    pass
            _dump_depth(team, '03_after_obs', depth)

        # Drop rows without a player name to prevent ghost rows from absorbing shares
        try:
            pl_str = depth['player'].astype(str)
            mask_named = depth['player'].notna() & pl_str.str.strip().ne('') & pl_str.str.strip().str.lower().ne('nan')
            depth = depth[mask_named].copy()
        except Exception:
            pass
        _dump_depth(team, '04_after_drop_noname', depth)

        # If all base shares are zero (no priors/ESPN depth), synthesize a conservative baseline
        # to avoid zero/NaN projections. Distribute targets across WR/TE/RB and rush across RBs
        # while respecting any QB rush share already set.
        try:
            share_cols = ['rush_share','target_share','rz_rush_share','rz_target_share']
            for c in share_cols:
                if c not in depth.columns:
                    depth[c] = 0.0
                depth[c] = pd.to_numeric(depth[c], errors='coerce').fillna(0.0)
            # Consider only active players when allocating baselines
            act_mask = pd.Series(True, index=depth.index)
            if 'is_active' in depth.columns:
                try:
                    act_mask = depth['is_active'].astype('Int64').fillna(1).eq(1)
                except Exception:
                    act_mask = pd.Series(True, index=depth.index)
            # Receiving groups present in roster
            depth['pos_up'] = depth['position'].astype(str).str.upper()
            recv_mask_all = depth['pos_up'].isin(['WR','TE','RB']) & act_mask
            rush_mask_all = depth['pos_up'].isin(['RB','QB']) & act_mask
            t_sum0 = float(pd.to_numeric(depth.loc[recv_mask_all, 'target_share'], errors='coerce').fillna(0.0).sum())
            r_sum0 = float(pd.to_numeric(depth.loc[rush_mask_all, 'rush_share'], errors='coerce').fillna(0.0).sum())
            if (t_sum0 <= 1e-12) and (r_sum0 <= 1e-12):
                # Determine present groups for receiving
                has_wr = bool((depth['pos_up'] == 'WR') & act_mask).any()
                has_te = bool((depth['pos_up'] == 'TE') & act_mask).any()
                has_rb = bool((depth['pos_up'] == 'RB') & act_mask).any()
                # Desired group weights; renormalize to present groups
                desired_group = {'WR': 0.65, 'TE': 0.25, 'RB': 0.10}
                present = {k: v for k, v in desired_group.items() if ((k == 'WR' and has_wr) or (k == 'TE' and has_te) or (k == 'RB' and has_rb))}
                s_present = sum(present.values()) or 1.0
                present = {k: (v / s_present) for k, v in present.items()}
                # Helper: pick top N by depth_chart_order if present else stable order
                def _top_idxs(mask, n, order_col='depth_chart_order'):
                    idxs = depth.index[mask & act_mask]
                    if len(idxs) == 0:
                        return []
                    df_sub = depth.loc[idxs, [order_col]].copy() if order_col in depth.columns else pd.DataFrame(index=idxs)
                    if order_col in df_sub.columns:
                        df_sub['_ord'] = pd.to_numeric(df_sub[order_col], errors='coerce').fillna(99).astype(int)
                        df_sub = df_sub.sort_values(['_ord'])
                        return list(df_sub.index[:n])
                    # fallback: preserve existing order
                    return list(idxs[:n])
                # Allocate targets within each group
                allocs = []
                if has_wr:
                    wr_total = present.get('WR', 0.0)
                    wr_weights = [0.50, 0.30, 0.20]
                    wr_idx = _top_idxs(depth['pos_up'].eq('WR'), len(wr_weights))
                    if wr_idx:
                        wsum = sum(wr_weights[:len(wr_idx)])
                        for i, idx in enumerate(wr_idx):
                            allocs.append(('WR', idx, wr_total * (wr_weights[i] / wsum)))
                if has_te:
                    te_total = present.get('TE', 0.0)
                    te_weights = [0.80, 0.20]
                    te_idx = _top_idxs(depth['pos_up'].eq('TE'), len(te_weights))
                    if te_idx:
                        wsum = sum(te_weights[:len(te_idx)])
                        for i, idx in enumerate(te_idx):
                            allocs.append(('TE', idx, te_total * (te_weights[i] / wsum)))
                if has_rb:
                    rb_total_tgt = present.get('RB', 0.0)
                    rb_t_weights = [0.70, 0.30]
                    rb_idx_t = _top_idxs(depth['pos_up'].eq('RB'), len(rb_t_weights))
                    if rb_idx_t:
                        wsum = sum(rb_t_weights[:len(rb_idx_t)])
                        for i, idx in enumerate(rb_idx_t):
                            allocs.append(('RB_T', idx, rb_total_tgt * (rb_t_weights[i] / wsum)))
                # Zero current targets then assign
                depth.loc[recv_mask_all, 'target_share'] = 0.0
                for grp, idx, val in allocs:
                    depth.at[idx, 'target_share'] = float(val)
                # Red-zone target share: WR 0.55, TE 0.35, RB 0.10, same pattern
                rz_group = {'WR': 0.55, 'TE': 0.35, 'RB': 0.10}
                present_rz = {k: v for k, v in rz_group.items() if ((k == 'WR' and has_wr) or (k == 'TE' and has_te) or (k == 'RB' and has_rb))}
                s_rz = sum(present_rz.values()) or 1.0
                present_rz = {k: (v / s_rz) for k, v in present_rz.items()}
                # Reuse same indices for simplicity
                depth.loc[recv_mask_all, 'rz_target_share'] = 0.0
                for k, v in present_rz.items():
                    if k == 'WR' and has_wr:
                        wr_idx = _top_idxs(depth['pos_up'].eq('WR'), 3)
                        wts = [0.50, 0.30, 0.20]
                        wsum = sum(wts[:len(wr_idx)]) or 1.0
                        for i, idx in enumerate(wr_idx):
                            depth.at[idx, 'rz_target_share'] = float(v * (wts[i] / wsum))
                    if k == 'TE' and has_te:
                        te_idx = _top_idxs(depth['pos_up'].eq('TE'), 2)
                        wts = [0.80, 0.20]
                        wsum = sum(wts[:len(te_idx)]) or 1.0
                        for i, idx in enumerate(te_idx):
                            depth.at[idx, 'rz_target_share'] = float(v * (wts[i] / wsum))
                    if k == 'RB' and has_rb:
                        rb_idx = _top_idxs(depth['pos_up'].eq('RB'), 2)
                        wts = [0.70, 0.30]
                        wsum = sum(wts[:len(rb_idx)]) or 1.0
                        for i, idx in enumerate(rb_idx):
                            depth.at[idx, 'rz_target_share'] = float(v * (wts[i] / wsum))
                # Rushing: keep any existing QB rush share; give rest to RBs 75/25
                qb_mask2 = depth['pos_up'].eq('QB') & act_mask
                qb_r_share = float(pd.to_numeric(depth.loc[qb_mask2, 'rush_share'], errors='coerce').fillna(0.0).sum())
                remain = max(0.0, 1.0 - qb_r_share)
                rb_idx_r = _top_idxs(depth['pos_up'].eq('RB'), 2)
                if rb_idx_r:
                    r_wts = [0.75, 0.25]
                    wsum = sum(r_wts[:len(rb_idx_r)]) or 1.0
                    # Zero current RB rush shares and assign
                    depth.loc[(depth['pos_up'].eq('RB')) & act_mask, 'rush_share'] = 0.0
                    for i, idx in enumerate(rb_idx_r):
                        depth.at[idx, 'rush_share'] = float(remain * (r_wts[i] / wsum))
                # RZ rush share proportional to rush_share, with a small bias factor toward non-QB
                try:
                    depth['rz_rush_share'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                    r_tot = float(depth.loc[rush_mask_all, 'rz_rush_share'].sum())
                    if r_tot > 0:
                        depth.loc[rush_mask_all, 'rz_rush_share'] = depth.loc[rush_mask_all, 'rz_rush_share'] / r_tot
                except Exception:
                    pass
        except Exception:
            pass

        # Week 1: apply TE group enforcement (80/20 TE1/TE2; TE3+ zero) on base shares before building t_eff
        try:
            if int(tr.get('week')) == 1:
                te_mask = depth['pos_up'].eq('TE') if 'pos_up' in depth.columns else depth['position'].astype(str).str.upper().eq('TE')
                if te_mask.any():
                    # Ensure helper columns
                    if 'pos_up' not in depth.columns:
                        depth['pos_up'] = depth['position'].astype(str).str.upper()
                    te_df = depth.loc[te_mask].copy()
                    te_df['_t_base'] = pd.to_numeric(te_df.get('target_share'), errors='coerce').fillna(0.0)
                    # Determine TE1: prefer override match, then Week 1 central-derived TE1; else score by priors (prior_targets), depth order, and base share
                    m_override = pd.Series([False]*len(te_df), index=te_df.index)
                    try:
                        if TE1_OVERRIDES and team in TE1_OVERRIDES:
                            if 'player_alias' not in te_df.columns:
                                try:
                                    te_df['player_alias'] = te_df['player'].astype(str).map(normalize_alias_init_last)
                                except Exception:
                                    te_df['player_alias'] = te_df['player'].astype(str)
                            cand_names = TE1_OVERRIDES.get(team) or []
                            if isinstance(cand_names, str):
                                cand_names = [cand_names]
                            cand_alias = set(); cand_lnames = set()
                            for nm in cand_names:
                                try:
                                    cand_alias.add(normalize_alias_init_last(str(nm)))
                                except Exception:
                                    pass
                                parts = str(nm).split()
                                if parts:
                                    cand_lnames.add(parts[-1].lower())
                            m_a = te_df['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(te_df), index=te_df.index)
                            pl = te_df['player'].astype(str).str.lower()
                            m_l = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(te_df), index=te_df.index)
                            m_override = (m_a | m_l)
                        # Add Week 1 dynamic TE1 from central stats
                        if len(wk1_te1_aliases) > 0:
                            try:
                                if 'player_alias' not in te_df.columns:
                                    te_df['player_alias'] = te_df['player'].astype(str).map(normalize_alias_init_last)
                            except Exception:
                                te_df['player_alias'] = te_df['player'].astype(str)
                            m_a2 = te_df['player_alias'].astype(str).isin(wk1_te1_aliases)
                            pll = te_df['player'].astype(str).str.lower()
                            m_l2 = pll.apply(lambda x: any(ln in x for ln in wk1_te1_lnames)) if wk1_te1_lnames else pd.Series([False]*len(te_df), index=te_df.index)
                            m_override = m_override | m_a2 | m_l2
                    except Exception:
                        pass
                    te1_idx = None
                    if m_override.any():
                        cand = te_df[m_override]
                        if 'depth_chart_order' in te_df.columns:
                            dco = pd.to_numeric(cand['depth_chart_order'], errors='coerce')
                            mind = dco.min()
                            cand = cand[dco == mind]
                        cand = cand.sort_values('_t_base', ascending=False)
                        te1_idx = cand.index[0]
                    if te1_idx is None:
                        # Attach efficiency priors prior_targets to score pass-catching ability
                        try:
                            pri = eff_priors[['__dummy' if False else '_pid','_nm','targets']].copy().rename(columns={'_pid':'player_id','targets':'prior_targets'})
                            te_df = te_df.copy()
                            te_df['_nm'] = te_df['player'].astype(str).map(normalize_name_loose)
                            # Try id-merge if player_id present
                            if 'player_id' in te_df.columns and te_df['player_id'].notna().any():
                                te_df['player_id'] = te_df['player_id'].astype(str)
                                te_df = te_df.merge(pri[['player_id','prior_targets']], on='player_id', how='left')
                            else:
                                te_df = te_df.merge(pri, on='_nm', how='left')
                        except Exception:
                            te_df['prior_targets'] = np.nan
                        # Normalize components
                        pt = pd.to_numeric(te_df.get('prior_targets'), errors='coerce').fillna(0.0)
                        pt_n = pt / (pt.max() if float(pt.max()) > 0 else 1.0)
                        dco = pd.to_numeric(te_df.get('depth_chart_order'), errors='coerce')
                        if dco.notna().any():
                            d_norm = (dco.max() - dco) / (dco.max() - dco.min() + 1e-6)
                        else:
                            d_norm = pd.Series([0.5]*len(te_df), index=te_df.index)
                        b = te_df['_t_base']
                        b_n = b / (b.max() if float(b.max()) > 0 else 1.0)
                        # Weights: priors 0.6, depth 0.3, base 0.1
                        score = 0.6*pt_n + 0.3*d_norm + 0.1*b_n
                        te1_idx = score.sort_values(ascending=False).index[0]
                    rest = te_df.drop(index=[te1_idx]).sort_values('_t_base', ascending=False)
                    # Apply 80/20 split to both target_share and rz_target_share (when present) and renormalize globally
                    for col in ['target_share', 'rz_target_share']:
                        if col in depth.columns:
                            te_total_col = float(pd.to_numeric(depth.loc[te_mask, col], errors='coerce').fillna(0.0).sum())
                            if te_total_col > 0:
                                te1_share = 0.80 * te_total_col
                                te2_share = 0.20 * te_total_col if len(rest) > 0 else 0.0
                                depth.loc[te1_idx, col] = te1_share
                                if len(rest) > 0:
                                    depth.loc[rest.index[0], col] = te2_share
                                if len(rest) > 1:
                                    depth.loc[rest.index[1:], col] = 0.0
                                tot_all = float(pd.to_numeric(depth[col], errors='coerce').fillna(0.0).sum())
                                if tot_all > 0:
                                    depth[col] = pd.to_numeric(depth[col], errors='coerce').fillna(0.0) / tot_all
                    if 'recv_strength' in depth.columns and 'target_share' in depth.columns:
                        depth['recv_strength'] = np.where(depth['pos_up'].eq('TE'), pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0), depth['recv_strength'])
        except Exception:
            pass

        # Build effective receiving target shares with rank multipliers and renormalize across receiving positions
        recv_mask = depth['pos_up'].isin(['WR','TE','RB'])
        depth['t_mult'] = depth.apply(lambda r: _rank_mult_targets(str(r['pos_up']), r.get('recv_rank')), axis=1)
        # Use blended targets when available; fallback to base column if blended is NaN
        if 't_blend' in depth.columns:
            base_t_share = pd.to_numeric(depth['t_blend'], errors='coerce')
            # Fallback to base target_share, not RZ targets, to drive receiving volume
            if 'target_share' in depth.columns:
                base_t_share = base_t_share.fillna(pd.to_numeric(depth['target_share'], errors='coerce'))
            base_t_share = base_t_share.fillna(0.0)
        else:
            base_t_share = pd.to_numeric(depth['target_share'], errors='coerce').fillna(0.0)
        depth['t_eff'] = np.where(recv_mask, base_t_share * depth['t_mult'], 0.0)
        # Preserve group totals (WR/TE/RB) from base_t_share by rescaling within each group
        try:
            base_group = (
                pd.DataFrame({'pos_up': depth['pos_up'], 'base': base_t_share})
                .loc[recv_mask]
                .groupby('pos_up', as_index=False)['base']
                .sum()
            )
            base_map = dict(zip(base_group['pos_up'].astype(str), pd.to_numeric(base_group['base'], errors='coerce').fillna(0.0)))
            for pos_k, tgt_sum in base_map.items():
                m = (depth['pos_up'] == pos_k) & recv_mask
                if m.any():
                    cur = float(pd.to_numeric(depth.loc[m, 't_eff'], errors='coerce').fillna(0.0).sum())
                    if cur > 1e-9:
                        fac = float(tgt_sum) / cur
                        depth.loc[m, 't_eff'] = pd.to_numeric(depth.loc[m, 't_eff'], errors='coerce').fillna(0.0) * fac
        except Exception:
            pass
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
        _dump_depth(team, '05_after_t_eff', depth)

        # RB1 override enforcement: ensure the configured RB1 leads rushing allocation
        try:
            if RB1_OVERRIDES and team in RB1_OVERRIDES:
                rb_mask = depth['pos_up'].eq('RB')
                if rb_mask.any():
                    # Build alias helpers if not present
                    if 'player_alias' not in depth.columns:
                        try:
                            depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                        except Exception:
                            depth['player_alias'] = depth['player'].astype(str)
                    cand_names = RB1_OVERRIDES.get(team) or []
                    if isinstance(cand_names, str):
                        cand_names = [cand_names]
                    cand_alias = set(); cand_lnames = set()
                    for nm in cand_names:
                        try:
                            cand_alias.add(normalize_alias_init_last(str(nm)))
                        except Exception:
                            pass
                        parts = str(nm).split()
                        if parts:
                            cand_lnames.add(parts[-1].lower())
                    m_alias = depth['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(depth), index=depth.index)
                    pl = depth['player'].astype(str).str.lower()
                    m_lname = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(depth), index=depth.index)
                    m_override = rb_mask & (m_alias | m_lname)
                    if m_override.any():
                        # Promote override to top rushing rank and increase r_blend/rush_share slightly above current max
                        base_col = 'r_blend' if 'r_blend' in depth.columns else 'rush_share'
                        rb_vals = pd.to_numeric(depth.loc[rb_mask, base_col], errors='coerce').fillna(0.0)
                        rb_max = float(rb_vals.max())
                        cur = pd.to_numeric(depth.loc[m_override, base_col], errors='coerce').fillna(0.0)
                        bump = max(rb_max, float(cur.max()))
                        depth.loc[m_override, base_col] = float(min(1.0, bump * 1.02))
                        # Keep rush_strength aligned
                        if 'rush_strength' in depth.columns:
                            depth.loc[m_override, 'rush_strength'] = depth.loc[m_override, base_col]
                        # Force rush_rank=1 for override; demote other RBs if tied
                        depth.loc[m_override, 'rush_rank'] = 1
                        dup_ones = rb_mask & (depth['rush_rank'] == 1) & (~m_override)
                        if dup_ones.any():
                            depth.loc[dup_ones, 'rush_rank'] = 2
        except Exception:
            pass

        # Week 1: hard-cap any single TE's t_eff and redistribute excess to WRs proportionally
        try:
            if int(tr.get('week')) == 1:
                te_mask = depth['pos_up'].eq('TE')
                wr_mask2 = depth['pos_up'].eq('WR')
                if te_mask.any() and wr_mask2.any():
                    TE1_EFF_CAP = 0.10  # max share of team targets attributable to a single TE in Week 1
                    # Identify top TE by t_eff
                    te_eff = pd.to_numeric(depth.loc[te_mask, 't_eff'], errors='coerce').fillna(0.0)
                    if not te_eff.empty:
                        idx_te1 = te_eff.sort_values(ascending=False).index[0]
                        cur = float(te_eff.loc[idx_te1])
                        if cur > TE1_EFF_CAP:
                            excess = cur - TE1_EFF_CAP
                            # Cap TE1
                            depth.loc[idx_te1, 't_eff'] = TE1_EFF_CAP
                            # Redistribute excess to WRs proportionally to current t_eff
                            wr_eff = pd.to_numeric(depth.loc[wr_mask2, 't_eff'], errors='coerce').fillna(0.0)
                            wr_sum = float(wr_eff.sum())
                            if wr_sum > 0 and excess > 0:
                                scale_add = (wr_eff / wr_sum) * excess
                                depth.loc[wr_eff.index, 't_eff'] = wr_eff + scale_add
                            else:
                                # If WR sum is zero (unlikely), spread evenly among WRs
                                nwr = int(wr_mask2.sum())
                                if nwr > 0 and excess > 0:
                                    depth.loc[wr_mask2, 't_eff'] = pd.to_numeric(depth.loc[wr_mask2, 't_eff'], errors='coerce').fillna(0.0) + (excess / nwr)
                            # Renormalize t_eff across all receivers to 1.0
                            t_sum2 = float(pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0).sum())
                            if t_sum2 > 0:
                                depth.loc[recv_mask, 't_eff'] = pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0) / t_sum2
        except Exception:
            pass

        # Week 1: gate TE2+ to zero unless Week 1 central stats indicate meaningful TE2 usage; redistribute to WRs
        try:
            if int(tr.get('week')) == 1:
                te_mask = depth['pos_up'].eq('TE')
                wr_mask = depth['pos_up'].eq('WR')
                if te_mask.any():
                    allow_te2 = False
                    try:
                        # Determine if team meaningfully used two TEs in Week 1 (central stats)
                        use = _week1_usage_from_central(int(tr.get('season')))
                        if use is not None and not use.empty:
                            team_use = use[use['team'] == team].copy()
                            if not team_use.empty:
                                # Attach positions via league roster map to identify TEs
                                rm = _league_roster_map(int(tr.get('season')))
                                if rm is not None and not rm.empty:
                                    rm = rm.copy()
                                    rm['team'] = rm['team'].astype(str).apply(normalize_team_name)
                                    team_use = team_use.merge(rm[['team','player_id','position']].drop_duplicates(), on=['team','player_id'], how='left')
                                pos_up = team_use.get('position').astype(str).str.upper()
                                team_use['_pos_up'] = pos_up
                                te_usage = team_use[team_use['_pos_up'] == 'TE'].copy()
                                if not te_usage.empty and 'target_share_obs' in te_usage.columns:
                                    te_usage = te_usage.sort_values('target_share_obs', ascending=False)
                                    if len(te_usage) >= 2:
                                        second_share = float(pd.to_numeric(te_usage.iloc[1]['target_share_obs'], errors='coerce') or 0.0)
                                        # Threshold: only allow TE2 if second TE had >= 12% of team targets in Week 1
                                        allow_te2 = second_share >= 0.12
                    except Exception:
                        allow_te2 = False
                    # Apply gating on effective shares t_eff
                    te_eff = pd.to_numeric(depth.loc[te_mask, 't_eff'], errors='coerce').fillna(0.0)
                    if not te_eff.empty:
                        # Keep the top TE; zero out others unless allowed
                        idx_sorted = te_eff.sort_values(ascending=False).index
                        idx_top = idx_sorted[0]
                        idx_rest = idx_sorted[1:]
                        if len(idx_rest) > 0:
                            if not allow_te2:
                                rest_sum = float(pd.to_numeric(depth.loc[idx_rest, 't_eff'], errors='coerce').fillna(0.0).sum())
                                if rest_sum > 0:
                                    # Zero TE2+ and give their share to WRs proportionally
                                    depth.loc[idx_rest, 't_eff'] = 0.0
                                    wr_eff = pd.to_numeric(depth.loc[wr_mask, 't_eff'], errors='coerce').fillna(0.0)
                                    wr_sum = float(wr_eff.sum())
                                    if wr_sum > 0:
                                        depth.loc[wr_eff.index, 't_eff'] = wr_eff + (wr_eff / wr_sum) * rest_sum
                                    else:
                                        # If no WRs (unlikely), add to top TE bounded by cap
                                        cap = 0.10
                                        cur_top = float(pd.to_numeric(depth.loc[idx_top, 't_eff'], errors='coerce').fillna(0.0))
                                        add = max(0.0, min(rest_sum, cap - cur_top))
                                        depth.loc[idx_top, 't_eff'] = cur_top + add
                                # Renormalize down if slight numeric drift
                                t_sum3 = float(pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0).sum())
                                if t_sum3 > 1.0:
                                    depth.loc[recv_mask, 't_eff'] = pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0) / t_sum3
        except Exception:
            pass

        # Secondary enforcement: if a TE1 override exists for this team, ensure that TE leads TE t_eff while preserving TE group sum
        try:
            te_mask = depth['pos_up'].eq('TE')
            if TE1_OVERRIDES and team in TE1_OVERRIDES and te_mask.any():
                if 'player_alias' not in depth.columns:
                    try:
                        depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                    except Exception:
                        depth['player_alias'] = depth['player'].astype(str)
                cand_names = TE1_OVERRIDES.get(team) or []
                if isinstance(cand_names, str):
                    cand_names = [cand_names]
                cand_alias = set(); cand_lnames = set()
                for nm in cand_names:
                    try:
                        cand_alias.add(normalize_alias_init_last(str(nm)))
                    except Exception:
                        pass
                    parts = str(nm).split()
                    if parts:
                        cand_lnames.add(parts[-1].lower())
                m_alias = depth['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(depth), index=depth.index)
                pl = depth['player'].astype(str).str.lower()
                m_lname = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(depth), index=depth.index)
                m_override = te_mask & (m_alias | m_lname)
                if m_override.any():
                    te_sub = depth.loc[te_mask, 't_eff'].astype(float).fillna(0.0)
                    te_sum = float(te_sub.sum())
                    if te_sum > 0:
                        te_max = float(te_sub.max())
                        # bump override slightly above current max and rebalance others
                        new_override = float(min(1.0, te_max * 1.01))
                        idx_ovr = depth.index[m_override]
                        idx_oth = depth.index[te_mask & (~m_override)]
                        rem = max(te_sum - new_override, 0.0)
                        if len(idx_oth) > 0:
                            oth_vals = depth.loc[idx_oth, 't_eff'].astype(float).fillna(0.0)
                            oth_sum = float(oth_vals.sum())
                            if oth_sum > 0:
                                scale = rem / oth_sum
                                depth.loc[idx_oth, 't_eff'] = oth_vals * scale
                            else:
                                depth.loc[idx_oth, 't_eff'] = rem / float(len(idx_oth))
                        depth.loc[idx_ovr, 't_eff'] = new_override
                        # Enforce Week 1 TE cap again post-override
                        try:
                            if int(tr.get('week')) == 1:
                                TE1_EFF_CAP = 0.10
                                cur_val = float(depth.loc[idx_ovr, 't_eff'].astype(float).fillna(0.0).max())
                                if cur_val > TE1_EFF_CAP:
                                    excess = cur_val - TE1_EFF_CAP
                                    depth.loc[idx_ovr, 't_eff'] = TE1_EFF_CAP
                                    # Give excess to WRs proportionally
                                    wr_mask3 = depth['pos_up'].eq('WR')
                                    wr_eff3 = pd.to_numeric(depth.loc[wr_mask3, 't_eff'], errors='coerce').fillna(0.0)
                                    wr_sum3 = float(wr_eff3.sum())
                                    if wr_sum3 > 0 and excess > 0:
                                        depth.loc[wr_eff3.index, 't_eff'] = wr_eff3 + (wr_eff3 / wr_sum3) * excess
                                    # Renormalize
                                    t_sum3 = float(pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0).sum())
                                    if t_sum3 > 0:
                                        depth.loc[recv_mask, 't_eff'] = pd.to_numeric(depth.loc[recv_mask, 't_eff'], errors='coerce').fillna(0.0) / t_sum3
                        except Exception:
                            pass
                        # keep ranks consistent
                        depth.loc[idx_ovr, 'recv_rank'] = 1
                        if len(idx_oth) > 0:
                            depth.loc[idx_oth, 'recv_rank'] = np.maximum(2, pd.to_numeric(depth.loc[idx_oth, 'recv_rank'], errors='coerce').fillna(2)).astype(int)
        except Exception:
            pass

        # Apply per-player target share caps by position to avoid extreme projections, then renormalize across receivers
        try:
            caps = {
                'WR': float(os.environ.get('PROPS_CAP_WR', '0.34')),
                'TE': float(os.environ.get('PROPS_CAP_TE', '0.22')),
                'RB': float(os.environ.get('PROPS_CAP_RB', '0.20')),
            }
            if recv_mask.any():
                for _ in range(2):  # up to two passes if scaling causes new breaches
                    sub = depth.loc[recv_mask, ['pos_up','t_eff']].copy()
                    cap_vals = sub['pos_up'].map(caps).fillna(0.34)
                    exceeded = sub['t_eff'] > cap_vals
                    if not exceeded.any():
                        break
                    # Cap exceeded
                    sub.loc[exceeded, 't_eff'] = cap_vals[exceeded]
                    # Redistribute remaining to non-exceeded proportionally
                    remaining = 1.0 - float(sub['t_eff'].sum())
                    if remaining > 1e-9:
                        pool_mask = ~exceeded
                        pool_sum = float(sub.loc[pool_mask, 't_eff'].sum())
                        if pool_sum > 1e-9:
                            scale = 1.0 + (remaining / pool_sum)
                            sub.loc[pool_mask, 't_eff'] = sub.loc[pool_mask, 't_eff'] * scale
                        else:
                            # distribute evenly if pool empty
                            n = int(pool_mask.sum())
                            if n > 0:
                                sub.loc[pool_mask, 't_eff'] = remaining / n
                    depth.loc[recv_mask, 't_eff'] = sub['t_eff'].values
                # Final clip for safety and renorm
                depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'].clip(lower=0.0)
                s2 = float(depth.loc[recv_mask, 't_eff'].sum())
                # Only renormalize down if the sum exceeds 1.0; if below 1.0, keep as-is to respect caps
                if s2 > 1.0:
                    depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'] / s2
        except Exception:
            pass

        # Secondary enforcement: WR1 override ensure top WR t_eff while preserving WR group sum
        try:
            wr_mask = depth['pos_up'].eq('WR')
            if WR1_OVERRIDES and team in WR1_OVERRIDES and wr_mask.any():
                if 'player_alias' not in depth.columns:
                    try:
                        depth['player_alias'] = depth['player'].astype(str).map(normalize_alias_init_last)
                    except Exception:
                        depth['player_alias'] = depth['player'].astype(str)
                cand_names = WR1_OVERRIDES.get(team) or []
                if isinstance(cand_names, str):
                    cand_names = [cand_names]
                cand_alias = set(); cand_lnames = set()
                for nm in cand_names:
                    try:
                        cand_alias.add(normalize_alias_init_last(str(nm)))
                    except Exception:
                        pass
                    parts = str(nm).split()
                    if parts:
                        cand_lnames.add(parts[-1].lower())
                m_alias = depth['player_alias'].astype(str).isin(cand_alias) if cand_alias else pd.Series([False]*len(depth), index=depth.index)
                pl = depth['player'].astype(str).str.lower()
                m_lname = pl.apply(lambda x: any(ln in x for ln in cand_lnames)) if cand_lnames else pd.Series([False]*len(depth), index=depth.index)
                m_override = wr_mask & (m_alias | m_lname)
                if m_override.any():
                    wr_sub = depth.loc[wr_mask, 't_eff'].astype(float).fillna(0.0)
                    wr_sum = float(wr_sub.sum())
                    if wr_sum > 0:
                        wr_max = float(wr_sub.max())
                        new_override = float(min(1.0, wr_max * 1.01))
                        idx_ovr = depth.index[m_override]
                        idx_oth = depth.index[wr_mask & (~m_override)]
                        rem = max(wr_sum - new_override, 0.0)
                        if len(idx_oth) > 0:
                            oth_vals = depth.loc[idx_oth, 't_eff'].astype(float).fillna(0.0)
                            oth_sum = float(oth_vals.sum())
                            if oth_sum > 0:
                                scale = rem / oth_sum
                                depth.loc[idx_oth, 't_eff'] = oth_vals * scale
                            else:
                                depth.loc[idx_oth, 't_eff'] = rem / float(len(idx_oth))
                        depth.loc[idx_ovr, 't_eff'] = new_override
                        depth.loc[idx_ovr, 'recv_rank'] = 1
                        if len(idx_oth) > 0:
                            depth.loc[idx_oth, 'recv_rank'] = np.maximum(2, pd.to_numeric(depth.loc[idx_oth, 'recv_rank'], errors='coerce').fillna(2)).astype(int)
        except Exception:
            pass

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
        # Only renormalize down if sum exceeds 1.0; maintain <=1 sums to preserve caps with small receiver pools
        if t_sum > 1.0:
            depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'] / t_sum

        # Final hard cap enforcement after defense multipliers and overrides
        try:
            caps_final = {
                'WR': float(os.environ.get('PROPS_CAP_WR', '0.34')),
                'TE': float(os.environ.get('PROPS_CAP_TE', '0.22')),
                'RB': float(os.environ.get('PROPS_CAP_RB', '0.20')),
            }
            if recv_mask.any():
                for _ in range(3):
                    sub = depth.loc[recv_mask, ['pos_up','t_eff']].copy()
                    cap_vals = sub['pos_up'].map(caps_final).fillna(0.34)
                    exceeded = sub['t_eff'] > cap_vals
                    if not exceeded.any():
                        break
                    # Cap exceeded values
                    sub.loc[exceeded, 't_eff'] = cap_vals[exceeded]
                    # Redistribute only within headroom to avoid creating new breaches
                    current_sum = float(sub['t_eff'].sum())
                    deficit = 1.0 - current_sum
                    if deficit > 1e-9:
                        headroom = (cap_vals - sub['t_eff']).clip(lower=0.0)
                        pool_mask = headroom > 1e-12
                        hr_sum = float(headroom[pool_mask].sum())
                        if hr_sum > 1e-9:
                            add = headroom.copy()
                            add[:] = 0.0
                            add.loc[pool_mask] = headroom[pool_mask] * (deficit / hr_sum)
                            sub['t_eff'] = sub['t_eff'] + add
                    depth.loc[recv_mask, 't_eff'] = sub['t_eff'].values
                # Final safety: clip and renormalize down if slightly above 1 due to numeric noise
                depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'].clip(lower=0.0)
                s3 = float(depth.loc[recv_mask, 't_eff'].sum())
                if s3 > 1.0:
                    depth.loc[recv_mask, 't_eff'] = depth.loc[recv_mask, 't_eff'] / s3
        except Exception:
            pass

        # Build effective rushing shares with rank multipliers and renormalize across RB+QB
        rush_mask = depth['pos_up'].isin(['RB','QB'])
        depth['r_mult'] = depth.apply(lambda r: _rank_mult_carries(str(r['pos_up']), r.get('rush_rank')), axis=1)
        if 'r_blend' in depth.columns:
            base_r_share = pd.to_numeric(depth['r_blend'], errors='coerce').fillna(pd.to_numeric(depth['rush_share'], errors='coerce')).fillna(0.0)
        else:
            base_r_share = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
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
        # 1) ESPN depth chart (active QB1 for that week)
        # 2) Week 1 PBP starter mapping (for stability across early weeks)
        # 3) Roster depth_chart_order (lowest number)
        # 4) Highest combined shares proxy from usage depth
        qb_name = None
        try:
            dc = _load_weekly_depth_chart(int(tr.get('season')), int(tr.get('week')))
            if dc is not None and not dc.empty:
                sub = dc[(dc['team'] == team) & (dc['position'].astype(str).str.upper() == 'QB')].copy()
                if not sub.empty:
                    if 'active' in sub.columns:
                        sub = sub[sub['active'].astype(bool)]
                    if 'depth_rank' in sub.columns:
                        sub = sub.sort_values(['depth_rank','player'])
                    qb_name = str(sub.iloc[0]['player'])
        except Exception:
            pass
        try:
            wk1 = _week1_qb_starters(int(tr.get("season")))
            if wk1 is not None and not wk1.empty:
                if not qb_name:
                    qb_row = wk1[wk1['team'] == team]
                    if not qb_row.empty:
                        qb_name = str(qb_row.iloc[0]['player'])
        except Exception:
            pass
    # Note: trust Week 1 PBP starter even if not present in current roster_map.
    # This ensures actual Week 1 starter remains primary in Week 2 props.
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

        # Safety: if the selected qb_name is not a QB in roster_map (e.g., alias collision), swap to top roster QB
        try:
            if qb_name and (roster_map is not None) and not roster_map.empty and ('position' in roster_map.columns):
                qbs_rm = roster_map[roster_map['position'].astype(str).str.upper() == 'QB'].copy()
                if not qbs_rm.empty:
                    # If current qb_name not in roster QB names, replace with top QB by depth order when available
                    names_qb = set(qbs_rm['player'].astype(str)) if 'player' in qbs_rm.columns else set()
                    if (qb_name not in names_qb) or (not qb_name.strip()):
                        if 'depth_chart_order' in qbs_rm.columns:
                            qbs_rm['_ord'] = pd.to_numeric(qbs_rm['depth_chart_order'], errors='coerce').fillna(99).astype(int)
                            qb_name = str(qbs_rm.sort_values(['_ord','player']).iloc[0].get('player'))
                        else:
                            qb_name = str(qbs_rm.sort_values(['player']).iloc[0].get('player'))
        except Exception:
            pass

        # Final override: force a specific QB by team when configured (handles alias collisions like "C. Williams")
        try:
            forced = QB_OVERRIDE_BY_TEAM.get(team)
            if forced:
                qb_name = str(forced)
        except Exception:
            pass

    # Enforce selected starter as the sole QB row in depth to avoid backups getting passing stats
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
                        qb_share_est = float(np.clip(att_pg / 26.0, qb_share_min, qb_share_max))
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
                            qb_share_est = float(np.clip((att / 17.0) / 26.0, qb_share_min, qb_share_max))
                if qb_share_est is None:
                    # Conservative default
                    qb_share_est = float(qb_share_default)
                # Assign estimated QB rush share; other shares will be renormalized downstream
                depth.loc[qb_mask, 'rush_share'] = float(qb_share_est)
                # Derive QB RZ rush share from non-RZ with tunable scaling (higher captures sneak tendency)
                if 'rz_rush_share' in depth.columns:
                    depth.loc[qb_mask, 'rz_rush_share'] = float(np.clip(qb_share_est * qb_rz_share_scale, qb_rz_share_min, qb_rz_share_max))
                # If we have observed SoD usage (week > 1), blend it into QB effective rushing share to increase differentiation
                try:
                    if int(tr.get('week')) > 1:
                        has_obs = ('rush_share_obs' in depth.columns)
                        if has_obs:
                            qb_obs = pd.to_numeric(depth.loc[qb_mask, 'rush_share_obs'], errors='coerce').fillna(np.nan)
                            # If no observed data for this QB, fall back to estimate
                            qb_obs = qb_obs.where(qb_obs.notna(), float(qb_share_est))
                            qb_blend = float(np.clip((1.0 - qb_obs_blend) * float(qb_share_est) + qb_obs_blend * float(qb_obs.iloc[0] if not qb_obs.empty else qb_share_est), 0.0, 1.0))
                            # Initialize r_blend if missing
                            if 'r_blend' not in depth.columns:
                                depth['r_blend'] = pd.to_numeric(depth['rush_share'], errors='coerce').fillna(0.0)
                            depth.loc[qb_mask, 'r_blend'] = qb_blend
                            # Keep RZ share roughly in line with non-RZ after blending
                            if 'rz_rush_share' in depth.columns:
                                depth.loc[qb_mask, 'rz_rush_share'] = float(np.clip(qb_blend * qb_rz_share_scale, qb_rz_share_min, qb_rz_share_max))
                except Exception:
                    pass
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
        # Ensure player names are present: backfill from roster_map by player_id if needed
        try:
            if roster_map is not None and not roster_map.empty and ('player_id' in depth.columns):
                pid2name = {}
                try:
                    pid2name = dict(zip(roster_map['player_id'].astype(str), roster_map['player'].astype(str)))
                except Exception:
                    pass
                if 'player' not in depth.columns:
                    depth['player'] = depth['player_id'].astype(str).map(pid2name)
                else:
                    m_missing = depth['player'].isna() | depth['player'].astype(str).str.strip().eq('')
                    if m_missing.any():
                        depth.loc[m_missing, 'player'] = depth.loc[missing := m_missing, 'player_id'].astype(str).map(pid2name)
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
            def _get_share(row, primary: str, fallback: str) -> float:
                v = row.get(primary)
                try:
                    v = float(pd.to_numeric(v, errors='coerce'))
                except Exception:
                    v = np.nan
                if not np.isfinite(v):
                    try:
                        v2 = float(pd.to_numeric(row.get(fallback), errors='coerce'))
                    except Exception:
                        v2 = 0.0
                    return float(v2 if np.isfinite(v2) else 0.0)
                return float(v)
            r_share = _get_share(prw, 'r_eff', 'rush_share')
            t_share = _get_share(prw, 't_eff', 'target_share')
            # Volumes
            pl_rush_att = rush_att * r_share
            # Use pass attempts (not dropbacks) to set targets; attempts already excludes sacks
            pl_targets = attempts * t_share
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

            # Apply small position-level receiving tweaks
            if pos_up == 'WR':
                pl_rec_yards *= wr_rec_yards_mult
            elif pos_up == 'TE':
                pl_rec *= te_rec_mult
                pl_rec_yards *= te_rec_yards_mult

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
                            base_rr = 0.12 if pos_up == 'RB' else qb_rz_base
                            rz_r_bias = float(np.clip(1.0 + w2 * (rr_ - base_rr), 0.8, qb_rz_cap))
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
                # Blend in QB passing priors if available (by id or loose name)
                try:
                    pri = None
                    if qb_priors is not None and not qb_priors.empty:
                        cand = None
                        if pl_pid:
                            cand = qb_priors[qb_priors['player_id'].astype(str) == str(pl_pid)]
                        if (cand is None or cand.empty) and 'player' in qb_priors.columns:
                            # fallback by loose name
                            nm = normalize_name_loose(str(pl_name))
                            cand = qb_priors[qb_priors.get('_nm').astype(str) == nm]
                        if cand is not None and not cand.empty:
                            pri = cand.iloc[0]
                    if pri is not None:
                        # Compute priors-derived estimates for current game
                        att_pg = float(pd.to_numeric(pri.get('attempts_pg'), errors='coerce'))
                        ypa = float(pd.to_numeric(pri.get('ypa'), errors='coerce'))
                        td_rate = float(pd.to_numeric(pri.get('td_rate'), errors='coerce'))
                        # Reasonable clamps to avoid wild priors
                        att_pg = float(np.clip(att_pg, 22.0, 44.0))
                        ypa = float(np.clip(ypa, 6.2, 8.2))
                        # 3%..7% default; allow a small bump for elite ceilings via env (e.g., 7.5%)
                        td_rate = float(np.clip(td_rate, 0.030, float(qb_td_rate_hi)))  # per attempt
                        patt_prior = att_pg  # single game expectation
                        py_prior = patt_prior * ypa
                        ptd_prior = patt_prior * td_rate
                        # Blend
                        w = float(qb_prior_w)
                        patt = (1.0 - w) * patt + w * patt_prior
                        py = (1.0 - w) * py + w * py_prior
                        ptd = (1.0 - w) * ptd + w * ptd_prior
                except Exception:
                    pass
                # Calibrate QB passing toward prior-week QB bias
                if pos_bias and int(week) > 1 and 'QB' in pos_bias:
                    qb = pos_bias['QB']
                    patt *= float(np.clip(1.0 - qb_pass * qb.get('patt_frac', 0.0), 0.85, 1.15))
                    py *= float(np.clip(1.0 - qb_pass * qb.get('pyds_frac', 0.0), 0.85, 1.15))
                    ptd *= float(np.clip(1.0 - qb_pass * qb.get('ptd_frac', 0.0), 0.85, 1.15))
                    pint *= float(np.clip(1.0 - qb_pass * qb.get('int_frac', 0.0), 0.85, 1.15))
                # Apply small global QB passing tweaks
                py *= qb_py_mult
                ptd *= qb_ptd_mult
                # Final clamps for sanity
                patt = float(np.clip(patt, 20.0, 48.0))
                py = float(np.clip(py, 150.0, 370.0))
                ptd = float(np.clip(ptd, 0.5, 3.2))
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
    # Post-process: enforce team-level QB override names in final output and consolidate to a single QB row
    try:
        if QB_OVERRIDE_BY_TEAM and not out.empty:
            # Load Week 1 starter ids to help map IDs for forced QBs
            try:
                wk1_map = _week1_qb_starters(int(season))
            except Exception:
                wk1_map = pd.DataFrame()
            for t, forced_name in QB_OVERRIDE_BY_TEAM.items():
                sub = out[(out['team'].astype(str) == t) & (out['position'].astype(str).str.upper() == 'QB')].copy()
                if sub is None or sub.empty:
                    continue
                # Combine QB stats conservatively by sum where it makes sense; else take max/first
                num_cols_sum = ['pass_attempts','pass_yards','pass_tds','interceptions','rush_attempts','rush_yards','rush_tds']
                combined = sub.iloc[0].copy()
                for c in num_cols_sum:
                    if c in sub.columns:
                        combined[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0).sum()
                # Keep other team context from the row with max pass_attempts if available
                if 'pass_attempts' in sub.columns and pd.to_numeric(sub['pass_attempts'], errors='coerce').notna().any():
                    base = sub.iloc[pd.to_numeric(sub['pass_attempts'], errors='coerce').fillna(0).idxmax()]
                else:
                    base = sub.iloc[0]
                for c in base.index:
                    if c not in combined.index:
                        combined[c] = base[c]
                # Force the name and try to set player_id from Week 1 mapping when present
                combined['player'] = str(forced_name)
                if wk1_map is not None and not wk1_map.empty:
                    row = wk1_map[wk1_map['team'] == t]
                    if not row.empty and 'player_id' in row.columns and pd.notna(row.iloc[0].get('player_id')):
                        combined['player_id'] = str(row.iloc[0]['player_id'])
                # Drop all existing QB rows for team and append the consolidated forced row
                out = out[~((out['team'].astype(str) == t) & (out['position'].astype(str).str.upper() == 'QB'))].copy()
                out = pd.concat([out, pd.DataFrame([combined])], ignore_index=True)
            # Backfill QB passing stats from team context if zero/missing for forced starters
            for t, _forced_name in QB_OVERRIDE_BY_TEAM.items():
                mask = (out['team'].astype(str) == t) & (out['position'].astype(str).str.upper() == 'QB')
                if not mask.any():
                    continue
                qb_row = out.loc[mask].iloc[0]
                # If pass_attempts not set or <= 0, assign team-level passing to QB
                def _get_num(s, col):
                    try:
                        return float(pd.to_numeric(s.get(col), errors='coerce'))
                    except Exception:
                        return float('nan')
                pa = _get_num(qb_row, 'pass_attempts')
                if (not np.isfinite(pa)) or (pa <= 0.0):
                    for col_src, col_dst in [
                        ('team_pass_attempts','pass_attempts'),
                        ('team_pass_yards','pass_yards'),
                        ('team_exp_pass_tds','pass_tds'),
                        ('team_exp_int','interceptions'),
                    ]:
                        if (col_src in out.columns) and (col_dst in out.columns):
                            out.loc[mask, col_dst] = pd.to_numeric(out.loc[mask, col_src], errors='coerce')
    except Exception:
        pass
    # Attach is_active from weekly rosters
    try:
        act = _active_roster(int(season), int(week))
        if act is not None and not act.empty:
            out['_nm'] = out['player'].astype(str).map(normalize_name_loose)
            out['player_id'] = out['player_id'].astype(str)
            # Prefer id join, fallback to name+team
            out = out.merge(act.rename(columns={'_pid':'player_id'})[['team','player_id','is_active','status']].rename(columns={'is_active':'is_active_pid','status':'status_pid'}), on=['team','player_id'], how='left')
            # Fill where id missing via name key
            act2 = act.copy()
            out = out.merge(act2.rename(columns={'_nm':'_nm_act'})[['team','_nm_act','is_active','status']].rename(columns={'is_active':'is_active_nm','status':'status_nm'}), left_on=['team','_nm'], right_on=['team','_nm_act'], how='left')
            # Reconcile flags: prefer name-based when available; otherwise id-based; ensure consistency with status
            a = pd.to_numeric(out.get('is_active_pid'), errors='coerce')
            b = pd.to_numeric(out.get('is_active_nm'), errors='coerce')
            # Start with ones as float to avoid silent downcasting warnings, will cast to Int at end
            is_active_vals = pd.Series(1.0, index=out.index, dtype='float64')
            if a is not None:
                # fill from a where available
                ain = ~a.isna()
                if ain.any():
                    av = a.astype(float)
                    is_active_vals.loc[ain] = av.loc[ain]
            if b is not None:
                # combine with b via minimum, treating NaN as 1.0
                bv = b.astype(float).fillna(1.0)
                is_active_vals = np.minimum(is_active_vals.fillna(1.0), bv)
            out['is_active'] = pd.to_numeric(is_active_vals, errors='coerce').fillna(1.0)
            # Week 1: be cautious with missing weekly roster coverage. Only mark as inactive
            # when the team has weekly roster entries and the player is explicitly missing.
            try:
                if int(week) == 1 and ('team' in out.columns) and (act is not None) and (not act.empty):
                    covered_teams = set(str(x) for x in act['team'].dropna().unique())
                    missing_both = (a.isna() & b.isna()) if (a is not None and b is not None) else pd.Series(False, index=out.index)
                    has_coverage = out['team'].astype(str).isin(covered_teams)
                    strict_mask = missing_both & has_coverage
                    if strict_mask.any():
                        out.loc[strict_mask.fillna(False), 'is_active'] = 0
            except Exception:
                pass
            # Status: prefer name-based text when present
            out['status'] = out.get('status').fillna(out.get('status_nm')).fillna(out.get('status_pid'))
            out = out.drop(columns=['_nm_act','is_active_nm','status_nm','is_active_pid','status_pid'], errors='ignore')
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
    # Week 1 hygiene: ensure each team retains a QB row as active, but do not drop inactives for Week 1.
    try:
        if int(week) == 1 and 'is_active' in out.columns and 'position' in out.columns and 'team' in out.columns:
            # Guarantee at least one active QB per team (force activate the sole QB row if needed)
            qb_mask_all = out['position'].astype(str).str.upper() == 'QB'
            if qb_mask_all.any():
                for t in out.loc[qb_mask_all, 'team'].astype(str).unique():
                    m = (out['team'].astype(str) == t) & qb_mask_all
                    if m.any():
                        if not (out.loc[m, 'is_active'].astype('Int64').fillna(1) == 1).any():
                            out.loc[m, 'is_active'] = 1
            # Do NOT drop inactive players for Week 1; keep full roster for calibration and visibility
    except Exception:
        pass
    # Weeks > 1: by default, drop inactive players unless explicitly kept via env flag
    try:
        keep_inactive = str(os.environ.get('PROPS_KEEP_INACTIVE', '0')).strip().lower() in {'1','true','yes'}
        if (not keep_inactive) and ('is_active' in out.columns) and (int(week) > 1):
            out = out[out['is_active'].astype('Int64').fillna(1) == 1].copy()
    except Exception:
        pass
    # Optional: enforce per-team consistency by scaling player sums to match team totals
    try:
        def _env_bool(name: str, default: str = '0') -> bool:
            return str(os.environ.get(name, default)).strip().lower() in {'1','true','yes','on'}
        enforce_usage = _env_bool('PROPS_ENFORCE_TEAM_USAGE', '0')
        enforce_yards = _env_bool('PROPS_ENFORCE_TEAM_YARDS', '0')
        enforce_tds = _env_bool('PROPS_ENFORCE_TEAM_TDS', '0')
        # Scale bounds
        rec_min = float(os.environ.get('PROPS_TEAM_RECV_YDS_SCALE_MIN', '0.60'))
        rec_max = float(os.environ.get('PROPS_TEAM_RECV_YDS_SCALE_MAX', '1.60'))
        rush_min = float(os.environ.get('PROPS_TEAM_RUSH_YDS_SCALE_MIN', '0.60'))
        rush_max = float(os.environ.get('PROPS_TEAM_RUSH_YDS_SCALE_MAX', '1.60'))
        use_min = float(os.environ.get('PROPS_TEAM_USAGE_SCALE_MIN', '0.80'))
        use_max = float(os.environ.get('PROPS_TEAM_USAGE_SCALE_MAX', '1.20'))
        tds_min = float(os.environ.get('PROPS_TEAM_TDS_SCALE_MIN', '0.80'))
        tds_max = float(os.environ.get('PROPS_TEAM_TDS_SCALE_MAX', '1.20'))
        grp_keys = [c for c in ['game_id','team'] if c in out.columns]
        if grp_keys:
            def _scale_group(df_g: pd.DataFrame) -> pd.DataFrame:
                g = df_g.copy()
                # Work on active players only when scaling so team sums match displayed (active) props
                try:
                    if 'is_active' in g.columns:
                        act_mask = g['is_active'].astype('Int64').fillna(0) == 1
                    else:
                        act_mask = pd.Series(True, index=g.index)
                except Exception:
                    act_mask = pd.Series(True, index=g.index)
                # Usage scaling
                try:
                    if enforce_usage:
                        if {'targets','team_pass_attempts'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_pass_attempts'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'targets'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, use_min, use_max))
                                g.loc[act_mask, 'targets'] = pd.to_numeric(g.loc[act_mask, 'targets'], errors='coerce').fillna(0.0) * f
                                # Recompute receptions and rec_yards proportionally to targets adjustment
                                if 'receptions' in g.columns:
                                    g.loc[act_mask, 'receptions'] = pd.to_numeric(g.loc[act_mask, 'receptions'], errors='coerce').fillna(0.0) * f
                                if 'rec_yards' in g.columns and not enforce_yards:
                                    g.loc[act_mask, 'rec_yards'] = pd.to_numeric(g.loc[act_mask, 'rec_yards'], errors='coerce').fillna(0.0) * f
                        if {'rush_attempts','team_rush_attempts'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_rush_attempts'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'rush_attempts'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, use_min, use_max))
                                g.loc[act_mask, 'rush_attempts'] = pd.to_numeric(g.loc[act_mask, 'rush_attempts'], errors='coerce').fillna(0.0) * f
                                if 'rush_yards' in g.columns and not enforce_yards:
                                    g.loc[act_mask, 'rush_yards'] = pd.to_numeric(g.loc[act_mask, 'rush_yards'], errors='coerce').fillna(0.0) * f
                except Exception:
                    pass
                # Yards scaling
                try:
                    if enforce_yards:
                        if {'rec_yards','team_pass_yards'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_pass_yards'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'rec_yards'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, rec_min, rec_max))
                                g.loc[act_mask, 'rec_yards'] = pd.to_numeric(g.loc[act_mask, 'rec_yards'], errors='coerce').fillna(0.0) * f
                        if {'rush_yards','team_rush_yards'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_rush_yards'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'rush_yards'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, rush_min, rush_max))
                                g.loc[act_mask, 'rush_yards'] = pd.to_numeric(g.loc[act_mask, 'rush_yards'], errors='coerce').fillna(0.0) * f
                except Exception:
                    pass
                # TDs scaling (optional, conservative)
                try:
                    if enforce_tds:
                        if {'rec_tds','team_exp_pass_tds'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_exp_pass_tds'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'rec_tds'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, tds_min, tds_max))
                                g.loc[act_mask, 'rec_tds'] = pd.to_numeric(g.loc[act_mask, 'rec_tds'], errors='coerce').fillna(0.0) * f
                        if {'rush_tds','team_exp_rush_tds'}.issubset(g.columns):
                            tot = float(pd.to_numeric(g['team_exp_rush_tds'], errors='coerce').fillna(0.0).iloc[0])
                            s = float(pd.to_numeric(g.loc[act_mask, 'rush_tds'], errors='coerce').fillna(0.0).sum())
                            if s > 0 and tot > 0:
                                f = float(np.clip(tot / s, tds_min, tds_max))
                                g.loc[act_mask, 'rush_tds'] = pd.to_numeric(g.loc[act_mask, 'rush_tds'], errors='coerce').fillna(0.0) * f
                        # Recompute any_td_prob from adjusted expected TDs
                        if {'rec_tds','rush_tds','any_td_prob'}.issubset(g.columns):
                            lam = (
                                pd.to_numeric(g.loc[act_mask, 'rec_tds'], errors='coerce').fillna(0.0)
                                + pd.to_numeric(g.loc[act_mask, 'rush_tds'], errors='coerce').fillna(0.0)
                            )
                            # Only update any_td_prob for active rows
                            g.loc[act_mask, 'any_td_prob'] = (1.0 - np.exp(-lam)).astype(float)
                except Exception:
                    pass
                return g

            out = out.groupby(grp_keys, group_keys=False).apply(_scale_group)
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
    # Final sanity: drop accidental duplicate rows for the same player on the same team/week
    try:
        dedup_keys = [c for c in ["season","week","team","player","position"] if c in out.columns]
        if dedup_keys:
            out = out.drop_duplicates(subset=dedup_keys, keep="first").copy()
    except Exception:
        pass
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

@lru_cache(maxsize=4)
def _week1_usage_from_pbp(season: int) -> pd.DataFrame:
    """Build Week 1 usage shares (targets and rush attempts) per team-player from PBP.
    Returns columns: team, player_id, player (name when available), rush_share_obs, target_share_obs.
    Persists a cache CSV for transparency.
    """
    fp = USAGE_WK1_PBP_TMPL.with_name(USAGE_WK1_PBP_TMPL.name.format(season=int(season)))
    # Try cached
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            if df is not None and not df.empty:
                df['team'] = df['team'].astype(str).apply(normalize_team_name)
                # Ensure types
                for c in ['rush_share_obs','target_share_obs']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
                return df[['team','player_id','player','rush_share_obs','target_share_obs']]
        except Exception:
            pass
    # Build from PBP
    pbp = pd.DataFrame()
    try:
        pbp_fp = DATA_DIR / f"pbp_{int(season)}.parquet"
        if pbp_fp.exists():
            pbp = pd.read_parquet(pbp_fp)
    except Exception:
        pbp = pd.DataFrame()
    if pbp is None or pbp.empty:
        try:
            import nfl_data_py as nfl  # type: ignore
            pbp = nfl.import_pbp_data([int(season)])
        except Exception:
            pbp = pd.DataFrame()
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    df = pbp.copy()
    # Regular season, week 1
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper() == 'REG']
    elif 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    if 'week' not in df.columns:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    df = df[df['week'] == 1].copy()
    if df.empty:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    # Team
    tcol = 'posteam' if 'posteam' in df.columns else ('pos_team' if 'pos_team' in df.columns else None)
    if tcol is None:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    # Targets from pass plays with a receiver identified
    pass_mask = pd.to_numeric(df.get('pass'), errors='coerce').fillna(0).astype(int).eq(1) if 'pass' in df.columns else None
    # Some datasets have 'pass_attempt'
    if pass_mask is None and 'pass_attempt' in df.columns:
        pass_mask = pd.to_numeric(df['pass_attempt'], errors='coerce').fillna(0).astype(int).eq(1)
    rec_id_col = 'receiver_player_id' if 'receiver_player_id' in df.columns else ('receiver_id' if 'receiver_id' in df.columns else None)
    rec_name_col = None
    for c in ['receiver_player_name','receiver_name','receiver']:
        if c in df.columns:
            rec_name_col = c; break
    rec = pd.DataFrame(columns=[tcol,'player_id','player'])
    if pass_mask is not None and (rec_id_col or rec_name_col):
        sub = df[pass_mask].copy()
        # Only plays where a receiver is captured
        has_any = None
        if rec_id_col:
            has_any = sub[rec_id_col].astype(str).str.len() > 0
        if rec_name_col:
            has_name = sub[rec_name_col].astype(str).str.strip().str.len() > 0
            has_any = has_name if has_any is None else (has_any | has_name)
        sub = sub[has_any.fillna(False)].copy()
        cols = [tcol]
        if rec_id_col:
            cols.append(rec_id_col)
        if rec_name_col:
            cols.append(rec_name_col)
        sub = sub[cols].copy()
        sub = sub.rename(columns={tcol:'team_src', (rec_id_col or 'player'):'player_id', (rec_name_col or 'player'):'player'})
        rec = sub
    # Rush attempts by rusher
    rush_id_col = 'rusher_player_id' if 'rusher_player_id' in df.columns else ('rusher_id' if 'rusher_id' in df.columns else None)
    rush_name_col = None
    for c in ['rusher_player_name','rusher_name','rusher']:
        if c in df.columns:
            rush_name_col = c; break
    # Heuristic rush mask: rush attempts include scrambles; use rusher id presence
    ru = pd.DataFrame(columns=[tcol,'player_id','player'])
    if rush_id_col or rush_name_col:
        subr = df.copy()
        has_any = None
        if rush_id_col:
            has_any = subr[rush_id_col].astype(str).str.len() > 0
        if rush_name_col:
            has_name = subr[rush_name_col].astype(str).str.strip().str.len() > 0
            has_any = has_name if has_any is None else (has_any | has_name)
        subr = subr[has_any.fillna(False)].copy()
        cols = [tcol]
        if rush_id_col:
            cols.append(rush_id_col)
        if rush_name_col:
            cols.append(rush_name_col)
        subr = subr[cols].copy()
        subr = subr.rename(columns={tcol:'team_src', (rush_id_col or 'player'):'player_id', (rush_name_col or 'player'):'player'})
        ru = subr
    # Normalize team and compute shares
    def agg_shares(df_in: pd.DataFrame, count_col: str) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=['team','player_id','player',count_col])
        d = df_in.copy()
        d['team'] = d['team_src'].astype(str).apply(normalize_team_name)
        d['cnt'] = 1
        g = d.groupby(['team','player_id','player'], as_index=False)['cnt'].sum()
        tot = g.groupby('team', as_index=False)['cnt'].sum().rename(columns={'cnt':'tot'})
        out = g.merge(tot, on='team', how='left')
        out[count_col] = np.where(out['tot']>0, out['cnt']/out['tot'], 0.0)
        return out[['team','player_id','player',count_col]]
    tg = agg_shares(rec, 'target_share_obs')
    ca = agg_shares(ru, 'rush_share_obs')
    out = tg.merge(ca, on=['team','player_id','player'], how='outer')
    for c in ['target_share_obs','rush_share_obs']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
    # Cache
    try:
        if out is not None and not out.empty:
            fp.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(fp, index=False)
    except Exception:
        pass
    return out[['team','player_id','player','rush_share_obs','target_share_obs']]


@lru_cache(maxsize=4)
def _week1_usage_from_central(season: int) -> pd.DataFrame:
    """Build Week 1 observed usage from central stats CSV if present.
    Returns columns: team, player_id, player, rush_share_obs, target_share_obs
    """
    fp = USAGE_WK1_CENTRAL_TMPL.with_name(USAGE_WK1_CENTRAL_TMPL.name.format(season=int(season)))
    if not fp.exists():
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    if df is None or df.empty:
        return pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    # Normalize
    df['team'] = df['team'].astype(str).apply(normalize_team_name)
    # Aggregate to shares within team: targets and rush attempts
    tg = df.groupby(['team','player_id','player'], as_index=False)['targets'].sum(min_count=1)
    tot_t = tg.groupby('team', as_index=False)['targets'].sum(min_count=1).rename(columns={'targets':'_tot_t'})
    tg = tg.merge(tot_t, on='team', how='left')
    tg['target_share_obs'] = np.where(tg['_tot_t']>0, tg['targets']/tg['_tot_t'], 0.0)
    tg = tg[['team','player_id','player','target_share_obs']]

    ru = df.groupby(['team','player_id','player'], as_index=False)['rush_att'].sum(min_count=1)
    tot_r = ru.groupby('team', as_index=False)['rush_att'].sum(min_count=1).rename(columns={'rush_att':'_tot_r'})
    ru = ru.merge(tot_r, on='team', how='left')
    ru['rush_share_obs'] = np.where(ru['_tot_r']>0, ru['rush_att']/ru['_tot_r'], 0.0)
    ru = ru[['team','player_id','player','rush_share_obs']]

    out = tg.merge(ru, on=['team','player_id','player'], how='outer')
    out['rush_share_obs'] = pd.to_numeric(out['rush_share_obs'], errors='coerce').fillna(0.0)
    out['target_share_obs'] = pd.to_numeric(out['target_share_obs'], errors='coerce').fillna(0.0)
    return out[['team','player_id','player','rush_share_obs','target_share_obs']]

@lru_cache(maxsize=2)
def _week1_top_pos_from_central(season: int) -> pd.DataFrame:
    """Week 1 WR/TE leaders by team from central stats.
    Columns: team, pos_up, player, player_id, player_alias
    """
    try:
        use = _week1_usage_from_central(int(season))
    except Exception:
        use = pd.DataFrame(columns=['team','player_id','player','rush_share_obs','target_share_obs'])
    if use is None or use.empty:
        return pd.DataFrame(columns=['team','pos_up','player','player_id','player_alias'])
    try:
        rm = _league_roster_map(int(season))
    except Exception:
        rm = pd.DataFrame(columns=['team','player','player_id','position'])
    if rm is None or rm.empty:
        return pd.DataFrame(columns=['team','pos_up','player','player_id','player_alias'])
    use = use.copy(); rm = rm.copy()
    use['team'] = use['team'].astype(str).apply(normalize_team_name)
    rm['team'] = rm['team'].astype(str).apply(normalize_team_name)
    try:
        from .name_normalizer import normalize_alias_init_last
        use['_alias'] = use['player'].astype(str).map(normalize_alias_init_last)
        rm['_alias'] = rm['player'].astype(str).map(normalize_alias_init_last)
    except Exception:
        use['_alias'] = use['player'].astype(str)
        rm['_alias'] = rm['player'].astype(str)
    merged = use.copy()
    if 'player_id' in use.columns and use['player_id'].notna().any() and 'player_id' in rm.columns:
        merged = merged.merge(rm[['team','player_id','position']], on=['team','player_id'], how='left')
    if 'position' not in merged.columns or merged['position'].isna().any():
        missing = merged['position'].isna() if 'position' in merged.columns else pd.Series([True]*len(merged), index=merged.index)
        if missing.any():
            add = rm[['team','_alias','position']].drop_duplicates()
            merged = merged.merge(add, left_on=['team','_alias'], right_on=['team','_alias'], how='left', suffixes=(None, '_alias'))
            if 'position' not in merged.columns:
                merged['position'] = merged.get('position_alias')
            else:
                merged['position'] = merged['position'].fillna(merged.get('position_alias'))
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_alias')], errors='ignore')
    merged['pos_up'] = merged.get('position').astype(str).str.upper()
    merged['target_share_obs'] = pd.to_numeric(merged.get('target_share_obs'), errors='coerce').fillna(0.0)
    merged = merged[merged['pos_up'].isin(['WR','TE'])].copy()
    if merged.empty:
        return pd.DataFrame(columns=['team','pos_up','player','player_id','player_alias'])
    merged['_rn'] = merged.groupby(['team','pos_up'])['target_share_obs'].rank(ascending=False, method='first')
    top = merged[merged['_rn'] == 1].copy()
    try:
        from .name_normalizer import normalize_alias_init_last
        top['player_alias'] = top['player'].astype(str).map(normalize_alias_init_last)
    except Exception:
        top['player_alias'] = top['player'].astype(str)
    cols = ['team','pos_up','player','player_id','player_alias']
    for c in cols:
        if c not in top.columns:
            top[c] = None
    return top[cols].drop_duplicates()
