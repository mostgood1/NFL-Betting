import os
import pandas as pd
import numpy as np
from pathlib import Path
from .schemas import GameRow, TeamStatRow, LineRow
import json
from datetime import datetime
from typing import Any, Dict

# Respect NFL_DATA_DIR env when present; fallback to package data folder
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (Path(__file__).resolve().parents[1] / "data")

# CSV readers; swap out to real APIs when ready

def load_games() -> pd.DataFrame:
    fp = DATA_DIR / "games.csv"
    if not fp.exists():
        return pd.DataFrame(columns=GameRow.model_fields.keys())
    df = pd.read_csv(fp)
    return df

def load_team_stats() -> pd.DataFrame:
    fp = DATA_DIR / "team_stats.csv"
    if not fp.exists():
        return pd.DataFrame(columns=TeamStatRow.model_fields.keys())
    df = pd.read_csv(fp)
    return df

def load_lines() -> pd.DataFrame:
    fp = DATA_DIR / "lines.csv"
    if not fp.exists():
        return pd.DataFrame(columns=LineRow.model_fields.keys())
    df = pd.read_csv(fp)

    # Defensive cleanup: some feeds/snapshots occasionally contain bad open lines.
    # When close_* exists, treat it as the more reliable reference and fix obvious outliers.
    try:
        for c in ("spread_home", "close_spread_home", "total", "close_total"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "total" in df.columns and "close_total" in df.columns:
            total = df["total"]
            close_total = df["close_total"]
            # Totals outside plausible NFL range or wildly different from close are treated as bad.
            bad_total = (
                total.notna()
                & close_total.notna()
                & (((total < 30.0) | (total > 70.0)) | ((total - close_total).abs() >= 15.0))
            )
            df.loc[bad_total, "total"] = close_total.loc[bad_total]

        if "spread_home" in df.columns and "close_spread_home" in df.columns:
            spread = df["spread_home"]
            close_spread = df["close_spread_home"]
            # Large jumps or sign flips are usually feed errors (keyed to wrong side).
            bad_spread = (
                spread.notna()
                & close_spread.notna()
                & (
                    ((spread - close_spread).abs() >= 10.0)
                    | ((np.sign(spread) != np.sign(close_spread)) & ((spread - close_spread).abs() >= 2.0))
                )
            )
            df.loc[bad_spread, "spread_home"] = close_spread.loc[bad_spread]
    except Exception:
        pass

    return df

def load_predictions() -> pd.DataFrame:
    """Lightweight reader for predictions.csv used as a fallback source in pipelines.
    Returns an empty DataFrame if the file is missing.
    """
    fp = DATA_DIR / "predictions.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return df

# --- Extended team/context loaders (optional, degrade gracefully) ---
def load_pfr_drive_stats() -> pd.DataFrame:
    """Load team drive-level efficiency.
    Expected columns (per team-week or team-season):
      season, week, team, drives, points, points_per_drive, td_per_drive, fg_per_drive,
      avg_start_fp, yards_per_drive, seconds_per_drive
    """
    fp = DATA_DIR / "pfr_drive_stats.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_redzone_splits() -> pd.DataFrame:
    """Load red-zone offense/defense splits (inside 20/10).
    Expected columns: season, week, team, rzd_off_eff, rzd_def_eff, rzd_off_play_mix_run,
      rzd_off_play_mix_pass, rzd_def_td_rate, rzd_off_td_rate
    """
    fp = DATA_DIR / "redzone_splits.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_explosive_rates() -> pd.DataFrame:
    """Load explosive play rates derived from pbp (e.g., 20+ yards).
    Expected columns: season, week, team, explosive_pass_rate, explosive_run_rate
    """
    fp = DATA_DIR / "explosive_rates.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_penalties_stats() -> pd.DataFrame:
    """Load team penalty rates and turnover-adjusted rates.
    Expected columns: season, week, team, penalty_rate, off_holding_rate, dpi_drawn_rate,
      turnover_adj_rate, int_rate_adj, fumble_rate_adj
    """
    fp = DATA_DIR / "penalties_stats.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_special_teams() -> pd.DataFrame:
    """Load special teams performance.
    Expected columns: season, week, team, fg_acc, punt_epa, kick_return_epa, touchback_rate
    """
    fp = DATA_DIR / "special_teams.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_officiating_crews() -> pd.DataFrame:
    """Load officiating crew tendencies.
    Expected columns: season, week, game_id or matchup keys, crew_name, crew_penalty_rate, crew_dpi_rate, crew_pace_adj
    """
    fp = DATA_DIR / "officiating_crews.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d

def load_weather_noaa() -> pd.DataFrame:
    """Optional NOAA-derived weather augmentations.
    Expected columns: game_id, wx_gust_mph, wx_dew_point_f
    """
    fp = DATA_DIR / "weather_noaa.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return d


# --- Real betting lines JSON support (for daily_updater) ---
def _parse_real_lines_json(blob: Dict[str, Any]) -> pd.DataFrame:
    """Parse unified odds JSON produced by odds_api_client into a flat DataFrame.

    Expected shape:
      {
        "lines": {
          "Away @ Home": {
            "moneyline": {"home": int, "away": int} | None,
            "total_runs": {"line": float, "over": int, "under": int} | None,
            "run_line": {"home": float} | None,
            "markets": [ ... ]
          },
          ...
        },
        "source": "odds_api",
        "fetched_at": "..."
      }
    Returns columns: [home_team, away_team, spread_home, total, moneyline_home, moneyline_away,
                      spread_home_price, spread_away_price, total_over_price, total_under_price]
    Missing fields are left as NaN.
    """
    try:
        lines = blob.get("lines", {}) if isinstance(blob, dict) else {}
    except Exception:
        lines = {}
    rows = []
    for key, v in (lines or {}).items():
        try:
            # Key format "Away @ Home"
            away, home = [x.strip() for x in str(key).split("@", 1)]
        except Exception:
            continue
        spread_home = None
        total = None
        ml_home = None
        ml_away = None
        # prices if available in snapshot
        spread_home_price = None
        spread_away_price = None
        total_over_price = None
        total_under_price = None

        try:
            ml = v.get("moneyline") if isinstance(v, dict) else None
            if isinstance(ml, dict):
                ml_home = ml.get("home")
                ml_away = ml.get("away")
        except Exception:
            pass
        try:
            tr = v.get("total_runs") if isinstance(v, dict) else None
            if isinstance(tr, dict):
                total = tr.get("line")
                total_over_price = tr.get("over")
                total_under_price = tr.get("under")
        except Exception:
            pass
        try:
            rl = v.get("run_line") if isinstance(v, dict) else None
            if isinstance(rl, dict):
                spread_home = rl.get("home")
        except Exception:
            pass

        rows.append({
            "home_team": home,
            "away_team": away,
            "spread_home": spread_home,
            "total": total,
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "spread_home_price": spread_home_price,
            "spread_away_price": spread_away_price,
            "total_over_price": total_over_price,
            "total_under_price": total_under_price,
        })
    return pd.DataFrame(rows)


def _try_load_latest_real_lines() -> pd.DataFrame:
    """Load the most recent non-empty real_betting_lines_*.json snapshot.
    Prefer the newest file that parses to at least one event; if none, return the newest snapshot (may be empty).
    Returns empty DataFrame if none are found or parsing fails.
    """
    snaps = sorted(DATA_DIR.glob("real_betting_lines_*.json"))
    if not snaps:
        return pd.DataFrame(columns=[
            "home_team","away_team","spread_home","total","moneyline_home","moneyline_away",
            "spread_home_price","spread_away_price","total_over_price","total_under_price"
        ])
    # Iterate newest to oldest to find a snapshot with events
    chosen_blob = None
    for p in reversed(snaps):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            df_test = _parse_real_lines_json(blob)
            if df_test is not None and not df_test.empty:
                chosen_blob = blob
                break
        except Exception:
            continue
    # Fallback: use newest snapshot even if empty
    if chosen_blob is None:
        try:
            chosen_blob = json.loads(snaps[-1].read_text(encoding="utf-8"))
        except Exception:
            return pd.DataFrame()
    try:
        df = _parse_real_lines_json(chosen_blob)
    except Exception:
        return pd.DataFrame()
    # Coerce numeric columns
    for c in ["spread_home","total","moneyline_home","moneyline_away","spread_home_price","spread_away_price","total_over_price","total_under_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

