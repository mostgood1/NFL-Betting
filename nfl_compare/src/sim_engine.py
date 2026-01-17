from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _load_sigma_calibration(data_dir: Path = DATA_DIR) -> Dict[str, float]:
    fp = data_dir / "sigma_calibration.json"
    try:
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict) and "ats_sigma" in j and "total_sigma" in j:
                return {"ats_sigma": float(j["ats_sigma"]), "total_sigma": float(j["total_sigma"])}
            if isinstance(j, dict) and "sigma" in j and isinstance(j["sigma"], dict):
                s = j["sigma"]
                return {
                    "ats_sigma": float(s.get("ats_sigma", np.nan)),
                    "total_sigma": float(s.get("total_sigma", np.nan)),
                }
    except Exception:
        pass
    return {"ats_sigma": 12.0, "total_sigma": 11.0}


def _load_totals_calibration(data_dir: Path = DATA_DIR) -> Dict[str, float]:
    fp = data_dir / "totals_calibration.json"
    try:
        allow_unsafe = str(os.environ.get('ALLOW_UNSAFE_TOTALS_CALIBRATION', '0')).strip().lower() in {'1','true','yes','y','on'}
        try:
            scale_min = float(os.environ.get('TOTALS_CAL_SCALE_MIN', '0.85'))
            scale_max = float(os.environ.get('TOTALS_CAL_SCALE_MAX', '1.15'))
            shift_min = float(os.environ.get('TOTALS_CAL_SHIFT_MIN', '-7.0'))
            shift_max = float(os.environ.get('TOTALS_CAL_SHIFT_MAX', '7.0'))
            min_n = int(float(os.environ.get('TOTALS_CAL_MIN_N', '80')))
        except Exception:
            scale_min, scale_max = 0.85, 1.15
            shift_min, shift_max = -7.0, 7.0
            min_n = 80

        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict):
                out: Dict[str, float] = {}
                if "scale" in j:
                    out["scale"] = float(j["scale"])
                if "shift" in j:
                    out["shift"] = float(j["shift"])
                if not allow_unsafe:
                    try:
                        sc = float(out.get('scale', 1.0))
                        sh = float(out.get('shift', 0.0))
                        if not (float(scale_min) <= sc <= float(scale_max)):
                            return {}
                        if not (float(shift_min) <= sh <= float(shift_max)):
                            return {}
                        # Optional metrics.n gate
                        try:
                            metrics = j.get('metrics')
                            if isinstance(metrics, dict) and metrics.get('n') is not None:
                                n = int(float(metrics.get('n')))
                                if n < int(min_n):
                                    return {}
                        except Exception:
                            pass
                    except Exception:
                        return {}
                return out
    except Exception:
        pass
    return {}


def _coerce_float(v: Any) -> float:
    try:
        x = float(pd.to_numeric(v, errors="coerce"))
        return x
    except Exception:
        return float("nan")


def _spread_total_refs(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Spread: prefer close (most reliable for backtests), fallback open.
    spread_home = pd.to_numeric(df.get("spread_home"), errors="coerce") if "spread_home" in df.columns else pd.Series(index=df.index, dtype=float)
    close_spread_home = pd.to_numeric(df.get("close_spread_home"), errors="coerce") if "close_spread_home" in df.columns else pd.Series(index=df.index, dtype=float)
    spread_ref = close_spread_home.fillna(spread_home)
    if "spread_ref" in df.columns:
        spread_ref = pd.to_numeric(df.get("spread_ref"), errors="coerce").fillna(spread_ref)

    # Total: prefer close (most reliable for backtests), fallback open.
    market_total = pd.to_numeric(df.get("total"), errors="coerce") if "total" in df.columns else pd.Series(index=df.index, dtype=float)
    close_total = pd.to_numeric(df.get("close_total"), errors="coerce") if "close_total" in df.columns else pd.Series(index=df.index, dtype=float)
    total_ref = close_total.fillna(market_total)
    if "total_ref" in df.columns:
        total_ref = pd.to_numeric(df.get("total_ref"), errors="coerce").fillna(total_ref)

    return spread_ref, total_ref


def _per_game_sigmas(row: pd.Series, ats_sigma: float, total_sigma: float) -> tuple[float, float]:
    a = float(ats_sigma)
    t = float(total_sigma)
    k_roof = _env_f("SIM_SIGMA_K_ROOF_CLOSED", -0.08)
    k_wind_ats = _env_f("SIM_ATS_SIGMA_K_WIND", 0.015)
    k_wind_tot = _env_f("SIM_TOTAL_SIGMA_K_WIND", 0.025)
    k_rest = _env_f("SIM_SIGMA_K_REST", 0.02)
    k_tot_line = _env_f("SIM_TOTAL_SIGMA_K_LINE", 0.004)
    k_neutral = _env_f("SIM_SIGMA_K_NEUTRAL", 0.0)
    a_min = _env_f("SIM_ATS_SIGMA_MIN", 7.0)
    a_max = _env_f("SIM_ATS_SIGMA_MAX", 20.0)
    t_min = _env_f("SIM_TOTAL_SIGMA_MIN", 6.0)
    t_max = _env_f("SIM_TOTAL_SIGMA_MAX", 20.0)

    roof_closed = _coerce_float(row.get("roof_closed_flag"))
    if np.isfinite(roof_closed):
        a *= 1.0 + k_roof * roof_closed
        t *= 1.0 + k_roof * roof_closed
    w_open = _coerce_float(row.get("wind_open"))
    if np.isfinite(w_open):
        a *= 1.0 + k_wind_ats * (w_open / 10.0)
        t *= 1.0 + k_wind_tot * (w_open / 10.0)
    rest_diff = _coerce_float(row.get("rest_days_diff"))
    if np.isfinite(rest_diff):
        a *= 1.0 + k_rest * (abs(rest_diff) / 7.0)
    tot_line = _coerce_float(row.get("total_ref"))
    if np.isfinite(tot_line):
        t *= 1.0 + k_tot_line * ((tot_line - 44.0) / 10.0)
    neutral_flag = _coerce_float(row.get("neutral_site_flag"))
    if np.isfinite(neutral_flag) and neutral_flag >= 0.5:
        a *= 1.0 + float(k_neutral)
        t *= 1.0 + float(k_neutral)

    a = float(np.clip(a, a_min, a_max))
    t = float(np.clip(t, t_min, t_max))
    return (a, t)


def _compute_game_means_and_lines(
    row: pd.Series,
    totals_cal: Dict[str, float],
    blend_margin: float,
    blend_total: float,
    k_mean_total_wind: float,
    k_mean_margin_rest: float,
    k_mean_margin_elo: float,
    k_mean_total_inj: float,
    k_mean_total_defppg: float,
    k_mean_total_pressure: float,
    pressure_baseline: float,
    k_mean_margin_rating: float,
    k_mean_margin_neutral: float,
    k_mean_total_precip: float,
    k_mean_total_cold: float,
    cold_threshold_f: float,
) -> tuple[float, float, float, float]:
    """Shared mean/line logic used by game/quarter/drive sims.

    Returns (m_mu, t_mu, spread_ref, total_ref), each possibly NaN.
    """
    # Means
    m_mu = _coerce_float(row.get("pred_margin"))
    if not np.isfinite(m_mu):
        ph = _coerce_float(row.get("pred_home_points"))
        pa = _coerce_float(row.get("pred_away_points"))
        if np.isfinite(ph) and np.isfinite(pa):
            m_mu = float(ph - pa)
    if not np.isfinite(m_mu):
        sref = _coerce_float(row.get("spread_ref"))
        if np.isfinite(sref):
            m_mu = -float(sref)

    t_mu = _coerce_float(row.get("pred_total_cal"))
    if not np.isfinite(t_mu):
        t_mu = _coerce_float(row.get("pred_total"))
    if not np.isfinite(t_mu):
        tref = _coerce_float(row.get("total_ref"))
        if np.isfinite(tref):
            t_mu = float(tref)

    s = _coerce_float(row.get("spread_ref"))
    t_line = _coerce_float(row.get("total_ref"))

    # Totals calibration
    try:
        scale = float(totals_cal.get("scale", np.nan))
        shift = float(totals_cal.get("shift", np.nan))
        if np.isfinite(t_mu) and np.isfinite(scale) and np.isfinite(shift):
            t_mu = scale * t_mu + shift
    except Exception:
        pass

    # Market blends
    try:
        if np.isfinite(m_mu) and np.isfinite(s) and 0.0 < float(blend_margin) <= 1.0:
            m_mu = (1.0 - float(blend_margin)) * float(m_mu) + float(blend_margin) * (-float(s))
    except Exception:
        pass
    try:
        if np.isfinite(t_mu) and np.isfinite(t_line):
            eff_blend_total = float(blend_total) if np.isfinite(float(blend_total)) else 0.0

            # Adaptive market pull for totals on upcoming games.
            # This prevents extreme model totals (often from missing/unstable features) from drifting far from market.
            # Defaults: off for finals; on for upcoming with large deltas.
            try:
                hs = _coerce_float(row.get("home_score"))
                a_s = _coerce_float(row.get("away_score"))
                is_final = np.isfinite(hs) and np.isfinite(a_s)
            except Exception:
                is_final = False

            pull_max = _env_f("SIM_TOTAL_MARKET_PULL_MAX_BLEND", 0.75)
            pull_start = _env_f("SIM_TOTAL_MARKET_PULL_START", 6.0)
            pull_delta_at_max = _env_f("SIM_TOTAL_MARKET_PULL_DELTA_AT_MAX", 14.0)

            pull_blend = 0.0
            if (not is_final) and pull_max > 0:
                delta = abs(float(t_mu) - float(t_line))
                if delta > float(pull_start):
                    denom = max(1e-9, float(pull_delta_at_max) - float(pull_start))
                    frac = min(1.0, max(0.0, (delta - float(pull_start)) / denom))
                    pull_blend = float(pull_max) * float(frac)

            eff_blend_total = max(0.0, min(1.0, max(eff_blend_total, pull_blend)))
            if eff_blend_total > 0:
                t_mu = (1.0 - eff_blend_total) * float(t_mu) + eff_blend_total * float(t_line)
    except Exception:
        pass

    # Feature adjustments
    try:
        w_open = _coerce_float(row.get("wind_open"))
        if np.isfinite(t_mu) and np.isfinite(w_open) and w_open > 0:
            t_mu = float(t_mu) + float(k_mean_total_wind) * (w_open / 10.0)
    except Exception:
        pass
    try:
        rest_diff = _coerce_float(row.get("rest_days_diff"))
        if np.isfinite(m_mu) and np.isfinite(rest_diff) and rest_diff != 0:
            m_mu = float(m_mu) + float(k_mean_margin_rest) * (rest_diff / 7.0)
    except Exception:
        pass
    try:
        elo_diff = _coerce_float(row.get("elo_diff"))
        if np.isfinite(m_mu) and np.isfinite(elo_diff):
            m_mu = float(m_mu) + float(k_mean_margin_elo) * (elo_diff / 100.0)
    except Exception:
        pass
    try:
        neutral_flag = _coerce_float(row.get("neutral_site_flag"))
        if np.isfinite(neutral_flag) and neutral_flag >= 0.5 and np.isfinite(m_mu):
            m_mu = float(m_mu) + float(k_mean_margin_neutral)
    except Exception:
        pass
    try:
        h_inj = _coerce_float(row.get("home_inj_starters_out"))
        a_inj = _coerce_float(row.get("away_inj_starters_out"))
        total_out = 0.0
        for v in (h_inj, a_inj):
            if np.isfinite(v):
                total_out += float(v)
        if np.isfinite(t_mu) and total_out > 0:
            t_mu = float(t_mu) + float(k_mean_total_inj) * float(total_out)
    except Exception:
        pass
    try:
        h_def_ppg = _coerce_float(row.get("home_def_ppg"))
        a_def_ppg = _coerce_float(row.get("away_def_ppg"))
        vals = [v for v in (h_def_ppg, a_def_ppg) if np.isfinite(v)]
        def_avg = float(np.mean(vals)) if vals else float("nan")
        if not np.isfinite(def_avg):
            def_diff = _coerce_float(row.get("def_ppg_diff"))
            if np.isfinite(def_diff):
                def_avg = 22.0 + (def_diff / 2.0)
        if np.isfinite(t_mu) and np.isfinite(def_avg):
            # home_def_ppg / away_def_ppg are points allowed (higher = worse defense).
            # Bad defenses should increase totals; good defenses should decrease totals.
            t_mu = float(t_mu) + float(k_mean_total_defppg) * ((22.0 - def_avg) / 5.0)
    except Exception:
        pass
    try:
        h_def_sr = _coerce_float(row.get("home_def_sack_rate_ema"))
        if not np.isfinite(h_def_sr):
            h_def_sr = _coerce_float(row.get("home_def_sack_rate"))
        a_def_sr = _coerce_float(row.get("away_def_sack_rate_ema"))
        if not np.isfinite(a_def_sr):
            a_def_sr = _coerce_float(row.get("away_def_sack_rate"))
        sr_vals = [v for v in (h_def_sr, a_def_sr) if np.isfinite(v)]
        if sr_vals and np.isfinite(t_mu):
            sr_avg = float(np.mean(sr_vals))
            delta_units = (sr_avg - float(pressure_baseline)) / 0.05
            t_mu = float(t_mu) + float(k_mean_total_pressure) * float(delta_units)
    except Exception:
        pass
    try:
        nm_diff = _coerce_float(row.get("net_margin_diff"))
        if np.isfinite(m_mu) and np.isfinite(nm_diff):
            m_mu = float(m_mu) + float(k_mean_margin_rating) * float(nm_diff)
    except Exception:
        pass
    try:
        precip = _coerce_float(row.get("wx_precip_pct"))
        if np.isfinite(t_mu) and np.isfinite(precip) and precip > 0:
            t_mu = float(t_mu) + float(k_mean_total_precip) * (precip / 100.0)
    except Exception:
        pass
    try:
        temp_f = _coerce_float(row.get("wx_temp_f"))
        if np.isfinite(t_mu) and np.isfinite(temp_f) and temp_f < float(cold_threshold_f):
            units = (float(cold_threshold_f) - float(temp_f)) / 10.0
            t_mu = float(t_mu) + float(k_mean_total_cold) * float(units)
    except Exception:
        pass

    # Realism clamps
    try:
        # Default: do NOT silently anchor model totals toward market.
        # Use SIM_TOTAL_DELTA_MAX > 0 to enable market-anchoring clamps.
        delta_t_max = _env_f("SIM_TOTAL_DELTA_MAX", 0.0)
        t_min = _env_f("SIM_TOTAL_MEAN_MIN", 20.0)
        t_max = _env_f("SIM_TOTAL_MEAN_MAX", 62.0)
        if np.isfinite(t_mu):
            if np.isfinite(t_line) and delta_t_max > 0:
                t_mu = float(t_line) + float(np.clip(t_mu - float(t_line), -delta_t_max, delta_t_max))
            t_mu = float(np.clip(t_mu, t_min, t_max))
    except Exception:
        pass
    try:
        # Default: do NOT silently anchor model margins toward market.
        delta_m_max = _env_f("SIM_MARGIN_DELTA_MAX", 0.0)
        m_abs_max = _env_f("SIM_MARGIN_MEAN_ABS_MAX", 20.0)
        if np.isfinite(m_mu):
            m_anchor = -float(s) if np.isfinite(s) else 0.0
            if delta_m_max > 0:
                m_mu = float(m_anchor) + float(np.clip(m_mu - float(m_anchor), -delta_m_max, delta_m_max))
            m_mu = float(np.clip(m_mu, -m_abs_max, m_abs_max))
    except Exception:
        pass

    return (m_mu, t_mu, s, t_line)


def compute_margin_total_draws(
    view_df: pd.DataFrame,
    n_sims: int = 2000,
    ats_sigma_override: float | None = None,
    total_sigma_override: float | None = None,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute and return per-game (margins, totals) draws so all artifacts can be aligned."""
    if view_df is None or view_df.empty:
        return {}
    df = view_df.copy()
    spread_ref, total_ref = _spread_total_refs(df)
    df["spread_ref"] = spread_ref
    df["total_ref"] = total_ref

    sigmas = _load_sigma_calibration(data_dir=data_dir)
    ats_sigma = float(sigmas.get("ats_sigma", 12.0))
    total_sigma = float(sigmas.get("total_sigma", 11.0))
    if isinstance(ats_sigma_override, (int, float)) and float(ats_sigma_override) > 0:
        ats_sigma = float(ats_sigma_override)
    if isinstance(total_sigma_override, (int, float)) and float(total_sigma_override) > 0:
        total_sigma = float(total_sigma_override)
    totals_cal = _load_totals_calibration(data_dir=data_dir)

    blend_margin = _env_f("SIM_MARKET_BLEND_MARGIN", 0.0)
    blend_total = _env_f("SIM_MARKET_BLEND_TOTAL", 0.0)

    k_mean_total_wind = _env_f("SIM_MEAN_TOTAL_K_WIND", -0.5)
    k_mean_margin_rest = _env_f("SIM_MEAN_MARGIN_K_REST", 0.15)
    k_mean_margin_elo = _env_f("SIM_MEAN_MARGIN_K_ELO", 0.08)
    k_mean_total_inj = _env_f("SIM_MEAN_TOTAL_K_INJ", -0.4)
    k_mean_total_defppg = _env_f("SIM_MEAN_TOTAL_K_DEFPPG", -0.4)
    k_mean_total_pressure = _env_f("SIM_MEAN_TOTAL_K_PRESSURE", -0.8)
    pressure_baseline = _env_f("SIM_PRESSURE_BASELINE", 0.065)
    k_mean_margin_rating = _env_f("SIM_MEAN_MARGIN_K_RATING", 0.08)
    k_mean_margin_neutral = _env_f("SIM_MEAN_MARGIN_K_NEUTRAL", 0.0)
    k_mean_total_precip = _env_f("SIM_MEAN_TOTAL_K_PRECIP", 0.0)
    k_mean_total_cold = _env_f("SIM_MEAN_TOTAL_K_COLD", 0.0)
    cold_threshold_f = _env_f("SIM_COLD_TEMP_F", 45.0)

    rho = _env_f("SIM_CORR_MARGIN_TOTAL", 0.10)
    rng = np.random.default_rng(seed)

    # Deterministic ordering: sort by game_id string when present.
    if "game_id" in df.columns:
        try:
            df = df.assign(_gid_sort=df["game_id"].astype(str)).sort_values(["_gid_sort"]).drop(columns=["_gid_sort"])
        except Exception:
            pass

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx in df.index:
        row = df.loc[idx]
        gid = row.get("game_id")
        if gid is None:
            continue
        gid_s = str(gid)

        m_mu, t_mu, s, t_line = _compute_game_means_and_lines(
            row,
            totals_cal=totals_cal,
            blend_margin=blend_margin,
            blend_total=blend_total,
            k_mean_total_wind=k_mean_total_wind,
            k_mean_margin_rest=k_mean_margin_rest,
            k_mean_margin_elo=k_mean_margin_elo,
            k_mean_total_inj=k_mean_total_inj,
            k_mean_total_defppg=k_mean_total_defppg,
            k_mean_total_pressure=k_mean_total_pressure,
            pressure_baseline=pressure_baseline,
            k_mean_margin_rating=k_mean_margin_rating,
            k_mean_margin_neutral=k_mean_margin_neutral,
            k_mean_total_precip=k_mean_total_precip,
            k_mean_total_cold=k_mean_total_cold,
            cold_threshold_f=cold_threshold_f,
        )
        if not np.isfinite(m_mu) and not np.isfinite(t_mu):
            continue

        eff_ats_sigma, eff_total_sigma = _per_game_sigmas(row, ats_sigma=ats_sigma, total_sigma=total_sigma)

        margins = None
        totals = None
        try:
            if np.isfinite(m_mu) and np.isfinite(t_mu):
                cov = np.array(
                    [
                        [eff_ats_sigma**2, float(rho) * eff_ats_sigma * eff_total_sigma],
                        [float(rho) * eff_ats_sigma * eff_total_sigma, eff_total_sigma**2],
                    ],
                    dtype=float,
                )
                draws = rng.multivariate_normal(mean=np.array([m_mu, t_mu], dtype=float), cov=cov, size=int(n_sims))
                margins = draws[:, 0]
                totals = draws[:, 1]
        except Exception:
            margins = None
            totals = None
        if margins is None:
            margins = rng.normal(loc=m_mu if np.isfinite(m_mu) else 0.0, scale=eff_ats_sigma, size=int(n_sims))
        if totals is None:
            totals = rng.normal(loc=t_mu if np.isfinite(t_mu) else 0.0, scale=eff_total_sigma, size=int(n_sims))

        out[gid_s] = (np.array(margins, dtype=float), np.array(totals, dtype=float))
    return out


def simulate_mc_probs(
    view_df: pd.DataFrame,
    n_sims: int = 2000,
    ats_sigma_override: float | None = None,
    total_sigma_override: float | None = None,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
    draws_by_game: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Compute per-game Monte Carlo probabilities from mean margin/total and optional features.

    Expected columns (best effort; all optional):
    - Keys: season, week, game_id, home_team, away_team
    - Means: pred_margin, pred_total (or pred_total_cal; or pred_home_points/pred_away_points)
    - Market: spread_home/close_spread_home, total/close_total
    - Features used for mean/sigma tweaks: roof_closed_flag, wind_open, rest_days_diff, elo_diff,
      home_inj_starters_out, away_inj_starters_out, home_def_ppg, away_def_ppg,
      home_def_sack_rate(_ema), away_def_sack_rate(_ema), net_margin_diff,
      neutral_site_flag, wx_precip_pct, wx_temp_f

    Returns a DataFrame matching the app's sim_probs.csv schema.
    """
    if view_df is None or view_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "home_team",
                "away_team",
                "pred_margin",
                "pred_total",
                "spread_ref",
                "total_ref",
                "prob_home_win_mc",
                "prob_home_cover_mc",
                "prob_over_total_mc",
            ]
        )

    df = view_df.copy()

    spread_ref, total_ref = _spread_total_refs(df)
    df["spread_ref"] = spread_ref
    df["total_ref"] = total_ref

    # Load calibration artifacts
    sigmas = _load_sigma_calibration(data_dir=data_dir)
    ats_sigma = float(sigmas.get("ats_sigma", 12.0))
    total_sigma = float(sigmas.get("total_sigma", 11.0))
    if isinstance(ats_sigma_override, (int, float)) and float(ats_sigma_override) > 0:
        ats_sigma = float(ats_sigma_override)
    if isinstance(total_sigma_override, (int, float)) and float(total_sigma_override) > 0:
        total_sigma = float(total_sigma_override)
    totals_cal = _load_totals_calibration(data_dir=data_dir)

    # Mean blending knobs
    blend_margin = _env_f("SIM_MARKET_BLEND_MARGIN", 0.0)
    blend_total = _env_f("SIM_MARKET_BLEND_TOTAL", 0.0)

    # Feature mean knobs (existing)
    k_mean_total_wind = _env_f("SIM_MEAN_TOTAL_K_WIND", -0.5)  # points per 10mph wind_open
    k_mean_margin_rest = _env_f("SIM_MEAN_MARGIN_K_REST", 0.15)  # points per 7-day rest diff
    k_mean_margin_elo = _env_f("SIM_MEAN_MARGIN_K_ELO", 0.08)  # points per 100 Elo diff
    k_mean_total_inj = _env_f("SIM_MEAN_TOTAL_K_INJ", -0.4)  # points per starter out
    k_mean_total_defppg = _env_f("SIM_MEAN_TOTAL_K_DEFPPG", -0.4)  # points per 5 PPG above/below league avg
    k_mean_total_pressure = _env_f("SIM_MEAN_TOTAL_K_PRESSURE", -0.8)  # points per 0.05 above baseline
    pressure_baseline = _env_f("SIM_PRESSURE_BASELINE", 0.065)
    k_mean_margin_rating = _env_f("SIM_MEAN_MARGIN_K_RATING", 0.08)  # points per 1.0 net margin rating diff

    # Optional mean knobs (new, default 0.0 to keep current behavior)
    k_mean_margin_neutral = _env_f("SIM_MEAN_MARGIN_K_NEUTRAL", 0.0)  # points to add when neutral-site flag true
    k_mean_total_precip = _env_f("SIM_MEAN_TOTAL_K_PRECIP", 0.0)  # points per 100% precip
    k_mean_total_cold = _env_f("SIM_MEAN_TOTAL_K_COLD", 0.0)  # points per 10F below cold_threshold
    cold_threshold_f = _env_f("SIM_COLD_TEMP_F", 45.0)

    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for idx in df.index:
        row = df.loc[idx]
        season_i = int(_coerce_float(row.get("season"))) if "season" in df.columns else None
        week_i = int(_coerce_float(row.get("week"))) if "week" in df.columns else None
        game_id = row.get("game_id")
        home_team = row.get("home_team")
        away_team = row.get("away_team")

        m_mu, t_mu, s, t_line = _compute_game_means_and_lines(
            row,
            totals_cal=totals_cal,
            blend_margin=blend_margin,
            blend_total=blend_total,
            k_mean_total_wind=k_mean_total_wind,
            k_mean_margin_rest=k_mean_margin_rest,
            k_mean_margin_elo=k_mean_margin_elo,
            k_mean_total_inj=k_mean_total_inj,
            k_mean_total_defppg=k_mean_total_defppg,
            k_mean_total_pressure=k_mean_total_pressure,
            pressure_baseline=pressure_baseline,
            k_mean_margin_rating=k_mean_margin_rating,
            k_mean_margin_neutral=k_mean_margin_neutral,
            k_mean_total_precip=k_mean_total_precip,
            k_mean_total_cold=k_mean_total_cold,
            cold_threshold_f=cold_threshold_f,
        )

        if not np.isfinite(m_mu) and not np.isfinite(t_mu):
            continue
        eff_ats_sigma, eff_total_sigma = _per_game_sigmas(row, ats_sigma=ats_sigma, total_sigma=total_sigma)
        rho = _env_f("SIM_CORR_MARGIN_TOTAL", 0.10)

        margins = None
        totals = None
        gid_s = str(game_id) if game_id is not None else None
        if draws_by_game is not None and gid_s and gid_s in draws_by_game:
            try:
                dm, dt = draws_by_game[gid_s]
                if hasattr(dm, "__len__") and hasattr(dt, "__len__") and len(dm) == int(n_sims) and len(dt) == int(n_sims):
                    margins = np.array(dm, dtype=float)
                    totals = np.array(dt, dtype=float)
            except Exception:
                margins = None
                totals = None
        try:
            if np.isfinite(m_mu) and np.isfinite(t_mu):
                cov = np.array(
                    [
                        [eff_ats_sigma**2, float(rho) * eff_ats_sigma * eff_total_sigma],
                        [float(rho) * eff_ats_sigma * eff_total_sigma, eff_total_sigma**2],
                    ],
                    dtype=float,
                )
                if margins is None or totals is None:
                    draws = rng.multivariate_normal(mean=np.array([m_mu, t_mu], dtype=float), cov=cov, size=int(n_sims))
                    margins = draws[:, 0]
                    totals = draws[:, 1]
        except Exception:
            margins = None
            totals = None

        if margins is None:
            margins = rng.normal(loc=m_mu if np.isfinite(m_mu) else 0.0, scale=eff_ats_sigma, size=int(n_sims))
        if totals is None:
            totals = rng.normal(loc=t_mu if np.isfinite(t_mu) else 0.0, scale=eff_total_sigma, size=int(n_sims))

        p_ml = float(np.mean(margins > 0.0)) if np.isfinite(m_mu) else np.nan
        p_ats = float(np.mean(margins + s > 0.0)) if np.isfinite(s) and np.isfinite(m_mu) else np.nan
        p_over = float(np.mean(totals > t_line)) if np.isfinite(t_line) and np.isfinite(t_mu) else np.nan

        # Derived team points means (useful for aligning player-props and other downstream artifacts).
        try:
            home_pts = 0.5 * (totals + margins)
            away_pts = totals - home_pts
            home_points_mean = float(np.mean(home_pts))
            away_points_mean = float(np.mean(away_pts))
            total_points_mean = float(np.mean(totals))
        except Exception:
            home_points_mean = np.nan
            away_points_mean = np.nan
            total_points_mean = np.nan

        rows.append(
            {
                "season": season_i,
                "week": week_i,
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "pred_margin": float(m_mu) if np.isfinite(m_mu) else np.nan,
                "pred_total": float(t_mu) if np.isfinite(t_mu) else np.nan,
                "spread_ref": float(s) if np.isfinite(s) else np.nan,
                "total_ref": float(t_line) if np.isfinite(t_line) else np.nan,
                "home_points_mean": home_points_mean,
                "away_points_mean": away_points_mean,
                "total_points_mean": total_points_mean,
                "prob_home_win_mc": p_ml,
                "prob_home_cover_mc": p_ats,
                "prob_over_total_mc": p_over,
            }
        )

    return pd.DataFrame(rows)


def _quarter_weights_for_team(is_winner: bool) -> np.ndarray:
    """Heuristic quarter scoring weights.

    Winners skew a bit toward Q2; losers skew toward Q4 (garbage-time / comeback).
    """
    w = np.array([0.23, 0.27, 0.23, 0.27], dtype=float)
    if is_winner:
        w = w + np.array([0.01, 0.03, 0.00, -0.04], dtype=float)
    else:
        w = w + np.array([-0.04, 0.00, 0.00, 0.04], dtype=float)
    w = np.clip(w, 0.05, None)
    w = w / float(w.sum())
    return w


def _decompose_points_to_scoring_plays(points: int, rng: np.random.Generator) -> list[int]:
    """Decompose a team's final points into a list of scoring play point-values.

    This is a best-effort football-ish decomposition using TD(6/7/8), FG(3), safety(2).
    It guarantees sum(plays) == points for non-negative points.
    """
    p = int(points)
    if p <= 0:
        return []

    # Fast path for small values
    if p in (2, 3, 6, 7, 8):
        return [p]

    # Try stochastic + constrained search for a plausible mix.
    # Target ~55-65% of points from TDs.
    mean_tds = max(0.0, (p * float(rng.uniform(0.55, 0.65))) / 7.0)
    td_cap = max(0, p // 6)

    for _ in range(200):
        tds = int(min(td_cap, rng.poisson(mean_tds)))
        tds = max(0, tds)

        for td_delta in (0, -1, 1, -2, 2, -3, 3):
            t = tds + int(td_delta)
            if t < 0 or t > td_cap:
                continue
            rem_after_tds = p - 7 * t
            if rem_after_tds < 0:
                continue
            fg_guess = int(round(rem_after_tds / 3.0))
            fg_guess = int(np.clip(fg_guess, 0, rem_after_tds // 3))

            for fg_delta in (0, -1, 1, -2, 2, -3, 3):
                f = fg_guess + int(fg_delta)
                if f < 0:
                    continue
                base = 7 * t + 3 * f
                if base > p:
                    continue
                leftover = p - base
                if leftover > 8:
                    continue

                # Solve: base + (x - y) + 2*s = p
                # x = number of TDs upgraded to 8 (adds +1)
                # y = number of TDs downgraded to 6 (adds -1)
                # s = safeties (adds +2)
                for s in range(0, 5):
                    for x in range(0, t + 1):
                        for y in range(0, t - x + 1):
                            if base + x - y + 2 * s != p:
                                continue
                            plays: list[int] = [7] * t + [3] * f + [2] * s
                            # Apply TD conversions.
                            if x + y > 0 and t > 0:
                                td_idxs = list(range(t))
                                rng.shuffle(td_idxs)
                                for j in td_idxs[:x]:
                                    plays[j] = 8
                                for j in td_idxs[x : x + y]:
                                    plays[j] = 6
                            rng.shuffle(plays)
                            return plays

    # Deterministic fallback: find any solution with small brute force.
    for t in range(0, td_cap + 1):
        for f in range(0, (p // 3) + 1):
            base = 7 * t + 3 * f
            if base > p:
                continue
            leftover = p - base
            for s in range(0, 6):
                for x in range(0, t + 1):
                    for y in range(0, t - x + 1):
                        if base + x - y + 2 * s != p:
                            continue
                        plays = [7] * t + [3] * f + [2] * s
                        if x + y > 0 and t > 0:
                            td_idxs = list(range(t))
                            rng.shuffle(td_idxs)
                            for j in td_idxs[:x]:
                                plays[j] = 8
                            for j in td_idxs[x : x + y]:
                                plays[j] = 6
                        rng.shuffle(plays)
                        return plays
            # leftover unused; continue

    # Last resort: make it sum correctly even if odd.
    plays = []
    remaining = p
    while remaining >= 7:
        plays.append(7)
        remaining -= 7
    while remaining >= 3:
        plays.append(3)
        remaining -= 3
    if remaining > 0:
        plays.append(remaining)
    return plays


def _allocate_plays_to_quarters(plays: list[int], weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    if not plays:
        return q
    w = np.array(weights, dtype=float)
    w = w / float(w.sum())
    for pts in plays:
        qi = int(rng.choice(4, p=w))
        q[qi] += float(pts)
    return q


def _bucket_plays_to_quarters(plays: list[int], weights: np.ndarray, rng: np.random.Generator) -> list[list[int]]:
    """Assign each scoring play to a quarter, returning a list-of-lists."""
    out: list[list[int]] = [[], [], [], []]
    if not plays:
        return out
    w = np.array(weights, dtype=float)
    w = w / float(w.sum())
    for pts in plays:
        qi = int(rng.choice(4, p=w))
        out[qi].append(int(pts))
    return out


def _split_count_across_quarters(n: int) -> list[int]:
    """Deterministically split a count across 4 quarters, preserving sum."""
    n = int(max(0, n))
    if n == 0:
        return [0, 0, 0, 0]
    base = [n // 4] * 4
    rem = n - sum(base)
    # Bias extra possessions slightly to Q2/Q4 (more end-of-half drives)
    order = [1, 3, 0, 2]
    for i in range(rem):
        base[order[i % 4]] += 1
    return base


def _allocate_quarter_plays_to_drives(
    plays_by_q: list[list[int]],
    drives_by_q: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Allocate scoring plays within each quarter to specific drives.

    Returns a list of per-drive points in quarter order.
    """
    drive_points: list[int] = []
    for qi in range(4):
        nd = int(max(0, drives_by_q[qi] if qi < len(drives_by_q) else 0))
        # Ensure we always have a place to put scoring plays.
        if nd <= 0:
            nd = 1
        pts = [0] * nd
        plays = plays_by_q[qi] if qi < len(plays_by_q) else []
        for p in plays:
            di = int(rng.integers(0, nd))
            pts[di] += int(p)
        drive_points.extend(pts)
    return drive_points


def _valid_team_scores(max_points: int = 90) -> set[int]:
    """Return the set of team point totals reachable by {2,3,6,7,8} scoring plays."""
    max_points = int(max(0, max_points))
    allowed = (2, 3, 6, 7, 8)
    reachable = {0}
    for p in range(1, max_points + 1):
        ok = False
        for a in allowed:
            if p - a >= 0 and (p - a) in reachable:
                ok = True
                break
        if ok:
            reachable.add(p)
    return reachable


def _valid_game_totals(max_total: int = 90) -> set[int]:
    team = _valid_team_scores(max_points=max_total)
    totals: set[int] = set()
    for h in team:
        for a in team:
            s = h + a
            if 0 <= s <= int(max_total):
                totals.add(int(s))
    return totals


def _nearest_in_set(x: int, vals: set[int]) -> int:
    if not vals:
        return int(x)
    if int(x) in vals:
        return int(x)
    # Deterministic nearest; ties -> smaller.
    best = None
    best_d = None
    xi = int(x)
    for v in vals:
        d = abs(int(v) - xi)
        if best is None or d < best_d or (d == best_d and int(v) < int(best)):
            best = int(v)
            best_d = int(d)
    return int(best) if best is not None else int(x)


def _make_valid_score_pair(total: int, home_guess: int, away_guess: int, valid_team: set[int]) -> tuple[int, int]:
    """Return (home, away) such that home+away==total and both are valid team scores.

    Chooses the pair closest to the provided guesses.
    """
    t = int(total)
    hg = int(home_guess)
    ag = int(away_guess)
    best_pair: tuple[int, int] | None = None
    best_cost: tuple[int, int] | None = None
    for h in valid_team:
        if h < 0 or h > t:
            continue
        a = t - int(h)
        if a not in valid_team:
            continue
        # Primary: minimize deviation from guessed scores.
        dev = abs(int(h) - hg) + abs(int(a) - ag)
        # Secondary: minimize deviation from guessed margin.
        dev_m = abs((int(h) - int(a)) - (hg - ag))
        cost = (int(dev), int(dev_m))
        if best_pair is None or cost < best_cost:
            best_pair = (int(h), int(a))
            best_cost = cost
    if best_pair is not None:
        return best_pair
    # Fallback: clamp into a valid team score and preserve total.
    h2 = _nearest_in_set(hg, valid_team)
    h2 = int(max(0, min(t, h2)))
    a2 = int(t - h2)
    if a2 not in valid_team:
        a2 = _nearest_in_set(a2, valid_team)
        a2 = int(max(0, min(t, a2)))
        h2 = int(t - a2)
    return (int(h2), int(a2))


def _force_realistic_final_scores(total_i: np.ndarray, home_i: np.ndarray, away_i: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjust integer final scores so both teams and game total are reachable by football scoring plays."""
    if total_i is None or home_i is None or away_i is None:
        return total_i, home_i, away_i
    try:
        max_total = int(np.nanmax(total_i)) if len(total_i) else 90
    except Exception:
        max_total = 90
    max_total = int(max(0, min(120, max_total)))
    valid_team = _valid_team_scores(max_points=max_total)
    valid_tot = _valid_game_totals(max_total=max_total)

    t_out = np.array(total_i, dtype=int, copy=True)
    h_out = np.array(home_i, dtype=int, copy=True)
    a_out = np.array(away_i, dtype=int, copy=True)
    for j in range(len(t_out)):
        t = int(t_out[j])
        h = int(h_out[j])
        a = int(a_out[j])
        # Ensure non-negative and consistent sum.
        t = int(max(0, t))
        h = int(max(0, h))
        a = int(max(0, a))
        if h + a != t:
            # Preserve total; adjust away.
            a = int(max(0, t - h))
        # Snap total to a reachable total if needed (rare tail case).
        if t not in valid_tot:
            t = _nearest_in_set(t, valid_tot)
        # Find best reachable (home, away) pair.
        h2, a2 = _make_valid_score_pair(t, h, a, valid_team)
        t_out[j] = int(t)
        h_out[j] = int(h2)
        a_out[j] = int(a2)
    return t_out, h_out, a_out


def _interleave_possessions(home_drives: int, away_drives: int, start_home: bool) -> list[str]:
    """Return a possession sequence like ['home','away',...] of total length home_drives+away_drives."""
    seq: list[str] = []
    h = int(max(0, home_drives))
    a = int(max(0, away_drives))
    turn = "home" if start_home else "away"
    while h > 0 or a > 0:
        if turn == "home":
            if h > 0:
                seq.append("home")
                h -= 1
            turn = "away"
        else:
            if a > 0:
                seq.append("away")
                a -= 1
            turn = "home"
        # If one side is exhausted, dump the rest for the other.
        if h == 0 and a > 0:
            seq.extend(["away"] * a)
            break
        if a == 0 and h > 0:
            seq.extend(["home"] * h)
            break
    return seq


def simulate_drive_timeline(
    view_df: pd.DataFrame,
    props_df: Optional[pd.DataFrame] = None,
    n_sims: int = 2000,
    ats_sigma_override: float | None = None,
    total_sigma_override: float | None = None,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
    draws_by_game: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Simulate a drive-level game timeline.

    Output rows per game per drive with cumulative expected score and lead probabilities.

    Notes / philosophy:
    - This is a "football-ish" possession model, not a full play-by-play engine.
    - It uses the same margin/total draws as simulate_quarter_means, then allocates scoring plays
      to quarters and then to drives.
    - Drive counts are estimated from player props cache team_plays when available.
    """
    if view_df is None or view_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "home_team",
                "away_team",
                "drive_no",
                "quarter",
                "poss_side",
                "poss_team",
                "drive_sec_mean",
                "drive_pts_mean",
                "p_drive_score",
                "p_drive_fg",
                "p_drive_td",
                "drive_outcome_mode",
                "home_score_mean",
                "away_score_mean",
                "home_score_drive_mean",
                "away_score_drive_mean",
                "p_home_lead",
                "p_tie",
                "p_away_lead",
                "drives_home",
                "drives_away",
                "drives_total",
            ]
        )

    df = view_df.copy()
    spread_ref, total_ref = _spread_total_refs(df)
    df["spread_ref"] = spread_ref
    df["total_ref"] = total_ref

    sigmas = _load_sigma_calibration(data_dir=data_dir)
    ats_sigma = float(sigmas.get("ats_sigma", 12.0))
    total_sigma = float(sigmas.get("total_sigma", 11.0))
    if isinstance(ats_sigma_override, (int, float)) and float(ats_sigma_override) > 0:
        ats_sigma = float(ats_sigma_override)
    if isinstance(total_sigma_override, (int, float)) and float(total_sigma_override) > 0:
        total_sigma = float(total_sigma_override)
    totals_cal = _load_totals_calibration(data_dir=data_dir)

    blend_margin = _env_f("SIM_MARKET_BLEND_MARGIN", 0.0)
    blend_total = _env_f("SIM_MARKET_BLEND_TOTAL", 0.0)

    # Drive model knobs
    plays_per_drive = _env_f("SIM_PLAYS_PER_DRIVE", 6.2)
    seconds_per_play = _env_f("SIM_SECONDS_PER_PLAY", 26.0)
    seconds_per_play_sigma = _env_f("SIM_SECONDS_PER_PLAY_SIGMA", 4.0)
    # Drive-level duration multiplier to introduce realistic variation (penalties, tempo, stoppages).
    # Use lognormal so we get a right tail (some long drives) without negative durations.
    drive_time_mult_sigma = _env_f("SIM_DRIVE_TIME_MULT_SIGMA", 0.45)
    drive_time_cap_sec = _env_f("SIM_DRIVE_TIME_CAP_SEC", 780.0)
    drive_extra_plays_td = _env_f("SIM_DRIVE_EXTRA_PLAYS_TD", 2.0)
    drive_extra_plays_fg = _env_f("SIM_DRIVE_EXTRA_PLAYS_FG", 1.0)

    # Outcome mix for non-scoring drives (simple priors; should sum to 1.0 after normalization).
    # These are coarse, league-ish rates and intentionally lightweight.
    p_ns_punt = _env_f("SIM_DRIVE_P_NS_PUNT", 0.62)
    p_ns_int = _env_f("SIM_DRIVE_P_NS_INT", 0.08)
    p_ns_fumble = _env_f("SIM_DRIVE_P_NS_FUMBLE", 0.03)
    p_ns_downs = _env_f("SIM_DRIVE_P_NS_DOWNS", 0.05)
    p_ns_missed_fg = _env_f("SIM_DRIVE_P_NS_MISSED_FG", 0.07)
    p_ns_end_half = _env_f("SIM_DRIVE_P_NS_END_HALF", 0.15)

    def _norm_probs(ps: list[float]) -> list[float]:
        try:
            arr = np.array([max(0.0, float(x)) for x in ps], dtype=float)
            s = float(np.sum(arr))
            if s <= 0:
                return [1.0 / len(ps)] * len(ps)
            return [float(x / s) for x in arr]
        except Exception:
            return [1.0 / len(ps)] * len(ps)

    # Normalized and cumulative for sampling.
    _ns_names = ["Punt", "INT", "Fumble", "Downs", "Missed FG", "End Half"]
    _ns_ps = _norm_probs([p_ns_punt, p_ns_int, p_ns_fumble, p_ns_downs, p_ns_missed_fg, p_ns_end_half])
    _ns_cum = np.cumsum(np.array(_ns_ps, dtype=float))
    drives_default = int(_env_f("SIM_DRIVES_PER_TEAM_DEFAULT", 11.0))
    drives_min = int(_env_f("SIM_DRIVES_PER_TEAM_MIN", 8.0))
    drives_max = int(_env_f("SIM_DRIVES_PER_TEAM_MAX", 16.0))

    # Feature mean knobs (match quarter sim)
    k_mean_total_wind = _env_f("SIM_MEAN_TOTAL_K_WIND", -0.5)
    k_mean_margin_rest = _env_f("SIM_MEAN_MARGIN_K_REST", 0.15)
    k_mean_margin_elo = _env_f("SIM_MEAN_MARGIN_K_ELO", 0.08)
    k_mean_total_inj = _env_f("SIM_MEAN_TOTAL_K_INJ", -0.4)
    k_mean_total_defppg = _env_f("SIM_MEAN_TOTAL_K_DEFPPG", -0.4)
    k_mean_total_pressure = _env_f("SIM_MEAN_TOTAL_K_PRESSURE", -0.8)
    pressure_baseline = _env_f("SIM_PRESSURE_BASELINE", 0.065)
    k_mean_margin_rating = _env_f("SIM_MEAN_MARGIN_K_RATING", 0.08)
    k_mean_margin_neutral = _env_f("SIM_MEAN_MARGIN_K_NEUTRAL", 0.0)
    k_mean_total_precip = _env_f("SIM_MEAN_TOTAL_K_PRECIP", 0.0)
    k_mean_total_cold = _env_f("SIM_MEAN_TOTAL_K_COLD", 0.0)
    cold_threshold_f = _env_f("SIM_COLD_TEMP_F", 45.0)

    # Optional: map (game_id, team) -> team_plays for drive count estimation
    team_plays_map: dict[tuple[str, str], float] = {}
    if props_df is not None and not props_df.empty:
        try:
            pdf = props_df.copy()
            if "game_id" in pdf.columns and "team" in pdf.columns and "team_plays" in pdf.columns:
                pdf["team_plays"] = pd.to_numeric(pdf["team_plays"], errors="coerce")
                grp = pdf.dropna(subset=["game_id", "team"]).groupby(["game_id", "team"], dropna=False)["team_plays"].mean()
                for (gid, tm), v in grp.items():
                    try:
                        if pd.notna(v):
                            team_plays_map[(str(gid), str(tm))] = float(v)
                    except Exception:
                        continue
        except Exception:
            team_plays_map = {}

    rng = np.random.default_rng(seed)
    rho = _env_f("SIM_CORR_MARGIN_TOTAL", 0.10)

    out_rows: list[dict[str, Any]] = []
    for idx in df.index:
        row = df.loc[idx]
        season_i = int(_coerce_float(row.get("season"))) if "season" in df.columns else None
        week_i = int(_coerce_float(row.get("week"))) if "week" in df.columns else None
        game_id = row.get("game_id")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        gid_s = str(game_id) if game_id is not None else ""

        # Estimate drives per team from team_plays if available
        def _est_drives(team_name: Any) -> int:
            try:
                tp = team_plays_map.get((gid_s, str(team_name)))
                if tp is None or not np.isfinite(tp) or plays_per_drive <= 0:
                    return int(drives_default)
                d = int(np.clip(int(round(float(tp) / float(plays_per_drive))), drives_min, drives_max))
                return d
            except Exception:
                return int(drives_default)

        drives_home = _est_drives(home_team)
        drives_away = _est_drives(away_team)
        drives_total = int(drives_home + drives_away)
        if drives_total <= 0:
            continue

        # Means (same fallbacks as simulate_quarter_means)
        m_mu = _coerce_float(row.get("pred_margin")) if "pred_margin" in df.columns else float("nan")
        if not np.isfinite(m_mu):
            ph = _coerce_float(row.get("pred_home_points"))
            pa = _coerce_float(row.get("pred_away_points"))
            if np.isfinite(ph) and np.isfinite(pa):
                m_mu = float(ph - pa)
        if not np.isfinite(m_mu):
            sref = _coerce_float(row.get("spread_ref"))
            if np.isfinite(sref):
                m_mu = -float(sref)

        t_mu = float("nan")
        if "pred_total_cal" in df.columns:
            t_mu = _coerce_float(row.get("pred_total_cal"))
        if not np.isfinite(t_mu) and "pred_total" in df.columns:
            t_mu = _coerce_float(row.get("pred_total"))
        if not np.isfinite(t_mu):
            tref = _coerce_float(row.get("total_ref"))
            if np.isfinite(tref):
                t_mu = float(tref)

        s = _coerce_float(row.get("spread_ref"))
        t_line = _coerce_float(row.get("total_ref"))
        if not np.isfinite(m_mu) and not np.isfinite(t_mu):
            continue

        # Totals calibration
        scale = float(totals_cal.get("scale", np.nan))
        shift = float(totals_cal.get("shift", np.nan))
        if np.isfinite(t_mu) and np.isfinite(scale) and np.isfinite(shift):
            t_mu = scale * t_mu + shift

        # Market blends
        if np.isfinite(m_mu) and np.isfinite(s) and 0.0 < float(blend_margin) <= 1.0:
            m_mu = (1.0 - float(blend_margin)) * float(m_mu) + float(blend_margin) * (-float(s))
        if np.isfinite(t_mu) and np.isfinite(t_line) and 0.0 < float(blend_total) <= 1.0:
            t_mu = (1.0 - float(blend_total)) * float(t_mu) + float(blend_total) * float(t_line)

        # Feature adjustments (same as quarter sim)
        w_open = _coerce_float(row.get("wind_open"))
        if np.isfinite(t_mu) and np.isfinite(w_open) and w_open > 0:
            t_mu = float(t_mu) + float(k_mean_total_wind) * (w_open / 10.0)
        rest_diff = _coerce_float(row.get("rest_days_diff"))
        if np.isfinite(m_mu) and np.isfinite(rest_diff) and rest_diff != 0:
            m_mu = float(m_mu) + float(k_mean_margin_rest) * (rest_diff / 7.0)
        elo_diff = _coerce_float(row.get("elo_diff"))
        if np.isfinite(m_mu) and np.isfinite(elo_diff):
            m_mu = float(m_mu) + float(k_mean_margin_elo) * (elo_diff / 100.0)
        neutral_flag = _coerce_float(row.get("neutral_site_flag"))
        if np.isfinite(neutral_flag) and neutral_flag >= 0.5 and np.isfinite(m_mu):
            m_mu = float(m_mu) + float(k_mean_margin_neutral)
        h_inj = _coerce_float(row.get("home_inj_starters_out"))
        a_inj = _coerce_float(row.get("away_inj_starters_out"))
        total_out = 0.0
        for v in (h_inj, a_inj):
            if np.isfinite(v):
                total_out += float(v)
        if np.isfinite(t_mu) and total_out > 0:
            t_mu = float(t_mu) + float(k_mean_total_inj) * float(total_out)
        h_def_ppg = _coerce_float(row.get("home_def_ppg"))
        a_def_ppg = _coerce_float(row.get("away_def_ppg"))
        vals = [v for v in (h_def_ppg, a_def_ppg) if np.isfinite(v)]
        def_avg = float(np.mean(vals)) if vals else float("nan")
        if not np.isfinite(def_avg):
            def_diff = _coerce_float(row.get("def_ppg_diff"))
            if np.isfinite(def_diff):
                def_avg = 22.0 + (def_diff / 2.0)
        if np.isfinite(t_mu) and np.isfinite(def_avg):
            t_mu = float(t_mu) + float(k_mean_total_defppg) * ((def_avg - 22.0) / 5.0)
        h_def_sr = _coerce_float(row.get("home_def_sack_rate_ema"))
        if not np.isfinite(h_def_sr):
            h_def_sr = _coerce_float(row.get("home_def_sack_rate"))
        a_def_sr = _coerce_float(row.get("away_def_sack_rate_ema"))
        if not np.isfinite(a_def_sr):
            a_def_sr = _coerce_float(row.get("away_def_sack_rate"))
        sr_vals = [v for v in (h_def_sr, a_def_sr) if np.isfinite(v)]
        if sr_vals and np.isfinite(t_mu):
            sr_avg = float(np.mean(sr_vals))
            delta_units = (sr_avg - float(pressure_baseline)) / 0.05
            t_mu = float(t_mu) + float(k_mean_total_pressure) * float(delta_units)
        nm_diff = _coerce_float(row.get("net_margin_diff"))
        if np.isfinite(m_mu) and np.isfinite(nm_diff):
            m_mu = float(m_mu) + float(k_mean_margin_rating) * float(nm_diff)
        precip = _coerce_float(row.get("wx_precip_pct"))
        if np.isfinite(t_mu) and np.isfinite(precip) and precip > 0:
            t_mu = float(t_mu) + float(k_mean_total_precip) * (precip / 100.0)
        temp_f = _coerce_float(row.get("wx_temp_f"))
        if np.isfinite(t_mu) and np.isfinite(temp_f) and temp_f < float(cold_threshold_f):
            units = (float(cold_threshold_f) - float(temp_f)) / 10.0
            t_mu = float(t_mu) + float(k_mean_total_cold) * float(units)

        # Realism clamps
        delta_t_max = _env_f("SIM_TOTAL_DELTA_MAX", 10.0)
        t_min = _env_f("SIM_TOTAL_MEAN_MIN", 30.0)
        t_max = _env_f("SIM_TOTAL_MEAN_MAX", 62.0)
        if np.isfinite(t_mu):
            if np.isfinite(t_line) and delta_t_max > 0:
                t_mu = float(t_line) + float(np.clip(t_mu - float(t_line), -delta_t_max, delta_t_max))
            t_mu = float(np.clip(t_mu, t_min, t_max))
        delta_m_max = _env_f("SIM_MARGIN_DELTA_MAX", 10.0)
        m_abs_max = _env_f("SIM_MARGIN_MEAN_ABS_MAX", 20.0)
        if np.isfinite(m_mu):
            m_anchor = -float(s) if np.isfinite(s) else 0.0
            if delta_m_max > 0:
                m_mu = float(m_anchor) + float(np.clip(m_mu - float(m_anchor), -delta_m_max, delta_m_max))
            m_mu = float(np.clip(m_mu, -m_abs_max, m_abs_max))

        eff_ats_sigma, eff_total_sigma = _per_game_sigmas(row, ats_sigma=ats_sigma, total_sigma=total_sigma)
        if np.isfinite(m_mu) and np.isfinite(t_mu):
            cov = np.array(
                [
                    [eff_ats_sigma**2, float(rho) * eff_ats_sigma * eff_total_sigma],
                    [float(rho) * eff_ats_sigma * eff_total_sigma, eff_total_sigma**2],
                ],
                dtype=float,
            )
            gid_s2 = str(game_id) if game_id is not None else None
            if draws_by_game is not None and gid_s2 and gid_s2 in draws_by_game:
                try:
                    dm, dt = draws_by_game[gid_s2]
                    if hasattr(dm, "__len__") and hasattr(dt, "__len__") and len(dm) == int(n_sims) and len(dt) == int(n_sims):
                        margins = np.array(dm, dtype=float)
                        totals = np.array(dt, dtype=float)
                except Exception:
                    margins = None
                    totals = None
            if margins is None or totals is None:
                draws = rng.multivariate_normal(mean=np.array([m_mu, t_mu], dtype=float), cov=cov, size=int(n_sims))
                margins = draws[:, 0]
                totals = draws[:, 1]
        else:
            gid_s2 = str(game_id) if game_id is not None else None
            if draws_by_game is not None and gid_s2 and gid_s2 in draws_by_game:
                try:
                    dm, dt = draws_by_game[gid_s2]
                    if hasattr(dm, "__len__") and hasattr(dt, "__len__") and len(dm) == int(n_sims) and len(dt) == int(n_sims):
                        margins = np.array(dm, dtype=float)
                        totals = np.array(dt, dtype=float)
                except Exception:
                    margins = None
                    totals = None
            if margins is None:
                margins = rng.normal(loc=m_mu if np.isfinite(m_mu) else 0.0, scale=eff_ats_sigma, size=int(n_sims))
            if totals is None:
                totals = rng.normal(loc=t_mu if np.isfinite(t_mu) else 44.0, scale=eff_total_sigma, size=int(n_sims))

        total_i = np.clip(np.rint(totals), 0, 90).astype(int)
        # Derive team scores from the integer total for consistency.
        home_i = np.rint((total_i.astype(float) + margins) / 2.0).astype(int)
        away_i = total_i - home_i
        neg_away = away_i < 0
        if np.any(neg_away):
            home_i[neg_away] = total_i[neg_away]
            away_i[neg_away] = 0
        neg_home = home_i < 0
        if np.any(neg_home):
            away_i[neg_home] = total_i[neg_home]
            home_i[neg_home] = 0

        # Force (total, home, away) into reachable football score combinations (TD/FG/safety).
        total_i, home_i, away_i = _force_realistic_final_scores(total_i, home_i, away_i)

        # Simulate drive sequences
        seq = _interleave_possessions(drives_home, drives_away, start_home=bool(rng.random() < 0.45))
        n_drives = len(seq)
        if n_drives <= 0:
            continue

        home_cum = np.zeros((int(n_sims), n_drives), dtype=float)
        away_cum = np.zeros((int(n_sims), n_drives), dtype=float)
        home_drive_pts = np.zeros((int(n_sims), n_drives), dtype=float)
        away_drive_pts = np.zeros((int(n_sims), n_drives), dtype=float)
        drive_sec = np.zeros((int(n_sims), n_drives), dtype=float)
        # Outcome code per sim per drive
        # 0: Punt, 1: INT, 2: Fumble, 3: Downs, 4: Missed FG, 5: End Half, 6: FG, 7: TD
        drive_out = np.zeros((int(n_sims), n_drives), dtype=np.int8)

        home_drives_by_q = _split_count_across_quarters(drives_home)
        away_drives_by_q = _split_count_across_quarters(drives_away)

        for j in range(int(n_sims)):
            hp = int(home_i[j]); ap = int(away_i[j])
            hw = _quarter_weights_for_team(is_winner=(hp >= ap))
            aw = _quarter_weights_for_team(is_winner=(ap > hp))
            h_plays_q = _bucket_plays_to_quarters(_decompose_points_to_scoring_plays(hp, rng), hw, rng)
            a_plays_q = _bucket_plays_to_quarters(_decompose_points_to_scoring_plays(ap, rng), aw, rng)
            h_drive_points = _allocate_quarter_plays_to_drives(h_plays_q, home_drives_by_q, rng)
            a_drive_points = _allocate_quarter_plays_to_drives(a_plays_q, away_drives_by_q, rng)
            hi = 0; ai = 0
            hscore = 0; ascore = 0
            for d, who in enumerate(seq):
                if who == "home":
                    pts = int(h_drive_points[hi]) if hi < len(h_drive_points) else 0
                    hi += 1
                    hscore += pts
                    home_drive_pts[j, d] = float(pts)
                else:
                    pts = int(a_drive_points[ai]) if ai < len(a_drive_points) else 0
                    ai += 1
                    ascore += pts
                    away_drive_pts[j, d] = float(pts)

                # Sample a coarse drive outcome (used for UI narrative only).
                try:
                    if pts >= 6:
                        ocode = 7  # TD
                    elif pts == 3:
                        ocode = 6  # FG
                    else:
                        u = float(rng.random())
                        k = int(np.searchsorted(_ns_cum, u, side="right"))
                        k = int(np.clip(k, 0, len(_ns_names) - 1))
                        ocode = int(k)  # 0..5
                    drive_out[j, d] = np.int8(ocode)
                except Exception:
                    drive_out[j, d] = np.int8(0)

                # Estimate time of possession for this drive (seconds).
                try:
                    lam = float(max(1.0, plays_per_drive))
                    ocode_i = int(drive_out[j, d])
                    if ocode_i == 7:
                        lam += float(max(0.0, drive_extra_plays_td))
                    elif ocode_i == 6:
                        lam += float(max(0.0, drive_extra_plays_fg))
                    elif ocode_i in (1, 2):
                        # INT/Fumble often shorter than average
                        lam = float(max(1.0, lam - 2.0))
                    elif ocode_i == 3:
                        # Turnover on downs tends to be longer
                        lam += 2.0
                    elif ocode_i == 4:
                        # Missed FG implies drive reached range
                        lam += 1.0
                    elif ocode_i == 5:
                        # End half/game can be short or medium; bias slightly shorter
                        lam = float(max(1.0, lam - 1.0))

                    n_plays = int(max(1, rng.poisson(lam=lam)))
                    sec_mu = float(max(10.0, seconds_per_play))
                    sec_sd = float(max(0.0, seconds_per_play_sigma))

                    # Sample a per-drive pace (seconds/play), then sum play times via a gamma draw.
                    # Gamma introduces a heavier right tail vs a simple normal, which better matches
                    # real drive TOP variability.
                    sec_mu_eff = sec_mu
                    if sec_sd > 0:
                        sec_mu_eff = float(rng.normal(loc=sec_mu, scale=sec_sd))
                    sec_mu_eff = float(np.clip(sec_mu_eff, 12.0, 45.0))
                    dur = float(rng.gamma(shape=float(n_plays), scale=sec_mu_eff))

                    # Outcome-based tweaks
                    ocode_i = int(drive_out[j, d])
                    if ocode_i in (1, 2):
                        # INT/Fumble often happen earlier in a drive
                        dur *= 0.75
                    elif ocode_i == 0:
                        # Punts skew a bit shorter
                        dur *= 0.90
                    elif ocode_i == 3:
                        # Turnover on downs tends to be a longer series
                        dur *= 1.10
                    elif ocode_i == 5:
                        # End half/game sequences are often hurry-up or abbreviated
                        dur *= 0.85

                    # Global heavy-tail multiplier (tempo, penalties, stoppages). Keep mean ~1.0.
                    if drive_time_mult_sigma > 0:
                        sig = float(max(0.01, drive_time_mult_sigma))
                        mult = float(rng.lognormal(mean=-0.5 * sig * sig, sigma=sig))
                        dur *= mult

                    cap = float(max(120.0, drive_time_cap_sec))
                    drive_sec[j, d] = float(np.clip(dur, 20.0, cap))
                except Exception:
                    drive_sec[j, d] = float(max(15.0, float(plays_per_drive) * float(seconds_per_play)))
                home_cum[j, d] = float(hscore)
                away_cum[j, d] = float(ascore)

        # Aggregate per-drive statistics
        for d in range(n_drives):
            hmean = float(np.mean(home_cum[:, d]))
            amean = float(np.mean(away_cum[:, d]))
            p_home = float(np.mean(home_cum[:, d] > away_cum[:, d]))
            p_tie = float(np.mean(home_cum[:, d] == away_cum[:, d]))
            p_away = float(np.mean(home_cum[:, d] < away_cum[:, d]))
            q = int(np.clip(int((d / max(1, n_drives)) * 4) + 1, 1, 4))

            # Possession and drive outcome probabilities.
            who = seq[d] if d < len(seq) else ""
            poss_team = home_team if who == "home" else away_team
            pts_drive = home_drive_pts[:, d] + away_drive_pts[:, d]
            p_score = float(np.mean(pts_drive > 0))
            p_fg = float(np.mean(pts_drive == 3))
            # Treat 6/7/8 as TD-family.
            p_td = float(np.mean(pts_drive >= 6))
            p_safety = float(np.mean(pts_drive == 2))

            p_punt = float(np.mean(drive_out[:, d] == 0))
            p_int = float(np.mean(drive_out[:, d] == 1))
            p_fum = float(np.mean(drive_out[:, d] == 2))
            p_downs = float(np.mean(drive_out[:, d] == 3))
            p_missfg = float(np.mean(drive_out[:, d] == 4))
            p_end = float(np.mean(drive_out[:, d] == 5))
            dur_mean = float(np.mean(drive_sec[:, d]))
            try:
                dur_p25 = float(np.quantile(drive_sec[:, d], 0.25))
                dur_p75 = float(np.quantile(drive_sec[:, d], 0.75))
            except Exception:
                dur_p25 = np.nan
                dur_p75 = np.nan
            p_noscore = float(max(0.0, 1.0 - p_score))

            # Mode selection for UI: if scoring is plausibly in play, show the most likely scoring outcome
            # (TD/FG/Safety). Otherwise, show the most likely no-score outcome.
            mode = "No score"
            try:
                score_floor = _env_f("SIM_DRIVE_MODE_SCORE_MIN", 0.28)
            except Exception:
                score_floor = 0.28
            try:
                if p_score >= float(score_floor):
                    best_score = max(
                        [
                            (p_td, "TD"),
                            (p_fg, "FG"),
                            (p_safety, "Safety"),
                        ],
                        key=lambda x: x[0],
                    )
                    mode = best_score[1] if best_score and best_score[1] else "Score"
                else:
                    best_ns = max(
                        [
                            (p_punt, "Punt"),
                            (p_int, "INT"),
                            (p_fum, "Fumble"),
                            (p_downs, "Downs"),
                            (p_missfg, "Missed FG"),
                            (p_end, "End Half"),
                            (p_noscore, "No score"),
                        ],
                        key=lambda x: x[0],
                    )
                    mode = best_ns[1] if best_ns and best_ns[1] else "No score"
            except Exception:
                mode = "No score"
            pts_mean = float(np.mean(pts_drive))
            out_rows.append(
                {
                    "season": season_i,
                    "week": week_i,
                    "game_id": game_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "drive_no": int(d + 1),
                    "quarter": q,
                    "poss_side": who,
                    "poss_team": poss_team,
                    "drive_sec_mean": dur_mean,
                    "drive_sec_p25": dur_p25,
                    "drive_sec_p75": dur_p75,
                    "drive_pts_mean": pts_mean,
                    "p_drive_score": p_score,
                    "p_drive_fg": p_fg,
                    "p_drive_td": p_td,
                    "p_drive_safety": p_safety,
                    "p_drive_punt": p_punt,
                    "p_drive_int": p_int,
                    "p_drive_fumble": p_fum,
                    "p_drive_downs": p_downs,
                    "p_drive_missed_fg": p_missfg,
                    "p_drive_end_half": p_end,
                    "drive_outcome_mode": mode,
                    "home_score_mean": hmean,
                    "away_score_mean": amean,
                    "home_score_drive_mean": float(np.mean(home_drive_pts[:, d])),
                    "away_score_drive_mean": float(np.mean(away_drive_pts[:, d])),
                    "p_home_lead": p_home,
                    "p_tie": p_tie,
                    "p_away_lead": p_away,
                    "drives_home": int(drives_home),
                    "drives_away": int(drives_away),
                    "drives_total": int(n_drives),
                }
            )

    return pd.DataFrame(out_rows)


def simulate_quarter_means(
    view_df: pd.DataFrame,
    n_sims: int = 2000,
    ats_sigma_override: float | None = None,
    total_sigma_override: float | None = None,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
    draws_by_game: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Simulate quarter-by-quarter expected scores.

    Writes a compact per-game artifact suitable for cards/UI:
      home_q1..home_q4, away_q1..away_q4 plus mean finals.
    """
    if view_df is None or view_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "home_team",
                "away_team",
                "home_q1",
                "home_q2",
                "home_q3",
                "home_q4",
                "away_q1",
                "away_q2",
                "away_q3",
                "away_q4",
                "home_points_mean",
                "away_points_mean",
            ]
        )

    df = view_df.copy()
    spread_ref, total_ref = _spread_total_refs(df)
    df["spread_ref"] = spread_ref
    df["total_ref"] = total_ref

    sigmas = _load_sigma_calibration(data_dir=data_dir)
    ats_sigma = float(sigmas.get("ats_sigma", 12.0))
    total_sigma = float(sigmas.get("total_sigma", 11.0))
    if isinstance(ats_sigma_override, (int, float)) and float(ats_sigma_override) > 0:
        ats_sigma = float(ats_sigma_override)
    if isinstance(total_sigma_override, (int, float)) and float(total_sigma_override) > 0:
        total_sigma = float(total_sigma_override)
    totals_cal = _load_totals_calibration(data_dir=data_dir)

    # Mean blending knobs
    blend_margin = _env_f("SIM_MARKET_BLEND_MARGIN", 0.0)
    blend_total = _env_f("SIM_MARKET_BLEND_TOTAL", 0.0)

    # Feature mean knobs
    k_mean_total_wind = _env_f("SIM_MEAN_TOTAL_K_WIND", -0.5)
    k_mean_margin_rest = _env_f("SIM_MEAN_MARGIN_K_REST", 0.15)
    k_mean_margin_elo = _env_f("SIM_MEAN_MARGIN_K_ELO", 0.08)
    k_mean_total_inj = _env_f("SIM_MEAN_TOTAL_K_INJ", -0.4)
    k_mean_total_defppg = _env_f("SIM_MEAN_TOTAL_K_DEFPPG", -0.4)
    k_mean_total_pressure = _env_f("SIM_MEAN_TOTAL_K_PRESSURE", -0.8)
    pressure_baseline = _env_f("SIM_PRESSURE_BASELINE", 0.065)
    k_mean_margin_rating = _env_f("SIM_MEAN_MARGIN_K_RATING", 0.08)

    k_mean_margin_neutral = _env_f("SIM_MEAN_MARGIN_K_NEUTRAL", 0.0)
    k_mean_total_precip = _env_f("SIM_MEAN_TOTAL_K_PRECIP", 0.0)
    k_mean_total_cold = _env_f("SIM_MEAN_TOTAL_K_COLD", 0.0)
    cold_threshold_f = _env_f("SIM_COLD_TEMP_F", 45.0)

    rng = np.random.default_rng(seed)
    rho = _env_f("SIM_CORR_MARGIN_TOTAL", 0.10)

    out_rows: list[dict[str, Any]] = []
    for idx in df.index:
        row = df.loc[idx]
        season_i = int(_coerce_float(row.get("season"))) if "season" in df.columns else None
        week_i = int(_coerce_float(row.get("week"))) if "week" in df.columns else None
        game_id = row.get("game_id")
        home_team = row.get("home_team")
        away_team = row.get("away_team")

        # Means (same fallbacks as simulate_mc_probs)
        m_mu = _coerce_float(row.get("pred_margin")) if "pred_margin" in df.columns else float("nan")
        if not np.isfinite(m_mu):
            ph = _coerce_float(row.get("pred_home_points"))
            pa = _coerce_float(row.get("pred_away_points"))
            if np.isfinite(ph) and np.isfinite(pa):
                m_mu = float(ph - pa)
        if not np.isfinite(m_mu):
            sref = _coerce_float(row.get("spread_ref"))
            if np.isfinite(sref):
                m_mu = -float(sref)

        t_mu = float("nan")
        if "pred_total_cal" in df.columns:
            t_mu = _coerce_float(row.get("pred_total_cal"))
        if not np.isfinite(t_mu) and "pred_total" in df.columns:
            t_mu = _coerce_float(row.get("pred_total"))
        if not np.isfinite(t_mu):
            tref = _coerce_float(row.get("total_ref"))
            if np.isfinite(tref):
                t_mu = float(tref)

        s = _coerce_float(row.get("spread_ref"))
        t_line = _coerce_float(row.get("total_ref"))
        if not np.isfinite(m_mu) and not np.isfinite(t_mu):
            continue

        # Totals calibration
        scale = float(totals_cal.get("scale", np.nan))
        shift = float(totals_cal.get("shift", np.nan))
        if np.isfinite(t_mu) and np.isfinite(scale) and np.isfinite(shift):
            t_mu = scale * t_mu + shift

        # Market blends
        if np.isfinite(m_mu) and np.isfinite(s) and 0.0 < float(blend_margin) <= 1.0:
            m_mu = (1.0 - float(blend_margin)) * float(m_mu) + float(blend_margin) * (-float(s))
        if np.isfinite(t_mu) and np.isfinite(t_line) and 0.0 < float(blend_total) <= 1.0:
            t_mu = (1.0 - float(blend_total)) * float(t_mu) + float(blend_total) * float(t_line)

        # Feature adjustments
        w_open = _coerce_float(row.get("wind_open"))
        if np.isfinite(t_mu) and np.isfinite(w_open) and w_open > 0:
            t_mu = float(t_mu) + float(k_mean_total_wind) * (w_open / 10.0)
        rest_diff = _coerce_float(row.get("rest_days_diff"))
        if np.isfinite(m_mu) and np.isfinite(rest_diff) and rest_diff != 0:
            m_mu = float(m_mu) + float(k_mean_margin_rest) * (rest_diff / 7.0)
        elo_diff = _coerce_float(row.get("elo_diff"))
        if np.isfinite(m_mu) and np.isfinite(elo_diff):
            m_mu = float(m_mu) + float(k_mean_margin_elo) * (elo_diff / 100.0)
        neutral_flag = _coerce_float(row.get("neutral_site_flag"))
        if np.isfinite(neutral_flag) and neutral_flag >= 0.5 and np.isfinite(m_mu):
            m_mu = float(m_mu) + float(k_mean_margin_neutral)
        h_inj = _coerce_float(row.get("home_inj_starters_out"))
        a_inj = _coerce_float(row.get("away_inj_starters_out"))
        total_out = 0.0
        for v in (h_inj, a_inj):
            if np.isfinite(v):
                total_out += float(v)
        if np.isfinite(t_mu) and total_out > 0:
            t_mu = float(t_mu) + float(k_mean_total_inj) * float(total_out)
        h_def_ppg = _coerce_float(row.get("home_def_ppg"))
        a_def_ppg = _coerce_float(row.get("away_def_ppg"))
        vals = [v for v in (h_def_ppg, a_def_ppg) if np.isfinite(v)]
        def_avg = float(np.mean(vals)) if vals else float("nan")
        if not np.isfinite(def_avg):
            def_diff = _coerce_float(row.get("def_ppg_diff"))
            if np.isfinite(def_diff):
                def_avg = 22.0 + (def_diff / 2.0)
        if np.isfinite(t_mu) and np.isfinite(def_avg):
            t_mu = float(t_mu) + float(k_mean_total_defppg) * ((def_avg - 22.0) / 5.0)
        h_def_sr = _coerce_float(row.get("home_def_sack_rate_ema"))
        if not np.isfinite(h_def_sr):
            h_def_sr = _coerce_float(row.get("home_def_sack_rate"))
        a_def_sr = _coerce_float(row.get("away_def_sack_rate_ema"))
        if not np.isfinite(a_def_sr):
            a_def_sr = _coerce_float(row.get("away_def_sack_rate"))
        sr_vals = [v for v in (h_def_sr, a_def_sr) if np.isfinite(v)]
        if sr_vals and np.isfinite(t_mu):
            sr_avg = float(np.mean(sr_vals))
            delta_units = (sr_avg - float(pressure_baseline)) / 0.05
            t_mu = float(t_mu) + float(k_mean_total_pressure) * float(delta_units)
        nm_diff = _coerce_float(row.get("net_margin_diff"))
        if np.isfinite(m_mu) and np.isfinite(nm_diff):
            m_mu = float(m_mu) + float(k_mean_margin_rating) * float(nm_diff)
        precip = _coerce_float(row.get("wx_precip_pct"))
        if np.isfinite(t_mu) and np.isfinite(precip) and precip > 0:
            t_mu = float(t_mu) + float(k_mean_total_precip) * (precip / 100.0)
        temp_f = _coerce_float(row.get("wx_temp_f"))
        if np.isfinite(t_mu) and np.isfinite(temp_f) and temp_f < float(cold_threshold_f):
            units = (float(cold_threshold_f) - float(temp_f)) / 10.0
            t_mu = float(t_mu) + float(k_mean_total_cold) * float(units)

        # Realism clamps (same semantics)
        delta_t_max = _env_f("SIM_TOTAL_DELTA_MAX", 10.0)
        t_min = _env_f("SIM_TOTAL_MEAN_MIN", 30.0)
        t_max = _env_f("SIM_TOTAL_MEAN_MAX", 62.0)
        if np.isfinite(t_mu):
            if np.isfinite(t_line) and delta_t_max > 0:
                t_mu = float(t_line) + float(np.clip(t_mu - float(t_line), -delta_t_max, delta_t_max))
            t_mu = float(np.clip(t_mu, t_min, t_max))
        delta_m_max = _env_f("SIM_MARGIN_DELTA_MAX", 10.0)
        m_abs_max = _env_f("SIM_MARGIN_MEAN_ABS_MAX", 20.0)
        if np.isfinite(m_mu):
            m_anchor = -float(s) if np.isfinite(s) else 0.0
            if delta_m_max > 0:
                m_mu = float(m_anchor) + float(np.clip(m_mu - float(m_anchor), -delta_m_max, delta_m_max))
            m_mu = float(np.clip(m_mu, -m_abs_max, m_abs_max))

        eff_ats_sigma, eff_total_sigma = _per_game_sigmas(row, ats_sigma=ats_sigma, total_sigma=total_sigma)

        margins = None
        totals = None
        gid_s = str(game_id) if game_id is not None else None
        if draws_by_game is not None and gid_s and gid_s in draws_by_game:
            try:
                dm, dt = draws_by_game[gid_s]
                if hasattr(dm, "__len__") and hasattr(dt, "__len__") and len(dm) == int(n_sims) and len(dt) == int(n_sims):
                    margins = np.array(dm, dtype=float)
                    totals = np.array(dt, dtype=float)
            except Exception:
                margins = None
                totals = None
        if margins is None or totals is None:
            if np.isfinite(m_mu) and np.isfinite(t_mu):
                cov = np.array(
                    [
                        [eff_ats_sigma**2, float(rho) * eff_ats_sigma * eff_total_sigma],
                        [float(rho) * eff_ats_sigma * eff_total_sigma, eff_total_sigma**2],
                    ],
                    dtype=float,
                )
                draws = rng.multivariate_normal(mean=np.array([m_mu, t_mu], dtype=float), cov=cov, size=int(n_sims))
                margins = draws[:, 0]
                totals = draws[:, 1]
            else:
                margins = rng.normal(loc=m_mu if np.isfinite(m_mu) else 0.0, scale=eff_ats_sigma, size=int(n_sims))
                totals = rng.normal(loc=t_mu if np.isfinite(t_mu) else 44.0, scale=eff_total_sigma, size=int(n_sims))

        total_i = np.clip(np.rint(totals), 0, 90).astype(int)
        # Derive team scores from the integer total for consistency.
        home_i = np.rint((total_i.astype(float) + margins) / 2.0).astype(int)
        away_i = total_i - home_i
        # Clamp negatives while preserving total.
        neg_away = away_i < 0
        if np.any(neg_away):
            home_i[neg_away] = total_i[neg_away]
            away_i[neg_away] = 0
        neg_home = home_i < 0
        if np.any(neg_home):
            away_i[neg_home] = total_i[neg_home]
            home_i[neg_home] = 0

        # Force (total, home, away) into reachable football score combinations (TD/FG/safety).
        total_i, home_i, away_i = _force_realistic_final_scores(total_i, home_i, away_i)

        home_q_sum = np.zeros(4, dtype=float)
        away_q_sum = np.zeros(4, dtype=float)
        for j in range(int(n_sims)):
            hp = int(home_i[j])
            ap = int(away_i[j])
            hw = _quarter_weights_for_team(is_winner=(hp >= ap))
            aw = _quarter_weights_for_team(is_winner=(ap > hp))
            hq = _allocate_plays_to_quarters(_decompose_points_to_scoring_plays(hp, rng), hw, rng)
            aq = _allocate_plays_to_quarters(_decompose_points_to_scoring_plays(ap, rng), aw, rng)
            home_q_sum += hq
            away_q_sum += aq

        out_rows.append(
            {
                "season": season_i,
                "week": week_i,
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_q1": float(home_q_sum[0] / float(n_sims)),
                "home_q2": float(home_q_sum[1] / float(n_sims)),
                "home_q3": float(home_q_sum[2] / float(n_sims)),
                "home_q4": float(home_q_sum[3] / float(n_sims)),
                "away_q1": float(away_q_sum[0] / float(n_sims)),
                "away_q2": float(away_q_sum[1] / float(n_sims)),
                "away_q3": float(away_q_sum[2] / float(n_sims)),
                "away_q4": float(away_q_sum[3] / float(n_sims)),
                "home_points_mean": float(np.mean(home_i)),
                "away_points_mean": float(np.mean(away_i)),
            }
        )

    return pd.DataFrame(out_rows)
