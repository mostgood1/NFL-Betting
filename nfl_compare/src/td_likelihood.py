from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data_sources import load_games, load_team_stats, load_lines, load_predictions
from .features import merge_features
from .weather import load_weather_for_games


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _implied_points(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Return implied points (home, away) from total and home spread if available.

    Uses: home_points = (total + margin) / 2, where margin = -spread_home.
    Robustness:
    - Treat zero/invalid totals as missing and prefer close_* fallbacks.
    - If spread missing, assume 0 (pick'em) rather than failing.
    Returns (None, None) if insufficient info.
    """
    # 1) Prefer sim-derived expected points when present (keeps props aligned with sim artifacts).
    try:
        hp = row.get("home_points_mean")
        ap = row.get("away_points_mean")
        hp = float(hp) if hp is not None else None
        ap = float(ap) if ap is not None else None
        if hp is not None and ap is not None and np.isfinite(hp) and np.isfinite(ap) and hp > 0 and ap > 0:
            return float(hp), float(ap)
    except Exception:
        pass

    # 2) Prefer prediction-derived team points if present.
    try:
        hp = row.get("pred_home_points")
        ap = row.get("pred_away_points")
        hp = float(hp) if hp is not None else None
        ap = float(ap) if ap is not None else None
        if hp is not None and ap is not None and np.isfinite(hp) and np.isfinite(ap) and hp > 0 and ap > 0:
            return float(hp), float(ap)
    except Exception:
        pass

    # 3) Prediction-derived total/margin.
    try:
        t = row.get("pred_total_cal") if "pred_total_cal" in row else row.get("pred_total")
        m = row.get("pred_margin")
        t = float(t) if t is not None else None
        m = float(m) if m is not None else None
        if t is not None and m is not None and np.isfinite(t) and np.isfinite(m) and t > 0:
            home_pts = (t + m) / 2.0
            away_pts = t - home_pts
            if np.isfinite(home_pts) and np.isfinite(away_pts):
                return float(home_pts), float(away_pts)
    except Exception:
        pass

    # Prefer posted closing numbers when available; treat non-positive totals as missing.
    total_candidates = [row.get("total"), row.get("close_total") if "close_total" in row else None]
    spread_candidates = [row.get("spread_home"), row.get("close_spread_home") if "close_spread_home" in row else None]

    total_f: Optional[float] = None
    for v in total_candidates:
        try:
            f = float(v)
            if np.isfinite(f) and f > 0.0:
                total_f = f
                break
        except Exception:
            continue

    spread_f: Optional[float] = None
    for v in spread_candidates:
        try:
            f = float(v)
            if np.isfinite(f):
                spread_f = f
                break
        except Exception:
            continue

    if total_f is None:
        return None, None
    if spread_f is None:
        spread_f = 0.0

    margin = -spread_f  # home expected margin (home - away)
    home_pts = (total_f + margin) / 2.0
    away_pts = total_f - home_pts
    return home_pts, away_pts


def _weather_factor(row: pd.Series) -> float:
    """Return a multiplicative adjustment based on weather/roof.
    - Domes neutralize wind/precip.
    - Outdoor wind > 15 mph reduces by ~2.5% per 5 mph over threshold.
    - Precipitation chance > 60% reduces by 7.5%.
    """
    # Dome check (if present as derived in features)
    is_dome = 0.0
    if "is_dome" in row and pd.notna(row["is_dome"]):
        try:
            is_dome = float(row["is_dome"]) or 0.0
        except Exception:
            is_dome = 0.0

    if is_dome >= 0.5:
        return 1.0

    wind = row.get("wx_wind_mph")
    precip = row.get("wx_precip_pct")
    factor = 1.0
    try:
        w = float(wind)
        if w > 15.0:
            over = max(0.0, w - 15.0)
            factor *= max(0.85, 1.0 - 0.025 * (over / 5.0))
    except Exception:
        pass
    try:
        p = float(precip)
        if p >= 60.0:
            factor *= 0.925
    except Exception:
        pass
    return float(factor)


def _pace_factor(pace_secs_play_prior: Optional[float], league_avg_pace: float) -> float:
    """Convert pace (seconds per play) into a small factor around 1.0.
    Faster than league (lower seconds/play) => >1; slower => <1; clamp to [0.9, 1.1].
    Treat non-finite values as neutral (1.0).
    """
    try:
        if pace_secs_play_prior is None or pd.isna(pace_secs_play_prior):
            return 1.0
        num = float(league_avg_pace)
        den = float(pace_secs_play_prior)
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
            return 1.0
        f = num / den
        if not np.isfinite(f):
            return 1.0
        return float(np.clip(f, 0.9, 1.1))
    except Exception:
        return 1.0


def _epa_factor(off_prior: Optional[float], opp_def_prior: Optional[float], beta: float = 2.0) -> float:
    """Transform (off - opp_def) EPA/play into a multiplicative factor using exp(beta * diff),
    clamped to a modest range [0.7, 1.3] to avoid extreme uncalibrated swings.
    Non-finite inputs are treated as 0.
    """
    try:
        def _safe(x: Optional[float]) -> float:
            try:
                f = float(x) if x is not None else 0.0
                return f if np.isfinite(f) else 0.0
            except Exception:
                return 0.0

        o = _safe(off_prior)
        d = _safe(opp_def_prior)
        diff = o - d
        val = float(np.exp(beta * diff))
        if not np.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.7, 1.3))
    except Exception:
        return 1.0


def _team_rows_from_game(row: pd.Series, league_avg_pace: float) -> list[dict]:
    """Build per-team rows for a single game row with implied points and adjustments."""
    home_pts, away_pts = _implied_points(row)
    # If we can't derive implied points, fall back to a neutral baseline
    if home_pts is None or away_pts is None:
        # Fallback total and spread-neutral baseline
        total = 44.0
        home_pts = total / 2.0
        away_pts = total / 2.0

    # If sim_probs attached expected points, treat them as authoritative scoring means.
    # In that case, avoid re-scaling TD expectation with additional EPA/pace/weather factors
    # (those would create misalignment vs the sim artifacts).
    using_sim_points = False
    try:
        hp = float(row.get("home_points_mean"))
        ap = float(row.get("away_points_mean"))
        using_sim_points = bool(np.isfinite(hp) and np.isfinite(ap) and hp > 0 and ap > 0)
    except Exception:
        using_sim_points = False

    # Weather factors evaluated on same row
    wf = 1.0 if using_sim_points else _weather_factor(row)

    # Priors present via merge_features (fall back to non-_prior columns when needed)
    home_epa_prior = row.get("home_off_epa_prior") if pd.notna(row.get("home_off_epa_prior")) else row.get("home_off_epa")
    away_epa_prior = row.get("away_off_epa_prior") if pd.notna(row.get("away_off_epa_prior")) else row.get("away_off_epa")
    home_opp_def_prior = row.get("away_def_epa_prior") if pd.notna(row.get("away_def_epa_prior")) else row.get("away_def_epa")
    away_opp_def_prior = row.get("home_def_epa_prior") if pd.notna(row.get("home_def_epa_prior")) else row.get("home_def_epa")

    # Pace priors (seconds per play)
    home_pace_prior = row.get("home_pace_prior") if pd.notna(row.get("home_pace_prior")) else row.get("home_pace_secs_play")
    away_pace_prior = row.get("away_pace_prior") if pd.notna(row.get("away_pace_prior")) else row.get("away_pace_secs_play")

    # Pass/rush rate priors
    home_pass_rate_prior = row.get("home_pass_rate_prior") if pd.notna(row.get("home_pass_rate_prior")) else row.get("home_pass_rate")
    away_pass_rate_prior = row.get("away_pass_rate_prior") if pd.notna(row.get("away_pass_rate_prior")) else row.get("away_pass_rate")
    home_rush_rate_prior = row.get("home_rush_rate_prior") if pd.notna(row.get("home_rush_rate_prior")) else row.get("home_rush_rate")
    away_rush_rate_prior = row.get("away_rush_rate_prior") if pd.notna(row.get("away_rush_rate_prior")) else row.get("away_rush_rate")

    # Compute multiplicative factors
    if using_sim_points:
        home_factor = 1.0
        away_factor = 1.0
    else:
        home_factor = _epa_factor(home_epa_prior, home_opp_def_prior) * _pace_factor(home_pace_prior, league_avg_pace) * wf
        away_factor = _epa_factor(away_epa_prior, away_opp_def_prior) * _pace_factor(away_pace_prior, league_avg_pace) * wf

    # Base lambda ~ expected touchdowns (Poisson) as implied_points / 7.0
    home_lambda = max(0.05, float(home_pts) / 7.0) * home_factor
    away_lambda = max(0.05, float(away_pts) / 7.0) * away_factor

    # Probability of at least one TD: 1 - exp(-lambda)
    home_p_td = float(1.0 - np.exp(-home_lambda))
    away_p_td = float(1.0 - np.exp(-away_lambda))

    return [
        {
            "season": row.get("season"),
            "week": row.get("week"),
            "game_id": row.get("game_id"),
            "date": row.get("date"),
            "team": row.get("home_team"),
            "opponent": row.get("away_team"),
            "is_home": 1,
            "implied_points": float(home_pts),
            "expected_tds": float(home_lambda),
            "td_likelihood": home_p_td,
            "off_epa_prior": home_epa_prior,
            "opp_def_epa_prior": home_opp_def_prior,
            "pace_prior": home_pace_prior,
            "home_pass_rate_prior": home_pass_rate_prior,
            "home_rush_rate_prior": home_rush_rate_prior,
        },
        {
            "season": row.get("season"),
            "week": row.get("week"),
            "game_id": row.get("game_id"),
            "date": row.get("date"),
            "team": row.get("away_team"),
            "opponent": row.get("home_team"),
            "is_home": 0,
            "implied_points": float(away_pts),
            "expected_tds": float(away_lambda),
            "td_likelihood": away_p_td,
            "off_epa_prior": away_epa_prior,
            "opp_def_epa_prior": away_opp_def_prior,
            "pace_prior": away_pace_prior,
            "away_pass_rate_prior": away_pass_rate_prior,
            "away_rush_rate_prior": away_rush_rate_prior,
        },
    ]


def compute_td_likelihood(season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
    """Compute per-team touchdown likelihood for upcoming games or a specific season/week.

    Returns a DataFrame with per-team rows for each target game.
    """
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    preds = load_predictions()

    # Filter games if season/week provided
    if games is None:
        games = pd.DataFrame()

    # Filter by season/week if provided
    if season is not None:
        games = games[games.get("season").astype("Int64") == int(season)].copy() if not games.empty else games
    if week is not None:
        games = games[games.get("week").astype("Int64") == int(week)].copy() if not games.empty else games

    # If requested slice is empty or we want upcoming games but schedule is missing, synthesize from lines
    def _synth_from_lines(_lines: pd.DataFrame, _season: Optional[int], _week: Optional[int]) -> pd.DataFrame:
        if _lines is None or _lines.empty:
            return pd.DataFrame()
        df = _lines.copy()
        if _season is not None:
            df = df[df.get("season").astype("Int64") == int(_season)]
        if _week is not None:
            df = df[df.get("week").astype("Int64") == int(_week)]
        if df.empty:
            return pd.DataFrame()
        # Ensure required columns
        out = pd.DataFrame({
            "season": df.get("season"),
            "week": df.get("week"),
            "game_id": df.get("game_id", pd.Series([pd.NA]*len(df))),
            "date": df.get("date", pd.Series([pd.NA]*len(df))),
            "home_team": df.get("home_team"),
            "away_team": df.get("away_team"),
            "home_score": pd.NA,
            "away_score": pd.NA,
        })
        # Drop rows with missing teams
        out = out.dropna(subset=["home_team","away_team"])
        # If game_id missing, synthesize
        def _mk_gid(r):
            if pd.notna(r.get("game_id")) and str(r.get("game_id")).strip() != "":
                return r["game_id"]
            s = str(r.get("season")) if pd.notna(r.get("season")) else ""
            w = str(r.get("week")) if pd.notna(r.get("week")) else ""
            ht = str(r.get("home_team", "")).split()[-1][:3].upper()
            at = str(r.get("away_team", "")).split()[-1][:3].upper()
            return f"{s}-{w:0>2}-{ht}-{at}"
        out["game_id"] = out.apply(_mk_gid, axis=1)
        return out

    need_upcoming = (season is None and week is None)
    if games.empty:
        synth = _synth_from_lines(lines, season, week)
        if synth is not None and not synth.empty:
            games = synth
        else:
            # As a last resort, synthesize minimal schedule rows from predictions
            if preds is not None and not preds.empty:
                df = preds.copy()
                if season is not None:
                    df = df[df.get("season").astype("Int64") == int(season)]
                if week is not None:
                    df = df[df.get("week").astype("Int64") == int(week)]
                if not df.empty:
                    base_cols = {c for c in df.columns}
                    ht = "home_team" if "home_team" in base_cols else None
                    at = "away_team" if "away_team" in base_cols else None
                    gid = "game_id" if "game_id" in base_cols else None
                    dt = "date" if "date" in base_cols else None
                    games = pd.DataFrame({
                        "season": df.get("season"),
                        "week": df.get("week"),
                        "game_id": df.get(gid) if gid else df.get("game_key", pd.Series([pd.NA]*len(df))),
                        "date": df.get(dt, pd.Series([pd.NA]*len(df))),
                        "home_team": df.get(ht),
                        "away_team": df.get(at),
                        "home_score": pd.NA,
                        "away_score": pd.NA,
                    }).dropna(subset=["home_team","away_team"]).drop_duplicates(subset=["game_id","home_team","away_team"], keep="first")
            if games is None or games.empty:
                return pd.DataFrame(columns=["season","week","game_id","team","opponent","is_home","implied_points","expected_tds","td_likelihood"]) 

    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = None

    feat = merge_features(games, team_stats, lines, wx)

    # If sim artifacts exist for the requested season/week, attach sim-derived expected points.
    try:
        if season is not None and week is not None:
            sim_fp = DATA_DIR / "backtests" / f"{int(season)}_wk{int(week)}" / "sim_probs.csv"
            if sim_fp.exists():
                sim = pd.read_csv(sim_fp)
                keep = [c for c in [
                    "game_id",
                    "home_points_mean",
                    "away_points_mean",
                    "total_points_mean",
                    "pred_margin",
                    "pred_total",
                    "spread_ref",
                    "total_ref",
                ] if c in sim.columns]
                if keep and "game_id" in keep and "game_id" in feat.columns:
                    sim_small = sim[keep].drop_duplicates(subset=["game_id"], keep="first")
                    feat = feat.merge(sim_small, on="game_id", how="left", suffixes=("", "_sim"))
    except Exception:
        pass

    # Also attach raw model predictions (best-effort) so td likelihood can fall back to them
    # when market lines are missing.
    try:
        if preds is not None and not preds.empty and "game_id" in preds.columns and "game_id" in feat.columns:
            keep = [c for c in [
                "game_id",
                "pred_margin",
                "pred_total",
                "pred_total_cal",
                "pred_home_points",
                "pred_away_points",
            ] if c in preds.columns]
            if keep and "game_id" in keep:
                pred_small = preds[keep].drop_duplicates(subset=["game_id"], keep="first")
                feat = feat.merge(pred_small, on="game_id", how="left", suffixes=("", "_pred"))
                # Prefer already-present columns; only fill missing.
                for c in ["pred_margin", "pred_total", "pred_total_cal", "pred_home_points", "pred_away_points"]:
                    cp = f"{c}_pred"
                    if cp in feat.columns:
                        if c in feat.columns:
                            feat[c] = feat[c].where(feat[c].notna(), feat[cp])
                        else:
                            feat[c] = feat[cp]
                drop_cols = [c for c in feat.columns if c.endswith("_pred")]
                if drop_cols:
                    feat = feat.drop(columns=drop_cols)
    except Exception:
        pass

    # Target games: those without scores (future) or those in requested season/week regardless of scores
    if need_upcoming:
        target = feat[feat["home_score"].isna() | feat["away_score"].isna()].copy()
    else:
        target = feat.copy()

    if target.empty:
        return pd.DataFrame(columns=["season","week","game_id","team","opponent","is_home","implied_points","expected_tds","td_likelihood"]) 

    # League average pace from priors if present, else fallback to historical mean seconds/play ~ 27.5
    pace_cols = []
    for c in ["home_pace_prior","away_pace_prior","home_pace_secs_play","away_pace_secs_play"]:
        if c in target.columns:
            pace_cols.append(c)
    if pace_cols:
        league_avg_pace = pd.to_numeric(pd.concat([target[c] for c in pace_cols], axis=0), errors="coerce").dropna().mean()
    else:
        league_avg_pace = 27.5
    if pd.isna(league_avg_pace):
        league_avg_pace = 27.5

    # Build per-team rows
    rows: list[dict] = []
    for _, r in target.iterrows():
        rows.extend(_team_rows_from_game(r, float(league_avg_pace)))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Create a normalized score (0-100) for ranking
    try:
        # rank by td_likelihood; scale to 0-100
        ranks = out["td_likelihood"].rank(method="min").astype(float)
        out["td_score"] = ((ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9) * 100.0).round(2)
    except Exception:
        out["td_score"] = (out["td_likelihood"].fillna(0.0) * 100.0).round(2)

    # Order columns
    cols = [
        "season","week","date","game_id","team","opponent","is_home",
        "implied_points","expected_tds","td_likelihood","td_score",
        "off_epa_prior","opp_def_epa_prior","pace_prior",
        # include pass/rush priors to support player-level allocation
        "home_pass_rate_prior","away_pass_rate_prior","home_rush_rate_prior","away_rush_rate_prior",
    ]
    out = out.reindex(columns=[c for c in cols if c in out.columns])
    return out.sort_values(["season","week","td_score"], ascending=[True, True, False])


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compute per-team touchdown likelihood for upcoming NFL games or a specific week.")
    p.add_argument("--season", type=int, default=None, help="Season year to compute for (optional)")
    p.add_argument("--week", type=int, default=None, help="Week number to compute for (optional)")
    p.add_argument("--out", type=str, default=None, help="Output CSV path (optional); defaults to data/td_likelihood_<season>_wk<week>.csv or data/td_likelihood_upcoming.csv")
    args = p.parse_args(argv)

    df = compute_td_likelihood(args.season, args.week)
    if df is None or df.empty:
        print("No target games found or insufficient data to compute touchdown likelihood.")
        return

    if args.out:
        out_fp = Path(args.out)
    else:
        if args.season and args.week:
            out_fp = DATA_DIR / f"td_likelihood_{args.season}_wk{args.week}.csv"
        else:
            out_fp = DATA_DIR / "td_likelihood_upcoming.csv"

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    # Print a small summary
    show = df.head(10)[[c for c in ["season","week","team","opponent","td_likelihood","td_score","implied_points","expected_tds"] if c in df.columns]].copy()
    try:
        # round for readability
        for c in ["td_likelihood","expected_tds"]:
            if c in show.columns:
                show[c] = show[c].astype(float).round(3)
    except Exception:
        pass
    print(f"Wrote touchdown likelihoods to {out_fp}")
    try:
        print(show.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
