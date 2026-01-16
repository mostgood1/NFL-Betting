from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_lines, load_team_stats
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games

DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


_SIM_REQUIRED_FEATURES: dict[str, str] = {
    # Market refs
    "spread_home": "lines",
    "close_spread_home": "lines",
    "total": "lines",
    "close_total": "lines",
    # Weather/context
    "roof_closed_flag": "weather",
    "wind_open": "weather",
    "wx_temp_f": "weather",
    "wx_precip_pct": "weather",
    "neutral_site_flag": "schedule",
    # Schedule priors
    "rest_days_diff": "schedule",
    # Ratings/priors
    "elo_diff": "elo",
    "home_def_ppg": "team_ratings",
    "away_def_ppg": "team_ratings",
    "def_ppg_diff": "team_ratings",
    "net_margin_diff": "team_ratings",
    # Pressure priors (from team_stats)
    "home_def_sack_rate_ema": "team_stats",
    "away_def_sack_rate_ema": "team_stats",
    "home_def_sack_rate": "team_stats",
    "away_def_sack_rate": "team_stats",
    # Injuries (optional depth chart)
    "home_inj_starters_out": "injuries",
    "away_inj_starters_out": "injuries",
}


_SIM_OPTIONAL_EXTRAS: dict[str, str] = {
    # Team context diffs (from team_stats)
    "pace_secs_play_diff": "team_stats",
    "pass_rate_diff": "team_stats",
    "rush_rate_diff": "team_stats",
    "qb_adj_diff": "team_stats",
    "sos_diff": "team_stats",
    "pace_secs_play_ema_diff": "team_stats",
    "pass_rate_ema_diff": "team_stats",
    "rush_rate_ema_diff": "team_stats",
    "qb_adj_ema_diff": "team_stats",
    "sos_ema_diff": "team_stats",
    # Aggregates already derived in merge_features
    "def_pressure_avg": "team_stats",
    "def_pressure_avg_ema": "team_stats",
    # Extended (optional data sources)
    "ppd_diff": "pfr_drive_stats",
    "yards_per_drive_diff": "pfr_drive_stats",
    "seconds_per_drive_diff": "pfr_drive_stats",
    "drives_diff": "pfr_drive_stats",
    "rzd_off_eff_diff": "redzone",
    "rzd_def_eff_diff": "redzone",
    "rzd_off_td_rate_diff": "redzone",
    "rzd_def_td_rate_diff": "redzone",
    "explosive_pass_rate_diff": "explosive",
    "explosive_run_rate_diff": "explosive",
}


_CONTINUOUS_FEATURES = {
    "wind_open",
    "wx_temp_f",
    "wx_precip_pct",
    "rest_days_diff",
    "elo_diff",
    "home_def_ppg",
    "away_def_ppg",
    "def_ppg_diff",
    "net_margin_diff",
    "home_def_sack_rate_ema",
    "away_def_sack_rate_ema",
    "home_def_sack_rate",
    "away_def_sack_rate",
    "home_inj_starters_out",
    "away_inj_starters_out",
    "pace_secs_play_diff",
    "pass_rate_diff",
    "rush_rate_diff",
    "qb_adj_diff",
    "qb_adj_ema_diff",
    "sos_diff",
    "sos_ema_diff",
    "neutral_site_flag",
}


def _parse_weeks_arg(weeks: str) -> list[int]:
    s = str(weeks).strip()
    if not s:
        return []
    if "-" in s:
        a, b = s.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _best_row_per_game(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "game_id" not in df.columns:
        return df

    for c in ["spread_home", "close_spread_home", "total", "close_total"]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])
    have_cols = [c for c in ["spread_home", "close_spread_home", "total", "close_total"] if c in df.columns]
    if not have_cols:
        return df

    out = df.copy()
    out["_line_quality"] = out[have_cols].notna().sum(axis=1)
    out = out.sort_values(["game_id", "_line_quality"], ascending=[True, False]).drop_duplicates(subset=["game_id"], keep="first")
    out = out.drop(columns=["_line_quality"], errors="ignore")
    return out


def _coverage_for_columns(df: pd.DataFrame, columns: Iterable[str], week: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _depth_chart_inactive_starters_any(season_val: int, week_val: int) -> tuple[bool, bool]:
        """Return (exists, inactive_starter_any).

        We treat 'inactive starter' as the first depth_chart row per (team, position)
        having active == False.
        """
        try:
            dc_path = DATA_DIR / f"depth_chart_{int(season_val)}_wk{int(week_val)}.csv"
            if not dc_path.exists():
                return (False, False)
            dc = pd.read_csv(dc_path)
            if dc is None or dc.empty or "active" not in dc.columns:
                return (True, False)

            dcc = dc.copy()
            dcc["position"] = dcc.get("position", "").astype(str).str.upper()
            if "depth_rank" not in dcc.columns:
                dcc["depth_rank"] = dcc.groupby(["team", "position"]).cumcount() + 1

            starters = (
                dcc.sort_values(["team", "position", "depth_rank"])
                .groupby(["team", "position"], as_index=False)
                .first()
            )
            inactive_any = bool((starters["active"].astype(bool) == False).any())
            return (True, inactive_any)
        except Exception:
            return (False, False)

    n = int(len(df))
    for c in columns:
        present = c in df.columns
        if not present:
            rows.append(
                {
                    "feature": c,
                    "present": False,
                    "n_games": n,
                    "non_null_pct": 0.0,
                    "zero_pct": np.nan,
                    "unique_non_null": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "status": "missing_col",
                    "notes": "",
                }
            )
            continue

        s = df[c]
        s_num = _coerce_num(s) if (pd.api.types.is_numeric_dtype(s) or c in _CONTINUOUS_FEATURES) else None
        non_null = int(s.notna().sum())
        non_null_pct = float(non_null / n) if n else 0.0
        unique_non_null = int(s.dropna().nunique())

        zero_pct = np.nan
        v_min = v_max = v_mean = v_std = np.nan
        if s_num is not None:
            nn = s_num.dropna()
            if len(nn):
                zero_pct = float((nn == 0).mean())
                v_min = float(nn.min())
                v_max = float(nn.max())
                v_mean = float(nn.mean())
                v_std = float(nn.std(ddof=0))

        status = "ok"
        notes = ""
        if non_null == 0:
            status = "all_null"
        elif unique_non_null <= 1:
            if s_num is not None and len(s_num.dropna()) and float(s_num.dropna().iloc[0]) == 0.0:
                status = "constant_zero"
            else:
                status = "constant"

        # Heuristics for "real data" vs silent defaults
        if status == "constant_zero" and c in _CONTINUOUS_FEATURES:
            # Neutral site is legitimately 0 for most games/weeks.
            # Treat constant-zero as expected rather than a pipeline failure.
            if c == "neutral_site_flag":
                notes = "expected_no_neutral_games"
                status = "ok"
            # Early-week SOS can be legitimately 0.0 for all teams depending on how it's initialized.
            elif c in {"sos_diff", "sos_ema_diff"} and week is not None and int(week) <= 3:
                notes = "expected_sparse_early"
                status = "ok"
            # Week 1 can be legitimately sparse for priors, but weather/lines should still vary.
            elif week is not None and int(week) <= 1 and c in {"rest_days_diff", "home_def_ppg", "away_def_ppg", "def_ppg_diff", "net_margin_diff"}:
                notes = "expected_sparse_week1"
            else:
                # Injury features can legitimately be all-zero for one side (or even both sides)
                # if the depth chart indicates no inactive starters.
                if c in {"home_inj_starters_out", "away_inj_starters_out"}:
                    other = "away_inj_starters_out" if c == "home_inj_starters_out" else "home_inj_starters_out"
                    if other in df.columns:
                        other_num = _coerce_num(df[other])
                        # Decide if constant-zero is suspicious based on depth chart evidence
                        season_val = None
                        try:
                            if "season" in df.columns and df["season"].notna().any():
                                season_val = int(pd.to_numeric(df["season"].dropna().iloc[0], errors="coerce"))
                        except Exception:
                            season_val = None

                        if season_val is not None and week is not None:
                            dc_exists, inactive_any = _depth_chart_inactive_starters_any(season_val, int(week))
                        else:
                            dc_exists, inactive_any = (False, False)

                        # If depth chart missing, we can't distinguish "real zeros" vs silent defaults.
                        if not dc_exists:
                            notes = "suspect_default"
                        else:
                            # If there are no inactive starters anywhere, constant-zero is expected.
                            if not inactive_any:
                                notes = "legit_all_active"
                            else:
                                # Some inactive starters exist in the league that week.
                                # If the opposite side shows any non-zero values, this side being zero is plausible.
                                if other_num.notna().any() and (other_num != 0).any():
                                    notes = "legit_sparse_by_side"
                                else:
                                    # Both sides are zero but starters exist somewhere -> more suspicious.
                                    notes = "suspect_default"
                    else:
                        notes = "suspect_default"
                elif c in {"qb_adj_diff", "qb_adj_ema_diff"}:
                    # qb_adj comes from depth chart. If the depth chart exists for the week,
                    # constant-zero is often legitimately "both starting QBs active".
                    season_val = None
                    try:
                        if "season" in df.columns and df["season"].notna().any():
                            season_val = int(pd.to_numeric(df["season"].dropna().iloc[0], errors="coerce"))
                    except Exception:
                        season_val = None
                    if season_val is not None and week is not None:
                        dc_exists, _inactive_any = _depth_chart_inactive_starters_any(season_val, int(week))
                        notes = "legit_qb_adj_all_zero" if dc_exists else "suspect_default"
                    else:
                        notes = "suspect_default"
                else:
                    notes = "suspect_default"

        # Normalize: for a couple of known-sparse features, treat non-suspect constant-zeros as ok.
        if status == "constant_zero" and notes and not str(notes).startswith("suspect"):
            if c in {"home_inj_starters_out", "away_inj_starters_out", "qb_adj_diff", "qb_adj_ema_diff"}:
                status = "ok"
        if status == "all_null" and week is not None and int(week) <= 1 and c in {"home_def_ppg", "away_def_ppg", "def_ppg_diff", "net_margin_diff"}:
            notes = "expected_sparse_week1"

        rows.append(
            {
                "feature": c,
                "present": True,
                "n_games": n,
                "non_null_pct": non_null_pct,
                "zero_pct": zero_pct,
                "unique_non_null": unique_non_null,
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "status": status,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def coverage_report(season: int, weeks: list[int], include_extras: bool = True) -> pd.DataFrame:
    games = load_games()
    lines = load_lines()
    stats = load_team_stats()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx).copy()
    if "season" in feat.columns:
        feat["season"] = _coerce_num(feat["season"]).astype("Int64")
    if "week" in feat.columns:
        feat["week"] = _coerce_num(feat["week"]).astype("Int64")

    cols = list(_SIM_REQUIRED_FEATURES.keys())
    if include_extras:
        cols = cols + [c for c in _SIM_OPTIONAL_EXTRAS.keys() if c not in cols]

    out_rows: list[pd.DataFrame] = []
    for wk in weeks:
        df = feat.loc[(feat["season"] == int(season)) & (feat["week"] == int(wk))].copy()
        if df.empty:
            continue
        df = _best_row_per_game(df)
        cov = _coverage_for_columns(df, cols, week=int(wk))
        cov.insert(0, "week", int(wk))
        cov.insert(0, "season", int(season))

        # Attach source
        src = {**_SIM_REQUIRED_FEATURES, **(_SIM_OPTIONAL_EXTRAS if include_extras else {})}
        cov["source"] = cov["feature"].map(src).fillna("unknown")
        out_rows.append(cov)

    if not out_rows:
        return pd.DataFrame(columns=["season", "week", "feature", "source", "present", "n_games", "non_null_pct", "zero_pct", "unique_non_null", "min", "max", "mean", "std", "status", "notes"])

    return pd.concat(out_rows, ignore_index=True)


def audit_week(season: int, week: int, out_csv: str | None = None) -> pd.DataFrame:
    games = load_games()
    lines = load_lines()
    stats = load_team_stats()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx).copy()
    if "season" in feat.columns:
        feat["season"] = _coerce_num(feat["season"]).astype("Int64")
    if "week" in feat.columns:
        feat["week"] = _coerce_num(feat["week"]).astype("Int64")

    df = feat.loc[(feat["season"] == int(season)) & (feat["week"] == int(week))].copy()
    if df.empty:
        return df

    # merge_features can yield multiple rows per game (multiple books/snapshots). For auditing,
    # keep the "best" row per game_id based on available line fields.
    df = _best_row_per_game(df)

    # Normalized numeric columns we care about
    for c in [
        "spread_home",
        "close_spread_home",
        "total",
        "close_total",
        "wind_open",
        "roof_closed_flag",
        "neutral_site_flag",
        "rest_days_diff",
        "elo_diff",
        "wx_temp_f",
        "wx_precip_pct",
    ]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])

    # Derived diagnostics
    df["delta_total_open_close"] = df.get("total") - df.get("close_total")
    df["delta_spread_open_close"] = df.get("spread_home") - df.get("close_spread_home")

    # Flags (heuristics)
    df["flag_total_implausible"] = False
    if "total" in df.columns:
        df.loc[df["total"].notna() & ((df["total"] < 30) | (df["total"] > 70)), "flag_total_implausible"] = True

    df["flag_total_jump"] = False
    if "delta_total_open_close" in df.columns:
        df.loc[df["delta_total_open_close"].abs() >= 10, "flag_total_jump"] = True

    df["flag_spread_jump"] = False
    if "delta_spread_open_close" in df.columns:
        df.loc[df["delta_spread_open_close"].abs() >= 7, "flag_spread_jump"] = True

    df["flag_spread_sign_flip"] = False
    if "spread_home" in df.columns and "close_spread_home" in df.columns:
        sign_flip = (df["spread_home"].notna() & df["close_spread_home"].notna() & (np.sign(df["spread_home"]) != np.sign(df["close_spread_home"])) )
        df.loc[sign_flip & (df["delta_spread_open_close"].abs() >= 2), "flag_spread_sign_flip"] = True

    df["flag_weather_missing"] = False
    if "wind_open" in df.columns:
        df.loc[df["wind_open"].isna(), "flag_weather_missing"] = True

    # Output selection
    keep_cols = [
        c
        for c in [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "spread_home",
            "close_spread_home",
            "delta_spread_open_close",
            "total",
            "close_total",
            "delta_total_open_close",
            "wind_open",
            "roof_closed_flag",
            "neutral_site_flag",
            "rest_days_diff",
            "elo_diff",
            "wx_temp_f",
            "wx_precip_pct",
            "flag_total_implausible",
            "flag_total_jump",
            "flag_spread_jump",
            "flag_spread_sign_flip",
            "flag_weather_missing",
        ]
        if c in df.columns
    ]

    out = df[keep_cols].copy()

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit sim-relevant feature correctness for a given season/week")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, default=None, help="Single week to audit")
    ap.add_argument("--weeks", type=str, default=None, help="Coverage weeks: '1-18' or '10,11,12'")
    ap.add_argument("--coverage", action="store_true", help="Emit a feature-coverage report instead of per-game heuristics")
    ap.add_argument("--no-extras", action="store_true", help="Coverage mode: only sim_engine-required features")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    if args.coverage:
        weeks: list[int] = []
        if args.weeks:
            weeks = _parse_weeks_arg(args.weeks)
        elif args.week is not None:
            weeks = [int(args.week)]
        else:
            raise SystemExit("Coverage mode requires --week or --weeks")

        rep = coverage_report(int(args.season), weeks, include_extras=(not args.no_extras))
        if rep.empty:
            print("No rows found for that season/week(s).")
            return 0

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rep.to_csv(out_path, index=False)

        # Print a compact summary: show suspect/missing first.
        show = rep.copy()
        show["_rank"] = 0
        show.loc[show["status"].isin(["missing_col", "all_null"]), "_rank"] = 3
        show.loc[show["status"].isin(["constant_zero", "constant"]), "_rank"] = 2
        show.loc[show["notes"].astype(str).str.contains("suspect", na=False), "_rank"] = 4
        show = show.sort_values(["season", "week", "_rank", "source", "feature"], ascending=[True, True, False, True, True]).drop(columns=["_rank"])
        # Avoid dumping huge output; print the top issues per week.
        max_rows = 120
        if len(show) > max_rows:
            print(show.head(max_rows).to_string(index=False))
            print(f"... ({len(show) - max_rows} more rows)")
        else:
            print(show.to_string(index=False))
        return 0

    if args.week is None:
        raise SystemExit("Non-coverage mode requires --week")

    out = audit_week(args.season, int(args.week), out_csv=args.out)
    if out is None or out.empty:
        print("No rows found for that season/week.")
        return 0

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    if flag_cols:
        flagged = out.loc[out[flag_cols].any(axis=1)].copy()
        print(f"Rows: {len(out)} | Flagged: {len(flagged)}")
        if len(flagged):
            print(flagged.to_string(index=False))
    else:
        print(f"Rows: {len(out)}")
        print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
