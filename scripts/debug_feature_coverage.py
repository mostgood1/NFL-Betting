import os
import sys
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path so `nfl_compare` is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _data_dir() -> Path:
    env = os.environ.get("NFL_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


def _norm_team(s: str) -> str:
    try:
        from nfl_compare.src.team_normalizer import normalize_team_name

        return normalize_team_name(str(s)).strip()
    except Exception:
        return str(s).strip()


def _print_file_status(label: str, path: Path) -> None:
    if path.exists():
        try:
            size_kb = path.stat().st_size / 1024.0
            print(f"[OK] {label}: {path} ({size_kb:,.1f} KB)")
        except Exception:
            print(f"[OK] {label}: {path}")
    else:
        print(f"[MISSING] {label}: {path}")


def _match_row(df: pd.DataFrame, season: int, week: int, home: str, away: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ("season", "week"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if {"home_team", "away_team"}.issubset(out.columns):
        out["home_team"] = out["home_team"].astype(str).map(_norm_team)
        out["away_team"] = out["away_team"].astype(str).map(_norm_team)
        home_n = _norm_team(home)
        away_n = _norm_team(away)
        out = out[(out["season"] == season) & (out["week"] == week)]
        # accept either orientation
        mk = out["home_team"].astype(str) + "|" + out["away_team"].astype(str)
        keys = {f"{home_n}|{away_n}", f"{away_n}|{home_n}"}
        out = out[mk.isin(keys)]
        return out
    return out[(out.get("season") == season) & (out.get("week") == week)]


def _coverage_for_team_week(src: pd.DataFrame, season: int, week: int, team: str) -> dict:
    """Exact coverage: rows where (season, week, team) match."""
    if src is None or src.empty:
        return {"rows": 0}
    d = src.copy()
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "team" in d.columns:
        d["team"] = d["team"].astype(str).map(_norm_team)
        team_n = _norm_team(team)
        sub = d[(d.get("season") == season) & (d.get("week") == week) & (d.get("team") == team_n)]
        return {"rows": int(len(sub))}
    return {"rows": 0}


def _coverage_for_team_asof(src: pd.DataFrame, season: int, max_week: int, team: str) -> dict:
    """As-of coverage: latest available row where week <= max_week for (season, team)."""
    if src is None or src.empty:
        return {"rows": 0, "asof_week": None}
    d = src.copy()
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "team" not in d.columns or "week" not in d.columns or "season" not in d.columns:
        return {"rows": 0, "asof_week": None}
    d["team"] = d["team"].astype(str).map(_norm_team)
    team_n = _norm_team(team)
    sub = d[(d["season"] == season) & (d["team"] == team_n) & (d["week"].notna()) & (d["week"] <= max_week)]
    if sub.empty:
        return {"rows": 0, "asof_week": None}
    asof_week = int(sub["week"].max())
    rows = int((sub["week"] == asof_week).sum())
    return {"rows": rows, "asof_week": asof_week}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Report missing model features for a given matchup.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--home", type=str, required=True)
    ap.add_argument("--away", type=str, required=True)
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)
    home = args.home
    away = args.away

    dd = _data_dir()
    print(f"Using data dir: {dd}")

    # File presence (source of truth)
    _print_file_status("games.csv", dd / "games.csv")
    _print_file_status("lines.csv", dd / "lines.csv")
    _print_file_status("team_stats.csv", dd / "team_stats.csv")
    _print_file_status("pfr_drive_stats.csv", dd / "pfr_drive_stats.csv")
    _print_file_status("redzone_splits.csv", dd / "redzone_splits.csv")
    _print_file_status("explosive_rates.csv", dd / "explosive_rates.csv")
    _print_file_status("penalties_stats.csv", dd / "penalties_stats.csv")
    _print_file_status("special_teams.csv", dd / "special_teams.csv")
    _print_file_status("officiating_crews.csv", dd / "officiating_crews.csv")
    _print_file_status("weather_noaa.csv", dd / "weather_noaa.csv")

    from nfl_compare.src.data_sources import (
        load_games,
        load_team_stats,
        load_lines,
        load_pfr_drive_stats,
        load_redzone_splits,
        load_explosive_rates,
        load_penalties_stats,
        load_special_teams,
        load_officiating_crews,
        load_weather_noaa,
    )
    from nfl_compare.src.weather import load_weather_for_games
    from nfl_compare.src.features import merge_features
    from nfl_compare.src.models import FEATURES

    games = load_games()
    lines = load_lines()
    stats = load_team_stats()
    wx = load_weather_for_games(games)

    # Optional datasets: check exact prior-week coverage *and* as-of fallback coverage.
    # In postseason, many team-week feeds stop at the last regular season week,
    # so exact (week-1) is often empty even though as-of fill will work.
    prev_week = week - 1
    print("\nTeam-week coverage checks (exact prev_week and as-of <= prev_week)")
    for label, loader in [
        ("pfr_drive_stats", load_pfr_drive_stats),
        ("redzone_splits", load_redzone_splits),
        ("explosive_rates", load_explosive_rates),
        ("penalties_stats", load_penalties_stats),
        ("special_teams", load_special_teams),
    ]:
        try:
            d = loader()
        except Exception:
            d = pd.DataFrame()
        ch_exact = _coverage_for_team_week(d, season, prev_week, home)
        ca_exact = _coverage_for_team_week(d, season, prev_week, away)
        ch_asof = _coverage_for_team_asof(d, season, prev_week, home)
        ca_asof = _coverage_for_team_asof(d, season, prev_week, away)
        print(
            f"- {label}: "
            f"home_rows(prev_wk={prev_week})={ch_exact['rows']} "
            f"away_rows(prev_wk={prev_week})={ca_exact['rows']} "
            f"| asof_home_wk={ch_asof['asof_week']} rows={ch_asof['rows']} "
            f"asof_away_wk={ca_asof['asof_week']} rows={ca_asof['rows']}"
        )

    # Game-level optional feeds
    try:
        oc = load_officiating_crews()
        noa = load_weather_noaa()
    except Exception:
        oc = pd.DataFrame()
        noa = pd.DataFrame()

    feat = merge_features(games, stats, lines, wx)
    row = _match_row(feat, season, week, home, away)
    if row.empty:
        print("\n[ERROR] No feature row found for matchup/week. Check games/lines team names and week.")
        return 2

    # Prefer a single row
    row = row.iloc[[0]].copy()

    print("\nMatch found:")
    for c in ["game_id", "home_team", "away_team", "game_date", "date", "spread_home", "total", "close_spread_home", "close_total", "is_postseason", "rest_days_diff"]:
        if c in row.columns:
            print(f"- {c}: {row.iloc[0].get(c)}")

    # Feature availability report
    missing_cols = [c for c in FEATURES if c not in row.columns]
    present_cols = [c for c in FEATURES if c in row.columns]
    nan_cols = [c for c in present_cols if pd.isna(row.iloc[0].get(c))]
    ok_cols = [c for c in present_cols if c not in nan_cols]

    print("\nModel feature coverage (for this row)")
    print(f"- FEATURES total: {len(FEATURES)}")
    print(f"- Missing columns (not created by merge_features): {len(missing_cols)}")
    print(f"- Present but NaN: {len(nan_cols)}")
    print(f"- Present with value: {len(ok_cols)}")

    if missing_cols:
        print("\nMissing columns:")
        for c in missing_cols:
            print(f"  - {c}")

    if nan_cols:
        print("\nPresent-but-NaN (likely missing data for this matchup):")
        for c in nan_cols:
            print(f"  - {c}")

    # Quick spotlight on known high-impact blocks
    blocks = {
        "Market": ["spread_home", "total"],
        "Core diffs": ["elo_diff", "off_epa_diff", "def_epa_diff", "pace_secs_play_diff", "pass_rate_diff"],
        "Pressure": ["def_pressure_avg_ema", "def_pressure_avg"],
        "Weather": ["wx_temp_f", "wx_wind_mph", "wx_precip_pct", "roof_closed_flag", "wind_open"],
        "PhaseA": ["ppd_diff", "td_per_drive_diff", "explosive_pass_rate_diff", "penalty_rate_diff", "fg_acc_diff", "phase_a_total_delta"],
        "Crew/NOAA": ["crew_penalty_rate", "wx_gust_mph", "wx_dew_point_f"],
    }

    print("\nBlock values (NaN shown as blank)")
    for name, cols in blocks.items():
        parts = []
        for c in cols:
            if c in row.columns:
                v = row.iloc[0].get(c)
                if pd.isna(v):
                    parts.append(f"{c}=")
                else:
                    try:
                        parts.append(f"{c}={float(v):.4g}")
                    except Exception:
                        parts.append(f"{c}={v}")
            else:
                parts.append(f"{c}=(missing)")
        print(f"- {name}: " + ", ".join(parts))

    # Game-level optional feed keys
    try:
        gid = row.iloc[0].get("game_id")
        if gid is not None:
            if not oc.empty and "game_id" in oc.columns:
                print(f"\nOfficiating crews rows for game_id={gid}: {int((oc['game_id'].astype(str)==str(gid)).sum())}")
            if not noa.empty and "game_id" in noa.columns:
                print(f"NOAA weather rows for game_id={gid}: {int((noa['game_id'].astype(str)==str(gid)).sum())}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
