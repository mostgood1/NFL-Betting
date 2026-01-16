from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_compare.src.data_sources import load_games, load_lines, load_team_stats
from nfl_compare.src.features import merge_features
from nfl_compare.src.weather import load_weather_for_games


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
    if df is None or df.empty:
        return df
    if "game_id" not in df.columns:
        return df
    cols = [c for c in ["season", "week", "game_id"] if c in df.columns]
    if cols:
        return df.sort_values(cols).drop_duplicates(subset=["game_id"], keep="first")
    return df.drop_duplicates(subset=["game_id"], keep="first")


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "(no rows)\n"
    work = df.copy()
    cols = [c for c in cols if c in work.columns]
    if not cols:
        return "(no columns)\n"
    work = work[cols].head(max_rows)

    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.3f}".rstrip("0").rstrip(".")
        return str(v)

    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = "".join(
        "| " + " | ".join(fmt(v) for v in row) + " |\n" for row in work.itertuples(index=False, name=None)
    )
    return header + sep + rows


def inspect(season: int, weeks: Iterable[int]) -> dict:
    weeks = sorted(set(int(w) for w in weeks))
    if not weeks:
        return {"error": "No weeks provided"}

    games = load_games()
    lines = load_lines()
    stats = load_team_stats()
    wx = load_weather_for_games(games)

    feat = merge_features(games, stats, lines, wx).copy()
    for c in ["season", "week"]:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")

    feat = _best_row_per_game(feat)
    feat = feat[(feat["season"] == int(season)) & (feat["week"].isin(weeks))].copy()
    if feat.empty:
        return {"error": f"No rows for season={season} weeks={weeks}"}

    # Ensure injury columns exist
    inj_cols = [
        "home_inj_qb_out",
        "away_inj_qb_out",
        "home_inj_rb1_out",
        "away_inj_rb1_out",
        "home_inj_te1_out",
        "away_inj_te1_out",
        "home_inj_wr_top2_out",
        "away_inj_wr_top2_out",
        "home_inj_starters_out",
        "away_inj_starters_out",
    ]
    missing_cols = [c for c in inj_cols if c not in feat.columns]

    for c in ["home_inj_starters_out", "away_inj_starters_out"]:
        if c not in feat.columns:
            feat[c] = np.nan

    feat["inj_starters_out_total"] = _coerce_num(feat["home_inj_starters_out"]).fillna(0) + _coerce_num(
        feat["away_inj_starters_out"]
    ).fillna(0)
    feat["inj_starters_out_abs_diff"] = (
        _coerce_num(feat["home_inj_starters_out"]).fillna(0) - _coerce_num(feat["away_inj_starters_out"]).fillna(0)
    ).abs()

    # Per-week summaries
    def _week_summ(col: str) -> pd.DataFrame:
        s = _coerce_num(feat[col])
        g = feat.copy()
        g["_v"] = s
        out = (
            g.groupby("week")["_v"]
            .agg([
                ("n_games", "size"),
                ("mean", "mean"),
                ("p_nonzero", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
                ("max", "max"),
            ])
            .reset_index()
        )
        return out

    wk_home = _week_summ("home_inj_starters_out")
    wk_away = _week_summ("away_inj_starters_out")
    wk_total = _week_summ("inj_starters_out_total")

    # Top games
    top_total = feat.sort_values(["inj_starters_out_total", "inj_starters_out_abs_diff"], ascending=False).head(15).copy()
    top_absdiff = feat.sort_values(["inj_starters_out_abs_diff", "inj_starters_out_total"], ascending=False).head(15).copy()

    # Baseline-week check
    try:
        baseline_week = int(pd.to_numeric(os.environ.get("INJURY_BASELINE_WEEK", 1), errors="coerce"))
        if baseline_week < 1:
            baseline_week = 1
    except Exception:
        baseline_week = 1
    bw = feat[feat["week"] == baseline_week]
    baseline_nonzero = None
    if not bw.empty:
        h = _coerce_num(bw["home_inj_starters_out"]).fillna(0)
        a = _coerce_num(bw["away_inj_starters_out"]).fillna(0)
        baseline_nonzero = {
            "week": int(baseline_week),
            "n_games": int(len(bw)),
            "p_home_nonzero": float((h > 0).mean()),
            "p_away_nonzero": float((a > 0).mean()),
            "p_any_nonzero": float(((h + a) > 0).mean()),
            "max_total": float((h + a).max()),
        }

    return {
        "feat": feat,
        "missing_cols": missing_cols,
        "wk_home": wk_home,
        "wk_away": wk_away,
        "wk_total": wk_total,
        "top_total": top_total,
        "top_absdiff": top_absdiff,
        "baseline_check": baseline_nonzero,
        "baseline_week": baseline_week,
    }


def write_report(res: dict, season: int, weeks: list[int], out_fp: Path) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    baseline_week = res.get("baseline_week")
    missing_cols = res.get("missing_cols") or []

    weeks_str = f"{min(weeks)}-{max(weeks)}" if weeks else ""

    lines: list[str] = []
    lines.append(f"# Injury Feature Sanity Check — Season {season}, Weeks {weeks_str}\n")
    lines.append("## Context\n")
    lines.append(f"- INJURY_BASELINE_WEEK: `{baseline_week}`\n")
    lines.append(f"- Weeks requested: `{weeks}`\n")
    if missing_cols:
        lines.append(f"- Missing injury columns in merged features: `{missing_cols}`\n")
    else:
        lines.append("- Missing injury columns in merged features: `[]`\n")

    if res.get("baseline_check"):
        bc = res["baseline_check"]
        lines.append("\n## Baseline Week Self-Consistency\n")
        lines.append(
            f"Baseline week = {bc['week']}. Percent non-zero (should be low-ish): "
            f"home={bc['p_home_nonzero']:.3f}, away={bc['p_away_nonzero']:.3f}, any={bc['p_any_nonzero']:.3f}. "
            f"Max total starters-out in a game: {bc['max_total']}.\n"
        )

    lines.append("\n## Per-Week Summary (home_inj_starters_out)\n")
    lines.append(_md_table(res["wk_home"], ["week", "n_games", "mean", "p_nonzero", "max"], max_rows=50))

    lines.append("\n## Per-Week Summary (away_inj_starters_out)\n")
    lines.append(_md_table(res["wk_away"], ["week", "n_games", "mean", "p_nonzero", "max"], max_rows=50))

    lines.append("\n## Per-Week Summary (total starters-out in game)\n")
    lines.append(_md_table(res["wk_total"], ["week", "n_games", "mean", "p_nonzero", "max"], max_rows=50))

    lines.append("\n## Top Games by Total Starters Out\n")
    cols = [
        "week",
        "game_id",
        "away_team",
        "home_team",
        "home_inj_starters_out",
        "away_inj_starters_out",
        "inj_starters_out_total",
        "inj_starters_out_abs_diff",
    ]
    lines.append(_md_table(res["top_total"], cols, max_rows=15))

    lines.append("\n## Top Games by Abs(Home-Away) Starters Out\n")
    lines.append(_md_table(res["top_absdiff"], cols, max_rows=15))

    out_fp.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect injury features (baseline-starters-out) across weeks")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", type=str, required=True, help='"1-4" or "10-18" or "1,2,3"')
    ap.add_argument("--out", type=str, default=None, help="Output markdown path (default reports/injury_sanity_...) ")
    args = ap.parse_args(argv)

    weeks = _parse_weeks_arg(args.weeks)
    if not weeks:
        raise SystemExit("No weeks parsed from --weeks")

    res = inspect(int(args.season), weeks)
    if "error" in res:
        print(res["error"])
        return 1

    if args.out:
        out_fp = Path(args.out)
    else:
        out_fp = Path("reports") / f"injury_sanity_{int(args.season)}_wk{min(weeks)}-{max(weeks)}.md"

    write_report(res, int(args.season), weeks, out_fp)
    print(f"Wrote {out_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
