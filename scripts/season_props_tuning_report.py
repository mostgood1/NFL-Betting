"""Aggregate weekly scenario-props evaluation into a tuning-oriented season report.

This does NOT auto-tune model parameters. It summarizes:
- Which stats have the weakest min..max scenario envelope coverage
- Which weeks are worst by baseline MAE / Brier
- Which players recur as top error contributors

Typical usage:
  python scripts/season_props_tuning_report.py --season 2025 --end-week 19 --scenario-set v2

Outputs (default):
  nfl_compare/data/backtests/<season>_season_scenarios/props_scenarios_tuning_report_thru_wk<end>.{md,csv}

Notes:
- This is meant to guide the *next* tuning step (e.g. usage/role-shock scenarios,
  depth_overrides cleanup, QB/pass-volume priors). It won’t magically make each
  historical week better on re-run.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(REPO_ROOT / "nfl_compare" / "data"))).resolve()


@dataclass
class WeekInputs:
    season: int
    week: int
    backtest_dir: Path


def _season_summary_fp(season: int, end_week: int) -> Path:
    return DATA_DIR / "backtests" / f"{int(season)}_season_scenarios" / f"props_scenarios_accuracy_summary_thru_wk{int(end_week)}.csv"


def _week_dir(season: int, week: int) -> Path:
    return DATA_DIR / "backtests" / f"{int(season)}_wk{int(week)}"


def _read_csv(fp: Path) -> Optional[pd.DataFrame]:
    try:
        if fp.exists():
            return pd.read_csv(fp)
    except Exception:
        return None
    return None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return ""


def _fmt_float(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def _top_errors_for_week(season: int, week: int, scenario_id: str, limit: int = 25) -> pd.DataFrame:
    fp = _week_dir(season, week) / "player_props_scenarios_top_errors.csv"
    df = _read_csv(fp)
    if df is None or df.empty:
        return pd.DataFrame()
    if "scenario_id" in df.columns:
        df = df[df["scenario_id"].astype(str) == str(scenario_id)].copy()
    if "abs_error_sum" in df.columns:
        df["abs_error_sum"] = pd.to_numeric(df["abs_error_sum"], errors="coerce")
        df = df.dropna(subset=["abs_error_sum"]).sort_values("abs_error_sum", ascending=False)
    return df.head(int(limit))


def _coverage_for_week(season: int, week: int) -> pd.DataFrame:
    fp = _week_dir(season, week) / "player_props_scenarios_coverage.csv"
    df = _read_csv(fp)
    if df is None or df.empty:
        return pd.DataFrame()
    if "coverage_min_max" in df.columns:
        df["coverage_min_max"] = pd.to_numeric(df["coverage_min_max"], errors="coerce")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a tuning-oriented season report from weekly props-scenarios eval")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--scenario-set", type=str, default="v2")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    season = int(args.season)
    start_week = max(1, int(args.start_week))
    end_week = int(args.end_week)

    summ_fp = _season_summary_fp(season, end_week)
    summ = _read_csv(summ_fp)
    if summ is None or summ.empty:
        raise SystemExit(f"Missing season summary: {summ_fp}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (summ_fp.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"props_scenarios_tuning_report_thru_wk{end_week}.csv"
    out_md = out_dir / f"props_scenarios_tuning_report_thru_wk{end_week}.md"

    # Filter to actuals weeks
    has_actuals = summ["baseline_mae_mean"].notna().copy()
    actuals_weeks = summ.loc[has_actuals, "week"].astype(int).tolist()
    actuals_weeks = [w for w in actuals_weeks if start_week <= w <= end_week]

    # 1) Coverage aggregation
    cov_rows: List[Dict] = []
    for w in actuals_weeks:
        cov = _coverage_for_week(season, w)
        if cov is None or cov.empty:
            continue
        if "stat" not in cov.columns or "coverage_min_max" not in cov.columns:
            continue
        for _, r in cov.iterrows():
            cov_rows.append({
                "week": int(w),
                "stat": str(r.get("stat") or "").strip(),
                "coverage_min_max": float(pd.to_numeric(r.get("coverage_min_max"), errors="coerce")) if pd.notna(pd.to_numeric(r.get("coverage_min_max"), errors="coerce")) else None,
            })

    cov_df = pd.DataFrame(cov_rows)
    cov_summary = pd.DataFrame()
    if not cov_df.empty:
        cov_summary = (
            cov_df.dropna(subset=["coverage_min_max"])  # type: ignore[arg-type]
            .groupby("stat", as_index=False)["coverage_min_max"]
            .agg(["mean", "median", "min", "max", "count"])  # type: ignore[call-arg]
            .reset_index()
            .rename(columns={"mean": "coverage_mean", "median": "coverage_median", "min": "coverage_min", "max": "coverage_max", "count": "n_weeks"})
        )
        cov_summary = cov_summary.sort_values(["coverage_mean", "coverage_median"], ascending=True)

    # 2) Top error aggregation (baseline only)
    top_n = max(5, int(args.top_n))
    baseline_errors: List[Tuple[str, str, str]] = []  # (player, position, stat_bucket)
    player_counter: Counter[str] = Counter()
    pos_counter: Counter[str] = Counter()
    stat_counter: Counter[str] = Counter()

    for w in actuals_weeks:
        row = summ[summ["week"].astype(int) == int(w)]
        if row.empty:
            continue
        baseline_id = str(row.iloc[0].get("baseline_id") or "baseline")
        te = _top_errors_for_week(season, w, baseline_id, limit=top_n)
        if te is None or te.empty:
            continue
        for _, r in te.iterrows():
            player = str(r.get("player") or "").strip()
            pos = str(r.get("position") or "").strip().upper()
            # crude: infer which stat contributed by looking at which act column exists and differs most
            # keep it simple: bucket based on presence of columns
            stat_bucket = "mixed"
            for st in ["rec_yards", "targets", "rush_yards", "pass_yards"]:
                if st in te.columns and f"{st}_act" in te.columns:
                    # these rows always include all four stats, so just bucket by the largest abs delta
                    try:
                        deltas = {}
                        for s2 in ["rec_yards", "targets", "rush_yards", "pass_yards"]:
                            v1 = pd.to_numeric(r.get(s2), errors="coerce")
                            v2 = pd.to_numeric(r.get(f"{s2}_act"), errors="coerce")
                            if pd.notna(v1) and pd.notna(v2):
                                deltas[s2] = float(abs(float(v1) - float(v2)))
                        if deltas:
                            stat_bucket = max(deltas.items(), key=lambda kv: kv[1])[0]
                    except Exception:
                        stat_bucket = "mixed"
                    break

            if player:
                player_counter[player] += 1
            if pos:
                pos_counter[pos] += 1
            if stat_bucket:
                stat_counter[stat_bucket] += 1
            baseline_errors.append((player, pos, stat_bucket))

    # 3) Build compact export CSV (weeks + key columns)
    week_cols = [
        "season",
        "week",
        "scenario_set",
        "baseline_id",
        "baseline_mae_mean",
        "baseline_brier_any_td",
        "best_mae_scenario_id",
        "best_mae_mean",
        "coverage_targets",
        "coverage_rec_yards",
        "coverage_rush_yards",
        "coverage_pass_yards",
        "roster_matched_on_team_pct",
        "roster_unmatched_pct",
        "roster_ext_inactive_pct",
        "roster_qb_starter_match_pct",
        "note",
    ]
    out_weeks = summ.copy()
    for c in week_cols:
        if c not in out_weeks.columns:
            out_weeks[c] = pd.NA
    out_weeks = out_weeks[(out_weeks["week"].astype(int) >= start_week) & (out_weeks["week"].astype(int) <= end_week)]
    out_weeks = out_weeks[week_cols]
    out_weeks.to_csv(out_csv, index=False)

    # 4) Markdown report
    lines: List[str] = []
    lines.append("# Props Scenarios Tuning Report")
    lines.append("")
    lines.append(f"- Season: {season}")
    lines.append(f"- Weeks: {start_week}..{end_week}")
    lines.append(f"- Scenario set: {args.scenario_set}")
    lines.append(f"- Actuals weeks: {len(actuals_weeks)}")
    lines.append(f"- Source: {summ_fp.as_posix()}")
    lines.append("")

    # Worst weeks table
    try:
        worst = summ[summ["baseline_mae_mean"].notna()].copy()
        worst["baseline_mae_mean"] = pd.to_numeric(worst["baseline_mae_mean"], errors="coerce")
        worst = worst.dropna(subset=["baseline_mae_mean"]).sort_values("baseline_mae_mean", ascending=False).head(8)
        if not worst.empty:
            lines.append("## Worst Weeks (Baseline MAE)")
            lines.append("")
            cols = ["week", "baseline_mae_mean", "best_mae_mean", "coverage_targets", "coverage_rec_yards"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, r in worst.iterrows():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(int(r.get("week"))),
                            _fmt_float(r.get("baseline_mae_mean")),
                            _fmt_float(r.get("best_mae_mean")),
                            _fmt_pct(r.get("coverage_targets")),
                            _fmt_pct(r.get("coverage_rec_yards")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    except Exception:
        pass

    # Coverage summary
    if cov_summary is not None and not cov_summary.empty:
        lines.append("## Coverage (Min..Max Envelope)")
        lines.append("")
        lines.append("Lowest-average coverage stats (scenario envelope misses actuals often):")
        lines.append("")
        view = cov_summary.head(12)
        cols = ["stat", "coverage_mean", "coverage_median", "n_weeks"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in view.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("stat")),
                        _fmt_pct(r.get("coverage_mean")),
                        _fmt_pct(r.get("coverage_median")),
                        str(int(r.get("n_weeks") or 0)),
                    ]
                )
                + " |"
            )
        lines.append("")

    # Recurring top errors
    lines.append("## Recurring Top Errors (Baseline)")
    lines.append("")
    if player_counter:
        lines.append("Most frequent players appearing in per-week top error list:")
        lines.append("")
        lines.append("| player | appearances |")
        lines.append("| --- | --- |")
        for player, n in player_counter.most_common(15):
            lines.append(f"| {player} | {n} |")
        lines.append("")

    if pos_counter:
        lines.append("Top-error positions (count of appearances):")
        lines.append("")
        lines.append("| position | appearances |")
        lines.append("| --- | --- |")
        for pos, n in pos_counter.most_common():
            lines.append(f"| {pos} | {n} |")
        lines.append("")

    if stat_counter:
        lines.append("Top-error stat buckets (largest delta within row):")
        lines.append("")
        lines.append("| stat | appearances |")
        lines.append("| --- | --- |")
        for st, n in stat_counter.most_common():
            lines.append(f"| {st} | {n} |")
        lines.append("")

    # Roster validation (if present)
    if "roster_unmatched_pct" in summ.columns and summ["roster_unmatched_pct"].notna().any():
        lines.append("## Roster / Actives Sanity")
        lines.append("")
        lines.append("These come from `nfl_compare/src/roster_validation.py` (nfl_data_py weekly rosters + depth charts).")
        lines.append("")
        view = summ[summ["baseline_mae_mean"].notna()].copy()
        view = view.sort_values("roster_unmatched_pct", ascending=False).head(8)
        cols = ["week", "roster_matched_on_team_pct", "roster_unmatched_pct", "roster_ext_inactive_pct", "roster_qb_starter_match_pct"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in view.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(r.get("week"))),
                        _fmt_pct(r.get("roster_matched_on_team_pct")),
                        _fmt_pct(r.get("roster_unmatched_pct")),
                        _fmt_pct(r.get("roster_ext_inactive_pct")),
                        _fmt_pct(r.get("roster_qb_starter_match_pct")),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
