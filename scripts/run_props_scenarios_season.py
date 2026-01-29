"""Run deterministic scenario sims + player-props scenarios + accuracy checks week-by-week.

This is an orchestration/backfill helper. It does NOT change model training or base props
logic; it only ensures the scenario artifacts exist and runs the evaluation if actuals are
available.

Typical use:
  python scripts/run_props_scenarios_season.py --season 2025 --end-week 17 --n-sims 2000 --scenario-set v2

What it does per week W:
  1) Ensure backtests/<season>_wk<W>/sim_probs_scenarios.csv exists (run simulate_scenarios.py if missing)
  2) Ensure backtests/<season>_wk<W>/player_props_scenarios.csv exists (run simulate_player_props_scenarios.py)
  3) If nfl_compare/data/player_props_vs_actuals_<season>_wk<W>.csv exists, run player_props_scenarios_accuracy.py
  4) Append a compact summary row to a season-level CSV/MD report

Notes on "tuning":
- The current player-props scenarios only scale volume/yardage/TD expectations off team scoring
  environment shifts. This can improve robustness/coverage somewhat, but it cannot model player-level
  usage variance (targets distribution, role changes) by itself. The per-week reports help you see
  where that limitation matters so you can add new scenario knobs (or enhance the props model).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(REPO_ROOT / "nfl_compare" / "data"))).resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_current_week_marker() -> Optional[Tuple[int, int]]:
    # Try env first, then nfl_compare/data/current_week.json
    s = os.environ.get("CURRENT_SEASON") or os.environ.get("DEFAULT_SEASON")
    w = os.environ.get("CURRENT_WEEK") or os.environ.get("DEFAULT_WEEK")
    try:
        if s and w:
            return int(s), int(w)
    except Exception:
        pass

    fp = DATA_DIR / "current_week.json"
    if fp.exists():
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            ss = int(obj.get("season"))
            ww = int(obj.get("week"))
            if ss and ww:
                return ss, ww
        except Exception:
            return None
    return None


def _baseline_id_for_set(scenario_set: str) -> str:
    ss = (scenario_set or "").strip().lower()
    if ss == "v2":
        return "v2_baseline"
    if ss == "v1":
        return "v1_baseline"
    return "baseline"


def _run(cmd: List[str], *, cwd: Path, label: str, fail_ok: bool) -> int:
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {label}\n  {cmd_str}")
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), check=False)
        rc = int(proc.returncode)
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        rc = 1

    if rc != 0:
        msg = f"[WARN] {label} failed rc={rc}"
        if fail_ok:
            print(msg + " (continuing)")
        else:
            print(msg)
    return rc


@dataclass
class WeekSummary:
    season: int
    week: int
    scenario_set: str
    n_sims: int
    baseline_id: str
    baseline_mae_mean: Optional[float] = None
    baseline_brier_any_td: Optional[float] = None
    best_mae_scenario_id: Optional[str] = None
    best_mae_mean: Optional[float] = None
    coverage_rec_yards: Optional[float] = None
    coverage_targets: Optional[float] = None
    coverage_rush_yards: Optional[float] = None
    coverage_pass_yards: Optional[float] = None
    roster_matched_on_team_pct: Optional[float] = None
    roster_unmatched_pct: Optional[float] = None
    roster_ext_inactive_pct: Optional[float] = None
    roster_qb_starter_match_pct: Optional[float] = None
    note: str = ""


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _read_week_metrics(out_dir: Path, baseline_id: str) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    acc_fp = out_dir / "player_props_scenarios_accuracy.csv"
    cov_fp = out_dir / "player_props_scenarios_coverage.csv"

    metrics = None
    if acc_fp.exists():
        try:
            metrics = pd.read_csv(acc_fp)
        except Exception:
            metrics = None

    coverage: Dict[str, float] = {}
    if cov_fp.exists():
        try:
            cov = pd.read_csv(cov_fp)
            if not cov.empty and "stat" in cov.columns and "coverage_min_max" in cov.columns:
                for _, row in cov.iterrows():
                    st = str(row.get("stat") or "").strip()
                    cv = _safe_float(row.get("coverage_min_max"))
                    if st and cv is not None:
                        coverage[st] = float(cv)
        except Exception:
            coverage = {}

    return metrics, coverage


def _eval_outputs_present(out_dir: Path) -> bool:
    return (out_dir / "player_props_scenarios_accuracy.csv").exists() and (
        out_dir / "player_props_scenarios_coverage.csv"
    ).exists()


def _read_roster_validation_metrics(season: int, week: int) -> Dict[str, Optional[float]]:
    """Read roster_validation_summary_{season}_wk{week}.csv and compute compact quality metrics.

    These are best-effort sanity checks for weekly actives/roster alignment.
    """
    out: Dict[str, Optional[float]] = {
        "roster_matched_on_team_pct": None,
        "roster_unmatched_pct": None,
        "roster_ext_inactive_pct": None,
        "roster_qb_starter_match_pct": None,
    }

    fp = DATA_DIR / f"roster_validation_summary_{int(season)}_wk{int(week)}.csv"
    if not fp.exists():
        return out

    try:
        summ = pd.read_csv(fp)
    except Exception:
        return out

    if summ is None or summ.empty:
        return out

    try:
        total = pd.to_numeric(summ.get("total_model_players"), errors="coerce").fillna(0)
        matched = pd.to_numeric(summ.get("matched_on_team"), errors="coerce").fillna(0)
        unmatched = pd.to_numeric(summ.get("unmatched"), errors="coerce").fillna(0)
        ext_inactive = pd.to_numeric(summ.get("ext_inactive"), errors="coerce").fillna(0)
        qb_match = pd.to_numeric(summ.get("qb_starter_match"), errors="coerce").fillna(0)

        denom = float(total.sum()) if float(total.sum()) > 0 else 0.0
        if denom > 0:
            out["roster_matched_on_team_pct"] = float(matched.sum()) / denom
            out["roster_unmatched_pct"] = float(unmatched.sum()) / denom
            out["roster_ext_inactive_pct"] = float(ext_inactive.sum()) / denom
        # QB metric is team-level: percent of teams where our starter matches external starter
        out["roster_qb_starter_match_pct"] = float(qb_match.mean()) if len(qb_match) else None
    except Exception:
        return out

    return out


def _pick_best_by_mae_mean(metrics: pd.DataFrame) -> Optional[pd.Series]:
    if metrics is None or metrics.empty:
        return None
    if "mae_mean" not in metrics.columns:
        return None
    df = metrics.copy()
    df["mae_mean"] = pd.to_numeric(df["mae_mean"], errors="coerce")
    df = df.dropna(subset=["mae_mean"])
    if df.empty:
        return None
    return df.sort_values("mae_mean", ascending=True).head(1).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run scenario props artifacts + accuracy checks for a season")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, default=0, help="If 0, infer from current_week marker")
    ap.add_argument("--scenario-set", type=str, default="v2")
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--drives", action="store_true")
    ap.add_argument(
        "--roster-validate",
        action="store_true",
        help="Run weekly roster validation (uses nfl_data_py weekly rosters/depth charts) and record summary metrics",
    )
    ap.add_argument("--baseline-scenario-id", type=str, default="", help="Override baseline scenario id")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first non-zero exit code")
    ap.add_argument(
        "--stop-on-missing-actuals",
        action="store_true",
        help="Stop if player_props_vs_actuals_<season>_wk<week>.csv missing (to allow manual reconcile/tune)",
    )
    args = ap.parse_args()

    season = int(args.season)
    start_week = max(1, int(args.start_week))
    end_week = int(args.end_week)
    if end_week <= 0:
        marker = _infer_current_week_marker()
        if marker and int(marker[0]) == season:
            end_week = int(marker[1])
        else:
            print("ERROR: --end-week not provided and current_week marker not usable")
            return 2

    scenario_set = str(args.scenario_set or "v2")
    n_sims = max(200, int(args.n_sims))
    baseline_id = str(args.baseline_scenario_id).strip() or _baseline_id_for_set(scenario_set)

    season_out_dir = DATA_DIR / "backtests" / f"{season}_season_scenarios"
    season_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = season_out_dir / f"props_scenarios_accuracy_summary_thru_wk{end_week}.csv"
    out_md = season_out_dir / f"props_scenarios_accuracy_summary_thru_wk{end_week}.md"

    py = sys.executable

    rows: List[Dict[str, Any]] = []
    print(f"Season={season} Weeks={start_week}..{end_week} scenario_set={scenario_set} n_sims={n_sims} baseline_id={baseline_id}")

    for week in range(start_week, end_week + 1):
        bt_dir = DATA_DIR / "backtests" / f"{season}_wk{week}"
        bt_dir.mkdir(parents=True, exist_ok=True)

        sim_fp = bt_dir / "sim_probs_scenarios.csv"
        props_fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
        pps_fp = bt_dir / "player_props_scenarios.csv"
        actuals_fp = DATA_DIR / f"player_props_vs_actuals_{season}_wk{week}.csv"

        note = ""

        if not props_fp.exists():
            note = f"skip: missing props cache {props_fp.name}"
            print(f"\n[WEEK {week}] {note}")
            rows.append(WeekSummary(season, week, scenario_set, n_sims, baseline_id, note=note).__dict__)
            continue

        # 0) Optional roster validation / injury-actives sanity check
        if args.roster_validate:
            summ_fp = DATA_DIR / f"roster_validation_summary_{int(season)}_wk{int(week)}.csv"
            if args.force or (not summ_fp.exists()):
                cmd = [
                    py,
                    "-m",
                    "nfl_compare.src.roster_validation",
                    "--season",
                    str(season),
                    "--week",
                    str(week),
                ]
                rc = _run(cmd, cwd=REPO_ROOT, label=f"roster_validation wk{week}", fail_ok=True)
                if rc != 0:
                    # Don't fail the whole season run; external feeds can be flaky.
                    pass

        # 1) Sim scenarios
        if args.force or (not sim_fp.exists()):
            cmd = [
                py,
                "scripts/simulate_scenarios.py",
                "--season",
                str(season),
                "--week",
                str(week),
                "--n-sims",
                str(n_sims),
                "--scenario-set",
                scenario_set,
                "--out-dir",
                str(bt_dir),
            ]
            if args.drives:
                cmd.append("--drives")
            rc = _run(cmd, cwd=REPO_ROOT, label=f"simulate_scenarios wk{week}", fail_ok=not args.fail_fast)
            if rc != 0 and args.fail_fast:
                return rc

        # 2) Player props scenarios
        if args.force or (not pps_fp.exists()):
            cmd = [
                py,
                "scripts/simulate_player_props_scenarios.py",
                "--season",
                str(season),
                "--week",
                str(week),
                "--baseline-scenario-id",
                baseline_id,
                "--out-dir",
                str(bt_dir),
            ]
            rc = _run(cmd, cwd=REPO_ROOT, label=f"simulate_player_props_scenarios wk{week}", fail_ok=not args.fail_fast)
            if rc != 0 and args.fail_fast:
                return rc

        # 3) Evaluate (if actuals exist)
        if not actuals_fp.exists():
            note = f"no actuals: {actuals_fp.name}"
            print(f"\n[WEEK {week}] {note}")
            if args.stop_on_missing_actuals:
                rows.append(WeekSummary(season, week, scenario_set, n_sims, baseline_id, note=note).__dict__)
                break
            rows.append(WeekSummary(season, week, scenario_set, n_sims, baseline_id, note=note).__dict__)
            continue

        if args.force or (not _eval_outputs_present(bt_dir)):
            cmd = [
                py,
                "scripts/player_props_scenarios_accuracy.py",
                "--season",
                str(season),
                "--week",
                str(week),
                "--out-dir",
                str(bt_dir),
            ]
            rc = _run(cmd, cwd=REPO_ROOT, label=f"player_props_scenarios_accuracy wk{week}", fail_ok=not args.fail_fast)
            if rc != 0 and args.fail_fast:
                return rc

        metrics, coverage = _read_week_metrics(bt_dir, baseline_id)

        baseline_mae = None
        baseline_brier = None
        if metrics is not None and not metrics.empty and "scenario_id" in metrics.columns:
            m0 = metrics.copy()
            m0 = m0[m0["scenario_id"].astype(str) == str(baseline_id)]
            if not m0.empty:
                baseline_mae = _safe_float(m0.iloc[0].get("mae_mean"))
                baseline_brier = _safe_float(m0.iloc[0].get("brier_any_td"))

        best = _pick_best_by_mae_mean(metrics) if metrics is not None else None
        best_id = str(best.get("scenario_id")) if best is not None and "scenario_id" in best else None
        best_mae = _safe_float(best.get("mae_mean")) if best is not None else None

        ws = WeekSummary(
            season=season,
            week=week,
            scenario_set=scenario_set,
            n_sims=n_sims,
            baseline_id=baseline_id,
            baseline_mae_mean=baseline_mae,
            baseline_brier_any_td=baseline_brier,
            best_mae_scenario_id=best_id,
            best_mae_mean=best_mae,
            coverage_rec_yards=_safe_float(coverage.get("rec_yards")),
            coverage_targets=_safe_float(coverage.get("targets")),
            coverage_rush_yards=_safe_float(coverage.get("rush_yards")),
            coverage_pass_yards=_safe_float(coverage.get("pass_yards")),
            **(_read_roster_validation_metrics(season, week) if args.roster_validate else {}),
            note=note,
        )

        print(
            f"\n[WEEK {week}] baseline_mae_mean={ws.baseline_mae_mean} best_mae_mean={ws.best_mae_mean} "
            f"cov(rec_yards)={ws.coverage_rec_yards} cov(targets)={ws.coverage_targets}"
        )

        rows.append(ws.__dict__)

        # Write partial report each week (useful when interrupted)
        try:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_csv, index=False)
        except Exception:
            pass

    # Final write
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    # Small markdown summary
    try:
        lines: List[str] = []
        lines.append(f"# Props Scenarios Accuracy Summary")
        lines.append("")
        lines.append(f"- Season: {season}")
        lines.append(f"- Weeks: {start_week}..{end_week}")
        lines.append(f"- Scenario set: {scenario_set}")
        lines.append(f"- n_sims: {n_sims}")
        lines.append(f"- Baseline scenario id: {baseline_id}")
        lines.append(f"- Generated: {_utc_now_iso()}")
        lines.append("")

        cols = [
            "week",
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
        view = df_out.copy()
        for c in cols:
            if c not in view.columns:
                view[c] = None
        view = view[cols]

        # Markdown table (simple)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in view.iterrows():
            row = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                elif v is None or (isinstance(v, float) and pd.isna(v)):
                    row.append("")
                else:
                    row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")

        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote season summary: {out_csv}")
        print(f"Wrote season summary: {out_md}")
    except Exception as e:
        print(f"WARN: markdown write failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
