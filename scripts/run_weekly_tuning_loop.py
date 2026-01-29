"""Walk-forward weekly loop: props -> reconcile -> scenarios -> accuracy (+ roster/injury sanity).

Goal
- Provide a deterministic, repeatable workflow that can *actually improve* week N+1
  by regenerating props for week N+1 using the model's existing reconciliation-driven
  calibration (which keys off week N actuals).

What this script does per week W:
1) (Optional) Archive existing player_props_{season}_wk{W}.csv
2) Generate props cache via scripts/gen_props.py (compute_player_props)
   - For W>1, compute_player_props uses prior-week reconciliation summaries when
     player_props_vs_actuals_{season}_wk{W-1}.csv exists.
3) Reconcile week W props vs actuals (writes player_props_vs_actuals_{season}_wk{W}.csv)
   - Uses in-package reconciliation (more reliable than standalone script).
4) Run scenario sims + props-scenarios + accuracy evaluation for week W
5) Run roster validation for week W (weekly rosters + depth charts) and record summary metrics

Outputs
- Per-week artifacts under nfl_compare/data/backtests/{season}_wk{week}/ (existing scripts)
- Rolling summary under nfl_compare/data/backtests/{season}_season_tuning/

Notes
- This is a *workflow* tuner, not an auto-optimizer. It relies on your existing
  calibration logic in nfl_compare/src/player_props.py.
- Use --dry-run to see commands without modifying files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(REPO_ROOT / "nfl_compare" / "data"))).resolve()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    timeout_sec: Optional[int] = None,
) -> int:
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {label}\n  {cmd_str}")
    if dry_run:
        return 0
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False, timeout=timeout_sec)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(f"[WARN] {label}: timed out after {timeout_sec}s")
        return 124
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return 1


def _archive_file(fp: Path, *, archive_dir: Path, stamp: str, dry_run: bool) -> Optional[Path]:
    if not fp.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    out = archive_dir / f"{fp.stem}__arch_{stamp}{fp.suffix}"
    if dry_run:
        print(f"[DRY] archive {fp.name} -> {out}")
        return out
    shutil.copy2(fp, out)
    return out


def _reconcile_in_package(season: int, week: int, *, dry_run: bool) -> Optional[Path]:
    """Write player_props_vs_actuals_{season}_wk{week}.csv using in-package reconciliation."""
    out_fp = DATA_DIR / f"player_props_vs_actuals_{int(season)}_wk{int(week)}.csv"
    if dry_run:
        print(f"[DRY] reconcile -> {out_fp}")
        return out_fp

    try:
        from nfl_compare.src.reconciliation import reconcile_props

        df = reconcile_props(int(season), int(week))
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_fp, index=False)
        print(f"Wrote {out_fp} rows={len(df)}")
        return out_fp
    except Exception as e:
        print(f"[WARN] reconcile failed for {season} wk{week}: {e}")
        return None


def _read_roster_validation_metrics(season: int, week: int) -> Dict[str, Optional[float]]:
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
        out["roster_qb_starter_match_pct"] = float(qb_match.mean()) if len(qb_match) else None
    except Exception:
        return out

    return out


@dataclass
class WeekRow:
    season: int
    week: int
    scenario_set: str
    n_sims: int
    props_rows: Optional[int] = None
    reconcile_rows: Optional[int] = None
    baseline_mae_mean: Optional[float] = None
    best_mae_mean: Optional[float] = None
    coverage_targets: Optional[float] = None
    coverage_rec_yards: Optional[float] = None
    roster_matched_on_team_pct: Optional[float] = None
    roster_unmatched_pct: Optional[float] = None
    roster_qb_starter_match_pct: Optional[float] = None
    note: str = ""


def _read_eval_summary(season: int, week: int) -> Dict[str, Optional[float]]:
    bt_dir = DATA_DIR / "backtests" / f"{int(season)}_wk{int(week)}"
    acc_fp = bt_dir / "player_props_scenarios_accuracy.csv"
    cov_fp = bt_dir / "player_props_scenarios_coverage.csv"

    out: Dict[str, Optional[float]] = {
        "baseline_mae_mean": None,
        "best_mae_mean": None,
        "coverage_targets": None,
        "coverage_rec_yards": None,
    }

    try:
        if acc_fp.exists():
            acc = pd.read_csv(acc_fp)
            if not acc.empty and "mae_mean" in acc.columns:
                acc["mae_mean"] = pd.to_numeric(acc["mae_mean"], errors="coerce")
                # baseline is any row with 'baseline' in id if present, else the min row
                if "scenario_id" in acc.columns:
                    base = acc[acc["scenario_id"].astype(str).str.contains("baseline", case=False, na=False)]
                    if not base.empty:
                        out["baseline_mae_mean"] = float(base.sort_values("mae_mean").iloc[0]["mae_mean"])
                best = acc.dropna(subset=["mae_mean"]).sort_values("mae_mean").head(1)
                if not best.empty:
                    out["best_mae_mean"] = float(best.iloc[0]["mae_mean"])
    except Exception:
        pass

    try:
        if cov_fp.exists():
            cov = pd.read_csv(cov_fp)
            if not cov.empty and {"stat", "coverage_min_max"}.issubset(cov.columns):
                m = {str(r.get("stat")): float(pd.to_numeric(r.get("coverage_min_max"), errors="coerce")) for _, r in cov.iterrows()}
                out["coverage_targets"] = m.get("targets")
                out["coverage_rec_yards"] = m.get("rec_yards")
    except Exception:
        pass

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run weekly walk-forward props tuning loop")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--scenario-set", type=str, default="v2")
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--drives", action="store_true")
    ap.add_argument("--skip-scenarios", action="store_true")
    ap.add_argument("--skip-reconcile", action="store_true")
    ap.add_argument("--skip-roster-validate", action="store_true")
    ap.add_argument(
        "--prefetch-rosters",
        action="store_true",
        help="Best-effort prefetch nfl_data_py roster caches once per run",
    )
    ap.add_argument(
        "--roster-validate-timeout-sec",
        type=int,
        default=90,
        help="Hard timeout for roster validation subprocess (prevents hangs on external fetch)",
    )
    ap.add_argument("--archive", action="store_true", help="Archive existing props before overwriting")
    ap.add_argument("--force", action="store_true", help="Force regenerate scenario outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    season = int(args.season)
    start_week = max(1, int(args.start_week))
    end_week = int(args.end_week)
    scenario_set = str(args.scenario_set or "v2")
    n_sims = max(200, int(args.n_sims))

    stamp = _utc_stamp()
    py = sys.executable

    season_out_dir = DATA_DIR / "backtests" / f"{season}_season_tuning"
    season_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = season_out_dir / f"weekly_tuning_summary_thru_wk{end_week}__{stamp}.csv"

    if args.prefetch_rosters and (not args.dry_run):
        try:
            from nfl_compare.src.roster_cache import get_seasonal_rosters, get_weekly_rosters

            _ = get_seasonal_rosters(season)
            _ = get_weekly_rosters(season)
            print(f"[INFO] Prefetched roster caches for season={season}")
        except Exception as e:
            print(f"[WARN] Prefetch rosters failed: {e}")

    rows: List[Dict[str, Any]] = []

    for week in range(start_week, end_week + 1):
        note = ""
        props_fp = DATA_DIR / f"player_props_{season}_wk{week}.csv"
        arch_dir = season_out_dir / "archives" / f"wk{week}"

        if args.archive:
            _archive_file(props_fp, archive_dir=arch_dir, stamp=stamp, dry_run=args.dry_run)

        # 1) Generate props (will use prior-week reconciliation calibration if present)
        rc = _run([py, "scripts/gen_props.py", str(season), str(week)], cwd=REPO_ROOT, label=f"gen_props wk{week}", dry_run=args.dry_run)
        props_regen_ok = (rc == 0)
        if rc != 0:
            note = f"gen_props_failed rc={rc}"
            print(f"[WEEK {week}] {note}")

        # Count props rows
        props_rows = None
        try:
            if props_fp.exists():
                props_rows = len(pd.read_csv(props_fp))
        except Exception:
            props_rows = None

        # 2) Reconcile vs actuals
        reconcile_rows = None
        if not args.skip_reconcile:
            recon_fp = _reconcile_in_package(season, week, dry_run=args.dry_run)
            if recon_fp is not None and recon_fp.exists():
                try:
                    reconcile_rows = len(pd.read_csv(recon_fp))
                except Exception:
                    reconcile_rows = None

        # 3) Scenario sims + props scenarios + accuracy
        if not args.skip_scenarios:
            bt_dir = DATA_DIR / "backtests" / f"{season}_wk{week}"
            bt_dir.mkdir(parents=True, exist_ok=True)

            sim_fp = bt_dir / "sim_probs_scenarios.csv"
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
                _run(cmd, cwd=REPO_ROOT, label=f"simulate_scenarios wk{week}", dry_run=args.dry_run)

            # Because we just regenerated base props, always refresh derived artifacts.
            if props_regen_ok or args.force:
                _run(
                    [
                        py,
                        "scripts/simulate_player_props_scenarios.py",
                        "--season",
                        str(season),
                        "--week",
                        str(week),
                        "--baseline-scenario-id",
                        "v2_baseline" if scenario_set.strip().lower() == "v2" else "baseline",
                        "--out-dir",
                        str(bt_dir),
                    ],
                    cwd=REPO_ROOT,
                    label=f"simulate_player_props_scenarios wk{week}",
                    dry_run=args.dry_run,
                )

                _run(
                    [
                        py,
                        "scripts/player_props_scenarios_accuracy.py",
                        "--season",
                        str(season),
                        "--week",
                        str(week),
                        "--out-dir",
                        str(bt_dir),
                    ],
                    cwd=REPO_ROOT,
                    label=f"player_props_scenarios_accuracy wk{week}",
                    dry_run=args.dry_run,
                )

        # 4) Roster validation
        if not args.skip_roster_validate:
            summ_fp = DATA_DIR / f"roster_validation_summary_{season}_wk{week}.csv"
            if props_regen_ok or args.force or (not summ_fp.exists()):
                _run(
                    [py, "-m", "nfl_compare.src.roster_validation", "--season", str(season), "--week", str(week)],
                    cwd=REPO_ROOT,
                    label=f"roster_validation wk{week}",
                    dry_run=args.dry_run,
                    timeout_sec=max(10, int(args.roster_validate_timeout_sec)),
                )

        evals = _read_eval_summary(season, week)
        roster = _read_roster_validation_metrics(season, week) if not args.skip_roster_validate else {}

        row = WeekRow(
            season=season,
            week=week,
            scenario_set=scenario_set,
            n_sims=n_sims,
            props_rows=props_rows,
            reconcile_rows=reconcile_rows,
            baseline_mae_mean=evals.get("baseline_mae_mean"),
            best_mae_mean=evals.get("best_mae_mean"),
            coverage_targets=evals.get("coverage_targets"),
            coverage_rec_yards=evals.get("coverage_rec_yards"),
            roster_matched_on_team_pct=roster.get("roster_matched_on_team_pct"),
            roster_unmatched_pct=roster.get("roster_unmatched_pct"),
            roster_qb_starter_match_pct=roster.get("roster_qb_starter_match_pct"),
            note=note,
        )

        print(
            f"\n[WEEK {week}] props_rows={props_rows} recon_rows={reconcile_rows} "
            f"mae={row.baseline_mae_mean} cov_targets={row.coverage_targets} roster_unmatched={row.roster_unmatched_pct}"
        )

        rows.append(row.__dict__)
        try:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        except Exception:
            pass

    print(f"\nWrote tuning summary: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
