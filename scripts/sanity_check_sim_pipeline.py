"""Sanity checks for the simulation pipeline (games -> sims -> quarters/drives).

Goal: catch obvious regressions (totals collapsing, artifacts inconsistent) fast.

Usage:
  python scripts/sanity_check_sim_pipeline.py --season 2025 --week 20

Exit codes:
  0 = ok
  1 = warnings/errors detected

Notes:
- This script is intentionally defensive: missing optional artifacts produce warnings, not crashes.
- It does not attempt to judge "model correctness"; it enforces internal consistency and basic plausibility.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"


@dataclass
class Issue:
    level: str  # WARN | ERROR
    game_id: Optional[str]
    msg: str


def _read_csv(fp: Path) -> pd.DataFrame:
    try:
        if not fp.exists():
            return pd.DataFrame()
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _market_total_from_lines(lines: pd.DataFrame) -> pd.Series:
    ct = pd.to_numeric(lines.get("close_total"), errors="coerce")
    tt = pd.to_numeric(lines.get("total"), errors="coerce")
    return ct.where(ct.notna(), tt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--backtests-dir", type=str, default=None, help="Override backtests folder (default nfl_compare/data/backtests/{season}_wk{week})")
    ap.add_argument("--max-total-gap", type=float, default=20.0, help="Warn if |pred_total - market_total| exceeds this")
    ap.add_argument("--max-artifact-gap", type=float, default=2.0, help="Warn if means disagree across artifacts by more than this")
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)

    issues: list[Issue] = []

    pred_fp = DATA_DIR / "predictions_week.csv"
    lines_fp = DATA_DIR / "lines.csv"

    bt_dir = Path(args.backtests_dir) if args.backtests_dir else (DATA_DIR / "backtests" / f"{season}_wk{week}")
    sim_probs_fp = bt_dir / "sim_probs.csv"
    sim_quarters_fp = bt_dir / "sim_quarters.csv"
    sim_drives_fp = bt_dir / "sim_drives.csv"

    pred = _read_csv(pred_fp)
    lines = _read_csv(lines_fp)
    sim_probs = _read_csv(sim_probs_fp)
    sim_q = _read_csv(sim_quarters_fp)
    sim_d = _read_csv(sim_drives_fp)

    if pred.empty:
        issues.append(Issue("ERROR", None, f"Missing/empty {pred_fp}"))
    if lines.empty:
        issues.append(Issue("WARN", None, f"Missing/empty {lines_fp} (market comparisons skipped)"))
    if sim_probs.empty:
        issues.append(Issue("ERROR", None, f"Missing/empty {sim_probs_fp}"))

    if issues and any(i.level == "ERROR" for i in issues):
        for i in issues:
            print(f"[{i.level}] {i.game_id or ''} {i.msg}".strip())
        return 1

    # Normalize keys
    for df in (pred, lines, sim_probs, sim_q, sim_d):
        if df is None or df.empty:
            continue
        for c in ("season", "week"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)

    pred_w = pred[(pred.get("season") == season) & (pred.get("week") == week)].copy() if {"season", "week"}.issubset(pred.columns) else pred.copy()
    lines_w = lines[(lines.get("season") == season) & (lines.get("week") == week)].copy() if (not lines.empty and {"season", "week"}.issubset(lines.columns)) else pd.DataFrame()
    sim_probs_w = sim_probs[(sim_probs.get("season") == season) & (sim_probs.get("week") == week)].copy() if {"season", "week"}.issubset(sim_probs.columns) else sim_probs.copy()

    # Build lookup frames
    pred_tot = None
    if not pred_w.empty:
        if "pred_total" in pred_w.columns:
            pred_tot = pd.to_numeric(pred_w["pred_total"], errors="coerce")
        elif {"pred_home_points", "pred_away_points"}.issubset(pred_w.columns):
            pred_tot = pd.to_numeric(pred_w["pred_home_points"], errors="coerce") + pd.to_numeric(pred_w["pred_away_points"], errors="coerce")
        else:
            pred_tot = pd.Series([np.nan] * len(pred_w))
        pred_w = pred_w.assign(_pred_total=pred_tot)

    if not lines_w.empty:
        lines_w = lines_w.assign(_market_total=_market_total_from_lines(lines_w))

    # Join by game_id (most robust)
    base = sim_probs_w.copy()
    if base.empty:
        issues.append(Issue("ERROR", None, "No sim_probs rows for requested season/week"))
        for i in issues:
            print(f"[{i.level}] {i.game_id or ''} {i.msg}".strip())
        return 1

    if (not pred_w.empty) and ("game_id" in pred_w.columns) and ("game_id" in base.columns):
        base = base.merge(pred_w[["game_id", "_pred_total"]].drop_duplicates("game_id"), on="game_id", how="left")
    else:
        base["_pred_total"] = np.nan

    if (not lines_w.empty) and ("game_id" in lines_w.columns) and ("game_id" in base.columns):
        base = base.merge(lines_w[["game_id", "_market_total"]].drop_duplicates("game_id"), on="game_id", how="left")
    else:
        base["_market_total"] = np.nan

    # Basic plausibility checks
    for _, r in base.iterrows():
        gid = str(r.get("game_id"))
        mt = r.get("_market_total")
        pt = r.get("_pred_total")
        st = r.get("pred_total")  # sim mean used
        sm = r.get("total_points_mean")

        # probs
        for pc in ("prob_home_win_mc", "prob_home_cover_mc", "prob_over_total_mc"):
            pv = r.get(pc)
            if pv is None or (isinstance(pv, float) and np.isnan(pv)):
                continue
            try:
                pvf = float(pv)
                if not (0.0 <= pvf <= 1.0):
                    issues.append(Issue("ERROR", gid, f"{pc} out of [0,1]: {pv}"))
            except Exception:
                issues.append(Issue("ERROR", gid, f"{pc} not numeric: {pv}"))

        # total alignment (sim_probs)
        try:
            if pd.notna(st) and pd.notna(sm) and abs(float(st) - float(sm)) > float(args.max_artifact_gap):
                issues.append(Issue("WARN", gid, f"sim pred_total vs total_points_mean gap: {float(st):.2f} vs {float(sm):.2f}"))
        except Exception:
            pass

        # compare to market
        try:
            if pd.notna(pt) and pd.notna(mt) and abs(float(pt) - float(mt)) > float(args.max_total_gap):
                issues.append(Issue("WARN", gid, f"model pred_total far from market: pred={float(pt):.1f} market={float(mt):.1f}"))
        except Exception:
            pass

    # Quarters consistency
    if not sim_q.empty and "game_id" in sim_q.columns:
        for _, r in sim_q.iterrows():
            gid = str(r.get("game_id"))
            try:
                hsum = sum(float(r.get(c, 0.0)) for c in ("home_q1", "home_q2", "home_q3", "home_q4"))
                asum = sum(float(r.get(c, 0.0)) for c in ("away_q1", "away_q2", "away_q3", "away_q4"))
                hp = float(r.get("home_points_mean"))
                ap = float(r.get("away_points_mean"))
                if abs(hsum - hp) > float(args.max_artifact_gap):
                    issues.append(Issue("WARN", gid, f"quarters home sum {hsum:.2f} != home_points_mean {hp:.2f}"))
                if abs(asum - ap) > float(args.max_artifact_gap):
                    issues.append(Issue("WARN", gid, f"quarters away sum {asum:.2f} != away_points_mean {ap:.2f}"))
            except Exception:
                issues.append(Issue("WARN", gid, "quarters rows not numeric / incomplete"))
    else:
        issues.append(Issue("WARN", None, f"Missing/empty {sim_quarters_fp} (quarters consistency skipped)"))

    # Drives consistency:
    # - sim_drives.csv contains cumulative mean scores per drive in home_score_mean/away_score_mean
    # - drive_pts_mean is per-drive expected points (not cumulative)
    # We validate the final cumulative score against sim_probs home/away_points_mean.
    if not sim_d.empty and "game_id" in sim_d.columns:
        try:
            sim_probs_idx = None
            try:
                if (sim_probs_w is not None) and (not sim_probs_w.empty) and ("game_id" in sim_probs_w.columns):
                    sim_probs_idx = sim_probs_w.set_index("game_id")
            except Exception:
                sim_probs_idx = None

            for gid, gdf in sim_d.groupby("game_id"):
                gid_s = str(gid)
                gdf2 = gdf.sort_values([c for c in ("drive_no",) if c in gdf.columns])
                last = gdf2.iloc[-1]

                try:
                    hs = float(last.get("home_score_mean"))
                    a_s = float(last.get("away_score_mean"))
                except Exception:
                    issues.append(Issue("WARN", gid_s, "drives missing home_score_mean/away_score_mean"))
                    continue

                if sim_probs_idx is not None and gid_s in sim_probs_idx.index:
                    try:
                        hp = float(sim_probs_idx.loc[gid_s].get("home_points_mean"))
                        ap = float(sim_probs_idx.loc[gid_s].get("away_points_mean"))
                        if abs(hs - hp) > float(args.max_artifact_gap):
                            issues.append(Issue("WARN", gid_s, f"drives final home_score_mean {hs:.2f} != sim_probs home_points_mean {hp:.2f}"))
                        if abs(a_s - ap) > float(args.max_artifact_gap):
                            issues.append(Issue("WARN", gid_s, f"drives final away_score_mean {a_s:.2f} != sim_probs away_points_mean {ap:.2f}"))
                    except Exception:
                        issues.append(Issue("WARN", gid_s, "unable to compare drives final score to sim_probs means"))

                # Optional: validate cumulative sums from drive_pts_mean by possession side.
                try:
                    if {"drive_pts_mean", "poss_side"}.issubset(gdf2.columns):
                        pts = pd.to_numeric(gdf2["drive_pts_mean"], errors="coerce")
                        side = gdf2["poss_side"].astype(str).str.lower()
                        home_sum = float(pts.where(side == "home", 0.0).sum())
                        away_sum = float(pts.where(side == "away", 0.0).sum())
                        # This is expected to be close to final cumulative means.
                        if abs(home_sum - hs) > float(args.max_artifact_gap) * 2:
                            issues.append(Issue("WARN", gid_s, f"sum(home drive_pts_mean) {home_sum:.2f} != drives final home_score_mean {hs:.2f}"))
                        if abs(away_sum - a_s) > float(args.max_artifact_gap) * 2:
                            issues.append(Issue("WARN", gid_s, f"sum(away drive_pts_mean) {away_sum:.2f} != drives final away_score_mean {a_s:.2f}"))
                except Exception:
                    pass
        except Exception:
            issues.append(Issue("WARN", None, "drives grouping failed"))
    else:
        issues.append(Issue("WARN", None, f"Missing/empty {sim_drives_fp} (drives consistency skipped)"))

    # Report
    if not issues:
        print(f"OK: season={season} week={week} ({len(base)} games) :: {bt_dir}")
        return 0

    errs = [i for i in issues if i.level == "ERROR"]
    warns = [i for i in issues if i.level == "WARN"]
    print(f"Issues: {len(errs)} errors, {len(warns)} warnings :: season={season} week={week} :: {bt_dir}")
    for i in issues:
        prefix = f"[{i.level}]"
        gid = f"{i.game_id} " if i.game_id else ""
        print(f"{prefix} {gid}{i.msg}")

    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
