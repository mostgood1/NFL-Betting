"""Compare backtest outputs between line modes (open vs close).

Reads existing backtest outputs under nfl_compare/data/backtests/ and computes:
- realized ROI per bet (1u stake) per market
- total realized units
- overlap rate of ATS/TOTAL selections by game

Usage:
  python scripts/compare_backtest_line_modes.py --season 2025 --end-week 17 --tag-open open_wk12_17 --tag-close close_wk12_17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _american_to_decimal(odds) -> float | None:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / abs(o)


def _realized_units(row: pd.Series) -> float:
    res = str(row.get("result", "")).upper()
    if res == "PUSH":
        return 0.0
    if res not in {"WIN", "LOSS"}:
        return float("nan")

    typ = str(row.get("type", "")).upper()
    odds = row.get("odds")
    dec = _american_to_decimal(odds)
    if dec is None:
        # Default pricing for ATS/TOTAL
        if typ in {"SPREAD", "TOTAL"}:
            dec = 1.0 + 100.0 / 110.0
        else:
            return float("nan")

    if res == "WIN":
        return float(dec) - 1.0
    return -1.0


def _load_details(backtests_dir: Path, season: int, end_week: int, tag: str) -> pd.DataFrame:
    out_dir = backtests_dir / f"{season}_wk{end_week}_{tag}"
    fp = out_dir / "recs_backtest_details.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    df["typeU"] = df["type"].astype(str).str.upper()
    df["resultU"] = df.get("result", "").astype(str).str.upper()
    df["realized_units"] = df.apply(_realized_units, axis=1)
    return df


def _market_stats(df: pd.DataFrame) -> dict:
    out: dict = {}
    for m in ["MONEYLINE", "SPREAD", "TOTAL"]:
        sub = df[df["typeU"].eq(m)].copy()
        graded = sub[sub["resultU"].isin(["WIN", "LOSS", "PUSH"])].copy()
        roi = float(np.nanmean(graded["realized_units"])) if len(graded) else float("nan")
        out[m] = {
            "rows": int(len(sub)),
            "graded": int(len(graded)),
            "wins": int((graded["resultU"] == "WIN").sum()),
            "losses": int((graded["resultU"] == "LOSS").sum()),
            "pushes": int((graded["resultU"] == "PUSH").sum()),
            "realized_roi_per_bet": roi,
            "realized_units_total": float(np.nansum(graded["realized_units"])) if len(graded) else float("nan"),
        }
    return out


def _pick_one_per_game_market(df: pd.DataFrame) -> pd.DataFrame:
    # With RECS_ONE_PER_MARKET=true this should already be 1 per (game_id,type)
    # but we dedupe defensively.
    d = df.copy()
    if "game_id" not in d.columns:
        return d
    d = d.sort_values(["game_id", "typeU"], kind="mergesort")
    return d.drop_duplicates(["game_id", "typeU"], keep="first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--tag-open", type=str, required=True)
    ap.add_argument("--tag-close", type=str, required=True)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    backtests_dir = base_dir / "nfl_compare" / "data" / "backtests"

    open_df = _load_details(backtests_dir, args.season, args.end_week, args.tag_open)
    close_df = _load_details(backtests_dir, args.season, args.end_week, args.tag_close)

    open_stats = _market_stats(open_df)
    close_stats = _market_stats(close_df)

    print("=== Realized (per-bet ROI, 1u stake) ===")
    for m in ["MONEYLINE", "SPREAD", "TOTAL"]:
        o = open_stats[m]
        c = close_stats[m]
        print(
            f"{m:9s} "
            f"open: rows={o['rows']:3d} graded={o['graded']:3d} roi={o['realized_roi_per_bet']:.4f} units={o['realized_units_total']:.2f} | "
            f"close: rows={c['rows']:3d} graded={c['graded']:3d} roi={c['realized_roi_per_bet']:.4f} units={c['realized_units_total']:.2f}"
        )

    # Overlap of selections by game/type for ATS + TOTAL
    od = _pick_one_per_game_market(open_df[open_df["typeU"].isin(["SPREAD", "TOTAL"])])
    cd = _pick_one_per_game_market(close_df[close_df["typeU"].isin(["SPREAD", "TOTAL"])])
    if "selection" in od.columns and "selection" in cd.columns:
        od["selU"] = od["selection"].astype(str)
        cd["selU"] = cd["selection"].astype(str)

        merged = od.merge(cd[["game_id", "typeU", "selU"]], on=["game_id", "typeU"], how="inner", suffixes=("_open", "_close"))
        merged["same"] = merged["selU_open"].eq(merged["selU_close"])
        print("\n=== Selection overlap (ATS/TOTAL) ===")
        for t in ["SPREAD", "TOTAL"]:
            sub = merged[merged["typeU"].eq(t)]
            if len(sub) == 0:
                continue
            print(f"{t:6s} overlap rows={len(sub):3d} same={int(sub['same'].sum()):3d} share_same={float(sub['same'].mean()):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
