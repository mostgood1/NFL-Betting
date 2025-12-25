from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import sys

# Load backtest_games module directly from file to access backtest()
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parents[1]
BG_PATH = ROOT / "scripts" / "backtest_games.py"
DATA_DIR = ROOT / "nfl_compare" / "data" / "backtests"

# Ensure repo root is importable for nfl_compare
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
bg_mod = SourceFileLoader("backtest_games", str(BG_PATH)).load_module()

DEFAULT_PAIRS: List[Tuple[float, float]] = [
    (0.00, 0.00),
    (0.05, 0.00),
    (0.10, 0.05),
    (0.15, 0.05),
    (0.20, 0.10),
    (0.25, 0.15),
    (0.30, 0.15),
    (0.35, 0.15),
]


def run_sweep(season: int, start_week: int, end_week: int, pairs: List[Tuple[float, float]], out_dir: Path) -> Path:
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for bm, bt in pairs:
        print(f"[blend-sweep] season={season} weeks={start_week}-{end_week} blend_margin={bm} blend_total={bt}")
        summ_df, det_df = bg_mod.backtest(
            season=season,
            start_week=start_week,
            end_week=end_week,
            include_same=True,
            blend_margin=bm,
            blend_total=bt,
        )
        if summ_df is None or summ_df.empty:
            continue
        row = summ_df.iloc[0].to_dict()
        row["blend_margin"] = bm
        row["blend_total"] = bt
        rows.append(row)
        # Write per-blend outputs for inspection
        try:
            det_fp = out_dir / f"games_details_bm{bm}_bt{bt}.csv"
            if det_df is not None and not det_df.empty:
                det_df.to_csv(det_fp, index=False)
        except Exception:
            pass
    res = pd.DataFrame(rows)
    sweep_fp = out_dir / "blend_sweep.csv"
    res.to_csv(sweep_fp, index=False)
    print(f"Wrote {sweep_fp}")
    try:
        print(res.to_string(index=False))
    except Exception:
        pass
    return sweep_fp


def parse_pairs(arg: str | None) -> List[Tuple[float, float]]:
    if not arg:
        return DEFAULT_PAIRS
    items = []
    for p in arg.split(";"):
        p = p.strip()
        if not p:
            continue
        try:
            bm_str, bt_str = p.split(",")
            items.append((float(bm_str), float(bt_str)))
        except Exception:
            continue
    return items or DEFAULT_PAIRS


def parse_grid(grid: str | None, default: List[float]) -> List[float]:
    if not grid:
        return default
    grid = grid.strip()
    if ":" in grid:
        try:
            start, end, step = [float(x) for x in grid.split(":")]
            vals = []
            x = start
            # inclusive end with float steps
            while x <= end + 1e-9:
                vals.append(round(x, 10))
                x += step
            return vals
        except Exception:
            pass
    # comma-separated list
    try:
        return [float(x) for x in grid.split(",") if x.strip()]
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description="Run blend sweep backtests for margin/total blends")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, required=True)
    ap.add_argument("--pairs", type=str, default=None, help="Semicolon-separated pairs bm,bt (e.g., '0.0,0.0;0.2,0.1')")
    ap.add_argument("--bm-grid", type=str, default=None, help="Blend margin grid as 'start:end:step' or comma list")
    ap.add_argument("--bt-grid", type=str, default=None, help="Blend total grid as 'start:end:step' or comma list")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    season = int(args.season)
    start_week = int(args.start_week)
    end_week = int(args.end_week)
    pairs = parse_pairs(args.pairs)
    if args.bm_grid or args.bt_grid:
        bm_vals = parse_grid(args.bm_grid, default=[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35])
        bt_vals = parse_grid(args.bt_grid, default=[0.0,0.05,0.10,0.15])
        pairs = [(bm, bt) for bm in bm_vals for bt in bt_vals]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = DATA_DIR / f"{season}_wk{end_week}" / "blend_sweep"

    run_sweep(season, start_week, end_week, pairs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
