"""
One-click pipeline to:
  1) Read current season/week from nfl_compare/data/current_week.json
  2) Fetch Bovada props to CSV
  3) Join edges vs model
  4) Generate ladder options (synthesized if explicit ladders are absent)

Outputs go under nfl_compare/data for the inferred week.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Optional, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "nfl_compare" / "data"


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def _csv_has_data(fp: Path) -> bool:
    """True if CSV exists and appears to have at least one data row (header + >=1 row).
    Works for UTF-8 and UTF-16 by counting newline bytes.
    """
    try:
        if not fp.exists() or fp.stat().st_size == 0:
            return False
        # Count newlines in binary mode to be agnostic to encoding (\n or \n\x00)
        with open(fp, 'rb') as f:
            b = f.read(1024 * 1024)  # read up to 1MB; enough for header + small data
        # Count both LF and UTF-16 LE/BE patterns conservatively
        nl = b.count(b"\n") + b.count(b"\n\x00") + b.count(b"\x00\n")
        return nl >= 2  # header + at least one data line
    except Exception:
        return False


def _normalize_csv_encoding(src: Path, dst: Path) -> bool:
    """Attempt to read a possibly non-UTF8 CSV (e.g., UTF-16) and rewrite as UTF-8.
    Returns True on success; False otherwise.
    """
    encodings: List[str] = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(src, encoding=enc)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst, index=False)
            return True
        except Exception:
            continue
    return False


def main() -> int:
    cw_path = DATA_DIR / "current_week.json"
    if not cw_path.exists():
        print(f"ERROR: {cw_path} not found.")
        return 2
    cw = json.loads(cw_path.read_text())
    season = int(cw.get("season"))
    week = int(cw.get("week"))
    # Player props artifacts
    predictions_csv = DATA_DIR / f"player_props_{season}_wk{week}.csv"  # model predictions we generate
    oddsapi_csv = DATA_DIR / f"oddsapi_player_props_{season}_wk{week}.csv"  # market lines fetched from OddsAPI
    edges_csv = DATA_DIR / f"edges_player_props_{season}_wk{week}.csv"
    ladders_csv = DATA_DIR / f"ladder_options_{season}_wk{week}.csv"
    # Game props artifacts
    game_bov_csv = DATA_DIR / f"bovada_game_props_{season}_wk{week}.csv"
    game_edges_csv = DATA_DIR / f"edges_game_props_{season}_wk{week}.csv"

    # 0) Refresh depth chart and player props for this week
    # Rebuild weekly depth chart to capture latest actives/injuries
    dc_csv = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    # predictions output path
    props_csv = predictions_csv
    print("Pre-step: refresh depth chart and player props")
    rc = run([
        sys.executable,
        "scripts/build_depth_chart.py",
        str(season),
        str(week),
    ])
    if rc != 0:
        print(f"WARNING: build_depth_chart returned {rc}")
    rc = run([
        sys.executable,
        "scripts/gen_props.py",
        str(season),
        str(week),
    ])
    if rc != 0:
        print(f"WARNING: gen_props returned {rc} (may be locked or failed)")
    # Require props to exist for downstream edges join
    if not _csv_has_data(predictions_csv):
        print(f"ERROR: Missing or empty props CSV: {predictions_csv}")
        return 3

    # 1) Fetch player props from OddsAPI
    rc = run([
        sys.executable,
        "scripts/fetch_oddsapi_props.py",
        "--season", str(season),
        "--week", str(week),
        "--out", str(oddsapi_csv),
    ])
    if rc != 0:
        print(f"WARNING: fetch_oddsapi_props exited with {rc} for players.")
    # Determine source CSV for edges/ladders: prefer OddsAPI if it has data; else fall back to Bovada scrape
    props_source_csv = oddsapi_csv
    if not _csv_has_data(props_source_csv):
        print("INFO: OddsAPI player props missing/empty; attempting Bovada player props as fallback…")
        bov_p_csv = DATA_DIR / f"bovada_player_props_{season}_wk{week}.csv"
        rc_b = run([
            sys.executable,
            "scripts/fetch_bovada_props.py",
            "--season", str(season),
            "--week", str(week),
            "--out", str(bov_p_csv),
        ])
        if rc_b == 0 and _csv_has_data(bov_p_csv):
            print(f"INFO: Using Bovada player props fallback: {bov_p_csv.name}")
            props_source_csv = bov_p_csv
        else:
            print("WARNING: Bovada player props fallback failed or empty. Edges and ladders may be empty.")

    # 2) Join edges
    rc = run([
        sys.executable,
        "scripts/props_edges_join.py",
        "--season", str(season),
        "--week", str(week),
        # The join script expects --bovada=<csv>; it accepts any props CSV with the expected columns
        "--bovada", str(props_source_csv),
        "--out", str(edges_csv),
    ])
    if rc != 0:
        print("WARNING: props_edges_join failed. Continuing pipeline; ladders may still generate if Bovada CSV is readable.")

    # 3) Generate ladders for all games (synthesize if absent)
    rc = run([
        sys.executable,
        "scripts/gen_ladder_options.py",
        "--season", str(season),
        "--week", str(week),
        "--bovada", str(props_source_csv),
        "--out", str(ladders_csv),
        "--synthesize",
        "--max-rungs", "6",
        "--yard-step", "10",
        "--rec-step", "1",
    ])
    if rc != 0:
        print("WARNING: gen_ladder_options failed. Continuing to game props.")

    # 4) Fetch Bovada GAME props
    rc = run([
        sys.executable,
        "scripts/fetch_bovada_game_props.py",
        "--season", str(season),
        "--week", str(week),
        "--out", str(game_bov_csv),
    ])
    if rc != 0:
        print(f"WARNING: fetch_bovada_game_props exited with {rc}. Will attempt to proceed if a previous CSV exists.")

    # Optional: normalize game props if existing file is non-UTF8 or header-only and an archive exists
    if not _csv_has_data(game_bov_csv):
        arch_g = DATA_DIR / f"bovada_game_props_{season}_wk{week}.archive.csv"
        if arch_g.exists():
            tmp_g = DATA_DIR / f"bovada_game_props_{season}_wk{week}.fallback.csv"
            ok_g = _normalize_csv_encoding(arch_g, tmp_g)
            if ok_g and _csv_has_data(tmp_g):
                print(f"INFO: Using archived game props fallback: {arch_g.name} -> {tmp_g.name}")
                try:
                    tmp_g.replace(game_bov_csv)
                except Exception:
                    try:
                        df = pd.read_csv(tmp_g)
                        df.to_csv(game_bov_csv, index=False)
                    except Exception:
                        pass
        else:
            print(f"WARNING: No archived game props found at {arch_g}")

    # 5) Compute GAME props edges
    rc = run([
        sys.executable,
        "scripts/game_props_edges_join.py",
        "--season", str(season),
        "--week", str(week),
        "--game-csv", str(game_bov_csv),
        "--out", str(game_edges_csv),
    ])
    if rc != 0:
        print("WARNING: game_props_edges_join failed.")
        return rc

    print("Pipeline complete:")
    print(" -", predictions_csv)
    print(" -", edges_csv)
    print(" -", ladders_csv)
    print(" -", props_source_csv)
    print(" -", game_bov_csv)
    print(" -", game_edges_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
