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


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "nfl_compare" / "data"


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def main() -> int:
    cw_path = DATA_DIR / "current_week.json"
    if not cw_path.exists():
        print(f"ERROR: {cw_path} not found.")
        return 2
    cw = json.loads(cw_path.read_text())
    season = int(cw.get("season"))
    week = int(cw.get("week"))
    # Player props artifacts
    bov_csv = DATA_DIR / f"bovada_player_props_{season}_wk{week}.csv"
    edges_csv = DATA_DIR / f"edges_player_props_{season}_wk{week}.csv"
    ladders_csv = DATA_DIR / f"ladder_options_{season}_wk{week}.csv"
    # Game props artifacts
    game_bov_csv = DATA_DIR / f"bovada_game_props_{season}_wk{week}.csv"
    game_edges_csv = DATA_DIR / f"edges_game_props_{season}_wk{week}.csv"

    # 1) Fetch Bovada
    rc = run([
        sys.executable,
        "scripts/fetch_bovada_props.py",
        "--season", str(season),
        "--week", str(week),
        "--out", str(bov_csv),
    ])
    if rc != 0:
        return rc

    # 2) Join edges
    rc = run([
        sys.executable,
        "scripts/props_edges_join.py",
        "--season", str(season),
        "--week", str(week),
        "--bovada", str(bov_csv),
        "--out", str(edges_csv),
    ])
    if rc != 0:
        return rc

    # 3) Generate ladders for all games (synthesize if absent)
    rc = run([
        sys.executable,
        "scripts/gen_ladder_options.py",
        "--season", str(season),
        "--week", str(week),
        "--bovada", str(bov_csv),
        "--out", str(ladders_csv),
        "--synthesize",
        "--max-rungs", "6",
        "--yard-step", "10",
        "--rec-step", "1",
    ])
    if rc != 0:
        return rc

    # 4) Fetch Bovada GAME props
    rc = run([
        sys.executable,
        "scripts/fetch_bovada_game_props.py",
        "--season", str(season),
        "--week", str(week),
        "--out", str(game_bov_csv),
    ])
    if rc != 0:
        return rc

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
        return rc

    print("Pipeline complete:")
    print(" -", bov_csv)
    print(" -", edges_csv)
    print(" -", ladders_csv)
    print(" -", game_bov_csv)
    print(" -", game_edges_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
