import argparse
from pathlib import Path
import sys
import warnings
import pandas as pd

# Ensure repo root is on sys.path so `nfl_compare` package is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.player_props import compute_player_props
from nfl_compare.src.util import silence_stdout_stderr
from nfl_compare.src.depth_charts import save_depth_chart as save_weekly_depth


def main():
    # Silence noisy FutureWarnings (e.g., pandas downcasting notices)
    warnings.filterwarnings("ignore", category=FutureWarning)
    p = argparse.ArgumentParser(description="Generate player props CSV for a season/week")
    p.add_argument("season", type=int)
    p.add_argument("week", type=int)
    args = p.parse_args()

    # Respect props lock: if a lock marker or locked CSV exists, skip regeneration
    data_dir = REPO_ROOT / 'nfl_compare' / 'data'
    lock_marker = data_dir / f"props_lock_{args.season}_wk{args.week}.lock"
    locked_csv = data_dir / f"player_props_{args.season}_wk{args.week}.locked.csv"
    if lock_marker.exists() or locked_csv.exists():
        print(f"Props locked for season={args.season}, week={args.week}; skipping regeneration.")
        return

    # Build ESPN depth chart for this week if missing
    dc_fp = REPO_ROOT / 'nfl_compare' / 'data' / f"depth_chart_{args.season}_wk{args.week}.csv"
    if not dc_fp.exists():
        with silence_stdout_stderr():
            save_weekly_depth(int(args.season), int(args.week))

    with silence_stdout_stderr():
        df = compute_player_props(args.season, args.week)
    out = Path(__file__).resolve().parents[1] / "nfl_compare" / "data" / f"player_props_{args.season}_wk{args.week}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")


if __name__ == "__main__":
    main()
