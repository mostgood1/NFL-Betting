import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure repo root is on sys.path so `nfl_compare` package is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / 'nfl_compare' / 'data'


def main() -> None:
    p = argparse.ArgumentParser(description='Lock player props for a season/week (prevent regeneration).')
    p.add_argument('season', type=int)
    p.add_argument('week', type=int)
    p.add_argument('--copy', action='store_true', help='Also save a frozen copy as player_props_<season>_wk<week>.locked.csv')
    args = p.parse_args()

    csv_fp = DATA_DIR / f"player_props_{args.season}_wk{args.week}.csv"
    if not csv_fp.exists():
        print(f"Props CSV not found: {csv_fp}. Generate first, then lock.")
        return

    # Create lock marker file
    lock_fp = DATA_DIR / f"props_lock_{args.season}_wk{args.week}.lock"
    lock_fp.write_text('locked')
    print(f"Created {lock_fp} (props locked).")

    if args.copy:
        frozen_fp = DATA_DIR / f"player_props_{args.season}_wk{args.week}.locked.csv"
        try:
            df = pd.read_csv(csv_fp)
            df.to_csv(frozen_fp, index=False)
            print(f"Wrote frozen copy -> {frozen_fp} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to write frozen copy: {e}")


if __name__ == '__main__':
    main()
