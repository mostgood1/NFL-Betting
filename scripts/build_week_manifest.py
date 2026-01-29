from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfl_compare.src.artifacts import DATA_DIR, build_week_manifest, default_week_manifest_path, write_week_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build reproducibility manifest for a given season/week")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--no-rows-cols", action="store_true", help="Skip reading CSVs for row/col info")
    args = ap.parse_args()

    data_dir = DATA_DIR
    # Allow explicit override via env for local workflows.
    env_dd = os.environ.get("NFL_DATA_DIR")
    if env_dd:
        data_dir = Path(env_dd)

    manifest = build_week_manifest(
        args.season,
        args.week,
        data_dir=data_dir,
        include_rows_cols=not args.no_rows_cols,
    )

    out_path = Path(args.out) if args.out else default_week_manifest_path(args.season, args.week, data_dir=data_dir)
    write_week_manifest(manifest, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
