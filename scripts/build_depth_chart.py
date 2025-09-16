import argparse
from pathlib import Path
import sys

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.depth_charts import save_depth_chart


def main() -> int:
    p = argparse.ArgumentParser(description="Build and save weekly depth chart (ESPN)")
    p.add_argument("season", type=int)
    p.add_argument("week", type=int)
    p.add_argument("--source", choices=["espn"], default="espn")
    args = p.parse_args()
    out = save_depth_chart(args.season, args.week, source=args.source)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
