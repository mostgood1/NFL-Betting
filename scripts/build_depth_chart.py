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
    p.add_argument(
        "--team",
        action="append",
        default=None,
        help="Optional team name to refresh (repeatable). If omitted, refreshes all teams.",
    )
    p.add_argument(
        "--teams",
        default=None,
        help="Optional comma-separated team names to refresh (alternative to --team).",
    )
    args = p.parse_args()

    teams = []
    if args.team:
        teams.extend([t.strip() for t in args.team if str(t).strip()])
    if args.teams:
        teams.extend([t.strip() for t in str(args.teams).split(",") if t.strip()])
    teams = teams or None

    out = save_depth_chart(args.season, args.week, source=args.source, teams=teams)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
