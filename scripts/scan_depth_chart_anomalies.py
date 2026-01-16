import argparse
import os
from pathlib import Path

import pandas as pd

# Ensure repo root is importable
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_compare.src.depth_charts import save_depth_chart


DATA_DIR = REPO_ROOT / "nfl_compare" / "data"


def _parse_weeks_arg(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def scan_depth_chart(season: int, week: int) -> tuple[set[str], dict[str, dict]]:
    fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    if not fp.exists():
        return {"__missing_file__"}, {}

    df = pd.read_csv(fp)
    if df.empty:
        return {"__empty__"}, {}

    df["position"] = df["position"].astype(str).str.upper().str.strip()

    required_pos = {"QB", "RB", "WR", "TE"}
    flags: set[str] = set()
    details: dict[str, dict] = {}

    # Baseline overlap heuristic: compare baseline starter slots to current week players.
    # This catches the "defense table parsed as offense" bug even when QB/RB/WR/TE exist.
    baseline_week = 1
    try:
        baseline_week = int(pd.to_numeric(os.environ.get("INJURY_BASELINE_WEEK", 1), errors="coerce"))
        if baseline_week < 1:
            baseline_week = 1
    except Exception:
        baseline_week = 1

    baseline_fp = DATA_DIR / f"depth_chart_{season}_wk{baseline_week}.csv"
    baseline = None
    if baseline_fp.exists():
        try:
            baseline = pd.read_csv(baseline_fp)
            baseline["position"] = baseline["position"].astype(str).str.upper().str.strip()
            if "depth_rank" not in baseline.columns:
                baseline["depth_rank"] = baseline.groupby(["team", "position"]).cumcount() + 1
        except Exception:
            baseline = None

    def _baseline_slot_players(team: str) -> set[str]:
        if baseline is None or baseline.empty:
            return set()
        slots = {"QB": 1, "RB": 1, "TE": 1, "WR": 2}
        b = baseline[(baseline["team"] == team) & (baseline["position"].isin(list(slots.keys())))].copy()
        if b.empty:
            return set()
        b = b.sort_values(["position", "depth_rank", "player"])
        b["_slot"] = b.groupby(["position"]).cumcount() + 1
        b["_max_slot"] = b["position"].map(slots).fillna(1).astype(int)
        b = b[b["_slot"] <= b["_max_slot"]].copy()
        return set(b["player"].astype(str).tolist())

    for team in sorted(df["team"].astype(str).unique().tolist()):
        sub = df[df["team"] == team]
        pos_present = set(sub["position"].unique().tolist())
        missing = sorted(required_pos - pos_present)

        reasons: list[str] = []
        if missing:
            reasons.append(f"missing_pos={missing}")
        if len(sub) < 12:
            reasons.append(f"low_rows={len(sub)}")
        wr_n = int((sub[sub["position"] == "WR"].shape[0]))
        if wr_n < 4:
            reasons.append(f"low_wr_rows={wr_n}")

        if baseline is not None and not baseline.empty:
            base_players = _baseline_slot_players(team)
            if base_players:
                cur_players = set(
                    sub[sub["position"].isin(list(required_pos))]["player"].astype(str).tolist()
                )
                overlap = len(base_players & cur_players)
                if overlap <= 1:
                    reasons.append(f"baseline_overlap={overlap}/5")

        if reasons:
            flags.add(team)
            details[team] = {
                "rows": int(len(sub)),
                "positions": sorted(pos_present),
                "reasons": reasons,
            }

    return flags, details


def main() -> int:
    p = argparse.ArgumentParser(description="Scan weekly depth charts for offense-table scrape anomalies")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", type=str, required=True, help="Weeks: e.g. 1-18 or 10-18 or 1,2,5")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="If set, re-scrape only the flagged teams and patch into the weekly CSVs.",
    )
    args = p.parse_args()
    weeks = _parse_weeks_arg(args.weeks)
    if not weeks:
        raise SystemExit("No weeks parsed")

    any_flags = False
    for week in weeks:
        flags, details = scan_depth_chart(args.season, week)
        if not flags:
            continue
        any_flags = True

        if "__missing_file__" in flags or "__empty__" in flags:
            print(f"wk{week}: {flags}")
            continue

        print(f"wk{week}: flagged {len(flags)} team(s)")
        for team in sorted(flags):
            d = details.get(team, {})
            print(f"  - {team}: {d.get('reasons')}")

        if args.refresh:
            refreshed = []
            for team in sorted(flags):
                out = save_depth_chart(args.season, week, source="espn", teams=[team])
                refreshed.append(team)
            print(f"wk{week}: refreshed {len(refreshed)} team(s) -> {out}")

    if not any_flags:
        print("No anomalies detected in requested weeks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
