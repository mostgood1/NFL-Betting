"""Backfill lines.csv from archived real_betting_lines_*.json snapshots.

Goal
----
For historical weeks where `lines.csv` is missing usable market lines, backfill
close-style fields (`close_spread_home`, `close_total`, moneylines/prices) from the
nearest snapshot on/before each game's kickoff date.

This uses only real captured snapshots (no synthetic defaults).

Usage
-----
  python scripts/backfill_lines_from_real_snapshots.py --season 2025 --weeks 1-18

Notes
-----
- Matchups are joined by normalized (home_team, away_team) names.
- Snapshot selection: choose the latest snapshot date <= game_date; if none exist,
  choose the earliest snapshot date > game_date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import pandas as pd


def _ensure_repo_on_path() -> None:
	here = Path(__file__).resolve()
	repo_root = here.parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))


def _parse_weeks(s: str) -> list[int]:
	s = str(s).strip()
	if not s:
		return []
	if "-" in s:
		a, b = s.split("-", 1)
		return list(range(int(a), int(b) + 1))
	return [int(x) for x in s.split(",") if str(x).strip()]


def _snap_date_from_path(p: Path) -> dt.date | None:
	# real_betting_lines_2025_09_04.json
	stem = p.stem
	if not stem.startswith("real_betting_lines_"):
		return None
	tail = stem.replace("real_betting_lines_", "")
	try:
		y, m, d = tail.split("_")
		return dt.date(int(y), int(m), int(d))
	except Exception:
		return None


def _choose_snapshot(game_date: dt.date, snaps: list[tuple[dt.date, Path]]) -> Path | None:
	if not snaps:
		return None
	# Latest snapshot on/before game_date
	prior = [p for (d, p) in snaps if d <= game_date]
	if prior:
		return prior[-1]
	# Else earliest after
	after = [p for (d, p) in snaps if d > game_date]
	if after:
		return after[0]
	return None


def _candidate_snapshots(
	game_date: dt.date,
	snaps: list[tuple[dt.date, Path]],
	lookback_days: int = 7,
	lookahead_days: int = 2,
) -> list[Path]:
	"""Return candidate snapshot paths around game_date.

	Order is important:
	1) Prior snapshots within lookback window (latest -> earliest)
	2) After snapshots within lookahead window (earliest -> latest)
	3) Fallback to the default choice if nothing in window
	"""
	if not snaps:
		return []
	prior: list[tuple[dt.date, Path]] = [t for t in snaps if 0 <= (game_date - t[0]).days <= int(lookback_days)]
	prior.sort(key=lambda t: t[0], reverse=True)
	after: list[tuple[dt.date, Path]] = [t for t in snaps if 0 < (t[0] - game_date).days <= int(lookahead_days)]
	after.sort(key=lambda t: t[0])
	out = [p for _, p in prior] + [p for _, p in after]
	if out:
		return out
	fallback = _choose_snapshot(game_date, snaps)
	return [fallback] if fallback is not None else []


def main() -> None:
	_ensure_repo_on_path()

	ap = argparse.ArgumentParser(description="Backfill lines.csv from real_betting_lines_*.json snapshots")
	ap.add_argument("--season", type=int, required=True)
	ap.add_argument("--weeks", type=str, default="1-18", help="Week list like '17-18' or '1,2,3'")
	args = ap.parse_args()

	from nfl_compare.src.data_sources import DATA_DIR, load_games  # type: ignore
	from nfl_compare.src.team_normalizer import normalize_team_name  # type: ignore
	from nfl_compare.src.data_sources import _parse_real_lines_json  # type: ignore

	weeks = _parse_weeks(args.weeks)
	if not weeks:
		raise SystemExit("No weeks provided")

	games = load_games()
	if games is None or games.empty:
		raise SystemExit("No games.csv rows found")

	for c in ("season", "week"):
		if c in games.columns:
			games[c] = pd.to_numeric(games[c], errors="coerce")

	g = games[(games["season"] == int(args.season)) & (games["week"].isin([int(w) for w in weeks]))].copy()
	if g.empty:
		raise SystemExit(f"No games found for season={args.season} weeks={weeks}")

	if "game_date" not in g.columns:
		raise SystemExit("games.csv missing game_date column")

	g["home_team_norm"] = g["home_team"].astype(str).apply(normalize_team_name)
	g["away_team_norm"] = g["away_team"].astype(str).apply(normalize_team_name)
	g["game_date_d"] = pd.to_datetime(g["game_date"], errors="coerce").dt.date

	lines_fp = DATA_DIR / "lines.csv"
	if lines_fp.exists():
		lines_df = pd.read_csv(lines_fp)
	else:
		lines_df = pd.DataFrame(columns=[
			"season",
			"week",
			"game_id",
			"game_date",
			"home_team",
			"away_team",
			"spread_home",
			"total",
			"moneyline_home",
			"moneyline_away",
			"spread_home_price",
			"spread_away_price",
			"total_over_price",
			"total_under_price",
			"close_spread_home",
			"close_total",
			"date",
		])

	for c in ("season", "week"):
		if c in lines_df.columns:
			lines_df[c] = pd.to_numeric(lines_df[c], errors="coerce")

	if "home_team" in lines_df.columns:
		lines_df["home_team_norm"] = lines_df["home_team"].astype(str).apply(normalize_team_name)
	else:
		lines_df["home_team_norm"] = pd.NA
	if "away_team" in lines_df.columns:
		lines_df["away_team_norm"] = lines_df["away_team"].astype(str).apply(normalize_team_name)
	else:
		lines_df["away_team_norm"] = pd.NA

	# Enumerate snapshots
	snap_paths = sorted(DATA_DIR.glob("real_betting_lines_*.json"))
	snaps: list[tuple[dt.date, Path]] = []
	for p in snap_paths:
		d = _snap_date_from_path(p)
		if d is not None:
			snaps.append((d, p))
	snaps.sort(key=lambda t: t[0])

	if not snaps:
		raise SystemExit(f"No snapshots found under {DATA_DIR}")

	# Cache parsed snapshots by path
	snap_cache: dict[Path, pd.DataFrame] = {}

	n_upd = 0
	n_add = 0
	n_missing_snapshot = 0
	n_missing_matchup = 0
	missing_matchups: list[dict] = []

	for c in [
		"close_spread_home",
		"close_total",
		"moneyline_home",
		"moneyline_away",
		"spread_home_price",
		"spread_away_price",
		"total_over_price",
		"total_under_price",
	]:
		if c not in lines_df.columns:
			lines_df[c] = pd.NA

	for _, row in g.iterrows():
		gd = row.get("game_date_d")
		if gd is None or pd.isna(gd):
			continue

		ht = str(row.get("home_team_norm", ""))
		at = str(row.get("away_team_norm", ""))

		chosen_snap: Path | None = None
		chosen_df: pd.DataFrame | None = None
		chosen_match: pd.DataFrame | None = None

		for snap in _candidate_snapshots(gd, snaps, lookback_days=7, lookahead_days=2):
			if snap not in snap_cache:
				try:
					snap_cache[snap] = _parse_real_lines_json(json.loads(snap.read_text(encoding="utf-8")))
				except Exception:
					snap_cache[snap] = pd.DataFrame()

			sdf = snap_cache[snap]
			if sdf is None or sdf.empty:
				continue

			if "home_team_norm" not in sdf.columns:
				sdf = sdf.copy()
				sdf["home_team_norm"] = sdf["home_team"].astype(str).apply(normalize_team_name)
				sdf["away_team_norm"] = sdf["away_team"].astype(str).apply(normalize_team_name)
				snap_cache[snap] = sdf

			match = sdf[(sdf["home_team_norm"] == ht) & (sdf["away_team_norm"] == at)]
			if not match.empty:
				chosen_snap = snap
				chosen_df = sdf
				chosen_match = match
				break

		if chosen_snap is None or chosen_df is None or chosen_match is None or chosen_match.empty:
			n_missing_matchup += 1
			missing_matchups.append(
				{
					"season": int(row.get("season")),
					"week": int(row.get("week")),
					"game_date": str(row.get("game_date")),
					"home_team": str(row.get("home_team")),
					"away_team": str(row.get("away_team")),
					"home_team_norm": ht,
					"away_team_norm": at,
					"snapshot_path": "",
					"snapshot_date": "",
					"snapshot_rows": 0,
				}
			)
			continue

		snap = chosen_snap
		sdf = chosen_df
		match = chosen_match

		m0 = match.iloc[0]
		rec = {
			"season": int(row.get("season")),
			"week": int(row.get("week")),
			"game_id": row.get("game_id"),
			"game_date": row.get("game_date"),
			"home_team": row.get("home_team"),
			"away_team": row.get("away_team"),
			"home_team_norm": ht,
			"away_team_norm": at,
			"close_spread_home": m0.get("spread_home"),
			"close_total": m0.get("total"),
			"moneyline_home": m0.get("moneyline_home"),
			"moneyline_away": m0.get("moneyline_away"),
			"spread_home_price": m0.get("spread_home_price"),
			"spread_away_price": m0.get("spread_away_price"),
			"total_over_price": m0.get("total_over_price"),
			"total_under_price": m0.get("total_under_price"),
		}

		mask = (
			(lines_df["season"] == rec["season"])
			& (lines_df["week"] == rec["week"])
			& (lines_df["home_team_norm"] == rec["home_team_norm"])
			& (lines_df["away_team_norm"] == rec["away_team_norm"])
		)

		if mask.any():
			i = lines_df.index[mask][0]
			for c in [
				"close_spread_home",
				"close_total",
				"moneyline_home",
				"moneyline_away",
				"spread_home_price",
				"spread_away_price",
				"total_over_price",
				"total_under_price",
			]:
				if pd.isna(lines_df.at[i, c]) and pd.notna(rec.get(c)):
					lines_df.at[i, c] = rec.get(c)
			n_upd += 1
		else:
			lines_df = pd.concat([lines_df, pd.DataFrame([rec])], ignore_index=True)
			n_add += 1

	out = lines_df.drop(columns=["home_team_norm", "away_team_norm"], errors="ignore")
	lines_fp.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(lines_fp, index=False)

	if missing_matchups:
		reports_dir = Path("reports")
		reports_dir.mkdir(parents=True, exist_ok=True)
		wk_label = args.weeks.replace(" ", "")
		report_fp = reports_dir / f"lines_snapshot_misses_{int(args.season)}_weeks_{wk_label}.csv"
		pd.DataFrame(missing_matchups).to_csv(report_fp, index=False)

	print(
		f"Backfilled lines.csv from snapshots: updated={n_upd}, added={n_add}, "
		f"missing_snapshot={n_missing_snapshot}, missing_matchup={n_missing_matchup} -> {lines_fp}"
	)
	if missing_matchups:
		print(f"Wrote missing-matchup report: {report_fp}")


if __name__ == "__main__":
	main()