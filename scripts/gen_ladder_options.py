"""
Generate ladder options per player/market from Bovada CSV and join simple model edges.

Usage:
  python scripts/gen_ladder_options.py --season 2025 --week 3 \
    --bovada nfl_compare/data/bovada_player_props_2025_wk3.csv \
    --out nfl_compare/data/ladder_options_2025_wk3.csv

Notes:
  - Expects Bovada CSV produced by scripts/fetch_bovada_props.py, which includes is_ladder markers.
  - Joins to player props predictions (player_props_{season}_wk{week}.csv) to add naive edge = proj - line.
  - Focuses on yardage/receptions markets: Receiving Yards, Receptions, Rushing Yards, Passing Yards.
  - Keeps one row per ladder rung (does not collapse).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


MARKET_KEEP = {
    "receiving yards": "rec_yards",
    "receptions": "receptions",
    "rushing yards": "rush_yards",
    "passing yards": "pass_yards",
}

PROJ_COLS: Dict[str, List[str]] = {
    "rec_yards": ["rec_yards", "pred_rec_yards", "expected_rec_yards"],
    "receptions": ["receptions", "pred_receptions", "expected_receptions"],
    "rush_yards": ["rush_yards", "pred_rush_yards", "expected_rush_yards"],
    "pass_yards": ["pass_yards", "pred_pass_yards", "expected_pass_yards"],
}


def _nm_loose(s: Optional[str]) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_market_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a lowercase/stripped 'market_norm' column exists.
    If 'market' is missing, fill with empty string to keep downstream filters safe.
    """
    if "market_norm" in df.columns:
        return df
    if "market" in df.columns:
        df["market_norm"] = df["market"].astype(str).str.strip().str.lower()
    else:
        # Broadcast empty string to match length
        df["market_norm"] = pd.Series([""] * len(df))
    return df


def _ensure_key_player_loose(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'key_player_loose' column exists for downstream merges.
    Attempts to derive from a plausible player column; if none, creates an empty column.
    Safe for empty DataFrames.
    """
    if "key_player_loose" in df.columns:
        return df
    pcol = _pick_col(df, ["player", "name", "player_name", "display_name"])  # may be None
    if pcol:
        df[pcol] = df[pcol].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
        df["key_player_loose"] = df[pcol].map(_nm_loose)
    else:
        # Broadcast empty string; safe for empty DataFrames
        df["key_player_loose"] = ""
    return df


def load_bovada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bovada CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize market
    df = _ensure_market_norm(df)
    # Keep only ladders for the markets we care about
    df = df[df.get("is_ladder").fillna(False) == True].copy()
    df = df[df["market_norm"].isin(MARKET_KEEP.keys())].copy()

    # Normalize player and team
    pcol = _pick_col(df, ["player", "name", "player_name", "display_name"]) or "player"
    df[pcol] = df[pcol].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
    df["key_player_loose"] = df[pcol].map(_nm_loose)
    tcol = _pick_col(df, ["team", "team_abbr", "team_code", "posteam"])  # optional
    if tcol:
        df["team"] = df[tcol]

    # Coerce numerics
    df["line"] = pd.to_numeric(df.get("line"), errors="coerce")
    df["over_price"] = pd.to_numeric(df.get("over_price"), errors="coerce")
    df["under_price"] = pd.to_numeric(df.get("under_price"), errors="coerce")
    return df


def load_predictions(season: int, week: int, base_dir: Optional[Path] = None) -> pd.DataFrame:
    base = Path(base_dir) if base_dir else Path("nfl_compare/data")
    path = base / f"player_props_{season}_wk{week}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    preds = pd.read_csv(path)
    pcol = _pick_col(preds, ["display_name", "player", "name", "player_name"]) or "display_name"
    preds["key_player_loose"] = preds[pcol].map(_nm_loose)
    return preds


def choose_proj_col(preds: pd.DataFrame, key: str) -> Optional[str]:
    for c in PROJ_COLS.get(key, []):
        if c in preds.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ladder options from Bovada props")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--bovada", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--base-dir", type=str, default="nfl_compare/data")
    ap.add_argument("--filter-event", type=str, default=None, help="Regex to filter event description (optional)")
    ap.add_argument("--synthesize", action="store_true", help="If no explicit ladder rows exist, synthesize ladder rungs from baseline lines.")
    ap.add_argument("--max-rungs", type=int, default=8, help="Maximum number of ladder rungs to generate per player/market when synthesizing.")
    ap.add_argument("--yard-step", type=float, default=10.0, help="Step size for yardage ladder synthesis.")
    ap.add_argument("--rec-step", type=float, default=1.0, help="Step size for receptions ladder synthesis.")
    args = ap.parse_args()

    bov_all = pd.read_csv(Path(args.bovada))
    # Event filter (regex)
    if args.filter_event:
        mask_evt = bov_all.get("event", "").astype(str).str.contains(args.filter_event, case=False, na=False, regex=True)
        bov_all = bov_all[mask_evt].copy()

    # Explicit ladders first
    bov = bov_all.copy()
    bov = _ensure_market_norm(bov)
    bov = bov[bov.get("is_ladder").fillna(False) == True]
    bov = bov[bov["market_norm"].isin(MARKET_KEEP.keys())].copy()
    # Normalize names
    if not bov.empty:
        pcol_b = _pick_col(bov, ["player", "name", "player_name", "display_name"]) or "player"
        bov[pcol_b] = bov[pcol_b].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
        bov["key_player_loose"] = bov[pcol_b].map(_nm_loose)
        tcol_b = _pick_col(bov, ["team", "team_abbr", "team_code", "posteam"])  # optional
        if tcol_b:
            bov["team"] = bov[tcol_b]

    synthesized = pd.DataFrame()
    if bov.empty and args.synthesize:
        # Build from standard rows with numeric lines
        std = bov_all.copy()
        std = _ensure_market_norm(std)
        std = std[std["market_norm"].isin(MARKET_KEEP.keys())].copy()
        std["line"] = pd.to_numeric(std.get("line"), errors="coerce")
        std = std[std["line"].notna()].copy()
        # Normalize names
        pcol_s = _pick_col(std, ["player", "name", "player_name", "display_name"]) or "player"
        std[pcol_s] = std[pcol_s].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
        std["key_player_loose"] = std[pcol_s].map(_nm_loose)
        # Collapse per (event, player, market) to a representative baseline line near the median
        grp_keys = [k for k in ["event", "key_player_loose", "market_norm"] if k in std.columns]
        def _pick_row(g: pd.DataFrame) -> pd.DataFrame:
            if g["line"].notna().sum() == 0:
                return g.iloc[:1].copy()
            med = g["line"].median()
            g2 = g.copy()
            g2["_dist"] = (g2["line"] - med).abs()
            return g2.sort_values(["_dist", "line"], ascending=[True, True]).head(1)
        # Future pandas may exclude grouping columns by default; request current behavior explicitly
        try:
            base = std.groupby(grp_keys, as_index=False, group_keys=False).apply(_pick_row, include_groups=False)  # type: ignore[arg-type]
        except TypeError:
            # Older pandas versions: no include_groups arg
            base = std.groupby(grp_keys, as_index=False, group_keys=False).apply(_pick_row)
        try:
            base = base.reset_index(drop=True)
        except Exception:
            pass
        if "_dist" in base.columns:
            base = base.drop(columns=["_dist"])
        # Ensure market_norm survived grouping; if missing, derive from 'market' or set empty
        if "market_norm" not in base.columns:
            if "market" in base.columns:
                base["market_norm"] = base["market"].astype(str).str.strip().str.lower()
            else:
                base["market_norm"] = ""
        # Generate rungs around baseline
        rows: List[dict] = []
        for _, r in base.iterrows():
            # Safe access and fallback
            market_norm = r.get("market_norm")
            if market_norm is None or (isinstance(market_norm, float) and np.isnan(market_norm)):
                market_norm = str(r.get("market") or "").strip().lower()
            market = r.get("market")
            player = r.get(pcol_s)
            team = r.get("team", np.nan)
            event = r.get("event")
            game_time = r.get("game_time")
            home_team = r.get("home_team")
            away_team = r.get("away_team")
            key = MARKET_KEEP.get(str(market_norm))
            if key is None:
                continue
            baseline = float(r.get("line")) if pd.notna(r.get("line")) else None
            if baseline is None:
                continue
            # Steps
            step = args.rec_step if key == "receptions" else args.yard_step
            # Build symmetric rungs around baseline: include baseline, then +/- step up to max-rungs
            deltas = [0]
            for i in range(1, args.max_rungs + 1):
                deltas.extend([i * step, -i * step])
            # de-duplicate and clamp to >= 0
            rungs = sorted({round(max(0.0, baseline + d), 1) for d in deltas})
            for ln in rungs:
                rows.append({
                    "player": player,
                    "team": team,
                    "event": event,
                    "market": market,
                    "line": ln,
                    "over_price": np.nan,
                    "under_price": np.nan,
                    "game_time": game_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "is_ladder": True,
                    "market_norm": market_norm,
                    "key_player_loose": r.get("key_player_loose"),
                })
        synthesized = pd.DataFrame(rows)
        bov = synthesized.copy()

    preds = load_predictions(args.season, args.week, Path(args.base_dir))

    # Prepare market key for selecting projection column
    bov = bov.copy()
    if "market_norm" not in bov.columns:
        bov = _ensure_market_norm(bov)
    bov["market_key"] = bov["market_norm"].map(MARKET_KEEP)

    # Merge predictions by loose name (team not enforced to keep coverage high for ladder variants)
    if not bov.empty:
        # Ensure key column exists when using explicit ladders
        if "key_player_loose" not in bov.columns:
            pcol_b2 = _pick_col(bov, ["player", "name", "player_name", "display_name"]) or "player"
            bov[pcol_b2] = bov[pcol_b2].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
            bov["key_player_loose"] = bov[pcol_b2].map(_nm_loose)
    # Ensure merge key exists even if bov is empty
    bov = _ensure_key_player_loose(bov)
    if "key_player_loose" not in bov.columns:
        bov["key_player_loose"] = ""
    merged = bov.merge(preds, on="key_player_loose", how="left", suffixes=("", "_p"))

    # Choose projection per-row based on market
    proj_vals: List[float] = []
    for _, r in merged.iterrows():
        key = r.get("market_key")
        if not key:
            proj_vals.append(np.nan)
            continue
        col = choose_proj_col(merged, key)
        if not col:
            proj_vals.append(np.nan)
            continue
        proj_vals.append(r.get(col))
    merged["proj"] = pd.to_numeric(pd.Series(proj_vals), errors="coerce")
    merged["edge"] = merged["proj"] - merged["line"]

    # Tidy columns
    keep_cols = [
        "player", "team", "event", "market", "line", "over_price", "under_price",
        "game_time", "home_team", "away_team", "is_ladder", "proj", "edge",
    ]
    out_df = merged[[c for c in keep_cols if c in merged.columns]].copy()

    # Order rows by event, market, player, line
    out_df["line"] = pd.to_numeric(out_df.get("line"), errors="coerce")
    out_df = out_df.sort_values(["event", "market", "player", "line"], na_position="last")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
