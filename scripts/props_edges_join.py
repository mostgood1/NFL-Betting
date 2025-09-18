"""
Join Bovada player prop lines with model projections and compute edges/EV.

Usage (example):
  python scripts/props_edges_join.py --season 2025 --week 3 \
    --bovada nfl_compare/data/bovada_player_props_2025_wk3.csv \
    --out nfl_compare/data/edges_player_props_2025_wk3.csv

Expected Bovada CSV columns (flexible, best-effort parsing):
  - player: Player name (e.g., "Justin Jefferson")
  - team: Optional team code/name
  - market: e.g., "Receiving Yards", "Receptions", "Rushing Yards",
            "Passing Yards", "Anytime TD"
  - line: Numeric prop line (yards/receptions) or implied for TD markets (ignored)
  - over_price: American odds for over (e.g., -115, +120). Optional
  - under_price: American odds for under. Optional
  - book: Optional (e.g., "Bovada")

Notes:
  - For yardage/receptions markets, we compute simple difference `edge = proj - line`.
  - For Anytime TD, we compute EV for over/under using model probability when present.
  - We do not synthesize prices. If odds/prices are missing, EV is left null.
  - Matching is performed by normalized player name (case-insensitive). When available,
    we also attempt to match team.

This script is intentionally dependency-light and robust to partial columns; if a particular
projection column is missing for a market, it will skip that row with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import math
import pandas as pd
import numpy as np

# Use shared name normalization utilities for better joins
try:
    from nfl_compare.src.name_normalizer import (
        normalize_name_loose as _nm_loose,
        normalize_alias_init_last as _nm_alias,
    )
except Exception:
    # Fallbacks if module not importable in some environments
    def _nm_loose(s: Optional[str]) -> str:
        if s is None:
            return ""
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _nm_alias(s: Optional[str]) -> str:
        if not s:
            return ""
        s = str(s).strip().lower()
        parts = [p for p in s.replace("-", " ").replace(".", " ").split() if p]
        if not parts:
            return ""
        first = parts[0][:1]
        last = parts[-1]
        return f"{first}{''.join(ch for ch in last if ch.isalnum())}"


# ----------------------------- Helpers ---------------------------------

def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower()


def american_to_decimal(odds: Optional[float | int | str]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    try:
        o = float(str(odds).replace("+", ""))
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))


def implied_prob_from_american(odds: Optional[float | int | str]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(str(odds).replace("+", ""))
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return abs(o) / (abs(o) + 100.0)


def ev_from_prob_and_american(prob: Optional[float], odds: Optional[float | int | str]) -> Optional[float]:
    """
    Returns expected value (per 1 stake) given success probability and American odds.
    EV = p*(decimal-1) - (1-p)*1
    """
    if prob is None:
        return None
    d = american_to_decimal(odds)
    if d is None:
        return None
    return float(prob) * (d - 1.0) - (1.0 - float(prob))


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except Exception:
        return 0.0


def _poisson_cdf(k: int, lam: float, max_k: int = 30) -> float:
    """Return P(X <= k) for X~Poisson(lam). For numerical safety, cap summation."""
    if lam < 0 or k < 0:
        return 0.0
    k = int(k)
    # Cap max k to avoid long loops; interceptions/TDs means are small so this is fine
    k_cap = min(max_k, k)
    s = 0.0
    for i in range(0, k_cap + 1):
        s += _poisson_pmf(i, lam)
    # If k exceeds cap, approximate tail is ~0 for small lam; acceptable here
    return min(max(s, 0.0), 1.0)


# Map inbound market strings to internal projection keys
MARKET_MAP: Dict[str, str] = {
    "receiving yards": "rec_yards",
    "rec yds": "rec_yards",
    "receptions": "receptions",
    "rushing yards": "rush_yards",
    "rush yds": "rush_yards",
    "passing yards": "pass_yards",
    "pass yds": "pass_yards",
    # QB passing touchdowns
    "passing touchdowns": "pass_tds",
    "passing tds": "pass_tds",
    "pass tds": "pass_tds",
    "pass touchdowns": "pass_tds",
    # QB/Player counting markets
    "passing attempts": "pass_attempts",
    "pass attempts": "pass_attempts",
    "rushing attempts": "rush_attempts",
    "rush attempts": "rush_attempts",
    "interceptions": "interceptions",
    "interceptions thrown": "interceptions",
    "anytime td": "any_td",
    "any time td": "any_td",
    # Combined yards and volume
    "rush+rec yards": "rush_rec_yards",
    "rushing + receiving yards": "rush_rec_yards",
    "rush + rec yards": "rush_rec_yards",
    "pass+rush yards": "pass_rush_yards",
    "pass + rush yards": "pass_rush_yards",
    "passing + rushing yards": "pass_rush_yards",
    "targets": "targets",
    # Multi-TD (2+ touchdowns)
    "2+ touchdowns": "multi_tds",
}


# Candidate projection columns for each internal key
PROJ_COLS: Dict[str, List[str]] = {
    "rec_yards": ["rec_yards", "pred_rec_yards", "expected_rec_yards"],
    "receptions": ["receptions", "pred_receptions", "expected_receptions"],
    "rush_yards": ["rush_yards", "pred_rush_yards", "expected_rush_yards"],
    "pass_yards": ["pass_yards", "pred_pass_yards", "expected_pass_yards"],
    # QB passing TD projections
    "pass_tds": ["pass_tds", "pred_pass_tds", "expected_pass_tds"],
    # QB/Player counting projections
    "pass_attempts": ["pass_attempts", "pred_pass_attempts", "expected_pass_attempts"],
    "rush_attempts": ["rush_attempts", "pred_rush_attempts", "expected_rush_attempts"],
    "interceptions": ["interceptions", "pred_interceptions", "expected_interceptions", "ints", "pred_ints"],
    # For TD, use model probability directly (not a line). We'll read:
    "any_td": ["any_td_prob", "anytime_td_prob", "prob_any_td"],
    # Derived/volume
    "rush_rec_yards": ["rush_yards", "rec_yards"],
    "pass_rush_yards": ["pass_yards", "rush_yards"],
    "targets": ["targets", "pred_targets", "expected_targets"],
    # Multi-TD uses combined rushing+receiving TD mean
    "multi_tds": ["rush_tds", "rec_tds"],
}


PLAYER_COL_CANDIDATES = ["display_name", "player", "name", "player_name"]
TEAM_COL_CANDIDATES = ["team", "posteam", "pos_team", "team_abbr", "team_code"]
POS_COL_CANDIDATES = ["position", "pos"]


def pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_proj_col(preds: pd.DataFrame, key: str) -> Optional[str]:
    for c in PROJ_COLS.get(key, []):
        if c in preds.columns:
            return c
    return None


def load_predictions(season: int, week: int, base_dir: Optional[Path] = None) -> pd.DataFrame:
    base = Path(base_dir) if base_dir else Path("nfl_compare/data")
    path = base / f"player_props_{season}_wk{week}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_csv(path)
    return df


def load_bovada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bovada CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize expected columns
    # Lower-case a 'market' string for mapping
    if "market" in df.columns:
        df["market_norm"] = df["market"].astype(str).str.strip().str.lower()
    else:
        df["market_norm"] = np.nan
    # Normalize player key(s)
    pcol = pick_first_col(df, PLAYER_COL_CANDIDATES) or "player"
    if pcol not in df.columns:
        raise ValueError("Bovada CSV must contain a player column (e.g., 'player').")
    # Strip trailing team tags like "(MIA)" from player names for robust matches
    df[pcol] = df[pcol].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
    df["key_player"] = df[pcol].astype(str).str.strip().str.lower()
    df["key_player_loose"] = df[pcol].astype(str).map(_nm_loose)
    df["key_player_alias"] = df[pcol].astype(str).map(_nm_alias)
    # Team if available
    tcol = pick_first_col(df, TEAM_COL_CANDIDATES)
    # If team column missing/empty, try to detect a tag that may remain (already stripped from player); keep as-is otherwise
    if tcol and tcol in df.columns:
        df["key_team"] = df[tcol].astype(str).str.strip().str.lower()
    else:
        df["key_team"] = np.nan
    # Optional: collapse multiple lines per (player, market, event) to a single representative
    # Choose the line closest to the group's median; if tie, prefer the lower absolute line.
    try:
        df["line"] = pd.to_numeric(df.get("line"), errors="coerce")
        # Preserve ladder rows exactly as-is; collapse only non-ladder rows
        is_ladder_mask = df.get("is_ladder").fillna(False) == True
        base = df.loc[~is_ladder_mask].copy()
        ladd = df.loc[is_ladder_mask].copy()
        grp_keys = [k for k in ["key_player", "market_norm", "event"] if k in base.columns]
        if grp_keys and not base.empty:
            def _pick_df(g: pd.DataFrame) -> pd.DataFrame:
                if "line" not in g.columns or g["line"].notna().sum() == 0:
                    return g.iloc[:1].copy()
                med = g["line"].median()
                g2 = g.copy()
                g2["_dist"] = (g2["line"] - med).abs()
                return g2.sort_values(["_dist", "line"], ascending=[True, True]).head(1)
            base = base.groupby(grp_keys, as_index=False, group_keys=False).apply(_pick_df)
            if isinstance(base, pd.Series):
                base = base.to_frame().T
            try:
                base = base.reset_index(drop=True)
            except Exception:
                pass
            if "_dist" in base.columns:
                base = base.drop(columns=["_dist"])  # cleanup helper column
        # Recombine
        df = pd.concat([base, ladd], ignore_index=True)
    except Exception:
        # Be forgiving; if collapsing fails, keep original rows
        pass
    # Ensure is_ladder exists if provided
    if "is_ladder" not in df.columns:
        df["is_ladder"] = np.nan
    return df


def compute_edges(
    season: int,
    week: int,
    bovada_csv: Path,
    out_csv: Path,
    base_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds = load_predictions(season, week, base_dir)
    bov = load_bovada(bovada_csv)

    p_pcol = pick_first_col(preds, PLAYER_COL_CANDIDATES)
    if not p_pcol:
        raise ValueError("Predictions file missing player column (display_name/player/name)")
    preds = preds.copy()
    # Add a merge marker to detect unmatched rows reliably (Bovada also has a 'player' column)
    preds["_pred_marker"] = 1
    preds["key_player"] = preds[p_pcol].astype(str).str.strip().str.lower()
    preds["key_player_loose"] = preds[p_pcol].astype(str).map(_nm_loose)
    preds["key_player_alias"] = preds[p_pcol].astype(str).map(_nm_alias)
    p_tcol = pick_first_col(preds, TEAM_COL_CANDIDATES)
    preds["key_team"] = preds[p_tcol].astype(str).str.strip().str.lower() if p_tcol else np.nan
    p_poscol = pick_first_col(preds, POS_COL_CANDIDATES)

    # Helper to perform a merge and fill back unmatched slots
    def _merge_fill(base: pd.DataFrame, right: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
        merged_local = base.merge(right, on=on_cols, how="left", suffixes=("", "_p"))
        # Unmatched when prediction marker is NaN
        mask_unmatched = merged_local["_pred_marker"].isna()
        return merged_local, mask_unmatched

    # Multi-pass join for robustness:
    merged, um = _merge_fill(bov, preds, ["key_player", "key_team"])
    left_cols = list(bov.columns)
    if um.any():
        # Pass 2: loose + team
        fallback = merged.loc[um, left_cols].copy()
        fallback["_row_idx"] = fallback.index
        merged2 = fallback.merge(preds, on=["key_player_loose", "key_team"], how="left", suffixes=("", "_p"))
        fill_cols = [c for c in merged2.columns if c not in left_cols and c != "_row_idx"]
        if fill_cols:
            for _, r in merged2.iterrows():
                idx = r["_row_idx"]
                if pd.notna(r.get("_pred_marker")):
                    for c in fill_cols:
                        merged.at[idx, c] = r.get(c)
        um = merged["_pred_marker"].isna()
    if um.any():
        # Pass 3: alias + team
        fallback = merged.loc[um, left_cols].copy()
        fallback["_row_idx"] = fallback.index
        merged3 = fallback.merge(preds, on=["key_player_alias", "key_team"], how="left", suffixes=("", "_p"))
        fill_cols = [c for c in merged3.columns if c not in left_cols and c != "_row_idx"]
        if fill_cols:
            for _, r in merged3.iterrows():
                idx = r["_row_idx"]
                if pd.notna(r.get("_pred_marker")):
                    for c in fill_cols:
                        merged.at[idx, c] = r.get(c)
        um = merged["_pred_marker"].isna()
    if um.any():
        # Pass 4: loose only
        fallback = merged.loc[um, left_cols].copy()
        fallback["_row_idx"] = fallback.index
        preds4 = preds.drop(columns=["key_team"]).copy()
        merged4 = fallback.merge(preds4, on=["key_player_loose"], how="left", suffixes=("", "_p"))
        fill_cols = [c for c in merged4.columns if c not in left_cols and c != "_row_idx"]
        if fill_cols:
            for _, r in merged4.iterrows():
                idx = r["_row_idx"]
                if pd.notna(r.get("_pred_marker")):
                    for c in fill_cols:
                        merged.at[idx, c] = r.get(c)
        um = merged["_pred_marker"].isna()
    if um.any():
        # Pass 5: alias only
        fallback = merged.loc[um, left_cols].copy()
        fallback["_row_idx"] = fallback.index
        preds5 = preds.drop(columns=["key_team"]).copy()
        merged5 = fallback.merge(preds5, on=["key_player_alias"], how="left", suffixes=("", "_p"))
        fill_cols = [c for c in merged5.columns if c not in left_cols and c != "_row_idx"]
        if fill_cols:
            for _, r in merged5.iterrows():
                idx = r["_row_idx"]
                if pd.notna(r.get("_pred_marker")):
                    for c in fill_cols:
                        merged.at[idx, c] = r.get(c)
        um = merged["_pred_marker"].isna()

    # Drop internal marker
    if "_pred_marker" in merged.columns:
        merged = merged.drop(columns=["_pred_marker"]) 

    # Map market -> internal key and projection column
    def market_key(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        s2 = s.strip().lower()
        return MARKET_MAP.get(s2)

    merged["market_key"] = merged["market_norm"].map(market_key)

    # Coerce numeric line/prices if present
    for c in ("line", "over_price", "under_price"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Compute projection per row
    proj_vals: List[Optional[float]] = []
    notes: List[str] = []
    for i, row in merged.iterrows():
        key = row.get("market_key")
        if not key:
            proj_vals.append(np.nan)
            notes.append("unsupported_market")
            continue
        # Derived sums
        if key in {"rush_rec_yards", "pass_rush_yards"}:
            parts = PROJ_COLS.get(key, [])
            vals = []
            for c in parts:
                v = row.get(c)
                vals.append(float(v)) if (v is not None and not pd.isna(v)) else None
            if all(v is not None for v in vals) and len(vals) == 2:
                proj_vals.append(float(vals[0]) + float(vals[1]))
                notes.append("")
            else:
                proj_vals.append(np.nan)
                notes.append("projection_missing")
            continue
        # Direct columns
        pcol = choose_proj_col(merged, key)
        if not pcol or pcol not in merged.columns:
            proj_vals.append(np.nan)
            notes.append("no_projection_column")
            continue
        val = row.get(pcol)
        if pd.isna(val):
            proj_vals.append(np.nan)
            notes.append("projection_missing")
            continue
        # Multi-TD: sum rush_tds + rec_tds to get λ
        if key == "multi_tds":
            try:
                lam = 0.0
                for c in ["rush_tds", "rec_tds"]:
                    vc = row.get(c)
                    if vc is not None and not pd.isna(vc):
                        lam += float(vc)
                proj_vals.append(lam if lam > 0 else np.nan)
                notes.append("")
                continue
            except Exception:
                proj_vals.append(np.nan)
                notes.append("projection_missing")
                continue
        proj_vals.append(float(val))
        notes.append("")

    merged["proj"] = proj_vals
    merged["note"] = notes

    # Edge and EV
    merged["edge"] = np.nan
    if "line" in merged.columns:
        # Only meaningful for yardage/receptions/counting stat markets
        yard_markets = {"rec_yards", "rush_yards", "pass_yards", "receptions", "pass_tds", "pass_attempts", "rush_attempts", "interceptions", "rush_rec_yards", "pass_rush_yards", "targets"}
        mask_yard = merged["market_key"].isin(list(yard_markets)) & merged["proj"].notna() & merged["line"].notna()
        merged.loc[mask_yard, "edge"] = merged.loc[mask_yard, "proj"] - merged.loc[mask_yard, "line"]

    # EV for Anytime TD (over/under)
    merged["over_ev"] = np.nan
    merged["under_ev"] = np.nan
    mask_td = (merged["market_key"] == "any_td") & merged["proj"].notna()
    if mask_td.any():
        p = merged.loc[mask_td, "proj"].astype(float)
        over_odds = merged.loc[mask_td, "over_price"] if "over_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
        under_odds = merged.loc[mask_td, "under_price"] if "under_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
        over_vals: List[float] = []
        under_vals: List[float] = []
        for pi, oi, ui in zip(p, over_odds, under_odds):
            ove = ev_from_prob_and_american(float(pi), oi)
            une = ev_from_prob_and_american(1.0 - float(pi), ui)
            over_vals.append(np.nan if ove is None else float(ove))
            under_vals.append(np.nan if une is None else float(une))
        merged.loc[mask_td, "over_ev"] = over_vals
        merged.loc[mask_td, "under_ev"] = under_vals

    # EV for Interceptions and Passing TDs via Poisson model
    for mkey in ["interceptions", "pass_tds"]:
        mask = (merged["market_key"] == mkey) & merged["proj"].notna() & merged["line"].notna()
        if mask.any():
            mu = merged.loc[mask, "proj"].astype(float)
            line_vals = merged.loc[mask, "line"].astype(float)
            over_odds = merged.loc[mask, "over_price"] if "over_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
            under_odds = merged.loc[mask, "under_price"] if "under_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
            over_list: List[float] = []
            under_list: List[float] = []
            for lam, L, oi, ui in zip(mu, line_vals, over_odds, under_odds):
                try:
                    # Over threshold: X >= floor(L) + 1 for typical .5 lines; general case works too
                    k_over = int(math.floor(L) + 1)
                    p_over = 1.0 - _poisson_cdf(k_over - 1, float(lam))
                    p_under = 1.0 - p_over
                    ove = ev_from_prob_and_american(p_over, oi)
                    une = ev_from_prob_and_american(p_under, ui)
                except Exception:
                    ove = None; une = None
                over_list.append(np.nan if ove is None else float(ove))
                under_list.append(np.nan if une is None else float(une))
            merged.loc[mask, "over_ev"] = over_list
            merged.loc[mask, "under_ev"] = under_list

    # EV for Multi-TD (2+ Touchdowns) via Poisson model (usually single-sided Yes price)
    mask_mt = (merged["market_key"] == "multi_tds") & merged["proj"].notna()
    if mask_mt.any():
        mu = merged.loc[mask_mt, "proj"].astype(float)
        # If a numeric line is present (e.g., 2), use it; else default to 2
        line_vals = merged.loc[mask_mt, "line"] if "line" in merged.columns else pd.Series(index=merged.index, dtype=float)
        over_odds = merged.loc[mask_mt, "over_price"] if "over_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
        under_odds = merged.loc[mask_mt, "under_price"] if "under_price" in merged.columns else pd.Series(index=merged.index, dtype=float)
        over_list: List[float] = []
        under_list: List[float] = []
        for lam, L, oi, ui in zip(mu, line_vals, over_odds, under_odds):
            try:
                thr = 2.0 if (L is None or (isinstance(L, float) and math.isnan(L))) else float(L)
                k_over = int(math.floor(thr))
                p_over = 1.0 - _poisson_cdf(k_over - 1, float(lam))
                p_under = 1.0 - p_over
                ove = ev_from_prob_and_american(p_over, oi)
                une = ev_from_prob_and_american(p_under, ui)
            except Exception:
                ove = None; une = None
            over_list.append(np.nan if ove is None else float(ove))
            under_list.append(np.nan if une is None else float(une))
        merged.loc[mask_mt, "over_ev"] = over_list
        merged.loc[mask_mt, "under_ev"] = under_list

    # Output selection
    out_cols = []
    for c in [
        # Bovada columns
        pick_first_col(bov, PLAYER_COL_CANDIDATES) or "player",
        pick_first_col(bov, TEAM_COL_CANDIDATES) or "team",
        "market",
        "line",
        "over_price",
        "under_price",
        "is_ladder",
        "event",
        "home_team",
        "away_team",
        "game_time",
        "book",
        # Predictions context
        p_pcol,
        p_tcol,
        p_poscol,
        # Our computed
        "market_key",
        "proj",
        "edge",
        "over_ev",
        "under_ev",
        "note",
    ]:
        if c and c in merged.columns and c not in out_cols:
            out_cols.append(c)

    out_df = merged[out_cols].copy()

    # Rank suggestions: by EV (if TD) then by absolute edge for yardage
    out_df["rank_score"] = 0.0
    # Prefer EV for TD markets
    is_td = out_df["market_key"] == "any_td"
    if "over_ev" in out_df.columns:
        out_df.loc[is_td & out_df["over_ev"].notna(), "rank_score"] = out_df.loc[is_td & out_df["over_ev"].notna(), "over_ev"].astype(float)
    # For yardage/receptions, use absolute edge
    is_yard = out_df["market_key"].isin(["rec_yards", "rush_yards", "pass_yards", "receptions", "pass_attempts", "rush_attempts", "interceptions", "pass_tds", "rush_rec_yards", "pass_rush_yards", "targets"]) & out_df["edge"].notna()
    out_df.loc[is_yard, "rank_score"] = out_df.loc[is_yard, "edge"].abs().astype(float)

    # Sort descending by rank_score
    out_df = out_df.sort_values(by=["rank_score"], ascending=False)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    return merged, out_df


def main():
    ap = argparse.ArgumentParser(description="Join Bovada player props with model projections to compute edges/EV.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--bovada", type=str, required=True, help="Path to Bovada player props CSV")
    ap.add_argument("--out", type=str, required=True, help="Path to write edges CSV")
    ap.add_argument("--data-dir", type=str, default="nfl_compare/data", help="Base directory for predictions data")
    args = ap.parse_args()

    try:
        merged, out_df = compute_edges(
            season=args.season,
            week=args.week,
            bovada_csv=Path(args.bovada),
            out_csv=Path(args.out),
            base_dir=Path(args.data_dir),
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Hint: ensure you have generated player props and exported Bovada CSV with columns: player, market, line, over_price, under_price, team (optional).")
        return 2
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Wrote {args.out} with {len(out_df)} rows.")
    # Show quick top-10 preview
    preview_cols = [c for c in ["player", "team", "market", "line", "proj", "edge", "over_price", "over_ev", "under_price", "under_ev", "rank_score"] if c in out_df.columns]
    try:
        print(out_df.head(10)[preview_cols].to_string(index=False))
    except Exception:
        # Fallback if preview_cols missing
        print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
