"""
Compute edges/EV for game-level markets using Bovada game props CSV and model predictions.

Inputs:
  - nfl_compare/data/bovada_game_props_{season}_wk{week}.csv
  - predictions/games via app._load_predictions fallbacks (we'll load from csv helpers if available)

Outputs:
  - nfl_compare/data/edges_game_props_{season}_wk{week}.csv

Markets handled:
  - moneyline: EV for home/away using implied vs pred_home_win_prob
  - spread/alt_spread: EV for home/away using edge -> cover prob via NFL_ATS_SIGMA
  - total/alt_total: EV for Over/Under using edge -> over prob via NFL_TOTAL_SIGMA
  - team_total: EV for Over/Under using model per-team points when available (pred_home_points/pred_away_points)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
from typing import Optional

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"


def american_to_decimal(amer: Optional[float]) -> Optional[float]:
    if amer is None or (isinstance(amer, float) and np.isnan(amer)):
        return None
    try:
        a = float(amer)
    except Exception:
        return None
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / abs(a)
    return None


def ev_from_prob_and_decimal(p: Optional[float], d: Optional[float]) -> Optional[float]:
    if p is None or d is None or not np.isfinite(p) or not np.isfinite(d):
        return None
    try:
        return p * (d - 1.0) - (1.0 - p)
    except Exception:
        return None


def cover_prob_from_edge(edge_pts: float, sigma_pts: float) -> float:
    # Approximate via Normal CDF; convert edge to probability
    try:
        from math import erf, sqrt
        z = edge_pts / float(sigma_pts)
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    except Exception:
        return 0.5


def load_predictions() -> pd.DataFrame:
    # Try to read main predictions from data/predictions_week.csv or nfl_compare/data...
    # Reuse app conventions without importing heavy Flask modules
    cand = [
        BASE_DIR / "data" / "predictions_week.csv",
        DATA_DIR / "predictions_week.csv",
        BASE_DIR / "data" / "predictions.csv",
        DATA_DIR / "predictions.csv",
    ]
    for fp in cand:
        if fp.exists():
            try:
                df = pd.read_csv(fp)
                return df
            except Exception:
                continue
    return pd.DataFrame()


def _load_team_assets() -> dict:
    try:
        fp = DATA_DIR / 'nfl_team_assets.json'
        if fp.exists():
            with open(fp, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _to_abbr_builder():
    assets = _load_team_assets()
    abbr_map = {}
    nick_to_abbr = {}
    if isinstance(assets, dict):
        for full, meta in assets.items():
            try:
                ab = str((meta or {}).get('abbr') or full).strip().upper()
                full_up = str(full).strip().upper()
                abbr_map[full_up] = ab
                abbr_map[ab] = ab
                parts = [p for p in str(full).strip().split() if p]
                if parts:
                    nick = parts[-1].upper()
                    if nick not in nick_to_abbr:
                        nick_to_abbr[nick] = ab
            except Exception:
                continue

    def to_abbr_any(x: object) -> str:
        s = str(x or '').strip()
        if not s:
            return ''
        u = s.upper()
        if u in abbr_map:
            return abbr_map[u]
        parts = [p for p in u.split() if p]
        if parts:
            nick = parts[-1]
            if nick in nick_to_abbr:
                return nick_to_abbr[nick]
        return u
    return to_abbr_any


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute edges for game-level props")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--game-csv", type=str, default=None, help="Override path to bovada_game_props CSV")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = ap.parse_args()

    season = args.season
    week = args.week
    in_fp = Path(args.game_csv) if args.game_csv else (DATA_DIR / f"bovada_game_props_{season}_wk{week}.csv")
    out_fp = Path(args.out) if args.out else (DATA_DIR / f"edges_game_props_{season}_wk{week}.csv")

    if not in_fp.exists():
        print(f"Input not found: {in_fp}")
        return 2
    gdf = pd.read_csv(in_fp)
    pdf = load_predictions()
    if pdf is None or pdf.empty:
        print("WARNING: predictions not found; EVs may be missing")
        pdf = pd.DataFrame()

    # Normalize join keys
    for c in ("season","week"):
        if c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    # Restrict to season/week when present
    if "season" in pdf.columns:
        pdf = pdf[pdf["season"].astype("Int64") == pd.Series([season]*len(pdf), dtype="Int64")]
    if "week" in pdf.columns:
        pdf = pdf[pdf["week"].astype("Int64") == pd.Series([week]*len(pdf), dtype="Int64")]

    # Attach model predictions: pred_home_points, pred_away_points, pred_total, pred_home_win_prob
    keep_cols = [c for c in [
        "game_id","season","week","home_team","away_team",
        "pred_home_points","pred_away_points","pred_total","pred_home_win_prob"
    ] if c in pdf.columns]
    pdf2 = pdf[keep_cols].drop_duplicates() if keep_cols else pd.DataFrame()
    # Normalize team names to abbreviations for robust joins (handles 'Bills' vs 'Buffalo Bills')
    to_abbr_any = _to_abbr_builder()
    df = gdf.copy()
    if not df.empty and {"home_team","away_team"}.issubset(df.columns):
        df["__home_abbr"] = df["home_team"].map(to_abbr_any)
        df["__away_abbr"] = df["away_team"].map(to_abbr_any)
    if not pdf2.empty:
        if {"home_team","away_team"}.issubset(pdf2.columns):
            pdf2 = pdf2.copy()
            pdf2["__home_abbr"] = pdf2["home_team"].map(to_abbr_any)
            pdf2["__away_abbr"] = pdf2["away_team"].map(to_abbr_any)
        # Merge on abbr; if missing, try swapped
        if {"__home_abbr","__away_abbr"}.issubset(pdf2.columns) and {"__home_abbr","__away_abbr"}.issubset(df.columns):
            merged = df.merge(pdf2, on=["__home_abbr","__away_abbr"], how="left")
            # Preserve original home_team/away_team columns when collisions create *_x/*_y
            if 'home_team' not in merged.columns and 'home_team_x' in merged.columns:
                merged['home_team'] = merged['home_team_x'].where(merged['home_team_x'].notna(), merged.get('home_team_y'))
            if 'away_team' not in merged.columns and 'away_team_x' in merged.columns:
                merged['away_team'] = merged['away_team_x'].where(merged['away_team_x'].notna(), merged.get('away_team_y'))
            # Drop helper collision columns if present
            drop_cols = [c for c in merged.columns if c.endswith('_x') or c.endswith('_y')]
            if drop_cols:
                merged = merged.drop(columns=drop_cols)
            miss = merged["pred_total"].isna() if "pred_total" in merged.columns else merged["pred_home_win_prob"].isna()
            if miss.any():
                sw = pdf2.rename(columns={"__home_abbr":"__away_abbr","__away_abbr":"__home_abbr"})
                merged2 = df.merge(sw, on=["__home_abbr","__away_abbr"], how="left", suffixes=("","_sw"))
                for c in ["pred_home_points","pred_away_points","pred_total","pred_home_win_prob"]:
                    c_sw = f"{c}_sw"
                    if c in merged.columns and c_sw in merged2.columns:
                        merged[c] = merged[c].where(merged[c].notna(), merged2[c_sw])
            df = merged

    # Compute EVs per row
    def _safe_float(v: Optional[float]) -> Optional[float]:
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    ats_sigma = _safe_float(os.environ.get("NFL_ATS_SIGMA") or 9.0) or 9.0
    total_sigma = _safe_float(os.environ.get("NFL_TOTAL_SIGMA") or 10.0) or 10.0

    out_rows = []
    for _, r in df.iterrows():
        mk = str(r.get("market_key") or "").strip().lower()
        home = r.get("home_team"); away = r.get("away_team")
        pred_home_win_prob = _safe_float(r.get("pred_home_win_prob"))
        pred_total = _safe_float(r.get("pred_total"))
        pred_hp = _safe_float(r.get("pred_home_points")); pred_ap = _safe_float(r.get("pred_away_points"))
        line = _safe_float(r.get("line"))

        base = {
            "event": r.get("event"),
            "game_time": r.get("game_time"),
            "home_team": home,
            "away_team": away,
            "market_key": r.get("market_key"),
            "team_side": r.get("team_side"),
            "line": line,
            "price_home": r.get("price_home"),
            "price_away": r.get("price_away"),
            "over_price": r.get("over_price"),
            "under_price": r.get("under_price"),
            "is_alternate": r.get("is_alternate"),
        }

        # Moneyline EV (if prices present)
        if mk == "moneyline":
            # Home
            d_home = american_to_decimal(r.get("price_home"))
            d_away = american_to_decimal(r.get("price_away"))
            if pred_home_win_prob is not None:
                evh = ev_from_prob_and_decimal(pred_home_win_prob, d_home) if d_home is not None else None
                eva = ev_from_prob_and_decimal(1.0 - pred_home_win_prob, d_away) if d_away is not None else None
            else:
                evh = eva = None
            out_rows.append({**base, "side": "home", "ev_units": evh})
            out_rows.append({**base, "side": "away", "ev_units": eva})
            continue

        # Spread EV via cover prob
        if mk in {"spread","alt_spread"} and line is not None and pred_hp is not None and pred_ap is not None:
            margin = pred_hp - pred_ap
            # For home side, edge = margin + spread (covers if > 0)
            edge_home = margin + line if line is not None else None
            edge_away = -(margin + line) if line is not None else None
            p_home_cover = cover_prob_from_edge(edge_home, ats_sigma) if edge_home is not None else None
            # Prices
            d_home = american_to_decimal(r.get("price_home"))
            d_away = american_to_decimal(r.get("price_away"))
            evh = ev_from_prob_and_decimal(p_home_cover, d_home) if (p_home_cover is not None and d_home is not None) else None
            eva = ev_from_prob_and_decimal(1.0 - p_home_cover, d_away) if (p_home_cover is not None and d_away is not None) else None
            out_rows.append({**base, "side": "home", "ev_units": evh, "edge_pts": edge_home})
            out_rows.append({**base, "side": "away", "ev_units": eva, "edge_pts": edge_away})
            continue

        # Game Total EV via over prob
        if mk in {"total","alt_total"} and line is not None and pred_total is not None:
            edge_t = pred_total - line
            p_over = cover_prob_from_edge(edge_t, total_sigma)
            d_over = american_to_decimal(r.get("over_price"))
            d_under = american_to_decimal(r.get("under_price"))
            evo = ev_from_prob_and_decimal(p_over, d_over) if d_over is not None else None
            evu = ev_from_prob_and_decimal(1.0 - p_over, d_under) if d_under is not None else None
            out_rows.append({**base, "side": "Over", "ev_units": evo, "edge_pts": edge_t})
            out_rows.append({**base, "side": "Under", "ev_units": evu, "edge_pts": -edge_t})
            continue

        # Team Total EV via per-team points
        if mk == "team_total" and line is not None and (pred_hp is not None or pred_ap is not None):
            # Determine which team
            team_side = str(r.get("team_side") or "").strip().lower()
            pred_team = pred_hp if team_side == "home" else pred_ap if team_side == "away" else None
            if pred_team is not None:
                edge = pred_team - line
                p_over = cover_prob_from_edge(edge, total_sigma)
                d_over = american_to_decimal(r.get("over_price"))
                d_under = american_to_decimal(r.get("under_price"))
                evo = ev_from_prob_and_decimal(p_over, d_over) if d_over is not None else None
                evu = ev_from_prob_and_decimal(1.0 - p_over, d_under) if d_under is not None else None
                out_rows.append({**base, "side": "Over", "ev_units": evo, "edge_pts": edge})
                out_rows.append({**base, "side": "Under", "ev_units": evu, "edge_pts": -edge})

    out = pd.DataFrame(out_rows)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fp, index=False)
    print(f"Wrote {out_fp} with {len(out)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
