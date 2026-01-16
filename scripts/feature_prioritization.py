import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

from nfl_compare.src.data_sources import load_games, load_team_stats, load_lines
from nfl_compare.src.weather import load_weather_for_games
from nfl_compare.src.features import merge_features


def _numeric_series(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                out.append(c)
        except Exception:
            continue
    return out


def _build_targets(feat: pd.DataFrame) -> Dict[str, pd.Series]:
    # Moneyline home win
    ml = None
    try:
        hs = pd.to_numeric(feat.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(feat.get("away_score"), errors="coerce")
        ml = (hs > as_).astype(int)
    except Exception:
        ml = pd.Series(index=feat.index, dtype=float)

    # ATS home cover: margin + spread_home > 0
    ats = None
    try:
        spread = pd.to_numeric(feat.get("close_spread_home"), errors="coerce")
        if spread.isna().all():
            spread = pd.to_numeric(feat.get("spread_home"), errors="coerce")
        hs = pd.to_numeric(feat.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(feat.get("away_score"), errors="coerce")
        margin = hs - as_
        ats = ((margin + spread) > 0).astype(int)
    except Exception:
        ats = pd.Series(index=feat.index, dtype=float)

    # Total over: home_score + away_score > total
    tot_over = None
    try:
        total = pd.to_numeric(feat.get("close_total"), errors="coerce")
        if total.isna().all():
            total = pd.to_numeric(feat.get("total"), errors="coerce")
        hs = pd.to_numeric(feat.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(feat.get("away_score"), errors="coerce")
        game_total = hs + as_
        tot_over = (game_total > total).astype(int)
    except Exception:
        tot_over = pd.Series(index=feat.index, dtype=float)

    return {"ML_HOME_WIN": ml, "ATS_HOME_COVER": ats, "TOTAL_OVER": tot_over}


def _rank_features(df: pd.DataFrame, target: pd.Series, candidate_cols: List[str], top_k: int = 25) -> pd.DataFrame:
    # Filter rows with valid target
    mask = target.notna()
    y = target[mask].astype(int)
    work = df.loc[mask, candidate_cols].copy()
    # numeric coercion
    for c in list(work.columns):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.fillna(work.median(numeric_only=True))
    X = work.values
    # Mutual information
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    except Exception:
        mi = np.zeros(len(candidate_cols))
    # Simple AUC for single-feature logistic thresholds: rank correlation proxy
    aucs = []
    for i, c in enumerate(candidate_cols):
        try:
            s = work[c]
            # If low variance, skip
            if s.nunique() <= 2:
                aucs.append(np.nan)
                continue
            # Use raw feature as score; higher implies class 1
            auc = roc_auc_score(y, s)
            aucs.append(auc)
        except Exception:
            aucs.append(np.nan)
    out = pd.DataFrame({"feature": candidate_cols, "mutual_info": mi, "auc_proxy": aucs})
    out = out.sort_values(["mutual_info"], ascending=False).head(top_k)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rank data features by mutual information and AUC proxies vs outcomes")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, default=17)
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()

    games = load_games()
    stats = load_team_stats()
    lines = load_lines()
    wx = load_weather_for_games(games)
    feat = merge_features(games, stats, lines, wx).copy()
    # Deduplicate by game_id
    if "game_id" in feat.columns:
        feat = feat.sort_values(["season", "week", "game_id"]).drop_duplicates(subset=["game_id"], keep="first")

    # Limit to target season window and completed games
    feat["season"] = pd.to_numeric(feat.get("season"), errors="coerce")
    feat["week"] = pd.to_numeric(feat.get("week"), errors="coerce")
    mask = (feat["season"] == int(args.season)) & (feat["week"] >= int(args.start_week)) & (feat["week"] <= int(args.end_week))
    feat = feat[mask].copy()
    feat = feat.dropna(subset=["home_score", "away_score"]).copy()
    if feat.empty:
        print("No completed games in the selected window.")
        return 0

    # Candidate columns: numeric, exclude target/leakage columns
    exclude = set(["home_score", "away_score", "game_total", "margin", "prob_home_win", "prob_over_total", "prob_home_cover"])  # predictions/leaks
    leak_substrings = [
        "score", "points", "margin", "first_half", "second_half", "q1", "q2", "q3", "q4",
    ]
    def _is_leak(c: str) -> bool:
        lc = c.lower()
        if c in exclude:
            return True
        return any(s in lc for s in leak_substrings)
    candidate_cols = [c for c in feat.columns if not _is_leak(c)]
    candidate_cols = _numeric_series(feat, candidate_cols)

    targets = _build_targets(feat)
    results = {}
    for name, y in targets.items():
        try:
            out = _rank_features(feat, y, candidate_cols, top_k=args.top_k)
            results[name] = out.to_dict(orient="records")
        except Exception as e:
            results[name] = {"error": str(e)}

    out_dir = Path(__file__).resolve().parents[1] / "nfl_compare" / "data" / "backtests" / f"{args.season}_wk{args.end_week}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    out_fp = out_dir / "feature_priority.json"
    try:
        import json
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_fp}")
    except Exception as e:
        print(f"Failed writing feature_priority.json: {e}")
    # Also print concise top-10 per market
    for name, rows in results.items():
        print(name, ":", rows[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
