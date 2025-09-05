from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .td_likelihood import compute_td_likelihood
from .team_normalizer import normalize_team_name
from .data_sources import DATA_DIR as TEAM_DATA_DIR


DATA_DIR = TEAM_DATA_DIR  # reuse same data folder


def _load_player_usage() -> pd.DataFrame:
    """Load optional player usage priors.

    Expected file: data/player_usage_priors.csv with columns:
      season, team, player, position, rush_share, target_share, rz_rush_share, rz_target_share
    Shares should be in [0,1]. Rows can be per-season team priors.
    Returns empty DataFrame if absent; the script will fallback to defaults.
    """
    fp = DATA_DIR / "player_usage_priors.csv"
    if not fp.exists():
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])
    return df


def _enrich_player_names(players_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Enrich player names with last names using nfl_data_py seasonal rosters when available.

    Strategy:
    - Load nfl_data_py.import_seasonal_rosters for the season.
    - Normalize team names and build lookup of first-name->best display name per team and position.
    - If a player's name in players_df has no space (likely first name only), and we find an exact
      first-name match on the same team and position, replace with the display/full name.
    - Leave as-is when no confident match exists.
    """
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return players_df

    try:
        ros = nfl.import_seasonal_rosters([season])
    except Exception:
        return players_df
    if ros is None or ros.empty:
        return players_df

    df = players_df.copy()
    # Build team-normalized lookup
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if team_src is None:
        return df
    ros = ros.copy()
    ros['team_norm'] = ros[team_src].astype(str).apply(normalize_team_name)
    # Best available display name
    def best_name(r: pd.Series) -> str:
        for k in ['display_name','player_display_name','full_name','player_name','football_name']:
            v = r.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        fn = str(r.get('first_name') or '').strip()
        ln = str(r.get('last_name') or '').strip()
        nm = f"{fn} {ln}".strip()
        return nm if nm else str(r.get('gsis_id') or '').strip()

    ros['best_name'] = ros.apply(best_name, axis=1)
    pos_col = 'position' if 'position' in ros.columns else None
    # Build mappings with normalized tokens to be robust to punctuation (e.g., D.J. vs DJ)
    def first_token(s: str) -> str:
        parts = s.split()
        return parts[0] if parts else s
    def norm_token(s: str) -> str:
        return ''.join(ch for ch in s if ch.isalnum()).lower()
    lut_pos: dict[tuple[str, str, str], set[str]] = {}
    lut_any: dict[tuple[str, str], set[str]] = {}
    for _, rr in ros.iterrows():
        team_norm = str(rr['team_norm'])
        pos = str(rr.get(pos_col) or '') if pos_col else ''
        bn = best_name(rr)
        tok = norm_token(first_token(bn))
        # also include normalized first_name as key when available
        fn = str(rr.get('first_name') or '')
        fn_tok = norm_token(first_token(fn)) if fn else ''
        for key_tok in filter(None, [tok, fn_tok]):
            kpos = (team_norm, pos, key_tok)
            lut_pos.setdefault(kpos, set()).add(rr['best_name'])
            kany = (team_norm, key_tok)
            lut_any.setdefault(kany, set()).add(rr['best_name'])

    # Apply enrichment
    def enrich_row(r: pd.Series) -> str:
        name = str(r.get('player') or '').strip()
        if not name:
            return name
        # if it already has a space, assume full name and leave
        if ' ' in name:
            return name
        team_norm = normalize_team_name(str(r.get('team') or ''))
        pos = str(r.get('position') or '')
        tok = norm_token(name)
        cands = lut_pos.get((team_norm, pos, tok))
        if not cands:
            cands = lut_any.get((team_norm, tok))
        if not cands:
            return name
        # If unique, return it; else keep original to avoid wrong merge
        if len(cands) == 1:
            return next(iter(cands))
        return name

    df['player'] = df.apply(enrich_row, axis=1)
    return df


def _default_team_depth(team: str) -> pd.DataFrame:
    """Fallback depth with generic buckets by position if no player priors present.
    Allocates plausible shares to RB/WR/TE and QB for rushing TDs, passing TDs allocated to receivers.
    """
    rows = [
        {"player": f"{team} QB1", "position": "QB", "rush_share": 0.10, "target_share": 0.00, "rz_rush_share": 0.10, "rz_target_share": 0.00},
        {"player": f"{team} RB1", "position": "RB", "rush_share": 0.45, "target_share": 0.10, "rz_rush_share": 0.50, "rz_target_share": 0.08},
        {"player": f"{team} RB2", "position": "RB", "rush_share": 0.25, "target_share": 0.05, "rz_rush_share": 0.25, "rz_target_share": 0.05},
        {"player": f"{team} WR1", "position": "WR", "rush_share": 0.03, "target_share": 0.25, "rz_rush_share": 0.02, "rz_target_share": 0.25},
        {"player": f"{team} WR2", "position": "WR", "rush_share": 0.02, "target_share": 0.20, "rz_rush_share": 0.01, "rz_target_share": 0.20},
        {"player": f"{team} WR3", "position": "WR", "rush_share": 0.01, "target_share": 0.12, "rz_rush_share": 0.01, "rz_target_share": 0.12},
        {"player": f"{team} TE1", "position": "TE", "rush_share": 0.00, "target_share": 0.15, "rz_rush_share": 0.00, "rz_target_share": 0.20},
        {"player": f"{team} TE2", "position": "TE", "rush_share": 0.00, "target_share": 0.05, "rz_rush_share": 0.00, "rz_target_share": 0.10},
    ]
    return pd.DataFrame(rows)


def _normalize_shares(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        s = out[c].sum()
        if s > 0:
            out[c] = out[c] / s
    return out


def _team_player_shares(usage: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    if usage is None or usage.empty:
        base = _default_team_depth(team)
    else:
        u = usage.copy()
        u = u[(pd.to_numeric(u.get("season"), errors='coerce') == season) & (u.get("team").astype(str) == team)]
        base = u if not u.empty else _default_team_depth(team)
    base = _normalize_shares(base, ["rush_share","target_share","rz_rush_share","rz_target_share"])
    base["team"] = team
    return base


def _split_team_tds(row: pd.Series) -> Dict[str, float]:
    """Split team expected TDs into rushing vs passing using pass/rush priors when present."""
    exp_tds = float(row.get("expected_tds") or 0.0)
    # derive pass/rush mix
    # prefer home_* when is_home==1 else away_*
    is_home = int(row.get("is_home") or 0)
    if is_home:
        pr = row.get("home_pass_rate_prior")
        rr = row.get("home_rush_rate_prior")
    else:
        pr = row.get("away_pass_rate_prior")
        rr = row.get("away_rush_rate_prior")
    try:
        pr = float(pr) if pr is not None else np.nan
    except Exception:
        pr = np.nan
    try:
        rr = float(rr) if rr is not None else np.nan
    except Exception:
        rr = np.nan
    # If both missing, use neutral 58/42 pass/rush split
    if not np.isfinite(pr) and not np.isfinite(rr):
        pr, rr = 0.58, 0.42
    elif not np.isfinite(pr) and np.isfinite(rr):
        pr = max(0.0, min(1.0, 1.0 - rr))
    elif not np.isfinite(rr) and np.isfinite(pr):
        rr = max(0.0, min(1.0, 1.0 - pr))
    # Convert to TD split heuristic favoring passing TDs overall (~65% league-wide) while respecting team pass/rush tendency.
    # Scale back to unity for expected TDs
    w_rush = 0.35 * rr
    w_pass = 0.65 * pr
    s = w_rush + w_pass
    if s <= 0:
        w_rush, w_pass = 0.42, 0.58
        s = 1.0
    rush_tds = exp_tds * (w_rush / s)
    pass_tds = exp_tds * (w_pass / s)
    return {"rush_tds": float(rush_tds), "pass_tds": float(pass_tds)}


def compute_player_td_likelihood(season: int, week: int) -> pd.DataFrame:
    teams = compute_td_likelihood(season=season, week=week)
    if teams is None or teams.empty:
        return pd.DataFrame(columns=["season","week","team","player","position","anytime_td_prob","expected_td"])

    usage = _load_player_usage()

    rows: list[dict] = []
    for _, r in teams.iterrows():
        team = str(r.get("team"))
        opp = str(r.get("opponent"))
        exp_tds = float(r.get("expected_tds") or 0.0)
        split = _split_team_tds(r)
        rush_tds = split["rush_tds"]
        pass_tds = split["pass_tds"]

        depth = _team_player_shares(usage, int(r.get("season")), team)
        # Allocate rushing expected TDs to QB/RB by rz_rush_share when present, else rush_share
        rush_col = "rz_rush_share" if "rz_rush_share" in depth.columns else "rush_share"
        pass_col = "rz_target_share" if "rz_target_share" in depth.columns else "target_share"
        depth[rush_col] = pd.to_numeric(depth[rush_col], errors='coerce').fillna(0.0)
        depth[pass_col] = pd.to_numeric(depth[pass_col], errors='coerce').fillna(0.0)
        # Normalize safeguard
        if depth[rush_col].sum() > 0:
            depth[rush_col] = depth[rush_col] / depth[rush_col].sum()
        if depth[pass_col].sum() > 0:
            depth[pass_col] = depth[pass_col] / depth[pass_col].sum()

        # Rebalance pass allocation across positions to reflect league rates (WR heavy)
        desired_group = {"WR": 0.65, "TE": 0.25, "RB": 0.10}
        desired_group_rz = {"WR": 0.55, "TE": 0.35, "RB": 0.10}
        desired = desired_group_rz if pass_col == "rz_target_share" else desired_group
        # compute current group sums
        def _group_sum(pos: str) -> float:
            m = depth["position"].astype(str).str.upper() == pos
            return float(depth.loc[m, pass_col].sum())
        for pos in ("WR", "TE", "RB"):
            cur = _group_sum(pos)
            tgt = desired.get(pos, 0.0)
            if cur > 0 and tgt >= 0:
                factor = tgt / cur
                m = depth["position"].astype(str).str.upper() == pos
                depth.loc[m, pass_col] = depth.loc[m, pass_col] * factor
        # renormalize to 1
        s = depth[pass_col].sum()
        if s > 0:
            depth[pass_col] = depth[pass_col] / s

        for _, p in depth.iterrows():
            exp_rush_td = float(rush_tds) * float(p[rush_col])
            rec_weight = float(p[pass_col])
            # Passing TDs go to receivers (WR/TE/RB) catching the TD. Attribute by target shares; QB gets 0 receiving TDs.
            exp_rec_td = 0.0 if str(p.get("position")).upper() == "QB" else float(pass_tds) * rec_weight
            exp_any_td = exp_rush_td + exp_rec_td
            # Poisson approx any-time TD probability
            prob_any = float(1.0 - np.exp(-max(0.0, exp_any_td)))
            rows.append({
                "season": int(r.get("season")),
                "week": int(r.get("week")),
                "date": r.get("date"),
                "game_id": r.get("game_id"),
                "team": team,
                "opponent": opp,
                "player": p.get("player"),
                "position": p.get("position"),
                "is_home": int(r.get("is_home") or 0),
                "expected_td": exp_any_td,
                "anytime_td_prob": prob_any,
                "exp_rush_td": exp_rush_td,
                "exp_rec_td": exp_rec_td,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Try to enrich player names to include last names using nfl_data_py rosters
    out = _enrich_player_names(out, season)
    # Sort by game/team and descending probability
    out = out.sort_values(["season","week","game_id","team","anytime_td_prob"], ascending=[True, True, True, True, False])
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compute per-player anytime TD likelihoods from team TD expectations.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    df = compute_player_td_likelihood(args.season, args.week)
    if df is None or df.empty:
        print("No player TD results.")
        return
    out_fp = Path(args.out) if args.out else (DATA_DIR / f"player_td_likelihood_{args.season}_wk{args.week}.csv")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f"Wrote player anytime TD likelihoods to {out_fp}")
    try:
        print(df.groupby(["team","position"]).head(1)[["team","player","position","anytime_td_prob","expected_td"]].to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
