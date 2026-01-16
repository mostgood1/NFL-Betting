import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfl_compare.src.data_sources import load_games  # type: ignore
from nfl_compare.src.team_ratings import compute_team_ratings_from_games  # type: ignore

try:
    from nfl_compare.src.player_props import _load_weekly_depth_chart  # type: ignore
except Exception:
    _load_weekly_depth_chart = None  # type: ignore

DATA_DIR = ROOT / 'nfl_compare' / 'data'


def _safe_num(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def build_stats(season: int, week: int) -> pd.DataFrame:
    games = load_games()
    if games is None or games.empty:
        return pd.DataFrame(columns=['season','week','team','off_epa','def_epa','pace_secs_play','pass_rate','rush_rate','qb_adj','sos'])

    # Normalize types
    for c in ('season','week'):
        if c in games.columns:
            games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games[games['season'].eq(int(season))].copy()
    # Ratings up to current week
    R = compute_team_ratings_from_games(games, int(season), int(week))

    # Prepare per-team per-week rows (for target week)
    # Teams appearing in any game for the target week
    wk = games[pd.to_numeric(games['week'], errors='coerce').eq(int(week))]
    teams = pd.unique(pd.concat([wk['home_team'], wk['away_team']], ignore_index=True))
    rows = []

    # Optional team context EMA file for pass rate / pace if present
    ema_path = DATA_DIR / f'team_context_ema_{season}_wk{week}.csv'
    ema = None
    if ema_path.exists():
        try:
            ema = pd.read_csv(ema_path)
        except Exception:
            ema = None
    # Depth chart for QB adjust
    dc = None
    if _load_weekly_depth_chart is not None:
        try:
            dc = _load_weekly_depth_chart(int(season), int(week))
        except Exception:
            dc = None
    if dc is not None and not dc.empty:
        # starter active per team at QB
        dc['pos_up'] = dc.get('position', '').astype(str).str.upper()
        qb = dc[dc['pos_up'] == 'QB'].copy()
        qb = qb.sort_values(['team', 'depth_rank', 'player']) if 'depth_rank' in qb.columns else qb
        qb_main = qb.groupby('team', as_index=False).first()[['team','active']]
        qb_main.rename(columns={'active':'qb_starter_active'}, inplace=True)

    for team in teams:
        team = str(team)
        # Defaults
        off_epa = 0.0
        def_epa = 0.0
        pace_secs_play = 26.5  # conservative baseline
        pass_rate = 0.57
        rush_rate = 1.0 - pass_rate
        qb_adj = 0.0
        sos = 0.0

        # Use EMA context if available
        if ema is not None and not ema.empty and 'team' in ema.columns:
            e = ema[ema['team'].astype(str) == team]
            if not e.empty:
                # Try common fields
                pr = _safe_num(e.iloc[0].get('pass_rate'))
                if pr is None:
                    pr = _safe_num(e.iloc[0].get('pass_rate_ema'))
                if pr is not None:
                    pass_rate = max(0.0, min(1.0, pr))
                    rush_rate = 1.0 - pass_rate
                # pace approximations
                pace = _safe_num(e.iloc[0].get('pace_secs_play'))
                if pace is None:
                    plays_pg = _safe_num(e.iloc[0].get('plays_per_game'))
                    # If plays per game present, convert to secs/play ~ (60*60)/(plays_pg*~60mins) ~ 3600/(plays_pg*3600/60)
                    if plays_pg and plays_pg > 0:
                        # Rough approximation: 60 minutes = 3600 seconds; secs/play ~ 3600/plays
                        pace = 3600.0 / plays_pg
                if pace is not None:
                    pace_secs_play = pace

        # QB adjust from depth chart
        try:
            if dc is not None and not dc.empty:
                row = qb_main[qb_main['team'].astype(str) == team]
                if not row.empty:
                    active = bool(row.iloc[0]['qb_starter_active'])
                    qb_adj = 0.0 if active else -1.0
        except Exception:
            pass

        # Strength of schedule: average opponent net_margin prior to this week
        try:
            if R is not None and not R.empty:
                # Determine opponents up to prev week
                g_team = games[(games['home_team'].astype(str) == team) | (games['away_team'].astype(str) == team)].copy()
                g_team['w'] = pd.to_numeric(g_team['week'], errors='coerce')
                prev = g_team[g_team['w'] < int(week)]
                if not prev.empty:
                    opps = []
                    for _, gr in prev.iterrows():
                        home = str(gr.get('home_team'))
                        away = str(gr.get('away_team'))
                        opp = away if home == team else home
                        opps.append((int(season), int(gr['week']), opp))
                    # For each opponent, take their net_margin rating at that week (from R)
                    vals = []
                    for s, w, opp in opps:
                        r_opp = R[(R['season'] == s) & (R['week'] == int(w)) & (R['team'].astype(str) == opp)]
                        if not r_opp.empty:
                            nm = _safe_num(r_opp.iloc[0].get('net_margin'))
                            if nm is not None:
                                vals.append(nm)
                    sos = float(np.mean(vals)) if vals else 0.0
        except Exception:
            sos = 0.0

        rows.append({
            'season': int(season), 'week': int(week), 'team': team,
            'off_epa': float(off_epa), 'def_epa': float(def_epa),
            'pace_secs_play': float(pace_secs_play), 'pass_rate': float(pass_rate), 'rush_rate': float(rush_rate),
            'qb_adj': float(qb_adj), 'sos': float(sos)
        })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='Build team_stats.csv for a given season/week')
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--week', type=int, required=True)
    args = p.parse_args()

    df = build_stats(args.season, args.week)
    out = DATA_DIR / 'team_stats.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    # If the file exists, update/replace rows for (season, week) teams
    if out.exists():
        try:
            cur = pd.read_csv(out)
        except Exception:
            cur = pd.DataFrame()
        if cur is None or cur.empty:
            cur = df
        else:
            cur['season'] = pd.to_numeric(cur.get('season'), errors='coerce')
            cur['week'] = pd.to_numeric(cur.get('week'), errors='coerce')
            mask = ~((cur['season'] == int(args.season)) & (cur['week'] == int(args.week)))
            cur = pd.concat([cur[mask], df], ignore_index=True)
        cur.to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows to {out}')


if __name__ == '__main__':
    main()
