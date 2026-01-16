import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Ensure project root on path to allow `import app`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int)
    ap.add_argument('--week', type=int)
    ap.add_argument('--min-ev', type=float, default=8.0)
    ap.add_argument('--one-per-game', type=str, default='false')
    ap.add_argument('--allowed-markets', type=str, default='MONEYLINE,SPREAD,TOTAL')
    ap.add_argument('--min-ats-delta', type=float, default=0.18)
    ap.add_argument('--min-total-delta', type=float, default=0.18)
    ap.add_argument('--min-ev-ml', type=float, default=8.0)
    ap.add_argument('--min-ev-ats', type=float, default=12.0)
    ap.add_argument('--min-ev-total', type=float, default=12.0)
    ap.add_argument('--conf-ats', type=str, default='High')
    ap.add_argument('--conf-total', type=str, default='High')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    # Configure gating via environment variables consumed by app
    os.environ['RECS_ALLOWED_MARKETS'] = str(args.allowed_markets)
    os.environ['RECS_ONE_PER_MARKET'] = 'true'
    os.environ['RECS_ONE_PER_GAME'] = str(args.one_per_game)
    os.environ['RECS_MIN_WP_DELTA'] = '0.12'
    os.environ['RECS_MIN_EV_PCT_ML'] = str(args.min_ev_ml)
    os.environ['RECS_MIN_ATS_DELTA'] = str(args.min_ats_delta)
    os.environ['RECS_MIN_EV_PCT_ATS'] = str(args.min_ev_ats)
    os.environ['RECS_MIN_TOTAL_DELTA'] = str(args.min_total_delta)
    os.environ['RECS_MIN_EV_PCT_TOTAL'] = str(args.min_ev_total)
    os.environ['RECS_UPCOMING_CONF_MIN_ATS'] = str(args.conf_ats)
    os.environ['RECS_UPCOMING_CONF_MIN_TOTAL'] = str(args.conf_total)

    # Build request path with optional season/week
    qs = []
    if args.season is not None:
        qs.append(f"season={args.season}")
    if args.week is not None:
        qs.append(f"week={args.week}")
    # Pass min_ev via query to ensure API uses it
    qs.append(f"min_ev={args.min_ev}")
    qs.append(f"one_per_game={args.one_per_game}")
    path = "/api/recommendations"
    if qs:
        path += "?" + "&".join(qs)

    with app.test_client() as c:
        r = c.get(path)
        if r.status_code != 200:
            raise SystemExit(f"API call failed: status={r.status_code}")
        js = r.get_json() or {}
        data = js.get('data') or []
        df = pd.DataFrame(data)
        # Ensure columns are stable and ordered
        prefer = [
            'type', 'confidence', 'ev_pct', 'odds', 'side', 'market',
            'home_team', 'away_team', 'game_id', 'game_date',
            'season', 'week', 'book', 'note'
        ]
        cols = [c for c in prefer if c in df.columns]
        if cols:
            df = df[cols]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote upcoming recommendations: rows={len(df)}, out={args.out}")


if __name__ == '__main__':
    main()
