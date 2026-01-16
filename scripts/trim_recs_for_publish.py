import argparse
from pathlib import Path
import pandas as pd

TIERS = {"Low": 1, "Medium": 2, "High": 3}


def tier_num(val: str) -> int:
    return TIERS.get(str(val or '').strip().capitalize(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', type=Path, required=True)
    ap.add_argument('--out', dest='out', type=Path, required=True)
    ap.add_argument('--per-market', type=int, default=3)
    ap.add_argument('--min-conf', type=str, default='High')
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if df.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"No rows to trim. Wrote empty file: {args.out}")
        return

    df['conf_num'] = df['confidence'].apply(tier_num)
    conf_floor = tier_num(args.min_conf)
    df = df[df['conf_num'] >= conf_floor].copy()
    # Sort by confidence then EV desc
    df['ev_pct'] = pd.to_numeric(df.get('ev_pct'), errors='coerce')
    df = df.sort_values(['conf_num', 'ev_pct'], ascending=[False, False])

    out_rows = []
    for mkt in ['MONEYLINE', 'SPREAD', 'TOTAL']:
        sub = df[(df['type'].str.upper() == mkt)].copy()
        if sub.empty:
            continue
        take = sub.head(args.per_market)
        out_rows.append(take)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else df.head(0)
    # Keep preferred columns if present
    prefer = ['type', 'confidence', 'ev_pct', 'odds', 'side', 'market',
              'home_team', 'away_team', 'game_id', 'game_date', 'season', 'week', 'book', 'note']
    cols = [c for c in prefer if c in out.columns]
    if cols:
        out = out[cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Trimmed publish list: rows={len(out)}, out={args.out}")


if __name__ == '__main__':
    main()
