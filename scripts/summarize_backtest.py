import argparse
from pathlib import Path
import pandas as pd


def summarize(details_path: Path):
    df = pd.read_csv(details_path)
    def agg(sub):
        wins = int((sub['result'] == 'Win').sum())
        losses = int((sub['result'] == 'Loss').sum())
        pushes = int((sub['result'] == 'Push').sum()) if 'Push' in sub['result'].unique() else 0
        rows = len(sub)
        win_rate = wins / rows if rows else 0.0
        return {
            'rows': rows,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': round(win_rate, 3)
        }

    out = {}
    for market in ['MONEYLINE', 'SPREAD', 'TOTAL']:
        sub = df[df['type'] == market]
        out[market] = agg(sub)
    high_out = {}
    for market in ['MONEYLINE', 'SPREAD', 'TOTAL']:
        sub = df[(df['type'] == market) & (df['confidence'] == 'High')]
        high_out[market] = agg(sub)

    print('Overall:', out)
    print('High-only:', high_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--details', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.details)
