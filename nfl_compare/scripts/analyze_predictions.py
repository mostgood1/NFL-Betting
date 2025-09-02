import pandas as pd
import numpy as np
from pathlib import Path


def main():
    pred_path = Path(__file__).resolve().parents[1] / 'data' / 'predictions.csv'
    if not pred_path.exists():
        print(f'Predictions file not found: {pred_path}')
        print('Run predictions first: python -m nfl_compare.src.predict')
        return

    df = pd.read_csv(pred_path)
    cols = set(df.columns)

    if 'pred_margin' in cols:
        m = df['pred_margin']
    else:
        m = df.get('pred_home_score', pd.Series(0)) - df.get('pred_away_score', pd.Series(0))

    n = len(df)
    abs_m = m.abs()
    metrics = {
        'n_games': n,
        'mean_margin': float(m.mean()),
        'mean_abs_margin': float(abs_m.mean()),
        'median_abs_margin': float(abs_m.median()),
        'pct_within_3': float((abs_m <= 3).mean() * 100.0),
        'pct_within_7': float((abs_m <= 7).mean() * 100.0),
        'pct_within_10': float((abs_m <= 10).mean() * 100.0),
    }

    # Quarter sweep analysis
    qs = [f'pred_q{i}_winner' for i in range(1, 5)]
    if all(q in cols for q in qs):
        winners = df[qs].astype(str).values
        sweeps = three_one = two_two = ties_any = 0
        for row in winners:
            unique = set(row)
            if 'Tie' in unique:
                ties_any += 1
            uniq_no_tie = [w for w in row if w != 'Tie']
            if len(uniq_no_tie) == 0:
                two_two += 1
                continue
            vals, counts = np.unique(uniq_no_tie, return_counts=True)
            mx = int(counts.max())
            if mx == 4:
                sweeps += 1
            elif mx == 3:
                three_one += 1
            elif mx == 2 and len(vals) == 2:
                two_two += 1
        metrics.update({
            'pct_sweeps': sweeps * 100.0 / n,
            'pct_3_1': three_one * 100.0 / n,
            'pct_2_2': two_two * 100.0 / n,
            'pct_any_tie_label': ties_any * 100.0 / n,
        })

    # Quarter totals sanity
    qt = [f'pred_q{i}_total' for i in range(1, 5)]
    if all(q in cols for q in qt) and 'pred_total' in cols:
        tot = df['pred_total']
        sum_q = df[qt].sum(axis=1)
        metrics['mean_q_sum_err'] = float((sum_q - tot).abs().mean())

    # Print metrics
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


if __name__ == '__main__':
    main()
