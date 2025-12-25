import argparse
from pathlib import Path
import pandas as pd

TIERS = {"Low": 1, "Medium": 2, "High": 3}

def tier_num(val: str) -> int:
    return TIERS.get(str(val or '').strip().capitalize(), 1)


def compute_win_rate(sub: pd.DataFrame) -> tuple[int, float]:
    rows = len(sub)
    if rows == 0:
        return 0, 0.0
    wins = int((sub['result'] == 'Win').sum())
    return rows, wins / rows


def sweep(details_path: Path, min_conf_ats: str, min_conf_total: str,
          ats_start: float, ats_stop: float, ats_step: float,
          total_start: float, total_stop: float, total_step: float,
          min_picks: int) -> None:
    df = pd.read_csv(details_path)
    # Normalize confidence
    df['confidence'] = df['confidence'].fillna('')
    df['conf_num'] = df['confidence'].apply(tier_num)
    # Prepare subsets
    spreads = df[df['type'].str.upper() == 'SPREAD'].copy()
    totals = df[df['type'].str.upper() == 'TOTAL'].copy()

    conf_floor_ats = tier_num(min_conf_ats)
    conf_floor_total = tier_num(min_conf_total)

    best = {
        'ATS': {'threshold': None, 'rows': 0, 'win_rate': 0.0},
        'TOTAL': {'threshold': None, 'rows': 0, 'win_rate': 0.0}
    }

    print(f"Sweep: ATS {ats_start}-{ats_stop} step {ats_step}, TOTAL {total_start}-{total_stop} step {total_step}; conf floors ATS={min_conf_ats}, TOTAL={min_conf_total}")
    print("ATS thresholds:")
    t = ats_start
    while t <= ats_stop + 1e-9:
        sub = spreads[(spreads['conf_num'] >= conf_floor_ats) & (spreads['ev_pct'] >= t)]
        rows, wr = compute_win_rate(sub)
        if rows >= min_picks and wr >= best['ATS']['win_rate']:
            best['ATS'] = {'threshold': round(t, 2), 'rows': rows, 'win_rate': round(wr, 3)}
        print(f"  ATS EV%>={t:.2f}: rows={rows}, win_rate={wr:.3f}")
        t += ats_step

    print("TOTAL thresholds:")
    t = total_start
    while t <= total_stop + 1e-9:
        sub = totals[(totals['conf_num'] >= conf_floor_total) & (totals['ev_pct'] >= t)]
        rows, wr = compute_win_rate(sub)
        if rows >= min_picks and wr >= best['TOTAL']['win_rate']:
            best['TOTAL'] = {'threshold': round(t, 2), 'rows': rows, 'win_rate': round(wr, 3)}
        print(f"  TOTAL EV%>={t:.2f}: rows={rows}, win_rate={wr:.3f}")
        t += total_step

    print("Best thresholds meeting min_picks:")
    print(f"  ATS: {best['ATS']}")
    print(f"  TOTAL: {best['TOTAL']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', type=Path, required=True)
    ap.add_argument('--min-conf-ats', type=str, default='High')
    ap.add_argument('--min-conf-total', type=str, default='High')
    ap.add_argument('--ats-ev-start', type=float, default=18.0)
    ap.add_argument('--ats-ev-stop', type=float, default=28.0)
    ap.add_argument('--ats-ev-step', type=float, default=2.0)
    ap.add_argument('--total-ev-start', type=float, default=20.0)
    ap.add_argument('--total-ev-stop', type=float, default=30.0)
    ap.add_argument('--total-ev-step', type=float, default=2.0)
    ap.add_argument('--min-picks', type=int, default=10)
    args = ap.parse_args()

    sweep(args.details, args.min_conf_ats, args.min_conf_total,
          args.ats_ev_start, args.ats_ev_stop, args.ats_ev_step,
          args.total_ev_start, args.total_ev_stop, args.total_ev_step,
          args.min_picks)
