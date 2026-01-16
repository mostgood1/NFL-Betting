import argparse
import pandas as pd


def parse_selection(row):
    t = str(row.get("type", "")).upper()
    sel = str(row.get("selection", ""))
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    return t, sel, home, away


def prob_selected_ml(row):
    p_home = pd.to_numeric(row.get("prob_home_win_mc"), errors="coerce")
    if pd.isna(p_home):
        return None
    _, sel, home, away = parse_selection(row)
    # Selection contains team name and "ML"
    if home and home in sel:
        return float(p_home)
    if away and away in sel:
        return float(1.0 - float(p_home))
    return None


def prob_selected_total(row):
    p_over = pd.to_numeric(row.get("prob_over_total_mc"), errors="coerce")
    if pd.isna(p_over):
        return None
    t, sel, _, _ = parse_selection(row)
    if not t.startswith("TOTAL"):
        return None
    is_over = sel.strip().lower().startswith("over")
    return float(p_over) if is_over else float(1.0 - float(p_over))


def compute_accuracy(df, market, p_thresh):
    if market == "MONEYLINE":
        probs = df.apply(prob_selected_ml, axis=1)
        mask = probs.apply(lambda x: x is not None and x >= p_thresh)
    else:
        probs = df.apply(prob_selected_total, axis=1)
        mask = probs.apply(lambda x: x is not None and x >= p_thresh)
    sub = df[mask].copy()
    if sub.empty:
        return {"rows": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": 0.0}
    wins = int((sub["result"].str.upper() == "WIN").sum())
    losses = int((sub["result"].str.upper() == "LOSS").sum())
    pushes = int((sub["result"].str.upper() == "PUSH").sum()) if "PUSH" in sub["result"].unique() else 0
    win_rate = float(wins / max(1, (wins + losses)))
    return {"rows": int(len(sub)), "wins": wins, "losses": losses, "pushes": pushes, "win_rate": win_rate}


def main():
    ap = argparse.ArgumentParser(description="Sweep MC probability thresholds to target accuracy bands")
    ap.add_argument("--details", type=str, required=True, help="Path to recs_backtest_details.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.details)
    ml_thresholds = [0.60, 0.62, 0.64, 0.66, 0.68]
    total_thresholds = [0.58, 0.60, 0.62, 0.64]

    out = {"MONEYLINE": {}, "TOTAL": {}}
    for th in ml_thresholds:
        out["MONEYLINE"][th] = compute_accuracy(df[df["type"].str.upper() == "MONEYLINE"], "MONEYLINE", th)
    for th in total_thresholds:
        out["TOTAL"][th] = compute_accuracy(df[df["type"].str.upper().str.startswith("TOTAL")], "TOTAL", th)

    print(out)


if __name__ == "__main__":
    main()
