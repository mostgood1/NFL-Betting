import argparse
import pandas as pd


def prob_selected_ml(row):
    p_home = pd.to_numeric(row.get("prob_home_win_mc"), errors="coerce")
    sel = str(row.get("selection", ""))
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    if pd.isna(p_home):
        return None
    if home and home in sel:
        return float(p_home)
    if away and away in sel:
        return float(1.0 - float(p_home))
    return None


def prob_selected_total(row):
    p_over = pd.to_numeric(row.get("prob_over_total_mc"), errors="coerce")
    sel = str(row.get("selection", ""))
    if pd.isna(p_over):
        return None
    is_over = sel.strip().lower().startswith("over")
    return float(p_over) if is_over else float(1.0 - float(p_over))


def main():
    ap = argparse.ArgumentParser(description="Preview publishable picks under MC probability thresholds")
    ap.add_argument("--details", type=str, required=True)
    ap.add_argument("--p-ml", type=float, default=0.64)
    ap.add_argument("--p-total", type=float, default=0.60)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.details)
    if args.season is not None:
        df = df[df["season"] == int(args.season)]
    if args.week is not None:
        df = df[df["week"] == int(args.week)]
    df["p_sel_ml_mc"] = df.apply(lambda r: prob_selected_ml(r) if str(r.get("type", "")).upper() == "MONEYLINE" else None, axis=1)
    df["p_sel_total_mc"] = df.apply(lambda r: prob_selected_total(r) if str(r.get("type", "")).upper().startswith("TOTAL") else None, axis=1)

    ml_pub = df[(df["type"].str.upper() == "MONEYLINE") & (df["p_sel_ml_mc"].notna()) & (df["p_sel_ml_mc"] >= args.p_ml)]
    total_pub = df[(df["type"].str.upper().str.startswith("TOTAL")) & (df["p_sel_total_mc"].notna()) & (df["p_sel_total_mc"] >= args.p_total)]

    print({
        "MONEYLINE": ml_pub[["selection", "odds", "p_sel_ml_mc", "confidence", "home_team", "away_team", "season", "week", "game_id"]].to_dict(orient="records"),
        "TOTAL": total_pub[["selection", "odds", "p_sel_total_mc", "confidence", "home_team", "away_team", "season", "week", "game_id"]].to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
