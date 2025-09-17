import pandas as pd
from pathlib import Path


def main():
    # Locate Week 1 props cache
    data_dir = Path(__file__).resolve().parents[1] / "nfl_compare" / "data"
    f = data_dir / "player_props_2025_wk1.csv"
    if not f.exists():
        print(f"Missing file: {f}")
        return

    df = pd.read_csv(f)

    # Keep only active players for player sums to avoid duplicates/inactive noise
    df_players = df[df["is_active"].fillna(0) == 1.0].copy()

    # Compute per-team sums and reference team totals from the team rows (they repeat per player; use first)
    team_cols = [
        "team_pass_yards",
        "team_rush_yards",
        "team_pass_attempts",
        "team_rush_attempts",
        "team_exp_pass_tds",
        "team_exp_rush_tds",
    ]

    ref = (
        df_players.groupby(["game_id", "team"], as_index=False)[team_cols]
        .first()
    )

    sums = (
        df_players.groupby(["game_id", "team"], as_index=False)
        .agg(
            rec_yards_sum=("rec_yards", "sum"),
            rush_yards_sum=("rush_yards", "sum"),
            targets_sum=("targets", "sum"),
            receptions_sum=("receptions", "sum"),
            rush_attempts_sum=("rush_attempts", "sum"),
            rec_tds_sum=("rec_tds", "sum"),
            rush_tds_sum=("rush_tds", "sum"),
        )
    )

    out = ref.merge(sums, on=["game_id", "team"], how="left")

    # Ratios; guard against divide-by-zero
    def ratio(a, b):
        a = a.fillna(0.0)
        b = b.fillna(0.0)
        return (a / b.replace(0, pd.NA)).astype(float)

    out["ratio_rec_yards"] = ratio(out["rec_yards_sum"], out["team_pass_yards"])  # expect ~1.0
    out["ratio_rush_yards"] = ratio(out["rush_yards_sum"], out["team_rush_yards"])  # expect ~1.0
    out["ratio_targets_attempts"] = ratio(out["targets_sum"], out["team_pass_attempts"])  # expect ~1.0
    out["ratio_rush_att"] = ratio(out["rush_attempts_sum"], out["team_rush_attempts"])  # expect ~1.0
    out["ratio_rec_tds"] = ratio(out["rec_tds_sum"], out["team_exp_pass_tds"])  # expect ~1.0 (pass TDs -> receiving TDs)
    out["ratio_rush_tds"] = ratio(out["rush_tds_sum"], out["team_exp_rush_tds"])  # expect ~1.0

    def summarize(col):
        s = out[col].dropna()
        if s.empty:
            return "n/a"
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p10": float(s.quantile(0.10)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "max": float(s.max()),
        }

    print("Week 1 team-to-player correlation ratios (target ~1.0):")
    for col in [
        "ratio_rec_yards",
        "ratio_rush_yards",
        "ratio_targets_attempts",
        "ratio_rush_att",
        "ratio_rec_tds",
        "ratio_rush_tds",
    ]:
        print(f"- {col}: {summarize(col)}")

    # Show worst offenders (absolute deviation from 1.0)
    def worst(col, k=8):
        s = out[["game_id", "team", col]].dropna().copy()
        s["dev"] = (s[col] - 1.0).abs()
        return s.sort_values("dev", ascending=False).head(k)

    print("\nWorst deviations (top 8) per metric:")
    for col in [
        "ratio_rec_yards",
        "ratio_rush_yards",
        "ratio_targets_attempts",
        "ratio_rush_att",
        "ratio_rec_tds",
        "ratio_rush_tds",
    ]:
        print(f"\n{col}:")
        print(worst(col).to_string(index=False))


if __name__ == "__main__":
    main()
