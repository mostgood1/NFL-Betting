import pandas as pd
from pathlib import Path
from joblib import load
from .data_sources import load_games, load_team_stats, load_lines
from .features import merge_features
from .models import predict as model_predict
from .recommendations import make_recommendations

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'


def main():
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()

    if (MODELS_DIR / 'nfl_models.joblib').exists():
        models = load(MODELS_DIR / 'nfl_models.joblib')
    else:
        print('Model file not found. Run training first: python -m src.train')
        return

    # For upcoming week predictions, we expect lines without scores
    df = merge_features(games, team_stats, lines)
    df_future = df[df['home_score'].isna() | df['away_score'].isna()].copy()
    if df_future.empty:
        print('No future games found (rows without scores). Ensure lines.csv has upcoming matchups.')
        return

    df_pred = model_predict(models, df_future)
    df_rec = make_recommendations(df_pred)

    out_fp = Path(__file__).resolve().parents[1] / 'data' / 'predictions.csv'
    df_rec.to_csv(out_fp, index=False)
    print(f'Predictions written to {out_fp}')


if __name__ == '__main__':
    main()
