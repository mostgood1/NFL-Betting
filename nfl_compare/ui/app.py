import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load
from src.data_sources import load_games, load_team_stats, load_lines
from src.features import merge_features
from src.models import predict as model_predict
from src.recommendations import make_recommendations

st.set_page_config(page_title='NFL Compare', layout='wide')

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'

@st.cache_data
def load_data():
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    return games, team_stats, lines

@st.cache_data
def predict_df():
    games, team_stats, lines = load_data()
    df = merge_features(games, team_stats, lines)
    # If no model file, just display merged data
    if not (MODELS_DIR / 'nfl_models.joblib').exists():
        return df, None
    models = load(MODELS_DIR / 'nfl_models.joblib')

    # Future games: rows without scores
    df_future = df[df['home_score'].isna() | df['away_score'].isna()].copy()
    if df_future.empty:
        return df, None

    df_pred = model_predict(models, df_future)
    df_rec = make_recommendations(df_pred)
    return df_rec, models

st.title('NFL Predictions & Betting Recommendations')

with st.sidebar:
    st.header('Filters')
    season = st.number_input('Season', min_value=2000, max_value=2100, value=2025)
    week = st.number_input('Week', min_value=1, max_value=22, value=1)
    st.markdown('Upload or drop CSVs in ./data: games.csv, team_stats.csv, lines.csv')


df_pred, models = predict_df()

if models is None:
    st.warning('Model not trained yet or no future games found. Train with: python -m src.train')

if isinstance(df_pred, pd.DataFrame) and not df_pred.empty:
    filt = (df_pred['season'] == season) & (df_pred['week'] == week)
    view = df_pred[filt] if 'season' in df_pred and 'week' in df_pred else df_pred

    st.subheader('Matchups')
    cols = ['season','week','home_team','away_team','spread_home','total','pred_home_score','pred_away_score','pred_total','pred_margin','prob_home_win','units_spread','units_total','units_home_ml','units_away_ml']
    cols = [c for c in cols if c in view.columns]
    st.dataframe(view[cols].sort_values('prob_home_win', ascending=False), use_container_width=True)

    st.subheader('Quarter/Half Projections')
    qcols = [
        'home_team','away_team','pred_home_q1','pred_home_q2','pred_home_q3','pred_home_q4','pred_away_q1','pred_away_q2','pred_away_q3','pred_away_q4','pred_home_1h','pred_home_2h','pred_away_1h','pred_away_2h'
    ]
    qcols = [c for c in qcols if c in view.columns]
    if qcols:
        st.dataframe(view[qcols], use_container_width=True)

    st.subheader('Raw Predictions CSV')
    st.download_button('Download CSV', data=view.to_csv(index=False), file_name='nfl_predictions.csv', mime='text/csv')
else:
    st.info('No prediction data to display yet.')
