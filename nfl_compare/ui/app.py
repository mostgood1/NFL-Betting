import streamlit as st
import os
import pandas as pd
from pathlib import Path
import sys
from joblib import load
from datetime import date as _date, timedelta

# Ensure imports work in two ways:
# - `from src...` (by adding the package dir)
# - joblib unpickling referencing `nfl_compare.src...` (by adding the repo root)
PKG_ROOT = Path(__file__).resolve().parents[1]   # .../nfl_compare
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../NFL-Betting
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_sources import load_games, load_team_stats, load_lines
from src.features import merge_features
from src.models import predict as model_predict
from src.recommendations import make_recommendations
from src.team_assets import get_team_logo, get_team_colors
from src.weather import load_weather_for_games, WeatherCols
try:
    # When run as a package (python -m), relative import works
    from .components import inject_css, game_card
except Exception:
    # When run directly by Streamlit, fall back to absolute import after sys.path injection
    from ui.components import inject_css, game_card

st.set_page_config(page_title='NFL Compare', layout='wide')
_CSS_PATH = Path(__file__).resolve().parent / 'styles.css'
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'

# Lock tuned calibration parameters into the UI runtime (2025)
_TUNED_2025 = {
    'NFL_MARKET_MARGIN_BLEND': '0.20',
    'NFL_MARGIN_TARGET_STD': '8.0',
    'NFL_MARGIN_SHARPEN': '1.10',
    'NFL_MARGIN_MIN_SCALE': '0.80',
    'NFL_MARGIN_MAX_SCALE': '2.50',
    'NFL_Q_SHARE_BLEND': '0.60',
    'NFL_HALF_BLEND': '0.65',
    'NFL_ANTI_SWEEP_ENABLE': '1',
    'NFL_ANTI_SWEEP_CLOSE_THR': '1.4',
    'NFL_ANTI_SWEEP_BLOWOUT_THR': '3.0',
    'NFL_ANTI_SWEEP_BUFFER': '0.1',
}
for k, v in _TUNED_2025.items():
    try:
        os.environ[k] = str(v)
    except Exception:
        pass


def _infer_current_season_week(games: pd.DataFrame) -> tuple[int, int]:
    if games is None or games.empty or 'season' not in games.columns or 'week' not in games.columns:
        return 2025, 1
    today = _date.today()
    g = games.copy()
    try:
        g['date'] = pd.to_datetime(g.get('date'), errors='coerce').dt.date
    except Exception:
        g['date'] = pd.NaT
    seasons = sorted(pd.to_numeric(g['season'], errors='coerce').dropna().astype(int).unique())
    if not seasons:
        return 2025, 1
    def _season_score(s: int) -> int:
        sd = g[g['season']==s]['date']
        sd = sd[sd.notna()]
        if sd.empty:
            return 10**6
        try:
            return int(min(abs((d - today).days) for d in sd))
        except Exception:
            return 10**6
    season = min(seasons, key=_season_score)
    gs = g[g['season']==season].copy()
    if gs.empty:
        return season, 1
    wk = (
        gs.groupby('week')['date']
          .agg(['min','max'])
          .dropna()
          .reset_index()
          .rename(columns={'min':'start','max':'end'})
    )
    wk['start'] = pd.to_datetime(wk['start'], errors='coerce').dt.date
    wk['end'] = pd.to_datetime(wk['end'], errors='coerce').dt.date
    cur = wk[(wk['start'] - timedelta(days=1) <= today) & (today <= wk['end'] + timedelta(days=1))]
    if not cur.empty:
        week = int(cur['week'].iloc[0])
    else:
        upcoming = wk[wk['start'] >= today].sort_values('start')
        week = int(upcoming['week'].iloc[0]) if not upcoming.empty else int(wk['week'].max())
    return int(season), int(week)


def _fmt_num(x, decimals=1):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "-"


def _render_game_card(r: dict):
    home = r.get('home_team', '')
    away = r.get('away_team', '')
    home_logo = get_team_logo(home)
    away_logo = get_team_logo(away)
    _ = get_team_colors(home)

    # Styled card container
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="gc-header gc-teams">
            <div class="gc-team"><img src="{away_logo}" width="36"/><span class="name">{away}</span></div>
            <div style="text-align:center; color:#777; font-size:12px;">@</div>
            <div class="gc-team" style="justify-content:flex-end"><span class="name">{home}</span><img src="{home_logo}" width="36"/></div>
        </div>''',
        unsafe_allow_html=True
    )

    # Market and predictions / finals
    spread = r.get('spread_home')
    total = r.get('total')
    prob = r.get('prob_home_win')
    phs = r.get('pred_home_score')
    pas = r.get('pred_away_score')
    ptotal = r.get('pred_total')
    h1 = r.get('pred_1h_total')
    h2 = r.get('pred_2h_total')
    # Team half scores
    a1 = r.get('pred_away_1h'); h1h = r.get('pred_home_1h')
    a2 = r.get('pred_away_2h'); h2h = r.get('pred_home_2h')
    units_spread = r.get('units_spread')
    units_total = r.get('units_total')
    # Actual finals
    ah = r.get('home_score')
    aa = r.get('away_score')
    is_final = (ah is not None and not pd.isna(ah)) and (aa is not None and not pd.isna(aa))
    # Weather
    temp = r.get(WeatherCols.temp_f, None)
    wind = r.get(WeatherCols.wind_mph, None)
    precip = r.get(WeatherCols.precip_pct, None)
    roof = r.get(WeatherCols.roof, None)

    # Market chips
    chips = []
    if spread is not None and not pd.isna(spread):
        chips.append(f"<span class='chip'>Spread {spread:+}</span>")
    if total is not None and not pd.isna(total) and float(total) != 0.0:
        chips.append(f"<span class='chip'>Total {float(total):.1f}</span>")
    if prob is not None and not pd.isna(prob):
        chips.append(f"<span class='chip'>Home WP {float(prob):.2f}</span>")
    if is_final:
        chips.append("<span class='chip edge-pos'>FINAL</span>")
    if chips:
        st.markdown(f"<div class='gc-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)
        # Score block: show finals if present, else predictions
        if is_final:
            try:
                atot = (float(aa) if aa is not None else 0) + (float(ah) if ah is not None else 0)
            except Exception:
                atot = None
            st.markdown(
                f"""
                <div class='gc-pred'>
                    <div class='score'>{_fmt_num(aa)}</div>
                    <div style='color:#666'>{'Final Total ' + _fmt_num(atot) if atot is not None else 'Final'}</div>
                    <div class='score'>{_fmt_num(ah)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif phs is not None and pas is not None and ptotal is not None:
            st.markdown(
                f"""
                <div class='gc-pred'>
                    <div class='score'>{_fmt_num(pas)}</div>
                    <div style='color:#666'>Projected Total {_fmt_num(ptotal)}</div>
                    <div class='score'>{_fmt_num(phs)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    # Show 1H/2H projected scores (away-home) and totals (skip if final)
    if (not is_final) and ((a1 is not None and h1h is not None) or (a2 is not None and h2h is not None)):
        st.markdown(
            f"<div class='gc-meta'>"
            f"1H { _fmt_num(a1) } - { _fmt_num(h1h) } (tot {_fmt_num(h1)}) • "
            f"2H { _fmt_num(a2) } - { _fmt_num(h2h) } (tot {_fmt_num(h2)})"
            f"</div>",
            unsafe_allow_html=True
        )

    # Weather chip
    wx_bits = []
    if roof and str(roof).strip():
        wx_bits.append(str(roof))
    if temp is not None and not pd.isna(temp):
        wx_bits.append(f"{float(temp):.0f}F")
    if wind is not None and not pd.isna(wind):
        wx_bits.append(f"Wind {float(wind):.0f}mph")
    if precip is not None and not pd.isna(precip):
        wx_bits.append(f"Precip {float(precip):.0f}%")
    if wx_bits:
        st.markdown(f"<div class='gc-meta'>{' | '.join(wx_bits)}</div>", unsafe_allow_html=True)

    # Quick picks summary (skip if final)
    if not is_final:
        recs = []
        if isinstance(units_spread, (int, float)) and not pd.isna(units_spread) and abs(units_spread) >= 0.5:
            side = f"{home} {spread:+}" if spread is not None and not pd.isna(spread) else f"{home} spread"
            recs.append(f"Spread: {side} ({_fmt_num(units_spread, 1)}u)")
        if isinstance(units_total, (int, float)) and not pd.isna(units_total) and abs(units_total) >= 0.5:
            if ptotal is not None and total is not None and not pd.isna(ptotal) and not pd.isna(total):
                o_u = 'Over' if float(ptotal) > float(total) else 'Under'
                recs.append(f"Total: {o_u} {total} ({_fmt_num(units_total, 1)}u)")
        if recs:
            cls = 'edge-pos' if ('Over' in ' '.join(recs) or '+' in ' '.join(recs)) else 'edge-neg'
            st.markdown(f"<div class='gc-chips'><span class='chip {cls}'>{' • '.join(recs)}</span></div>", unsafe_allow_html=True)

    # Probability bar (skip if final)
    if not is_final:
        try:
            p = float(prob) if prob is not None and not pd.isna(prob) else 0.5
            pct = int(round(p * 100))
            st.markdown(
                f"<div class='probbar'><div class='home' style='width:{pct}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='gc-meta'>Home win prob: {pct}%</div>", unsafe_allow_html=True)
        except Exception:
            pass

    # Pop-out detail (expander)
    with st.expander('Details', expanded=False):
        _render_game_detail(pd.Series(r))

    st.markdown('</div>', unsafe_allow_html=True)


def _render_game_detail(r: pd.Series):
    # Render the same styled game card inside the modal for a consistent look
    try:
        game_card(dict(r))
    except Exception:
        # Fallback minimal header if card rendering fails
        home = r.get('home_team', '')
        away = r.get('away_team', '')
        st.markdown(f"### {away} @ {home}")

    # Add quarter-by-quarter chips with team scores (away-home), total, and winner
    chips = []
    for i in [1, 2, 3, 4]:
        tot_key = f'pred_q{i}_total'
        win_key = f'pred_q{i}_winner'
        h_key = f'pred_home_q{i}'
        a_key = f'pred_away_q{i}'
        tot_val = r.get(tot_key, None)
        win_val = r.get(win_key, None)
        h_val = r.get(h_key, None)
        a_val = r.get(a_key, None)
        if tot_val is not None and not pd.isna(tot_val):
            score_txt = f"{_fmt_num(a_val)} - {_fmt_num(h_val)}"
            win_txt = f" • W: {win_val}" if (win_val is not None and not pd.isna(win_val)) else ""
            chips.append(
                f"<span class='chip'><span class='dot dot-blue'></span>Q{i} {score_txt} (tot {float(tot_val):.1f}){win_txt}</span>"
            )
    if chips:
        st.markdown(f"<div class='gc-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_data():
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()
    # Attach weather rows for games (non-destructive; NaNs if no file)
    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = pd.DataFrame()
    return games, team_stats, lines, wx

@st.cache_data(ttl=30)
def predict_df():
    games, team_stats, lines, wx = load_data()
    df = merge_features(games, team_stats, lines, wx)
    # If no model file, just display merged data (will include finals and future)
    if not (MODELS_DIR / 'nfl_models.joblib').exists():
        if not df.empty and not (wx is None or wx.empty):
            df = df.merge(wx, on=['game_id','date','home_team','away_team'], how='left')
        return df, None

    models = load(MODELS_DIR / 'nfl_models.joblib')
    # Split into completed and future games
    df_completed = df[df['home_score'].notna() & df['away_score'].notna()].copy()
    df_future = df[df['home_score'].isna() | df['away_score'].isna()].copy()

    out_frames = []
    if not df_completed.empty:
        # Attach weather for UI parity
        if not (wx is None or wx.empty):
            df_completed = df_completed.merge(wx, on=['game_id','date','home_team','away_team'], how='left')
        out_frames.append(df_completed)

    if not df_future.empty:
        df_pred = model_predict(models, df_future)
        if not (wx is None or wx.empty):
            df_pred = df_pred.merge(wx, on=['game_id','date','home_team','away_team'], how='left')
        df_rec = make_recommendations(df_pred)
        out_frames.append(df_rec)

    if out_frames:
        return pd.concat(out_frames, ignore_index=True), models
    else:
        return df, models

st.title('NFL Compare — Week Picks & Game Models')
st.markdown("<div class='gc-chips'><span class='chip'>Calibration: 2025 tuned</span></div>", unsafe_allow_html=True)
inject_css()

with st.sidebar:
    st.header('Filters')
    # Determine defaults from data
    try:
        games, _, _, _ = load_data()
    except Exception:
        games = pd.DataFrame()
    d_season, d_week = _infer_current_season_week(games)
    season = st.number_input('Season', min_value=2000, max_value=2100, value=int(d_season))
    week = st.number_input('Week', min_value=1, max_value=22, value=int(d_week))
    st.markdown('Upload or drop CSVs in ./data: games.csv, team_stats.csv, lines.csv')
    if st.button('Refresh data & predictions'):
        # Clear all cached data so file changes are picked up immediately
        st.cache_data.clear()
        st.rerun()


df_pred, models = predict_df()

if models is None:
    st.warning('Model not trained yet or no future games found. Train with: python -m src.train')

if isinstance(df_pred, pd.DataFrame) and not df_pred.empty:
    filt = (df_pred['season'] == season) & (df_pred['week'] == week)
    view = df_pred[filt] if {'season','week'}.issubset(df_pred.columns) else df_pred

    tabs = st.tabs(["Cards", "Table"])

    with tabs[0]:
        st.subheader('This Week — Game Cards')
        if not view.empty:
            # Sort by kickoff date if available, else by probability
            if 'date' in view.columns:
                view_cards = view.sort_values('date')
            elif 'prob_home_win' in view.columns:
                view_cards = view.sort_values('prob_home_win', ascending=False)
            else:
                view_cards = view

            rows = view_cards.to_dict(orient='records')
            ncols = 2
            cols = st.columns(ncols, gap="medium")
            for i, r in enumerate(rows):
                with cols[i % ncols]:
                    game_card(r)

    with tabs[1]:
        st.subheader('Predictions — Table')
        if not view.empty:
            show_cols = [
                'date','away_team','home_team','home_score','away_score','spread_home','total','moneyline_home','moneyline_away',
                'prob_home_win','pred_away_score','pred_home_score','pred_total',
                'pred_1h_total','pred_2h_total','pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total',
                'pred_q1_winner','pred_q2_winner','pred_q3_winner','pred_q4_winner',
                'pred_1h_winner','pred_2h_winner','units_spread','units_total'
            ]
            tbl = view[[c for c in show_cols if c in view.columns]].copy()
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.download_button('Download CSV', data=tbl.to_csv(index=False), file_name='nfl_predictions.csv', mime='text/csv')
else:
    st.info('No prediction data to display yet.')
