from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Dict, Any

PKG_ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = PKG_ROOT / 'ui' / 'styles.css'


def inject_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _fmt(x, d=1, dash='–') -> str:
    try:
        if x is None:
            return dash
        return f"{float(x):.{d}f}"
    except Exception:
        return dash


def game_card(r: Dict[str, Any]):
    # Lazy import assets (path already injected by app.py)
    try:
        from src.team_assets import get_team_logo, get_team_colors
    except Exception:
        get_team_logo = lambda team: ""
        get_team_colors = lambda team: {"primary": "#1f2937", "text": "#ffffff"}

    away, home = r.get('away_team', ''), r.get('home_team', '')
    date_str = r.get('date', '') or ''
    # Markets (try multiple possible keys)
    spread = r.get('spread_home', r.get('home_spread'))
    total = r.get('total', r.get('game_total'))
    mlh = r.get('moneyline_home', r.get('ml_home'))
    mla = r.get('moneyline_away', r.get('ml_away'))
    prob = r.get('prob_home_win')
    phs = r.get('pred_home_score')
    pas = r.get('pred_away_score')
    ptotal = r.get('pred_total')
    h1 = r.get('pred_1h_total')
    h2 = r.get('pred_2h_total')
    # Team half scores
    a1 = r.get('pred_away_1h'); h1h = r.get('pred_home_1h')
    a2 = r.get('pred_away_2h'); h2h = r.get('pred_home_2h')
    temp = r.get('wx_temp_f') or r.get('temp_f')
    wind = r.get('wx_wind_mph') or r.get('wind_mph')
    precip = r.get('wx_precip_pct') or r.get('precip_pct')
    roof = r.get('wx_roof') or r.get('roof')
    units_spread = r.get('units_spread')
    units_total = r.get('units_total')

    # Extract team colors safely
    _colors = get_team_colors(home) if home else {"primary": "#1f2937", "text": "#ffffff"}
    if isinstance(_colors, dict):
        home_color = _colors.get("primary", "#1f2937")
        text_on_home = _colors.get("text", "#ffffff")
    else:
        # Backward compatibility if a tuple is ever returned
        try:
            home_color, text_on_home = _colors  # type: ignore[misc]
        except Exception:
            home_color, text_on_home = "#1f2937", "#ffffff"

    away_logo = get_team_logo(away)
    home_logo = get_team_logo(home)

    chips = []
    if spread is not None and spread == spread:
        chips.append(f"<span class='chip'><span class='dot dot-green'></span>Home {spread:+}</span>")
    if total is not None and total == total and float(total) != 0.0:
        chips.append(f"<span class='chip'><span class='dot dot-blue'></span>Total {total}</span>")
    if prob is not None and prob == prob:
        chips.append(f"<span class='chip'><span class='dot dot-gold'></span>WP {float(prob):.2f}</span>")
    if mlh is not None and mlh == mlh:
        chips.append(f"<span class='chip'><span class='dot dot-red'></span>ML {home} {int(mlh):+}</span>")
    if mla is not None and mla == mla:
        chips.append(f"<span class='chip'><span class='dot dot-red'></span>ML {away} {int(mla):+}</span>")

    wx_bits = []
    if roof:
        wx_bits.append(str(roof))
    if temp is not None and temp == temp:
        wx_bits.append(f"{float(temp):.0f}F")
    if wind is not None and wind == wind:
        wx_bits.append(f"{float(wind):.0f} mph")
    if precip is not None and precip == precip:
        wx_bits.append(f"{float(precip):.0f}%")

    prob_pct = 0 if prob is None else max(0, min(100, int(float(prob) * 100)))

    recs = []
    if isinstance(units_spread, (int, float)) and units_spread == units_spread and abs(units_spread) >= 0.5:
        side = f"{home} {spread:+}" if spread is not None and spread == spread else f"{home} spread"
        recs.append(f"Spread: {side} ({float(units_spread):.1f}u)")
    if isinstance(units_total, (int, float)) and units_total == units_total and abs(units_total) >= 0.5 and total is not None and ptotal is not None:
        ou = 'Over' if float(ptotal) > float(total) else 'Under'
        recs.append(f"Total: {ou} {total} ({float(units_total):.1f}u)")

    # Winner helper with tie guard to avoid contradictory displays due to rounding
    def _winner_text(a_val, h_val, away_name, home_name, eps=0.05):
        try:
            a = float(a_val)
            h = float(h_val)
        except Exception:
            return None
        if abs(a - h) < eps:
            return "Tie"
        return home_name if h > a else away_name

    w1t = _winner_text(a1, h1h, away, home)
    w2t = _winner_text(a2, h2h, away, home)

    html = f"""
    <div class='game-card'>
      <div class='gc-header'>
        <div class='gc-team'>
          <img class='gc-logo' src='{away_logo}' alt='{away}' />
          <div>
            <div class='name'>{away}</div>
            <div class='meta'>{date_str}</div>
          </div>
        </div>
        <div class='gc-vs'>@</div>
        <div class='gc-team'>
          <img class='gc-logo' src='{home_logo}' alt='{home}' />
          <div>
            <div class='name'>{home}</div>
            <div class='meta'>{' • '.join(wx_bits)}</div>
          </div>
        </div>
      </div>
      <div class='gc-chips'>{''.join(chips)}</div>
            <div class='gc-scores'>
                <div class='pred'><strong>Pred:</strong> {away} {_fmt(pas)} — {home} {_fmt(phs)}</div>
                <div class='total'><strong>Total</strong> {_fmt(ptotal)}</div>
            </div>
            <!-- Half predictions with team scores and consistent winners -->
            <div class='gc-chips'>
                {f"<span class='chip'><span class='dot dot-blue'></span>1H {_fmt(a1)} - {_fmt(h1h)} (tot {_fmt(h1)}){'' if w1t is None else ' • W: ' + w1t}</span>" if (a1 is not None and h1h is not None and h1 is not None) else ''}
                {f"<span class='chip'><span class='dot dot-blue'></span>2H {_fmt(a2)} - {_fmt(h2h)} (tot {_fmt(h2)}){'' if w2t is None else ' • W: ' + w2t}</span>" if (a2 is not None and h2h is not None and h2 is not None) else ''}
            </div>
      <div class='gc-prob'>
        <div class='prob-bar'><div class='prob-fill' style='width:{prob_pct}%'></div></div>
      </div>
      <div class='gc-footer'>
        <div class='rec'>{' • '.join(recs)}</div>
      </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
