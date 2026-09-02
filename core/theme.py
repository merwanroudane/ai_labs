"""
core.theme
==========
Global CSS + plotly template for the platform.

`inject()` is called once at the top of every page (it is idempotent because
Streamlit re-runs the whole script anyway).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .palette import C, SEQ

# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------

_TEMPLATE = go.layout.Template()
_TEMPLATE.layout = go.Layout(
    colorway=SEQ,
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif",
              size=13, color=C["ink"]),
    title=dict(font=dict(size=17, color=C["ink"]), x=0.01, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C["surface_alt"],
    margin=dict(l=60, r=30, t=52, b=52),
    xaxis=dict(gridcolor=C["line"], zerolinecolor="#B9C0D4",
               linecolor=C["line"], ticks="outside", ticklen=5,
               tickcolor=C["line"], title=dict(font=dict(size=13))),
    yaxis=dict(gridcolor=C["line"], zerolinecolor="#B9C0D4",
               linecolor=C["line"], ticks="outside", ticklen=5,
               tickcolor=C["line"], title=dict(font=dict(size=13))),
    legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor=C["line"],
                borderwidth=1, font=dict(size=12)),
    hoverlabel=dict(font_size=12, font_family="Inter, Segoe UI, sans-serif"),
    colorscale=dict(sequential=[[i / 8, c] for i, c in enumerate(
        ["#352A87", "#0363E1", "#1485D4", "#06A7C6", "#38B99E",
         "#92BF73", "#D9BA56", "#FCCE2E", "#F9FB0E"])]),
)
pio.templates["mlplat"] = _TEMPLATE
pio.templates.default = "mlplat"


PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "scrollZoom": False,
}


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}}

/* ---------- page frame ---------- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1250px; }}

/* ---------- chapter hero ---------- */
.mp-hero {{
    background: linear-gradient(118deg, {C['primary']} 0%, {C['primary_dark']} 45%, {C['info']} 100%);
    border-radius: 20px; padding: 26px 30px 24px 30px; color: #fff;
    box-shadow: 0 16px 40px -18px rgba(76,47,209,.75);
    margin-bottom: 20px; position: relative; overflow: hidden;
}}
.mp-hero::after {{
    content:''; position:absolute; right:-70px; top:-90px; width:290px; height:290px;
    background: radial-gradient(circle, rgba(0,194,168,.55) 0%, rgba(0,194,168,0) 68%);
}}
.mp-hero .mp-kicker {{
    text-transform: uppercase; letter-spacing: .18em; font-size: .70rem;
    font-weight: 700; opacity: .85; margin-bottom: 6px;
}}
.mp-hero h1 {{ font-size: 2.0rem; font-weight: 800; margin: 0 0 8px 0; line-height: 1.15; color:#fff; }}
.mp-hero p  {{ font-size: 1.0rem; opacity: .93; margin: 0; max-width: 78ch; line-height:1.55; }}
.mp-hero .mp-chips {{ margin-top: 14px; }}
.mp-chip {{
    display:inline-block; background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.32); border-radius: 999px;
    padding: 3px 12px; font-size: .74rem; font-weight: 600; margin-right: 7px; margin-top:5px;
}}

/* ---------- section heading ---------- */
.mp-sec {{
    display:flex; align-items:center; gap:12px;
    margin: 26px 0 12px 0; padding-bottom: 9px;
    border-bottom: 2px solid {C['surface_deep']};
}}
.mp-sec .mp-num {{
    background: linear-gradient(135deg, {C['primary']}, {C['info']});
    color:#fff; font-weight:800; font-size:.82rem; border-radius: 9px;
    padding: 5px 11px; letter-spacing:.02em; white-space:nowrap;
}}
.mp-sec h2 {{ margin:0; font-size: 1.42rem; font-weight: 750; color: {C['ink']}; }}

.mp-sub {{
    font-size: 1.06rem; font-weight: 700; color: {C['primary_dark']};
    margin: 22px 0 8px 0; padding-left: 11px;
    border-left: 4px solid {C['accent']};
}}

/* ---------- lead paragraph ---------- */
.mp-lead {{
    font-size: 1.03rem; line-height: 1.68; color: {C['ink_soft']};
    background: {C['surface_alt']}; border-radius: 13px;
    padding: 15px 19px; border-left: 5px solid {C['primary']};
    margin-bottom: 16px;
}}

/* ---------- callouts ---------- */
.mp-call {{
    border-radius: 13px; padding: 14px 18px 13px 18px; margin: 14px 0;
    border: 1px solid; line-height: 1.62; font-size: .955rem;
}}
.mp-call .mp-ct {{
    font-weight: 750; font-size: .80rem; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: 6px; display:block;
}}
.mp-call p:last-child {{ margin-bottom: 0; }}
.mp-idea    {{ background:{C['primary_soft']}; border-color:#C9BCFF; color:#2C1D6B; }}
.mp-idea .mp-ct    {{ color:{C['primary_dark']}; }}
.mp-tip     {{ background:{C['accent_soft']}; border-color:#98E8DC; color:#08544A; }}
.mp-tip .mp-ct     {{ color:{C['accent_dark']}; }}
.mp-warn    {{ background:#FFF4E2; border-color:#FFD79C; color:#7A4A05; }}
.mp-warn .mp-ct    {{ color:#B36D00; }}
.mp-pitfall {{ background:#FFE9EE; border-color:#FFC0CE; color:#7C0F2B; }}
.mp-pitfall .mp-ct {{ color:{C['danger']}; }}
.mp-proof   {{ background:#F4EBFF; border-color:#DBC4FF; color:#3F0F73; }}
.mp-proof .mp-ct   {{ color:{C['proof']}; }}
.mp-note    {{ background:#E6F4F9; border-color:#B4DCEA; color:#0B4457; }}
.mp-note .mp-ct    {{ color:{C['info']}; }}
.mp-code    {{ background:#EAF7F1; border-color:#A8E3C9; color:#08462F; }}
.mp-code .mp-ct    {{ color:#0B7A50; }}

/* ---------- key points ---------- */
.mp-keys {{
    background: linear-gradient(140deg, #12172B 0%, #221A4E 100%);
    border-radius: 16px; padding: 18px 22px; color:#E9E7FF; margin: 20px 0;
    box-shadow: 0 12px 34px -20px rgba(18,23,43,.9);
}}
.mp-keys h4 {{ margin:0 0 10px 0; font-size:.80rem; letter-spacing:.14em;
              text-transform:uppercase; color:{C['accent']}; font-weight:750; }}
.mp-keys ul {{ margin:0; padding-left: 20px; }}
.mp-keys li {{ margin-bottom: 7px; line-height:1.6; font-size:.95rem; }}
.mp-keys code {{ background: rgba(255,255,255,.13); color:#9BF3E4;
                 padding:1px 6px; border-radius:5px; font-size:.87em; }}

/* ---------- animation panel ---------- */
.mp-anim {{
    border: 1px solid {C['line']}; border-radius: 16px; padding: 4px 6px 2px 6px;
    background: linear-gradient(180deg, #FFFFFF 0%, {C['surface_alt']} 100%);
    box-shadow: 0 8px 26px -20px rgba(14,20,40,.55); margin: 8px 0 6px 0;
}}
.mp-animhdr {{
    display:flex; align-items:center; gap:9px; margin: 16px 0 6px 0;
    font-weight:700; color:{C['ink']}; font-size:1.0rem;
}}
.mp-animhdr .mp-badge {{
    background: linear-gradient(135deg,{C['danger']},{C['warning']});
    color:#fff; border-radius:999px; padding:3px 11px; font-size:.70rem;
    font-weight:750; letter-spacing:.07em; text-transform:uppercase;
}}

/* ---------- code lab ---------- */
.mp-lab {{
    display:flex; align-items:center; gap:9px; margin: 18px 0 4px 0;
    font-weight:700; color:{C['ink']}; font-size:1.0rem;
}}
.mp-lab .mp-badge {{
    background: linear-gradient(135deg,#0B7A50,{C['success']});
    color:#fff; border-radius:999px; padding:3px 11px; font-size:.70rem;
    font-weight:750; letter-spacing:.07em; text-transform:uppercase;
}}
.stCodeBlock, pre, code {{ font-family: 'JetBrains Mono', Consolas, monospace !important; }}

/* ---------- equation caption ---------- */
.mp-eqcap {{
    text-align:center; font-size:.83rem; color:{C['muted']};
    margin-top:-6px; margin-bottom:14px; font-style: italic;
}}

/* ---------- tables ---------- */
.mp-tbl table {{ width:100%; border-collapse:collapse; font-size:.93rem; margin: 10px 0 16px 0; }}
.mp-tbl th {{
    background:{C['primary']}; color:#fff; text-align:left;
    padding:9px 12px; font-weight:650; font-size:.85rem;
}}
.mp-tbl td {{ padding:8px 12px; border-bottom:1px solid {C['line']}; vertical-align: top; }}
.mp-tbl tr:nth-child(even) td {{ background:{C['surface_alt']}; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {C['surface_alt']}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
.mp-sbtitle {{
    font-size:.72rem; letter-spacing:.15em; text-transform:uppercase;
    font-weight:750; color:{C['muted']}; margin: 12px 0 4px 2px;
}}

/* ---------- progress / pills ---------- */
.mp-pill {{
    display:inline-block; border-radius:999px; padding:2px 10px;
    font-size:.72rem; font-weight:650; margin-right:6px;
}}
.mp-pill-a {{ background:{C['primary_soft']}; color:{C['primary_dark']}; }}
.mp-pill-b {{ background:{C['accent_soft']}; color:{C['accent_dark']}; }}
.mp-pill-c {{ background:#FFF4E2; color:#B36D00; }}

/* ---------- misc ---------- */
hr {{ border:none; border-top:1px solid {C['line']}; margin: 26px 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px 9px 0 0; padding: 6px 14px; font-weight:600; font-size:.92rem;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 750; }}
</style>
"""


def inject() -> None:
    """Inject the global stylesheet. Safe to call on every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)
