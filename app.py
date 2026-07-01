import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJM Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2.5rem;
}
.hero-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #3b82f6;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-title span {
    color: #3b82f6;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    flex: 1;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #f1f5f9;
}
.metric-card .value.good { color: #22c55e; }
.metric-card .value.best { color: #3b82f6; }
.metric-card .sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.25rem;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d45;
}

/* Model comparison table */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.model-table th {
    text-align: left;
    padding: 0.6rem 1rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid #1e2d45;
}
.model-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #0f172a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.model-table tr.winner td { color: #3b82f6; }
.model-table tr:hover td { background: #111827; }

/* Winner badge */
.badge {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    background: #1d3461;
    color: #3b82f6;
    margin-left: 0.5rem;
}

/* Chart container */
.chart-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Insight cards */
.insight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.insight-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.25rem;
}
.insight-card .insight-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.insight-card .insight-body {
    font-size: 0.875rem;
    color: #64748b;
    line-height: 1.6;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── LSTM Model definition (must match training) ────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ── Data & model loading ───────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    df = pd.read_csv("Energy_Data.csv")
    df.columns = df.columns.str.strip()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df


@st.cache_resource
def load_models():
    models = {}
    model_dir = Path("models")

    if (model_dir / "xgb_model.pkl").exists():
        models['xgb'] = joblib.load(model_dir / "xgb_model.pkl")
    if (model_dir / "rf_model.pkl").exists():
        models['rf'] = joblib.load(model_dir / "rf_model.pkl")
    if (model_dir / "lr_model.pkl").exists():
        models['lr'] = joblib.load(model_dir / "lr_model.pkl")
    if (model_dir / "lstm_model.pth").exists():
        lstm = LSTMModel()
        lstm.load_state_dict(torch.load(model_dir / "lstm_model.pth", map_location='cpu'))
        lstm.eval()
        models['lstm'] = lstm
    if (model_dir / "scaler.pkl").exists():
        models['scaler'] = joblib.load(model_dir / "scaler.pkl")

    return models


# ── Hardcoded results (from your training run) ─────────────────────────────────
RESULTS = {
    'Linear Regression': {'rmse': 5730.65, 'mae': None,     'r2': 0.2295},
    'Random Forest':     {'rmse': 4133.52, 'mae': None,     'r2': 0.5991},
    'XGBoost':           {'rmse': 4097.68, 'mae': 3065.31,  'r2': 0.6060},
    'LSTM':              {'rmse': 606.82,  'mae': None,      'r2': 0.9800},
}


# ── Chart helpers ──────────────────────────────────────────────────────────────
DARK_BG   = "#0a0e1a"
CARD_BG   = "#111827"
BORDER    = "#1e2d45"
BLUE      = "#3b82f6"
GREEN     = "#22c55e"
ORANGE    = "#f97316"
SLATE     = "#475569"
TEXT      = "#cbd5e1"

def style_ax(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=SLATE, labelsize=8)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle='--', alpha=0.6)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">PJM Interconnection · Hourly Load · 2002 – 2018</div>
    <div class="hero-title">Grid Demand <span>Forecasting</span></div>
    <div class="hero-sub">Comparing classical ML and deep learning on real electricity grid data</div>
</div>
""", unsafe_allow_html=True)


# ── Top metrics ────────────────────────────────────────────────────────────────
df = load_raw_data()

avg_load  = df['PJME_MW'].mean()
peak_load = df['PJME_MW'].max()
n_years   = (df.index.max() - df.index.min()).days // 365

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Dataset Size</div>
        <div class="value">{len(df):,}</div>
        <div class="sub">hourly observations · {n_years} years</div>
    </div>
    <div class="metric-card">
        <div class="label">Avg Grid Load</div>
        <div class="value">{avg_load:,.0f} <span style="font-size:1rem;color:#475569">MW</span></div>
        <div class="sub">across full time range</div>
    </div>
    <div class="metric-card">
        <div class="label">Peak Demand</div>
        <div class="value">{peak_load:,.0f} <span style="font-size:1rem;color:#475569">MW</span></div>
        <div class="sub">single hour maximum</div>
    </div>
    <div class="metric-card">
        <div class="label">Best Model RMSE</div>
        <div class="value best">606 <span style="font-size:1rem;color:#475569">MW</span></div>
        <div class="sub">LSTM · R² = 0.98</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load profile chart ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Historical Load Profile</div>', unsafe_allow_html=True)

sample = df['PJME_MW'].resample('W').mean()

fig, ax = plt.subplots(figsize=(12, 3.5))
style_ax(ax, fig)
ax.fill_between(sample.index, sample.values, alpha=0.15, color=BLUE)
ax.plot(sample.index, sample.values, color=BLUE, linewidth=1.2)
ax.set_ylabel("MW (weekly avg)", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout(pad=0.5)
st.pyplot(fig)
plt.close()


# ── Model comparison ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem">Model Comparison</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Table
    rows = ""
    for name, res in RESULTS.items():
        r2_pct  = f"{res['r2']*100:.1f}%"
        rmse    = f"{res['rmse']:,.0f}"
        winner  = name == "LSTM"
        badge   = '<span class="badge">best</span>' if winner else ""
        cls     = "winner" if winner else ""
        rows += f"""
        <tr class="{cls}">
            <td>{name}{badge}</td>
            <td>{rmse}</td>
            <td>{r2_pct}</td>
        </tr>"""

    st.markdown(f"""
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>RMSE (MW)</th>
                <th>R²</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

with col_right:
    # RMSE bar chart
    names  = list(RESULTS.keys())
    rmses  = [RESULTS[n]['rmse'] for n in names]
    colors = [BLUE if n == "LSTM" else SLATE for n in names]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    style_ax(ax, fig)
    bars = ax.barh(names, rmses, color=colors, height=0.5)
    ax.set_xlabel("RMSE (MW) — lower is better", fontsize=8)
    ax.invert_yaxis()
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=7.5,
                color=TEXT, fontfamily='monospace')
    ax.set_xlim(0, max(rmses) * 1.18)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()


# ── Hourly demand pattern ──────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem">Demand Patterns</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    hourly = df.groupby(df.index.hour)['PJME_MW'].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3))
    style_ax(ax, fig)
    ax.plot(hourly.index, hourly.values, color=BLUE, linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.1, color=BLUE)
    ax.set_xlabel("Hour of Day", fontsize=8)
    ax.set_ylabel("Avg MW", fontsize=8)
    ax.set_title("Average Load by Hour", fontsize=9, color=TEXT, pad=8)
    ax.set_xticks(range(0, 24, 3))
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()

with col2:
    monthly = df.groupby(df.index.month)['PJME_MW'].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(5.5, 3))
    style_ax(ax, fig)
    bar_colors = [ORANGE if m in [7, 8, 1, 2] else SLATE for m in range(1, 13)]
    ax.bar(range(12), monthly.values, color=bar_colors, width=0.6)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=7)
    ax.set_ylabel("Avg MW", fontsize=8)
    ax.set_title("Average Load by Month", fontsize=9, color=TEXT, pad=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()


# ── Why LSTM won ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem">Why LSTM Dominated</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-grid">
    <div class="insight-card">
        <div class="insight-title">⏱ Sequential memory</div>
        <div class="insight-body">
            Energy demand is time-dependent. What happened 24 hours ago is highly
            predictive of the current hour. LSTM processes data as a sequence and
            retains that context — Random Forest and XGBoost treat each hour as independent.
        </div>
    </div>
    <div class="insight-card">
        <div class="insight-title">📉 7× lower error</div>
        <div class="insight-body">
            LSTM achieved 606 MW RMSE vs 4,097 MW for XGBoost — a 7× improvement.
            On a grid averaging ~32,000 MW, that's the difference between a 1.9% and
            a 12.8% average error, which matters for real dispatch decisions.
        </div>
    </div>
    <div class="insight-card">
        <div class="insight-title">📅 Captures daily cycles</div>
        <div class="insight-body">
            With a 24-hour sequence window, the model sees a full day's context before
            predicting the next hour. It learns daily peaks, overnight troughs, and
            weekend patterns automatically — without manual feature engineering.
        </div>
    </div>
    <div class="insight-card">
        <div class="insight-title">🌡 Limitation: anomalies</div>
        <div class="insight-body">
            LSTM trained on normal patterns may underperform during grid anomalies
            like extreme weather events or blackouts — the same sequences it learned
            from become unreliable. XGBoost with weather features could close the gap
            in those edge cases.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1.5rem;
            border-top:1px solid #1e2d45; color:#334155; font-size:0.75rem;">
    PJM Interconnection Dataset · Models: Linear Regression · Random Forest · XGBoost · LSTM
</div>
""", unsafe_allow_html=True)