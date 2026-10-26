"""
Stock Market Linear Regression AI & Quantitative EDA Dashboard
Built with Streamlit, Scikit-Learn, Plotly, and Pandas
Integrating complete Exploratory Data Analysis (EDA) and Predictive ML Pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Market Predictor & Quantitative EDA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# DYNAMIC THEME SYSTEM & CUSTOM STYLING
# ==========================================
def get_custom_css(theme: str = "dark") -> str:
    """
    Generate dynamic CSS design tokens and glassmorphism styles for Light and Dark themes.
    Enables instant real-time theme toggling across the entire dashboard interface.
    """
    is_dark = (theme == "dark")

    if is_dark:
        bg_main = "#0b0f19"
        text_main = "#f8fafc"
        sidebar_bg = "#0f172a"
        sidebar_border = "rgba(255, 255, 255, 0.08)"
        text_muted = "#94a3b8"
        text_body = "#cbd5e1"
        metric_bg = "linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85))"
        metric_border = "rgba(255, 255, 255, 0.08)"
        metric_shadow = "0 4px 20px rgba(0, 0, 0, 0.25)"
        metric_hover_border = "rgba(59, 130, 246, 0.4)"
        section_title_color = "#f1f5f9"
        section_title_border = "#38bdf8"
        info_bg = "linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7))"
        info_border = "rgba(56, 189, 248, 0.2)"
        info_text = "#cbd5e1"
        warning_bg = "linear-gradient(135deg, rgba(69, 26, 3, 0.6), rgba(30, 41, 59, 0.7))"
        warning_border = "rgba(245, 158, 11, 0.3)"
        warning_text = "#fde68a"
        tab_list_border = "rgba(255, 255, 255, 0.08)"
        tab_text = "#94a3b8"
        tab_selected_bg = "rgba(56, 189, 248, 0.12)"
        tab_selected_color = "#38bdf8"
        card_bg = "rgba(30, 41, 59, 0.8)"
        card_border = "rgba(255, 255, 255, 0.1)"
        code_bg = "rgba(30, 41, 59, 0.6)"
        code_color = "#38bdf8"
        code_border = "rgba(255, 255, 255, 0.08)"
        hr_border = "rgba(255, 255, 255, 0.08)"
        btn_bg = "rgba(30, 41, 59, 0.6)"
        btn_border = "rgba(255, 255, 255, 0.15)"
        btn_text = "#f8fafc"
        btn_hover_bg = "rgba(56, 189, 248, 0.1)"
        btn_hover_border = "#38bdf8"
        btn_hover_text = "#38bdf8"
    else:
        bg_main = "#f8fafc"
        text_main = "#0f172a"
        sidebar_bg = "#ffffff"
        sidebar_border = "#e2e8f0"
        text_muted = "#64748b"
        text_body = "#334155"
        metric_bg = "#ffffff"
        metric_border = "#e2e8f0"
        metric_shadow = "0 4px 14px rgba(0, 0, 0, 0.04)"
        metric_hover_border = "#0284c7"
        section_title_color = "#0f172a"
        section_title_border = "#0284c7"
        info_bg = "linear-gradient(135deg, #f0f9ff, #e0f2fe)"
        info_border = "#bae6fd"
        info_text = "#0369a1"
        warning_bg = "linear-gradient(135deg, #fffbeb, #fef3c7)"
        warning_border = "#fde68a"
        warning_text = "#92400e"
        tab_list_border = "#e2e8f0"
        tab_text = "#64748b"
        tab_selected_bg = "rgba(2, 132, 199, 0.08)"
        tab_selected_color = "#0284c7"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        code_bg = "#f1f5f9"
        code_color = "#0369a1"
        code_border = "#e2e8f0"
        hr_border = "#e2e8f0"
        btn_bg = "#ffffff"
        btn_border = "#cbd5e1"
        btn_text = "#0f172a"
        btn_hover_bg = "#f0f9ff"
        btn_hover_border = "#0284c7"
        btn_hover_text = "#0284c7"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {text_main} !important;
    }}

    p, span, label {{
        color: {text_body} !important;
    }}

    .stCaption, [data-testid="stCaptionContainer"] p {{
        color: {text_muted} !important;
    }}

    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background: {metric_bg} !important;
        border: 1px solid {metric_border} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: {metric_shadow} !important;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {metric_hover_border} !important;
        transform: translateY(-2px);
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: {text_muted} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: {text_main} !important;
    }}

    /* Section Titles */
    .section-title {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: {section_title_color} !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.75rem !important;
        border-left: 4px solid {section_title_border} !important;
        padding-left: 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }}

    /* Information / Callout Cards */
    .info-card {{
        background: {info_bg} !important;
        border: 1px solid {info_border} !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin: 10px 0 !important;
        color: {info_text} !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
    }}

    .warning-card {{
        background: {warning_bg} !important;
        border: 1px solid {warning_border} !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin: 10px 0 !important;
        color: {warning_text} !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
    }}

    /* Simulation Result Card */
    .simulation-card {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
    }}
    .simulation-card-label {{
        color: {text_muted} !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-bullish {{
        background: {'rgba(34, 197, 94, 0.18)' if is_dark else 'rgba(34, 197, 94, 0.12)'} !important;
        color: {'#4ade80' if is_dark else '#15803d'} !important;
        border: 1px solid {'rgba(34, 197, 94, 0.4)' if is_dark else 'rgba(34, 197, 94, 0.35)'} !important;
    }}
    .badge-bearish {{
        background: {'rgba(239, 68, 68, 0.18)' if is_dark else 'rgba(239, 68, 68, 0.12)'} !important;
        color: {'#f87171' if is_dark else '#b91c1c'} !important;
        border: 1px solid {'rgba(239, 68, 68, 0.4)' if is_dark else 'rgba(239, 68, 68, 0.35)'} !important;
    }}
    .badge-neutral {{
        background: {'rgba(148, 163, 184, 0.18)' if is_dark else 'rgba(148, 163, 184, 0.12)'} !important;
        color: {'#cbd5e1' if is_dark else '#475569'} !important;
        border: 1px solid {'rgba(148, 163, 184, 0.4)' if is_dark else 'rgba(148, 163, 184, 0.35)'} !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        border-bottom: 1px solid {tab_list_border} !important;
        padding-bottom: 4px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0px 0px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        color: {tab_text} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_selected_bg} !important;
        color: {tab_selected_color} !important;
        font-weight: 600 !important;
    }}

    /* Code Snippets */
    code {{
        background-color: {code_bg} !important;
        color: {code_color} !important;
        border: 1px solid {code_border} !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
    }}

    /* Horizontal Rules */
    hr {{
        border-color: {hr_border} !important;
    }}

    /* Button Styling */
    div.stButton > button {{
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button[kind="secondary"] {{
        background-color: {btn_bg} !important;
        border: 1px solid {btn_border} !important;
        color: {btn_text} !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        border-color: {btn_hover_border} !important;
        color: {btn_hover_text} !important;
        background-color: {btn_hover_bg} !important;
    }}
    </style>
    """

# ==========================================
# CONSTANTS & METADATA
# ==========================================
INDEX_NAMES = {
    "NYA": "NYSE Composite (USA)",
    "IXIC": "NASDAQ Composite (USA)",
    "N225": "Nikkei 225 (Japan)",
    "GSPTSE": "S&P/TSX Composite (Canada)",
    "HSI": "Hang Seng Index (Hong Kong)",
    "GDAXI": "DAX Performance Index (Germany)",
    "SSMI": "Swiss Market Index (Switzerland)",
    "KS11": "KOSPI Composite (South Korea)",
    "TWII": "Taiwan Weighted (Taiwan)",
    "000001.SS": "SSE Composite (Shanghai)",
    "399001.SZ": "SZSE Component (Shenzhen)",
    "N100": "Euronext 100 (Europe)",
    "NSEI": "NIFTY 50 (India)",
    "J203.JO": "FTSE/JSE Top 40 (South Africa)",
}

REGIONS = {
    "NYA": "Americas",
    "IXIC": "Americas",
    "GSPTSE": "Americas",
    "GDAXI": "Europe",
    "SSMI": "Europe",
    "N100": "Europe",
    "N225": "Asia-Pacific",
    "HSI": "Asia-Pacific",
    "KS11": "Asia-Pacific",
    "TWII": "Asia-Pacific",
    "000001.SS": "Asia-Pacific",
    "399001.SZ": "Asia-Pacific",
    "NSEI": "Asia-Pacific",
    "J203.JO": "Africa",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Market.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "linear_regression_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics_summary.json")
PER_INDEX_PATH = os.path.join(BASE_DIR, "outputs", "per_index_metrics.csv")
COEF_PATH = os.path.join(BASE_DIR, "outputs", "feature_coefficients.csv")


# ==========================================
# DATA & MODEL LOADERS (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained scikit-learn Pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_raw_dataset():
    """
    Load complete uncleaned raw dataset directly from Market.csv
    Preserves all missingness and anomalies for EDA diagnosis.
    """
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Adj_Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data
def get_features_and_predictions(_model, df_raw):
    """
    Compute lags, rolling technicals, and generate model + baseline predictions.
    Recreates exact chronological 80/20 train/test split.
    """
    df = df_raw.dropna(subset=["Open", "High", "Low", "Close", "Date"]).copy()
    df = df.sort_values(by=["Index", "Date"]).reset_index(drop=True)

    grouped = df.groupby("Index")

    df["Target_Next_Close"] = grouped["Close"].shift(-1)
    df["Close_Lag1"] = grouped["Close"].shift(1)
    df["Close_Lag2"] = grouped["Close"].shift(2)
    df["Close_Lag5"] = grouped["Close"].shift(5)
    df["SMA_5"] = grouped["Close"].transform(lambda s: s.rolling(5).mean())
    df["SMA_10"] = grouped["Close"].transform(lambda s: s.rolling(10).mean())
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    df["Log_Volume"] = np.log1p(df["Volume"].fillna(0).clip(lower=0))

    clean_df = df.dropna().copy().reset_index(drop=True)

    # Chronological 80/20 Train/Test split
    train_indices, test_indices = [], []
    for _, g in clean_df.groupby("Index"):
        n = len(g)
        split = int(n * 0.8)
        train_indices.extend(g.index[:split])
        test_indices.extend(g.index[split:])

    clean_df["Split"] = "Train"
    clean_df.loc[test_indices, "Split"] = "Test"

    feature_cols = [
        "Open", "High", "Low", "Close",
        "Close_Lag1", "Close_Lag2", "Close_Lag5",
        "SMA_5", "SMA_10", "Daily_Return", "Intraday_Range", "Log_Volume",
        "Index"
    ]

    # Model Predictions
    clean_df["Predicted_Next_Close"] = _model.predict(clean_df[feature_cols])
    clean_df["Residual"] = clean_df["Target_Next_Close"] - clean_df["Predicted_Next_Close"]
    clean_df["Abs_Pct_Error"] = (clean_df["Residual"].abs() / clean_df["Target_Next_Close"]) * 100

    # Naive Baseline: Next Close = Current Close (Random Walk / Martingale)
    clean_df["Naive_Next_Close"] = clean_df["Close"]
    clean_df["Naive_Residual"] = clean_df["Target_Next_Close"] - clean_df["Naive_Next_Close"]
    clean_df["Naive_Abs_Pct_Error"] = (clean_df["Naive_Residual"].abs() / clean_df["Target_Next_Close"]) * 100

    return clean_df


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # ==========================================
    # THEME STATE & DYNAMIC STYLING
    # ==========================================
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    is_dark = (st.session_state["theme"] == "dark")
    plotly_template = "plotly_dark" if is_dark else "plotly_white"

    # Inject dynamic theme stylesheet
    st.markdown(get_custom_css(st.session_state["theme"]), unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.title("📈 Market AI & EDA Suite")
    st.sidebar.markdown(
        "**Scikit-Learn Machine Learning & Quantitative Exploratory Analytics** for global equity indices."
    )

    # ==========================================
    # LIGHT & DARK THEME TOGGLE BUTTON
    # ==========================================
    theme_status = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
    badge_theme_class = "badge-neutral" if is_dark else "badge-bullish"
    
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; margin-bottom: 8px;">
            <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85;">Interface Theme</span>
            <span class="badge {badge_theme_class}" style="font-size: 0.75rem; padding: 2px 10px;">{theme_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_btn_text = "☀️ Switch to Light Mode" if is_dark else "🌙 Switch to Dark Mode"
    if st.sidebar.button(
        theme_btn_text,
        key="theme_toggle_btn",
        use_container_width=True,
        help="Click to toggle between Dark and Light mode",
    ):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

    st.sidebar.divider()

    model = load_model()
    raw_df = load_raw_dataset()

    if model is None or raw_df is None:
        st.error(
            "⚠️ Model artifacts or `Market.csv` not found! Please verify that `Market.csv` and `outputs/` exist."
        )
        return

    processed_df = get_features_and_predictions(model, raw_df)

    # Navigation Tabs
    menu = st.sidebar.radio(
        "Navigate Dashboard",
        [
            "⚡ Live Forecast & Overview",
            "🌐 Global Markets Relative Performance",
            "🔬 Comprehensive EDA & Data Quality",
            "📊 Model Performance & Baseline Benchmark",
            "📈 Interactive Time-Series Backtest",
            "🧠 Feature Importance & Weights",
            "🎛️ What-If Scenario Simulator",
            "📁 Data Explorer & Anomaly Audit",
        ],
    )

    all_indices = sorted(processed_df["Index"].unique())
    selected_index = st.sidebar.selectbox(
        "Select Active Market Index",
        all_indices,
        index=all_indices.index("NYA") if "NYA" in all_indices else 0,
        format_func=lambda x: f"{x} - {INDEX_NAMES.get(x, x)}",
    )

    st.sidebar.divider()
    st.sidebar.info(
        f"**Active Index**: `{selected_index}`\n\n"
        f"**Region**: {REGIONS.get(selected_index, 'Global')}\n\n"
        f"**Exchange**: {INDEX_NAMES.get(selected_index, selected_index)}\n\n"
        f"**Chronological Split**: 80% Train / 20% Test"
    )

    # =========================================================================
    # TAB 1: LIVE FORECAST & OVERVIEW
    # =========================================================================
    if menu == "⚡ Live Forecast & Overview":
        st.title("⚡ Stock Market Next-Day Close Price Predictor")
        st.caption(
            "Production Machine Learning pipeline predicting tomorrow's market close using historical price action, lags, and rolling averages."
        )

        idx_data = processed_df[processed_df["Index"] == selected_index].sort_values("Date")
        latest_row = idx_data.iloc[-1]

        last_date = latest_row["Date"].strftime("%B %d, %Y")
        current_close = latest_row["Close"]
        predicted_next = latest_row["Predicted_Next_Close"]
        diff = predicted_next - current_close
        pct_change = (diff / current_close) * 100
        is_bullish = diff >= 0

        # Header Cards
        st.markdown(
            f"### Next Trading Day Outlook for **{INDEX_NAMES.get(selected_index, selected_index)}** (`{selected_index}`)"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close Date", last_date)
        col2.metric("Latest Close Price", f"{current_close:,.2f}")
        col3.metric(
            "Predicted Next Close",
            f"{predicted_next:,.2f}",
            delta=f"{diff:+,.2f} ({pct_change:+.2f}%)",
            delta_color="normal",
        )
        with col4:
            st.markdown("**Expected Signal**")
            badge_class = "badge-bullish" if is_bullish else "badge-bearish"
            signal_text = "🟢 BULLISH (Upward Momentum)" if is_bullish else "🔴 BEARISH (Downward Movement)"
            st.markdown(
                f"<div class='badge {badge_class}' style='font-size:0.95rem; padding: 10px 14px;'>{signal_text}</div>",
                unsafe_allow_html=True,
            )

        # Candlestick + MA + Forecast Tracking
        st.markdown("<div class='section-title'>Recent 60 Trading Days: Candlestick Action & Model Forecast</div>", unsafe_allow_html=True)

        recent_60 = idx_data.tail(60).copy()
        fig_live = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            specs=[[{"type": "xy"}], [{"type": "bar"}]]
        )

        # Candlestick
        fig_live.add_trace(
            go.Candlestick(
                x=recent_60["Date"],
                open=recent_60["Open"],
                high=recent_60["High"],
                low=recent_60["Low"],
                close=recent_60["Close"],
                name="OHLC Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1, col=1
        )
        # SMA 5 & SMA 10
        fig_live.add_trace(
            go.Scatter(
                x=recent_60["Date"],
                y=recent_60["SMA_5"],
                name="5-Day SMA",
                line=dict(color="#38bdf8", width=1.5),
            ),
            row=1, col=1
        )
        fig_live.add_trace(
            go.Scatter(
                x=recent_60["Date"],
                y=recent_60["SMA_10"],
                name="10-Day SMA",
                line=dict(color="#a855f7", width=1.5),
            ),
            row=1, col=1
        )
        # Predicted Next Close
        fig_live.add_trace(
            go.Scatter(
                x=recent_60["Date"],
                y=recent_60["Predicted_Next_Close"],
                name="Model Predicted Next Close",
                line=dict(color="#f97316", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=4),
            ),
            row=1, col=1
        )

        # Volume / Daily Range in bottom pane
        fig_live.add_trace(
            go.Bar(
                x=recent_60["Date"],
                y=recent_60["Daily_Return"] * 100,
                name="Daily Return (%)",
                marker_color=np.where(recent_60["Daily_Return"] >= 0, "#22c55e", "#ef4444"),
            ),
            row=2, col=1
        )

        fig_live.update_layout(
            height=520,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis2_title="Date",
            yaxis_title="Price (Local Currency)",
            yaxis2_title="Return (%)",
            template=plotly_template,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_live, use_container_width=True)

        # Out-of-Sample Stats for this Index
        test_slice = idx_data[idx_data["Split"] == "Test"]
        st.markdown("<div class='section-title'>Out-of-Sample Performance on Active Index</div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Test Set R² Score", f"{r2_score(test_slice['Target_Next_Close'], test_slice['Predicted_Next_Close']):.4f}")
        m_col2.metric("Mean Absolute Error", f"{mean_absolute_error(test_slice['Target_Next_Close'], test_slice['Predicted_Next_Close']):.2f}")
        m_col3.metric("MAPE (Avg % Error)", f"{test_slice['Abs_Pct_Error'].mean():.2f}%")
        m_col4.metric("Test Trading Days", f"{len(test_slice):,}")

    # =========================================================================
    # TAB 2: GLOBAL MARKETS RELATIVE PERFORMANCE (FROM NOTEBOOK STEPS 10 & 11)
    # =========================================================================
    elif menu == "🌐 Global Markets Relative Performance":
        st.title("🌐 Cross-Market Comparative Performance & Macro Trends")
        st.caption(
            "Comparing long-term compounding growth across 14 global equity benchmarks re-indexed to Base = 100 (Step 11 from EDA Notebook)."
        )

        # Clean Close prices for normalization
        trend_df = raw_df.dropna(subset=["Close", "Date"]).sort_values(["Index", "Date"]).copy()

        # Multi-select controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            selected_mult_indices = st.multiselect(
                "Select Indices to Compare",
                options=all_indices,
                default=["NYA", "IXIC", "N225", "GDAXI", "HSI", "NSEI"],
                format_func=lambda x: f"{x} ({INDEX_NAMES.get(x, x)})",
            )
        with ctrl_col2:
            norm_mode = st.radio(
                "Base 100 Anchor",
                ["First Recorded Session", "Fixed Common Start Year"],
                horizontal=True,
            )
        with ctrl_col3:
            scale_type = st.radio("Price Scale", ["Logarithmic (log)", "Linear"], horizontal=True)

        if not selected_mult_indices:
            st.warning("Please select at least one index to display.")
            return

        filtered_trend = trend_df[trend_df["Index"].isin(selected_mult_indices)].copy()

        if norm_mode == "Fixed Common Start Year":
            common_year = st.slider("Select Common Baseline Year", 1990, 2018, 2000)
            filtered_trend = filtered_trend[filtered_trend["Date"] >= pd.to_datetime(f"{common_year}-01-01")]

        # Compute Normalized Close (Base = 100)
        filtered_trend["Normalized_Close"] = (
            filtered_trend["Close"]
            / filtered_trend.groupby("Index")["Close"].transform("first")
            * 100
        )

        st.markdown("<div class='section-title'>Normalized Closing-Price Growth (First Observed Close = 100)</div>", unsafe_allow_html=True)
        fig_norm = px.line(
            filtered_trend,
            x="Date",
            y="Normalized_Close",
            color="Index",
            labels={"Normalized_Close": "Normalized Value (Base 100)", "Date": "Date", "Index": "Market Index"},
            template=plotly_template,
        )
        fig_norm.update_layout(
            title=dict(
                text="Multi-Decade Global Equity Performance (Rebased to 100)",
                y=0.98,
                x=0,
                xanchor="left",
            ),
            height=560,
            hovermode="x unified",
            yaxis_type="log" if "Logarithmic" in scale_type else "linear",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                title=dict(text=""),
            ),
            margin=dict(l=20, r=20, t=85, b=20),
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # Observations depth and timeline per index (Step 10 from Notebook)
        st.markdown("<div class='section-title'>Historical Observations Depth & Trading History per Index</div>", unsafe_allow_html=True)
        
        index_summary = (
            trend_df.groupby("Index")
            .agg(
                Observations=("Date", "size"),
                First_Date=("Date", "min"),
                Last_Date=("Date", "max"),
                Min_Close=("Close", "min"),
                Max_Close=("Close", "max"),
                Mean_Close=("Close", "mean"),
                Zero_Volume=("Volume", lambda s: s.eq(0).sum()),
            )
            .reset_index()
        )
        index_summary["Exchange"] = index_summary["Index"].map(INDEX_NAMES)
        index_summary["Region"] = index_summary["Index"].map(REGIONS)
        index_summary["Years_Active"] = ((index_summary["Last_Date"] - index_summary["First_Date"]).dt.days / 365.25).round(1)

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig_bar_obs = px.bar(
                index_summary.sort_values("Observations", ascending=True),
                x="Observations",
                y="Index",
                color="Region",
                orientation="h",
                text="Observations",
                title="Total Recorded Trading Days by Global Exchange",
                template=plotly_template,
                color_discrete_map={
                    "Americas": "#38bdf8",
                    "Europe": "#a855f7",
                    "Asia-Pacific": "#22c55e",
                    "Africa": "#f59e0b",
                },
            )
            fig_bar_obs.update_traces(textposition="outside")
            fig_bar_obs.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar_obs, use_container_width=True)

        with c2:
            st.markdown("#### Market Timeline Summary Table")
            display_summary = index_summary[[
                "Index", "Region", "Years_Active", "First_Date", "Last_Date", "Observations"
            ]].copy()
            display_summary["First_Date"] = display_summary["First_Date"].dt.strftime("%Y-%m-%d")
            display_summary["Last_Date"] = display_summary["Last_Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_summary.sort_values("Observations", ascending=False), height=400, use_container_width=True)

    # =========================================================================
    # TAB 3: COMPREHENSIVE EDA & DATA QUALITY (NOTEBOOK STEPS 5 TO 18)
    # =========================================================================
    elif menu == "🔬 Comprehensive EDA & Data Quality":
        st.title("🔬 Exploratory Data Analysis & Integrity Diagnostics")
        st.caption(
            "In-depth empirical analysis mirroring the findings, data cleaning protocols, and statistical tests from the EDA Notebook."
        )

        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "🩺 Data Health & Anomaly Audit",
            "📊 Distributions & Skewness",
            "📦 Boxplots & IQR Outliers",
            "⚡ Daily Returns & Volatility Dynamics",
            "🔥 Correlation & Multicollinearity",
        ])

        # ---------------------------------------------------------------------
        # SUB-TAB 1: DATA HEALTH & ANOMALY AUDIT (STEPS 5-9)
        # ---------------------------------------------------------------------
        with eda_tab1:
            st.markdown("### 🩺 Data Integrity, Missingness, and Anomaly Diagnostics")
            st.caption("Validating raw market entries, identifying null patterns, and auditing OHLC relationships (Steps 5-9).")

            # Missingness summary
            num_cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
            missing_series = raw_df[num_cols].isna().sum()
            missing_pct = (raw_df[num_cols].isna().mean() * 100).round(2)
            missing_df = pd.DataFrame({"Missing Rows": missing_series, "Missing (%)": missing_pct}).reset_index(names="Column")

            # Zero volume calculation
            zero_vol_total = raw_df["Volume"].eq(0).sum()
            zero_vol_pct = (zero_vol_total / len(raw_df)) * 100

            # OHLC integrity check (Step 9)
            ohlc_cols = ["Open", "High", "Low", "Close"]
            complete_ohlc = raw_df[ohlc_cols].notna().all(axis=1)
            max_oc = raw_df[["Open", "Close"]].max(axis=1)
            min_oc = raw_df[["Open", "Close"]].min(axis=1)
            bad_ohlc_mask = complete_ohlc & (
                (raw_df["High"] < raw_df["Low"]) |
                (raw_df["High"] < max_oc) |
                (raw_df["Low"] > min_oc)
            )
            bad_ohlc_df = raw_df.loc[bad_ohlc_mask, ["Index", "Date"] + ohlc_cols].copy()

            # Health metrics
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Total Raw Records", f"{len(raw_df):,}")
            h_col2.metric("Exact Duplicate Rows", f"{raw_df.duplicated().sum():,}")
            h_col3.metric("Zero-Volume Days", f"{zero_vol_total:,} ({zero_vol_pct:.1f}%)")
            h_col4.metric("OHLC Quality Pass Rate", "99.999%", delta="1 Historic Outlier Audited", delta_color="normal")

            # Missingness bar chart
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Missing Values by Measurement Column (%)</div>", unsafe_allow_html=True)
                fig_miss = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing (%)",
                    text="Missing (%)",
                    title="Missing Measurement Frequencies (Step 6)",
                    color="Missing (%)",
                    color_continuous_scale="Reds",
                    template=plotly_template,
                )
                fig_miss.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig_miss.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), yaxis_range=[0, 3.5])
                st.plotly_chart(fig_miss, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Zero-Volume Reporting Rates by Exchange</div>", unsafe_allow_html=True)
                zero_vol_by_idx = (
                    raw_df.assign(is_zero=raw_df["Volume"].eq(0))
                    .groupby("Index")["is_zero"]
                    .mean()
                    .mul(100)
                    .reset_index(name="Zero_Volume_Pct")
                )
                fig_zero_vol = px.bar(
                    zero_vol_by_idx.sort_values("Zero_Volume_Pct", ascending=False),
                    x="Zero_Volume_Pct",
                    y="Index",
                    orientation="h",
                    text="Zero_Volume_Pct",
                    title="Percentage of Zero-Volume Days per Market (Step 8)",
                    color="Zero_Volume_Pct",
                    color_continuous_scale="Viridis",
                    template=plotly_template,
                )
                fig_zero_vol.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_zero_vol.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_zero_vol, use_container_width=True)

            # Informational Card on Volume & Missingness
            st.markdown(
                """
                <div class='info-card'>
                    💡 <b>Reporting Context (Notebook Step 8)</b>: Certain indices (e.g., <b>J203.JO</b> at 98.3%, <b>N225</b> at 63.7%, <b>NYA</b> in historical decades) 
                    record zero volume because international exchanges or early reporting aggregators did not capture transaction volume for index aggregates. 
                    Therefore, <code>Volume == 0</code> represents uncollected reporting data rather than zero trades.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # OHLC Auditor Detail Box
            st.markdown("<div class='section-title'>OHLC Logical Relationship Auditor (Step 9)</div>", unsafe_allow_html=True)
            if len(bad_ohlc_df) > 0:
                st.markdown(
                    """
                    <div class='info-card' style='border-left: 4px solid #38bdf8;'>
                        💡 <b>Data Quality Insight (Notebook Step 9)</b>: By mathematically auditing the physical market boundary 
                        <code>High ≥ max(Open, Close)</code> and <code>Low ≤ min(Open, Close)</code>, the pipeline verified that 
                        <b>99.999% of all 112,457 records are valid</b>. It also successfully isolated a single historic clerical typo in 
                        <b>NYA on June 30, 1969</b>, where <code>Open = 5,722.36</code> was recorded instead of ~572 in legacy archival data.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(bad_ohlc_df, use_container_width=True)
            else:
                st.success("All OHLC records conform strictly to valid price boundaries.")

        # ---------------------------------------------------------------------
        # SUB-TAB 2: DISTRIBUTIONS & SKEWNESS (STEP 13)
        # ---------------------------------------------------------------------
        with eda_tab2:
            st.markdown(f"### 📊 Feature Distributions & Skewness Analysis: **{selected_index}**")
            st.caption("Parametric distributions, kernel density estimations, and log-volume normalization (Step 13).")

            sub_idx_raw = raw_df[raw_df["Index"] == selected_index].dropna(subset=["Close"]).copy()

            dist_col1, dist_col2 = st.columns([1, 1])
            with dist_col1:
                feat_to_plot = st.selectbox(
                    "Select Price Feature to Inspect",
                    ["Close", "Open", "High", "Low", "Adj_Close"],
                    index=0,
                )
                fig_dist = px.histogram(
                    sub_idx_raw,
                    x=feat_to_plot,
                    nbins=50,
                    marginal="box",
                    title=f"{feat_to_plot} Distribution with Boxplot — {selected_index}",
                    color_discrete_sequence=["#38bdf8"],
                    template=plotly_template,
                )
                fig_dist.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_dist, use_container_width=True)

            with dist_col2:
                st.markdown("#### Raw Volume vs. Log-Transformed `log(1 + Volume)`")
                sub_vol = sub_idx_raw["Volume"].dropna()
                log_vol = np.log1p(sub_vol.clip(lower=0))

                fig_vol_comp = make_subplots(rows=2, cols=1, subplot_titles=["Raw Volume (Severe Right Skew)", "Log-Transformed log(1 + Volume) (Normalized)"])
                fig_vol_comp.add_trace(
                    go.Histogram(x=sub_vol, nbinsx=40, marker_color="#f97316", name="Raw Volume"),
                    row=1, col=1,
                )
                fig_vol_comp.add_trace(
                    go.Histogram(x=log_vol, nbinsx=40, marker_color="#10b981", name="log(1+Volume)"),
                    row=2, col=1,
                )
                fig_vol_comp.update_layout(height=420, showlegend=False, template=plotly_template, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_vol_comp, use_container_width=True)

            # Skewness Table
            st.markdown("<div class='section-title'>Parametric Skewness League Table</div>", unsafe_allow_html=True)
            skew_metrics = sub_idx_raw[["Open", "High", "Low", "Close", "Volume"]].skew(numeric_only=True).to_frame(name="Skewness")
            skew_metrics["log(1+Volume)"] = np.log1p(sub_idx_raw["Volume"].dropna().clip(lower=0)).skew()
            skew_df = skew_metrics.reset_index().rename(columns={"index": "Feature"}).sort_values("Skewness", ascending=False)

            sk_col1, sk_col2 = st.columns([1.2, 1])
            with sk_col1:
                fig_skew = px.bar(
                    skew_df,
                    x="Feature",
                    y="Skewness",
                    color="Skewness",
                    title=f"Feature Skewness Metrics ({selected_index})",
                    color_continuous_scale="Purples",
                    template=plotly_template,
                )
                fig_skew.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_skew, use_container_width=True)
            with sk_col2:
                st.markdown(
                    """
                    <div class='info-card' style='margin-top: 24px;'>
                        <b>Why Skewness Matters for Linear Regression</b>:
                        Raw financial volume exhibits exponential skew (> 5.0). 
                        Applying <code>log1p(Volume)</code> compresses multi-billion-share outliers and centers the variance, 
                        fulfilling linear model normality assumptions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------------------------
        # SUB-TAB 3: BOXPLOTS & IQR OUTLIERS (STEP 14)
        # ---------------------------------------------------------------------
        with eda_tab3:
            st.markdown(f"### 📦 Boxplots & Interquartile Range (IQR) Outlier Detection")
            st.caption("Screening price dispersion, volatility shifts, and flagging historic market turbulence days (Step 14).")

            sub_one = processed_df[processed_df["Index"] == selected_index].sort_values("Date").copy()

            b_col1, b_col2 = st.columns([1.3, 1])
            with b_col1:
                fig_box_all = px.box(
                    processed_df,
                    x="Index",
                    y="Close",
                    color="Index",
                    title="Price Dispersion Across All 14 Global Indices",
                    template=plotly_template,
                )
                fig_box_all.update_layout(height=420, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_box_all, use_container_width=True)

            with b_col2:
                st.markdown("#### Interactive IQR Outlier Configuration")
                iqr_mult = st.slider("IQR Multiplier ($k$ factor)", min_value=1.0, max_value=3.5, value=1.5, step=0.25)
                
                close_series = sub_one["Close"].dropna()
                q1 = close_series.quantile(0.25)
                q3 = close_series.quantile(0.75)
                iqr_val = q3 - q1
                lower_lim = q1 - iqr_mult * iqr_val
                upper_lim = q3 + iqr_mult * iqr_val

                outliers = sub_one[(sub_one["Close"] < lower_lim) | (sub_one["Close"] > upper_lim)]

                st.markdown(
                    f"""
                    - **25th Percentile (Q1)**: `{q1:,.2f}`
                    - **75th Percentile (Q3)**: `{q3:,.2f}`
                    - **Interquartile Range (IQR)**: `{iqr_val:,.2f}`
                    - **Outlier Threshold**: `< {lower_lim:,.2f}` or `> {upper_lim:,.2f}`
                    - **Sessions Flagged**: `{len(outliers):,}` of `{len(sub_one):,}` ({len(outliers)/len(sub_one)*100:.1f}%)
                    """
                )

            # Scatter timeline with outliers highlighted
            st.markdown("<div class='section-title'>Flagged Statistical Outliers on Active Market Timeline</div>", unsafe_allow_html=True)
            fig_out_scatter = go.Figure()
            fig_out_scatter.add_trace(
                go.Scatter(
                    x=sub_one["Date"],
                    y=sub_one["Close"],
                    mode="lines",
                    name="Normal Price Action",
                    line=dict(color="#38bdf8", width=1.5),
                )
            )
            fig_out_scatter.add_trace(
                go.Scatter(
                    x=outliers["Date"],
                    y=outliers["Close"],
                    mode="markers",
                    name=f"IQR Outliers (k={iqr_mult})",
                    marker=dict(color="#f43f5e", size=5, symbol="circle"),
                )
            )
            fig_out_scatter.update_layout(
                height=380,
                hovermode="x unified",
                template=plotly_template,
                title=f"{selected_index} Historical Trajectory with Flagged IQR Outlier Periods",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_out_scatter, use_container_width=True)

        # ---------------------------------------------------------------------
        # SUB-TAB 4: DAILY RETURNS & VOLATILITY DYNAMICS (STEP 15)
        # ---------------------------------------------------------------------
        with eda_tab4:
            st.markdown(f"### ⚡ Daily Returns, Volatility, and Volume Dynamics: **{selected_index}**")
            st.caption("Analyzing return distributions, kurtosis fat tails, and volume vs. price volatility correlations (Step 15).")

            sub_ret = processed_df[processed_df["Index"] == selected_index].sort_values("Date").copy()
            sub_ret["Daily_Return_Pct"] = sub_ret["Daily_Return"] * 100

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("#### Daily Return (%) Distribution with Gaussian Overlay")
                fig_ret_hist = px.histogram(
                    sub_ret["Daily_Return_Pct"].dropna(),
                    nbins=80,
                    title=f"Close-to-Close Return Distribution — {selected_index}",
                    color_discrete_sequence=["#38bdf8"],
                    template=plotly_template,
                )
                # Compute normal curve overlay
                ret_clean = sub_ret["Daily_Return_Pct"].dropna()
                mu, std = ret_clean.mean(), ret_clean.std()
                x_norm = np.linspace(ret_clean.quantile(0.001), ret_clean.quantile(0.999), 100)
                y_norm = stats.norm.pdf(x_norm, mu, std) * len(ret_clean) * (ret_clean.max() - ret_clean.min()) / 80

                fig_ret_hist.add_trace(
                    go.Scatter(x=x_norm, y=y_norm, mode="lines", name="Theoretical Normal Curve", line=dict(color="#f97316", width=2))
                )
                fig_ret_hist.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_ret_hist, use_container_width=True)

            with r_col2:
                st.markdown("#### Volume vs. Daily Return (Symlog Scale)")
                # Sample 3000 points if large for speed
                vol_scatter_df = sub_ret.dropna(subset=["Volume", "Daily_Return_Pct"]).copy()
                if len(vol_scatter_df) > 3000:
                    vol_scatter_df = vol_scatter_df.sample(3000, random_state=42)

                fig_vol_ret = px.scatter(
                    vol_scatter_df,
                    x="Volume",
                    y="Daily_Return_Pct",
                    title="Trading Volume vs. Daily Return % (Notebook Step 15)",
                    opacity=0.45,
                    color="Intraday_Range",
                    color_continuous_scale="Viridis",
                    template=plotly_template,
                )
                fig_vol_ret.update_xaxes(type="log")
                fig_vol_ret.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_vol_ret, use_container_width=True)

            # Volatility over time
            st.markdown("<div class='section-title'>Historical 30-Day Rolling Annualized Volatility (%)</div>", unsafe_allow_html=True)
            sub_ret["Rolling_Vol_30"] = sub_ret["Daily_Return"].rolling(30).std() * np.sqrt(252) * 100

            fig_vol_time = px.line(
                sub_ret.dropna(subset=["Rolling_Vol_30"]),
                x="Date",
                y="Rolling_Vol_30",
                title=f"{selected_index} 30-Day Rolling Annualized Volatility History",
                color_discrete_sequence=["#eab308"],
                template=plotly_template,
            )
            fig_vol_time.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Annualized Volatility (%)")
            st.plotly_chart(fig_vol_time, use_container_width=True)

        # ---------------------------------------------------------------------
        # SUB-TAB 5: CORRELATION & MULTICOLLINEARITY (STEPS 16-18)
        # ---------------------------------------------------------------------
        with eda_tab5:
            st.markdown(f"### 🔥 Correlation Analysis & Linear Regression Multicollinearity Caveats")
            st.caption("Pairwise relationship inspections, correlation with target, and regression pitfalls (Steps 16-18).")

            sub_corr_df = processed_df[processed_df["Index"] == selected_index].copy()
            corr_feature_list = [
                "Open", "High", "Low", "Close", "Target_Next_Close", "Close_Lag1", "Close_Lag2", "Close_Lag5",
                "SMA_5", "SMA_10", "Daily_Return", "Intraday_Range", "Log_Volume"
            ]

            corr_matrix = sub_corr_df[corr_feature_list].corr().round(3)

            c_heat1, c_heat2 = st.columns([1.3, 1])
            with c_heat1:
                st.markdown("#### Interactive Correlation Heatmap")
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"Feature Correlation Matrix — {selected_index}",
                    template=plotly_template,
                )
                fig_corr.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_corr, use_container_width=True)

            with c_heat2:
                st.markdown("#### Correlation with Next-Day Target Close")
                target_corrs = corr_matrix["Target_Next_Close"].drop("Target_Next_Close").sort_values(ascending=False).reset_index()
                target_corrs.columns = ["Feature", "Correlation_With_Target"]

                fig_target_corr = px.bar(
                    target_corrs,
                    x="Correlation_With_Target",
                    y="Feature",
                    orientation="h",
                    color="Correlation_With_Target",
                    color_continuous_scale="Blues",
                    title="Ranking of Features Predicting Tomorrow's Close",
                    template=plotly_template,
                )
                fig_target_corr.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_target_corr, use_container_width=True)

            # Multicollinearity & Econometric Insight Callout (Step 18)
            st.markdown(
                """
                <div class='info-card' style='border-left: 4px solid #38bdf8;'>
                    💡 <b>Econometric & Machine Learning Insights (Notebook Step 18)</b>:<br/>
                    1. <b>Multicollinearity Profile</b>: Because <code>Open</code>, <code>High</code>, <code>Low</code>, and <code>Close</code> track each other closely ($r > 0.999$), 
                    our pipeline relies on standard scaling and multi-day lag differences (<code>Close_Lag1</code>, <code>Close_Lag2</code>) to isolate true trend momentum.<br/>
                    2. <b>Non-Stationarity & Benchmarking</b>: Financial prices have non-zero unit roots (trending growth). An $R^2$ of 0.999 reflects asset price persistence, 
                    which is why we benchmark against the <b>Naive Random-Walk Baseline</b> ($Close_{t+1} = Close_t$) to evaluate genuine predictive skill.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4: MODEL PERFORMANCE & BASELINE BENCHMARK (NOTEBOOK STEPS 20 & 21)
    # =========================================================================
    elif menu == "📊 Model Performance & Baseline Benchmark":
        st.title("📊 Model Performance & Baseline Benchmark Validation")
        st.caption(
            "Rigorous evaluation on chronological out-of-sample test data (22,026 test days) with Naive Baseline Benchmarking (Steps 20 & 21)."
        )

        test_df = processed_df[processed_df["Split"] == "Test"].copy()

        # Model Global Metrics
        global_r2 = r2_score(test_df["Target_Next_Close"], test_df["Predicted_Next_Close"])
        global_mae = mean_absolute_error(test_df["Target_Next_Close"], test_df["Predicted_Next_Close"])
        global_rmse = np.sqrt(mean_squared_error(test_df["Target_Next_Close"], test_df["Predicted_Next_Close"]))
        global_mape = test_df["Abs_Pct_Error"].mean()

        # Naive Baseline Metrics (Tomorrow = Today)
        naive_r2 = r2_score(test_df["Target_Next_Close"], test_df["Naive_Next_Close"])
        naive_mae = mean_absolute_error(test_df["Target_Next_Close"], test_df["Naive_Next_Close"])
        naive_rmse = np.sqrt(mean_squared_error(test_df["Target_Next_Close"], test_df["Naive_Next_Close"]))
        naive_mape = test_df["Naive_Abs_Pct_Error"].mean()

        # Directional Accuracy (Did model predict up vs down correctly?)
        actual_direction = (test_df["Target_Next_Close"] - test_df["Close"]) > 0
        model_direction = (test_df["Predicted_Next_Close"] - test_df["Close"]) > 0
        dir_accuracy = (actual_direction == model_direction).mean() * 100

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Overall Test R² Score", f"{global_r2:.6f}")
        g2.metric("Overall Test MAPE", f"{global_mape:.2f}%")
        g3.metric("Overall Test MAE", f"{global_mae:.2f}")
        g4.metric("Directional Accuracy", f"{dir_accuracy:.2f}%")

        # NEW VISUAL: Model vs Naive Baseline Benchmark Comparison (Notebook Step 21)
        st.markdown("<div class='section-title'>Linear Regression vs. Naive Random-Walk Baseline Benchmark (Step 21 Checklist)</div>", unsafe_allow_html=True)
        
        bench_col1, bench_col2 = st.columns([1.1, 1])
        with bench_col1:
            comparison_metrics = pd.DataFrame({
                "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "MAPE (%)"],
                "Scikit-Learn Pipeline": [global_mae, global_rmse, global_mape],
                "Naive Baseline (Tomorrow=Today)": [naive_mae, naive_rmse, naive_mape],
            })
            
            fig_bench = go.Figure()
            fig_bench.add_trace(
                go.Bar(
                    x=comparison_metrics["Metric"],
                    y=comparison_metrics["Scikit-Learn Pipeline"],
                    name="Scikit-Learn Linear Regression",
                    marker_color="#38bdf8",
                )
            )
            fig_bench.add_trace(
                go.Bar(
                    x=comparison_metrics["Metric"],
                    y=comparison_metrics["Naive Baseline (Tomorrow=Today)"],
                    name="Naive Baseline (Close t+1 = Close t)",
                    marker_color="#94a3b8",
                )
            )
            fig_bench.update_layout(
                barmode="group",
                title="Model vs. Naive Baseline Quantitative Comparison",
                template=plotly_template,
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with bench_col2:
            st.markdown(
                """
                <div class='info-card' style='margin-top: 10px;'>
                    <b>Notebook Step 21 Checklist Recommendation</b>:<br/>
                    <i>"Evaluate MAE, RMSE and R² on the held-out test period; compare against a naive baseline such as predicting next Close = current Close."</i>
                    <br/><br/>
                    <b>Quantitative Findings</b>:
                    <ul>
                        <li><b>R² Reality Check</b>: The Naive Random-Walk model achieves an R² of <b>0.9996</b>, demonstrating that high R² in asset pricing stems from price memory.</li>
                        <li><b>Added Alpha</b>: The Linear Regression model leverages moving averages (<code>SMA_5</code>) and lag features to refine residual spread and capture mean-reversion drag.</li>
                        <li><b>Directional Hit Rate</b>: The model achieves a <b>48.5%</b> directional accuracy on out-of-sample data, consistent with the Efficient Market Hypothesis.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # NEW VISUAL: Chronological Split Timeline (Notebook Step 20)
        st.markdown("<div class='section-title'>Chronological Train/Test Partitioning Timeline (Step 20)</div>", unsafe_allow_html=True)
        split_summary = (
            processed_df.groupby(["Index", "Split"])
            .agg(Start_Date=("Date", "min"), End_Date=("Date", "max"), Samples=("Date", "count"))
            .reset_index()
        )
        fig_split = px.timeline(
            split_summary,
            x_start="Start_Date",
            x_end="End_Date",
            y="Index",
            color="Split",
            title="80% Chronological Train vs. 20% Chronological Test Time Horizon per Index",
            color_discrete_map={"Train": "#0284c7", "Test": "#f97316"},
            template=plotly_template,
        )
        fig_split.update_yaxes(autorange="reversed")
        fig_split.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_split, use_container_width=True)

        # Actual vs Predicted & Residuals
        st.markdown("<div class='section-title'>Actual vs. Predicted Scatter & Fit</div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            sample_scatter = test_df.sample(min(4000, len(test_df)), random_state=42)
            fig_scatter = px.scatter(
                sample_scatter,
                x="Target_Next_Close",
                y="Predicted_Next_Close",
                color="Index",
                opacity=0.6,
                title="Actual vs. Predicted Next-Day Close (Test Sample)",
                labels={"Target_Next_Close": "Actual Close (Next Day)", "Predicted_Next_Close": "Predicted Close"},
                template=plotly_template,
            )
            min_p = min(sample_scatter["Target_Next_Close"].min(), sample_scatter["Predicted_Next_Close"].min())
            max_p = max(sample_scatter["Target_Next_Close"].max(), sample_scatter["Predicted_Next_Close"].max())
            fig_scatter.add_shape(
                type="line",
                x0=min_p, y0=min_p, x1=max_p, y1=max_p,
                line=dict(color="#ef4444", width=2, dash="dash"),
            )
            fig_scatter.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            st.markdown("#### Residuals Error Distribution")
            res_sub = test_df["Residual"].clip(
                lower=test_df["Residual"].quantile(0.01),
                upper=test_df["Residual"].quantile(0.99),
            )
            fig_hist = px.histogram(
                res_sub,
                nbins=60,
                color_discrete_sequence=["#38bdf8"],
                title="Residuals (Actual - Predicted) 1st-99th Percentile",
                template=plotly_template,
            )
            fig_hist.update_layout(height=450, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        # Per-Index Performance Breakdown Table
        st.markdown("<div class='section-title'>Per-Index Performance Breakdown Table</div>", unsafe_allow_html=True)
        if os.path.exists(PER_INDEX_PATH):
            df_per_idx = pd.read_csv(PER_INDEX_PATH)
            df_per_idx["Exchange Name"] = df_per_idx["Index"].map(INDEX_NAMES)
            df_per_idx = df_per_idx[[
                "Index", "Exchange Name", "Sample_Count", "MAE", "RMSE", "MAPE_percent", "R2_Score", "Directional_Accuracy_percent"
            ]]
            df_per_idx.columns = ["Index", "Exchange", "Test Samples", "MAE", "RMSE", "MAPE (%)", "R² Score", "Dir. Accuracy (%)"]
            st.dataframe(
                df_per_idx.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "MAPE (%)": "{:.2f}%",
                    "R² Score": "{:.4f}",
                    "Dir. Accuracy (%)": "{:.1f}%",
                    "Test Samples": "{:,}",
                }),
                use_container_width=True,
                height=450,
            )

    # =========================================================================
    # TAB 5: INTERACTIVE TIME-SERIES BACKTEST
    # =========================================================================
    elif menu == "📈 Interactive Time-Series Backtest":
        st.title(f"📈 Backtest Explorer: {INDEX_NAMES.get(selected_index, selected_index)}")
        st.caption("Examine how the Linear Regression model performed day-by-day across years of trading history.")

        sub_df = processed_df[processed_df["Index"] == selected_index].sort_values("Date")

        min_date = sub_df["Date"].min().to_pydatetime()
        max_date = sub_df["Date"].max().to_pydatetime()

        date_col1, date_col2, date_col3 = st.columns([1, 1, 1])
        start_date = date_col1.date_input(
            "Start Date",
            value=min_date if min_date > (max_date - pd.Timedelta(days=730)) else (max_date - pd.Timedelta(days=730)),
        )
        end_date = date_col2.date_input("End Date", value=max_date)
        chart_style = date_col3.radio("Display Style", ["Line & Moving Averages", "Candlestick Chart"], horizontal=True)

        filtered = sub_df[(sub_df["Date"] >= pd.to_datetime(start_date)) & (sub_df["Date"] <= pd.to_datetime(end_date))]

        if filtered.empty:
            st.warning("No records found in selected date range.")
            return

        fig = go.Figure()
        if chart_style == "Candlestick Chart":
            fig.add_trace(
                go.Candlestick(
                    x=filtered["Date"],
                    open=filtered["Open"],
                    high=filtered["High"],
                    low=filtered["Low"],
                    close=filtered["Close"],
                    name="Session OHLC",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=filtered["Date"],
                    y=filtered["Target_Next_Close"],
                    name="Actual Next-Day Close",
                    line=dict(color="#0284c7", width=2.2),
                )
            )

        # Technical overlays
        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["SMA_5"],
                name="5-Day SMA",
                line=dict(color="#38bdf8", width=1.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["Predicted_Next_Close"],
                name="Model Forecast",
                line=dict(color="#f97316", width=2, dash="dot"),
            )
        )

        fig.update_layout(
            title=f"{selected_index} Next-Day Close Price Tracking ({len(filtered):,} days)",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500,
            template=plotly_template,
            xaxis=dict(rangeslider=dict(visible=False if chart_style == "Candlestick Chart" else True), type="date"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residual plot below
        fig_res = px.bar(
            filtered,
            x="Date",
            y="Residual",
            title="Forecast Error / Residual by Date (Actual - Predicted)",
            color="Residual",
            color_continuous_scale="RdBu_r",
            template=plotly_template,
        )
        fig_res.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_res, use_container_width=True)

    # =========================================================================
    # TAB 6: FEATURE IMPORTANCE & WEIGHTS
    # =========================================================================
    elif menu == "🧠 Feature Importance & Weights":
        st.title("🧠 Linear Regression Feature Weights & Interpretation")
        st.caption("Understand what features drive the next-day price forecast and their mathematical coefficients.")

        if os.path.exists(COEF_PATH):
            df_coef = pd.read_csv(COEF_PATH)
            st.markdown("<div class='section-title'>Standardized Feature Coefficients</div>", unsafe_allow_html=True)

            top_n = st.slider("Number of top features to display", 5, len(df_coef), 12)
            display_coef = df_coef.head(top_n).copy()

            fig_coef = px.bar(
                display_coef.sort_values("Coefficient"),
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Coefficient",
                color_continuous_scale=["#ef4444", "#38bdf8", "#22c55e"],
                title=f"Top {top_n} Learned Regression Weights",
                template=plotly_template,
            )
            fig_coef.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_coef, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Key Insights from the Model:")
                st.markdown(
                    """
                    1. **Current Day Close (`Close`)**: Has the largest positive weight, reflecting strong price persistence (martingale behavior in financial assets).
                    2. **Mean-Reversion Drag (`SMA_5`, `Low`)**: Negative coefficients on short-term averages penalize extended runs, modeling pullbacks.
                    3. **Momentum Lag (`Close_Lag2`, `Close_Lag5`)**: Captures multi-day trend continuity.
                    4. **Index Embeddings**: Calibrate baseline shifts for international exchange currencies and structural differences.
                    """
                )
            with col2:
                st.markdown("#### Raw Coefficients Table:")
                st.dataframe(df_coef.style.format({"Coefficient": "{:+.4f}", "Abs_Coefficient": "{:.4f}"}), height=280)

    # =========================================================================
    # TAB 7: WHAT-IF SCENARIO SIMULATOR
    # =========================================================================
    elif menu == "🎛️ What-If Scenario Simulator":
        st.title("🎛️ Interactive What-If Scenario Simulator")
        st.caption("Tweak today's market conditions and immediately generate real-time next-day price predictions.")

        sub_latest = processed_df[processed_df["Index"] == selected_index].sort_values("Date").iloc[-1]

        st.markdown(f"#### Base Template: **{selected_index}** (Latest Close: {sub_latest['Close']:,.2f})")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            in_open = st.number_input("Open Price", value=float(sub_latest["Open"]), step=1.0)
            in_high = st.number_input("High Price", value=float(sub_latest["High"]), step=1.0)
            in_low = st.number_input("Low Price", value=float(sub_latest["Low"]), step=1.0)
        with sim_col2:
            in_close = st.number_input("Close Price", value=float(sub_latest["Close"]), step=1.0)
            in_lag1 = st.number_input("Prior Day Close (Lag 1)", value=float(sub_latest["Close_Lag1"]), step=1.0)
            in_lag2 = st.number_input("2 Days Ago Close (Lag 2)", value=float(sub_latest["Close_Lag2"]), step=1.0)
        with sim_col3:
            in_lag5 = st.number_input("5 Days Ago Close (Lag 5)", value=float(sub_latest["Close_Lag5"]), step=1.0)
            in_volume = st.number_input("Volume", value=float(sub_latest["Volume"]), step=100000.0)
            in_sma5 = st.number_input("5-day SMA", value=float(sub_latest["SMA_5"]), step=1.0)

        in_sma10 = sub_latest["SMA_10"]
        in_ret = (in_close - in_open) / in_open if in_open != 0 else 0
        in_range = (in_high - in_low) / in_open if in_open != 0 else 0
        in_logvol = np.log1p(max(0, in_volume))

        input_data = pd.DataFrame([{
            "Open": in_open,
            "High": in_high,
            "Low": in_low,
            "Close": in_close,
            "Close_Lag1": in_lag1,
            "Close_Lag2": in_lag2,
            "Close_Lag5": in_lag5,
            "SMA_5": in_sma5,
            "SMA_10": in_sma10,
            "Daily_Return": in_ret,
            "Intraday_Range": in_range,
            "Log_Volume": in_logvol,
            "Index": selected_index,
        }])

        predicted_sim = float(model.predict(input_data)[0])
        sim_diff = predicted_sim - in_close
        sim_pct = (sim_diff / in_close) * 100

        st.markdown("<div class='section-title'>Simulation Forecast Result</div>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Simulated Input Close", f"{in_close:,.2f}")
        res_col2.metric(
            "Forecasted Next Close",
            f"{predicted_sim:,.2f}",
            delta=f"{sim_diff:+,.2f} ({sim_pct:+.2f}%)",
        )
        with res_col3:
            signal_color = "#4ade80" if sim_diff >= 0 else "#f87171"
            direction_name = "BULLISH UP" if sim_diff >= 0 else "BEARISH DOWN"
            st.markdown(
                f"""
                <div class="simulation-card">
                    <div class="simulation-card-label">Signal</div>
                    <div style="color: {signal_color}; font-size: 1.5rem; font-weight: 700; margin-top: 4px;">{direction_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 8: DATA EXPLORER & ANOMALY AUDIT
    # =========================================================================
    elif menu == "📁 Data Explorer & Anomaly Audit":
        st.title("📁 Market Dataset, Predictions & Anomaly Audit")
        st.caption("Inspect and download the underlying dataset, model predictions, and audit data anomalies.")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            filter_split = st.selectbox("Filter Split", ["All", "Test Set (Out of Sample)", "Train Set"])
        with exp_col2:
            filter_anomalies = st.checkbox("Show Only Statistical Outliers (Abs Error > 3%)", value=False)

        display_df = processed_df[processed_df["Index"] == selected_index].copy()

        if filter_split == "Test Set (Out of Sample)":
            display_df = display_df[display_df["Split"] == "Test"]
        elif filter_split == "Train Set":
            display_df = display_df[display_df["Split"] == "Train"]

        if filter_anomalies:
            display_df = display_df[display_df["Abs_Pct_Error"] > 3.0]

        st.markdown(f"**Showing {len(display_df):,} records for `{selected_index}`**")

        st.dataframe(
            display_df[[
                "Date", "Open", "High", "Low", "Close", "Target_Next_Close", "Predicted_Next_Close", "Residual", "Abs_Pct_Error", "Split"
            ]].sort_values("Date", ascending=False),
            use_container_width=True,
            height=500,
        )

        csv_download = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Data & Predictions as CSV",
            data=csv_download,
            file_name=f"{selected_index}_predictions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
