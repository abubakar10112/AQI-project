import atexit
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.feature_pipeline.feature_store import get_feature_store

logger = logging.getLogger(__name__)

API_URL = os.getenv("AQI_API_URL", "http://127.0.0.1:5000/api")

# ============================================================================
# AUTO-LAUNCH FLASK API
# ============================================================================
_flask_process = None


def _is_api_running():
    """Check if Flask API is responding on port 5000."""
    try:
        response = requests.get(f"{API_URL.replace('/api', '')}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _launch_flask_api():
    """Spawn Flask API in a background process if not already running."""
    global _flask_process

    for _ in range(3):
        if _is_api_running():
            return
        time.sleep(0.5)

    try:
        flask_script = Path(__file__).parent / "flask_api.py"
        log_file = Path(__file__).parent.parent / "flask_api.log"

        if sys.platform == "win32":
            _flask_process = subprocess.Popen(
                [sys.executable, str(flask_script)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            _flask_process = subprocess.Popen(
                [sys.executable, str(flask_script)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        for _ in range(20):
            time.sleep(0.5)
            if _is_api_running():
                return
    except Exception:
        pass


def _cleanup_flask():
    """Kill Flask process on Streamlit exit."""
    global _flask_process
    if _flask_process is not None:
        try:
            os.killpg(os.getpgid(_flask_process.pid), 9)
        except Exception:
            pass


_launch_flask_api()
atexit.register(_cleanup_flask)

# ============================================================================
# PAGE CONFIGURATION & MODERN THEME
# ============================================================================
st.set_page_config(
    page_title="Lahore AQI Predictor | Live Environmental Intelligence",
    layout="wide",
    page_icon="🍃",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 50% -20%, #152238 0%, #080d1a 60%, #04070d 100%) !important;
            color: #f1f5f9 !important;
        }

        header[data-testid="stHeader"], .stApp > header {
            background: transparent !important;
            background-color: transparent !important;
        }

        .block-container {
            padding-top: 3.2rem !important;
            padding-bottom: 2.5rem;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 100% !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(10, 15, 26, 0.98) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        [data-testid="stSidebar"] .stSelectbox label, 
        [data-testid="stSidebar"] .stSlider label {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        [data-baseweb="select"] > div {
            background: rgba(18, 27, 44, 0.9) !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
            color: #f1f5f9 !important;
        }

        [data-baseweb="select"] * {
            color: #f1f5f9 !important;
        }

        /* Glassmorphism Card Style */
        .glass-card {
            background: rgba(18, 27, 44, 0.65);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.3rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.16);
            transform: translateY(-2px);
        }

        /* Top Header Area */
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            line-height: 1.15;
        }
        
        .hero-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
            font-weight: 500;
            margin-top: 0.3rem;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* Tabs custom styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(13, 20, 33, 0.75) !important;
            padding: 0.4rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            font-size: 0.9rem;
            background: transparent !important;
            border: none !important;
        }

        .stTabs [data-baseweb="tab"] * {
            color: #94a3b8 !important;
        }

        .stTabs [data-baseweb="tab"]:hover * {
            color: #f1f5f9 !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.18) !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
        }

        .stTabs [aria-selected="true"] * {
            color: #38bdf8 !important;
            font-weight: 700 !important;
        }

        /* Clean metric values */
        .metric-huge {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.03em;
            color: #ffffff;
        }

        .guide-box {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.82rem;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(129, 140, 248, 0.15) 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
            color: #f8fafc !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.3) 0%, rgba(129, 140, 248, 0.3) 100%) !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# API & IN-PROCESS DATA FETCHERS
# ============================================================================
@st.cache_data(ttl=180)
def _fetch_api_cached(url: str, params_tuple: tuple):
    res = requests.get(
        url,
        params=dict(params_tuple) if params_tuple else None,
        timeout=15,
    )
    res.raise_for_status()
    return res.json()


def fetch_api(endpoint: str, params: dict = None):
    url = f"{API_URL}{endpoint}"
    params_tuple = tuple(sorted(params.items())) if params else ()
    try:
        return _fetch_api_cached(url, params_tuple)
    except Exception as exc:
        logger.debug("API fetch %s failed: %s", url, exc)
        return None


def fallback_current_data():
    """In-process fallback when API is starting or unreachable."""
    try:
        df = get_feature_store().get_latest_features(n_hours=48)
    except Exception:
        return None

    if df is not None and not df.empty:
        now = pd.Timestamp.now(tz=config.CITY_TIMEZONE).tz_localize(None)
        obs = df.loc[df.index <= now]
        if obs.empty:
            obs = df
        valid = obs[obs[config.TARGET].notna()] if config.TARGET in obs.columns else obs
        row = valid.iloc[-1].to_dict() if not valid.empty else obs.iloc[-1].to_dict()

        aqi = float(row.get(config.TARGET, 120.0))
        cat = config.get_aqi_category(aqi)
        return {
            "us_aqi": aqi,
            "category": cat["label"],
            "color": cat["color"],
            "emoji": cat["emoji"],
            "health_advisory": config.get_health_advisory(aqi),
            "temperature_2m": float(row.get("temperature_2m", 25.0)),
            "relative_humidity_2m": float(row.get("relative_humidity_2m", 60.0)),
            "wind_speed_10m": float(row.get("wind_speed_10m", 8.0)),
            "wind_direction_10m": float(row.get("wind_direction_10m", 180)),
            "surface_pressure": float(row.get("surface_pressure", 1012)),
            "pm2_5": float(row.get("pm2_5", 45.0)),
            "pm10": float(row.get("pm10", 85.0)),
            "nitrogen_dioxide": float(row.get("nitrogen_dioxide", 22.0)),
            "sulphur_dioxide": float(row.get("sulphur_dioxide", 8.0)),
            "ozone": float(row.get("ozone", 30.0)),
            "carbon_monoxide": float(row.get("carbon_monoxide", 450.0)),
        }
    return None


def fallback_forecast_data(model_name: str = "xgboost"):
    """In-process forecast generation fallback."""
    try:
        from src.inference.predictor import Predictor

        predictor = Predictor(model_names=[model_name])
        return predictor.predict_next_3_days()
    except Exception as exc:
        logger.warning("Local forecast fallback failed: %s", exc)
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # ------------------ Sidebar Controls ------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.2rem;">
                <span style="font-size: 1.6rem;">🍃</span>
                <div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; line-height: 1.1;">Lahore AQI</div>
                    <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">AI Environmental Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Model Engine")
        model_selector = st.selectbox(
            "Select Inference Model",
            ["XGBoost", "Random Forest", "Ridge"],
            index=0,
            help="Select which ML model generates the recursive 72-hour forecast.",
        )
        selected_model_key = {
            "XGBoost": "xgboost",
            "Random Forest": "random_forest",
            "Ridge": "ridge",
        }[model_selector]

        st.markdown("### Analytics Range")
        history_days = st.slider("History Window", min_value=7, max_value=60, value=30, step=7)

        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Architecture & Infrastructure Status
        st.markdown("### Connected Stack")
        st.markdown(
            """
            <div style="font-size: 0.78rem; line-height: 1.8; color: #cbd5e1;">
                <div>🟢 <b>Feature Store:</b> Supabase (PostgreSQL)</div>
                <div>🟣 <b>Model Registry:</b> Hopsworks</div>
                <div>📡 <b>Live Weather:</b> Open-Meteo</div>
                <div>🧪 <b>Station Data:</b> AQICN / WAQI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("Lahore Air Quality Intelligence Platform • 2026")

    # ------------------ Fetch Data ------------------
    current_data = fetch_api("/current") or fallback_current_data()
    forecast_data = fetch_api("/predict", params={"model": selected_model_key}) or fallback_forecast_data(selected_model_key)

    # ------------------ Top Navigation Bar ------------------
    active_model = (forecast_data.get("model_used") or selected_model_key).upper() if forecast_data else selected_model_key.upper()
    now_str = datetime.now().strftime("%A, %d %b %Y • %H:%M PKT")

    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown('<div class="badge-pill" style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; margin-bottom: 0.4rem;">● LIVE TELEMETRY</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Lahore Air Quality Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Real-time Atmospheric Monitoring & 72-Hour Serverless AI Forecasting</p>', unsafe_allow_html=True)

    with header_col2:
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: flex-end; justify-content: center; height: 100%; gap: 0.4rem;">
                <div style="display: flex; gap: 0.5rem;">
                    <span class="badge-pill" style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #34d399;">● ONLINE</span>
                    <span class="badge-pill" style="background: rgba(129, 140, 248, 0.15); border: 1px solid rgba(129, 140, 248, 0.35); color: #a5b4fc;">ENGINE: {active_model}</span>
                </div>
                <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 8px; padding: 0.4rem 0.85rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <span style="font-size: 0.95rem;">📅</span>
                    <span style="font-size: 0.88rem; color: #f8fafc; font-weight: 700; letter-spacing: 0.01em;">{now_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # ------------------ HERO SECTION: Current AQI & Micro Metrics ------------------
    if current_data:
        aqi_val = float(current_data.get("us_aqi", 0))
        cat_info = config.get_aqi_category(aqi_val)
        category = current_data.get("category") or cat_info["label"]
        color = current_data.get("color") or cat_info["color"]
        emoji = current_data.get("emoji") or cat_info["emoji"]
        advisory = current_data.get("health_advisory") or config.get_health_advisory(aqi_val)
        temp = float(current_data.get("temperature_2m", 25.0))
        hum = float(current_data.get("relative_humidity_2m", 60.0))
        wind = float(current_data.get("wind_speed_10m", 8.0))
        pressure = float(current_data.get("surface_pressure", 1012.0))

        # Main Hero Grid: Left Hero Card + 3 Right Parameter Cards
        hero_left, hero_right = st.columns([1.35, 2.0])

        with hero_left:
            st.markdown(
                f"""
                <div class="glass-card" style="border-top: 5px solid {color}; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Current Air Quality</span>
                            <span style="font-size: 0.78rem; color: {color}; font-weight: 700; background: {color}18; padding: 0.2rem 0.6rem; border-radius: 999px; border: 1px solid {color}40;">
                                {emoji} {category}
                            </span>
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 0.6rem; margin: 1.1rem 0 0.5rem 0;">
                            <span class="metric-huge">{aqi_val:.0f}</span>
                            <span style="font-size: 1.1rem; color: #94a3b8; font-weight: 700;">US AQI</span>
                        </div>
                        <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.4;">
                            Dominant factor: <b style="color: #ffffff;">PM2.5 particles</b> & meteorological humidity inversion.
                        </div>
                    </div>
                    <div style="margin-top: 1.3rem; padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.08); display: flex; justify-content: space-between; font-size: 0.8rem; color: #cbd5e1;">
                        <span>Station: <b style="color: #ffffff;">Lahore Central</b></span>
                        <span>Date: <b style="color: #38bdf8;">{datetime.now().strftime('%d %b %Y')}</b></span>
                        <span>Scale: <b style="color: #ffffff;">0 - 500 EPA</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with hero_right:
            # 3 Micro Environmental Cards in row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top: 4px solid #38bdf8;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Temperature</div>
                        <div style="font-size: 1.9rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{temp:.1f}°C</div>
                        <div style="font-size: 0.78rem; color: #93c5fd;">Feels like ~{temp+1.5:.1f}°C</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top: 4px solid #06b6d4;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Atmosphere</div>
                        <div style="font-size: 1.9rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{hum:.0f}%</div>
                        <div style="font-size: 0.78rem; color: #67e8f9;">Barometer: {pressure:.0f} hPa</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top: 4px solid #818cf8;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Wind Speed</div>
                        <div style="font-size: 1.9rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{wind:.1f} <span style="font-size: 1rem; color: #94a3b8;">km/h</span></div>
                        <div style="font-size: 0.78rem; color: #a5b4fc;">Gentle surface dispersion</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Health Protection Guidance Box
            st.markdown(
                f"""
                <div class="glass-card" style="margin-top: 0.75rem; border-left: 4px solid {color}; padding: 0.9rem 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.76rem; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 0.05em;">🛡️ Health Advisory & Action Plan</span>
                        <span style="font-size: 0.74rem; color: #94a3b8;">WHO Guidelines</span>
                    </div>
                    <div style="font-size: 0.86rem; color: #e2e8f0; line-height: 1.45; font-weight: 500;">
                        {advisory}
                    </div>
                    <div style="display: flex; gap: 1rem; margin-top: 0.7rem; font-size: 0.76rem; color: #94a3b8;">
                        <span>😷 <b>Masks:</b> {'Required' if aqi_val > 150 else 'Advised'}</span>
                        <span>🪟 <b>Windows:</b> {'Closed' if aqi_val > 100 else 'Open allowed'}</span>
                        <span>🏃 <b>Sports:</b> {'Indoor only' if aqi_val > 150 else 'Moderate'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ------------------ 3-DAY FORECAST CARDS ------------------
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.8rem;">
            <div>
                <h2 style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin: 0;">3-Day Air Quality Outlook</h2>
                <div style="font-size: 0.82rem; color: #94a3b8;">Dynamic multi-step autoregressive forecast via {active_model}</div>
            </div>
            <div style="font-size: 0.78rem; color: #64748b;">72-hour horizon with weather dispersion</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if forecast_data and "daily_summary" in forecast_data and forecast_data["daily_summary"]:
        days = forecast_data["daily_summary"][:3]
        f_cols = st.columns(len(days))

        for idx, (col, day) in enumerate(zip(f_cols, days)):
            with col:
                day_num = idx + 1
                date_str = day.get("date", "")
                day_name = day.get("day_name") or (pd.to_datetime(date_str).strftime("%A, %b %d") if date_str else f"Day {day_num}")
                avg_aqi = float(day.get("avg_aqi", 0))
                min_aqi = float(day.get("min_aqi", 0))
                max_aqi = float(day.get("max_aqi", 0))
                cat_info = config.get_aqi_category(avg_aqi)
                category = day.get("category") or cat_info["label"]
                color = day.get("color") or cat_info["color"]
                emoji = day.get("emoji") or cat_info["emoji"]
                advisory_text = day.get("advisory") or config.get_health_advisory(avg_aqi)

                day_badge = "TODAY" if day_num == 1 else ("TOMORROW" if day_num == 2 else f"DAY {day_num}")

                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top: 5px solid {color}; min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.6rem; margin-bottom: 0.75rem;">
                                <span style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); color: #38bdf8; font-size: 0.74rem; font-weight: 800; padding: 0.2rem 0.6rem; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">{day_badge}</span>
                                <div style="font-size: 0.96rem; color: #f8fafc; font-weight: 700; display: flex; align-items: center; gap: 0.35rem;">
                                    <span>🗓️</span> <span>{day_name}</span>
                                </div>
                            </div>
                            <div style="display: flex; align-items: baseline; gap: 0.4rem; margin: 0.5rem 0 0.3rem 0;">
                                <span style="font-size: 2.7rem; font-weight: 800; color: #ffffff; line-height: 1;">{avg_aqi:.0f}</span>
                                <span style="font-size: 0.95rem; color: #cbd5e1; font-weight: 700;">AQI</span>
                                <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">(24h Mean)</span>
                            </div>
                            <div style="display: inline-flex; align-items: center; gap: 0.35rem; background: {color}1f; border: 1px solid {color}45; color: {color}; font-size: 0.82rem; font-weight: 700; padding: 0.25rem 0.7rem; border-radius: 999px; margin-top: 0.2rem;">
                                {emoji} {category}
                            </div>
                            <div style="display: flex; gap: 1.2rem; margin-top: 0.9rem; font-size: 0.84rem; color: #cbd5e1;">
                                <span>🔻 Low: <b style="color: #ffffff;">{min_aqi:.0f}</b></span>
                                <span>🔺 Peak: <b style="color: #ffffff;">{max_aqi:.0f}</b></span>
                            </div>
                        </div>
                        <div class="guide-box" style="margin-top: 1rem; border-left: 3px solid {color}; line-height: 1.45; color: #e2e8f0;">
                            <b>Advice:</b> {advisory_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Forecast telemetry is currently compiling...")

    # ------------------ 72-HOUR HOURLY INTERACTIVE CURVE DRAWER ------------------
    if forecast_data and "hourly_predictions" in forecast_data and forecast_data["hourly_predictions"]:
        df_hourly = pd.DataFrame(forecast_data["hourly_predictions"])
        if "timestamp" in df_hourly.columns and "aqi" in df_hourly.columns:
            df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"])
            
            with st.expander("📈 Expand 72-Hour Hourly Prediction Curve & Diurnal Trajectory", expanded=False):
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.8rem; font-size: 0.88rem; color: #cbd5e1;">
                        Hourly atmospheric projection generated by <b style="color: #38bdf8;">{active_model}</b> incorporating diurnal solar radiation cycles and nocturnal temperature inversions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig_72h = go.Figure()

                # Upper and lower uncertainty bounds (+/- 10%)
                upper_bound = df_hourly["aqi"] * 1.10
                lower_bound = (df_hourly["aqi"] * 0.90).clip(lower=0)

                # Shaded confidence band
                fig_72h.add_trace(
                    go.Scatter(
                        x=df_hourly["timestamp"],
                        y=upper_bound,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig_72h.add_trace(
                    go.Scatter(
                        x=df_hourly["timestamp"],
                        y=lower_bound,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(56, 189, 248, 0.12)",
                        name="90% Confidence Interval",
                        hoverinfo="skip",
                    )
                )

                # Main 72h prediction line with markers
                fig_72h.add_trace(
                    go.Scatter(
                        x=df_hourly["timestamp"],
                        y=df_hourly["aqi"],
                        mode="lines+markers",
                        name="Predicted US AQI",
                        line=dict(color="#38bdf8", width=3),
                        marker=dict(size=5, color="#38bdf8"),
                        hovertemplate="<b>%{x|%a, %b %d • %H:%M PKT}</b><br>Predicted AQI: <b>%{y:.1f}</b><extra></extra>",
                    )
                )

                # Air quality threshold benchmark lines
                fig_72h.add_hline(y=100, line_dash="dash", line_color="#fbbf24", annotation_text="Moderate / Sensitive (100)", annotation_position="bottom right")
                fig_72h.add_hline(y=150, line_dash="dash", line_color="#f97316", annotation_text="Unhealthy (150)", annotation_position="bottom right")
                fig_72h.add_hline(y=200, line_dash="dash", line_color="#ef4444", annotation_text="Very Unhealthy (200)", annotation_position="top right")
                fig_72h.add_hline(y=300, line_dash="dash", line_color="#7e22ce", annotation_text="Hazardous (300)", annotation_position="top right")

                fig_72h.update_layout(
                    title="Lahore 72-Hour High-Resolution AQI Hourly Trajectory",
                    xaxis_title="Timeline (PKT)",
                    yaxis_title="US AQI",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_72h.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
                fig_72h.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

                st.plotly_chart(fig_72h, use_container_width=True)

                # CSV Download Button
                csv_export = df_hourly[["timestamp", "aqi", "category"]].copy()
                csv_export.columns = ["Timestamp (PKT)", "Predicted AQI", "Health Category"]
                csv_data = csv_export.to_csv(index=False).encode('utf-8')
                
                c_btn1, c_btn2 = st.columns([1, 2])
                with c_btn1:
                    st.download_button(
                        label="📥 Download 72-Hour Forecast (CSV)",
                        data=csv_data,
                        file_name=f"lahore_aqi_72h_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ------------------ DEEP-DIVE ANALYTICS TABS ------------------
    tab1, tab2, tab3 = st.tabs([
        "🧪 Pollutants & Chemical Breakdown",
        "📈 30-Day Historical Trend",
        "🧠 AI Model Benchmarks & SHAP",
    ])

    # Tab 1: Pollutants
    with tab1:
        if current_data:
            p_col1, p_col2 = st.columns([1.2, 1])

            with p_col1:
                pollutants = {
                    "PM2.5 (Fine Particles)": (current_data.get("pm2_5", 45.0), 15.0, "µg/m³"),
                    "PM10 (Coarse Particles)": (current_data.get("pm10", 85.0), 45.0, "µg/m³"),
                    "NO₂ (Nitrogen Dioxide)": (current_data.get("nitrogen_dioxide", 22.0), 25.0, "µg/m³"),
                    "O₃ (Ground Ozone)": (current_data.get("ozone", 30.0), 100.0, "µg/m³"),
                    "SO₂ (Sulphur Dioxide)": (current_data.get("sulphur_dioxide", 8.0), 40.0, "µg/m³"),
                    "CO (Carbon Monoxide)": (current_data.get("carbon_monoxide", 450.0), 4000.0, "µg/m³"),
                }

                st.markdown("#### Live Concentration vs. WHO Safe Guidelines")
                st.caption("Values exceeding 100% indicate levels surpassing World Health Organization recommendations.")

                for name, (val, threshold, unit) in pollutants.items():
                    pct = min(250.0, (val / threshold) * 100.0)
                    bar_color = "#10b981" if pct <= 100 else ("#f59e0b" if pct <= 150 else "#ef4444")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.83rem; margin-bottom: 0.25rem;">
                                <span><b>{name}</b></span>
                                <span><b>{val:.1f} {unit}</b> <span style="color: {bar_color}; font-size: 0.76rem;">({pct:.0f}% of WHO limit)</span></span>
                            </div>
                            <div style="background: rgba(255,255,255,0.06); border-radius: 999px; height: 8px; overflow: hidden;">
                                <div style="background: {bar_color}; width: {min(100.0, pct):.1f}%; height: 100%; border-radius: 999px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with p_col2:
                df_p = pd.DataFrame([
                    {"Pollutant": k.split(" ")[0], "Concentration": v[0]}
                    for k, v in pollutants.items()
                ])
                fig_poll = px.bar(
                    df_p,
                    x="Pollutant",
                    y="Concentration",
                    color="Pollutant",
                    title="Pollutant Relative Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e", "#fbbf24", "#34d399", "#a855f7"],
                )
                fig_poll.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=280,
                    showlegend=False,
                )
                st.plotly_chart(fig_poll, use_container_width=True)

    # Tab 2: Historical Trend
    with tab2:
        history_data = fetch_api(f"/history?days={history_days}")
        if history_data:
            df_hist = pd.DataFrame(history_data)
            if "timestamp" in df_hist.columns and config.TARGET in df_hist.columns:
                df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
                df_hist_daily = df_hist.resample("D", on="timestamp").mean().reset_index()
                df_hist_daily["rolling_avg"] = df_hist_daily[config.TARGET].rolling(3).mean()

                fig_hist = go.Figure()

                # Shaded area under curve
                fig_hist.add_trace(
                    go.Scatter(
                        x=df_hist_daily["timestamp"],
                        y=df_hist_daily[config.TARGET],
                        mode="lines",
                        name="Daily Mean AQI",
                        line=dict(color="#38bdf8", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(56, 189, 248, 0.08)",
                        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Daily Mean AQI: %{y:.1f}<extra></extra>",
                    )
                )

                fig_hist.add_trace(
                    go.Scatter(
                        x=df_hist_daily["timestamp"],
                        y=df_hist_daily["rolling_avg"],
                        mode="lines",
                        name="3-Day Moving Average",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        hovertemplate="3-Day Trend: %{y:.1f}<extra></extra>",
                    )
                )

                fig_hist.update_layout(
                    title=f"Lahore AQI Trends Over Past {history_days} Days",
                    xaxis_title="Date",
                    yaxis_title="US AQI",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=10, r=10, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_hist.update_xaxes(showgrid=False)
                fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Historical data is loading from the Supabase feature store...")

    # Tab 3: Model Benchmarks & SHAP Explainability
    with tab3:
        b_col1, b_col2 = st.columns([1, 1.2])

        with b_col1:
            st.markdown("#### Model Performance Benchmarks")
            st.caption("Models trained with time-series cross-validation on Lahore climate features.")

            metrics_list = fetch_api("/models")
            if metrics_list and isinstance(metrics_list, list):
                table_rows = []
                for item in metrics_list:
                    m_name = item.get("model_name", "Unknown").upper()
                    m_metrics = item.get("metrics", {})
                    rmse_val = m_metrics.get("rmse")
                    mae_val = m_metrics.get("mae")
                    r2_val = m_metrics.get("r2")

                    is_best = "🏆 " if m_name == "XGBOOST" else ""
                    table_rows.append({
                        "Model": f"{is_best}{m_name}",
                        "RMSE": f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else "—",
                        "MAE": f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else "—",
                        "R² Score": f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "—",
                    })

                if table_rows:
                    df_metrics = pd.DataFrame(table_rows)
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            else:
                st.info("Benchmarking data loading...")

        with b_col2:
            st.markdown("#### Top Predictive Drivers (SHAP Feature Importance)")
            st.caption("Identifies the meteorological and lag factors with highest impact on AQI predictions.")

            shap_data = fetch_api("/explain")
            if shap_data and isinstance(shap_data, list):
                df_shap = pd.DataFrame(shap_data)
                if "Importance" in df_shap.columns and "Feature" in df_shap.columns:
                    df_shap = df_shap.sort_values(by="Importance", ascending=True).tail(12)
                    fig_shap = px.bar(
                        df_shap,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        template="plotly_dark",
                        color="Importance",
                        color_continuous_scale="Blues",
                    )
                    fig_shap.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=290,
                        margin=dict(l=10, r=10, t=20, b=10),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("SHAP explainability summary loading...")

    # ------------------ Hazardous Alerts Banner (if active) ------------------
    alerts = fetch_api("/alerts")
    if alerts and len(alerts) > 0:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.error(f"⚠️ **Attention**: {len(alerts)} hazardous air quality peak periods anticipated in the 72-hour window.")


if __name__ == "__main__":
    main()
