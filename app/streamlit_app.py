import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sys
import subprocess
import time
import atexit
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.feature_pipeline.feature_store import get_feature_store

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
    
    # Check if API is already running
    for attempt in range(3):
        if _is_api_running():
            return  # API already running
        time.sleep(0.5)
    
    try:
        flask_script = Path(__file__).parent / "flask_api.py"
        log_file = Path(__file__).parent.parent / "flask_api.log"
        
        if sys.platform == "win32":
            # Windows: spawn with detached process
            _flask_process = subprocess.Popen(
                [sys.executable, str(flask_script)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Unix: spawn with nohup
            _flask_process = subprocess.Popen(
                [sys.executable, str(flask_script)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        
        # Wait for API to be ready
        for attempt in range(20):
            time.sleep(0.5)
            if _is_api_running():
                return
    except Exception as e:
        pass  # Silent fail—user can start API manually if needed

def _cleanup_flask():
    """Kill Flask process on Streamlit exit."""
    global _flask_process
    if _flask_process is not None:
        try:
            if sys.platform == "win32":
                os.killpg(os.getpgid(_flask_process.pid), 9)
            else:
                os.killpg(os.getpgid(_flask_process.pid), 9)
        except Exception:
            pass

# Auto-launch API on startup
_launch_flask_api()
atexit.register(_cleanup_flask)

st.set_page_config(page_title='Pearls AQI Predictor - Lahore', layout='wide', page_icon='🌍')

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #071421 0%, #0d1b2a 100%);
            color: #eaf2ff;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(12, 19, 29, 0.95);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .css-1d391kg, .css-18e3th9 {
            background: transparent;
        }
        .metric-container {
            background: rgba(20, 30, 44, 0.75);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
        .title-badge {
            display: inline-block;
            margin-bottom: 0.25rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(48, 166, 255, 0.15);
            border: 1px solid rgba(48, 166, 255, 0.35);
            color: #72d5ff;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .main-header {
            font-size: 2.45rem;
            font-weight: 800;
            margin-bottom: 0rem;
            color: #f4f8ff;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #b7c7dd;
            margin-top: 0.2rem;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: rgba(32, 64, 89, 0.72);
            border: 1px solid rgba(120, 205, 255, 0.25);
            color: #dfefff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=1800)
def fetch_api(endpoint, params=None):
    url = f"{API_URL}{endpoint}"
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.warning(
            f"Failed to fetch from API ({url}). "
            "Make sure the Flask backend is running or set AQI_API_URL to the correct server address."
        )
        return None

def fallback_current_data():
    try:
        df = get_feature_store().get_latest_features(n_hours=24)
    except Exception as exc:
        st.error(f"Supabase is unavailable: {exc}")
        return None

    if df is not None and not df.empty:
        valid_df = df[df[config.TARGET].notna()] if config.TARGET in df.columns else df
        row = valid_df.iloc[-1].to_dict() if not valid_df.empty else df.iloc[-1].to_dict()
        aqi = row.get(config.TARGET, 160.0)
        cat = config.get_aqi_category(aqi)
        return {
            'us_aqi': aqi,
            'category': cat['label'],
            'color': cat['color'],
            'emoji': cat['emoji'],
            'health_advisory': config.get_health_advisory(aqi),
            'temperature_2m': row.get('temperature_2m', 25.0),
            'relative_humidity_2m': row.get('relative_humidity_2m', 60.0),
            'wind_speed_10m': row.get('wind_speed_10m', 10.0),
            'wind_direction_10m': row.get('wind_direction_10m', 180),
        }
    return None

def main():
    # ------------------ Sidebar ------------------
    st.sidebar.title("Configuration")
    model_selector = st.sidebar.selectbox("Model Selector", ["XGBoost", "Random Forest", "Ridge"])
    history_days = st.sidebar.slider("Historical View (Days)", min_value=7, max_value=90, value=30)
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        
    st.sidebar.markdown("### About")
    st.sidebar.info("Pearls AQI Predictor forecasts Air Quality Index for Lahore, Pakistan using ML models.")
    
    # ------------------ Header Section ------------------
    st.markdown("<div class='title-badge'>AQI Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-header'>Pearls AQI Predictor — Lahore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time Air Quality Monitoring & 3-Day Forecast</div>", unsafe_allow_html=True)
    
    current_data = fetch_api("/current")
    if current_data is None:
        current_data = fallback_current_data()
        
    if current_data:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ------------------ Row 1: Current Conditions ------------------
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label=f"Current AQI ({current_data.get('category', 'Unknown')} {current_data.get('emoji', '')})",
                value=f"{current_data.get('us_aqi', 0):.1f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            temp = current_data.get('temperature_2m', 0)
            hum = current_data.get('relative_humidity_2m', 0)
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(label="Temperature / Humidity", value=f"{temp}°C / {hum}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            wind = current_data.get('wind_speed_10m', 0)
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(label="Wind Speed", value=f"{wind} km/h")
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            st.write(current_data.get('health_advisory', 'No specific health advisory.'))
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.divider()

    # ------------------ Row 2: 3-Day Forecast (Individual Days) ------------------
    st.subheader(f"3-Day Forecast (Model: {forecast_data.get('model_used', 'N/A').upper()})")
    
    if forecast_data and 'daily_summary' in forecast_data and forecast_data['daily_summary']:
        days = forecast_data['daily_summary'][:3]
        cols = st.columns(len(days))
        
        for idx, (col, day) in enumerate(zip(cols, days)):
            with col:
                day_num = idx + 1
                date_str = day.get('date', '')
                day_name = day.get('day_name') or (pd.to_datetime(date_str).strftime('%A, %b %d') if date_str else f"Day {day_num}")
                avg_aqi = float(day.get('avg_aqi', 0))
                min_aqi = float(day.get('min_aqi', 0))
                max_aqi = float(day.get('max_aqi', 0))
                category = day.get('category', 'Unknown')
                cat_info = config.get_aqi_category(avg_aqi)
                color = day.get('color') or cat_info['color']
                emoji = day.get('emoji') or cat_info['emoji']
                advisory = day.get('advisory') or cat_info.get('health_advisory') or config.get_health_advisory(avg_aqi)

                day_label = "Day 1" if day_num == 1 else f"Day {day_num}"

                st.markdown(
                    f"""
                    <div style="
                        background: rgba(20, 30, 44, 0.85);
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-top: 5px solid {color};
                        border-radius: 14px;
                        padding: 1.2rem;
                        min-height: 250px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                        margin-bottom: 1rem;
                    ">
                        <div>
                            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                                {day_label} • {day_name}
                            </div>
                            <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-top: 0.5rem;">
                                <span style="font-size: 2.3rem; font-weight: 800; color: #f8fafc;">{avg_aqi:.0f}</span>
                                <span style="font-size: 1rem; color: #94a3b8; font-weight: 600;">AQI</span>
                            </div>
                            <div style="
                                display: inline-block;
                                background: {color}22;
                                border: 1px solid {color}55;
                                color: {color};
                                font-size: 0.8rem;
                                font-weight: 700;
                                padding: 0.25rem 0.6rem;
                                border-radius: 999px;
                                margin-top: 0.25rem;
                            ">
                                {emoji} {category}
                            </div>
                            <div style="display: flex; gap: 1rem; margin-top: 0.9rem; font-size: 0.85rem; color: #cbd5e1;">
                                <span>🔻 Min: <strong style="color: #f8fafc;">{min_aqi:.0f}</strong></span>
                                <span>🔺 Max: <strong style="color: #f8fafc;">{max_aqi:.0f}</strong></span>
                            </div>
                        </div>
                        <div style="
                            margin-top: 1rem;
                            padding: 0.75rem;
                            border-radius: 8px;
                            background: rgba(0, 0, 0, 0.25);
                            border-left: 3px solid {color};
                            font-size: 0.8rem;
                            color: #e2e8f0;
                            line-height: 1.4;
                        ">
                            <strong>Advisory:</strong> {advisory}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("Forecast data is currently unavailable.")

    # ------------------ Row 3: History & Pollutants ------------------
    col_hist, col_poll = st.columns(2)
    
    history_data = fetch_api(f"/history?days={history_days}")
    if history_data:
        df_hist = pd.DataFrame(history_data)
        if 'timestamp' in df_hist.columns:
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
            
            with col_hist:
                st.subheader(f"Historical Trend (Last {history_days} Days)")
                df_hist_daily = df_hist.resample('D', on='timestamp').mean().reset_index()
                df_hist_daily['rolling_avg'] = df_hist_daily[config.TARGET].rolling(3).mean()
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_hist_daily['timestamp'], y=df_hist_daily[config.TARGET], mode='lines', name='Daily Mean AQI'))
                fig2.add_trace(go.Scatter(x=df_hist_daily['timestamp'], y=df_hist_daily['rolling_avg'], mode='lines', name='3-Day Rolling Avg', line=dict(dash='dash')))
                st.plotly_chart(fig2, width='stretch')
                
            with col_poll:
                st.subheader("Current Pollutant Levels")
                if current_data:
                    pollutants = {p: current_data.get(p, 0) for p in config.POLLUTANT_FEATURES if p in current_data}
                    if pollutants:
                        df_p = pd.DataFrame(list(pollutants.items()), columns=['Pollutant', 'Level'])
                        fig3 = px.bar(df_p, x='Pollutant', y='Level', color='Pollutant', title="Major Pollutants")
                        st.plotly_chart(fig3, width='stretch')

    # ------------------ Row 4: Performance & Explainability ------------------
    st.divider()
    col_perf, col_exp = st.columns(2)
    
    with col_perf:
        st.subheader("Model Performance Comparison")
        metrics_list = fetch_api("/models")
        if metrics_list and isinstance(metrics_list, list) and len(metrics_list) > 0:
            table_rows = []
            for item in metrics_list:
                m_name = item.get("model_name", "Unknown").upper()
                m_metrics = item.get("metrics", {})
                rmse_val = m_metrics.get('rmse')
                mae_val = m_metrics.get('mae')
                r2_val = m_metrics.get('r2')
                table_rows.append({
                    "Model": m_name,
                    "RMSE": f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else "—",
                    "MAE": f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else "—",
                    "R²": f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "—",
                })
            if table_rows:
                df_metrics = pd.DataFrame(table_rows)
                st.dataframe(df_metrics, width='stretch')
            else:
                st.info("No trained models available yet.")
        else:
            st.info("Metrics not available.")
            
    with col_exp:
        st.subheader("Feature Importance (SHAP)")
        shap_data = fetch_api("/explain")
        if shap_data and isinstance(shap_data, list):
            df_shap = pd.DataFrame(shap_data)
            if "Importance" in df_shap.columns and "Feature" in df_shap.columns:
                df_shap = df_shap.sort_values(by="Importance", ascending=True).tail(15)
                fig4 = px.bar(df_shap, x="Importance", y="Feature", orientation='h', title="Top 15 Predictive Features")
                st.plotly_chart(fig4, width='stretch')
            else:
                st.info("Feature importance data formatting not recognized.")
        else:
            st.info("SHAP explainability not available yet.")
            
    # ------------------ Row 5: Alerts Panel ------------------
    st.divider()
    alerts = fetch_api("/alerts")
    if alerts:
        st.error(f"⚠️ {len(alerts)} Hazardous AQI Periods Forecasted!")
        for alert in alerts:
            st.warning(f"**Time:** {alert['timestamp']} | **AQI:** {alert['aqi']} | **Advisory:** {alert['message']}")

if __name__ == "__main__":
    main()
