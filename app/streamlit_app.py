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

st.set_page_config(page_title='Lahore AQI Predictor', layout='wide', page_icon='🌍')

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

@st.cache_data(ttl=300)
def _fetch_api_cached(url, params_tuple):
    res = requests.get(url, params=dict(params_tuple) if params_tuple else None, timeout=20)
    res.raise_for_status()
    return res.json()

def fetch_api(endpoint, params=None):
    url = f"{API_URL}{endpoint}"
    params_tuple = tuple(sorted(params.items())) if params else ()
    try:
        return _fetch_api_cached(url, params_tuple)
    except Exception as e:
        logger.debug(f"API fetch {url} failed: {e}")
        return None

def fallback_current_data():
    try:
        df = get_feature_store().get_latest_features(n_hours=48)
    except Exception as exc:
        st.error(f"Supabase is unavailable: {exc}")
        return None

    if df is not None and not df.empty:
        now = pd.Timestamp.now(tz=config.CITY_TIMEZONE).tz_localize(None)
        observed_df = df.loc[df.index <= now]
        if observed_df.empty:
            observed_df = df
        valid_df = observed_df[observed_df[config.TARGET].notna()] if config.TARGET in observed_df.columns else observed_df
        row = valid_df.iloc[-1].to_dict() if not valid_df.empty else observed_df.iloc[-1].to_dict()
        aqi = float(row.get(config.TARGET, 120.0))
        cat = config.get_aqi_category(aqi)
        return {
            'us_aqi': aqi,
            'category': cat['label'],
            'color': cat['color'],
            'emoji': cat['emoji'],
            'health_advisory': config.get_health_advisory(aqi),
            'temperature_2m': float(row.get('temperature_2m', 25.0)),
            'relative_humidity_2m': float(row.get('relative_humidity_2m', 60.0)),
            'wind_speed_10m': float(row.get('wind_speed_10m', 10.0)),
            'wind_direction_10m': float(row.get('wind_direction_10m', 180)),
        }
    return None

def fallback_forecast_data(model_name: str = "xgboost"):
    try:
        from src.inference.predictor import Predictor
        predictor = Predictor(model_names=[model_name])
        return predictor.predict_next_3_days()
    except Exception as exc:
        logger.warning(f"Local forecast fallback failed: {exc}")
        return None

def main():
    # ------------------ Sidebar ------------------
    st.sidebar.title("Configuration")
    model_selector = st.sidebar.selectbox("Model Selector", ["XGBoost", "Random Forest", "Ridge"])
    history_days = st.sidebar.slider("Historical View (Days)", min_value=7, max_value=90, value=30)
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        
    st.sidebar.markdown("### About")
    st.sidebar.info("Lahore AQI Predictor forecasts Air Quality Index for Lahore, Pakistan using ML models.")
    
    # ------------------ Header Section ------------------
    st.markdown("<div class='title-badge'>LIVE AIR QUALITY FORECAST</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-header'>Lahore AQI Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time Air Quality Monitoring & 3-Day Machine Learning Forecast</div>", unsafe_allow_html=True)
    
    current_data = fetch_api("/current")
    if current_data is None:
        current_data = fallback_current_data()
        
    if current_data:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ------------------ Row 1: Current Conditions (Modern Cards) ------------------
        aqi_val = float(current_data.get('us_aqi', 0))
        category = current_data.get('category', 'Unknown')
        emoji = current_data.get('emoji', '🌫️')
        color = current_data.get('color', '#38bdf8')
        temp = float(current_data.get('temperature_2m', 0))
        hum = float(current_data.get('relative_humidity_2m', 0))
        wind = float(current_data.get('wind_speed_10m', 0))
        pressure = float(current_data.get('surface_pressure', 1013))
        advisory = current_data.get('health_advisory', 'No specific health advisory.')

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div style="background: rgba(20, 30, 44, 0.85); border: 1px solid rgba(255,255,255,0.08); border-top: 4px solid {color}; border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Current Air Quality</div>
                    <div style="display: flex; align-items: baseline; gap: 0.4rem; margin: 0.3rem 0;">
                        <span style="font-size: 2.6rem; font-weight: 800; color: #ffffff; line-height: 1;">{aqi_val:.1f}</span>
                        <span style="font-size: 0.95rem; font-weight: 600; color: #94a3b8;">AQI</span>
                    </div>
                    <div style="display: inline-flex; align-items: center; gap: 0.35rem; background: {color}22; border: 1px solid {color}55; color: {color}; font-size: 0.8rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 999px; width: fit-content;">
                        <span>{emoji}</span> <span>{category}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div style="background: rgba(20, 30, 44, 0.85); border: 1px solid rgba(255,255,255,0.08); border-top: 4px solid #38bdf8; border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Temperature & Humidity</div>
                    <div style="display: flex; align-items: baseline; gap: 0.3rem; margin: 0.3rem 0;">
                        <span style="font-size: 2.3rem; font-weight: 800; color: #ffffff; line-height: 1;">{temp:.1f}°C</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #93c5fd; font-weight: 500;">
                        💧 <b>{hum:.0f}%</b> Humidity • <b>{pressure:.0f}</b> hPa
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div style="background: rgba(20, 30, 44, 0.85); border: 1px solid rgba(255,255,255,0.08); border-top: 4px solid #818cf8; border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Wind Conditions</div>
                    <div style="display: flex; align-items: baseline; gap: 0.3rem; margin: 0.3rem 0;">
                        <span style="font-size: 2.3rem; font-weight: 800; color: #ffffff; line-height: 1;">{wind:.1f}</span>
                        <span style="font-size: 0.95rem; font-weight: 600; color: #94a3b8;">km/h</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #a5b4fc; font-weight: 500;">
                        🧭 Natural atmospheric dispersion
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""
                <div style="background: rgba(20, 30, 44, 0.85); border: 1px solid rgba(255,255,255,0.08); border-top: 4px solid {color}; border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Health Guidance</div>
                    <div style="font-size: 0.84rem; color: #e2e8f0; line-height: 1.4; margin: 0.2rem 0; font-weight: 500;">
                        {advisory}
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">
                        Target: Sensitive & general populations
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.divider()

    # ------------------ Row 2: 3-Day Forecast (Individual Days) ------------------
    selected_model_key = {
        "XGBoost": "xgboost",
        "Random Forest": "random_forest",
        "Ridge": "ridge",
    }[model_selector]
    forecast_data = fetch_api("/predict", params={"model": selected_model_key})
    if not forecast_data:
        forecast_data = fallback_forecast_data(selected_model_key)
    model_name = (forecast_data.get('model_used') or selected_model_key).upper() if forecast_data else selected_model_key.upper()
    st.subheader(f"3-Day Forecast (Model: {model_name})")
    
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

                if day_num == 1:
                    day_label = "Today (Projected Mean)"
                elif day_num == 2:
                    day_label = "Tomorrow"
                else:
                    day_label = f"Day {day_num}"

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
                            <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                                {day_label} • {day_name}
                            </div>
                            <div style="display: flex; align-items: baseline; gap: 0.4rem; margin-top: 0.5rem;">
                                <span style="font-size: 2.3rem; font-weight: 800; color: #f8fafc;">{avg_aqi:.0f}</span>
                                <span style="font-size: 0.95rem; color: #94a3b8; font-weight: 600;">AQI</span>
                                <span style="font-size: 0.78rem; color: #64748b; font-weight: 500;">(24h Avg)</span>
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
