import os
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.feature_pipeline.feature_store import get_feature_store

logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load all features from feature store."""
    store = get_feature_store()
    df = store.get_latest_features(n_hours=365 * 24)
    if df is None or df.empty:
        parquet_path = config.FEATURES_DIR / "aqi_features_all.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            return pd.DataFrame()
    return df

def plot_aqi_distribution(df: pd.DataFrame):
    """Histogram + KDE of AQI values."""
    fig = px.histogram(
        df, x=config.TARGET, 
        marginal="box", # acts like KDE/boxplot representation
        nbins=50,
        title="Distribution of AQI Values"
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix of all numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=False, aspect="auto", title="Feature Correlation Heatmap")
    return fig

def plot_time_series_decomposition(df: pd.DataFrame):
    """Trend, seasonality, residual using manual decomposition."""
    if df.empty or config.TARGET not in df.columns:
        return go.Figure()
        
    y = df[config.TARGET].dropna()
    trend = y.rolling(window=24*7, center=True).mean() # 7-day rolling mean
    detrended = y - trend
    
    # Very basic seasonality (hourly mean of detrended)
    hourly_seasonal = detrended.groupby(detrended.index.hour).mean()
    seasonality = pd.Series([hourly_seasonal[h] for h in y.index.hour], index=y.index)
    
    residual = detrended - seasonality
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=('Original', 'Trend', 'Seasonality', 'Residuals'))
    
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonality.index, y=seasonality, mode='lines', name='Seasonality'), row=3, col=1)
    fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residuals'), row=4, col=1)
    
    fig.update_layout(height=800, title_text="Time Series Decomposition (AQI)")
    return fig

def plot_diurnal_pattern(df: pd.DataFrame):
    """Average AQI by hour of day, grouped by day_of_week."""
    if 'hour' not in df.columns or 'day_of_week' not in df.columns:
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.day_name()
        
    grouped = df.groupby(['hour', 'day_of_week'])[config.TARGET].mean().reset_index()
    fig = px.line(grouped, x='hour', y=config.TARGET, color='day_of_week', 
                  title="Diurnal AQI Pattern by Day of Week", markers=True)
    return fig

def plot_monthly_pattern(df: pd.DataFrame):
    """Average AQI by month (should show Oct-Feb spike for Lahore)."""
    if 'month' not in df.columns:
        df['month'] = df.index.month_name()
        
    grouped = df.groupby('month')[config.TARGET].mean().reset_index()
    # Sort months correctly
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    grouped['month'] = pd.Categorical(grouped['month'], categories=months_order, ordered=True)
    grouped = grouped.sort_values('month')
    
    fig = px.bar(grouped, x='month', y=config.TARGET, title="Monthly Average AQI Pattern")
    return fig

def plot_pollutant_trends(df: pd.DataFrame):
    """Multi-line chart of all pollutants over time."""
    polls = [p for p in config.POLLUTANT_FEATURES if p in df.columns]
    if not polls:
        return go.Figure()
        
    fig = px.line(df, y=polls, title="Pollutant Trends Over Time")
    return fig

def plot_weather_vs_aqi(df: pd.DataFrame):
    """Scatter plots: temp vs AQI, humidity vs AQI, wind vs AQI."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Temp vs AQI", "Humidity vs AQI", "Wind Speed vs AQI"))
    
    if 'temperature_2m' in df.columns:
        fig.add_trace(go.Scatter(x=df['temperature_2m'], y=df[config.TARGET], mode='markers', marker=dict(opacity=0.3)), row=1, col=1)
    if 'relative_humidity_2m' in df.columns:
        fig.add_trace(go.Scatter(x=df['relative_humidity_2m'], y=df[config.TARGET], mode='markers', marker=dict(opacity=0.3)), row=1, col=2)
    if 'wind_speed_10m' in df.columns:
        fig.add_trace(go.Scatter(x=df['wind_speed_10m'], y=df[config.TARGET], mode='markers', marker=dict(opacity=0.3)), row=1, col=3)
        
    fig.update_layout(title="Weather Variables vs AQI", showlegend=False)
    return fig

def generate_eda_report(df: pd.DataFrame):
    """Run all plots, save as HTML and PNG files in reports/ directory."""
    logger.info("Generating EDA report...")
    
    plots = {
        'aqi_distribution': plot_aqi_distribution(df),
        'correlation_heatmap': plot_correlation_heatmap(df),
        'time_series_decomp': plot_time_series_decomposition(df),
        'diurnal_pattern': plot_diurnal_pattern(df),
        'monthly_pattern': plot_monthly_pattern(df),
        'pollutant_trends': plot_pollutant_trends(df),
        'weather_vs_aqi': plot_weather_vs_aqi(df)
    }
    
    for basename, fig in plots.items():
        try:
            html_path = config.REPORTS_DIR / f"{basename}.html"
            fig.write_html(str(html_path))
            logger.info(f"Saved {basename}.html")
        except Exception as e:
            logger.error(f"Failed to save {basename}.html: {e}")
            
        try:
            png_path = config.REPORTS_DIR / f"{basename}.png"
            fig.write_image(str(png_path))
            logger.info(f"Saved {basename}.png")
        except Exception:
            pass  # Kaleido not required when HTML is saved
            
    logger.info("EDA report generation complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        df = load_data()
        generate_eda_report(df)
    except Exception as e:
        logger.error(f"EDA failed: {e}")
