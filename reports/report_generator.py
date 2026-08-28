"""
Pearls AQI Predictor — Report Generator

Generates a comprehensive PDF report documenting the entire AQI prediction system,
including EDA findings, model comparisons, SHAP analysis, and forecast results.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CITY_NAME,
    CITY_COUNTRY,
    REPORTS_DIR,
    MODELS_DIR,
    AQI_CATEGORIES,
)

logger = logging.getLogger(__name__)


class AQIReportGenerator:
    """Generates a detailed PDF report documenting the AQI prediction system."""

    def __init__(self):
        """Initialize the report generator."""
        if FPDF is None:
            raise ImportError("fpdf2 is required for report generation. Install with: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _add_title_page(self):
        """Add the title page."""
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 26)
        self.pdf.cell(0, 40, "", new_x="LMARGIN", new_y="NEXT")
        self.pdf.cell(0, 15, "Pearls AQI Predictor", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.set_font("Helvetica", "", 15)
        self.pdf.cell(0, 10, f"Air Quality Index Forecasting for {CITY_NAME}, {CITY_COUNTRY}", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.cell(0, 15, "", new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font("Helvetica", "", 12)
        self.pdf.cell(0, 8, f"Generated: {self.timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.cell(0, 8, "3-Day AQI Forecast Using Machine Learning", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.cell(0, 25, "", new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font("Helvetica", "I", 10)
        self.pdf.cell(0, 8, "End-to-End ML Pipeline with Automated Data Collection,", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.cell(0, 8, "Feature Engineering, Model Training, and Real-Time Predictions", new_x="LMARGIN", new_y="NEXT", align="C")

    def _add_section_header(self, title: str):
        """Add a section header."""
        self.pdf.set_font("Helvetica", "B", 15)
        self.pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.ln(1)

    def _add_subsection_header(self, title: str):
        """Add a subsection header."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font("Helvetica", "", 10)

    def _add_paragraph(self, text: str):
        """Add a paragraph of text."""
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.multi_cell(0, 5.5, text)
        self.pdf.ln(2)

    def _add_table(self, headers: list, data: list, col_widths: list = None):
        """Add a table to the report."""
        if col_widths is None:
            col_widths = [self.pdf.epw / len(headers)] * len(headers)

        # Header
        self.pdf.set_font("Helvetica", "B", 9)
        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 7, str(header), border=1, align="C")
        self.pdf.ln()

        # Data rows
        self.pdf.set_font("Helvetica", "", 9)
        for row in data:
            for i, cell in enumerate(row):
                self.pdf.cell(col_widths[i], 6.5, str(cell), border=1, align="C")
            self.pdf.ln()
        self.pdf.ln(4)

    def _add_executive_summary(self):
        """Add the executive summary section."""
        self.pdf.add_page()
        self._add_section_header("1. Executive Summary")
        self._add_paragraph(
            f"This report documents the Pearls AQI Predictor, an end-to-end machine learning "
            f"system for forecasting Air Quality Index (AQI) in {CITY_NAME}, {CITY_COUNTRY} "
            f"for the next 3 days (72 hours). The system uses a 100% serverless architecture "
            f"with automated data collection, feature engineering, model training, and real-time "
            f"predictions through an interactive web dashboard."
        )
        self._add_paragraph(
            f"{CITY_NAME} is one of the most polluted cities globally, particularly during the "
            f"smog season (October-February) when AQI values regularly exceed 300. Accurate "
            f"AQI forecasting is critical for public health advisories and planning."
        )

    def _add_system_architecture(self):
        """Add the system architecture section."""
        self._add_section_header("2. System Architecture")
        self._add_paragraph(
            "The system consists of four main pipelines:\n"
            "1. Feature Pipeline: Fetches raw weather and pollutant data from AQICN and "
            "Open-Meteo APIs, computes features, and stores them in the Feature Store.\n"
            "2. Training Pipeline: Fetches historical features, trains multiple ML models, "
            "evaluates performance, and registers the best model.\n"
            "3. Inference Pipeline: Loads the best model, generates 3-day forecasts with "
            "a model fallback chain for reliability.\n"
            "4. Web Dashboard: Displays real-time predictions, historical trends, and "
            "model explanations via Streamlit + Flask."
        )

        self._add_subsection_header("2.1 Technology Stack")
        tech_data = [
            ["Python 3.12", "Core programming language"],
            ["Scikit-learn, XGBoost", "Gradient boosting & ensemble ML models"],
            ["TensorFlow (LSTM)", "Deep learning sequence modeling"],
            ["Hopsworks / Local Parquet", "Dual-mode Feature Store"],
            ["AQICN + Open-Meteo", "Real-time and historical data sources"],
            ["Streamlit + Flask", "Interactive dashboard & REST API"],
            ["GitHub Actions", "CI/CD automation pipelines"],
            ["SHAP", "Model explainability & feature importance"],
        ]
        self._add_table(
            ["Technology", "Purpose"],
            tech_data,
            [55, 135],
        )

    def _add_data_description(self):
        """Add data description section."""
        self._add_section_header("3. Data Description")
        self._add_paragraph(
            "The system collects hourly data from multiple sources. Weather data includes "
            "temperature, humidity, wind speed/direction, pressure, and precipitation from "
            "Open-Meteo. Air quality data includes PM2.5, PM10, NO2, SO2, O3, and CO "
            "concentrations, plus the US AQI index."
        )

        self._add_subsection_header("3.1 Feature Categories")
        feature_data = [
            ["Weather", "8", "Temperature, humidity, wind, pressure, cloud cover"],
            ["Pollutants", "6", "PM2.5, PM10, NO2, SO2, O3, CO"],
            ["Time-based", "6", "Hour, day, month, weekend, season"],
            ["Lahore-specific", "5", "Smog season, crop burning, brick kilns, wind direction"],
            ["Derived/Lag", "10", "Rolling stats, lag features, interaction terms"],
            ["TOTAL", "35", "All model input features"],
        ]
        self._add_table(
            ["Category", "Count", "Description"],
            feature_data,
            [35, 20, 135],
        )

    def _add_model_comparison(self, metrics: dict = None):
        """Add model comparison section."""
        self.pdf.add_page()
        self._add_section_header("4. Model Performance Comparison")
        self._add_paragraph(
            "Three ML models were trained and evaluated using time-based train/validation/test "
            "splits (80/10/10) to prevent temporal data leakage. Models were evaluated using RMSE "
            "(Root Mean Squared Error), MAE (Mean Absolute Error), R-squared, and MAPE metrics."
        )

        if metrics:
            model_data = []
            for model_name, m in metrics.items():
                rmse_val = m.get('rmse', 'N/A')
                mae_val = m.get('mae', 'N/A')
                r2_val = m.get('r2', 'N/A')
                mape_val = m.get('mape', 'N/A')

                model_data.append([
                    model_name.upper(),
                    f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else str(rmse_val),
                    f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else str(mae_val),
                    f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val),
                    f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else str(mape_val),
                ])
            self._add_table(
                ["Model", "RMSE", "MAE", "R2", "MAPE"],
                model_data,
                [45, 35, 35, 35, 40],
            )
        else:
            self._add_paragraph(
                "Model metrics will be populated after the training pipeline has been "
                "executed. Run the training pipeline to generate performance metrics."
            )

        self._add_subsection_header("4.1 Model Fallback Chain")
        self._add_paragraph(
            "The system implements a model fallback chain for 100% production reliability:\n"
            "Primary: XGBoost -> Fallback 1: Random Forest -> Fallback 2: Ridge -> "
            "Emergency: Last known AQI values.\n\n"
            "Each model undergoes validation checks before serving predictions. If it "
            "produces errors, NaN values, negative AQI, or values exceeding 600, "
            "the system automatically cascades to the next model."
        )

    def _add_aqi_categories(self):
        """Add AQI categories reference."""
        self._add_section_header("5. AQI Categories & Health Advisories")
        cat_data = []
        for cat in AQI_CATEGORIES:
            cat_data.append([
                f"{cat['min']}-{cat['max']}",
                cat["label"],
                cat["color"],
            ])
        self._add_table(
            ["AQI Range", "Category", "Color Code"],
            cat_data,
            [35, 110, 45],
        )

    def _add_lahore_analysis(self):
        """Add Lahore-specific analysis section."""
        self._add_section_header("6. Lahore-Specific Analysis")
        self._add_paragraph(
            f"{CITY_NAME} has unique pollution patterns driven by several factors:\n\n"
            "- Smog Season (Oct-Feb): Temperature inversions trap pollutants near the surface, "
            "causing severe air quality deterioration. AQI regularly exceeds 300.\n"
            "- Crop Burning (Oct-Nov): Stubble burning in Punjab contributes significantly "
            "to PM2.5 levels during the post-harvest season.\n"
            "- Brick Kilns (Nov-Mar): Traditional brick kilns operate during winter months, "
            "adding to industrial emissions.\n"
            "- Cross-border Pollution: Easterly winds carry pollution across the border.\n"
            "- Rain Effect: Rainfall is the primary natural air cleanser in Lahore."
        )

    def _add_automation_section(self):
        """Add CI/CD and automation section."""
        self.pdf.add_page()
        self._add_section_header("7. Automation & CI/CD")
        self._add_paragraph(
            "The system uses GitHub Actions for automated pipeline execution:\n\n"
            "- Feature Pipeline: Runs every hour to fetch fresh weather and air quality "
            "data, compute features, and store them in the Feature Store.\n"
            "- Training Pipeline: Runs daily at 2 AM UTC (7 AM PKT) to retrain models "
            "with the latest data and update the Model Registry.\n\n"
            "This ensures the prediction models are always up-to-date with the latest "
            "data patterns and seasonal changes."
        )

    def _add_conclusion(self):
        """Add conclusion section."""
        self._add_section_header("8. Conclusion & Deliverables")
        self._add_paragraph(
            "The Pearls AQI Predictor provides a comprehensive, automated solution for "
            f"air quality forecasting in {CITY_NAME}. Key achievements:\n\n"
            "1. End-to-end automated pipeline from data collection to prediction serving.\n"
            "2. Multiple ML models (Ridge, Random Forest, XGBoost) with automatic fallback.\n"
            "3. Lahore-specific features capturing local pollution drivers.\n"
            "4. Interactive dashboard with real-time forecasts and health advisories.\n"
            "5. SHAP-based model explainability for transparent predictions.\n"
            "6. Scalable architecture with CI/CD automation.\n"
        )

    def generate_report(self, metrics: dict = None, output_path: str = None) -> str:
        """
        Generate the complete PDF report.
        """
        if output_path is None:
            output_path = str(REPORTS_DIR / f"aqi_predictor_report_{datetime.now().strftime('%Y%m%d')}.pdf")

        logger.info("Generating AQI Predictor report...")

        # Load metrics from saved results if not provided
        if metrics is None:
            metrics_file = MODELS_DIR / "training_results.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    logger.info("Loaded metrics from training results")
                except Exception as e:
                    logger.warning(f"Could not load metrics: {e}")

        # Build the report
        self._add_title_page()
        self._add_executive_summary()
        self._add_system_architecture()
        self._add_data_description()
        self._add_model_comparison(metrics)
        self._add_aqi_categories()
        self._add_lahore_analysis()
        self._add_automation_section()
        self._add_conclusion()

        # Save
        self.pdf.output(output_path)
        logger.info(f"Report saved to: {output_path}")
        return output_path


def main():
    """Generate the project report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    generator = AQIReportGenerator()
    report_path = generator.generate_report()
    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()
