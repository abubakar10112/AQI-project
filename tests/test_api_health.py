import pytest

from app.flask_api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_route_returns_project_status(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["name"] == "Pearls AQI Predictor"


def test_health_route_contains_system_status(client):
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    body = response.get_json()
    assert body["status"] in ("healthy", "degraded")
    assert "feature_store" in body
    assert "model_registry" in body
    assert body["forecast_horizon_days"] == 3


def test_predict_rejects_unknown_model_without_contacting_services(client):
    response = client.get("/api/predict?model=unknown")
    assert response.status_code == 400
    assert "Unknown model" in response.get_json()["error"]
