import pytest
import time
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import PredictionResponse


client = TestClient(app)


# -----------------------------------------------------
# FIXTURE: SINGLE SOURCE OF TRUTH
# -----------------------------------------------------

@pytest.fixture(scope="module")
def valid_payload():
    return {
        "AMT_CREDIT": 100000,
        "AMT_INCOME_TOTAL": 50000,
        "AMT_ANNUITY": 10000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -3000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7
    }


# -----------------------------------------------------
# TEST 1: HEALTH CHECK (SMOKE TEST)
# -----------------------------------------------------

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -----------------------------------------------------
# TEST 2: PREDICT CONTRACT (HAPPY PATH)
# -----------------------------------------------------

def test_predict_endpoint_success(valid_payload):

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    # strict contract enforcement
    assert set(data.keys()) == {"probability", "prediction"}

    assert isinstance(data["probability"], (float, int))
    assert isinstance(data["prediction"], int)

    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in {0, 1}


# -----------------------------------------------------
# TEST 3: MISSING FIELD (FAST FAIL CONTRACT)
# -----------------------------------------------------

def test_predict_missing_fields(valid_payload):

    payload = valid_payload.copy()
    payload.pop("AMT_CREDIT")

    response = client.post("/predict", json=payload)

    # API may fail via FastAPI validation or custom validation layer
    assert response.status_code in {400, 422}


# -----------------------------------------------------
# TEST 4: TYPE SAFETY (INPUT HARDENING)
# -----------------------------------------------------

def test_predict_invalid_types(valid_payload):

    payload = valid_payload.copy()
    payload["AMT_CREDIT"] = "not_a_number"

    response = client.post("/predict", json=payload)

    assert response.status_code in {400, 422}


# -----------------------------------------------------
# TEST 5: MODEL FAILURE MODE (SYSTEM RESILIENCE)
# -----------------------------------------------------

def test_predict_model_unavailable(monkeypatch, valid_payload):

    import api.main as main

    monkeypatch.setattr(main, "model", None)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503
    assert "Model not loaded" in response.text


# -----------------------------------------------------
# TEST 6: RESPONSE SCHEMA VALIDATION (STRICT CONTRACT)
# -----------------------------------------------------

def test_predict_response_schema(valid_payload):

    response = client.post("/predict", json=valid_payload)

    data = response.json()

    parsed = PredictionResponse(**data)

    assert 0.0 <= parsed.probability <= 1.0
    assert parsed.prediction in {0, 1}


# -----------------------------------------------------
# TEST 7: LATENCY SLA GUARDRAIL (PRODUCTION METRIC)
# -----------------------------------------------------

def test_predict_latency_sla(valid_payload):

    start = time.perf_counter()

    response = client.post("/predict", json=valid_payload)

    duration = time.perf_counter() - start

    assert response.status_code == 200
    assert duration < 0.5  # 500ms SLA threshold