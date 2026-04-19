import pytest
import json
from app import app

VALID_PAYLOAD = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "DiffWalk": 0, "Sex": 1,
    "BMI": 28.5, "MentHlth": 5, "PhysHlth": 3,
    "Age": 7, "Education": 5, "Income": 5, "GenHlth": 3
}

LOW_RISK_PAYLOAD = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "DiffWalk": 0, "Sex": 0,
    "BMI": 22.0, "MentHlth": 0, "PhysHlth": 0,
    "Age": 3, "Education": 6, "Income": 8, "GenHlth": 1
}

HIGH_RISK_PAYLOAD = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "Smoker": 1, "Stroke": 1,
    "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 0, "NoDocbcCost": 1,
    "DiffWalk": 1, "Sex": 1,
    "BMI": 40.0, "MentHlth": 20, "PhysHlth": 25,
    "Age": 12, "Education": 2, "Income": 1, "GenHlth": 5
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Health & Index ────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<!DOCTYPE html>" in r.data or b"<html" in r.data


# ── /predict (all models) ─────────────────────────────────────────────────────

def test_predict_all_returns_all_models(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.get_json()
    for key in ("xgboost", "catboost", "neural_network", "ensemble"):
        assert key in body
        assert "probability" in body[key]
        assert "prediction" in body[key]


def test_predict_all_probability_range(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.get_json()
    for model in ("xgboost", "catboost", "neural_network", "ensemble"):
        prob = body[model]["probability"]
        assert 0.0 <= prob <= 1.0, f"{model} probability out of range: {prob}"


def test_predict_all_binary_prediction(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.get_json()
    for model in ("xgboost", "catboost", "neural_network", "ensemble"):
        assert body[model]["prediction"] in (0, 1)


def test_predict_all_missing_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["BMI"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert "Missing fields" in r.get_json()["error"]


def test_predict_all_empty_body(client):
    r = client.post("/predict", json={})
    assert r.status_code == 400


# ── /predict/<model_name> ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["xgboost", "catboost", "neural_network", "ensemble"])
def test_single_model_valid(client, model):
    r = client.post(f"/predict/{model}", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"] == model
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_invalid_model_name(client):
    r = client.post("/predict/random_forest", json=VALID_PAYLOAD)
    assert r.status_code == 400


@pytest.mark.parametrize("model", ["xgboost", "catboost", "neural_network", "ensemble"])
def test_single_model_missing_field(client, model):
    payload = dict(VALID_PAYLOAD)
    del payload["Age"]
    r = client.post(f"/predict/{model}", json=payload)
    assert r.status_code == 400


# ── Risk direction sanity checks ──────────────────────────────────────────────

@pytest.mark.parametrize("model", ["xgboost", "catboost", "neural_network", "ensemble"])
def test_high_risk_higher_than_low_risk(client, model):
    low = client.post(f"/predict/{model}", json=LOW_RISK_PAYLOAD).get_json()["probability"]
    high = client.post(f"/predict/{model}", json=HIGH_RISK_PAYLOAD).get_json()["probability"]
    assert high > low, (
        f"{model}: expected high-risk ({high:.4f}) > low-risk ({low:.4f})"
    )


# ── Preprocessing / scaling consistency ──────────────────────────────────────

def test_predictions_are_deterministic(client):
    r1 = client.post("/predict", json=VALID_PAYLOAD).get_json()
    r2 = client.post("/predict", json=VALID_PAYLOAD).get_json()
    for model in ("xgboost", "catboost", "neural_network", "ensemble"):
        assert r1[model]["probability"] == r2[model]["probability"]


def test_bmi_boundary_obese(client):
    """BMI just above 30 should flip BMI_Category to 3 (obese)."""
    p1 = dict(VALID_PAYLOAD, BMI=29.9)
    p2 = dict(VALID_PAYLOAD, BMI=30.1)
    r1 = client.post("/predict/ensemble", json=p1).get_json()["probability"]
    r2 = client.post("/predict/ensemble", json=p2).get_json()["probability"]
    # obese BMI should produce >= probability than overweight BMI
    assert r2 >= r1
