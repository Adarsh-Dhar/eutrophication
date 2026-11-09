import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eutro.api import app

client = TestClient(app)


def make_request(n=200, n_points=25, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d").tolist()
    doy = np.arange(n)
    rain = np.clip(rng.gamma(1.5, 4.0, size=(n, n_points)) * (1 + 0.8 * np.sin(2 * np.pi * (doy - 200) / 365))[:, None], 0, None)
    rain = rain.astype(object)
    rain[rng.random((n, n_points)) < 0.02] = None          # a few missing cells -> spatial mean fill
    tmean = 26 + 6 * np.sin(2 * np.pi * (doy - 100) / 365)
    return {"date": dates[-1],
            "rain": {"dates": dates, "points": rain.tolist()},
            "tmax": (tmean + 4).round(2).tolist(),
            "tmin": (tmean - 4).round(2).tolist(),
            "model": "eutro_model_swat.joblib"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and "eutro_model_swat.joblib" in r.json()["models"]


def test_predict_ok():
    r = client.post("/predict", json=make_request())
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == "eutro_model_swat.joblib"
    assert 0 <= out["TSI"] <= 100 and out["class"] in ("Oligotrophic", "Mesotrophic", "Eutrophic", "Hypereutrophic")
    assert abs(sum(out["class_probabilities"].values()) - 1) < 0.02
    assert out["window"]["days"] == 200 and out["n_rain_points"] == 25


def test_predict_rejects_short_history():
    r = client.post("/predict", json=make_request(n=100))
    assert r.status_code == 422 and "180" in r.text


def test_predict_rejects_missing_day():
    req = make_request()
    del req["rain"]["dates"][50]
    del req["rain"]["points"][50]
    req["tmax"].pop(50); req["tmin"].pop(50)
    r = client.post("/predict", json=req)
    assert r.status_code == 422 and "missing day" in r.text


def test_predict_rejects_target_outside_window():
    req = make_request()
    req["date"] = "2025-06-01"                     # after last data date
    r = client.post("/predict", json=req)
    assert r.status_code == 422


def test_predict_rejects_nan_temp():
    req = make_request()
    req["tmax"][10] = None
    r = client.post("/predict", json=req)
    assert r.status_code == 422 and "tmax" in r.text


def test_predict_rejects_bad_model():
    req = make_request()
    req["model"] = "../etc/passwd"
    r = client.post("/predict", json=req)
    assert r.status_code == 422
