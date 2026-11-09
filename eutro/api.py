"""FastAPI endpoint for eutrophication-status prediction.

POST /predict
  {"date": "YYYY-MM-DD",
   "rain":  {"dates": ["YYYY-MM-DD", ...], "points": [[p1..pN], ...]},   # mm/day, N grid points
   "tmax":  [degC, ...], "tmin": [degC, ...],                          # one value per rain date, same order
   "model": "eutro_model_swat.joblib"}                                  # optional, default eutro_model_swat.joblib

Rules (same as the CLI): >=180 consecutive days ending on `date`, no missing days,
rain gaps (null) are filled with the spatial mean of the other points, tmax/tmin must
be complete. The target date must be inside the supplied window (horizon=0 models).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MODEL_DIR, MODEL_PATH, MIN_HISTORY_DAYS
from .predict import predict_frames

# eutro_model.joblib (demo/synthetic) was pickled with sklearn 1.8 and may not load on newer
# sklearn; prefer the swat model when present.
DEFAULT_MODEL = MODEL_DIR / "eutro_model_swat.joblib"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = MODEL_PATH

app = FastAPI(title="Eutrophication status API", version="1.0",
              description="Predicts TSI / trophic class for one date from daily rain, Tmax and Tmin.")


class PredictRequest(BaseModel):
    date: str = Field(..., description="target date, YYYY-MM-DD; must be inside the data window")
    rain: dict = Field(..., description='{"dates": [...], "points": [[...], ...]} mm/day per grid point')
    tmax: list[float] = Field(..., description="daily max temperature, deg C, one per date")
    tmin: list[float] = Field(..., description="daily min temperature, deg C, one per date")
    model: str | None = Field(None, description="model file name inside models/ (default: eutro_model_swat.joblib)")


def _model_path(name):
    if not name:
        return DEFAULT_MODEL
    p = (MODEL_DIR / Path(name).name).resolve()
    if p.parent != MODEL_DIR.resolve() or not p.exists():
        raise HTTPException(422, f"unknown model '{name}' (available: {[f.name for f in MODEL_DIR.glob('*.joblib')]})")
    return p


def _build_frames(req: PredictRequest):
    """Validate the request and return (rain_df days x points, temp_df date,tmax,tmin)."""
    try:
        dates = pd.to_datetime(pd.Series(req.rain.get("dates")), errors="raise")
    except Exception:
        raise HTTPException(422, "rain.dates must be a list of YYYY-MM-DD dates")
    if dates.isna().any() or dates.duplicated().any():
        raise HTTPException(422, "rain.dates contains invalid or duplicate dates")
    dates = pd.DatetimeIndex(dates)

    try:
        pts = np.asarray(req.rain.get("points"), dtype=float)
    except (TypeError, ValueError):
        raise HTTPException(422, "rain.points must be a numeric matrix, one row per date")
    if pts.ndim != 2 or pts.shape[0] != len(dates) or pts.shape[1] < 1:
        raise HTTPException(422, f"rain.points must be {len(dates)} x >=1 (got shape {pts.shape})")
    if np.isinf(pts).any():
        raise HTTPException(422, "rain.points contains infinite values (use null for missing)")

    if len(req.tmax) != len(dates) or len(req.tmin) != len(dates):
        raise HTTPException(422, f"tmax/tmin must have one value per date ({len(dates)}); "
                                 f"got {len(req.tmax)} and {len(req.tmin)}")
    tmax, tmin = np.asarray(req.tmax, dtype=float), np.asarray(req.tmin, dtype=float)
    if not (np.isfinite(tmax).all() and np.isfinite(tmin).all()):
        raise HTTPException(422, "tmax/tmin must have no missing values")

    # no missing days anywhere in the supplied window (features need consecutive history)
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    if len(full) != len(dates):
        missing = full.difference(dates)
        raise HTTPException(422, f"{len(missing)} missing day(s) in {dates.min().date()}..{dates.max().date()} "
                                 f"(first: {missing[0].date()}); supply a complete daily series")

    # target date must be present, with >=180 consecutive days ending on it
    try:
        target = pd.Timestamp(req.date)
    except Exception:
        raise HTTPException(422, f"date '{req.date}' is not a valid date")
    need_from = target - pd.Timedelta(days=MIN_HISTORY_DAYS - 1)
    have = set(dates)
    if target not in have or need_from not in have:
        raise HTTPException(422, f"need {MIN_HISTORY_DAYS} consecutive days ending on {target.date()}; "
                                 f"window must cover {need_from.date()}..{target.date()} "
                                 f"(data covers {dates.min().date()}..{dates.max().date()})")

    rain = pd.DataFrame(pts, index=dates).sort_index()
    rain.index.name = "date"
    temp = pd.DataFrame({"tmax": tmax, "tmin": tmin}, index=dates).sort_index()
    return rain, temp


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(f.name for f in MODEL_DIR.glob("*.joblib"))}


@app.post("/predict")
def predict(req: PredictRequest):
    rain, temp = _build_frames(req)
    model_path = _model_path(req.model)
    try:
        out = predict_frames(rain, temp, date=req.date, model_path=model_path, quiet=True)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:                      # e.g. model pickle incompatible with installed sklearn
        raise HTTPException(500, f"prediction failed: {type(e).__name__}: {e}")
    out["model"] = Path(model_path).name
    out["n_rain_points"] = int(rain.shape[1])
    out["window"] = {"from": str(rain.index.min().date()), "to": str(rain.index.max().date()),
                     "days": int(len(rain))}
    return out
