#!/usr/bin/env python3
"""python main.py {swat|prepare|demo|simulate|validate|train|predict|forecast} [--rain F --temp F --nutrients F --obs F --horizon H --date D]

forecast = predict eutrophication status for FUTURE dates: give rain/temp files that include
           the future period (>=180 days of history before the first forecast date are needed)

swat    = REAL-DATA PIPELINE: data/*.xlsx (climate + SWAT reach) -> TSI labels -> train -> 1951-2024 hindcast
prepare = only convert the xlsx files to CSVs in data/processed"""
import argparse, warnings
warnings.filterwarnings("ignore")
from eutro import config as C
from eutro.simulate import make_synthetic
from eutro.train import train
from eutro.predict import predict, forecast
from eutro.io import load_rain, load_temp, load_nutrients, load_obs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["swat", "prepare", "demo", "simulate", "validate", "train", "predict", "forecast", "serve"])
    ap.add_argument("--rain", help="rain at many points: wide (date,p01..) or long (date,point_id,rainfall_mm)")
    ap.add_argument("--temp", help="date,tmax,tmin (any column names containing tmax / tmin)")
    ap.add_argument("--nutrients", help="date, tn (mg/L), tp (ug/L or mg/L); irregular dates OK")
    ap.add_argument("--obs", help="date + chl_ugL (chlorophyll-a) or tsi  <- the eutrophication target")
    ap.add_argument("--horizon", type=int, default=0, help="days ahead to predict (0 = nowcast)")
    ap.add_argument("--date", help="date to predict for (default: last data date + horizon)")
    ap.add_argument("--from", dest="start", help="forecast: first TARGET date (default: day after last training date)")
    ap.add_argument("--to", dest="end", help="forecast: last target date (default: last date in the input files)")
    ap.add_argument("--out", help="forecast: also write the full daily table to this CSV")
    ap.add_argument("--model", default=None, help="model file (default: models/eutro_model_swat.joblib; a model written by demo/generic train is models/eutro_model.joblib)")
    ap.add_argument("--port", type=int, default=8000, help="serve: port for the HTTP API")
    ap.add_argument("--data-dir", default=str(C.DATA_DIR), help="folder with precipitation/tas_max/tas_min/reach .xlsx (swat, prepare)")
    ap.add_argument("--reach", help="SWAT daily reach output as .xlsx (default: auto-detect in --data-dir)")
    ap.add_argument("--window", type=int, default=C.SWAT_LABEL_WINDOW, help="label = trailing N-day mean TSI (swat, prepare)")
    a = ap.parse_args()

    if a.cmd in ("swat", "prepare"):
        from eutro.swat import prepare, hindcast
        f = prepare(a.data_dir, C.PROCESSED_DIR, a.window, a.reach)
        if a.cmd == "swat":
            meta = dict(label=f"Carlson TSI(TP) from SWAT reach output, trailing {a.window}-day mean",
                        features="climate only: rain (all grid points), Tmax, Tmin", label_window_days=a.window,
                        chlorophyll_output=False,
                        caveat="labels are SWAT-simulated, not measured; river reach, not a reservoir")
            train(f["rain"], f["temp"], None, f["obs"], a.horizon, tag=C.SWAT_TAG, meta=meta,
                  test_fraction=C.SWAT_TEST_FRACTION, n_trees=C.SWAT_TREES, leaf=C.SWAT_LEAF, cv_splits=C.SWAT_CV_SPLITS)
            hindcast(C.MODEL_DIR / f"eutro_model_{C.SWAT_TAG}.joblib", f["rain"], f["temp"],
                     C.REPORT_DIR / "hindcast_swat.csv", f["obs"], C.REPORT_DIR / "hindcast_swat.png")
    elif a.cmd == "simulate":
        make_synthetic(out_dir=C.SYNTH_DIR); print("wrote", C.SYNTH_DIR)
    elif a.cmd == "validate":
        for f, fn in ((a.rain, load_rain), (a.temp, load_temp), (a.nutrients, load_nutrients), (a.obs, load_obs)):
            if f: fn(f)
    elif a.cmd == "train":
        if not (a.rain and a.obs): ap.error("--rain and --obs are required (temp/nutrients optional)")
        train(a.rain, a.temp, a.nutrients, a.obs, a.horizon)
    elif a.cmd == "predict":
        if not a.rain: ap.error("--rain required")
        predict(a.rain, a.temp, a.nutrients, a.date, a.model or C.MODEL_PATH)
    elif a.cmd == "forecast":
        if not a.rain: ap.error("--rain required")
        forecast(a.rain, a.temp, a.nutrients, a.model or C.MODEL_PATH,
                 a.start, a.end, a.out)
    elif a.cmd == "serve":
        import uvicorn
        from eutro.api import app
        uvicorn.run(app, host="0.0.0.0", port=a.port)
    else:
        train()
        d = C.SYNTH_DIR
        predict(d / "rain_points.csv", d / "temperature.csv", d / "nutrients.csv", None,
                C.MODEL_DIR / "eutro_model.joblib")


if __name__ == "__main__":
    main()
