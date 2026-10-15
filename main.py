#!/usr/bin/env python3
"""python main.py {demo|simulate|validate|train|predict} [--rain F --temp F --nutrients F --obs F --horizon H --date D]"""
import argparse, warnings
warnings.filterwarnings("ignore")
from eutro import config as C
from eutro.simulate import make_synthetic
from eutro.train import train
from eutro.predict import predict
from eutro.io import load_rain, load_temp, load_nutrients, load_obs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["demo", "simulate", "validate", "train", "predict"])
    ap.add_argument("--rain", help="rain at many points: wide (date,p01..) or long (date,point_id,rainfall_mm)")
    ap.add_argument("--temp", help="date,tmax,tmin (any column names containing tmax / tmin)")
    ap.add_argument("--nutrients", help="date, tn (mg/L), tp (ug/L or mg/L); irregular dates OK")
    ap.add_argument("--obs", help="date + chl_ugL (chlorophyll-a) or tsi  <- the eutrophication target")
    ap.add_argument("--horizon", type=int, default=0, help="days ahead to predict (0 = nowcast)")
    ap.add_argument("--date", help="date to predict for (default: last data date + horizon)")
    ap.add_argument("--model", default=str(C.MODEL_PATH))
    a = ap.parse_args()

    if a.cmd == "simulate":
        make_synthetic(out_dir=C.SYNTH_DIR); print("wrote", C.SYNTH_DIR)
    elif a.cmd == "validate":
        for f, fn in ((a.rain, load_rain), (a.temp, load_temp), (a.nutrients, load_nutrients), (a.obs, load_obs)):
            if f: fn(f)
    elif a.cmd == "train":
        if not (a.rain and a.obs): ap.error("--rain and --obs are required (temp/nutrients optional)")
        train(a.rain, a.temp, a.nutrients, a.obs, a.horizon)
    elif a.cmd == "predict":
        if not a.rain: ap.error("--rain required")
        predict(a.rain, a.temp, a.nutrients, a.date, a.model)
    else:
        train()
        d = C.SYNTH_DIR
        predict(d / "rain_points.csv", d / "temperature.csv", d / "nutrients.csv", None)


if __name__ == "__main__":
    main()
