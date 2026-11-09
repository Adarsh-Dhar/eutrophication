"""Generate predicted TSI for a date range (default 2023-01-01 .. 2024-12-31) from climate data.

Usage (from the project root):
    python generate_tsi.py
    python generate_tsi.py --from 2023-01-01 --to 2024-12-31 --out reports/tsi_2023_2024.csv
    python generate_tsi.py --monthly   # print monthly means instead of daily rows

Needs: data/processed/rain_points.csv + temperature.csv (daily, >=180 days of history before --from)
       models/eutro_model_swat.joblib
Output: one row per day: TSI, 80% interval, trophic class, class probabilities
        (+ reports/tsi_2023_2024_monthly.csv with monthly means).
These are MODEL ESTIMATES of the SWAT-derived TSI (30-day trailing mean), not measured values.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from eutro import config as C
from eutro.io import load_rain, load_temp
from eutro.predict import forecast

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default="2023-01-01")
ap.add_argument("--to", dest="end", default="2024-12-31")
ap.add_argument("--rain", default=str(C.PROCESSED_DIR / "rain_points.csv"))
ap.add_argument("--temp", default=str(C.PROCESSED_DIR / "temperature.csv"))
ap.add_argument("--model", default=str(C.MODEL_DIR / "eutro_model_swat.joblib"))
ap.add_argument("--out", default=str(C.REPORT_DIR / "tsi_2023_2024.csv"))
ap.add_argument("--daily", dest="view", action="store_const", const="daily", default="daily",
                help="print daily rows to the console (default)")
ap.add_argument("--monthly", dest="view", action="store_const", const="monthly",
                help="print monthly means to the console")
a = ap.parse_args()

# sanity: climate must cover the requested range
last = pd.read_csv(a.temp, usecols=["date"], parse_dates=["date"])["date"].max()
if pd.Timestamp(a.end) > last:
    raise SystemExit(f"climate data ends {last.date()}, cannot predict up to {a.end}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df = forecast(a.rain, a.temp, None, a.model, a.start, a.end, out_csv=None, quiet=True)
df = df.rename(columns={"TSI": "TSI_pred"})
# rough phosphorus equivalent (inverse of TSI = 14.42 ln(TP) + 4.15); only meaningful for 4 < TSI < 100
df["TP_equiv_ugL"] = np.where((df.TSI_pred > 5) & (df.TSI_pred < 99), np.exp((df.TSI_pred - 4.15) / 14.42), np.nan).round(1)

# attach the climate inputs (same files/frames used for the features)
T = load_temp(a.temp)
df["tmax"] = T["tmax"].reindex(df.index).round(1)
df["tmin"] = T["tmin"].reindex(df.index).round(1)
df["rainfall_mm"] = load_rain(a.rain).mean(axis=1).reindex(df.index).round(1)
df.to_csv(a.out)

m = df[["TSI_pred", "TSI_lo80", "TSI_hi80", "rainfall_mm", "tmax", "tmin"]].resample("MS").mean().round(1)
m["class"] = [C.CLASSES[int(np.digitize(v, C.TSI_BREAKS))] for v in m.TSI_pred]
mout = Path(a.out).with_name(Path(a.out).stem + "_monthly.csv")
m.to_csv(mout)

print(f"{len(df)} days  {df.index.min().date()} .. {df.index.max().date()}  ->  {a.out}")
print(f"monthly means -> {mout}")
if a.view == "monthly":
    print(m.to_string())
    print("class counts (months):", m["class"].value_counts().to_dict())
else:
    print(df.to_string())
    print("class counts (days):", df["class"].value_counts().to_dict())
