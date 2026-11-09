# Multivariate Reservoir Eutrophication Model

## Quick start on the supplied data (climate + SWAT output)
```bash
pip install -r requirements.txt
python main.py swat                 # xlsx -> TSI labels -> train (chronological 80/20) -> 1951-2024 hindcast   (~3 min on 1 CPU)
python main.py predict --rain data/processed/rain_points.csv --temp data/processed/temperature.csv \
       --date 2021-08-15 --model models/eutro_model_swat.joblib
python main.py forecast --rain data/processed/rain_points.csv --temp data/processed/temperature.csv \
       --model models/eutro_model_swat.joblib --out reports/forecast_2025.csv   # future dates: files must include history + the future period
pytest -q                           # 20 tests
```

### HTTP API
```bash
python main.py serve --port 8000    # POST /predict  (GET /health, /docs for Swagger UI)
```
```json
POST /predict
{"date": "2025-08-15",
 "rain":  {"dates": ["2025-02-16", "...", "2025-08-15"], "points": [[p1..p25], "..."]},
 "tmax":  [27.1, "..."], "tmin": [21.4, "..."],
 "model": "eutro_model_swat.joblib"}
```
Needs >=180 consecutive days ending on `date` (rain mm/day at N grid points — nulls are filled
by the spatial mean; tmax/tmin deg C, no gaps). Returns TSI, class, class probabilities,
80% interval. `model` is optional (default: `eutro_model_swat.joblib`, or any `*.joblib` in `models/`).
Inputs in `data/`: `precipitation.xlsx`, `tas_max.xlsx`, `tas_min.xlsx` (25 grid points, daily 1951-2024) and the SWAT daily reach
output exported to xlsx (`adarsh 2.xlsx`, auto-detected by its `FLOW_OUTcms` column).

**What the model is.** Features = climate only (rain at all 25 points, Tmax, Tmin; rolling windows, no look-ahead).
Target = Carlson TSI computed from SWAT-simulated reach total phosphorus, `TP = TOT_P / (FLOW_OUT * 86400)`,
`TSI = 14.42 ln(TP ug/L) + 4.15`, smoothed with a trailing 30-day mean. Outputs: TSI, class, class probabilities, 80% interval.

**Outputs** `models/eutro_model_swat.joblib`, `reports/metrics_swat.json`, `reports/results_swat.png`,
`reports/hindcast_swat.csv` + `.png` (prediction for every day 1951-2024, incl. years with no SWAT labels), `data/processed/*.csv`.

**Read before trusting it**
* Labels are SWAT-simulated, not measured, and the SWAT run is a river reach (no reservoir). TSI(TP) is a proxy, not chlorophyll-a.
* The SWAT run was forced by a different rainfall series than `precipitation.xlsx` (daily r = 0.17, monthly r = 0.75), so daily skill is capped.
* Hold-out (2013-12 .. 2022, chronological): Extra Trees R2 = 0.80, RMSE = 14 TSI vs seasonal-only baseline R2 = 0.73, RMSE = 16.5.
  Most of the skill is the monsoon seasonal cycle; class accuracy (0.74) is NOT better than the seasonal baseline (0.78).
* Excel date bug in the climate files (day/month swapped on cells Excel auto-converted) is repaired in `eutro/swat.py`.
* With real measurements (chlorophyll-a / TP), use `train` below instead: it replaces the SWAT labels.

## Generic use (your own CSVs)
Predicts the eutrophication level of ONE reservoir (chlorophyll-a, Carlson TSI, trophic class + 80% interval) from:
rainfall at many points (e.g. 50), daily Tmax / Tmin, and nitrogen / phosphorus measurements.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py demo          # synthetic data: train, ablation, sample prediction
pytest -q
```

## Use your data
```bash
python main.py validate --rain rain.csv --temp temp.csv --nutrients nutrients.csv --obs obs.csv   # audit first
python main.py train    --rain rain.csv --temp temp.csv --nutrients nutrients.csv --obs obs.csv [--horizon 7]
python main.py predict  --rain rain.csv --temp temp.csv --nutrients nutrients.csv --model models/eutro_model.joblib [--date 2026-10-14]
```
| File | Format |
|---|---|
| rain | wide `date,p01..p50` OR long `date,point_id,rainfall_mm`; daily, mm; negatives (-999) = missing |
| temp | `date,tmax,tmin` (any column names containing tmax / tmin), deg C |
| nutrients | `date,tn,tp` — TN mg/L, TP ug/L (mg/L auto-converted); irregular dates OK |
| obs (TARGET) | `date,chl_ugL` (chlorophyll-a) or `date,tsi`; this is the "eutrophication amount" |

* Rain/temp must start >= 180 days before the first observation. Need >= ~100 observations.
* `--horizon H`: predict H days ahead using only data up to H days before (0 = nowcast).
* Default model is `models/eutro_model_swat.joblib` (real data); a model written by `train`/`demo` is `models/eutro_model.joblib` — pass it with `--model`.
* Nutrients and temperature are optional; training reports an ablation showing what each data group adds.
* TSI = 30.6 + 9.81 ln(chl). Classes: <40 oligo, 40-50 meso, 50-60 eu, >60 hyper.

## Layout
```
main.py            CLI: swat | prepare | demo | simulate | validate | train | predict | forecast
eutro/swat.py      xlsx climate + SWAT reach -> TSI labels, CSV export, 1951-2024 hindcast
eutro/io.py        flexible loaders + unit handling
eutro/features.py  rain (mean + spatial), temperature, nutrient features (no look-ahead)
eutro/data.py      features <-> observations join with horizon
eutro/models.py    Lasso, RF, ExtraTrees, HistGB, seasonal baseline, CV, metrics
eutro/train.py     select by time-series CV, ablation, permutation importance, intervals, save
eutro/predict.py   inference with probabilities and interval
eutro/simulate.py  synthetic data in the same CSV format
tests/             pytest
```

## Caveat (synthetic demo)
The demo is trained on synthetic data (tuned, uncalibrated physics) so its scores are circular. Retrain on your observed data.
# eutrophication
