# Multivariate Reservoir Eutrophication Model

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
python main.py predict  --rain rain.csv --temp temp.csv --nutrients nutrients.csv [--date 2026-10-14]
```
| File | Format |
|---|---|
| rain | wide `date,p01..p50` OR long `date,point_id,rainfall_mm`; daily, mm; negatives (-999) = missing |
| temp | `date,tmax,tmin` (any column names containing tmax / tmin), deg C |
| nutrients | `date,tn,tp` — TN mg/L, TP ug/L (mg/L auto-converted); irregular dates OK |
| obs (TARGET) | `date,chl_ugL` (chlorophyll-a) or `date,tsi`; this is the "eutrophication amount" |

* Rain/temp must start >= 180 days before the first observation. Need >= ~100 observations.
* `--horizon H`: predict H days ahead using only data up to H days before (0 = nowcast).
* Nutrients and temperature are optional; training reports an ablation showing what each data group adds.
* TSI = 30.6 + 9.81 ln(chl). Classes: <40 oligo, 40-50 meso, 50-60 eu, >60 hyper.

## Layout
```
main.py            CLI: demo | simulate | validate | train | predict
eutro/io.py        flexible loaders + unit handling
eutro/features.py  rain (mean + spatial), temperature, nutrient features (no look-ahead)
eutro/data.py      features <-> observations join with horizon
eutro/models.py    Lasso, RF, ExtraTrees, HistGB, seasonal baseline, CV, metrics
eutro/train.py     select by time-series CV, ablation, permutation importance, intervals, save
eutro/predict.py   inference with probabilities and interval
eutro/simulate.py  synthetic data in the same CSV format
tests/             pytest
```

## Caveat
The demo is trained on synthetic data (tuned, uncalibrated physics) so its scores are circular. Retrain on your observed data.
# eutrophication
