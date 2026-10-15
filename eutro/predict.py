import json
from math import erf, sqrt
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH, CLASSES, TSI_BREAKS
from .features import build_features
from .io import load_rain, load_temp, load_nutrients
from .tsi import chl_from_tsi, tsi_to_class


def _cdf(x, mu, s): return 0.5 * (1 + erf((x - mu) / (s * sqrt(2))))


def predict(rain, temp=None, nutrients=None, date=None, model_path=MODEL_PATH, quiet=False):
    b = joblib.load(model_path)
    if "temp" in b["uses"] and not temp: raise ValueError("this model needs --temp")
    if "nutrients" in b["uses"] and not nutrients: raise ValueError("this model needs --nutrients")
    F, _ = build_features(load_rain(rain), load_temp(temp) if "temp" in b["uses"] else None,
                          load_nutrients(nutrients) if "nutrients" in b["uses"] else None)
    H = b["horizon"]
    feat_date = (pd.Timestamp(date) - pd.Timedelta(days=H)) if date else F.index[-1]
    if feat_date not in F.index: raise ValueError(f"no features for {feat_date.date()} (data covers {F.index[0].date()}..{F.index[-1].date()})")
    X = F.loc[[feat_date], b["features"]]
    mu = float(b["model"].predict(X)[0])
    ql, qh = sorted([float(b["lo"].predict(X)[0]), float(b["hi"].predict(X)[0])])
    lo, hi = min(ql, mu - 1.28 * b["sigma"]), max(qh, mu + 1.28 * b["sigma"])
    edges = [-np.inf] + TSI_BREAKS + [np.inf]
    probs = {CLASSES[k]: round(_cdf(edges[k + 1], mu, b["sigma"]) - _cdf(edges[k], mu, b["sigma"]), 3) for k in range(4)}
    out = {"features_as_of": str(feat_date.date()), "prediction_for": str((feat_date + pd.Timedelta(days=H)).date()),
           "TSI": round(mu, 1), "chlorophyll_a_ugL": round(float(chl_from_tsi(mu)), 1),
           "class": CLASSES[int(tsi_to_class(mu))], "class_probabilities": probs,
           "interval80": {"TSI": [round(lo, 1), round(hi, 1)],
                          "chlorophyll_a_ugL": [round(float(chl_from_tsi(lo)), 1), round(float(chl_from_tsi(hi)), 1)]}}
    if not quiet: print(json.dumps(out, indent=2))
    return out
