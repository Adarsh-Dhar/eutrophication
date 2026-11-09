import json
from functools import lru_cache
from math import erf, sqrt
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH, CLASSES, TSI_BREAKS
from .features import build_features
from .io import load_rain, load_temp, load_nutrients
from .tsi import chl_from_tsi, tsi_to_class


def _load_bundle(model_path, rain, temp, nutrients):
    b = joblib.load(model_path)
    if "temp" in b["uses"] and not temp: raise ValueError("this model needs --temp")
    if "nutrients" in b["uses"] and not nutrients: raise ValueError("this model needs --nutrients")
    F, _ = build_features(load_rain(rain), load_temp(temp) if "temp" in b["uses"] else None,
                          load_nutrients(nutrients) if "nutrients" in b["uses"] else None)
    return b, F


def _predict_array(b, X):
    """mu, lo80, hi80, probs[n,4] for feature rows X."""
    mu = np.clip(b["model"].predict(X), 0, 100)
    ql, qh = np.sort(np.c_[b["lo"].predict(X), b["hi"].predict(X)], axis=1).T
    lo = np.clip(np.minimum(ql, mu - 1.28 * b["sigma"]), 0, 100)
    hi = np.clip(np.maximum(qh, mu + 1.28 * b["sigma"]), 0, 100)
    edges = np.array([-np.inf] + TSI_BREAKS + [np.inf])
    z = (edges[None, :] - mu[:, None]) / (b["sigma"] * sqrt(2))
    c = 0.5 * (1 + np.vectorize(erf)(z))
    probs = np.diff(c, axis=1)
    return mu, lo, hi, probs


def _predict_features(b, F, date, quiet=False):
    H = b["horizon"]
    feat_date = (pd.Timestamp(date) - pd.Timedelta(days=H)) if date else F.index[-1]
    if feat_date not in F.index: raise ValueError(f"no features for {feat_date.date()} (data covers {F.index[0].date()}..{F.index[-1].date()})")
    X = F.loc[[feat_date], b["features"]]
    mu, lo, hi, probs = _predict_array(b, X)
    mu, lo, hi = float(mu[0]), float(lo[0]), float(hi[0])
    show_chl = b.get("meta", {}).get("chlorophyll_output", True)
    out = {"features_as_of": str(feat_date.date()), "prediction_for": str((feat_date + pd.Timedelta(days=H)).date()),
           "TSI": round(mu, 1), "class": CLASSES[int(tsi_to_class(mu))],
           "class_probabilities": {CLASSES[k]: round(float(probs[0, k]), 3) for k in range(4)},
           "interval80": {"TSI": [round(lo, 1), round(hi, 1)]}}
    if show_chl:       # only meaningful when the model was trained on chlorophyll-derived TSI
        out["chlorophyll_a_ugL"] = round(float(chl_from_tsi(mu)), 1)
        out["interval80"]["chlorophyll_a_ugL"] = [round(float(chl_from_tsi(lo)), 1), round(float(chl_from_tsi(hi)), 1)]
    if b.get("meta", {}).get("label"):
        out["label_definition"] = b["meta"]["label"]
    if not quiet: print(json.dumps(out, indent=2))
    return out


def predict(rain, temp=None, nutrients=None, date=None, model_path=MODEL_PATH, quiet=False):
    b, F = _load_bundle(model_path, rain, temp, nutrients)
    return _predict_features(b, F, date, quiet)


@lru_cache(maxsize=4)
def _cached_model(model_path):
    return joblib.load(str(model_path))


def predict_frames(rain, temp=None, nutrients=None, date=None, model_path=MODEL_PATH, quiet=False):
    """predict() but with already-loaded frames: rain = days x points (mm), temp = date,tmax,tmin (deg C)."""
    b = _cached_model(str(model_path))
    if "temp" in b["uses"] and temp is None: raise ValueError("this model needs temp")
    if "nutrients" in b["uses"] and nutrients is None: raise ValueError("this model needs nutrients")
    F, _ = build_features(rain, temp if "temp" in b["uses"] else None,
                          nutrients if "nutrients" in b["uses"] else None)
    return _predict_features(b, F, date, quiet)


def forecast(rain, temp=None, nutrients=None, model_path=MODEL_PATH, start=None, end=None,
             out_csv=None, quiet=False):
    """Predict eutrophication status for a range of FUTURE target dates.

    The rain/temp files must contain history AND the future dates (rolling features
    need >=180 days of history before the first forecast date). Target dates default to
    the day after the last date the model was trained on (bundle 'data_end') through the
    last date with features. With horizon H the feature row used is H days earlier.
    """
    b, F = _load_bundle(model_path, rain, temp, nutrients)
    H = b["horizon"]
    last_feat = F.index[-1]
    first_target = (pd.Timestamp(start) if start else
                    pd.Timestamp(b["data_end"]) + pd.Timedelta(days=1) if b.get("data_end")
                    else last_feat + pd.Timedelta(days=H))
    last_target = pd.Timestamp(end) if end else last_feat + pd.Timedelta(days=H)
    feat_idx = pd.date_range(first_target - pd.Timedelta(days=H), last_target - pd.Timedelta(days=H), freq="D")
    feat_idx = feat_idx[feat_idx.isin(F.index)]
    if not len(feat_idx):
        raise ValueError(f"no feature rows for target dates {first_target.date()}..{last_target.date()} "
                         f"(features cover {F.index[0].date()}..{last_feat.date()}; need >=180 d history before the first forecast date)")
    X = F.loc[feat_idx, b["features"]]
    mu, lo, hi, probs = _predict_array(b, X)
    targets = feat_idx + pd.Timedelta(days=H)
    df = pd.DataFrame({"TSI": mu.round(2), "TSI_lo80": lo.round(2), "TSI_hi80": hi.round(2),
                       "class": [CLASSES[int(k)] for k in tsi_to_class(mu)]}, index=targets)
    for k, c in enumerate(CLASSES):
        df[f"p_{c.lower()}"] = probs[:, k].round(3)
    df = df.rename_axis("date")
    if out_csv:
        df.to_csv(out_csv); print(f"[forecast] wrote {out_csv}")
    if not quiet:
        show_chl = b.get("meta", {}).get("chlorophyll_output", True)
        head = df.head(10)
        print(f"[forecast] model={Path(model_path).name} horizon={H}d trained_through={b.get('data_end','?')} "
              f"target {targets[0].date()}..{targets[-1].date()} ({len(df)} days)")
        print(head.to_string())
        if len(df) > 10: print(f"... ({len(df) - 10} more rows)")
        print("class counts:", df["class"].value_counts().to_dict())
        if show_chl:
            print("chlorophyll-a ug/L (from TSI):", round(float(chl_from_tsi(df["TSI"].mean())), 1), "(mean TSI)")
    return df
