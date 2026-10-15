"""Features from rain (many points), Tmax/Tmin, and N/P. Row at date t uses data up to t only."""
import numpy as np
import pandas as pd
from .config import RAIN_WINDOWS, NUTRIENT_MAX_AGE_DAYS


def _since(mask, index):
    pos = pd.Series(np.where(mask, np.arange(len(mask)), np.nan), index=index).ffill()
    return (np.arange(len(mask)) - pos).fillna(999)


def rain_features(R):
    m = R.mean(axis=1, skipna=True).fillna(0.0)                  # catchment-mean daily rain
    R2 = R.where(R.notna(), m, axis=0)
    f = pd.DataFrame(index=R.index)
    for w in RAIN_WINDOWS:
        f[f"rain_{w}d"] = m.rolling(w, min_periods=w).sum()
    for w in (30, 90):
        f[f"max_day_{w}d"] = m.rolling(w, min_periods=w).max()
        f[f"n_gt25_{w}d"] = (m > 25).astype(float).rolling(w, min_periods=w).sum()
        f[f"n_wet_{w}d"] = (m > 1).astype(float).rolling(w, min_periods=w).sum()
    f["n_gt50_90d"] = (m > 50).astype(float).rolling(90, min_periods=90).sum()
    for k in (0.90, 0.97):
        f[f"api_{k}"] = m.ewm(alpha=1 - k, adjust=False).mean()
    f["days_since_gt25"] = _since((m > 25).values, m.index)
    f["days_since_gt1"] = _since((m > 1).values, m.index)
    dry = (m <= 1).astype(int)
    run = dry.groupby((dry != dry.shift()).cumsum()).cumsum() * dry
    f["dry_spell_now"], f["dry_spell_max_60d"] = run, run.rolling(60, min_periods=1).max()
    f["storm_ratio_90d"] = f["max_day_90d"] / (f["rain_90d"] + 1)
    # spatial structure across the points
    f["pt_max_7d"] = R2.max(axis=1).rolling(7, min_periods=7).max()
    f["pt_max_30d"] = R2.max(axis=1).rolling(30, min_periods=30).max()
    f["heavy_frac_30d"] = (R2 > 25).mean(axis=1).rolling(30, min_periods=30).mean()
    f["wet_frac_30d"] = (R2 > 1).mean(axis=1).rolling(30, min_periods=30).mean()
    s30 = R2.rolling(30, min_periods=30).sum()
    f["spatial_cv_30d"] = s30.std(axis=1) / (s30.mean(axis=1) + 1)
    return f


def temp_features(T):
    tmean, dtr = (T["tmax"] + T["tmin"]) / 2, T["tmax"] - T["tmin"]
    f = pd.DataFrame(index=T.index)
    for w in (7, 14, 30, 60):
        f[f"tmean_{w}d"] = tmean.rolling(w, min_periods=w).mean()
    f["tmax_max_14d"] = T["tmax"].rolling(14, min_periods=14).max()
    f["tmax_max_30d"] = T["tmax"].rolling(30, min_periods=30).max()
    f["tmin_min_30d"] = T["tmin"].rolling(30, min_periods=30).min()
    f["dtr_30d"] = dtr.rolling(30, min_periods=30).mean()
    f["gdd25_30d"] = (tmean - 25).clip(lower=0).rolling(30, min_periods=30).sum()   # >25C favours cyanobacteria
    f["warm_days_30d"] = (tmean >= 25).astype(float).rolling(30, min_periods=30).sum()
    f["hot_days_30d"] = (T["tmax"] >= 33).astype(float).rolling(30, min_periods=30).sum()
    f["tmean_trend"] = f["tmean_7d"] - f["tmean_30d"]
    return f


def nutrient_features(N, index):
    N = N.reindex(index)
    f = pd.DataFrame(index=index)
    for col in ("tn_mgL", "tp_ugL"):
        s, key = N[col], col.split("_")[0]
        if s.notna().sum() == 0:
            continue
        pos = pd.Series(np.where(s.notna(), np.arange(len(s)), np.nan), index=index).ffill()
        age = np.arange(len(s)) - pos
        last = s.ffill().where(age <= NUTRIENT_MAX_AGE_DAYS)
        f[f"{key}_last"], f[f"{key}_age"] = last, age.fillna(999)
        f[f"{key}_mean30"] = s.rolling(30, min_periods=1).mean()
        f[f"{key}_mean90"] = s.rolling(90, min_periods=1).mean()
        f[f"{key}_dev90"] = last - f[f"{key}_mean90"]
    if {"tn_last", "tp_last"} <= set(f.columns):
        f["np_ratio"] = f["tn_last"] / (f["tp_last"] / 1000.0)          # mass ratio (mg N / mg P)
    return f


def build_features(R, T=None, N=None):
    """Returns (features, groups). T and N optional (ablation / missing data)."""
    parts, groups = {}, {}
    idx = R.index
    if T is not None:
        idx = idx.intersection(T.index)
    F = rain_features(R).loc[idx]
    groups["rain"] = list(F.columns)
    if T is not None:
        t = temp_features(T).loc[idx]; groups["temp"] = list(t.columns); F = F.join(t)
    if N is not None:
        n = nutrient_features(N, idx); groups["nutrients"] = list(n.columns); F = F.join(n)
    doy = idx.dayofyear.values
    F["sin_doy"], F["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    groups["season"] = ["sin_doy", "cos_doy"]
    core = groups["rain"] + groups.get("temp", [])
    return F.dropna(subset=core), groups
