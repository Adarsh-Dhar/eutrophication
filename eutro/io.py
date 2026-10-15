"""Flexible CSV loaders. All return DAILY, date-indexed frames and print what they assumed."""
import warnings
import numpy as np
import pandas as pd
from .tsi import tsi_chl


def _read(path):
    df = pd.read_csv(path)
    low = {c.strip().lower(): c for c in df.columns}
    if "date" not in low:
        raise ValueError(f"{path}: needs a 'date' column (found {list(df.columns)})")
    df = df.rename(columns={low["date"]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def _find(cols, keys, exclude=()):
    for c in cols:
        l = c.lower()
        if any(k in l for k in keys) and not any(e in l for e in exclude):
            return c
    return None


def _daily(df):
    df = df.groupby("date").mean(numeric_only=True).sort_index()
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))


def load_rain(path):
    """Wide (date, p01..p50) or long (date, point_id, rainfall_mm). Returns days x points (mm)."""
    df = _read(path)
    idc = _find(df.columns, ["point", "station", "gauge", "grid"]) or ("id" if "id" in df.columns else None)
    if idc:
        val = _find([c for c in df.columns if c not in ("date", idc)], ["rain", "precip", "mm", "value"]) \
              or [c for c in df.columns if c not in ("date", idc)][0]
        R = df.pivot_table(index="date", columns=idc, values=val, aggfunc="mean")
    else:
        R = df.set_index("date").select_dtypes("number")
    R = R.sort_index().reindex(pd.date_range(R.index.min(), R.index.max(), freq="D"))
    R = R.where(R >= 0)                                   # -999 etc. -> missing
    miss = R.isna().mean().mean()
    print(f"[rain] {R.shape[1]} points, {len(R)} days, {miss:.1%} missing (filled by spatial mean)")
    return R


def load_temp(path):
    df = _read(path)
    tmax, tmin = _find(df.columns, ["tmax", "max"]), _find(df.columns, ["tmin", "min"])
    if not (tmax and tmin):
        raise ValueError("temperature file needs columns containing 'tmax' and 'tmin'")
    T = _daily(df[["date", tmax, tmin]].rename(columns={tmax: "tmax", tmin: "tmin"}))
    T = T.interpolate(limit=5)
    sw = T["tmax"] < T["tmin"]
    if sw.any():
        print(f"[temp] {int(sw.sum())} days had tmax<tmin: swapped")
        T.loc[sw, ["tmax", "tmin"]] = T.loc[sw, ["tmin", "tmax"]].values
    print(f"[temp] {len(T)} days, {T['tmax'].isna().mean():.1%} missing")
    return T


def load_nutrients(path):
    """Columns containing 'tn'/'nitrogen' (mg/L) and 'tp'/'phosph' (ug/L; mg/L auto-converted)."""
    df = _read(path)
    n, p = _find(df.columns, ["tn", "nitrogen"]), _find(df.columns, ["tp", "phosph"])
    if not (n or p):
        raise ValueError("nutrient file needs a nitrogen (tn) and/or phosphorus (tp) column")
    out = pd.DataFrame({"date": df["date"]})
    out["tn_mgL"] = df[n] if n else np.nan
    out["tp_ugL"] = df[p] if p else np.nan
    if p and ("mg" in p.lower() or (out["tp_ugL"].median() < 1.0 and "ug" not in p.lower())):
        out["tp_ugL"] *= 1000.0
        print(f"[nutrients] '{p}' looks like mg/L -> converted to ug/L")
    N = _daily(out)
    print(f"[nutrients] {int(N['tn_mgL'].notna().sum())} TN and {int(N['tp_ugL'].notna().sum())} TP samples")
    return N


def load_obs(path):
    """Target = eutrophication level. Column with 'chl' (ug/L) preferred, else 'tsi'. Returns TSI series."""
    df = _read(path)
    chl, tsi = _find(df.columns, ["chl"]), _find(df.columns, ["tsi"])
    if chl:
        s = df.set_index("date")[chl]; s = tsi_chl(s.where(s > 0))
    elif tsi:
        s = df.set_index("date")[tsi]
    else:
        raise ValueError("obs file needs a chlorophyll-a column (chl_ugL) or a tsi column")
    s = s.dropna().groupby(level=0).mean().sort_index().rename("tsi")
    print(f"[obs] {len(s)} observations, TSI {s.min():.1f}..{s.max():.1f}")
    return s
