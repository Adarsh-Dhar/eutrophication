"""Synthetic dataset in the SAME CSV format as real inputs:
rain_points.csv (50 pts), temperature.csv, nutrients.csv, obs.csv.  Physics: SCS-CN -> MUSLE -> P/N load ->
reservoir mass balance -> chlorophyll (Rast&Lee x temperature x N-limitation)."""
import numpy as np
import pandas as pd
from . import config as C


def _rain_base(years, rng):
    dates = pd.date_range("2000-01-01", periods=int(365.25 * years), freq="D")
    doy = dates.dayofyear.values
    ws = np.exp(-((doy - 205) / 55.0) ** 2)
    yfac = dict(zip(range(dates.year.min(), dates.year.max() + 1), rng.lognormal(0, 0.25, years + 2)))
    yf = dates.year.map(yfac).values
    wet = rng.random(len(dates)) < np.clip((0.04 + 0.55 * ws) * np.sqrt(yf), 0, 0.95)
    return dates, doy, wet * rng.gamma(0.7, (6 + 22 * ws) * yf)


def _scs(r, cn):
    S = 25.4 * (1000 / cn - 10); ia = 0.2 * S
    return (r - ia) ** 2 / (r + 0.8 * S) if r > ia else 0.0


def _adj(cn2, r5):
    if r5 < 35: return cn2 - 20 * (100 - cn2) / (100 - cn2 + np.exp(2.533 - 0.0636 * (100 - cn2)))
    if r5 > 52.5: return cn2 * np.exp(0.00673 * (100 - cn2))
    return cn2


def make_synthetic(years=C.SIM_YEARS, out_dir=None, seed=C.SEED):
    rng = np.random.default_rng(seed)
    dates, doy, base = _rain_base(years, rng)
    n, k = len(dates), C.N_POINTS
    # --- 50 rain points: shared storm field + local variation
    wetpt = (rng.random((n, k)) < 0.92) & (base > 0)[:, None]
    local = (base == 0)[:, None] & (rng.random((n, k)) < 0.02)
    R = np.where(wetpt, base[:, None] * rng.lognormal(-0.06, 0.35, (n, k)), 0) + np.where(local, rng.gamma(0.7, 8, (n, k)), 0)
    R = np.round(R, 1); rain = R.mean(axis=1)
    # --- temperature: seasonal + persistent anomaly, cooler on rainy days
    a = np.zeros(n)
    for t in range(1, n): a[t] = 0.97 * a[t - 1] + rng.normal(0, 0.35)
    tmean = 26 + 7 * np.sin(2 * np.pi * (doy - 120) / 365) + a - 1.2 * (rain > 5)
    tmax = tmean + 5.5 + rng.normal(0, 0.8, n); tmin = np.minimum(tmean - 5.5 + rng.normal(0, 0.8, n), tmax - 1)
    # --- reservoir mass balance (P and N)
    cfrg = np.exp(-0.053 * C.ROCK); tp, tn, G = 25.0, 1.0, 20.0
    TP, TN = np.zeros(n), np.zeros(n)
    for t in range(n):
        Q = _scs(rain[t], _adj(C.CN2, rain[max(0, t - 4):t + 1].sum()))
        loadP = C.PT_P
        if Q > 0:
            qpk = rng.uniform(0.3, 0.7) * Q * (C.HRU_HA / 100) / (3.6 * C.T_CONC)
            sed = 11.8 * (Q * qpk * C.HRU_HA) ** 0.56 * C.K_USLE * C.C_USLE * C.P_USLE * C.LS_USLE * cfrg
            eps = min(0.78 * (sed / (10 * C.HRU_HA * Q)) ** -0.2468, C.EPS_CAP)
            kgha = 0.001 * C.CONC_SEDP * (sed / C.HRU_HA) * eps + C.P_SOL * Q / (C.RHO_B * 10 * C.KD)
            loadP += kgha * C.CATCH_HA * C.DELIVERY * rng.lognormal(0, 0.3)
        G += 0.25 * Q; bf = 0.012 * G; G -= bf
        qm3, bm3 = Q / 1000 * C.CATCH_HA * 1e4, bf / 1000 * C.CATCH_HA * 1e4
        loadP += C.BF_P * bm3 * 1e-6
        loadN = (C.RUNOFF_N * qm3 + C.BF_N * bm3) / 1000.0
        flush = min((qm3 + bm3) / C.V, 0.9)
        internal = 0.35 * np.clip((tmean[t] - 24) / 6, 0, 1) * rng.lognormal(0, 0.4)
        tp = max(tp + loadP * 1e9 / (C.V * 1000) - flush * tp - C.K_SET_P * tp + internal, 2.0)
        tn = max(tn + loadN * 1e6 / (C.V * 1000) - flush * tn - C.K_LOSS_N * tn, 0.05)
        TP[t], TN[t] = tp, tn
    t30 = pd.Series(tmean).rolling(30, min_periods=1).mean().values
    nlim = np.minimum(1.0, (TN / (TP / 1000.0) / 7.0) ** 0.6)                        # N limits when TN:TP < 7
    chl = 0.551 * TP ** 0.76 * 1.06 ** (t30 - 25) * nlim * rng.lognormal(0, 0.25, n)  # Rast&Lee x temp x N
    # --- write files in the real-data format
    rain_df = pd.DataFrame(R, columns=[f"p{i+1:02d}" for i in range(k)]); rain_df.insert(0, "date", dates)
    temp_df = pd.DataFrame({"date": dates, "tmax_c": tmax.round(1), "tmin_c": tmin.round(1)})
    ni = np.arange(C.SPINUP - 65, n, C.NUTR_EVERY); ni = ni[rng.random(len(ni)) > 0.10]
    nut_df = pd.DataFrame({"date": dates[ni], "tn_mgL": (TN[ni] * rng.lognormal(0, 0.10, len(ni))).round(3),
                           "tp_ugL": (TP[ni] * rng.lognormal(0, 0.12, len(ni))).round(1)})
    oi = np.arange(C.SPINUP, n, C.OBS_EVERY)
    obs_df = pd.DataFrame({"date": dates[oi], "chl_ugL": chl[oi].round(2)})
    out = {"rain_points.csv": rain_df, "temperature.csv": temp_df, "nutrients.csv": nut_df, "obs.csv": obs_df}
    if out_dir:
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for fn, d in out.items(): d.to_csv(Path(out_dir) / fn, index=False)
    return out
