"""Real-data path: climate .xlsx files + SWAT reach output (.rch as .xlsx)  ->  TSI labels -> CSVs the pipeline reads.

Labels are NOT measurements. They are Carlson TSI computed from SWAT-simulated total phosphorus in the reach:
    TP [ug/L] = TOT_P [kg/day] * 1e6 / (FLOW_OUT [m3/s] * 86400)       TSI = 14.42 ln(TP) + 4.15
smoothed with a trailing mean (default 30 d). Features use ONLY climate (rain at all grid points, Tmax, Tmin).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C
from .features import build_features
from .tsi import tsi_tp, tsi_to_class


def _fix_excel_date(x):
    """Strings are dd/mm/yyyy. Cells Excel auto-converted came out with day and month swapped (1 Feb = 2 Jan)."""
    if isinstance(x, str):
        return pd.to_datetime(x.strip(), format="%d/%m/%Y")
    return pd.Timestamp(year=x.year, month=x.day, day=x.month)


def load_climate_xlsx(path):
    """Sheet layout: row0 = Longitude, row1 = Latitude, then one row per day; columns = grid points.
    Returns days x points DataFrame with a continuous daily DatetimeIndex."""
    raw = pd.read_excel(path)
    if str(raw.iloc[0, 0]).strip().lower() != "longitude":
        raise ValueError(f"{path}: expected first row 'Longitude' and second row 'Latitude'")
    body = raw.iloc[2:].copy()
    dates = pd.DatetimeIndex(body.iloc[:, 0].map(_fix_excel_date))
    X = body.iloc[:, 1:].astype(float)
    X.index = dates
    X.columns = [str(c) for c in X.columns]
    if not X.index.is_unique:
        raise ValueError(f"{path}: duplicate dates after date repair")
    X = X.sort_index()
    full = pd.date_range(X.index.min(), X.index.max(), freq="D")
    gaps = len(full) - len(X)
    if gaps:
        print(f"[climate] {Path(path).name}: {gaps} missing days (left as NaN)")
        X = X.reindex(full)
    print(f"[climate] {Path(path).name}: {X.shape[1]} points, {X.index.min().date()}..{X.index.max().date()}, {len(X)} days")
    return X


def load_swat_reach(path):
    """SWAT output.rch daily, exported to xlsx (needs YEAR, MON=julian day, FLOW_OUTcms, TOT_Pkg)."""
    r = pd.read_excel(path)
    need = {"YEAR", "MON", "FLOW_OUTcms", "TOT_Pkg"}
    if not need <= set(r.columns):
        raise ValueError(f"{path}: not a daily reach file, missing {sorted(need - set(r.columns))}")
    if r["MON"].max() <= 12:
        raise ValueError("reach file looks monthly (MON <= 12); a DAILY output.rch is required")
    r["date"] = pd.to_datetime(r["YEAR"].astype(int).astype(str) + r["MON"].astype(int).astype(str).str.zfill(3), format="%Y%j")
    if "SUB" in r.columns and r["SUB"].nunique() > 1:
        raise ValueError(f"{path}: several reaches {sorted(r['SUB'].unique())[:5]}...; split to one reach first")
    r = r.drop_duplicates("date").set_index("date").sort_index()
    print(f"[swat] reach file: {len(r)} days, {r.index.min().date()}..{r.index.max().date()}")
    return r


def swat_tp_ugL(r):
    vol = r["FLOW_OUTcms"] * 86400.0                              # m3/day
    return (r["TOT_Pkg"] * 1e6 / vol).where(r["FLOW_OUTcms"] > C.MIN_FLOW_CMS)


def swat_labels(r, window=C.SWAT_LABEL_WINDOW):
    daily = pd.Series(tsi_tp(swat_tp_ugL(r), C.TP_FLOOR_UGL), index=r.index).where(swat_tp_ugL(r).notna())
    lab = daily.rolling(window, min_periods=max(1, int(np.ceil(0.8 * window)))).mean().dropna()
    return daily, lab.rename("tsi")


def find_reach_file(data_dir):
    """Find the xlsx whose header has FLOW_OUTcms (any file name)."""
    import openpyxl
    for f in sorted(Path(data_dir).glob("*.xlsx")):
        if f.name.startswith(("precip", "tas_")):
            continue
        try:
            wb = openpyxl.load_workbook(f, read_only=True)
            hdr = [c for c in next(wb.worksheets[0].iter_rows(max_row=1, values_only=True))]
            wb.close()
        except Exception:
            continue
        if "FLOW_OUTcms" in hdr:
            return f
    raise FileNotFoundError(f"no SWAT reach .xlsx (with a FLOW_OUTcms column) found in {data_dir}")


def _keep_later_rows(new, path):
    """Keep rows after the xlsx range from the existing CSV (e.g. appended 2025 climate), so re-running prepare/swat does not delete them."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, index_col="date", parse_dates=True)
        if list(old.columns) == list(new.columns):
            extra = old.loc[old.index > new.index.max()]
            if len(extra):
                print(f"[prepare] kept {len(extra)} later rows from {path.name} ({extra.index.min().date()}..{extra.index.max().date()})")
                new = pd.concat([new, extra])
    return new


def prepare(data_dir=C.DATA_DIR, out_dir=C.PROCESSED_DIR, window=C.SWAT_LABEL_WINDOW, reach_file=None):
    """Writes rain_points.csv, temperature.csv, obs.csv (+ daily_tsi.csv) in the format the pipeline expects."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R = load_climate_xlsx(data_dir / "precipitation.xlsx")
    Tx, Tn = load_climate_xlsx(data_dir / "tas_max.xlsx"), load_climate_xlsx(data_dir / "tas_min.xlsx")
    T = pd.DataFrame({"tmax": Tx.mean(axis=1), "tmin": Tn.mean(axis=1)}).dropna()
    r = load_swat_reach(reach_file or find_reach_file(data_dir))
    daily, lab = swat_labels(r, window)
    R.columns = [f"p{i + 1:02d}" for i in range(R.shape[1])]       # wide format; "Point1" would be read as a long-format id column
    R = _keep_later_rows(R.rename_axis("date"), out_dir / "rain_points.csv")
    R.to_csv(out_dir / "rain_points.csv")
    T = _keep_later_rows(T.rename_axis("date"), out_dir / "temperature.csv")
    T.to_csv(out_dir / "temperature.csv")
    lab.rename_axis("date").rename("tsi").to_csv(out_dir / "obs.csv")
    daily.rename("tsi_daily").rename_axis("date").to_csv(out_dir / "daily_tsi.csv")
    cls = np.bincount(tsi_to_class(lab.values), minlength=4)
    print(f"[labels] {len(lab)} days, TSI {lab.min():.1f}..{lab.max():.1f}; class counts "
          f"(oligo/meso/eu/hyper) = {cls.tolist()}")
    print("wrote", out_dir)
    return {k: out_dir / f for k, f in dict(rain="rain_points.csv", temp="temperature.csv", obs="obs.csv").items()}


def hindcast(model_path, rain_csv, temp_csv, out_csv, obs_csv=None, plot_path=None):
    """Predict TSI for EVERY climate day (also years with no SWAT/measured labels). Writes a CSV (+ plot)."""
    import joblib
    from .io import load_rain, load_temp
    b = joblib.load(model_path)
    F, _ = build_features(load_rain(rain_csv), load_temp(temp_csv), None)
    X = F[b["features"]]
    mu = pd.Series(b["model"].predict(X), index=X.index)
    ql, qh = np.sort(np.c_[b["lo"].predict(X), b["hi"].predict(X)], axis=1).T
    lo = np.clip(np.minimum(ql, mu - 1.28 * b["sigma"]), 0, 100)
    hi = np.clip(np.maximum(qh, mu + 1.28 * b["sigma"]), 0, 100)
    mu = mu.clip(0, 100)
    out = pd.DataFrame({"TSI_pred": mu.round(2), "TSI_lo80": lo.round(2), "TSI_hi80": hi.round(2)})
    out["class_pred"] = [C.CLASSES[int(k)] for k in tsi_to_class(mu.values)]
    if obs_csv and Path(obs_csv).exists():
        obs = pd.read_csv(obs_csv, parse_dates=["date"]).set_index("date")["tsi"]
        out["TSI_swat_label"] = obs.reindex(out.index).round(2)
    out["holdout_period"] = (out.index >= pd.Timestamp(b["test_start"])) & out["TSI_swat_label"].notna() if "TSI_swat_label" in out else False
    out.rename_axis("date").to_csv(out_csv)
    if plot_path:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(15, 4.5))
        ax.fill_between(out.index, out.TSI_lo80, out.TSI_hi80, color="C0", alpha=.15, label="80% interval")
        ax.plot(out.index, out.TSI_pred, lw=.7, color="C0", label="model prediction (climate only)")
        if "TSI_swat_label" in out:
            ax.plot(out.index, out.TSI_swat_label, lw=.7, color="C1", label="SWAT-derived label")
            ax.axvline(pd.Timestamp(b["test_start"]), color="k", ls="--", lw=1)
            ax.text(pd.Timestamp(b["test_start"]), 2, " hold-out ->", fontsize=9)
        for t in C.TSI_BREAKS: ax.axhline(t, color="grey", ls=":", lw=.6)
        ax.set(ylabel="TSI (30-day mean)", title="Predicted trophic state 1951-2024 (labels exist only 1978-2022)")
        ax.legend(loc="upper left", ncol=3); plt.tight_layout(); plt.savefig(plot_path, dpi=130); plt.close(fig)
    print(f"[hindcast] {len(out)} days {out.index.min().date()}..{out.index.max().date()} -> {out_csv}")
    return out
