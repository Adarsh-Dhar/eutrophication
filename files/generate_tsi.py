"""Generate predicted TSI for a date range (default 2023-01-01 .. 2100-12-31) from climate data.

Usage (from the files/ directory):
    python generate_tsi.py
    python generate_tsi.py graph monthly
    python generate_tsi.py graph yearly --out reports/tsi_yearly_ssp585.png

Defaults use the SSP585 future climate in future_data/processed (daily 2015-2100;
2015-2022 is retained as model-input history). Pass --rain/--temp/--from/--to to use other CSVs.

Needs: future_data/processed/rain_points.csv + temperature.csv (daily, >=180 days of history before --from)
       models/eutro_model_swat.joblib
Output: one row per day: TSI, 80% interval, trophic class, class probabilities
        (+ <output stem>_monthly.csv with monthly means).
        `graph` writes a PNG instead of the CSVs.
These are MODEL ESTIMATES of the SWAT-derived TSI (30-day trailing mean), not measured values.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from eutro import config as C
from eutro.io import load_rain, load_temp
from eutro.predict import forecast

FUTURE_PROCESSED_DIR = C.ROOT / "future_data" / "processed"
DATASET_TAG = "ssp585"


def add_common_args(parser, include_out=True):
    parser.add_argument("--from", dest="start", default="2023-01-01")
    parser.add_argument("--to", dest="end", default="2100-12-31")
    parser.add_argument("--rain", default=str(FUTURE_PROCESSED_DIR / "rain_points.csv"))
    parser.add_argument("--temp", default=str(FUTURE_PROCESSED_DIR / "temperature.csv"))
    parser.add_argument("--model", default=str(C.MODEL_DIR / "eutro_model_swat.joblib"))
    if include_out:
        parser.add_argument("--out", dest="out", default=None,
                            help="CSV output path (default: reports/tsi_ssp585_<from-year>_<to-year>.csv)")


def add_view_args(parser):
    parser.add_argument("--daily", dest="view", action="store_const", const="daily", default="daily",
                        help="print daily rows to the console (default)")
    parser.add_argument("--monthly", dest="view", action="store_const", const="monthly",
                        help="print monthly means to the console")


def default_csv_out(start, end):
    start_year, end_year = pd.Timestamp(start).year, pd.Timestamp(end).year
    stem = f"tsi_{DATASET_TAG}_{start_year}" if start_year == end_year else f"tsi_{DATASET_TAG}_{start_year}_{end_year}"
    return C.REPORT_DIR / f"{stem}.csv"


def aggregated_tsi(df, frequency):
    cols = ["TSI_pred", "TSI_lo80", "TSI_hi80"]
    if frequency == "daily":
        return df[cols]
    rule = {"monthly": "MS", "yearly": "YS"}[frequency]
    return df[cols].resample(rule).mean()


def plot_tsi(df, frequency, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = aggregated_tsi(df, frequency)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(s.index, s["TSI_lo80"], s["TSI_hi80"], color="tab:blue", alpha=0.15,
                    label="80% interval")
    ax.plot(s.index, s["TSI_pred"], color="tab:blue", lw=1.2 if frequency == "daily" else 2,
            marker=None if frequency == "daily" else "o", markersize=3, label="Predicted TSI")
    for value, label in zip(C.TSI_BREAKS,
                            ["Oligotrophic / Mesotrophic", "Mesotrophic / Eutrophic",
                             "Eutrophic / Hypereutrophic"]):
        ax.axhline(value, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.text(0.99, value, label, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=8, color="dimgray")
    ax.set_title(f"Predicted TSI ({frequency}) — {s.index.min().date()} to {s.index.max().date()}")
    ax.set_xlabel("Date" if frequency == "daily" else "Month" if frequency == "monthly" else "Year")
    ax.set_ylabel("TSI")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.01, 0.01, "Model estimates of SWAT-derived TSI (30-day trailing mean), not measurements.",
             fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


ap = argparse.ArgumentParser()
add_common_args(ap)
add_view_args(ap)
subparsers = ap.add_subparsers(dest="command")
graph_parser = subparsers.add_parser("graph", help="plot predicted TSI vs date")
add_common_args(graph_parser, include_out=False)
common_dests = {"start", "end", "rain", "temp", "model"}
for action in graph_parser._actions:
    if action.dest in common_dests:
        action.default = argparse.SUPPRESS
graph_parser.add_argument("frequency", nargs="?", choices=("daily", "monthly", "yearly"),
                          help="date aggregation for the plot (default: daily)")
graph_parser.add_argument("--frequency", "--freq", dest="frequency_opt",
                          choices=("daily", "monthly", "yearly"), default=None,
                          help="override the positional frequency")
graph_parser.add_argument("--out", dest="plot_out", default=None,
                          help="PNG output path (default: reports/tsi_<frequency>_ssp585_<from>_<to>.png)")
a = ap.parse_args()
if a.command == "graph":
    if a.frequency_opt and a.frequency and a.frequency_opt != a.frequency:
        ap.error("--frequency conflicts with the positional frequency")
    a.frequency = a.frequency_opt or a.frequency or "daily"

# sanity: climate must cover the requested range
last = pd.read_csv(a.temp, usecols=["date"], parse_dates=["date"])["date"].max()
if pd.Timestamp(a.end) > last:
    raise SystemExit(f"climate data ends {last.date()}, cannot predict up to {a.end}")

df = forecast(a.rain, a.temp, None, a.model, a.start, a.end, out_csv=None, quiet=True)
df = df.rename(columns={"TSI": "TSI_pred"})

if a.command == "graph":
    start_tag = pd.Timestamp(a.start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(a.end).strftime("%Y%m%d")
    plot_out = (a.plot_out or a.out or
                str(C.REPORT_DIR / f"tsi_{a.frequency}_{DATASET_TAG}_{start_tag}_{end_tag}.png"))
    plot_tsi(df, a.frequency, plot_out)
    print(f"{len(df)} days  {df.index.min().date()} .. {df.index.max().date()}")
    print(f"graph ({a.frequency}) -> {plot_out}")
else:
    out_csv = a.out or str(default_csv_out(a.start, a.end))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # rough phosphorus equivalent (inverse of TSI = 14.42 ln(TP) + 4.15); only meaningful for 4 < TSI < 100
    df["TP_equiv_ugL"] = np.where((df.TSI_pred > 5) & (df.TSI_pred < 99), np.exp((df.TSI_pred - 4.15) / 14.42), np.nan).round(1)

    # attach the climate inputs (same files/frames used for the features)
    T = load_temp(a.temp)
    df["tmax"] = T["tmax"].reindex(df.index).round(1)
    df["tmin"] = T["tmin"].reindex(df.index).round(1)
    df["rainfall_mm"] = load_rain(a.rain).mean(axis=1).reindex(df.index).round(1)
    df.to_csv(out_csv)

    m = df[["TSI_pred", "TSI_lo80", "TSI_hi80", "rainfall_mm", "tmax", "tmin"]].resample("MS").mean().round(1)
    m["class"] = [C.CLASSES[int(np.digitize(v, C.TSI_BREAKS))] for v in m.TSI_pred]
    mout = Path(out_csv).with_name(Path(out_csv).stem + "_monthly.csv")
    m.to_csv(mout)

    print(f"{len(df)} days  {df.index.min().date()} .. {df.index.max().date()}  ->  {out_csv}")
    print(f"monthly means -> {mout}")
    if a.view == "monthly":
        print(m.to_string())
        print("class counts (months):", m["class"].value_counts().to_dict())
    else:
        print(df.to_string())
        print("class counts (days):", df["class"].value_counts().to_dict())
