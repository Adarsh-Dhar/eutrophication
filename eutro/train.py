import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from . import config as C
from .data import make_dataset
from .features import build_features
from .io import load_rain, load_temp, load_nutrients, load_obs
from .models import SeasonalBaseline, candidates, quantile_model, metrics, cv_rmse
from .simulate import make_synthetic
from .tsi import chl_from_tsi


def _group_of(groups):
    return {f: g for g, fs in groups.items() for f in fs}


def train(rain=None, temp=None, nutrients=None, obs=None, horizon=0,
          model_dir=C.MODEL_DIR, report_dir=C.REPORT_DIR, synth_dir=C.SYNTH_DIR):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    if not (rain and obs):
        make_synthetic(out_dir=synth_dir); src = "synthetic (physics simulation)"
        rain, temp, nutrients, obs = [str(Path(synth_dir) / f) for f in
                                      ("rain_points.csv", "temperature.csv", "nutrients.csv", "obs.csv")]
    else:
        src = "observed"
    R = load_rain(rain)
    T = load_temp(temp) if temp else None
    N = load_nutrients(nutrients) if nutrients else None
    y_all = load_obs(obs)
    F, groups = build_features(R, T, N)
    X, y = make_dataset(F, groups, y_all, horizon)
    print(f"\nsource: {src} | horizon: {horizon} d | samples: {len(X)} | features: {X.shape[1]} | groups: "
          + ", ".join(f"{g}={len(v)}" for g, v in groups.items()))

    split = int(len(X) * (1 - C.TEST_FRACTION))
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    rows, fitted = [], {}
    base = SeasonalBaseline().fit(Xtr, ytr)
    rows.append(dict(model="Seasonal baseline", cv_rmse=np.nan, **metrics(yte, base.predict(Xte))))
    for name, mdl in candidates().items():
        cvm, cvs = cv_rmse(mdl, Xtr, ytr, C.CV_SPLITS)
        m = clone(mdl).fit(Xtr, ytr); fitted[name] = m
        rows.append(dict(model=name, cv_rmse=round(cvm, 2), **metrics(yte, m.predict(Xte))))
    res = pd.DataFrame(rows)
    print("\n", res.to_string(index=False))
    best = res[res.model != "Seasonal baseline"].sort_values("cv_rmse").iloc[0]["model"]
    print("\nbest (lowest time-series CV RMSE):", best)

    # ablation: which data groups carry the signal (hist gradient boosting, hold-out)
    combos = {"rain only": ["rain", "season"], "rain+temp": ["rain", "temp", "season"],
              "temp only": ["temp", "season"], "nutrients only": ["nutrients", "season"],
              "rain+temp+nutrients": ["rain", "temp", "nutrients", "season"]}
    abl = {}
    for label, gs in combos.items():
        if all(g in groups for g in gs):
            cols = [c for g in gs for c in groups[g]]
            m = clone(candidates()["Hist Gradient Boosting"]).fit(Xtr[cols], ytr)
            abl[label] = metrics(yte, m.predict(Xte[cols]))["R2"]
    print("\nablation (hold-out R2):", abl)

    # permutation importance on hold-out, summed per group
    pi = permutation_importance(fitted[best], Xte, yte, n_repeats=5, random_state=C.SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).clip(lower=0)
    g_of = _group_of(groups)
    group_imp = imp.groupby(imp.index.map(g_of)).sum()
    group_imp = (group_imp / max(group_imp.sum(), 1e-9)).round(3)
    print("importance share by group:", group_imp.to_dict())
    print("top features:", imp.sort_values(ascending=False).head(8).round(3).to_dict())

    # uncertainty: 80% interval + residual sigma
    lo, hi = quantile_model(0.1).fit(Xtr, ytr), quantile_model(0.9).fit(Xtr, ytr)
    pte = fitted[best].predict(Xte)
    sigma = float(np.std(yte.values - pte))
    ql, qh = np.sort(np.c_[lo.predict(Xte), hi.predict(Xte)], axis=1).T
    lo_te, hi_te = np.minimum(ql, pte - 1.28 * sigma), np.maximum(qh, pte + 1.28 * sigma)   # widened quantile band
    cover = float(np.mean((yte.values >= lo_te) & (yte.values <= hi_te)))
    print(f"80% interval hold-out coverage: {cover:.2f} | residual sigma: {sigma:.2f} TSI")

    # final refit on everything
    final = clone(candidates()[best]).fit(X, y)
    lo_f, hi_f = quantile_model(0.1).fit(X, y), quantile_model(0.9).fit(X, y)
    joblib.dump(dict(model=final, lo=lo_f, hi=hi_f, features=list(X.columns), groups=groups, sigma=sigma,
                     model_name=best, horizon=horizon, trained_on=src, uses=list(groups)),
                model_dir / "eutro_model.joblib")
    json.dump(dict(source=src, horizon=horizon, n=len(X), table=json.loads(res.to_json(orient="records")),
                   best=best, ablation_R2=abl, group_importance=group_imp.to_dict(),
                   interval80_coverage=round(cover, 3), residual_sigma=round(sigma, 2)),
              open(report_dir / "metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].scatter(yte, pte, s=12, alpha=.6); lim = [min(yte.min(), pte.min()), max(yte.max(), pte.max())]
    ax[0, 0].plot(lim, lim, "k--", lw=1)
    for b in C.TSI_BREAKS: ax[0, 0].axhline(b, color="grey", ls=":", lw=.6); ax[0, 0].axvline(b, color="grey", ls=":", lw=.6)
    ax[0, 0].set(xlabel="observed TSI", ylabel="predicted TSI", title=f"{best}: hold-out")
    ax[0, 1].plot(yte.index, yte.values, lw=1, label="observed"); ax[0, 1].plot(yte.index, pte, lw=1, label="predicted")
    ax[0, 1].fill_between(yte.index, lo_te, hi_te, alpha=.2, label="80% interval")
    ax[0, 1].legend(); ax[0, 1].set_title("Hold-out TSI over time")
    group_imp.sort_values().plot.barh(ax=ax[1, 0], color="teal"); ax[1, 0].set_title("Importance share by data group")
    pd.Series(abl).plot.barh(ax=ax[1, 1], color="darkorange"); ax[1, 1].set_title("Ablation: hold-out R²")
    plt.tight_layout(); plt.savefig(report_dir / "results.png", dpi=130); plt.close(fig)
    print("\nsaved:", model_dir / "eutro_model.joblib", "|", report_dir / "metrics.json", "|", report_dir / "results.png")
    return best
