import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import SEED
from .tsi import tsi_to_class, chl_from_tsi


class SeasonalBaseline(BaseEstimator, RegressorMixin):
    """Predict the mean TSI of the same time of year (12 bins). The bar ML must beat."""
    def _bin(self, X):
        ang = np.arctan2(X["sin_doy"].values, X["cos_doy"].values) % (2 * np.pi)
        return np.minimum((ang / (2 * np.pi) * 12).astype(int), 11)
    def fit(self, X, y):
        b = self._bin(X); self.m_ = np.array([y.values[b == k].mean() if (b == k).any() else y.mean() for k in range(12)])
        return self
    def predict(self, X): return self.m_[self._bin(X)]


def candidates():
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "Lasso": make_pipeline(imp(), StandardScaler(), LassoCV(cv=TimeSeriesSplit(3), max_iter=50000)),
        "Random Forest": make_pipeline(imp(), RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "Extra Trees": make_pipeline(imp(), ExtraTreesRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                                                l2_regularization=1.0, random_state=SEED),
    }


def quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.05,
                                         max_depth=3, random_state=SEED)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    cy, cp = tsi_to_class(y), tsi_to_class(p)
    return dict(R2=round(float(r2_score(y, p)), 3), RMSE_TSI=round(float(mean_squared_error(y, p) ** .5), 2),
                MAE_TSI=round(float(mean_absolute_error(y, p)), 2),
                MAE_chl_ugL=round(float(mean_absolute_error(chl_from_tsi(y), chl_from_tsi(p))), 2),
                class_acc=round(float((cy == cp).mean()), 3), within1=round(float((abs(cy - cp) <= 1).mean()), 3))


def cv_rmse(model, X, y, n_splits):
    s = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        s.append(mean_squared_error(y.iloc[te], m.predict(X.iloc[te])) ** .5)
    return float(np.mean(s)), float(np.std(s))
