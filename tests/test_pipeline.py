import numpy as np, pandas as pd, pytest
from eutro.simulate import make_synthetic
from eutro.io import load_rain, load_temp, load_nutrients, load_obs
from eutro.features import build_features
from eutro.data import make_dataset
from eutro.tsi import tsi_chl, chl_from_tsi, tsi_to_class


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    make_synthetic(years=10, out_dir=d)
    return d


def test_tsi_roundtrip():
    assert abs(chl_from_tsi(tsi_chl(12.0)) - 12.0) < 1e-9
    assert tsi_to_class(tsi_chl(1.0)) == 0 and tsi_to_class(tsi_chl(60.0)) >= 2


def test_loaders_and_shapes(files):
    R = load_rain(files / "rain_points.csv"); assert R.shape[1] == 50
    T = load_temp(files / "temperature.csv"); assert (T.tmax >= T.tmin).all()
    N = load_nutrients(files / "nutrients.csv"); assert N.tp_ugL.notna().sum() > 50
    assert len(load_obs(files / "obs.csv")) > 200


def test_long_format_rain(files, tmp_path):
    R = load_rain(files / "rain_points.csv").iloc[:200, :5]
    long = R.reset_index(names="date").melt(id_vars="date", var_name="point_id", value_name="rainfall_mm")
    long.to_csv(tmp_path / "long.csv", index=False)
    R2 = load_rain(tmp_path / "long.csv")
    assert R2.shape == R.shape and np.allclose(R2.values, R.values)


def test_tp_unit_conversion(tmp_path):
    pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "TN_mgL": [1, 1.2], "TP_mgL": [0.04, 0.05]}).to_csv(tmp_path / "n.csv", index=False)
    assert load_nutrients(tmp_path / "n.csv").tp_ugL.dropna().iloc[0] == pytest.approx(40.0)


def test_features_and_no_lookahead(files):
    R, T, N = (load_rain(files / "rain_points.csv"), load_temp(files / "temperature.csv"),
               load_nutrients(files / "nutrients.csv"))
    F, g = build_features(R, T, N)
    assert set(g) == {"rain", "temp", "nutrients", "season"}
    assert not F[g["rain"] + g["temp"]].isna().any().any()
    cut = 2500
    Fc, _ = build_features(R.iloc[:cut], T.iloc[:cut], N[N.index <= R.index[cut - 1]])
    d = Fc.index[-1]
    pd.testing.assert_series_equal(F.loc[d], Fc.loc[d], check_names=False, rtol=1e-9)


def test_dataset_horizon(files):
    R, T, N = (load_rain(files / "rain_points.csv"), load_temp(files / "temperature.csv"),
               load_nutrients(files / "nutrients.csv"))
    F, g = build_features(R, T, N)
    X0, y0 = make_dataset(F, g, load_obs(files / "obs.csv"), 0)
    X7, y7 = make_dataset(F, g, load_obs(files / "obs.csv"), 7)
    d = X7.index[10]
    assert X7.loc[d, "rain_7d"] == F.loc[d - pd.Timedelta(days=7), "rain_7d"]


def test_end_to_end(files, tmp_path):
    from eutro.train import train
    from eutro.predict import predict
    train(files / "rain_points.csv", files / "temperature.csv", files / "nutrients.csv", files / "obs.csv",
          0, tmp_path / "m", tmp_path / "r")
    out = predict(files / "rain_points.csv", files / "temperature.csv", files / "nutrients.csv", None,
                  tmp_path / "m" / "eutro_model.joblib", quiet=True)
    assert 10 < out["TSI"] < 90 and abs(sum(out["class_probabilities"].values()) - 1) < 0.01
