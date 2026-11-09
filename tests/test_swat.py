import datetime as dt
import numpy as np, pandas as pd, pytest
from eutro import swat
from eutro.tsi import tsi_tp


def _climate_xlsx(path, days=400, pts=3, seed=0):
    """Mimic the real files: Longitude/Latitude rows, then days; dd/mm strings for day>12, Excel-swapped datetimes for day<=12."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=days)
    cells = [dt.datetime(d.year, d.day, d.month) if d.day <= 12 else d.strftime("%d/%m/%Y") for d in idx]
    body = pd.DataFrame(rng.gamma(0.3, 8, (days, pts)), columns=[f"Point{i + 1}" for i in range(pts)])
    head = pd.DataFrame([[73.0] * pts, [24.0] * pts], columns=body.columns)
    out = pd.concat([head, body], ignore_index=True)
    out.insert(0, "Unnamed: 0", ["Longitude", "Latitude"] + cells)
    out.to_excel(path, index=False)
    return idx, body


def test_excel_date_repair(tmp_path):
    idx, body = _climate_xlsx(tmp_path / "c.xlsx")
    X = swat.load_climate_xlsx(tmp_path / "c.xlsx")
    assert X.index.equals(idx) and np.allclose(X.values, body.values)


def test_tsi_tp_values():
    assert tsi_tp(10.0) == pytest.approx(14.42 * np.log(10) + 4.15)
    assert tsi_tp(0.0) == pytest.approx(4.15)             # floor
    assert tsi_tp(1e9) == 100.0                           # clip


def test_tp_concentration_and_zero_flow():
    r = pd.DataFrame({"FLOW_OUTcms": [10.0, 0.0], "TOT_Pkg": [86.4, 5.0]}, index=pd.date_range("2000-01-01", periods=2))
    tp = swat.swat_tp_ugL(r)
    assert tp.iloc[0] == pytest.approx(100.0)             # 86.4 kg / 864000 m3 = 100 ug/L
    assert np.isnan(tp.iloc[1])


def test_reach_loader_rejects_monthly(tmp_path):
    pd.DataFrame({"YEAR": [2000] * 3, "MON": [1, 2, 3], "FLOW_OUTcms": [1, 1, 1], "TOT_Pkg": [1, 1, 1]}).to_excel(tmp_path / "r.xlsx", index=False)
    with pytest.raises(ValueError, match="monthly"):
        swat.load_swat_reach(tmp_path / "r.xlsx")


def test_prepare_roundtrip(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    for n in ("precipitation", "tas_max", "tas_min"):
        _climate_xlsx(d / f"{n}.xlsx", days=500)
    days = 400
    r = pd.DataFrame({"SUB": 1, "YEAR": 2000, "MON": np.arange(1, days + 1) % 366 + 1, "FLOW_OUTcms": np.linspace(1, 50, days),
                      "TOT_Pkg": np.linspace(10, 500, days)})
    r["YEAR"] = [2000 if i < 366 else 2001 for i in range(days)]
    r["MON"] = [i + 1 if i < 366 else i - 365 for i in range(days)]
    r.to_excel(d / "reach.xlsx", index=False)
    f = swat.prepare(d, tmp_path / "out", window=7)
    assert swat.find_reach_file(d).name == "reach.xlsx"
    for k in ("rain", "temp", "obs"):
        assert f[k].exists()
    from eutro.io import load_rain, load_obs
    assert load_rain(f["rain"]).shape[1] == 3          # wide format must NOT be mistaken for long format
    assert 0 <= load_obs(f["obs"]).min() and load_obs(f["obs"]).max() <= 100
