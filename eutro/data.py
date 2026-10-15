import pandas as pd


def make_dataset(F, groups, obs_tsi, horizon=0):
    """Observation at date d is explained by features at d-horizon (horizon=0 nowcast, >0 forecast)."""
    Fh = F.copy()
    Fh.index = Fh.index + pd.Timedelta(days=horizon)
    D = Fh.join(obs_tsi, how="inner")
    if len(D) < 30:
        raise ValueError(f"only {len(D)} usable observations: need >=30 (and >= {180} days of data before the first one)")
    return D.drop(columns="tsi"), D["tsi"]
