import numpy as np
from .config import TSI_BREAKS
def tsi_chl(c):  return 30.6 + 9.81 * np.log(c)
def chl_from_tsi(t): return np.exp((np.asarray(t, float) - 30.6) / 9.81)
def tsi_to_class(t): return np.digitize(t, TSI_BREAKS)


def tsi_tp(tp_ugL, floor=1.0):
    """Carlson (1977) TSI from total phosphorus (ug/L); TP floored at `floor` ug/L, TSI clipped to 0..100."""
    tp = np.maximum(np.asarray(tp_ugL, float), floor)
    return np.clip(14.42 * np.log(tp) + 4.15, 0, 100)
