import numpy as np
from .config import TSI_BREAKS
def tsi_chl(c):  return 30.6 + 9.81 * np.log(c)
def chl_from_tsi(t): return np.exp((np.asarray(t, float) - 30.6) / 9.81)
def tsi_to_class(t): return np.digitize(t, TSI_BREAKS)
