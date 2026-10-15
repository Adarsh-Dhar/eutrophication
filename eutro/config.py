from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, SYNTH_DIR = ROOT / "data", ROOT / "data" / "synthetic"
MODEL_DIR, REPORT_DIR = ROOT / "models", ROOT / "reports"
MODEL_PATH = MODEL_DIR / "eutro_model.joblib"

SEED = 42
CLASSES = ["Oligotrophic", "Mesotrophic", "Eutrophic", "Hypereutrophic"]
TSI_BREAKS = [40, 50, 60]
TEST_FRACTION, CV_SPLITS = 0.25, 5
RAIN_WINDOWS = (1, 3, 7, 14, 30, 60, 90, 180)
NUTRIENT_MAX_AGE_DAYS = 60       # last nutrient sample older than this is treated as missing
MIN_HISTORY_DAYS = 180           # longest rolling window

# ---- synthetic reservoir constants (tuned, NOT calibrated) ----
SIM_YEARS, SPINUP, OBS_EVERY, NUTR_EVERY, N_POINTS = 26, 365, 10, 14, 50
CN2, K_USLE, C_USLE, P_USLE, LS_USLE, ROCK = 75, 0.03, 0.2, 1.0, 1.5, 10
HRU_HA, T_CONC, CATCH_HA, DELIVERY, V = 100.0, 2.0, 15000.0, 0.30, 2.0e7
K_SET_P, K_LOSS_N = 0.02, 0.012
P_SOL, RHO_B, KD, CONC_SEDP, EPS_CAP = 0.4, 1.3, 175.0, 500.0, 5.0
BF_P, PT_P = 20.0, 2.0           # ug/L baseflow TP, kg/day point source
RUNOFF_N, BF_N = 3.0, 0.8        # g/m3 nitrogen in runoff / baseflow
