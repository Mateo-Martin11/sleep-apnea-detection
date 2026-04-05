from pathlib import Path

ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data" / "processed"
RAW_DIR     = ROOT_DIR / "data" / "raw"
OUTPUT_DIR  = ROOT_DIR / "data" / "submissions"
FIGURES_DIR = ROOT_DIR / "figures"

RANDOM_SEED = 42

N_SIGNALS            = 8
N_SAMPLES_PER_SIGNAL = 9000
N_LABEL_COLS         = 90
SIGNAL_FREQ          = 100
WINDOW_SECS          = 90

SIGNAL_NAMES = [
    "AbdoBelt", "AirFlow", "PPG", "ThorBelt",
    "Snoring", "SpO2", "EEG C4-A1", "EEG O2-A1"
]

VAL_SUBJECTS      = [0, 1, 8, 13, 15]
N_WINDOWS_PER_SUB = 200
N_FOLDS           = 5

BATCH_SIZE    = 32
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
POS_WEIGHT    = 12.23
