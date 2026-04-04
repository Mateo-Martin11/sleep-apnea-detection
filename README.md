# Sleep Apnea Detection

Automatic detection of sleep apnea events from raw polysomnography (PSG) signals.
Built for [ENS Challenge #45](https://challengedata.ens.fr/participants/challenges/45/).

**Current score: 0.549 F1** (benchmark: 0.525 — top 10)

## Problem

Given 90-second windows of 8 physiological signals recorded at 100Hz, predict a binary mask at 1Hz indicating the presence of apnea events second by second.

| Signal | Description |
|---|---|
| AbdoBelt | Abdominal respiratory belt |
| AirFlow | Nasal airflow |
| PPG | Photoplethysmogram (heart rate) |
| ThorBelt | Thoracic respiratory belt |
| Snoring | Snoring indicator |
| SPO2 | Blood oxygen saturation |
| C4-A1 | EEG derivation |
| O2-A1 | EEG derivation |

**Evaluation metric:** event-based F1 score (IoU threshold ≥ 0.3)

## Data

- 4400 training windows, 4400 test windows
- 22 subjects, 200 windows per subject
- Apnea rate: ~6.9% (heavy class imbalance)
- Average event duration: 18 seconds

Data files (not included, download from challenge):
```
X_train.h5, X_test.h5
data/X_train_h7ipJUo.csv, data/X_test_PN3J7aD.csv
data/y_train_tX9Br0C.csv
```

## Architecture

1D Attention U-Net with multi-scale temporal encoding:

```
Input (B, 8, 9000) — 8 signals at 100Hz
  → Encoder: 100Hz → 10Hz → 1Hz
  → Bottleneck: dilated convolutions at 1Hz
  → Decoder: attention-gated skip connections
  → Head: (B, 90) logits at 1Hz
```

**Loss:** BCE (pos_weight=10) + Dice Loss  
**Post-processing:** remove predicted events shorter than 9 seconds

## Results

| Version | Normalization | Training | Val F1 | Test F1 |
|---|---|---|---|---|
| v2 | Global | 17 subjects | 0.345 | 0.473 |
| v2 + post-proc | Global | 17 subjects | 0.423 | 0.549 |
| v3 + post-proc | Per-window | 17 subjects | 0.401 | TBD |
| v4 + post-proc | Per-window | 22 subjects | — | TBD |

## Project Structure

```
src/
├── data/
│   ├── dataset.py          # PyTorch Dataset, subject-aware train/val split
│   ├── preprocessing.py    # Global and per-window normalization
│   └── augmentation.py     # Scaling, jitter, channel dropout
├── models/
│   ├── cnn1d.py            # Baseline CNN
│   ├── unet1d.py           # Attention U-Net (main model)
│   └── factory.py
├── training/
│   ├── losses.py           # BCE, Dice, Focal, Combined
│   ├── metrics.py          # Event-based F1, post-processing
│   └── trainer.py          # Training loop, early stopping
└── utils/
    └── submission.py

scripts/
├── train.py                # Main training script
├── predict.py              # Generate submission CSV
├── evaluate.py             # Threshold sweep on val set
└── sanity_check.py         # Verify setup before training

configs/config.yaml         # All hyperparameters
notebooks/eda.ipynb         # Data exploration
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate

# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install h5py pandas numpy scikit-learn pyyaml tqdm matplotlib
```

## Usage

```bash
# Verify setup
python scripts/sanity_check.py

# Train (with subject-based val split)
python scripts/train.py

# Train on all subjects for final submission
python scripts/train.py --all_subjects

# Generate submission
python scripts/predict.py
```

## Key Design Choices

- **Subject-based split:** validation set never shares subjects with training to avoid data leakage
- **Per-window normalization:** each 90s window normalized independently to remove inter-subject baseline differences
- **Event post-processing:** predicted events shorter than 9s are removed (reduces false positives without hurting recall)
- **pos_weight=10:** addresses the 93/7 class imbalance in the BCE loss
