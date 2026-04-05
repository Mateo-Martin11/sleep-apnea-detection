# Sleep Apnea Detection

Automatic detection of sleep apnea events from raw polysomnography (PSG) signals.
Built for [ENS Challenge #45](https://challengedata.ens.fr/participants/challenges/45/).

**Best score: 0.636 F1 — 3rd place** (benchmark: 0.525)

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
- Apnea rate: ~6.9% (heavy class imbalance, 93/7)
- Average event duration: 18 seconds (min: 1s, max: 70s)
- 79.5% of windows have no apnea event

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
  → Encoder block 1 : Conv1D(8,  64, k=9)  + residual  →  (B, 64,  9000)
  → AvgPool1D(10)                                        →  (B, 64,  900)
  → Encoder block 2 : Conv1D(64, 128, k=7) + residual  →  (B, 128, 900)
  → AvgPool1D(10)                                        →  (B, 128, 90)
  → Bottleneck      : 3× Conv1D dilated (d=1,2,4)       →  (B, 256, 90)
  → Decoder         : upsample + attention gate + concat →  (B, 128, 900)
  → AvgPool1D(10)   : back to 1Hz                        →  (B, 128, 90)
  → Head            : Conv1D(384, 1)                     →  (B, 90)
```

**Loss:** BCE (pos_weight=10) + Dice Loss (50/50)

## Experimental Results

All experiments use:
- Instance normalization (per window, per channel)
- 5-fold GroupKFold cross-validation for hyperparameter calibration
- Subject-based splits to avoid data leakage

| Version | Model | Normalization | Subjects | Threshold | Min duration | Test F1 |
|---|---|---|---|---|---|---|
| v2 | UNet1D (base_filters=32) | Global | 17/22 | 0.60 | — | 0.473 |
| v2 + post-proc | UNet1D (base_filters=32) | Global | 17/22 | 0.60 | 9s | 0.549 |
| v3 | UNet1D (base_filters=32) | Per-window | 17/22 | 0.60 | 9s | 0.614 |
| v4 | UNet1D (base_filters=32) | Per-window | 22/22 | 0.60 | 9s | 0.624 |
| v5 | UNet1D (base_filters=64) | Per-window | 22/22 | 0.60 | 9s | 0.630 |
| **v5** | **UNet1D (base_filters=64)** | **Per-window** | **22/22** | **0.70** | **9s** | **0.636** |
| v5 + CV | UNet1D (base_filters=64) | Per-window | 22/22 | 0.70 | 10s | TBD |

### Key findings

| Improvement | Delta F1 | Method |
|---|---|---|
| Post-processing (min event duration) | +0.076 | Remove predicted events < 9s |
| Instance normalization | +0.065 | Normalize each window independently |
| Train on all 22 subjects | +0.010 | No held-out val split for training |
| Larger model (32→64 filters) | +0.006 | 745K → 3M parameters |
| Higher threshold (0.60→0.70) | +0.006 | Reduce false positives |

### Hyperparameter calibration (5-fold CV)

Threshold and min_duration were calibrated via 5-fold GroupKFold cross-validation on all 4400 training windows (no data leakage):

- **OOF F1: 0.5758** at threshold=0.70, min_duration=10s
- Grid search over threshold ∈ [0.40, 0.80] × min_duration ∈ [5, 15]

## Training Configuration

```yaml
model:
  name: unet1d
  base_filters: 64
  dropout: 0.4

training:
  batch_size: 32
  n_epochs: 12        # fixed (determined by early stopping on val split)
  lr: 0.001
  weight_decay: 0.001

loss:
  name: combined      # BCE (pos_weight=10) + Dice
  pos_weight: 10.0

data:
  instance_norm: true
```

## Project Structure

```
src/
├── data/
│   ├── dataset.py          # PyTorch Dataset, subject-aware train/val split
│   ├── preprocessing.py    # Global and per-window (instance) normalization
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
├── train.py                # Training (with or without val split)
├── cross_validate.py       # 5-fold CV — calibrates threshold & min_duration
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

# GPU (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install h5py pandas numpy scikit-learn pyyaml tqdm matplotlib
```

## Usage

```bash
# 1. Verify setup
python scripts/sanity_check.py

# 2. Run 5-fold CV to find optimal threshold and min_duration
python scripts/cross_validate.py

# 3. Train on all subjects with fixed epochs
python scripts/train.py --all_subjects

# 4. Generate submission with CV-calibrated hyperparameters
python scripts/predict.py --threshold 0.70 --min_duration 10
```

## Leaderboard

| Rank | Participant | Score |
|---|---|---|
| 1 | leo_h | 0.6699 |
| 2 | clementg & LB | 0.6503 |
| **3** | **Mateo & Theo** | **0.636** |
| 16 | benchmark | 0.5254 |
