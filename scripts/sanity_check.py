"""Quick sanity check: load data, run one forward pass, compute dummy F1.

Run this FIRST to verify everything is set up correctly before training.

Usage:
    python scripts/sanity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_data, SleepApneaDataset, subject_train_val_split
from src.data.preprocessing import compute_stats, normalize, clip_outliers
from src.models.cnn1d import CNN1D
from src.models.unet1d import UNet1D
from src.training.losses import CombinedLoss
from src.training.metrics import compute_f1_events, threshold_predictions


def main():
    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    # 1. Data loading
    print("\n[1] Loading data...")
    signals, labels, sample_ids, subject_ids = load_data(
        "X_train.h5",
        "data/X_train_h7ipJUo.csv",
        "data/y_train_tX9Br0C.csv",
    )
    print(f"  signals: {signals.shape}  dtype={signals.dtype}")
    print(f"  labels:  {labels.shape}   dtype={labels.dtype}")
    print(f"  Subjects: {np.unique(subject_ids).shape[0]}")
    print(f"  Class balance (apnea rate): {labels.mean():.3%}")

    # 2. Normalization
    print("\n[2] Normalization...")
    mean, std = compute_stats(signals)
    print(f"  mean per channel: {mean.ravel().round(2).tolist()}")
    print(f"  std  per channel: {std.ravel().round(2).tolist()}")
    signals_norm = clip_outliers(normalize(signals, mean, std))
    print(f"  After norm — min: {signals_norm.min():.2f}  max: {signals_norm.max():.2f}")

    # 3. Train/val split
    print("\n[3] Train/val split by subject...")
    X_tr, y_tr, sub_tr, X_val, y_val, sub_val = subject_train_val_split(
        signals_norm, labels, subject_ids
    )
    print(f"  Train: {len(X_tr)} windows, {np.unique(sub_tr).shape[0]} subjects")
    print(f"  Val:   {len(X_val)} windows, {np.unique(sub_val).shape[0]} subjects")
    overlap = set(sub_tr) & set(sub_val)
    assert len(overlap) == 0, f"DATA LEAKAGE: subjects {overlap} appear in both splits!"
    print("  No subject overlap — OK")

    # 4. Dataset + DataLoader
    print("\n[4] Dataset & DataLoader...")
    from torch.utils.data import DataLoader
    ds = SleepApneaDataset(X_tr[:16], y_tr[:16])
    loader = DataLoader(ds, batch_size=4)
    x_batch, y_batch = next(iter(loader))
    print(f"  x_batch: {x_batch.shape}  y_batch: {y_batch.shape}")

    # 5. Forward pass — CNN1D
    print("\n[5] CNN1D forward pass...")
    model = CNN1D(n_channels=8, base_filters=16)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n:,}")
    with torch.no_grad():
        out = model(x_batch)
    print(f"  Output shape: {out.shape}  (expected [4, 90])")
    assert out.shape == (4, 90)

    # 6. Forward pass — UNet1D
    print("\n[6] UNet1D forward pass...")
    model2 = UNet1D(n_channels=8, base_filters=16)
    n2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters: {n2:,}")
    with torch.no_grad():
        out2 = model2(x_batch)
    print(f"  Output shape: {out2.shape}  (expected [4, 90])")
    assert out2.shape == (4, 90)

    # 7. Loss
    print("\n[7] Loss...")
    loss_fn = CombinedLoss(pos_weight=3.0)
    loss = loss_fn(out2, y_batch)
    print(f"  Combined loss: {loss.item():.4f}")

    # 8. Metrics
    print("\n[8] Event-based F1 metric...")
    # Dummy all-zeros prediction (worst case)
    y_pred_zero = np.zeros_like(y_val[:50], dtype=int)
    f1, p, r = compute_f1_events(y_val[:50], y_pred_zero)
    print(f"  All-zero prediction: F1={f1:.3f}  P={p:.3f}  R={r:.3f}")

    # Random prediction
    rng = np.random.default_rng(42)
    y_pred_rand = (rng.random(y_val[:50].shape) > 0.8).astype(int)
    f1, p, r = compute_f1_events(y_val[:50], y_pred_rand)
    print(f"  Random prediction:   F1={f1:.3f}  P={p:.3f}  R={r:.3f}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — ready to train!")
    print("=" * 60)


if __name__ == "__main__":
    main()
