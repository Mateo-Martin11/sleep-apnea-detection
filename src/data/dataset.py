import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


SIGNAL_NAMES = ["AbdoBelt", "AirFlow", "PPG", "ThorBelt", "Snoring", "SPO2", "C4A1", "O2A1"]
N_SIGNALS = 8
N_TIMESTEPS = 9000   # 90s × 100Hz
N_LABELS = 90        # 90s × 1Hz


def load_data(h5_path: str, ids_csv: str, labels_csv: str = None):
    """Load signals and labels from H5 + CSV files.

    Returns:
        signals: np.ndarray (N, 8, 9000) float32
        labels:  np.ndarray (N, 90) float32, or None
        sample_ids:  np.ndarray (N,) int
        subject_ids: np.ndarray (N,) int
    """
    with h5py.File(h5_path, "r") as f:
        raw = f["data"][:]  # (N, 72002)

    sample_ids = raw[:, 0].astype(int)
    subject_ids = raw[:, 1].astype(int)
    signals = raw[:, 2:].reshape(-1, N_SIGNALS, N_TIMESTEPS).astype(np.float32)

    labels = None
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
        labels = df.iloc[:, 1:].values.astype(np.float32)  # (N, 90)

    return signals, labels, sample_ids, subject_ids


def subject_train_val_split(signals, labels, subject_ids, val_ratio=0.2, seed=42):
    """Split by subject to avoid data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx = np.arange(len(signals))
    train_idx, val_idx = next(gss.split(idx, groups=subject_ids))
    return (
        signals[train_idx], labels[train_idx], subject_ids[train_idx],
        signals[val_idx],   labels[val_idx],   subject_ids[val_idx],
    )


class SleepApneaDataset(Dataset):
    """PyTorch Dataset for sleep apnea detection."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray = None, transform=None):
        """
        Args:
            signals: (N, 8, 9000) float32
            labels:  (N, 90) float32, optional
            transform: callable applied to each signal tensor
        """
        self.signals = torch.from_numpy(signals)      # (N, 8, 9000)
        self.labels = torch.from_numpy(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx].clone()  # (8, 9000)
        if self.transform:
            x = self.transform(x)
        if self.labels is not None:
            return x, self.labels[idx]
        return x
