"""Per-channel normalization and outlier handling."""
import numpy as np


def compute_stats(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std over the training set.

    Args:
        signals: (N, 8, 9000)
    Returns:
        mean: (8, 1)
        std:  (8, 1)
    """
    # Reshape to (8, N*9000) to compute stats across all windows
    flat = signals.transpose(1, 0, 2).reshape(signals.shape[1], -1)  # (8, N*9000)
    mean = flat.mean(axis=1, keepdims=True)   # (8, 1)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize signals per channel.

    Args:
        signals: (N, 8, 9000)
        mean: (8, 1)
        std:  (8, 1)
    Returns:
        normalized: (N, 8, 9000)
    """
    return (signals - mean[np.newaxis]) / std[np.newaxis]


def instance_normalize(signals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each window independently per channel (instance normalization).

    Each (channel, window) pair is z-scored by its own mean and std.
    This removes inter-subject baseline differences without needing training stats.

    Args:
        signals: (N, 8, 9000)
    Returns:
        normalized: (N, 8, 9000)
    """
    mean = signals.mean(axis=2, keepdims=True)        # (N, 8, 1)
    std  = signals.std(axis=2, keepdims=True) + eps   # (N, 8, 1)
    return ((signals - mean) / std).astype(np.float32)


def clip_outliers(signals: np.ndarray, n_std: float = 5.0) -> np.ndarray:
    """Clip values beyond n_std standard deviations (applied after normalization)."""
    return np.clip(signals, -n_std, n_std)


def save_stats(mean: np.ndarray, std: np.ndarray, path: str):
    np.savez(path, mean=mean, std=std)


def load_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"], data["std"]
