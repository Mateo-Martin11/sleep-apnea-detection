"""
Dataset PyTorch pour les signaux PSG du Challenge ENS #45 — Dreem.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class DreemDataset(Dataset):
    """
    Dataset PyTorch pour les signaux PSG du challenge Dreem.

    Paramètres
    ----------
    X : np.ndarray ou np.memmap, shape (N, 8, 9000)
        Signaux normalisés. Peut être un tableau memory-mapped.
    y : np.ndarray ou None, shape (N, 90)
        Labels binaires (seconde par seconde). None pour le test set.

    Notes
    -----
    Le `.copy()` dans __getitem__ est obligatoire quand X est un memmap :
    torch.from_numpy() exige un tableau C-contigu avec ownership exclusif.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X          = X
        self.y          = y
        self.has_labels = y is not None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # .copy() : matérialise la ligne depuis le memmap en RAM contiguë
        x = torch.from_numpy(self.X[idx].copy())   # (8, 9000) float32
        if self.has_labels:
            y = torch.from_numpy(self.y[idx].copy())   # (90,) float32
            return x, y
        return x
