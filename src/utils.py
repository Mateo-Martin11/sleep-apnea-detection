"""
Utilitaires généraux : reproductibilité et early stopping.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Arrête l'entraînement si la métrique surveillée ne s'améliore pas
    pendant `patience` epochs consécutives.

    Paramètres
    ----------
    patience  : int — nombre d'epochs sans amélioration avant l'arrêt
    mode      : 'max' (maximiser F1) ou 'min' (minimiser loss)
    min_delta : float — amélioration minimale considérée comme significative
    """

    def __init__(self, patience: int = 10, mode: str = 'max',
                 min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.best      = -np.inf if mode == 'max' else np.inf
        self.counter   = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Retourne True si l'entraînement doit s'arrêter."""
        improved = (
            metric > self.best + self.min_delta if self.mode == 'max'
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered
