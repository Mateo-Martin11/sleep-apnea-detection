"""Data augmentation transforms for time-series signals."""
import torch
import numpy as np


class RandomScaling:
    """Scale signal amplitude by a random factor."""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.FloatTensor(x.shape[0], 1).uniform_(*self.scale_range)
        return x * scale


class RandomJitter:
    """Add Gaussian noise."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class RandomChannelDropout:
    """Randomly zero out a channel (forces model to not rely on one signal)."""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1).item() < self.p:
                x[i] = 0.0
        return x


class RandomTimeShift:
    """Shift signal in time with wrap-around (only valid for training)."""
    def __init__(self, max_shift=500):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=-1)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transform():
    return Compose([
        RandomScaling(scale_range=(0.85, 1.15)),
        RandomJitter(sigma=0.02),
        RandomChannelDropout(p=0.05),
    ])
