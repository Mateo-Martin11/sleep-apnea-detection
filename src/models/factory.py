"""Model factory."""
from .cnn1d import CNN1D
from .unet1d import UNet1D


def build_model(name: str, n_channels: int = 8, base_filters: int = 32, **kwargs):
    if name == "cnn1d":
        return CNN1D(n_channels=n_channels, base_filters=base_filters)
    elif name == "unet1d":
        return UNet1D(n_channels=n_channels, base_filters=base_filters, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from ['cnn1d', 'unet1d']")
