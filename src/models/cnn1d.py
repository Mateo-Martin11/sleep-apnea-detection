"""Baseline CNN1D model: 3 conv blocks with two 10x downsampling stages."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):
    """Simple fully-convolutional baseline.

    Input:  (B, 8, 9000)
    Output: (B, 90)
    """

    def __init__(self, n_channels: int = 8, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # 100Hz → 10Hz (9000 → 900)
        self.enc1 = ConvBlock(n_channels, f, kernel=7)
        self.pool1 = nn.AvgPool1d(kernel_size=10)

        # 10Hz → 1Hz (900 → 90)
        self.enc2 = ConvBlock(f, f * 2, kernel=7)
        self.pool2 = nn.AvgPool1d(kernel_size=10)

        # 1Hz processing with dilated convs for longer context
        self.enc3 = nn.Sequential(
            ConvBlock(f * 2, f * 4, kernel=5),
            ConvBlock(f * 4, f * 4, kernel=5, dilation=2),
            ConvBlock(f * 4, f * 4, kernel=5, dilation=4),
        )

        self.head = nn.Conv1d(f * 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 9000)
        x = self.pool1(self.enc1(x))   # (B, f, 900)
        x = self.pool2(self.enc2(x))   # (B, 2f, 90)
        x = self.enc3(x)               # (B, 4f, 90)
        x = self.head(x).squeeze(1)    # (B, 90)
        return x
