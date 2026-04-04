"""U-Net 1D with two encoder scales + skip connections.

Signal flow:
  Input (B, 8, 9000)
    → enc1 (B, f, 9000)  ──────────────────────────── skip1
    → pool ×10 (B, f, 900)
    → enc2 (B, 2f, 900)  ────────────────────────────  skip2
    → pool ×10 (B, 2f, 90)
    → bottleneck (B, 4f, 90)
    → up ×10 (B, 4f, 900)  + skip2 → dec2 (B, 2f, 900)
    → pool ×10 (B, 2f, 90)          ← aggregate back to 1Hz
    → concat(bottleneck, pooled_dec2) → head → (B, 90)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        # Residual projection if dimensions differ
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv1d(gate_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv1d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv1d(inter_ch, 1, 1, bias=True)

    def forward(self, gate, skip):
        # gate: (B, gate_ch, T_small)  — may need upsampling to match skip
        g = self.W_g(gate)
        x = self.W_x(skip)
        if g.shape[-1] != x.shape[-1]:
            g = F.interpolate(g, size=x.shape[-1], mode="linear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.gelu(g + x)))  # (B, 1, T)
        return skip * alpha


class UNet1D(nn.Module):
    """Attention U-Net for sleep apnea segmentation.

    Input:  (B, 8, 9000)
    Output: (B, 90) — logits, apply sigmoid for probabilities
    """

    def __init__(self, n_channels: int = 8, base_filters: int = 32, dropout: float = 0.2):
        super().__init__()
        f = base_filters

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(n_channels, f, kernel=9)          # (B, f, 9000)
        self.pool1 = nn.AvgPool1d(10)                           # → (B, f, 900)

        self.enc2 = ConvBlock(f, f * 2, kernel=7)               # (B, 2f, 900)
        self.pool2 = nn.AvgPool1d(10)                           # → (B, 2f, 90)

        # ── Bottleneck at 1Hz ────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBlock(f * 2, f * 4, kernel=5),
            ConvBlock(f * 4, f * 4, kernel=5, dilation=2),
            ConvBlock(f * 4, f * 4, kernel=5, dilation=4),
        )                                                        # (B, 4f, 90)

        # ── Decoder: back to 10Hz with skip from enc2 ────────────────────────
        self.attn2 = AttentionGate(gate_ch=f * 4, skip_ch=f * 2, inter_ch=f)
        self.up2 = nn.Upsample(scale_factor=10, mode="linear", align_corners=False)
        self.dec2 = ConvBlock(f * 4 + f * 2, f * 2, kernel=7)  # (B, 2f, 900)

        # Pool dec2 back to 1Hz and merge with bottleneck
        self.pool_dec2 = nn.AvgPool1d(10)                       # → (B, 2f, 90)

        # ── Head ─────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            ConvBlock(f * 4 + f * 2, f * 2, kernel=5),
            nn.Conv1d(f * 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)           # (B, f, 9000)
        e2 = self.enc2(self.pool1(e1))   # (B, 2f, 900)
        b  = self.bottleneck(self.pool2(e2))  # (B, 4f, 90)

        # Decoder
        e2_att = self.attn2(b, e2)                      # attended skip (B, 2f, 900)
        d2 = self.dec2(torch.cat([self.up2(b), e2_att], dim=1))  # (B, 2f, 900)
        d2_pooled = self.pool_dec2(d2)                   # (B, 2f, 90)

        # Merge multi-scale features at 1Hz
        out = torch.cat([b, d2_pooled], dim=1)           # (B, 6f, 90)
        out = self.dropout(out)
        out = self.head(out).squeeze(1)                  # (B, 90)
        return out
