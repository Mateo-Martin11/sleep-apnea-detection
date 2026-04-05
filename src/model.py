"""
Architecture DreemNet : CNN 1D + BiLSTM pour la segmentation temporelle d'apnées.

Flux des données :
    (batch, 8, 9000)
    → CNN encoder : (batch, 128, 90)   [réduction ×100 exacte]
    → permute     : (batch, 90, 128)
    → BiLSTM      : (batch, 90, 128)   [64 × 2 directions]
    → Dropout(0.3)
    → Linear      : (batch, 90, 1)
    → squeeze     : (batch, 90)        [logits bruts]
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc Conv1d + BatchNorm1d + ReLU réutilisable."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DreemNet(nn.Module):
    """
    Architecture CNN 1D + BiLSTM pour la segmentation temporelle d'apnées.

    Réduction temporelle exacte ×100 : 9000 → 1800 → 360 → 90
    Vérification analytique :
        Conv1(k=11, s=5, p=5) : floor((9000+10-11)/5)+1 = 1800
        Conv2(k=9,  s=5, p=4) : floor((1800+8-9)/5)+1   = 360
        Conv3(k=5,  s=4, p=2) : floor((360+4-5)/4)+1    = 90

    Paramètres
    ----------
    n_channels   : int — nombre de canaux d'entrée (8)
    cnn_channels : list[int] — filtres par bloc CNN ([32, 64, 128])
    lstm_hidden  : int — taille de l'état caché LSTM par direction (64)
    lstm_layers  : int — nombre de couches LSTM empilées (2)
    dropout      : float — taux de dropout dans LSTM et avant la tête (0.3)
    """

    def __init__(
        self,
        n_channels: int = 8,
        cnn_channels: list = None,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # ── Encodeur CNN 1D ────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            ConvBlock(n_channels,         cnn_channels[0], 11, stride=5, padding=5),
            ConvBlock(cnn_channels[0],    cnn_channels[1],  9, stride=5, padding=4),
            ConvBlock(cnn_channels[1],    cnn_channels[2],  5, stride=4, padding=2),
        )

        # ── BiLSTM ─────────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # ── Tête de classification ─────────────────────────────────────────────
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètres
        ----------
        x : (batch, 8, 9000)

        Retourne
        --------
        logits : (batch, 90) — logits bruts (pas de sigmoid)
        """
        x = self.encoder(x)           # (batch, 128, 90)
        x = x.permute(0, 2, 1)        # (batch, 90, 128)
        x, _ = self.bilstm(x)         # (batch, 90, 128)
        x = self.dropout(x)
        x = self.classifier(x).squeeze(-1)   # (batch, 90)
        return x
