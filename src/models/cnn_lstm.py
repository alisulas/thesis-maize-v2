"""CNN-LSTM model for maize yield prediction.

Adapted from You et al. 2017 for mean-aggregated MODIS features.
Original used histograms; here we use (T, F) scalar time-series.

Architecture:
    Input  (B, T=46, F=10)
    → CNN  per timestep: 1D conv over feature axis → abstract spatial features
    → LSTM: temporal sequence modelling
    → FC head: predict yield (ton/ha)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CropYieldCNNLSTM(nn.Module):
    """CNN-LSTM crop yield predictor.

    Args:
        n_features: Input feature dimension (default 10).
        cnn_channels: Number of 1D-conv output channels over feature axis.
        hidden_size: LSTM hidden state size.
        n_layers: LSTM stacked layers.
        dropout: Dropout applied in LSTM and FC head.
    """

    def __init__(
        self,
        n_features: int = 10,
        cnn_channels: int = 64,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # CNN: treat each timestep's feature vector as a 1-D "image" of length F
        # Conv1d expects (B, C_in, L); we use C_in=1, L=F
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B*T, cnn_channels, 1)
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            yield: (B,)
        """
        B, T, F = x.shape

        # Reshape for CNN: treat each timestep independently
        x_cnn = x.reshape(B * T, 1, F)          # (B*T, 1, F)
        feat = self.cnn(x_cnn).squeeze(-1)       # (B*T, cnn_channels)
        feat = feat.reshape(B, T, -1)            # (B, T, cnn_channels)

        # LSTM over time
        out, _ = self.lstm(feat)                 # (B, T, hidden)
        last = out[:, -1, :]                     # (B, hidden)

        return self.head(last).squeeze(-1)       # (B,)

    def freeze_feature_extractor(self) -> None:
        """Freeze CNN + LSTM; only train the FC head (for fine-tuning)."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class CropYieldLSTM(nn.Module):
    """Simpler LSTM-only baseline (no CNN).

    Use this when debugging or when CNN overhead isn't justified.
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

    def freeze_feature_extractor(self) -> None:
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
