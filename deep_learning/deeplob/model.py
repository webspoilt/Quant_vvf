"""
QUANT_VVF — DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.
Reference:
- Zhang, Zohren & Roberts (2019): DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.
  IEEE Transactions on Signal Processing, 67(14), 3701-3712.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class DeepLOB(nn.Module):
    """
    Faithful PyTorch reproduction of DeepLOB (Zhang et al. 2019):
    - Input: (Batch, 1, Sequence_Length, Features) where Features = 40 (10 levels of bid/ask price and size)
    - 3 Conv2D blocks capturing cross-level spatial dependencies
    - Inception module capturing multi-scale temporal dynamics
    - LSTM layer capturing sequential dependency
    - Fully connected linear layer predicting 3 classes: Down (0), Stationary (1), Up (2)
    """

    def __init__(
        self,
        num_classes: int = 3,
        seq_len: int = 100,
        num_features: int = 40,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_features = num_features

        # 1. Convolutional spatial blocks
        # Conv block 1: (1, 100, 40) -> (32, 100, 40) -> Conv(4,1) -> (32, 100, 40)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Conv block 2: (32, 100, 20) -> (32, 100, 10)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Conv block 3: (32, 100, 10) -> (32, 100, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # 2. Inception Module (capturing multi-scale temporal convolutions)
        # Branch 1: 1x1 conv
        self.inc_b1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Branch 2: 1x1 conv -> 5x1 conv
        self.inc_b2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Branch 3: MaxPool -> 1x1 conv
        self.inc_b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # 3. Recurrent Layer
        # Concatenated inception output channels: 64 + 64 + 64 = 192 features per time step
        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

        # 4. Dense Output
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 40) or (batch, 1, seq_len, 40)
        returns logits: (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dim (batch, 1, seq_len, 40)

        # Convolutional forward
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Inception forward
        b1 = self.inc_b1(out)
        b2 = self.inc_b2(out)
        b3 = self.inc_b3(out)
        inc = torch.cat([b1, b2, b3], dim=1)  # shape (batch, 192, seq_len, 1)

        # Reshape for LSTM: (batch, seq_len, 192)
        inc = inc.squeeze(-1).permute(0, 2, 1)

        # Recurrent forward
        lstm_out, _ = self.lstm(inc)
        last_step = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        last_step = self.dropout(last_step)

        # Logits
        logits = self.fc(last_step)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compute Softmax probabilities, predicted direction, and Shannon entropy uncertainty."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            # Shannon entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        return {
            "probabilities": probs.cpu().numpy(),
            "predicted_class": pred_class.cpu().numpy(),
            "uncertainty_entropy": entropy.cpu().numpy(),
            "class_names": {0: "DOWN", 1: "STATIONARY", 2: "UP"},
        }
