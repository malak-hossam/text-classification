from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        model_type: Literal["RNN", "LSTM"] = "RNN",
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if model_type not in {"RNN", "LSTM"}:
            raise ValueError("model_type must be one of {'RNN', 'LSTM'}")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = getattr(nn, model_type)(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

        recurrent_feature_dim = hidden_dim * (2 if bidirectional else 1)
        combined_feature_dim = recurrent_feature_dim * 2
        self.layer_norm = nn.LayerNorm(combined_feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(combined_feature_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [batch, seq=1, dim]
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        mean_pooled = out.mean(dim=1)
        combined = torch.cat([last_hidden, mean_pooled], dim=1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        return self.fc(combined)

