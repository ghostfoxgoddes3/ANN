"""
Modelo recorrente para classificação de janelas temporais.

Uma única classe cobre LSTM e GRU (selecionável por hiperparâmetro), seguida
de uma cabeça linear que produz o logit da classe "vai chover". Trabalhamos
com logits (sem sigmoid na saída) para usar BCEWithLogitsLoss, que é
numericamente estável e aceita peso de classe para o desbalanceamento.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RainClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        cell: str = "LSTM",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell = cell.upper()

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        output, _ = self.rnn(x)
        last = output[:, -1, :]          # estado do último instante da janela
        logit = self.head(last)          # (batch, 1)
        return logit.squeeze(-1)         # (batch,)
