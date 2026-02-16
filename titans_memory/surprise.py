"""Surprise metric for Titans memory mechanism."""

import torch
import torch.nn as nn


class SurpriseMetric(nn.Module):
    """Computes surprise score for incoming data."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, momentum: float = 0.0):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.momentum = momentum
        self._ema_surprise: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction = self.predictor(x)
        surprise = torch.mean((prediction - x) ** 2)
        return surprise

    def update_predictor(self, x: torch.Tensor, lr: float = 0.01) -> None:
        prediction = self.predictor(x)
        loss = torch.mean((prediction - x) ** 2)
        self.predictor.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.predictor.parameters():
                param -= lr * param.grad

    def compute_with_momentum(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward(x)
        if self._ema_surprise is None:
            self._ema_surprise = raw.clone()
        else:
            self._ema_surprise = self.momentum * self._ema_surprise + (1 - self.momentum) * raw
        return self._ema_surprise.clone()

    def reset_momentum(self) -> None:
        self._ema_surprise = None
