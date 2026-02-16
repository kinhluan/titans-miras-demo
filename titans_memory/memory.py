"""Memory module for Titans architecture."""

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """MLP-based long-term memory with surprise-gated writes and forgetting."""

    def __init__(
        self,
        input_dim: int,
        memory_dim: int = 32,
        forget_rate: float = 0.0,
        write_lr: float = 0.01,
    ):
        super().__init__()
        self.memory_net = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, input_dim),
        )
        self.forget_rate = forget_rate
        self.write_lr = write_lr

    def read(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.memory_net(x)

    def write(self, x: torch.Tensor, surprise_score: torch.Tensor) -> None:
        if surprise_score.item() < 1e-8:
            return
        prediction = self.memory_net(x)
        loss = torch.mean((prediction - x) ** 2)
        self.memory_net.zero_grad()
        loss.backward()
        effective_lr = self.write_lr * surprise_score.item()
        with torch.no_grad():
            for param in self.memory_net.parameters():
                param -= effective_lr * param.grad

    def apply_forgetting(self) -> None:
        if self.forget_rate <= 0:
            return
        with torch.no_grad():
            for param in self.memory_net.parameters():
                param *= (1.0 - self.forget_rate)

    def get_weight_snapshot(self) -> list[torch.Tensor]:
        return [p.data.clone() for p in self.memory_net.parameters()]
