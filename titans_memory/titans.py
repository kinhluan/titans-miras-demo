"""Full Titans memory layer combining surprise metric + memory module."""

import torch
from titans_memory.surprise import SurpriseMetric
from titans_memory.memory import MemoryModule


class TitansMemoryLayer:
    """Complete Titans-style memory layer."""

    def __init__(
        self,
        input_dim: int,
        memory_dim: int = 32,
        hidden_dim: int = 16,
        momentum: float = 0.0,
        forget_rate: float = 0.01,
        surprise_lr: float = 0.01,
        write_lr: float = 0.01,
    ):
        self.surprise_metric = SurpriseMetric(
            input_dim=input_dim, hidden_dim=hidden_dim, momentum=momentum
        )
        self.memory = MemoryModule(
            input_dim=input_dim, memory_dim=memory_dim,
            forget_rate=forget_rate, write_lr=write_lr,
        )
        self.surprise_lr = surprise_lr

    def process_sequence(self, seq: torch.Tensor) -> dict:
        outputs = []
        surprise_scores = []
        surprise_momentum = []
        memory_snapshots = []

        self.surprise_metric.reset_momentum()

        for t in range(seq.shape[0]):
            x = seq[t]
            raw_surprise = self.surprise_metric(x)
            momentum_surprise = self.surprise_metric.compute_with_momentum(x)
            mem_output = self.memory.read(x)
            self.memory.write(x, surprise_score=raw_surprise)
            self.surprise_metric.update_predictor(x, lr=self.surprise_lr)
            self.memory.apply_forgetting()
            outputs.append(mem_output.clone())
            surprise_scores.append(raw_surprise.clone())
            surprise_momentum.append(momentum_surprise.clone())
            memory_snapshots.append(self.memory.get_weight_snapshot())

        return {
            "outputs": outputs,
            "surprise_scores": surprise_scores,
            "surprise_momentum": surprise_momentum,
            "memory_snapshots": memory_snapshots,
        }
