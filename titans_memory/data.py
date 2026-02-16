"""Synthetic data generators for demonstrating surprise-based memory."""

import torch
import math


def generate_repeating_with_anomalies(
    pattern: list[float],
    repeats: int,
    anomaly_indices: list[int],
    anomaly_value: float = 99.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a repeating pattern with anomalies at specified positions."""
    seq = torch.tensor(pattern * repeats, dtype=torch.float32)
    anomaly_mask = torch.zeros_like(seq)
    for idx in anomaly_indices:
        if idx < len(seq):
            seq[idx] = anomaly_value
            anomaly_mask[idx] = 1.0
    return seq, anomaly_mask


def generate_frequency_shift(
    length: int,
    base_freq: float,
    shifted_freq: float,
    shift_at: list[int],
) -> tuple[torch.Tensor, list[int]]:
    """Generate a sine wave that changes frequency at specified points."""
    seq = torch.zeros(length, dtype=torch.float32)
    current_freq = base_freq
    shift_set = sorted(shift_at)
    shift_idx = 0

    for t in range(length):
        if shift_idx < len(shift_set) and t >= shift_set[shift_idx]:
            current_freq = shifted_freq if current_freq == base_freq else base_freq
            shift_idx += 1
        seq[t] = math.sin(2 * math.pi * current_freq * t)

    return seq, shift_at
