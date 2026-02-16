# Titans MIRAs Surprise-Based Memory Demo — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an educational Python/PyTorch demo of the Titans surprise-based memory mechanism with visualizations in Jupyter notebooks.

**Architecture:** A `titans_memory` Python package containing modular components (surprise metric, memory module, full layer, data generators). Notebooks import the package and build understanding step-by-step. A standalone script generates all figures as PNGs.

**Tech Stack:** Python 3.11+, PyTorch, matplotlib, numpy, uv, Jupyter

---

### Task 1: Project Scaffolding with uv

**Files:**
- Create: `pyproject.toml`
- Create: `titans_memory/__init__.py`
- Create: `.gitignore`
- Create: `figures/.gitkeep`
- Create: `scripts/.gitkeep`
- Create: `notebooks/.gitkeep`

**Step 1: Initialize uv project**

Run: `uv init --no-readme --python ">=3.11"`

**Step 2: Replace pyproject.toml with full config**

```toml
[project]
name = "titans-miras-demo"
version = "0.1.0"
description = "Educational demo of Titans surprise-based memory mechanism"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "jupyter>=1.0",
    "ipykernel>=6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

**Step 3: Create directory structure and package init**

Create directories: `titans_memory/`, `tests/`, `scripts/`, `figures/`, `notebooks/`

`titans_memory/__init__.py`:
```python
"""Titans surprise-based memory mechanism — educational demo."""
```

**Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
.ipynb_checkpoints/
figures/*.png
```

**Step 5: Install dependencies**

Run: `uv sync`
Expected: lockfile created, dependencies installed

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .python-version titans_memory/__init__.py figures/.gitkeep scripts/.gitkeep notebooks/.gitkeep
git commit -m "feat: scaffold project with uv, torch, matplotlib dependencies"
```

---

### Task 2: Synthetic Data Generator

**Files:**
- Create: `titans_memory/data.py`
- Create: `tests/test_data.py`

**Step 1: Write failing test**

`tests/test_data.py`:
```python
import torch
from titans_memory.data import generate_repeating_with_anomalies, generate_frequency_shift


def test_repeating_with_anomalies_shape():
    seq, anomaly_mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=10, anomaly_indices=[7, 15], anomaly_value=99.0
    )
    assert seq.shape == (30,)
    assert anomaly_mask.shape == (30,)


def test_repeating_with_anomalies_values():
    seq, anomaly_mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=4, anomaly_indices=[5], anomaly_value=99.0
    )
    # Index 5 should be anomaly
    assert seq[5].item() == 99.0
    assert anomaly_mask[5].item() == 1.0
    # Non-anomaly positions should follow pattern
    assert seq[0].item() == 1.0
    assert seq[1].item() == 2.0
    assert anomaly_mask[0].item() == 0.0


def test_frequency_shift_shape():
    seq, shift_points = generate_frequency_shift(
        length=200, base_freq=0.1, shifted_freq=0.5, shift_at=[100]
    )
    assert seq.shape == (200,)
    assert shift_points == [100]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — ImportError

**Step 3: Implement data generators**

`titans_memory/data.py`:
```python
"""Synthetic data generators for demonstrating surprise-based memory."""

import torch
import math


def generate_repeating_with_anomalies(
    pattern: list[float],
    repeats: int,
    anomaly_indices: list[int],
    anomaly_value: float = 99.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a repeating pattern with anomalies at specified positions.

    Args:
        pattern: Base pattern to repeat (e.g. [1, 2, 3])
        repeats: Number of times to repeat
        anomaly_indices: Positions to inject anomaly values
        anomaly_value: Value to inject at anomaly positions

    Returns:
        (sequence, anomaly_mask) — both 1D tensors of length len(pattern)*repeats
    """
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
    """Generate a sine wave that changes frequency at specified points.

    Args:
        length: Total sequence length
        base_freq: Frequency before shift
        shifted_freq: Frequency after shift
        shift_at: List of positions where frequency changes

    Returns:
        (sequence, shift_points)
    """
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add titans_memory/data.py tests/test_data.py
git commit -m "feat: add synthetic data generators with anomalies and frequency shifts"
```

---

### Task 3: Surprise Metric

**Files:**
- Create: `titans_memory/surprise.py`
- Create: `tests/test_surprise.py`

**Step 1: Write failing test**

`tests/test_surprise.py`:
```python
import torch
from titans_memory.surprise import SurpriseMetric


def test_surprise_metric_output_shape():
    metric = SurpriseMetric(input_dim=1, hidden_dim=16)
    x = torch.randn(1)
    surprise_score = metric(x)
    assert surprise_score.shape == ()  # scalar


def test_high_surprise_for_novel_input():
    torch.manual_seed(42)
    metric = SurpriseMetric(input_dim=1, hidden_dim=16)

    # Feed the same value many times so the metric "learns" it
    familiar = torch.tensor([1.0])
    for _ in range(50):
        metric(familiar)
        metric.update_predictor(familiar, lr=0.01)

    surprise_familiar = metric(familiar)

    # Novel input should have higher surprise
    novel = torch.tensor([99.0])
    surprise_novel = metric(novel)

    assert surprise_novel.item() > surprise_familiar.item()


def test_momentum_smooths_surprise():
    metric = SurpriseMetric(input_dim=1, hidden_dim=16, momentum=0.9)
    scores = []
    for val in [1.0, 1.0, 1.0, 99.0, 1.0, 1.0]:
        x = torch.tensor([val])
        score = metric.compute_with_momentum(x)
        scores.append(score.item())

    # After the spike at index 3, index 4 should still be elevated due to momentum
    assert scores[4] > scores[0]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_surprise.py -v`
Expected: FAIL — ImportError

**Step 3: Implement SurpriseMetric**

`titans_memory/surprise.py`:
```python
"""Surprise metric for Titans memory mechanism.

The surprise metric measures how unexpected a new input is relative to
what the memory's internal predictor expects. High surprise → strong
memory update. Low surprise → skip update.
"""

import torch
import torch.nn as nn


class SurpriseMetric(nn.Module):
    """Computes surprise score for incoming data.

    Uses a small predictor network. Surprise = MSE between
    the predictor's output and the actual input.
    """

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
        """Compute raw surprise score (MSE between prediction and actual).

        Args:
            x: Input tensor of shape (input_dim,)

        Returns:
            Scalar surprise score
        """
        with torch.no_grad():
            prediction = self.predictor(x)
        surprise = torch.mean((prediction - x) ** 2)
        return surprise

    def update_predictor(self, x: torch.Tensor, lr: float = 0.01) -> None:
        """Update the predictor to better predict the given input.

        Args:
            x: The actual input to learn from
            lr: Learning rate for the update
        """
        prediction = self.predictor(x)
        loss = torch.mean((prediction - x) ** 2)
        self.predictor.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.predictor.parameters():
                param -= lr * param.grad

    def compute_with_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute surprise with exponential moving average smoothing.

        Args:
            x: Input tensor

        Returns:
            Momentum-smoothed surprise score
        """
        raw = self.forward(x)
        if self._ema_surprise is None:
            self._ema_surprise = raw.clone()
        else:
            self._ema_surprise = self.momentum * self._ema_surprise + (1 - self.momentum) * raw
        return self._ema_surprise.clone()

    def reset_momentum(self) -> None:
        """Reset the EMA state."""
        self._ema_surprise = None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_surprise.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add titans_memory/surprise.py tests/test_surprise.py
git commit -m "feat: add SurpriseMetric with predictor network and momentum"
```

---

### Task 4: Memory Module

**Files:**
- Create: `titans_memory/memory.py`
- Create: `tests/test_memory.py`

**Step 1: Write failing test**

`tests/test_memory.py`:
```python
import torch
from titans_memory.memory import MemoryModule


def test_memory_module_output_shape():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)
    output = mem.read(x)
    assert output.shape == (4,)


def test_memory_write_changes_state():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)

    output_before = mem.read(x).clone()
    mem.write(x, surprise_score=torch.tensor(1.0))
    output_after = mem.read(x)

    # After a high-surprise write, the output should change
    assert not torch.allclose(output_before, output_after)


def test_low_surprise_minimal_update():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)

    output_before = mem.read(x).clone()
    mem.write(x, surprise_score=torch.tensor(0.0))
    output_after = mem.read(x)

    # Zero surprise → no update (weights unchanged)
    assert torch.allclose(output_before, output_after)


def test_forgetting_decays_weights():
    mem = MemoryModule(input_dim=4, memory_dim=16, forget_rate=0.1)
    x = torch.randn(4)

    # Write something strongly
    mem.write(x, surprise_score=torch.tensor(1.0))
    output_right_after = mem.read(x).clone()

    # Apply forgetting many times
    for _ in range(50):
        mem.apply_forgetting()

    output_after_forget = mem.read(x)

    # Output magnitude should decrease after forgetting
    assert output_after_forget.norm().item() < output_right_after.norm().item()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL — ImportError

**Step 3: Implement MemoryModule**

`titans_memory/memory.py`:
```python
"""Memory module for Titans architecture.

Uses a small MLP as long-term memory storage. The MLP's weights ARE the memory.
Writing to memory = updating MLP weights, gated by surprise score.
Reading from memory = forward pass through MLP.
"""

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """MLP-based long-term memory with surprise-gated writes and forgetting.

    The key insight: the network's weights store learned associations.
    High-surprise inputs cause large weight updates (strong memorization).
    Low-surprise inputs cause minimal updates (already known, skip).
    """

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
        """Read from memory: forward pass through the MLP.

        Args:
            x: Query tensor of shape (input_dim,)

        Returns:
            Memory output of shape (input_dim,)
        """
        with torch.no_grad():
            return self.memory_net(x)

    def write(self, x: torch.Tensor, surprise_score: torch.Tensor) -> None:
        """Write to memory: update MLP weights, gated by surprise.

        The effective learning rate = write_lr * surprise_score.
        So high surprise → big update, low/zero surprise → no update.

        Args:
            x: Input to memorize, shape (input_dim,)
            surprise_score: Scalar surprise value (0 = familiar, high = novel)
        """
        if surprise_score.item() < 1e-8:
            return  # Skip update for zero surprise

        prediction = self.memory_net(x)
        loss = torch.mean((prediction - x) ** 2)
        self.memory_net.zero_grad()
        loss.backward()

        effective_lr = self.write_lr * surprise_score.item()
        with torch.no_grad():
            for param in self.memory_net.parameters():
                param -= effective_lr * param.grad

    def apply_forgetting(self) -> None:
        """Apply weight decay to simulate forgetting old information."""
        if self.forget_rate <= 0:
            return
        with torch.no_grad():
            for param in self.memory_net.parameters():
                param *= (1.0 - self.forget_rate)

    def get_weight_snapshot(self) -> list[torch.Tensor]:
        """Return a snapshot of all memory weights (for visualization)."""
        return [p.data.clone() for p in self.memory_net.parameters()]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_memory.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add titans_memory/memory.py tests/test_memory.py
git commit -m "feat: add MemoryModule with surprise-gated writes and forgetting"
```

---

### Task 5: Full TitansMemoryLayer

**Files:**
- Create: `titans_memory/titans.py`
- Create: `tests/test_titans.py`

**Step 1: Write failing test**

`tests/test_titans.py`:
```python
import torch
from titans_memory.titans import TitansMemoryLayer


def test_titans_layer_processes_sequence():
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, momentum=0.9)
    seq = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 99.0, 1.0, 2.0])
    result = layer.process_sequence(seq.unsqueeze(-1))  # (9, 1)

    assert "outputs" in result
    assert "surprise_scores" in result
    assert "memory_snapshots" in result
    assert len(result["surprise_scores"]) == 9


def test_anomaly_has_high_surprise():
    torch.manual_seed(42)
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16)

    # Repeating pattern then anomaly
    values = [1.0, 2.0, 3.0] * 10 + [99.0]
    seq = torch.tensor(values).unsqueeze(-1)  # (31, 1)
    result = layer.process_sequence(seq)

    scores = result["surprise_scores"]
    # Last element (99.0) should have higher surprise than average of repeating part
    avg_normal = sum(s.item() for s in scores[:30]) / 30
    anomaly_surprise = scores[30].item()
    assert anomaly_surprise > avg_normal
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_titans.py -v`
Expected: FAIL — ImportError

**Step 3: Implement TitansMemoryLayer**

`titans_memory/titans.py`:
```python
"""Full Titans memory layer combining surprise metric + memory module.

This is the main entry point: feed it a sequence, it processes each step,
decides what to memorize based on surprise, and returns the full trace
(outputs, surprise scores, memory snapshots) for visualization.
"""

import torch
from titans_memory.surprise import SurpriseMetric
from titans_memory.memory import MemoryModule


class TitansMemoryLayer:
    """Complete Titans-style memory layer.

    For each timestep:
    1. Compute surprise (how unexpected is this input?)
    2. Read from memory (what does memory predict?)
    3. Write to memory gated by surprise (memorize if novel)
    4. Apply forgetting (decay old information)
    """

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
        """Process a full sequence through the Titans memory layer.

        Args:
            seq: Input tensor of shape (seq_len, input_dim)

        Returns:
            Dict with keys:
            - outputs: list of memory read outputs per timestep
            - surprise_scores: list of surprise scores per timestep
            - surprise_momentum: list of momentum-smoothed surprises
            - memory_snapshots: list of weight snapshots per timestep
        """
        outputs = []
        surprise_scores = []
        surprise_momentum = []
        memory_snapshots = []

        self.surprise_metric.reset_momentum()

        for t in range(seq.shape[0]):
            x = seq[t]

            # 1. Compute surprise
            raw_surprise = self.surprise_metric(x)
            momentum_surprise = self.surprise_metric.compute_with_momentum(x)

            # 2. Read from memory
            mem_output = self.memory.read(x)

            # 3. Write to memory (gated by surprise)
            self.memory.write(x, surprise_score=raw_surprise)

            # 4. Update surprise predictor (so it learns the pattern)
            self.surprise_metric.update_predictor(x, lr=self.surprise_lr)

            # 5. Apply forgetting
            self.memory.apply_forgetting()

            # Record trace
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_titans.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add titans_memory/titans.py tests/test_titans.py
git commit -m "feat: add TitansMemoryLayer combining surprise + memory + forgetting"
```

---

### Task 6: Update Package __init__.py

**Files:**
- Modify: `titans_memory/__init__.py`

**Step 1: Update exports**

`titans_memory/__init__.py`:
```python
"""Titans surprise-based memory mechanism — educational demo."""

from titans_memory.surprise import SurpriseMetric
from titans_memory.memory import MemoryModule
from titans_memory.titans import TitansMemoryLayer
from titans_memory.data import generate_repeating_with_anomalies, generate_frequency_shift

__all__ = [
    "SurpriseMetric",
    "MemoryModule",
    "TitansMemoryLayer",
    "generate_repeating_with_anomalies",
    "generate_frequency_shift",
]
```

**Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All 12 tests pass

**Step 3: Commit**

```bash
git add titans_memory/__init__.py
git commit -m "feat: export all public API from package __init__"
```

---

### Task 7: Generate Figures Script

**Files:**
- Create: `scripts/generate_figures.py`

**Step 1: Implement the script**

`scripts/generate_figures.py`:
```python
"""Generate all demo figures as PNG files in figures/ directory.

Usage: uv run python scripts/generate_figures.py
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from titans_memory import (
    TitansMemoryLayer,
    generate_repeating_with_anomalies,
    generate_frequency_shift,
)

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def fig1_surprise_over_time():
    """Surprise scores over time — spikes at anomalies."""
    torch.manual_seed(42)
    seq, mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=20, anomaly_indices=[14, 35, 50], anomaly_value=99.0
    )
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, momentum=0.9)
    result = layer.process_sequence(seq.unsqueeze(-1))

    scores = [s.item() for s in result["surprise_scores"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: input sequence
    ax1.plot(seq.numpy(), color="#2196F3", linewidth=1, label="Input sequence")
    anomaly_idx = mask.nonzero().squeeze(-1).numpy()
    ax1.scatter(anomaly_idx, seq[mask.bool()].numpy(), color="#F44336", s=80, zorder=5, label="Anomalies")
    ax1.set_ylabel("Input Value")
    ax1.set_title("Titans Surprise-Based Memory: Input Sequence & Surprise Response")
    ax1.legend()

    # Bottom: surprise scores
    ax2.plot(scores, color="#FF9800", linewidth=1.5, label="Raw surprise")
    for idx in anomaly_idx:
        ax2.axvline(x=idx, color="#F44336", alpha=0.3, linestyle="--")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Surprise Score")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "surprise_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: surprise_over_time.png")


def fig2_memory_update_heatmap():
    """Heatmap of memory weight changes over time."""
    torch.manual_seed(42)
    seq, _ = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0, 4.0], repeats=15, anomaly_indices=[20, 40], anomaly_value=50.0
    )
    layer = TitansMemoryLayer(input_dim=1, memory_dim=8, hidden_dim=8)
    result = layer.process_sequence(seq.unsqueeze(-1))

    # Track first layer weight changes
    snapshots = result["memory_snapshots"]
    first_layer_weights = [s[0].numpy() for s in snapshots]

    # Compute change magnitude per timestep
    changes = []
    for i in range(1, len(first_layer_weights)):
        diff = np.abs(first_layer_weights[i] - first_layer_weights[i - 1])
        changes.append(diff.flatten()[:16])  # take first 16 weights

    changes = np.array(changes).T

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(changes, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight Index")
    ax.set_title("Memory Weight Update Intensity (brighter = bigger update)")
    plt.colorbar(im, ax=ax, label="Δ|weight|")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "memory_update_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: memory_update_heatmap.png")


def fig3_momentum_vs_raw():
    """Compare raw surprise vs momentum-smoothed surprise."""
    torch.manual_seed(42)
    seq, mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=20, anomaly_indices=[15, 30, 45], anomaly_value=80.0
    )
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, momentum=0.9)
    result = layer.process_sequence(seq.unsqueeze(-1))

    raw = [s.item() for s in result["surprise_scores"]]
    momentum = [s.item() for s in result["surprise_momentum"]]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw, color="#FF9800", alpha=0.5, linewidth=1, label="Raw surprise")
    ax.plot(momentum, color="#E91E63", linewidth=2, label="With momentum (β=0.9)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Surprise Score")
    ax.set_title("Raw Surprise vs Momentum-Smoothed Surprise")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "momentum_vs_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: momentum_vs_raw.png")


def fig4_forgetting_effect():
    """Show how forgetting decays memory over time."""
    torch.manual_seed(42)

    # Run with different forget rates
    forget_rates = [0.0, 0.005, 0.02, 0.05]
    fig, ax = plt.subplots(figsize=(12, 4))

    for rate in forget_rates:
        seq, _ = generate_repeating_with_anomalies(
            pattern=[1.0, 2.0, 3.0], repeats=20, anomaly_indices=[10], anomaly_value=50.0
        )
        layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, forget_rate=rate)
        result = layer.process_sequence(seq.unsqueeze(-1))

        # Track memory norm over time
        norms = []
        for snapshot in result["memory_snapshots"]:
            total_norm = sum(w.norm().item() for w in snapshot)
            norms.append(total_norm)

        ax.plot(norms, linewidth=2, label=f"forget_rate={rate}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Memory Weight Norm")
    ax.set_title("Effect of Forgetting Rate on Memory Persistence")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "forgetting_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: forgetting_effect.png")


def fig5_full_pipeline():
    """Combined dashboard showing the full Titans pipeline."""
    torch.manual_seed(42)
    seq, mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0, 4.0, 5.0], repeats=12,
        anomaly_indices=[12, 30, 48], anomaly_value=50.0,
    )
    layer = TitansMemoryLayer(
        input_dim=1, memory_dim=16, hidden_dim=16, momentum=0.9, forget_rate=0.01
    )
    result = layer.process_sequence(seq.unsqueeze(-1))

    scores = [s.item() for s in result["surprise_scores"]]
    momentum = [s.item() for s in result["surprise_momentum"]]
    anomaly_idx = mask.nonzero().squeeze(-1).numpy()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # 1. Input sequence
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(seq.numpy(), color="#2196F3", linewidth=1)
    ax1.scatter(anomaly_idx, seq[mask.bool()].numpy(), color="#F44336", s=80, zorder=5)
    ax1.set_title("① Input Sequence (red = anomalies)")
    ax1.set_ylabel("Value")

    # 2. Surprise scores
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(scores, color="#FF9800", linewidth=1.5)
    ax2.set_title("② Raw Surprise Score")
    ax2.set_ylabel("Surprise")

    # 3. Momentum surprise
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(momentum, color="#E91E63", linewidth=1.5)
    ax3.set_title("③ Momentum-Smoothed Surprise")
    ax3.set_ylabel("Surprise")

    # 4. Memory weight changes
    snapshots = result["memory_snapshots"]
    first_layer = [s[0].numpy() for s in snapshots]
    changes = []
    for i in range(1, len(first_layer)):
        diff = np.abs(first_layer[i] - first_layer[i - 1])
        changes.append(diff.flatten()[:12])
    changes = np.array(changes).T

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.imshow(changes, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax4.set_title("④ Memory Update Intensity")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Weight Index")

    # 5. Memory norm over time
    norms = [sum(w.norm().item() for w in s) for s in snapshots]
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(norms, color="#4CAF50", linewidth=2)
    ax5.set_title("⑤ Memory Weight Norm (Forgetting Effect)")
    ax5.set_xlabel("Timestep")
    ax5.set_ylabel("Norm")

    fig.suptitle("Titans Surprise-Based Memory — Full Pipeline", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIGURES_DIR / "full_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: full_pipeline.png")


if __name__ == "__main__":
    print("Generating all figures...")
    fig1_surprise_over_time()
    fig2_memory_update_heatmap()
    fig3_momentum_vs_raw()
    fig4_forgetting_effect()
    fig5_full_pipeline()
    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
```

**Step 2: Run the script**

Run: `uv run python scripts/generate_figures.py`
Expected: 5 PNG files in `figures/`

**Step 3: Commit**

```bash
git add scripts/generate_figures.py
git commit -m "feat: add script to generate all demo figures as PNGs"
```

---

### Task 8: Notebook 01 — Surprise Metric

**Files:**
- Create: `notebooks/01_surprise_metric.ipynb`

**Step 1: Create notebook with cells**

Cells:
1. **Markdown**: Title + explanation of what surprise metric is
2. **Code**: Import + generate simple repeating data
3. **Markdown**: Explain the predictor network approach
4. **Code**: Create SurpriseMetric, feed familiar data, then novel data
5. **Code**: Plot surprise scores showing spikes at anomalies
6. **Markdown**: Key takeaway summary

**Step 2: Test notebook runs**

Run: `uv run jupyter execute notebooks/01_surprise_metric.ipynb`
Expected: Completes without errors

**Step 3: Commit**

```bash
git add notebooks/01_surprise_metric.ipynb
git commit -m "feat: add notebook 01 — surprise metric explanation and demo"
```

---

### Task 9: Notebook 02 — Memory Module

**Files:**
- Create: `notebooks/02_memory_module.ipynb`

**Step 1: Create notebook with cells**

Cells:
1. **Markdown**: Explain MLP-as-memory concept (weights ARE the memory)
2. **Code**: Create MemoryModule, demonstrate read/write
3. **Markdown**: Explain surprise-gated writes
4. **Code**: Show low-surprise vs high-surprise writes
5. **Code**: Heatmap of memory weight changes
6. **Markdown**: Explain momentum and forgetting
7. **Code**: Demonstrate forgetting effect with different rates
8. **Markdown**: Key takeaway

**Step 2: Test notebook**

Run: `uv run jupyter execute notebooks/02_memory_module.ipynb`

**Step 3: Commit**

```bash
git add notebooks/02_memory_module.ipynb
git commit -m "feat: add notebook 02 — memory module with heatmap visualization"
```

---

### Task 10: Notebook 03 — Full Titans Layer

**Files:**
- Create: `notebooks/03_full_titans.ipynb`

**Step 1: Create notebook with cells**

Cells:
1. **Markdown**: How all components fit together
2. **Code**: Create TitansMemoryLayer, process sequence with anomalies
3. **Code**: Plot surprise scores + memory state before/after anomaly
4. **Markdown**: Explain the full processing loop
5. **Code**: Try different hyperparameters (momentum, forget_rate)
6. **Markdown**: Key takeaway

**Step 2: Test notebook**

Run: `uv run jupyter execute notebooks/03_full_titans.ipynb`

**Step 3: Commit**

```bash
git add notebooks/03_full_titans.ipynb
git commit -m "feat: add notebook 03 — full TitansMemoryLayer demo"
```

---

### Task 11: Notebook 04 — Comprehensive Visualization

**Files:**
- Create: `notebooks/04_visualization.ipynb`

**Step 1: Create notebook with cells**

Cells:
1. **Markdown**: Overview — putting it all together
2. **Code**: Scenario 1 — repeating pattern with anomalies (full dashboard)
3. **Code**: Scenario 2 — frequency shift signal
4. **Code**: Scenario 3 — compare different momentum values
5. **Code**: Scenario 4 — compare different forget rates
6. **Markdown**: Summary of the Titans memory mechanism

**Step 2: Test notebook**

Run: `uv run jupyter execute notebooks/04_visualization.ipynb`

**Step 3: Commit**

```bash
git add notebooks/04_visualization.ipynb
git commit -m "feat: add notebook 04 — comprehensive visualization dashboard"
```

---

### Task 12: Final Integration & README

**Files:**
- Modify: `.gitignore` (ensure figures/*.png is tracked for README)
- Run: full test suite + generate all figures

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 2: Generate all figures**

Run: `uv run python scripts/generate_figures.py`

**Step 3: Commit everything**

```bash
git add -A
git commit -m "feat: complete Titans MIRAs demo with tests, notebooks, and figures"
```
