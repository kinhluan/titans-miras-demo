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

    ax1.plot(seq.numpy(), color="#2196F3", linewidth=1, label="Input sequence")
    anomaly_idx = mask.nonzero().squeeze(-1).numpy()
    ax1.scatter(anomaly_idx, seq[mask.bool()].numpy(), color="#F44336", s=80, zorder=5, label="Anomalies")
    ax1.set_ylabel("Input Value")
    ax1.set_title("Titans Surprise-Based Memory: Input Sequence & Surprise Response")
    ax1.legend()

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

    snapshots = result["memory_snapshots"]
    first_layer_weights = [s[0].numpy() for s in snapshots]

    changes = []
    for i in range(1, len(first_layer_weights)):
        diff = np.abs(first_layer_weights[i] - first_layer_weights[i - 1])
        changes.append(diff.flatten()[:16])

    changes = np.array(changes).T

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(changes, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight Index")
    ax.set_title("Memory Weight Update Intensity (brighter = bigger update)")
    plt.colorbar(im, ax=ax, label="|Δweight|")

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

    forget_rates = [0.0, 0.005, 0.02, 0.05]
    fig, ax = plt.subplots(figsize=(12, 4))

    for rate in forget_rates:
        seq, _ = generate_repeating_with_anomalies(
            pattern=[1.0, 2.0, 3.0], repeats=20, anomaly_indices=[10], anomaly_value=50.0
        )
        layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, forget_rate=rate)
        result = layer.process_sequence(seq.unsqueeze(-1))

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

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(seq.numpy(), color="#2196F3", linewidth=1)
    ax1.scatter(anomaly_idx, seq[mask.bool()].numpy(), color="#F44336", s=80, zorder=5)
    ax1.set_title("① Input Sequence (red = anomalies)")
    ax1.set_ylabel("Value")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(scores, color="#FF9800", linewidth=1.5)
    ax2.set_title("② Raw Surprise Score")
    ax2.set_ylabel("Surprise")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(momentum, color="#E91E63", linewidth=1.5)
    ax3.set_title("③ Momentum-Smoothed Surprise")
    ax3.set_ylabel("Surprise")

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
