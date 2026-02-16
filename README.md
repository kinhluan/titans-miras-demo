# Titans MIRAs Demo

Educational demo of the **Surprise-Based Memory** mechanism from Google Research's [Titans](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) architecture, built with Python and PyTorch.

## How Does Titans Memory Work?

Titans uses a **small MLP network** as long-term memory. Instead of storing everything, it only memorizes what is **surprising**:

```
Each timestep:
  1. Compute surprise (is this input unexpected?)
  2. Read from memory (what does memory predict?)
  3. Write to memory gated by surprise (store only if novel)
  4. Update surprise predictor (learn the pattern)
  5. Apply forgetting (decay old information)
```

**Three core mechanisms:**
- **Surprise Metric** — Measures "unexpectedness" via prediction error (MSE)
- **Surprise-Gated Writes** — Only updates memory when surprise is high
- **Adaptive Forgetting** — Automatically discards stale information via weight decay

## Demo Figures

### Surprise Score Over Time
![Surprise over time](figures/surprise_over_time.png)

### Memory Update Heatmap
![Memory update heatmap](figures/memory_update_heatmap.png)

### Raw Surprise vs Momentum-Smoothed
![Momentum vs raw](figures/momentum_vs_raw.png)

### Forgetting Effect
![Forgetting effect](figures/forgetting_effect.png)

### Full Pipeline Dashboard
![Full pipeline](figures/full_pipeline.png)

## Project Structure

```
titans-miras-demo/
├── titans_memory/              # Python package
│   ├── surprise.py             # SurpriseMetric (prediction error as surprise)
│   ├── memory.py               # MemoryModule (MLP weights as memory)
│   ├── titans.py               # TitansMemoryLayer (full pipeline)
│   └── data.py                 # Synthetic data generators
├── scripts/
│   └── generate_figures.py     # Generate all PNG figures
├── notebooks/
│   ├── 01_surprise_metric.ipynb
│   ├── 02_memory_module.ipynb
│   ├── 03_full_titans.ipynb
│   └── 04_visualization.ipynb
├── tests/                      # 12 unit tests
└── figures/                    # Generated demo images
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/kinhluan/titans-miras-demo.git
cd titans-miras-demo

# Install dependencies (requires uv — https://docs.astral.sh/uv/)
uv sync --extra dev
```

## Usage

```bash
# Run tests
uv run pytest tests/ -v

# Generate all demo figures
uv run python scripts/generate_figures.py

# Launch Jupyter notebooks
uv run jupyter notebook notebooks/
```

## Notebooks

| Notebook | Content | Key Visualization |
|----------|---------|-------------------|
| `01_surprise_metric` | How surprise metric works | Line chart of surprise scores |
| `02_memory_module` | MLP as memory + forgetting | Bar chart + weight decay curves |
| `03_full_titans` | Full TitansMemoryLayer | 3-panel plot + hyperparameter comparison |
| `04_visualization` | 4 comprehensive scenarios | Dashboards + grid comparisons |

## Dependencies

- Python >= 3.11
- PyTorch >= 2.0
- matplotlib >= 3.7
- numpy >= 1.24
- Jupyter >= 1.0

## References

- [Titans + MIRAs: Helping AI Have Long-Term Memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) — Google Research Blog
