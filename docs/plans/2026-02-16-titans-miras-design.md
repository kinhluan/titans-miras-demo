# Titans MIRAs Demo — Design Document

## Purpose
Educational demo simulating the Titans surprise-based memory mechanism with PyTorch, plus visualizations explaining how it works.

## Approach
Multiple small notebooks (Approach 3) + reusable Python package + script to generate figures. Managed with `uv`.

## Project Structure

```
titans-miras-demo/
├── pyproject.toml
├── titans_memory/
│   ├── __init__.py
│   ├── surprise.py          # SurpriseMetric
│   ├── memory.py            # MemoryModule (MLP + surprise-based update)
│   ├── titans.py            # TitansMemoryLayer (full pipeline)
│   └── data.py              # Synthetic data generators
├── scripts/
│   └── generate_figures.py  # One command → all PNG figures
├── figures/                 # Output images
├── notebooks/
│   ├── 01_surprise_metric.ipynb
│   ├── 02_memory_module.ipynb
│   ├── 03_full_titans.ipynb
│   └── 04_visualization.ipynb
```

## Components

### titans_memory/surprise.py
- Computes surprise = prediction error between memory output and actual input
- Low surprise → skip memory update; High surprise → strong update

### titans_memory/memory.py
- Small MLP (2-3 layers) as long-term memory
- Weight updates gated by surprise score
- Momentum: exponential moving average of recent surprises
- Forgetting: adaptive weight decay to discard stale info

### titans_memory/titans.py
- Combines SurpriseMetric + MemoryModule into single TitansMemoryLayer
- Processes sequence step-by-step, returns outputs + memory states

### titans_memory/data.py
- Generates synthetic sequences with repeating patterns + anomalies
- Example: [1,2,3,1,2,3,1,2,99,1,2,3...] where 99 triggers high surprise

### scripts/generate_figures.py
- Run: `uv run python scripts/generate_figures.py`
- Outputs: surprise_over_time.png, memory_update_heatmap.png, momentum_vs_raw.png, forgetting_effect.png, full_pipeline.png

### Notebooks
| Notebook | Focus | Key Visualization |
|----------|-------|-------------------|
| 01_surprise_metric | Surprise metric explanation + code | Line chart of surprise scores |
| 02_memory_module | MLP memory + momentum + forgetting | Heatmap of memory weight changes |
| 03_full_titans | Full TitansMemoryLayer | Before/after anomaly memory state |
| 04_visualization | Full pipeline on multiple scenarios | Combined dashboard |

## Dependencies
- python >=3.11
- torch >=2.0
- matplotlib >=3.7
- numpy >=1.24
- jupyter >=1.0

## Data
Synthetic numeric sequences only. No external datasets needed.
