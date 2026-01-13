# ThermoWilson

Wilson algorithm experiments for thermodynamic graph analysis. This project implements Wilson's algorithm and its variants (Wilson-Q) to sample spanning forests and reconstruct spectral properties of graphs.

## Features

- **Wilson Sampler**: Unified implementation of Wilson's algorithm for Uniform Spanning Trees (UST) and rooted spanning forests (Wilson-Q).
- **Spectral Density Reconstruction**: Tools to invert the Stieltjes transform of the forest-count function to recover the graph's Laplacian spectral density $\rho(\lambda)$.
- **Thermodynamic Analysis**: Estimation of the partition function $Z(\beta)$ and other thermodynamic quantities from Wilson samples.
- **Graph Families**: Support for Erdős-Rényi, Barabási-Albert, Regular, and Grid graphs.

## Installation

This project uses [astral-sh/uv](https://github.com/astral-sh/uv) for dependency management and execution. To set up the environment:

```bash
uv sync
```

## Testing

To run the test suite:

```bash
uv run pytest
```

## Library Usage

The core of the project is the `Wilson` class in `src/wilson/wilson.py`, which follows a scikit-learn-like API.

```python
import networkx as nx
from wilson import Wilson

# Create a graph
G = nx.grid_2d_graph(10, 10)

# Initialize Wilson sampler (q=None for classic UST, q > 0 for forests)
sampler = Wilson(q=0.1, random_state=42)

# Fit the sampler (pre-processes the graph)
sampler.fit(G, n_samples=100)

# Sample a forest
forest_edges, roots = sampler.sample()

# Get Monte Carlo estimates of s(q)
print(f"Estimate of s(q): {sampler.s_}")
```

## CLI Reference

The project provides a unified CLI `wilson` for running experiments. You can run it via `uv run`:

### Sampling $s(q)$

Sample the forest-count function $s(q)$ across a grid of $q$ values:

```bash
uv run wilson sample --graph ER --n 100 --p 0.05 --num-q 30 --samples-per-q 64
```

### Spectral Reconstruction

Reconstruct the spectral density and partition function $Z(\beta)$:

```bash
uv run wilson reconstruct --graph ER --n 50 --p 0.1
```

## Project Structure

```text
├── artifacts/          # Results and figures from experiments
├── scripts/            # Standalone experiment and plotting scripts
├── src/wilson/         # Core library implementation
│   ├── cli.py          # Typer CLI application
│   ├── inversion.py    # Spectral density inversion logic
│   ├── spectral.py     # Theoretical spectral calculations
│   └── wilson.py       # Wilson algorithm implementations
├── tests/              # Pytest suite
└── pyproject.toml      # Project metadata and dependencies
```

## License

MIT
