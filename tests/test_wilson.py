import networkx as nx
import numpy as np
import pytest

from wilson.spectral import s_from_spectrum
from wilson.wilson import Wilson as Wilson


@pytest.mark.parametrize("q", np.logspace(-2, 2, num=10))
def test_wilson_estimator(q):
    """Test the Wilson estimator on a small graph."""
    G = nx.grid_2d_graph(20, 20)
    random_state = None

    wilson_estimator = Wilson(q=q, random_state=random_state)
    wilson_estimator.fit(G, n_samples=500)
    n_roots = wilson_estimator.s_

    s_true = s_from_spectrum(q, nx.laplacian_spectrum(G))

    rel_error = abs(n_roots - s_true) / max(1e-12, abs(s_true))
    print(n_roots, s_true, rel_error)
    assert rel_error < 0.03, f"Relative error too large: {rel_error:.3e}"
