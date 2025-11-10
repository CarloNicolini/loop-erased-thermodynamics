"""Wilson algorithm for thermodynamic graph analysis."""

from wilson.graphs import build_graph, graph_id, relabel_consecutive
from wilson.inversion import (
    build_second_difference_matrix,
    compute_bin_widths,
    invert_stieltjes_density,
)
from wilson.parallel import (
    estimate_s_for_validate,
    estimate_s_via_wilson,
    sample_wilson_for_q,
)
from wilson.plotting import save_plot_both_formats
from wilson.spectral import (
    compute_s_true,
    heat_trace_from_spectrum,
    laplacian_eigenvalues,
    s_from_spectrum,
    s_from_Z_via_gauss_laguerre,
)
from wilson.wilson import Wilson, WilsonQ, draw_sampling, quad

__all__ = [
    # Core Wilson algorithm
    "Wilson",
    "WilsonQ",
    "draw_sampling",
    "quad",
    # Graph utilities
    "build_graph",
    "graph_id",
    "relabel_consecutive",
    # Spectral analysis
    "laplacian_eigenvalues",
    "compute_s_true",
    "heat_trace_from_spectrum",
    "s_from_spectrum",
    "s_from_Z_via_gauss_laguerre",
    # Inversion
    "compute_bin_widths",
    "build_second_difference_matrix",
    "invert_stieltjes_density",
    # Parallel execution
    "sample_wilson_for_q",
    "estimate_s_via_wilson",
    "estimate_s_for_validate",
    # Plotting
    "save_plot_both_formats",
]
