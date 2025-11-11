"""Wilson algorithm for thermodynamic graph analysis."""

from wilson.fractal import run_fractal_dimension_experiment
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
from wilson.plotting import (
    plot_g_vs_q,
    plot_s_vs_q,
    plot_spectrum_reconstruction,
    plot_validation_s_vs_q,
    plot_validation_Z_vs_beta,
    save_plot_both_formats,
)
from wilson.utils import (
    compute_eigenvalues_safe,
    compute_histogram_density,
    format_label,
    make_logspace_grid,
    setup_graph_and_output,
    setup_plotting_style,
)
from wilson.validation import build_validation_graphs
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
    "plot_s_vs_q",
    "plot_g_vs_q",
    "plot_spectrum_reconstruction",
    "plot_validation_s_vs_q",
    "plot_validation_Z_vs_beta",
    # Validation
    "build_validation_graphs",
    # Utilities
    "setup_plotting_style",
    "format_label",
    "make_logspace_grid",
    "setup_graph_and_output",
    "compute_eigenvalues_safe",
    "compute_histogram_density",
    # Fractal experiment
    "run_fractal_dimension_experiment",
]
