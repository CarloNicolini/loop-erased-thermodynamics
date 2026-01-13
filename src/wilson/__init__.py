"""Wilson algorithm for thermodynamic graph analysis."""

from wilson.graphs import build_graph, graph_id, relabel_consecutive
from wilson.inversion import (
    build_second_difference_matrix,
    compute_bin_widths,
    invert_stieltjes_density,
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
    compute_histogram_density,
    format_label,
    make_logspace_grid,
    setup_graph_and_output,
    setup_plotting_style,
)
from wilson.spectral import (
    z_from_spectrum,
    s_from_spectrum,
)
from wilson.wilson import Wilson


__all__ = [
    # Core Wilson algorithm
    "Wilson",
    # Graph utilities
    "build_graph",
    "graph_id",
    "relabel_consecutive",
    # Spectral analysis
    "z_from_spectrum",
    "s_from_spectrum",
    # Inversion
    "compute_bin_widths",
    "build_second_difference_matrix",
    "invert_stieltjes_density",
    # Plotting
    "save_plot_both_formats",
    "plot_s_vs_q",
    "plot_g_vs_q",
    "plot_spectrum_reconstruction",
    "plot_validation_s_vs_q",
    "plot_validation_Z_vs_beta",
    # Utilities
    "setup_plotting_style",
    "format_label",
    "make_logspace_grid",
    "setup_graph_and_output",
    "compute_histogram_density",
    "simulate_wilson_exploration",
    "grid_layout",
]
