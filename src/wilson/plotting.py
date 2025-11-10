"""Plotting utilities for Wilson algorithm visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
from rich import print as rprint


def save_plot_both_formats(
    fig: plt.Figure, out_path: Path, base_name: str
) -> None:
    """
    Save matplotlib figure in both PDF and PNG formats.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save.
    out_path : Path
        Output directory path.
    base_name : str
        Base filename without extension (e.g., 'validation_s_vs_q_ER_n100').

    Notes
    -----
    Saves with dpi=200 and bbox_inches='tight' for both formats.
    Automatically closes the figure after saving.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pathlib import Path
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_plot_both_formats(fig, Path("."), "test_plot")
    """
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fname = out_path / f"{base_name}.{ext}"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        rprint(f"  [green]✓[/green] Saved: {fname}")
    plt.close(fig)

