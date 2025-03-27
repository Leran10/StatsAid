"""StatsAid: A toolkit for data exploration, cleaning, and analysis in research."""

__version__ = "0.1.0"

from .explore import (
    load_data,
    explore,
    suggest_preprocessing,
    suggest_models,
)

from .visualize import (
    plot_missing_values,
    plot_missing_bar,
    plot_distributions,
    plot_correlation_matrix,
    plot_pairplot,
    plot_outliers,
    save_figure_to_base64,
)

__all__ = [
    "load_data",
    "explore",
    "suggest_preprocessing",
    "suggest_models",
    "plot_missing_values",
    "plot_missing_bar",
    "plot_distributions",
    "plot_correlation_matrix",
    "plot_pairplot",
    "plot_outliers",
    "save_figure_to_base64",
]