"""StatsAid: A comprehensive toolkit for data exploration, cleaning, and analysis in research."""

__version__ = "0.2.0"

# Core functionality
from .explore import (
    load_data,
    explore,
    suggest_preprocessing,
    suggest_models,
)

# Visualization module
from .visualize import (
    plot_missing_values,
    plot_missing_bar,
    plot_distributions,
    plot_correlation_matrix,
    plot_pairplot,
    plot_outliers,
    save_figure_to_base64,
)

# Enhanced missing data analysis
from .missing import (
    analyze_missing_patterns,
    test_missing_mechanism,
    compare_imputation_methods,
    assess_missing_impact,
)

# Data quality assessment
from .quality import (
    calculate_entropy,
    detect_data_leakage,
    check_multicollinearity,
    find_duplicates,
)

# Distribution analysis
from .distributions import (
    test_normality,
    detect_distribution_type,
    create_qq_plot,
    find_optimal_transformation,
)

# Outlier detection and treatment
from .outliers import (
    detect_univariate_outliers,
    detect_multivariate_outliers,
    calculate_influence,
    suggest_outlier_treatment,
)

# Feature importance and selection
from .features import (
    calculate_feature_importance,
    analyze_correlations,
    compute_mutual_information,
    select_features,
)

# Power analysis
from .power import (
    calculate_sample_size,
    estimate_effect_size,
    perform_power_analysis,
    recommend_study_design,
)

# Model diagnostics
from .diagnostics import (
    analyze_residuals,
    plot_calibration_curve,
    suggest_cross_validation,
    test_model_assumptions,
)

# Time series analysis
from .timeseries import (
    test_stationarity,
    detect_seasonality,
    analyze_autocorrelation,
    decompose_trend,
)

# Statistical testing
from .stats import (
    select_statistical_test,
    adjust_for_multiple_comparisons,
    calculate_effect_size,
    perform_bayesian_analysis,
)

# Reporting tools
from .report import (
    generate_report,
    create_cleaning_pipeline,
    visualize_confidence_intervals,
    interpret_results,
)

# Experimental design
from .design import (
    recommend_design,
    suggest_blocking,
    optimize_factorial_design,
    plan_sample_size,
)

__all__ = [
    # Core functionality
    "load_data",
    "explore",
    "suggest_preprocessing",
    "suggest_models",
    
    # Visualization
    "plot_missing_values",
    "plot_missing_bar",
    "plot_distributions",
    "plot_correlation_matrix",
    "plot_pairplot",
    "plot_outliers",
    "save_figure_to_base64",
    
    # Missing data analysis
    "analyze_missing_patterns",
    "test_missing_mechanism",
    "compare_imputation_methods",
    "assess_missing_impact",
    
    # Data quality
    "calculate_entropy",
    "detect_data_leakage",
    "check_multicollinearity",
    "find_duplicates",
    
    # Distribution analysis
    "test_normality",
    "detect_distribution_type",
    "create_qq_plot",
    "find_optimal_transformation",
    
    # Outlier treatment
    "detect_univariate_outliers",
    "detect_multivariate_outliers",
    "calculate_influence",
    "suggest_outlier_treatment",
    
    # Feature importance
    "calculate_feature_importance",
    "analyze_correlations",
    "compute_mutual_information",
    "select_features",
    
    # Power analysis
    "calculate_sample_size",
    "estimate_effect_size",
    "perform_power_analysis",
    "recommend_study_design",
    
    # Model diagnostics
    "analyze_residuals",
    "plot_calibration_curve",
    "suggest_cross_validation",
    "test_model_assumptions",
    
    # Time series
    "test_stationarity",
    "detect_seasonality",
    "analyze_autocorrelation",
    "decompose_trend",
    
    # Statistical testing
    "select_statistical_test",
    "adjust_for_multiple_comparisons",
    "calculate_effect_size",
    "perform_bayesian_analysis",
    
    # Reporting
    "generate_report",
    "create_cleaning_pipeline",
    "visualize_confidence_intervals",
    "interpret_results",
    
    # Experimental design
    "recommend_design",
    "suggest_blocking",
    "optimize_factorial_design",
    "plan_sample_size",
]