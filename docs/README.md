# StatsAid Documentation

## Introduction

StatsAid is a comprehensive Python package for statistical data analysis, designed to guide researchers through the entire data analysis workflow. It combines automated data exploration, advanced statistical methods, and intuitive visualizations with plain-language recommendations.

## Installation

To install StatsAid, run:

```bash
pip install statsaid
```

## Modules Overview

StatsAid is organized into specialized modules for different aspects of data analysis:

1. **Core Functions**: Data loading, exploration, preprocessing suggestions
2. **Missing Data Analysis**: Pattern detection, mechanism testing, imputation comparison
3. **Distribution Analysis**: Normality testing, distribution fitting, transformation selection
4. **Data Quality**: Multicollinearity, outliers, data leakage detection
5. **Statistical Testing**: Test selection, multiple comparisons, effect sizes
6. **Feature Analysis**: Importance assessment, mutual information, selection methods
7. **Power Analysis**: Sample size calculation, effect size estimation
8. **Model Diagnostics**: Residuals, calibration, assumptions testing
9. **Time Series**: Stationarity, seasonality, trend decomposition
10. **Reporting**: Automated report generation, visualization, interpretation

## Core Functionality

### Data Loading and Exploration

```python
# Load data from various formats
data = sa.load_data(file_path, **kwargs)

# Get comprehensive data overview
overview = sa.explore(data)

# Get preprocessing suggestions
suggestions = sa.suggest_preprocessing(data)

# Get model recommendations based on study design
models = sa.suggest_models(data, study_design="case_control")
```

### Basic Visualizations

```python
# Plot missing data patterns
fig1 = sa.plot_missing_values(data)
fig2 = sa.plot_missing_bar(data)

# Plot distributions
fig3 = sa.plot_distributions(data, max_cols=20)

# Plot correlation matrix
fig4 = sa.plot_correlation_matrix(data, method='pearson')

# Create pairplot for numeric variables
fig5 = sa.plot_pairplot(data, max_cols=5, hue='group')

# Plot outlier detection
fig6 = sa.plot_outliers(data, max_cols=20)
```

## Missing Data Analysis

The missing data module provides tools for understanding and handling missing values in your dataset.

```python
# Analyze missing data patterns
patterns = sa.analyze_missing_patterns(data)

# Test missing data mechanism (MCAR/MAR/MNAR)
mechanism = sa.test_missing_mechanism(data, target_col='column1')

# Compare different imputation methods
imputation = sa.compare_imputation_methods(data, column='column1', 
                                          methods=['mean', 'median', 'knn', 'iterative'])

# Assess the impact of missing values on model performance
impact = sa.assess_missing_impact(data, target_col='target', predictors=['x1', 'x2'])

# Visualize missing patterns
fig1 = sa.plot_missing_heatmap(data)
fig2 = sa.plot_missing_patterns(data)
fig3 = sa.plot_imputation_comparison(imputation)
```

## Distribution Analysis

The distribution module provides comprehensive tools for understanding and transforming your data's distributions.

```python
# Test normality of numeric columns
normality = sa.test_normality(data, alpha=0.05)

# Detect the underlying probability distribution
dist_type = sa.detect_distribution_type(data['column1'], n_candidates=5)

# Create a QQ plot for distribution assessment
fig1 = sa.create_qq_plot(data['column1'], dist='norm')

# Find the optimal transformation for normality
transforms = sa.find_optimal_transformation(data['column1'], 
                                          transformations=['log', 'sqrt', 'boxcox'])

# Plot distribution comparisons
fig2 = sa.plot_distribution_comparison(data['column1'], 
                                      best_dist=dist_type['best_fit']['distribution'],
                                      params=dist_type['best_fit']['params'])

# Plot transformation comparisons
fig3 = sa.plot_transformation_comparison(data['column1'], transforms)
```

## Data Quality Assessment

```python
# Calculate entropy for features
entropy = sa.calculate_entropy(data)

# Detect potential data leakage
leakage = sa.detect_data_leakage(data, target_col='target')

# Check for multicollinearity
collinearity = sa.check_multicollinearity(data[numeric_cols], threshold=10)

# Find duplicates and near-duplicates
dupes = sa.find_duplicates(data, threshold=0.95)
```

## Feature Importance

```python
# Calculate importance scores for features
importance = sa.calculate_feature_importance(data, target_col='target')

# Analyze correlations with significance testing
correlations = sa.analyze_correlations(data, method='pearson')

# Calculate mutual information between features and target
mi = sa.compute_mutual_information(data, target_col='target')

# Select best features for modeling
features = sa.select_features(data, target_col='target', method='recursive')
```

## Power Analysis

```python
# Calculate required sample size
sample_size = sa.calculate_sample_size(effect_size=0.5, power=0.8, alpha=0.05)

# Estimate effect size from existing data
effect = sa.estimate_effect_size(group1, group2, method='cohen')

# Perform power analysis for an experiment
power = sa.perform_power_analysis(data, design="independent_t")

# Get recommendations for study design
design_rec = sa.recommend_study_design(data, effect_size=0.3)
```

## Time Series Analysis

```python
# Test for stationarity
stationarity = sa.test_stationarity(time_series)

# Detect seasonality
seasonality = sa.detect_seasonality(time_series)

# Analyze autocorrelation
acf = sa.analyze_autocorrelation(time_series, lags=20)

# Decompose trend, seasonality and residual components
decomp = sa.decompose_trend(time_series, model='additive')
```

## Example Use Cases

### Comprehensive Data Exploration

```python
import statsaid as sa
import pandas as pd

# Load data
data = pd.read_csv("clinical_trial_data.csv")

# Get overview and initial insights
overview = sa.explore(data)

# Analyze missing data
missing = sa.analyze_missing_patterns(data)
fig1 = sa.plot_missing_patterns(data)

# Test distributions and find transformations
normality = sa.test_normality(data)
for col in data.select_dtypes('number').columns:
    if normality[col]['normality'] != "Normal":
        transformations = sa.find_optimal_transformation(data[col])
        if transformations['worth_transforming']:
            print(f"Column {col}: {transformations['recommendation']}")

# Get modeling suggestions
models = sa.suggest_models(data, study_design="rct")
for model in models['models']:
    print(f"- {model}")
```

### Advanced Missing Value Handling

```python
# Compare imputation methods for key clinical variables
key_vars = ['blood_pressure', 'weight', 'cholesterol']
for var in key_vars:
    if data[var].isna().any():
        # Test missing mechanism
        mechanism = sa.test_missing_mechanism(data, target_col=var)
        print(f"{var}: {mechanism[var]['mechanism']}")
        
        # Compare imputation methods
        comparison = sa.compare_imputation_methods(data, column=var)
        best = comparison['best_method']['name']
        print(f"Best imputation method for {var}: {best}")
        
        # Visualize comparison
        fig = sa.plot_imputation_comparison(comparison)
        fig.savefig(f"{var}_imputation.png")
```

## Additional Resources

- Check the `examples/` directory for more comprehensive examples and use cases
- See `docs/api_reference.md` for detailed API documentation
- Visit `docs/tutorials/` for step-by-step tutorials on specific analyses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.