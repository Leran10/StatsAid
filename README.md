# StatsAid

A comprehensive toolkit for data exploration, cleaning, and analysis in research.

## Overview

StatsAid helps researchers analyze their data with advanced statistical methods and intuitive visualizations:

- **Data Exploration**: Automatically analyze distributions, correlations, and patterns
- **Missing Values**: Identify patterns, test mechanisms (MCAR/MAR/MNAR), and compare imputation methods
- **Distribution Analysis**: Test for normality, detect probability distributions, optimize transformations
- **Statistical Guidance**: Get recommendations for appropriate models based on your study design
- **Automated Reporting**: Generate comprehensive reports with insights and visualizations

## Installation

```bash
pip install statsaid
```

## Quick Start

```python
import statsaid as sa
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Get data overview with comprehensive statistics
overview = sa.explore(data)

# Analyze missing values and patterns
missing_analysis = sa.analyze_missing_patterns(data)
missing_mechanism = sa.test_missing_mechanism(data)

# Find optimal imputation methods for a column
imputation_comparison = sa.compare_imputation_methods(data, 'column_with_missing_values')

# Test distributions and find transformations
normality_results = sa.test_normality(data)
transformation_results = sa.find_optimal_transformation(data['numeric_column'])

# Get model recommendations
models = sa.suggest_models(data, study_design="case_control")

# Create visualizations
fig1 = sa.plot_missing_heatmap(data)
fig2 = sa.plot_transformation_comparison(data['skewed_column'], transformation_results)
```

## Key Features

### 1. Enhanced Missing Data Analysis
- Pattern detection and visualization
- MCAR/MAR/MNAR mechanism testing
- Imputation method comparison and validation
- Impact assessment on predictive modeling

### 2. Distribution Analysis
- Multi-test normality assessment (Shapiro-Wilk, Anderson-Darling, D'Agostino's K²)
- Automated probability distribution fitting (20+ distributions)
- QQ plots and visualization tools
- Optimal transformation selection (Box-Cox, Yeo-Johnson, etc.)

### 3. Data Quality Assessment
- Multicollinearity detection
- Outlier analysis with influence measures
- Data leakage identification
- Duplicates and near-duplicates detection

### 4. Advanced Statistical Support
- Appropriate statistical test selection
- Multiple comparison corrections
- Effect size calculations with interpretation
- Power analysis and sample size planning

### 5. Model Diagnostics
- Residual analysis
- Cross-validation strategy recommendation
- Model assumption testing
- Prediction calibration tools

### 6. Specialized Features
- Time series analysis (stationarity, seasonality, decomposition)
- Experimental design assistance
- Reproducible reporting
- Feature importance and selection

## Documentation

For full documentation and examples, visit [docs/](docs/)

## License

MIT