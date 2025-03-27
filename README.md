# StatsAid

A comprehensive toolkit for data exploration, cleaning, and analysis in research.

## Overview

StatsAid helps researchers analyze their data with advanced statistical methods and intuitive visualizations:

- **Data Exploration**: Automatically analyze distributions, correlations, and patterns
- **Missing Values**: Identify patterns, test mechanisms (MCAR/MAR/MNAR), and compare imputation methods
- **Distribution Analysis**: Test for normality, detect probability distributions, optimize transformations
- **Statistical Test Selection**: Get intelligent recommendations for statistical tests based on your study design
- **Effect Size Analysis**: Calculate and interpret appropriate effect size measures
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

# Test distributions and find transformations
normality_results = sa.test_normality(data)

# Get model recommendations
models = sa.suggest_models(data, study_design="case_control")
```

## Key Features

### 1. Enhanced Missing Data Analysis
- Pattern detection and visualization
- MCAR/MAR/MNAR mechanism testing
- Imputation method comparison

### 2. Distribution Analysis
- Multi-test normality assessment
- Automated probability distribution fitting
- Optimal transformation selection

### 3. Advanced Statistical Support
- Study design-specific test selection
- Multiple comparison corrections
- Effect size calculations with interpretation

### 4. Model Diagnostics
- Residual analysis
- Cross-validation strategy recommendation
- Model assumption testing

## Documentation

For full documentation, examples, and tutorials, visit [docs/](docs/)

## License

MIT