# StatsAid

A comprehensive toolkit for data exploration, cleaning, and analysis in research.

## Overview

StatsAid helps researchers get an overview of their data, including:
- Data distributions and summary statistics
- Detection and visualization of missing values
- Automated suggestions for data cleaning and normalization
- Model selection based on study design
- Recommendations for relevant analysis packages

## Installation

```bash
pip install statsaid
```

## Quick Start

```python
import statsaid as sa

# Load your data
data = sa.load_data("your_data.csv")

# Get data overview
overview = sa.explore(data)
print(overview)

# Get suggestions for cleaning and normalization
suggestions = sa.suggest_preprocessing(data)
print(suggestions)

# Get model recommendations based on study design
models = sa.suggest_models(data, study_design="case_control")
print(models)
```

## Features

- **Data Overview**: Quick summaries and visualizations of your dataset
- **Missing Value Analysis**: Identify patterns in missing data
- **Data Quality Checks**: Detect outliers, inconsistencies, and data issues
- **Normalization Suggestions**: Get recommendations for data normalization
- **Model Selection**: Receive guidance on statistical approaches based on your study design
- **Package Recommendations**: Discover relevant R and Python packages for your analysis

## Documentation

For full documentation, visit [docs/](docs/)

## License

MIT