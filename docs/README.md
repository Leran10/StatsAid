# StatsAid Documentation

## Introduction

StatsAid is a Python package designed to help researchers get an overview of their data, including data distributions, missing values, and providing useful suggestions for data cleaning, normalization. Based on the study design, it can also suggest the best statistical models as well as the available packages for analysis.

## Installation

To install StatsAid, run:

```bash
pip install statsaid
```

## Basic Usage

```python
import statsaid as sa
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

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

## Core Functions

### Data Loading

```python
sa.load_data(file_path, **kwargs)
```

Loads data from various file formats including CSV, TSV, Excel, Parquet, and Feather.

Parameters:
- `file_path`: Path to the data file
- `**kwargs`: Additional arguments to pass to the pandas read function

Returns:
- A pandas DataFrame containing the loaded data

### Data Exploration

```python
sa.explore(data)
```

Generates a comprehensive overview of the dataset.

Parameters:
- `data`: Input pandas DataFrame

Returns:
- A dictionary containing various summary statistics including:
  - Basic information (shape, columns, dtypes)
  - Summary statistics
  - Missing values analysis
  - Variable type detection
  - Outlier detection for numeric columns
  - Unique value counts for categorical columns

### Preprocessing Suggestions

```python
sa.suggest_preprocessing(data)
```

Suggests preprocessing steps based on the dataset characteristics.

Parameters:
- `data`: Input pandas DataFrame

Returns:
- A dictionary containing preprocessing suggestions for:
  - Missing values handling
  - Normalization for numeric features
  - Encoding for categorical features

### Model Suggestions

```python
sa.suggest_models(data, study_design=None)
```

Suggests appropriate statistical models and packages based on the dataset and study design.

Parameters:
- `data`: Input pandas DataFrame
- `study_design`: Type of study design (e.g., 'case_control', 'cohort', 'cross_sectional', 'longitudinal', 'rct')

Returns:
- A dictionary containing suggested models and packages (both Python and R)

## Visualization Functions

StatsAid provides several visualization functions to help explore your data:

### Missing Values Visualization

```python
sa.plot_missing_values(data)
sa.plot_missing_bar(data)
```

### Distribution Plots

```python
sa.plot_distributions(data, max_cols=20)
```

### Correlation Matrix

```python
sa.plot_correlation_matrix(data, method='pearson')
```

### Pairwise Relationships

```python
sa.plot_pairplot(data, max_cols=5, hue=None)
```

### Outlier Detection

```python
sa.plot_outliers(data, max_cols=20)
```

## Example

See the `examples/basic_usage.py` file for a complete example of using StatsAid.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.