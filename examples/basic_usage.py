#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating basic usage of StatsAid.

This example shows how to:
1. Load and explore a dataset
2. Get data preprocessing suggestions
3. Get model recommendations
4. Generate visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import statsaid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import statsaid as sa

# Generate a sample dataset
def generate_sample_dataset(n_samples=1000):
    """Generate a sample dataset for demonstration."""
    np.random.seed(42)
    
    # Create a dataframe
    df = pd.DataFrame()
    
    # Add numeric columns
    df['age'] = np.random.normal(45, 15, n_samples)
    df['age'] = df['age'].clip(18, 90).round().astype(int)
    
    df['bmi'] = np.random.normal(25, 5, n_samples)
    df['bmi'] = df['bmi'].clip(15, 45)
    
    df['glucose'] = np.random.normal(100, 25, n_samples)
    df['glucose'] = df['glucose'].clip(70, 300)
    
    df['blood_pressure'] = np.random.normal(120, 15, n_samples)
    df['blood_pressure'] = df['blood_pressure'].clip(80, 200)
    
    # Add some skewed data
    df['cost'] = np.random.exponential(1000, n_samples)
    
    # Add categorical columns
    df['gender'] = np.random.choice(['Male', 'Female'], n_samples)
    df['smoking'] = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.2, 0.2])
    df['education'] = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Add a target variable for classification
    high_risk = (df['age'] > 50) & (df['bmi'] > 30) | (df['glucose'] > 150)
    medium_risk = (~high_risk) & (df['age'] > 40) | (df['bmi'] > 27) | (df['glucose'] > 120)
    df['risk_category'] = 'Low'
    df.loc[medium_risk, 'risk_category'] = 'Medium'
    df.loc[high_risk, 'risk_category'] = 'High'
    
    # Add some missing values
    for col in ['bmi', 'glucose', 'education']:
        mask = np.random.random(n_samples) < 0.1
        df.loc[mask, col] = np.nan
    
    return df

def main():
    """Main function to demonstrate StatsAid capabilities."""
    # Generate sample data
    print("Generating sample dataset...")
    data = generate_sample_dataset()
    
    # Show basic info about the dataset
    print("\n=== Dataset Information ===")
    print("Shape: {}".format(data.shape))
    print("Columns: {}".format(', '.join(data.columns.tolist())))
    
    # Use StatsAid to explore the data
    print("\n=== Data Exploration ===")
    overview = sa.explore(data)
    
    print("Variable Types:")
    for var_type, cols in overview['variable_types'].items():
        print("  - {}: {} columns".format(var_type, len(cols)))
    
    print("\nMissing Values:")
    for col, pct in sorted(overview['missing']['percentage'].items(), key=lambda x: x[1], reverse=True):
        if pct > 0:
            print("  - {}: {:.1f}%".format(col, pct))
    
    # Get preprocessing suggestions
    print("\n=== Preprocessing Suggestions ===")
    suggestions = sa.suggest_preprocessing(data)
    
    print("Handling Missing Values:")
    for col, suggestion in suggestions['missing_values'].items():
        print("  - {}: {}".format(col, suggestion))
    
    print("\nNormalization:")
    for col, suggestion in suggestions['normalization'].items():
        print("  - {}: {}".format(col, suggestion))
    
    # Get model suggestions
    print("\n=== Model Suggestions ===")
    model_suggestions = sa.suggest_models(data, study_design="cross_sectional")
    
    print("Recommended Models:")
    for model in model_suggestions['models']:
        print("  - {}".format(model))
    
    print("\nRecommended Python Packages:")
    for package in model_suggestions['packages']['python']:
        print("  - {}".format(package))
    
    # Generate some visualizations
    print("\n=== Generating Visualizations ===")
    
    # Create output directory if it doesn't exist
    os.makedirs("examples/output", exist_ok=True)
    
    # Missing values visualization
    missing_fig = sa.plot_missing_bar(data)
    missing_fig.savefig("examples/output/missing_values.png")
    print("- Missing values plot saved as 'examples/output/missing_values.png'")
    
    # Distribution plots
    distribution_figs = sa.plot_distributions(data, max_cols=5)
    if 'numeric' in distribution_figs:
        distribution_figs['numeric'].savefig("examples/output/numeric_distributions.png")
        print("- Numeric distributions plot saved as 'examples/output/numeric_distributions.png'")
    
    if 'categorical' in distribution_figs:
        distribution_figs['categorical'].savefig("examples/output/categorical_distributions.png")
        print("- Categorical distributions plot saved as 'examples/output/categorical_distributions.png'")
    
    # Correlation matrix
    corr_fig = sa.plot_correlation_matrix(data)
    corr_fig.savefig("examples/output/correlation_matrix.png")
    print("- Correlation matrix plot saved as 'examples/output/correlation_matrix.png'")
    
    # Outlier detection
    outlier_fig = sa.plot_outliers(data)
    outlier_fig.savefig("examples/output/outliers.png")
    print("- Outlier detection plot saved as 'examples/output/outliers.png'")
    
    # Generate a comprehensive report
    print("\nGenerating report...")
    
    # Create a report file
    with open("examples/output/report.md", "w") as f:
        f.write("# StatsAid Analysis Report\n\n")
        
        # Dataset information
        f.write("## Dataset Information\n\n")
        f.write("- Shape: {} rows x {} columns\n".format(data.shape[0], data.shape[1]))
        f.write("- Columns: {}\n\n".format(', '.join(data.columns.tolist())))
        
        # Variable types
        f.write("## Variable Types\n\n")
        for var_type, cols in overview['variable_types'].items():
            f.write("### {} Variables ({})\n\n".format(var_type.capitalize(), len(cols)))
            if cols:
                f.write("| Variable | Description |\n")
                f.write("|----------|-------------|\n")
                for col in cols:
                    if var_type == 'numeric':
                        desc = "Mean: {:.2f}, Median: {:.2f}, Std: {:.2f}".format(data[col].mean(), data[col].median(), data[col].std())
                    elif var_type == 'categorical':
                        top_val = data[col].value_counts().index[0] if not data[col].isna().all() else "N/A"
                        n_unique = data[col].nunique()
                        desc = "Most common: {}, Unique values: {}".format(top_val, n_unique)
                    else:
                        desc = "Time-based variable"
                    f.write("| {} | {} |\n".format(col, desc))
                f.write("\n")
        
        # Missing values
        f.write("## Missing Values\n\n")
        missing_cols = {col: pct for col, pct in overview['missing']['percentage'].items() if pct > 0}
        
        if missing_cols:
            f.write("| Variable | Missing % | Suggestion |\n")
            f.write("|----------|-----------|------------|\n")
            for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True):
                suggestion = suggestions['missing_values'].get(col, "No specific suggestion")
                f.write("| {} | {:.1f}% | {} |\n".format(col, pct, suggestion))
        else:
            f.write("No missing values found in the dataset.\n")
        f.write("\n")
        
        # Preprocessing suggestions
        f.write("## Preprocessing Suggestions\n\n")
        
        f.write("### Normalization\n\n")
        f.write("| Variable | Suggestion |\n")
        f.write("|----------|------------|\n")
        for col, suggestion in suggestions['normalization'].items():
            f.write("| {} | {} |\n".format(col, suggestion))
        f.write("\n")
        
        if 'encoding' in suggestions and suggestions['encoding']:
            f.write("### Categorical Encoding\n\n")
            f.write("| Variable | Suggestion |\n")
            f.write("|----------|------------|\n")
            for col, suggestion in suggestions['encoding'].items():
                f.write("| {} | {} |\n".format(col, suggestion))
            f.write("\n")
        
        # Model suggestions
        f.write("## Model Suggestions\n\n")
        
        f.write("### Recommended Models\n\n")
        for model in model_suggestions['models']:
            f.write("- {}\n".format(model))
        f.write("\n")
        
        f.write("### Recommended Python Packages\n\n")
        for package in model_suggestions['packages']['python']:
            f.write("- {}\n".format(package))
        f.write("\n")
        
        f.write("### Recommended R Packages\n\n")
        for package in model_suggestions['packages']['r']:
            f.write("- {}\n".format(package))
        f.write("\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        
        f.write("The following visualizations were generated:\n\n")
        f.write("- Missing values plot: [missing_values.png](missing_values.png)\n")
        f.write("- Numeric distributions: [numeric_distributions.png](numeric_distributions.png)\n")
        f.write("- Categorical distributions: [categorical_distributions.png](categorical_distributions.png)\n")
        f.write("- Correlation matrix: [correlation_matrix.png](correlation_matrix.png)\n")
        f.write("- Outlier detection: [outliers.png](outliers.png)\n")
    
    print("Report generated: examples/output/report.md")
    print("Example completed. Check the generated PNG files and report for results.")

if __name__ == "__main__":
    main()