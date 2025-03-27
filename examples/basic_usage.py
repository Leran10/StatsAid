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
    print(f"Shape: {data.shape}")
    print(f"Columns: {', '.join(data.columns.tolist())}")
    
    # Use StatsAid to explore the data
    print("\n=== Data Exploration ===")
    overview = sa.explore(data)
    
    print("Variable Types:")
    for var_type, cols in overview['variable_types'].items():
        print(f"  - {var_type}: {len(cols)} columns")
    
    print("\nMissing Values:")
    for col, pct in sorted(overview['missing']['percentage'].items(), key=lambda x: x[1], reverse=True):
        if pct > 0:
            print(f"  - {col}: {pct:.1f}%")
    
    # Get preprocessing suggestions
    print("\n=== Preprocessing Suggestions ===")
    suggestions = sa.suggest_preprocessing(data)
    
    print("Handling Missing Values:")
    for col, suggestion in suggestions['missing_values'].items():
        print(f"  - {col}: {suggestion}")
    
    print("\nNormalization:")
    for col, suggestion in suggestions['normalization'].items():
        print(f"  - {col}: {suggestion}")
    
    # Get model suggestions
    print("\n=== Model Suggestions ===")
    model_suggestions = sa.suggest_models(data, study_design="cross_sectional")
    
    print("Recommended Models:")
    for model in model_suggestions['models']:
        print(f"  - {model}")
    
    print("\nRecommended Python Packages:")
    for package in model_suggestions['packages']['python']:
        print(f"  - {package}")
    
    # Generate some visualizations
    print("\n=== Generating Visualizations ===")
    
    # Missing values visualization
    missing_fig = sa.plot_missing_bar(data)
    missing_fig.savefig("missing_values.png")
    print("- Missing values plot saved as 'missing_values.png'")
    
    # Distribution plots
    distribution_figs = sa.plot_distributions(data, max_cols=5)
    if 'numeric' in distribution_figs:
        distribution_figs['numeric'].savefig("numeric_distributions.png")
        print("- Numeric distributions plot saved as 'numeric_distributions.png'")
    
    if 'categorical' in distribution_figs:
        distribution_figs['categorical'].savefig("categorical_distributions.png")
        print("- Categorical distributions plot saved as 'categorical_distributions.png'")
    
    # Correlation matrix
    corr_fig = sa.plot_correlation_matrix(data)
    corr_fig.savefig("correlation_matrix.png")
    print("- Correlation matrix plot saved as 'correlation_matrix.png'")
    
    # Outlier detection
    outlier_fig = sa.plot_outliers(data)
    outlier_fig.savefig("outliers.png")
    print("- Outlier detection plot saved as 'outliers.png'")
    
    print("\nExample completed. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()