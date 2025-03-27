"""Enhanced missing data analysis module for StatsAid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Union, Optional, Tuple, Any

def analyze_missing_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns of missing values in the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing missing pattern analysis results
    """
    # Get basic missing value info
    missing_count = data.isna().sum()
    missing_percent = (missing_count / len(data)) * 100
    
    # Get columns with missing values
    missing_cols = missing_count[missing_count > 0].index.tolist()
    
    # Initialize results dict
    results = {
        'missing_count': missing_count.to_dict(),
        'missing_percent': missing_percent.to_dict(),
        'columns_with_missing': missing_cols,
        'total_missing_cells': data.isna().sum().sum(),
        'total_cells': data.size,
        'percent_missing_cells': (data.isna().sum().sum() / data.size) * 100,
    }
    
    if not missing_cols:
        results['patterns'] = {}
        results['joint_missing'] = {}
        return results
    
    # Analyze patterns
    # Create binary missing mask (1 if missing, 0 if not missing)
    missing_binary = data[missing_cols].isna().astype(int)
    
    # Get unique patterns
    pattern_counts = missing_binary.value_counts().reset_index()
    pattern_counts.columns = list(missing_binary.columns) + ['count']
    
    # Convert patterns to more readable format
    patterns = []
    for _, row in pattern_counts.iterrows():
        pattern = {}
        for col in missing_cols:
            pattern[col] = 'Missing' if row[col] == 1 else 'Present'
        pattern['count'] = row['count']
        pattern['percentage'] = (row['count'] / len(data)) * 100
        patterns.append(pattern)
    
    results['patterns'] = patterns
    
    # Analyze joint missingness
    joint_missing = {}
    if len(missing_cols) > 1:
        for i, col1 in enumerate(missing_cols):
            for col2 in missing_cols[i+1:]:
                both_missing = ((data[col1].isna()) & (data[col2].isna())).sum()
                either_missing = ((data[col1].isna()) | (data[col2].isna())).sum()
                joint_missing[f"{col1}_{col2}"] = {
                    'both_missing': both_missing,
                    'either_missing': either_missing,
                    'only_col1_missing': (data[col1].isna() & ~data[col2].isna()).sum(),
                    'only_col2_missing': (~data[col1].isna() & data[col2].isna()).sum(),
                    'jaccard_similarity': both_missing / either_missing if either_missing > 0 else 0
                }
    
    results['joint_missing'] = joint_missing
    
    return results


def test_missing_mechanism(data: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Test whether missing values appear to be MCAR, MAR, or MNAR.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_col : str, optional
        Target column to focus on (if None, will analyze all columns with missing values)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing missing mechanism test results
    """
    # Get columns with missing values
    missing_cols = data.columns[data.isna().any()].tolist()
    
    if not missing_cols:
        return {'mechanism': 'No missing values'}
    
    if target_col and target_col not in missing_cols:
        return {'error': f"Column '{target_col}' does not have missing values"}
    
    target_cols = [target_col] if target_col else missing_cols
    results = {}
    
    for col in target_cols:
        # Create a mask of missing values in the target column
        is_missing = data[col].isna()
        
        # Skip columns where all or no values are missing
        if is_missing.all() or not is_missing.any():
            results[col] = {
                'mechanism': 'Unable to determine',
                'reason': 'All values are missing or no values are missing'
            }
            continue
        
        # Create two groups: with and without missing values
        group1 = data[is_missing]
        group2 = data[~is_missing]
        
        # Compare other variables between groups
        tests = {}
        evidence_mar = []
        
        for other_col in data.columns:
            if other_col == col:
                continue
                
            # Skip if other column has missing values
            if data[other_col].isna().any():
                continue
                
            # For numeric data, use t-test
            if pd.api.types.is_numeric_dtype(data[other_col]):
                t_stat, p_value = stats.ttest_ind(
                    group2[other_col].dropna(),
                    group1[other_col].dropna(),
                    equal_var=False,  # Use Welch's t-test
                    nan_policy='omit'
                )
                test_type = "t-test"
            # For categorical data, use chi-square test
            else:
                # Create contingency table
                # Cross-tab of the categorical variable and missingness
                contingency = pd.crosstab(data[other_col], is_missing)
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    t_stat = chi2  # Just for consistency in output
                    test_type = "chi-square"
                except:
                    # Skip if test can't be performed (e.g., due to small counts)
                    continue
            
            # Store result
            tests[other_col] = {
                'test': test_type,
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # If significant, this is evidence for MAR
            if p_value < 0.05:
                evidence_mar.append(other_col)
        
        # Determine likely mechanism
        if not tests:
            mechanism = "Insufficient data to determine"
        elif not evidence_mar:
            mechanism = "Likely MCAR (Missing Completely At Random)"
        else:
            mechanism = "Likely MAR (Missing At Random)"
            
        results[col] = {
            'mechanism': mechanism,
            'tests': tests,
            'related_variables': evidence_mar
        }
    
    return results


def compare_imputation_methods(data: pd.DataFrame, 
                               column: str, 
                               methods: List[str] = ['mean', 'median', 'knn', 'iterative']) -> Dict[str, Any]:
    """
    Compare different imputation methods on a column with missing values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to impute
    methods : List[str], optional
        List of imputation methods to compare
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing imputation comparison results
    """
    # Verify that column exists and has missing values
    if column not in data.columns:
        return {'error': f"Column '{column}' not found in dataset"}
    
    if not data[column].isna().any():
        return {'error': f"Column '{column}' has no missing values"}
    
    # Verify that column is numeric (for now, only support numeric imputation)
    if not pd.api.types.is_numeric_dtype(data[column]):
        return {'error': f"Column '{column}' is not numeric. Currently only supporting numeric imputation comparison."}
    
    # Create a validation set by removing additional values
    # This helps us evaluate the imputation methods
    validation_idx = data[~data[column].isna()].sample(frac=0.3).index
    validation_values = data.loc[validation_idx, column].copy()
    
    # Create a working copy of the data
    df = data.copy()
    df.loc[validation_idx, column] = np.nan
    
    # Keep track of original missing values
    original_missing_idx = data[data[column].isna()].index
    
    results = {
        'total_missing': data[column].isna().sum(),
        'additional_validation_points': len(validation_idx),
        'methods': {}
    }
    
    # Apply different imputation methods
    for method in methods:
        # Skip unsupported methods
        if method not in ['mean', 'median', 'mode', 'knn', 'iterative']:
            continue
            
        # Create a copy for this method
        df_method = df.copy()
        numeric_cols = df_method.select_dtypes(include=['number']).columns.tolist()
        
        # Choose imputer based on method
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df_method[numeric_cols] = imputer.fit_transform(df_method[numeric_cols])
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            df_method[numeric_cols] = imputer.fit_transform(df_method[numeric_cols])
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_method[numeric_cols] = imputer.fit_transform(df_method[numeric_cols])
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_method[numeric_cols] = imputer.fit_transform(df_method[numeric_cols])
        elif method == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df_method[numeric_cols] = imputer.fit_transform(df_method[numeric_cols])
        
        # Calculate validation metrics (only for the validation set)
        imputed_values = df_method.loc[validation_idx, column]
        
        # Calculate error metrics
        mse = mean_squared_error(validation_values, imputed_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(validation_values, imputed_values)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((validation_values - imputed_values) / validation_values)) * 100
        
        # Store results
        results['methods'][method] = {
            'original_mean': data[column].mean(),
            'imputed_mean': df_method[column].mean(),
            'original_std': data[column].std(),
            'imputed_std': df_method[column].std(),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'imputed_values': {
                'validation': df_method.loc[validation_idx, column].to_dict(),
                'original_missing': df_method.loc[original_missing_idx, column].to_dict()
            }
        }
    
    # Determine best method based on lowest RMSE
    if results['methods']:
        best_method = min(results['methods'].items(), key=lambda x: x[1]['rmse'])[0]
        results['best_method'] = {
            'name': best_method,
            'rmse': results['methods'][best_method]['rmse']
        }
    
    return results


def assess_missing_impact(data: pd.DataFrame, target_col: str, predictors: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Assess the impact of missing values on predictive modeling for a target variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_col : str
        Target column for prediction
    predictors : List[str], optional
        List of predictor columns to use (if None, use all columns except target)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing missing impact analysis results
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    # Verify inputs
    if target_col not in data.columns:
        return {'error': f"Target column '{target_col}' not found in dataset"}
    
    # Determine if classification or regression
    is_classification = not pd.api.types.is_numeric_dtype(data[target_col]) or data[target_col].nunique() < 10
    
    # Get predictors if not specified
    if predictors is None:
        predictors = [col for col in data.columns if col != target_col]
    else:
        # Verify all predictors exist
        for col in predictors:
            if col not in data.columns:
                return {'error': f"Predictor column '{col}' not found in dataset"}
    
    # Check for missing values in target
    if data[target_col].isna().any():
        return {'error': f"Target column '{target_col}' has missing values"}
    
    # Get columns with missing values among predictors
    missing_cols = [col for col in predictors if data[col].isna().any()]
    
    # If no missing values, just return a message
    if not missing_cols:
        return {'message': "No missing values in predictor columns"}
    
    # Prepare three datasets:
    # 1. Original with missing values
    # 2. Complete cases (rows with no missing values)
    # 3. Imputed dataset
    
    # Complete cases
    complete_data = data.dropna(subset=predictors)
    
    # Imputed data (using median for numeric, mode for categorical)
    imputed_data = data.copy()
    
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())
        else:
            imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mode()[0])
    
    # Select only numeric predictors for now (for simplicity)
    numeric_predictors = [col for col in predictors if pd.api.types.is_numeric_dtype(data[col])]
    
    # Scale features
    scaler = StandardScaler()
    
    # Sets for analysis
    analysis_sets = {
        'complete_cases': {
            'X': scaler.fit_transform(complete_data[numeric_predictors]),
            'y': complete_data[target_col],
            'n_samples': len(complete_data)
        },
        'imputed': {
            'X': scaler.fit_transform(imputed_data[numeric_predictors]),
            'y': imputed_data[target_col],
            'n_samples': len(imputed_data)
        }
    }
    
    # Train models and get performance
    results = {
        'missing_impact': {},
        'performance': {}
    }
    
    for name, dataset in analysis_sets.items():
        # Skip if not enough samples
        if dataset['n_samples'] < 20:
            results['performance'][name] = {
                'error': f"Not enough samples ({dataset['n_samples']})"
            }
            continue
            
        # Choose model based on task
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring = 'neg_root_mean_squared_error'
        
        # Cross-validation
        cv_scores = cross_val_score(model, dataset['X'], dataset['y'], cv=5, scoring=scoring)
        
        if is_classification:
            results['performance'][name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'n_samples': dataset['n_samples']
            }
        else:
            # Convert negative RMSE to positive for easier interpretation
            rmse_scores = -cv_scores
            results['performance'][name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std(),
                'n_samples': dataset['n_samples']
            }
    
    # Assess impact by comparing performance
    if 'complete_cases' in results['performance'] and 'imputed' in results['performance']:
        if is_classification:
            acc_diff = (results['performance']['imputed']['mean_accuracy'] - 
                        results['performance']['complete_cases']['mean_accuracy'])
            
            results['missing_impact']['accuracy_difference'] = acc_diff
            results['missing_impact']['percent_difference'] = (acc_diff / 
                                                               results['performance']['complete_cases']['mean_accuracy']) * 100
            
            if abs(acc_diff) < 0.02:
                impact = "Minimal impact"
            elif acc_diff > 0:
                impact = "Positive impact (imputation improves model)"
            else:
                impact = "Negative impact (missing values harm model)"
                
        else:
            rmse_diff = (results['performance']['imputed']['mean_rmse'] - 
                         results['performance']['complete_cases']['mean_rmse'])
            
            results['missing_impact']['rmse_difference'] = rmse_diff
            results['missing_impact']['percent_difference'] = (rmse_diff / 
                                                              results['performance']['complete_cases']['mean_rmse']) * 100
            
            if abs(rmse_diff) < 0.05:
                impact = "Minimal impact"
            elif rmse_diff < 0:
                impact = "Positive impact (imputation improves model)"
            else:
                impact = "Negative impact (missing values harm model)"
        
        results['missing_impact']['impact_assessment'] = impact
        
        # Calculate sample size impact
        sample_diff = (results['performance']['imputed']['n_samples'] - 
                      results['performance']['complete_cases']['n_samples'])
        
        results['missing_impact']['additional_samples_with_imputation'] = sample_diff
        results['missing_impact']['percent_more_samples'] = (sample_diff / 
                                                           results['performance']['complete_cases']['n_samples']) * 100
    
    return results


def plot_missing_heatmap(data: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a heatmap of missing values correlation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
        
    Returns
    -------
    plt.Figure
        Figure object with the missing correlation heatmap
    """
    # Get columns with missing values
    missing_cols = [col for col in data.columns if data[col].isna().any()]
    
    if not missing_cols:
        # No missing values
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No missing values in the dataset", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create binary indicators for missingness
    missing_binary = pd.DataFrame()
    for col in missing_cols:
        missing_binary[f"Missing_{col}"] = data[col].isna().astype(int)
    
    # Calculate correlation matrix
    corr_matrix = missing_binary.corr()
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        fmt=".2f",
        annot_kws={"size": 8}
    )
    
    # Set title
    plt.title('Missing Value Correlation Matrix')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def plot_missing_patterns(data: pd.DataFrame, max_patterns: int = 10, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Visualize missing data patterns in the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    max_patterns : int, optional
        Maximum number of patterns to display, by default 10
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
        
    Returns
    -------
    plt.Figure
        Figure object with the missing patterns plot
    """
    # Get columns with missing values
    missing_cols = [col for col in data.columns if data[col].isna().any()]
    
    if not missing_cols:
        # No missing values
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No missing values in the dataset", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create binary indicators for missingness
    missing_binary = data[missing_cols].isna().astype(int)
    
    # Create pattern string for each row
    missing_binary['pattern'] = missing_binary.apply(lambda x: ''.join(x.astype(str)), axis=1)
    
    # Count pattern occurrences
    pattern_counts = missing_binary['pattern'].value_counts().head(max_patterns)
    
    # Create a DataFrame for each pattern
    patterns_df = pd.DataFrame(index=missing_cols)
    
    for pattern, count in pattern_counts.items():
        col_name = f"Pattern {pattern} (n={count})"
        patterns_df[col_name] = [int(c) for c in pattern]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        patterns_df,
        cmap=['#FFFFFF', '#2171b5'],  # White for not missing, blue for missing
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Set title and labels
    plt.title('Missing Data Patterns')
    plt.xlabel('Patterns (1 = Missing, 0 = Present)')
    plt.ylabel('Variables')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def plot_imputation_comparison(results: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize the comparison of different imputation methods.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from compare_imputation_methods function
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
        
    Returns
    -------
    plt.Figure
        Figure object with the imputation comparison plot
    """
    # Check for errors in results
    if 'error' in results:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, f"Error: {results['error']}", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Extract methods and metrics
    methods = list(results['methods'].keys())
    
    if not methods:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No imputation methods to compare", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Error metrics
    metrics = ['rmse', 'mae', 'mape']
    metric_data = {metric: [results['methods'][method][metric] for method in methods] for metric in metrics}
    
    # Plot error metrics
    bar_width = 0.25
    index = np.arange(len(methods))
    
    for i, metric in enumerate(metrics):
        axes[0].bar(index + i*bar_width, metric_data[metric], bar_width, 
                   label=metric.upper())
    
    axes[0].set_xlabel('Imputation Method')
    axes[0].set_ylabel('Error Value')
    axes[0].set_title('Error Metrics by Imputation Method')
    axes[0].set_xticks(index + bar_width)
    axes[0].set_xticklabels(methods)
    axes[0].legend()
    
    # Distribution comparison
    # Compare original vs imputed distribution stats
    orig_means = [results['methods'][method]['original_mean'] for method in methods]
    imp_means = [results['methods'][method]['imputed_mean'] for method in methods]
    orig_stds = [results['methods'][method]['original_std'] for method in methods]
    imp_stds = [results['methods'][method]['imputed_std'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1].bar(x - width/2, orig_means, width, yerr=orig_stds, label='Original', color='skyblue')
    axes[1].bar(x + width/2, imp_means, width, yerr=imp_stds, label='Imputed', color='lightcoral')
    
    axes[1].set_xlabel('Imputation Method')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title('Distribution Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].legend()
    
    # Highlight best method
    if 'best_method' in results:
        best_idx = methods.index(results['best_method']['name'])
        axes[0].get_children()[best_idx].set_facecolor('gold')
        axes[1].get_children()[best_idx + len(methods)].set_facecolor('gold')
        
        fig.suptitle(f"Imputation Comparison (Best Method: {results['best_method']['name']})", 
                    fontsize=16)
    else:
        fig.suptitle("Imputation Method Comparison", fontsize=16)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig