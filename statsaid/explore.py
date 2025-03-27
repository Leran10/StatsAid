"""Core functions for data exploration and analysis."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments to pass to the pandas read function
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_ext == '.tsv' or file_ext == '.txt':
        return pd.read_csv(file_path, sep='\t', **kwargs)
    elif file_ext == '.xlsx' or file_ext == '.xls':
        return pd.read_excel(file_path, **kwargs)
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif file_ext == '.feather':
        return pd.read_feather(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def explore(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive overview of the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing various summary statistics and visualizations
    """
    results = {}
    
    # Basic information
    results['shape'] = data.shape
    results['columns'] = data.columns.tolist()
    results['dtypes'] = data.dtypes.to_dict()
    
    # Summary statistics
    results['summary'] = data.describe(include='all').to_dict()
    
    # Missing values
    results['missing'] = {
        'total': data.isna().sum().to_dict(),
        'percentage': (data.isna().mean() * 100).to_dict()
    }
    
    # Detect variable types
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
    
    results['variable_types'] = {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }
    
    # Check for outliers in numeric columns
    if numeric_cols:
        results['outliers'] = {}
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            results['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'range': (lower_bound, upper_bound)
            }
    
    # Check for unique values in categorical columns
    if categorical_cols:
        results['categorical_counts'] = {}
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            results['categorical_counts'][col] = {
                'unique_count': data[col].nunique(),
                'top_values': value_counts.head(10).to_dict()
            }
    
    return results


def suggest_preprocessing(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Suggest preprocessing steps based on the dataset characteristics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessing suggestions
    """
    suggestions = {}
    
    # Get data overview
    overview = explore(data)
    
    # Suggest handling missing values
    missing_pct = overview['missing']['percentage']
    suggestions['missing_values'] = {}
    
    for col, pct in missing_pct.items():
        if pct > 0:
            if pct > 50:
                suggestions['missing_values'][col] = "Consider dropping this column due to high missingness (>50%)"
            elif pct > 20:
                suggestions['missing_values'][col] = "Consider imputation or creating a 'missing' indicator"
            else:
                if col in overview['variable_types']['numeric']:
                    suggestions['missing_values'][col] = "Impute with mean, median, or use KNN imputation"
                elif col in overview['variable_types']['categorical']:
                    suggestions['missing_values'][col] = "Impute with mode or create a new 'missing' category"
    
    # Suggest normalization for numeric features
    suggestions['normalization'] = {}
    for col in overview['variable_types']['numeric']:
        # Check if column might benefit from transformation
        if col in overview['outliers'] and overview['outliers'][col]['count'] > 0:
            skewness = data[col].skew()
            if abs(skewness) > 1:
                if skewness > 0:
                    suggestions['normalization'][col] = "Consider log or square root transformation (right-skewed)"
                else:
                    suggestions['normalization'][col] = "Consider square or cube transformation (left-skewed)"
            else:
                suggestions['normalization'][col] = "Consider min-max scaling or standardization"
        else:
            suggestions['normalization'][col] = "Standard scaling should be sufficient"
    
    # Suggest encoding for categorical features
    suggestions['encoding'] = {}
    for col in overview['variable_types']['categorical']:
        unique_count = overview['categorical_counts'][col]['unique_count']
        if unique_count == 2:
            suggestions['encoding'][col] = "Binary encoding (0/1)"
        elif 2 < unique_count <= 10:
            suggestions['encoding'][col] = "One-hot encoding"
        else:
            suggestions['encoding'][col] = "Consider label encoding, target encoding, or embedding techniques"
    
    return suggestions


def suggest_models(data: pd.DataFrame, study_design: str = None) -> Dict[str, List[str]]:
    """
    Suggest appropriate statistical models and packages based on the dataset and study design.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    study_design : str, optional
        Type of study design (e.g., 'case_control', 'cohort', 'cross_sectional', 'longitudinal', 'rct')
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing suggested models and packages
    """
    suggestions = {
        'models': [],
        'packages': {
            'python': [],
            'r': []
        }
    }
    
    # Get data overview
    overview = explore(data)
    
    # Basic assumptions
    num_features = len(overview['variable_types']['numeric'])
    num_samples = overview['shape'][0]
    has_categorical = len(overview['variable_types']['categorical']) > 0
    
    # Default packages
    suggestions['packages']['python'] = ['scikit-learn', 'statsmodels', 'scipy']
    suggestions['packages']['r'] = ['stats', 'lme4', 'ggplot2']
    
    # Study design specific suggestions
    if study_design is None:
        # General suggestions based on data characteristics
        if num_samples < 30:
            suggestions['models'].append("Non-parametric tests (small sample size)")
            suggestions['packages']['r'].append('nonpar')
            
        else:
            suggestions['models'].extend([
                "Linear regression",
                "Logistic regression",
                "Random Forest",
                "Gradient Boosting"
            ])
            suggestions['packages']['python'].extend(['xgboost', 'lightgbm'])
            suggestions['packages']['r'].extend(['randomForest', 'gbm', 'xgboost'])
    
    elif study_design.lower() == 'case_control':
        suggestions['models'].extend([
            "Logistic regression",
            "Conditional logistic regression",
            "Fisher's exact test (small sample size)"
        ])
        suggestions['packages']['r'].extend(['survival', 'epitools', 'epiR'])
        suggestions['packages']['python'].append('lifelines')
        
    elif study_design.lower() == 'cohort':
        suggestions['models'].extend([
            "Cox proportional hazards",
            "Kaplan-Meier survival analysis",
            "Poisson regression",
            "Negative binomial regression"
        ])
        suggestions['packages']['r'].extend(['survival', 'survminer', 'MASS'])
        suggestions['packages']['python'].extend(['lifelines', 'statsmodels.discrete_models'])
        
    elif study_design.lower() == 'cross_sectional':
        suggestions['models'].extend([
            "Chi-square test of independence",
            "Fisher's exact test",
            "Logistic regression",
            "Multinomial logistic regression"
        ])
        suggestions['packages']['r'].extend(['vcd', 'nnet', 'MASS'])
        
    elif study_design.lower() == 'longitudinal':
        suggestions['models'].extend([
            "Mixed effects models",
            "Generalized estimating equations (GEE)",
            "Repeated measures ANOVA"
        ])
        suggestions['packages']['r'].extend(['lme4', 'nlme', 'geepack'])
        suggestions['packages']['python'].append('statsmodels.genmod.gee')
        
    elif study_design.lower() == 'rct':
        suggestions['models'].extend([
            "Independent t-test",
            "Paired t-test",
            "Mixed ANOVA",
            "ANCOVA",
            "Intention-to-treat analysis"
        ])
        suggestions['packages']['r'].extend(['car', 'emmeans', 'effectsize'])
        
    # Add suggestions based on data features
    if has_categorical:
        if "Chi-square test" not in suggestions['models']:
            suggestions['models'].append("Chi-square test")
        
        if "Fisher's exact test" not in suggestions['models']:
            suggestions['models'].append("Fisher's exact test (for small cell counts)")
    
    # Handle high-dimensional data
    if num_features > 100:
        suggestions['models'].extend([
            "Principal Component Analysis (PCA)",
            "LASSO regression",
            "Ridge regression",
            "Elastic Net",
            "Dimensionality reduction techniques"
        ])
        suggestions['packages']['python'].append('sklearn.decomposition')
        suggestions['packages']['r'].extend(['glmnet', 'caret', 'pcaMethods'])
    
    # Handle imbalanced data (assuming classification)
    # This is a simplified check - you'd want to actually check the target variable
    if study_design and study_design.lower() in ['case_control']:
        suggestions['models'].extend([
            "SMOTE for class imbalance",
            "Weighted models",
            "AUC-ROC as evaluation metric"
        ])
        suggestions['packages']['python'].append('imbalanced-learn')
        suggestions['packages']['r'].extend(['ROSE', 'DMwR'])
    
    return suggestions