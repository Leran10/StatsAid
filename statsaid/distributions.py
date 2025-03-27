"""Distribution analysis module for StatsAid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import special
from typing import Dict, List, Union, Optional, Tuple, Any


def test_normality(data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Test numeric columns for normality using multiple tests.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    alpha : float, optional
        Significance level for the tests, by default 0.05
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing normality test results for each numeric column
    """
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    
    for col in numeric_cols:
        # Skip columns with too few values or all missing
        if data[col].count() < 8 or data[col].isna().all():
            results[col] = {
                'error': f"Not enough non-missing values for column '{col}'"
            }
            continue
        
        # Use clean data without missing values
        clean_data = data[col].dropna()
        
        # Store results for this column
        col_results = {}
        
        # 1. Shapiro-Wilk test (most powerful for n < 2000)
        if len(clean_data) < 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(clean_data)
                col_results['shapiro'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'normal': shapiro_p > alpha
                }
            except:
                col_results['shapiro'] = {
                    'error': "Could not perform Shapiro-Wilk test"
                }
        
        # 2. Anderson-Darling test (better for larger samples)
        try:
            ad_result = stats.anderson(clean_data, 'norm')
            # Convert from critical values to p-value (approximate)
            significance_levels = ad_result.significance_level / 100
            critical_values = ad_result.critical_values
            
            # Find the highest significance level that rejects the null hypothesis
            p_value = None
            for i, level in enumerate(significance_levels):
                if ad_result.statistic > critical_values[i]:
                    p_value = level
                    break
            
            if p_value is None:
                p_value = 0.15  # Higher than all tested levels
                
            col_results['anderson'] = {
                'statistic': ad_result.statistic,
                'critical_values': dict(zip(significance_levels, critical_values)),
                'p_value': p_value,
                'normal': p_value > alpha
            }
        except:
            col_results['anderson'] = {
                'error': "Could not perform Anderson-Darling test"
            }
        
        # 3. D'Agostino's K^2 test (focuses on skewness and kurtosis)
        try:
            k2_stat, k2_p = stats.normaltest(clean_data)
            col_results['dagostino'] = {
                'statistic': k2_stat,
                'p_value': k2_p,
                'normal': k2_p > alpha
            }
        except:
            col_results['dagostino'] = {
                'error': "Could not perform D'Agostino's test"
            }
        
        # Calculate basic statistics
        col_results['skewness'] = stats.skew(clean_data)
        col_results['kurtosis'] = stats.kurtosis(clean_data)
        
        # Determine overall normality
        test_results = [test.get('normal', False) for test_name, test 
                        in col_results.items() if test_name in ['shapiro', 'anderson', 'dagostino']]
        
        # If at least one test passed, we consider it possibly normal
        if any(test_results):
            if all(test_results):
                normality = "Normal"
            else:
                normality = "Probably Normal"
        else:
            # All tests reject normality
            if abs(col_results['skewness']) > 0.5:
                if col_results['skewness'] > 0:
                    normality = "Right-Skewed"
                else:
                    normality = "Left-Skewed"
            elif col_results['kurtosis'] > 0.5:
                normality = "Heavy-Tailed"
            elif col_results['kurtosis'] < -0.5:
                normality = "Light-Tailed"
            else:
                normality = "Non-Normal"
        
        col_results['normality'] = normality
        
        # Add to results
        results[col] = col_results
    
    return results


def detect_distribution_type(data: pd.Series, n_candidates: int = 5) -> Dict[str, Any]:
    """
    Detect the most likely probability distribution for the given data.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    n_candidates : int, optional
        Number of candidate distributions to return, by default 5
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing distribution detection results
    """
    # Clean data (no missing values)
    clean_data = data.dropna()
    
    # Return error if not enough data
    if len(clean_data) < 10:
        return {'error': "Not enough data points for distribution detection"}
    
    # List of distributions to test
    distributions = [
        # Continuous distributions
        'norm', 'lognorm', 'expon', 'gamma', 'beta',
        'weibull_min', 'uniform', 'chi2', 'f', 't',
        # Discrete distributions
        'poisson', 'binom', 'nbinom'
    ]
    
    # For discrete distributions, check if data is integer-like
    is_discrete = np.all(np.isclose(clean_data, np.round(clean_data), rtol=1e-5, atol=1e-5))
    
    # Descriptive statistics
    summary = {
        'mean': clean_data.mean(),
        'median': clean_data.median(),
        'std': clean_data.std(),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'is_discrete': is_discrete,
        'n': len(clean_data)
    }
    
    # Store results for each distribution
    distribution_fits = {}
    
    # Test each distribution
    for dist_name in distributions:
        # Skip discrete distributions if data doesn't look discrete
        if dist_name in ['poisson', 'binom', 'nbinom'] and not is_discrete:
            continue
            
        dist = getattr(stats, dist_name)
        
        try:
            # Fit the distribution
            if dist_name == 'binom':
                # For binomial, we need to provide n_trials parameter
                # Guess it as max value
                n_trials = int(clean_data.max())
                params = dist.fit(clean_data, n_trials)
            else:
                params = dist.fit(clean_data)
                
            # Calculate log-likelihood
            log_likelihood = np.sum(dist.logpdf(clean_data, *params))
            
            # Calculate AIC and BIC
            k = len(params)
            n = len(clean_data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Perform Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(clean_data, dist_name, params)
            
            # Store results
            distribution_fits[dist_name] = {
                'params': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_stat': ks_stat,
                'ks_p': ks_p
            }
        except:
            # Skip distributions that fail to fit
            continue
    
    # Rank distributions by AIC (lower is better)
    if distribution_fits:
        ranked_dists = sorted(distribution_fits.items(), key=lambda x: x[1]['aic'])
        
        # Get top candidates
        top_candidates = ranked_dists[:min(n_candidates, len(ranked_dists))]
        
        candidates = {}
        for dist_name, dist_data in top_candidates:
            candidates[dist_name] = {
                'aic': dist_data['aic'],
                'bic': dist_data['bic'],
                'ks_p': dist_data['ks_p'],
                'params': dist_data['params']
            }
        
        # Best distribution is the one with lowest AIC
        best_dist = top_candidates[0][0]
        best_params = top_candidates[0][1]['params']
        
        return {
            'summary': summary,
            'best_fit': {
                'distribution': best_dist,
                'params': best_params,
                'aic': top_candidates[0][1]['aic'],
                'p_value': top_candidates[0][1]['ks_p']
            },
            'candidates': candidates
        }
    else:
        return {
            'summary': summary,
            'error': "Could not fit any distribution to the data"
        }


def create_qq_plot(data: pd.Series, dist: str = 'norm', figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Create a QQ plot for the given data against a theoretical distribution.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    dist : str, optional
        Distribution to compare against ('norm', 'expon', 'lognorm'), by default 'norm'
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
        
    Returns
    -------
    plt.Figure
        Figure object with the QQ plot
    """
    # Clean data (no missing values)
    clean_data = data.dropna()
    
    # Return error figure if not enough data
    if len(clean_data) < 10:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Not enough data points for QQ plot", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create QQ plot
    if dist == 'norm':
        # Normal probability plot
        res = stats.probplot(clean_data, plot=ax)
        ax.set_title(f'Normal Q-Q Plot for {data.name}')
    else:
        # Other distributions
        data_sorted = np.sort(clean_data)
        n = len(data_sorted)
        p = np.arange(1, n + 1) / (n + 1)  # Empirical CDF
        
        if dist == 'expon':
            q_theor = stats.expon.ppf(p)
            ax.set_title(f'Exponential Q-Q Plot for {data.name}')
        elif dist == 'lognorm':
            q_theor = stats.lognorm.ppf(p, s=1)
            ax.set_title(f'Log-Normal Q-Q Plot for {data.name}')
        else:
            # Default to normal
            q_theor = stats.norm.ppf(p)
            ax.set_title(f'Normal Q-Q Plot for {data.name}')
        
        # Plot the points
        ax.scatter(q_theor, data_sorted)
        
        # Add reference line
        slope, intercept = np.polyfit(q_theor, data_sorted, 1)
        ax.plot(q_theor, intercept + slope * q_theor, 'r-')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def find_optimal_transformation(data: pd.Series, 
                               transformations: List[str] = ['log', 'sqrt', 'reciprocal', 'boxcox', 'yeojohnson']) -> Dict[str, Any]:
    """
    Find the optimal transformation to make the data more normal.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    transformations : List[str], optional
        List of transformations to try, by default all available
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing transformation comparison results
    """
    # Clean data (no missing values)
    clean_data = data.dropna()
    
    # Return error if not enough data
    if len(clean_data) < 10:
        return {'error': "Not enough data points for transformation analysis"}
    
    # Check data requirements for transformations
    valid_transformations = []
    for trans in transformations:
        # Log and reciprocal need positive data
        if trans in ['log', 'reciprocal']:
            if clean_data.min() <= 0:
                continue
                
        # Sqrt needs non-negative data
        if trans == 'sqrt':
            if clean_data.min() < 0:
                continue
                
        # Box-Cox needs positive data
        if trans == 'boxcox':
            if clean_data.min() <= 0:
                continue
                
        valid_transformations.append(trans)
    
    # Test normality of original data
    shapiro_orig = stats.shapiro(clean_data)
    
    results = {
        'original': {
            'shapiro_stat': shapiro_orig[0],
            'shapiro_p': shapiro_orig[1],
            'skewness': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data)
        },
        'transformations': {}
    }
    
    # Try each transformation
    for trans in valid_transformations:
        try:
            if trans == 'log':
                transformed = np.log(clean_data)
                formula = "log(x)"
                inverse = "exp(x)"
            elif trans == 'sqrt':
                transformed = np.sqrt(clean_data)
                formula = "sqrt(x)"
                inverse = "x^2"
            elif trans == 'reciprocal':
                transformed = 1 / clean_data
                formula = "1/x"
                inverse = "1/x"
            elif trans == 'boxcox':
                transformed, lmbda = stats.boxcox(clean_data)
                formula = f"boxcox(x, lambda={lmbda:.4f})"
                if abs(lmbda) < 0.01:
                    inverse = "exp(x)"
                else:
                    inverse = f"(x*{lmbda:.4f} + 1)^(1/{lmbda:.4f})"
            elif trans == 'yeojohnson':
                transformed, lmbda = stats.yeojohnson(clean_data)
                formula = f"yeojohnson(x, lambda={lmbda:.4f})"
                # Inverse is complex to express in a simple string
                inverse = "yeojohnson_inverse(x)"
            
            # Test normality
            shapiro_trans = stats.shapiro(transformed)
            
            # Store results
            results['transformations'][trans] = {
                'shapiro_stat': shapiro_trans[0],
                'shapiro_p': shapiro_trans[1],
                'skewness': stats.skew(transformed),
                'kurtosis': stats.kurtosis(transformed),
                'formula': formula,
                'inverse': inverse
            }
            
            # If Box-Cox or Yeo-Johnson, store lambda
            if trans in ['boxcox', 'yeojohnson']:
                results['transformations'][trans]['lambda'] = float(lmbda)
                
        except Exception as e:
            # Skip transformations that fail
            results['transformations'][trans] = {
                'error': str(e)
            }
    
    # Find best transformation (highest p-value from Shapiro-Wilk test)
    best_p = results['original']['shapiro_p']
    best_trans = 'original'
    
    for trans, trans_results in results['transformations'].items():
        if 'shapiro_p' in trans_results and trans_results['shapiro_p'] > best_p:
            best_p = trans_results['shapiro_p']
            best_trans = trans
    
    # Check if it's worth transforming
    is_worth_transforming = best_trans != 'original' and best_p > 0.05
    
    results['best_transformation'] = best_trans
    results['normality_improved'] = best_trans != 'original'
    results['worth_transforming'] = is_worth_transforming
    
    # Recommendation
    if not is_worth_transforming:
        if results['original']['shapiro_p'] > 0.05:
            results['recommendation'] = "No transformation needed, data is already approximately normal."
        else:
            results['recommendation'] = "No transformation significantly improves normality."
    else:
        results['recommendation'] = f"Apply {best_trans} transformation to improve normality."
    
    return results


def plot_distribution_comparison(data: pd.Series, 
                                best_dist: str, 
                                params: List[float],
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the empirical distribution against the fitted theoretical distribution.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    best_dist : str
        Name of the best-fitting distribution
    params : List[float]
        Parameters of the best-fitting distribution
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
        
    Returns
    -------
    plt.Figure
        Figure object with the distribution comparison plot
    """
    # Clean data (no missing values)
    clean_data = data.dropna()
    
    # Return error figure if not enough data
    if len(clean_data) < 10:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Not enough data points for distribution plot", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Histogram with PDF
    sns.histplot(clean_data, kde=False, stat='density', alpha=0.6, ax=axes[0])
    
    # Get distribution object
    dist = getattr(stats, best_dist)
    
    # Generate points for PDF
    x = np.linspace(min(clean_data), max(clean_data), 1000)
    y = dist.pdf(x, *params)
    
    # Plot PDF
    axes[0].plot(x, y, 'r-', linewidth=2)
    
    # Set labels
    axes[0].set_title(f'Histogram with {best_dist} PDF')
    axes[0].set_xlabel(data.name)
    axes[0].set_ylabel('Density')
    
    # 2. PP plot (Probability-Probability plot)
    # Sort the data
    data_sorted = np.sort(clean_data)
    n = len(data_sorted)
    
    # Calculate empirical CDF
    p_empirical = np.arange(1, n + 1) / (n + 1)
    
    # Calculate theoretical CDF
    p_theoretical = dist.cdf(data_sorted, *params)
    
    # Plot PP plot
    axes[1].scatter(p_theoretical, p_empirical, alpha=0.6)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    
    # Set labels
    axes[1].set_title('P-P Plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Empirical Probabilities')
    
    # Add grid
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Set figure title
    dist_name = best_dist.capitalize()
    param_str = ', '.join(f'{p:.3f}' for p in params)
    fig.suptitle(f"Fitted {dist_name} Distribution ({param_str})", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def plot_transformation_comparison(data: pd.Series, 
                                  results: Dict[str, Any],
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot original data against transformations to compare normality.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    results : Dict[str, Any]
        Results from find_optimal_transformation function
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
        
    Returns
    -------
    plt.Figure
        Figure object with the transformation comparison plot
    """
    # Clean data (no missing values)
    clean_data = data.dropna()
    
    # Return error figure if not enough data
    if len(clean_data) < 10 or 'error' in results:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Not enough data for transformation comparison", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Get transformations to plot
    trans_to_plot = ['original'] + list(results['transformations'].keys())
    
    # Determine grid dimensions
    n_plots = len(trans_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots - 1) // n_cols + 1
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes if more than one row
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = [axes]
        
    # Plot each transformation
    for i, trans in enumerate(trans_to_plot):
        if i < len(axes):
            # Get transformed data
            if trans == 'original':
                trans_data = clean_data
                label = "Original Data"
                p_value = results['original']['shapiro_p']
            elif 'error' in results['transformations'][trans]:
                # Skip if transformation failed
                axes[i].set_visible(False)
                continue
            else:
                # Apply transformation
                if trans == 'log':
                    trans_data = np.log(clean_data)
                elif trans == 'sqrt':
                    trans_data = np.sqrt(clean_data)
                elif trans == 'reciprocal':
                    trans_data = 1 / clean_data
                elif trans == 'boxcox':
                    lmbda = results['transformations'][trans]['lambda']
                    trans_data, _ = stats.boxcox(clean_data, lmbda=lmbda)
                elif trans == 'yeojohnson':
                    lmbda = results['transformations'][trans]['lambda']
                    trans_data, _ = stats.yeojohnson(clean_data, lmbda=lmbda)
                
                # Get labels and p-value
                label = f"{trans.capitalize()}"
                if trans in ['boxcox', 'yeojohnson']:
                    label += f" (λ={results['transformations'][trans]['lambda']:.2f})"
                    
                p_value = results['transformations'][trans]['shapiro_p']
            
            # Plot histogram with KDE
            sns.histplot(trans_data, kde=True, ax=axes[i])
            
            # Add title and p-value
            axes[i].set_title(f"{label}\nShapiro p={p_value:.4f}")
            
            # Highlight the best transformation
            if trans == results['best_transformation']:
                # Use a colored background to highlight
                axes[i].set_facecolor('#e6ffe6')  # Light green
                axes[i].set_title(f"{label} (BEST)\nShapiro p={p_value:.4f}")
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add recommendation as figure title
    fig.suptitle(f"Transformation Comparison: {results['recommendation']}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig