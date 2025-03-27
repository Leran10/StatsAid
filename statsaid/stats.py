"""Statistical testing and model selection module for StatsAid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, List, Union, Optional, Tuple, Any


def select_statistical_test(data: pd.DataFrame, 
                           outcome: str, 
                           predictors: Optional[List[str]] = None,
                           study_design: Optional[str] = None,
                           alpha: float = 0.05) -> Dict[str, Any]:
    """
    Select appropriate statistical tests based on data characteristics and research question.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Name of the outcome/dependent variable
    predictors : List[str], optional
        Names of predictor/independent variables
    study_design : str, optional
        Type of study design (e.g., 'case_control', 'cohort', 'cross_sectional', 'longitudinal', 'rct')
    alpha : float, optional
        Significance level, by default 0.05
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing recommended statistical tests and their justifications
    """
    # Verify inputs
    if outcome not in data.columns:
        return {'error': f"Outcome variable '{outcome}' not found in dataset"}
    
    # Get predictor variables if not specified
    if predictors is None:
        predictors = [col for col in data.columns if col != outcome]
    else:
        # Verify all predictors exist
        for col in predictors:
            if col not in data.columns:
                return {'error': f"Predictor variable '{col}' not found in dataset"}
    
    # Get variable types
    outcome_type = 'categorical' if not pd.api.types.is_numeric_dtype(data[outcome]) or data[outcome].nunique() <= 10 else 'continuous'
    
    predictor_types = {}
    for col in predictors:
        if not pd.api.types.is_numeric_dtype(data[col]) or data[col].nunique() <= 5:
            predictor_types[col] = 'categorical'
        else:
            predictor_types[col] = 'continuous'
    
    # Check sample size
    sample_size = len(data)
    small_sample = sample_size < 30
    
    # Check normality of continuous outcome
    normality_info = {}
    if outcome_type == 'continuous':
        # Check outcome normality
        shapiro_test = stats.shapiro(data[outcome].dropna())
        normality_info[outcome] = {
            'shapiro_stat': shapiro_test[0],
            'shapiro_p': shapiro_test[1],
            'normal': shapiro_test[1] > alpha
        }
    
    # Check for paired/repeated measures design
    has_repeated_measures = False
    subject_id_candidates = [col for col in data.columns if 'id' in col.lower() or 'subject' in col.lower() or 'participant' in col.lower()]
    
    if study_design == 'longitudinal' or study_design == 'repeated_measures':
        has_repeated_measures = True
    elif subject_id_candidates and any('time' in col.lower() or 'visit' in col.lower() or 'day' in col.lower() or 'week' in col.lower() for col in data.columns):
        # Likely has repeated measures if we have IDs and time indicators
        has_repeated_measures = True
        
    # Initialize results with data characteristics
    results = {
        'data_characteristics': {
            'outcome_variable': outcome,
            'outcome_type': outcome_type,
            'predictor_variables': predictor_types,
            'sample_size': sample_size,
            'small_sample': small_sample,
            'has_repeated_measures': has_repeated_measures,
            'study_design': study_design
        },
        'recommended_tests': [],
        'test_details': {},
        'warnings': []
    }
    
    if outcome_type == 'continuous':
        results['data_characteristics']['outcome_normality'] = normality_info.get(outcome, {}).get('normal', False)
    
    # Generate test recommendations based on data characteristics
    # We'll handle different scenarios: univariate, bivariate, and multivariate analysis
    
    # 1. Univariate analysis (descriptive statistics)
    results['test_details']['univariate'] = {
        'outcome': {
            'tests': ['descriptive_statistics'],
            'visualizations': []
        }
    }
    
    if outcome_type == 'continuous':
        results['test_details']['univariate']['outcome']['visualizations'] = ['histogram', 'boxplot', 'qq_plot']
        if not normality_info.get(outcome, {}).get('normal', True):
            results['warnings'].append(f"The outcome variable '{outcome}' appears non-normal. Consider non-parametric tests or data transformation.")
    else:
        results['test_details']['univariate']['outcome']['visualizations'] = ['bar_chart', 'pie_chart']
    
    # 2. Bivariate analysis (one predictor)
    results['test_details']['bivariate'] = {}
    
    for predictor, pred_type in predictor_types.items():
        test_info = {
            'predictor_type': pred_type,
            'tests': [],
            'visualizations': []
        }
        
        # Determine appropriate tests based on variable types
        if outcome_type == 'continuous':
            if pred_type == 'categorical':
                # Continuous outcome, categorical predictor = group comparison
                if data[predictor].nunique() == 2:
                    # Two groups
                    if small_sample or not normality_info.get(outcome, {}).get('normal', True):
                        test_info['tests'].append('mann_whitney_u_test')
                        test_info['visualizations'] = ['boxplot', 'violin_plot']
                    else:
                        if has_repeated_measures:
                            test_info['tests'].append('paired_t_test')
                        else:
                            test_info['tests'].append('independent_t_test')
                        test_info['visualizations'] = ['boxplot', 'violin_plot', 'means_plot']
                        
                else:
                    # More than two groups
                    if small_sample or not normality_info.get(outcome, {}).get('normal', True):
                        test_info['tests'].append('kruskal_wallis_h_test')
                        test_info['visualizations'] = ['boxplot', 'violin_plot']
                    else:
                        if has_repeated_measures:
                            test_info['tests'].append('repeated_measures_anova')
                        else:
                            test_info['tests'].append('one_way_anova')
                        test_info['visualizations'] = ['boxplot', 'violin_plot', 'means_plot']
                        
            else:
                # Continuous outcome, continuous predictor = correlation/regression
                test_info['tests'].extend(['pearsons_correlation', 'simple_linear_regression'])
                
                if not normality_info.get(outcome, {}).get('normal', True):
                    test_info['tests'].append('spearmans_rank_correlation')
                    
                test_info['visualizations'] = ['scatter_plot', 'regression_plot']
                
        else:
            # Categorical outcome
            if pred_type == 'categorical':
                # Categorical outcome, categorical predictor = contingency table
                if data[predictor].nunique() == 2 and data[outcome].nunique() == 2:
                    # 2x2 table
                    if small_sample:
                        test_info['tests'].append('fishers_exact_test')
                    else:
                        test_info['tests'].append('chi_square_test')
                        
                    if study_design == 'case_control':
                        test_info['tests'].append('odds_ratio')
                    elif study_design == 'cohort' or study_design == 'longitudinal':
                        test_info['tests'].append('relative_risk')
                        
                    test_info['visualizations'] = ['grouped_bar_chart', 'mosaic_plot']
                else:
                    # Larger contingency table
                    test_info['tests'].append('chi_square_test')
                    test_info['visualizations'] = ['grouped_bar_chart', 'heatmap']
            else:
                # Categorical outcome, continuous predictor = logistic regression
                test_info['tests'].append('logistic_regression')
                test_info['visualizations'] = ['box_plot', 'probability_plot']
        
        results['test_details']['bivariate'][predictor] = test_info
        
    # 3. Multivariate analysis (multiple predictors)
    if len(predictors) > 1:
        multivariate_tests = []
        
        if outcome_type == 'continuous':
            # Multiple regression models
            multivariate_tests.append('multiple_linear_regression')
            
            if has_repeated_measures:
                multivariate_tests.append('mixed_effects_model')
                
            # Check for interactions
            if len([p for p in predictor_types.values() if p == 'categorical']) > 0:
                multivariate_tests.append('ancova')
                
            # Add regularization for many predictors
            if len(predictors) > 10:
                multivariate_tests.extend(['ridge_regression', 'lasso_regression'])
                
        else:
            # Classification models
            multivariate_tests.append('multiple_logistic_regression')
            
            if has_repeated_measures:
                multivariate_tests.append('generalized_linear_mixed_model')
                
            if len(predictors) > 10:
                multivariate_tests.append('regularized_logistic_regression')
        
        # Add these to the results
        results['test_details']['multivariate'] = {
            'tests': multivariate_tests,
            'visualizations': ['partial_dependence_plots', 'coefficient_plot']
        }
    
    # 4. Specific tests based on study design
    if study_design:
        design_specific_tests = []
        
        if study_design.lower() == 'case_control':
            design_specific_tests.extend(['odds_ratio', 'conditional_logistic_regression'])
            
        elif study_design.lower() == 'cohort':
            design_specific_tests.extend(['relative_risk', 'cox_proportional_hazards', 'kaplan_meier_curves'])
            
        elif study_design.lower() == 'cross_sectional':
            if outcome_type == 'categorical':
                design_specific_tests.extend(['prevalence_ratio', 'prevalence_odds_ratio'])
                
        elif study_design.lower() in ['longitudinal', 'repeated_measures', 'time_series']:
            design_specific_tests.extend(['mixed_effects_model', 'generalized_estimating_equations'])
            
            if outcome_type == 'continuous':
                design_specific_tests.append('repeated_measures_anova')
                
        elif study_design.lower() == 'rct':
            design_specific_tests.extend(['intention_to_treat_analysis', 'per_protocol_analysis'])
            
            if outcome_type == 'continuous':
                design_specific_tests.append('ancova_for_baseline_adjustment')
            
        # Add these to the results
        if design_specific_tests:
            results['test_details']['design_specific'] = {
                'tests': design_specific_tests
            }
    
    # 5. Compile the final list of recommended tests
    primary_tests = []
    secondary_tests = []
    
    # Add primary tests based on the research question
    if len(predictors) == 1:
        # Single predictor - use bivariate analysis
        pred = predictors[0]
        primary_tests.extend(results['test_details']['bivariate'][pred]['tests'][:1])
        if len(results['test_details']['bivariate'][pred]['tests']) > 1:
            secondary_tests.extend(results['test_details']['bivariate'][pred]['tests'][1:])
    else:
        # Multiple predictors - use multivariate analysis
        if 'multivariate' in results['test_details']:
            primary_tests.extend(results['test_details']['multivariate']['tests'][:1])
            if len(results['test_details']['multivariate']['tests']) > 1:
                secondary_tests.extend(results['test_details']['multivariate']['tests'][1:])
    
    # Add design-specific tests
    if 'design_specific' in results['test_details']:
        primary_tests.extend(results['test_details']['design_specific']['tests'][:1])
        if len(results['test_details']['design_specific']['tests']) > 1:
            secondary_tests.extend(results['test_details']['design_specific']['tests'][1:])
    
    # Add tests to results
    results['recommended_tests'] = {
        'primary': primary_tests,
        'secondary': secondary_tests
    }
    
    return results


def adjust_for_multiple_comparisons(p_values: Union[List[float], Dict[str, float]], 
                                   method: str = 'bonferroni',
                                   alpha: float = 0.05) -> Dict[str, Any]:
    """
    Adjust p-values for multiple comparisons.
    
    Parameters
    ----------
    p_values : Union[List[float], Dict[str, float]]
        List or dictionary of p-values to adjust
    method : str, optional
        Method for adjustment ('bonferroni', 'holm', 'bh' for Benjamini-Hochberg, 
        'by' for Benjamini-Yekutieli), by default 'bonferroni'
    alpha : float, optional
        Significance level, by default 0.05
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing adjusted p-values and significance indicators
    """
    # Convert dictionary to lists of keys and values if needed
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        values = list(p_values.values())
        is_dict = True
    else:
        values = p_values
        keys = [f"test_{i+1}" for i in range(len(values))]
        is_dict = False
    
    # Remove NaN values
    valid_indices = []
    valid_values = []
    for i, val in enumerate(values):
        if not np.isnan(val):
            valid_indices.append(i)
            valid_values.append(val)
    
    # Initialize results
    n_tests = len(valid_values)
    adjusted_values = np.empty(len(values))
    adjusted_values.fill(np.nan)
    
    # If no valid p-values, return early
    if n_tests == 0:
        return {'error': "No valid p-values provided"}
    
    # Perform adjustment based on the selected method
    if method.lower() == 'bonferroni':
        # Bonferroni correction
        for i, p in zip(valid_indices, valid_values):
            adjusted_values[i] = min(p * n_tests, 1.0)
            
    elif method.lower() == 'holm':
        # Holm-Bonferroni method
        sorted_indices = [x for _, x in sorted(zip(valid_values, valid_indices))]
        sorted_values = sorted(valid_values)
        
        for k, (i, p) in enumerate(zip(sorted_indices, sorted_values)):
            adjusted_values[i] = min(p * (n_tests - k), 1.0)
            
    elif method.lower() in ['bh', 'fdr']:
        # Benjamini-Hochberg procedure (FDR)
        sorted_indices = [x for _, x in sorted(zip(valid_values, valid_indices))]
        sorted_values = sorted(valid_values)
        
        for k, (i, p) in enumerate(zip(sorted_indices, sorted_values)):
            adjusted_values[i] = min(p * n_tests / (k + 1), 1.0)
            
    elif method.lower() == 'by':
        # Benjamini-Yekutieli procedure (conservative FDR)
        c = sum(1 / np.arange(1, n_tests + 1))  # Harmonic number
        sorted_indices = [x for _, x in sorted(zip(valid_values, valid_indices))]
        sorted_values = sorted(valid_values)
        
        for k, (i, p) in enumerate(zip(sorted_indices, sorted_values)):
            adjusted_values[i] = min(p * n_tests * c / (k + 1), 1.0)
            
    else:
        return {'error': f"Unknown adjustment method: {method}"}
    
    # Create result dictionary
    results = {
        'method': method,
        'n_tests': n_tests,
        'alpha': alpha,
        'original_p_values': dict(zip(keys, values)) if is_dict else values,
        'adjusted_p_values': dict(zip(keys, adjusted_values)) if is_dict else adjusted_values.tolist(),
        'significant': {}
    }
    
    # Add significance indicators
    for i, key in enumerate(keys):
        if not np.isnan(adjusted_values[i]):
            results['significant'][key] = adjusted_values[i] < alpha
    
    # Add interpretation
    results['interpretation'] = f"Using the {method} method to control for {n_tests} tests at α={alpha}"
    if method.lower() in ['bh', 'fdr', 'by']:
        results['interpretation'] += f" (controlling the false discovery rate)"
    else:
        results['interpretation'] += f" (controlling the family-wise error rate)"
        
    # Add practical significance description
    sig_count = sum(results['significant'].values())
    results['summary'] = f"{sig_count} out of {n_tests} tests remain significant after adjustment"
    
    return results


def calculate_effect_size(data: pd.DataFrame,
                         group_col: Optional[str] = None,
                         value_col: Optional[str] = None,
                         x: Optional[pd.Series] = None,
                         y: Optional[pd.Series] = None,
                         test_type: str = 'auto') -> Dict[str, Any]:
    """
    Calculate appropriate effect size measures for statistical tests.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data frame (when using group_col and value_col)
    group_col : str, optional
        Column with group assignments
    value_col : str, optional
        Column with values to compare between groups
    x : pd.Series, optional
        First data series (alternative to using data frame)
    y : pd.Series, optional
        Second data series (alternative to using data frame)
    test_type : str, optional
        Type of test ('t_test', 'correlation', 'chi_square', 'anova', or 'auto'), by default 'auto'
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing effect size measures and interpretations
    """
    # Determine which data format we're using
    if x is not None and y is not None:
        # Using direct series input
        pass
    elif data is not None and group_col is not None and value_col is not None:
        # Using data frame with group and value columns
        if group_col not in data.columns:
            return {'error': f"Group column '{group_col}' not found in dataset"}
        if value_col not in data.columns:
            return {'error': f"Value column '{value_col}' not found in dataset"}
        
        # Check if group column has exactly 2 unique values for t-test measures
        unique_groups = data[group_col].dropna().unique()
        if len(unique_groups) == 2 and (test_type == 'auto' or test_type == 't_test'):
            # Set up for t-test/Mann-Whitney
            x = data[data[group_col] == unique_groups[0]][value_col].dropna()
            y = data[data[group_col] == unique_groups[1]][value_col].dropna()
            if test_type == 'auto':
                test_type = 't_test'
        elif test_type == 'auto':
            # Determine test type from data
            if pd.api.types.is_numeric_dtype(data[value_col]):
                if pd.api.types.is_numeric_dtype(data[group_col]) or len(unique_groups) > 10:
                    # Both numeric - likely correlation
                    test_type = 'correlation'
                    x = data[group_col].dropna()
                    y = data[value_col].dropna()
                else:
                    # Categorical group, numeric value - likely ANOVA/Kruskal-Wallis
                    test_type = 'anova'
            else:
                # Categorical outcome - likely chi-square/Fisher's exact
                test_type = 'chi_square'
    else:
        return {'error': "Must provide either (x and y) or (data, group_col, and value_col)"}
    
    # Initialize results dictionary
    results = {
        'test_type': test_type,
        'effect_sizes': {},
        'interpretation': {},
        'magnitude': {}
    }
    
    # Calculate effect sizes based on test type
    if test_type == 't_test':
        # Cohen's d and related measures
        n1, n2 = len(x), len(y)
        mean1, mean2 = x.mean(), y.mean()
        var1, var2 = x.var(ddof=1), y.var(ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_sd
        results['effect_sizes']['cohens_d'] = d
        
        # Hedges' g (bias-corrected d)
        correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        g = d * correction
        results['effect_sizes']['hedges_g'] = g
        
        # Glass's delta (using control group SD)
        glass_delta = (mean2 - mean1) / np.sqrt(var1)
        results['effect_sizes']['glass_delta'] = glass_delta
        
        # Cohen's U3
        u3 = stats.norm.cdf(abs(d))
        results['effect_sizes']['cohens_u3'] = u3
        
        # Common language effect size
        cles = stats.norm.cdf(abs(d) / np.sqrt(2))
        results['effect_sizes']['common_language_effect_size'] = cles
        
        # Interpretation of Cohen's d
        if abs(d) < 0.2:
            magnitude = "Negligible"
        elif abs(d) < 0.5:
            magnitude = "Small"
        elif abs(d) < 0.8:
            magnitude = "Medium"
        else:
            magnitude = "Large"
            
        results['magnitude']['cohens_d'] = magnitude
        results['interpretation']['cohens_d'] = (
            f"{magnitude.lower()} effect size (d = {d:.2f}). "
            f"The difference between groups is equivalent to {abs(d):.2f} standard deviations."
        )
        
        results['interpretation']['common_language_effect_size'] = (
            f"The probability that a randomly selected value from group 2 is greater than "
            f"a randomly selected value from group 1 is {cles:.2%}."
        )
    
    elif test_type == 'correlation':
        # Pearson's r
        r, p = stats.pearsonr(x, y)
        results['effect_sizes']['pearsons_r'] = r
        
        # r-squared (coefficient of determination)
        r_squared = r ** 2
        results['effect_sizes']['r_squared'] = r_squared
        
        # Interpretation of Pearson's r
        if abs(r) < 0.1:
            magnitude = "Negligible"
        elif abs(r) < 0.3:
            magnitude = "Small"
        elif abs(r) < 0.5:
            magnitude = "Medium"
        else:
            magnitude = "Large"
            
        results['magnitude']['pearsons_r'] = magnitude
        results['interpretation']['pearsons_r'] = (
            f"{magnitude.lower()} correlation (r = {r:.2f}). "
            f"The predictor explains {r_squared:.2%} of the variance in the outcome."
        )
        
    elif test_type == 'chi_square':
        # For chi-square tests, we need the contingency table
        if x is not None and y is not None and len(x.unique()) <= 10 and len(y.unique()) <= 10:
            # Create contingency table
            contingency = pd.crosstab(x, y)
            
            # Cramer's V
            chi2 = stats.chi2_contingency(contingency)[0]
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            v = np.sqrt(chi2 / (n * min_dim))
            results['effect_sizes']['cramers_v'] = v
            
            # Phi coefficient (for 2x2 tables)
            if contingency.shape == (2, 2):
                phi = np.sqrt(chi2 / n)
                results['effect_sizes']['phi_coefficient'] = phi
            
            # Interpretation of Cramer's V
            if v < 0.1:
                magnitude = "Negligible"
            elif v < 0.3:
                magnitude = "Small"
            elif v < 0.5:
                magnitude = "Medium"
            else:
                magnitude = "Large"
                
            results['magnitude']['cramers_v'] = magnitude
            results['interpretation']['cramers_v'] = (
                f"{magnitude.lower()} association (Cramer's V = {v:.2f}). "
                f"The variables show a {magnitude.lower()} degree of association."
            )
            
    elif test_type == 'anova':
        if group_col is not None and value_col is not None and data is not None:
            # Calculate eta-squared and omega-squared
            groups = data.groupby(group_col)[value_col].apply(list)
            group_means = data.groupby(group_col)[value_col].mean()
            grand_mean = data[value_col].mean()
            
            # Sum of squares calculations
            ss_total = sum((data[value_col] - grand_mean) ** 2)
            ss_between = sum(len(group) * (mean - grand_mean) ** 2 for group, mean in group_means.items())
            ss_within = ss_total - ss_between
            
            # Degrees of freedom
            df_between = len(groups) - 1
            df_within = len(data) - len(groups)
            df_total = len(data) - 1
            
            # Mean squares
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            
            # F-statistic
            f_stat = ms_between / ms_within
            
            # Effect size measures
            eta_squared = ss_between / ss_total
            omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
            
            results['effect_sizes']['eta_squared'] = eta_squared
            results['effect_sizes']['omega_squared'] = omega_squared
            
            # Interpretation of eta-squared
            if eta_squared < 0.01:
                magnitude = "Negligible"
            elif eta_squared < 0.06:
                magnitude = "Small"
            elif eta_squared < 0.14:
                magnitude = "Medium"
            else:
                magnitude = "Large"
                
            results['magnitude']['eta_squared'] = magnitude
            results['interpretation']['eta_squared'] = (
                f"{magnitude.lower()} effect (η² = {eta_squared:.2f}). "
                f"The grouping variable explains {eta_squared:.2%} of the variance in the outcome."
            )
            
    # Add additional information
    results['summary'] = f"Calculated {len(results['effect_sizes'])} effect size measure(s) for {test_type}"
    
    return results


def perform_bayesian_analysis(data: pd.DataFrame,
                             outcome: str,
                             predictor: Optional[str] = None,
                             priors: Optional[Dict[str, Any]] = None,
                             method: str = 'default') -> Dict[str, Any]:
    """
    Perform basic Bayesian analysis for common statistical tests.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Name of the outcome variable
    predictor : str, optional
        Name of the predictor variable
    priors : Dict[str, Any], optional
        Prior distributions for parameters
    method : str, optional
        Method to use ('default', 't_test', 'correlation', 'regression'), by default 'default'
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing Bayesian analysis results
    """
    try:
        import arviz as az
        from scipy import stats
    except ImportError:
        return {
            'error': "Bayesian analysis requires additional packages. Please install with: pip install arviz pymc"
        }
    
    # Verify inputs
    if outcome not in data.columns:
        return {'error': f"Outcome variable '{outcome}' not found in dataset"}
    
    if predictor is not None and predictor not in data.columns:
        return {'error': f"Predictor variable '{predictor}' not found in dataset"}
    
    # Basic information about variables
    outcome_data = data[outcome].dropna()
    
    # Default to simple Bayesian estimation of mean and standard deviation
    if method == 'default' or predictor is None:
        try:
            import pymc as pm
            
            # Prior specifications
            if priors is None:
                # Set default priors
                priors = {}
                
            # Default prior for mean: Normal(mean=data.mean(), sd=2*data.std())
            mu_prior_mean = priors.get('mu_mean', outcome_data.mean())
            mu_prior_sd = priors.get('mu_sd', 2 * outcome_data.std())
            
            # Default prior for standard deviation: HalfNormal(sd=data.std())
            sigma_prior_sd = priors.get('sigma_sd', outcome_data.std())
            
            # Create model
            with pm.Model() as model:
                # Priors
                mu = pm.Normal('mu', mu=mu_prior_mean, sigma=mu_prior_sd)
                sigma = pm.HalfNormal('sigma', sigma=sigma_prior_sd)
                
                # Likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=outcome_data)
                
                # Sample from posterior
                trace = pm.sample(1000, tune=1000, progressbar=False, cores=1)
                
            # Extract results
            summary = az.summary(trace)
            
            # Create results dictionary
            results = {
                'method': 'bayesian_estimation',
                'posterior': {
                    'mean': {
                        'mean': summary.loc['mu', 'mean'],
                        'sd': summary.loc['mu', 'sd'],
                        'hdi_3%': summary.loc['mu', 'hdi_3%'],
                        'hdi_97%': summary.loc['mu', 'hdi_97%']
                    },
                    'sigma': {
                        'mean': summary.loc['sigma', 'mean'],
                        'sd': summary.loc['sigma', 'sd'],
                        'hdi_3%': summary.loc['sigma', 'hdi_3%'],
                        'hdi_97%': summary.loc['sigma', 'hdi_97%']
                    }
                },
                'interpretation': {}
            }
            
            # Add interpretation
            results['interpretation']['mean'] = (
                f"The posterior mean is {summary.loc['mu', 'mean']:.2f} "
                f"with a 94% HDI (Highest Density Interval) of "
                f"[{summary.loc['mu', 'hdi_3%']:.2f}, {summary.loc['mu', 'hdi_97%']:.2f}]."
            )
            
            # Check if HDI excludes zero (evidence of effect)
            if summary.loc['mu', 'hdi_3%'] > 0 or summary.loc['mu', 'hdi_97%'] < 0:
                results['interpretation']['conclusion'] = (
                    "There is strong evidence that the true mean is different from zero, "
                    "as the 94% HDI does not include zero."
                )
            else:
                results['interpretation']['conclusion'] = (
                    "There is insufficient evidence to conclude that the true mean "
                    "is different from zero, as the 94% HDI includes zero."
                )
            
            return results
            
        except ImportError:
            return {
                'error': "Bayesian analysis requires PyMC. Please install with: pip install pymc"
            }
            
    elif method == 't_test' and predictor is not None:
        # Bayesian approach to t-test
        try:
            import pymc as pm
            
            # Get group data
            predictor_data = data[predictor].dropna()
            
            # Only works for binary predictors
            unique_groups = predictor_data.unique()
            if len(unique_groups) != 2:
                return {'error': "Bayesian t-test requires a binary predictor (exactly 2 groups)"}
            
            group1_data = outcome_data[predictor_data == unique_groups[0]]
            group2_data = outcome_data[predictor_data == unique_groups[1]]
            
            # Set default priors
            if priors is None:
                priors = {}
                
            # Prior for group means
            mu1_prior_mean = priors.get('mu1_mean', group1_data.mean())
            mu1_prior_sd = priors.get('mu1_sd', 2 * group1_data.std())
            
            mu2_prior_mean = priors.get('mu2_mean', group2_data.mean())
            mu2_prior_sd = priors.get('mu2_sd', 2 * group2_data.std())
            
            # Prior for standard deviations
            sigma_prior_sd = priors.get('sigma_sd', outcome_data.std())
            
            # Create model
            with pm.Model() as model:
                # Priors
                mu1 = pm.Normal('mu1', mu=mu1_prior_mean, sigma=mu1_prior_sd)
                mu2 = pm.Normal('mu2', mu=mu2_prior_mean, sigma=mu2_prior_sd)
                
                # Shared standard deviation (assuming equal variances)
                sigma = pm.HalfNormal('sigma', sigma=sigma_prior_sd)
                
                # Effect size (difference in means)
                delta = pm.Deterministic('delta', mu2 - mu1)
                
                # Standardized effect size (Cohen's d)
                cohens_d = pm.Deterministic('cohens_d', delta / sigma)
                
                # Likelihoods
                likelihood1 = pm.Normal('likelihood1', mu=mu1, sigma=sigma, observed=group1_data)
                likelihood2 = pm.Normal('likelihood2', mu=mu2, sigma=sigma, observed=group2_data)
                
                # Sample from posterior
                trace = pm.sample(1000, tune=1000, progressbar=False, cores=1)
                
            # Extract results
            summary = az.summary(trace)
            
            # Create results dictionary
            results = {
                'method': 'bayesian_t_test',
                'posterior': {
                    'mu1': {
                        'mean': summary.loc['mu1', 'mean'],
                        'sd': summary.loc['mu1', 'sd'],
                        'hdi_3%': summary.loc['mu1', 'hdi_3%'],
                        'hdi_97%': summary.loc['mu1', 'hdi_97%']
                    },
                    'mu2': {
                        'mean': summary.loc['mu2', 'mean'],
                        'sd': summary.loc['mu2', 'sd'],
                        'hdi_3%': summary.loc['mu2', 'hdi_3%'],
                        'hdi_97%': summary.loc['mu2', 'hdi_97%']
                    },
                    'delta': {
                        'mean': summary.loc['delta', 'mean'],
                        'sd': summary.loc['delta', 'sd'],
                        'hdi_3%': summary.loc['delta', 'hdi_3%'],
                        'hdi_97%': summary.loc['delta', 'hdi_97%']
                    },
                    'cohens_d': {
                        'mean': summary.loc['cohens_d', 'mean'],
                        'sd': summary.loc['cohens_d', 'sd'],
                        'hdi_3%': summary.loc['cohens_d', 'hdi_3%'],
                        'hdi_97%': summary.loc['cohens_d', 'hdi_97%']
                    }
                },
                'interpretation': {}
            }
            
            # Add interpretation
            results['interpretation']['difference'] = (
                f"The posterior mean difference between groups is {summary.loc['delta', 'mean']:.2f} "
                f"with a 94% HDI of [{summary.loc['delta', 'hdi_3%']:.2f}, {summary.loc['delta', 'hdi_97%']:.2f}]."
            )
            
            # Effect size interpretation
            d_mean = summary.loc['cohens_d', 'mean']
            if abs(d_mean) < 0.2:
                d_magnitude = "negligible"
            elif abs(d_mean) < 0.5:
                d_magnitude = "small"
            elif abs(d_mean) < 0.8:
                d_magnitude = "medium"
            else:
                d_magnitude = "large"
                
            results['interpretation']['effect_size'] = (
                f"The standardized effect size (Cohen's d) is estimated at {d_mean:.2f}, "
                f"which is considered a {d_magnitude} effect."
            )
            
            # Evidence interpretation
            if summary.loc['delta', 'hdi_3%'] > 0:
                results['interpretation']['conclusion'] = (
                    "There is strong evidence that Group 2 has a higher mean than Group 1, "
                    "as the 94% HDI for the difference is entirely above zero."
                )
            elif summary.loc['delta', 'hdi_97%'] < 0:
                results['interpretation']['conclusion'] = (
                    "There is strong evidence that Group 1 has a higher mean than Group 2, "
                    "as the 94% HDI for the difference is entirely below zero."
                )
            else:
                results['interpretation']['conclusion'] = (
                    "There is insufficient evidence to conclude that the groups differ, "
                    "as the 94% HDI for the difference includes zero."
                )
            
            # Calculate probability of direction
            samples = trace.posterior['delta'].values.flatten()
            if summary.loc['delta', 'mean'] > 0:
                prob_direction = (samples > 0).mean() * 100
                direction = "greater than"
            else:
                prob_direction = (samples < 0).mean() * 100
                direction = "less than"
                
            results['posterior']['probability_of_direction'] = prob_direction
            results['interpretation']['probability'] = (
                f"The probability that the true difference is {direction} zero is {prob_direction:.1f}%."
            )
            
            return results
            
        except ImportError:
            return {
                'error': "Bayesian t-test requires PyMC. Please install with: pip install pymc"
            }
            
    elif method == 'correlation' and predictor is not None:
        # Bayesian correlation analysis
        try:
            import pymc as pm
            
            # Get predictor data
            predictor_data = data[predictor].dropna()
            
            # Get only rows where both variables are non-missing
            valid_mask = ~outcome_data.isna() & ~predictor_data.isna()
            x = predictor_data[valid_mask].values
            y = outcome_data[valid_mask].values
            
            # Standardize for numerical stability
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()
            
            # Set default priors
            if priors is None:
                priors = {}
                
            # Create model
            with pm.Model() as model:
                # Prior for correlation coefficient (using Beta transformed to [-1, 1])
                # Default is a uniform prior over [-1, 1]
                alpha = priors.get('alpha', 1)
                beta = priors.get('beta', 1)
                
                rho = pm.Beta('rho_raw', alpha=alpha, beta=beta)
                rho = pm.Deterministic('rho', 2 * rho - 1)  # Transform to [-1, 1]
                
                # Multivariate normal likelihood
                cov = pm.math.stack([
                    [1, rho],
                    [rho, 1]
                ])
                
                likelihood = pm.MvNormal('likelihood', mu=[0, 0], cov=cov, observed=np.column_stack([x, y]))
                
                # Sample from posterior
                trace = pm.sample(1000, tune=1000, progressbar=False, cores=1)
                
            # Extract results
            summary = az.summary(trace)
            
            # Create results dictionary
            results = {
                'method': 'bayesian_correlation',
                'posterior': {
                    'rho': {
                        'mean': summary.loc['rho', 'mean'],
                        'sd': summary.loc['rho', 'sd'],
                        'hdi_3%': summary.loc['rho', 'hdi_3%'],
                        'hdi_97%': summary.loc['rho', 'hdi_97%']
                    }
                },
                'interpretation': {}
            }
            
            # Add interpretation
            results['interpretation']['correlation'] = (
                f"The posterior mean correlation is {summary.loc['rho', 'mean']:.2f} "
                f"with a 94% HDI of [{summary.loc['rho', 'hdi_3%']:.2f}, {summary.loc['rho', 'hdi_97%']:.2f}]."
            )
            
            # Effect size interpretation
            r_mean = summary.loc['rho', 'mean']
            if abs(r_mean) < 0.1:
                r_magnitude = "negligible"
            elif abs(r_mean) < 0.3:
                r_magnitude = "small"
            elif abs(r_mean) < 0.5:
                r_magnitude = "medium"
            else:
                r_magnitude = "large"
                
            results['interpretation']['effect_size'] = (
                f"The correlation coefficient is estimated at {r_mean:.2f}, "
                f"which is considered a {r_magnitude} effect."
            )
            
            # Evidence interpretation
            if summary.loc['rho', 'hdi_3%'] > 0:
                results['interpretation']['conclusion'] = (
                    "There is strong evidence for a positive correlation, "
                    "as the 94% HDI is entirely above zero."
                )
            elif summary.loc['rho', 'hdi_97%'] < 0:
                results['interpretation']['conclusion'] = (
                    "There is strong evidence for a negative correlation, "
                    "as the 94% HDI is entirely below zero."
                )
            else:
                results['interpretation']['conclusion'] = (
                    "There is insufficient evidence to conclude that the correlation is different from zero, "
                    "as the 94% HDI includes zero."
                )
            
            # Calculate probability of direction
            samples = trace.posterior['rho'].values.flatten()
            if r_mean > 0:
                prob_direction = (samples > 0).mean() * 100
                direction = "positive"
            else:
                prob_direction = (samples < 0).mean() * 100
                direction = "negative"
                
            results['posterior']['probability_of_direction'] = prob_direction
            results['interpretation']['probability'] = (
                f"The probability that the true correlation is {direction} is {prob_direction:.1f}%."
            )
            
            # Calculate Bayes Factor (using Savage-Dickey density ratio)
            # This is an approximation
            from scipy.stats import gaussian_kde
            
            # Compute the density at rho = 0
            kde = gaussian_kde(samples)
            bf10 = kde(0)[0] / (0.5)  # Compare to uniform prior density at 0
            
            results['posterior']['bayes_factor'] = bf10
            
            if bf10 > 100:
                evidence = "extreme"
            elif bf10 > 30:
                evidence = "very strong"
            elif bf10 > 10:
                evidence = "strong"
            elif bf10 > 3:
                evidence = "moderate"
            elif bf10 > 1:
                evidence = "anecdotal"
            else:
                evidence = "no"
                
            results['interpretation']['bayes_factor'] = (
                f"The Bayes Factor (BF10) is approximately {bf10:.2f}, indicating {evidence} "
                f"evidence for the alternative hypothesis compared to the null hypothesis."
            )
            
            return results
            
        except ImportError:
            return {
                'error': "Bayesian correlation requires PyMC. Please install with: pip install pymc"
            }
    else:
        return {
            'error': f"Unsupported Bayesian analysis method: {method}"
        }


def plot_test_selection_flowchart(data_characteristics: Dict[str, Any], figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a flowchart visualization to explain the test selection process.
    
    Parameters
    ----------
    data_characteristics : Dict[str, Any]
        Dictionary containing data characteristics (from select_statistical_test)
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 10)
        
    Returns
    -------
    plt.Figure
        Figure object with the test selection flowchart
    """
    import networkx as nx
    
    # Extract relevant characteristics
    outcome_type = data_characteristics.get('outcome_type', 'unknown')
    has_repeated_measures = data_characteristics.get('has_repeated_measures', False)
    small_sample = data_characteristics.get('small_sample', False)
    outcome_normal = data_characteristics.get('outcome_normality', True)
    predictor_types = data_characteristics.get('predictor_variables', {})
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for decision points and tests
    # Format: (node_type, text)
    # node_type can be 'decision', 'test', 'start', or 'note'
    G.add_node(1, type='start', text='Start')
    G.add_node(2, type='decision', text='Outcome\nVariable Type?')
    G.add_node(3, type='decision', text='# of Predictors?')
    G.add_node(4, type='decision', text='Predictor Type?')
    G.add_node(5, type='decision', text='# of Groups?')
    G.add_node(6, type='decision', text='Repeated\nMeasures?')
    G.add_node(7, type='decision', text='Normality?')
    G.add_node(8, type='decision', text='Sample Size?')
    
    # Add nodes for tests (continuous outcome)
    G.add_node(11, type='test', text='t-test')
    G.add_node(12, type='test', text='Mann-Whitney U')
    G.add_node(13, type='test', text='Paired t-test')
    G.add_node(14, type='test', text='Wilcoxon Signed-Rank')
    G.add_node(15, type='test', text='ANOVA')
    G.add_node(16, type='test', text='Kruskal-Wallis')
    G.add_node(17, type='test', text='RM-ANOVA')
    G.add_node(18, type='test', text='Friedman Test')
    G.add_node(19, type='test', text='Linear Regression')
    G.add_node(20, type='test', text='Multiple Linear\nRegression')
    G.add_node(21, type='test', text='Mixed Effects\nModel')
    
    # Add nodes for tests (categorical outcome)
    G.add_node(31, type='decision', text='# of Groups?')
    G.add_node(32, type='decision', text='Expected\nCell Counts?')
    G.add_node(33, type='test', text='Chi-Square Test')
    G.add_node(34, type='test', text='Fisher\'s Exact')
    G.add_node(35, type='test', text='McNemar\'s Test')
    G.add_node(36, type='test', text='Cochran\'s Q')
    G.add_node(37, type='test', text='Logistic Regression')
    G.add_node(38, type='test', text='Multinomial\nLogistic Regression')
    G.add_node(39, type='test', text='Multiple Logistic\nRegression')
    
    # Add edges for continuous outcome
    G.add_edge(1, 2)
    G.add_edge(2, 3, label='Continuous')
    G.add_edge(3, 4, label='1')
    G.add_edge(3, 20, label='≥2')
    G.add_edge(20, 21, label='With\nRepeated\nMeasures')
    G.add_edge(4, 5, label='Categorical')
    G.add_edge(4, 19, label='Continuous')
    G.add_edge(5, 6, label='2')
    G.add_edge(5, 7, label='>2')
    G.add_edge(6, 7, label='Yes')
    G.add_edge(6, 7, label='No')
    G.add_edge(7, 8, label='Normal')
    G.add_edge(7, 8, label='Non-normal')
    
    # Connect to appropriate tests for continuous outcome
    G.add_edge(8, 11, label='Large\nNormal\nIndependent')
    G.add_edge(8, 12, label='Small or\nNon-normal\nIndependent')
    G.add_edge(8, 13, label='Large\nNormal\nPaired')
    G.add_edge(8, 14, label='Small or\nNon-normal\nPaired')
    G.add_edge(8, 15, label='Large\nNormal\nIndependent\n>2 groups')
    G.add_edge(8, 16, label='Small or\nNon-normal\nIndependent\n>2 groups')
    G.add_edge(8, 17, label='Large\nNormal\nRepeated\n>2 groups')
    G.add_edge(8, 18, label='Small or\nNon-normal\nRepeated\n>2 groups')
    
    # Add edges for categorical outcome
    G.add_edge(2, 31, label='Categorical')
    G.add_edge(31, 32, label='2')
    G.add_edge(31, 38, label='>2')
    G.add_edge(32, 33, label='≥5')
    G.add_edge(32, 34, label='<5')
    G.add_edge(32, 35, label='Paired\nData')
    G.add_edge(32, 36, label='Repeated\nMeasures\n>2 timepoints')
    G.add_edge(31, 37, label='Binary\nOutcome')
    G.add_edge(37, 39, label='Multiple\nPredictors')
    
    # Set positions using a layered layout
    pos = {
        1: (5, 10),
        2: (5, 9),
        3: (2, 8),
        4: (2, 7),
        5: (1, 6),
        6: (1, 5),
        7: (1, 4),
        8: (1, 3),
        11: (2, 2),
        12: (3, 2),
        13: (4, 2),
        14: (5, 2),
        15: (6, 2),
        16: (7, 2),
        17: (8, 2),
        18: (9, 2),
        19: (4, 6),
        20: (6, 7),
        21: (6, 6),
        31: (8, 8),
        32: (8, 7),
        33: (10, 6),
        34: (10, 5),
        35: (10, 4),
        36: (10, 3),
        37: (8, 6),
        38: (8, 5),
        39: (8, 4)
    }
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw the nodes
    for node, data in G.nodes(data=True):
        node_type = data['type']
        text = data['text']
        
        if node_type == 'decision':
            # Diamond for decision points
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='d', 
                                  node_size=3000, node_color='lightblue', alpha=0.8)
        elif node_type == 'test':
            # Rectangle for tests
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='s', 
                                  node_size=3000, node_color='lightgreen', alpha=0.8)
        elif node_type == 'start':
            # Oval for start
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='o', 
                                  node_size=2000, node_color='lightcoral', alpha=0.8)
        elif node_type == 'note':
            # Rounded rectangle for notes
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='h', 
                                  node_size=2000, node_color='lightyellow', alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif',
                           horizontalalignment='center', verticalalignment='center')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=20, arrowstyle='->', alpha=0.5)
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if 'label' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, alpha=0.7)
    
    # Highlight recommended tests based on data characteristics
    recommended_tests = []
    
    if outcome_type == 'continuous':
        if len(predictor_types) == 1:
            pred_type = list(predictor_types.values())[0]
            
            if pred_type == 'continuous':
                recommended_tests.append(19)  # Linear Regression
            else:
                # Categorical predictor
                n_groups = 2  # Simplification, would need to check actual data
                
                if n_groups == 2:
                    if has_repeated_measures:
                        if small_sample or not outcome_normal:
                            recommended_tests.append(14)  # Wilcoxon
                        else:
                            recommended_tests.append(13)  # Paired t-test
                    else:
                        if small_sample or not outcome_normal:
                            recommended_tests.append(12)  # Mann-Whitney
                        else:
                            recommended_tests.append(11)  # t-test
                else:
                    if has_repeated_measures:
                        if small_sample or not outcome_normal:
                            recommended_tests.append(18)  # Friedman
                        else:
                            recommended_tests.append(17)  # RM-ANOVA
                    else:
                        if small_sample or not outcome_normal:
                            recommended_tests.append(16)  # Kruskal-Wallis
                        else:
                            recommended_tests.append(15)  # ANOVA
        else:
            if has_repeated_measures:
                recommended_tests.append(21)  # Mixed Effects Model
            else:
                recommended_tests.append(20)  # Multiple Linear Regression
    else:
        # Categorical outcome
        if len(predictor_types) == 1:
            pred_type = list(predictor_types.values())[0]
            
            if pred_type == 'continuous':
                recommended_tests.append(37)  # Logistic Regression
            else:
                # Categorical predictor
                n_groups = 2  # Simplification
                
                if n_groups == 2:
                    if has_repeated_measures:
                        recommended_tests.append(35)  # McNemar's Test
                    else:
                        cell_counts_adequate = True  # Simplification
                        if cell_counts_adequate:
                            recommended_tests.append(33)  # Chi-Square
                        else:
                            recommended_tests.append(34)  # Fisher's Exact
                else:
                    if has_repeated_measures:
                        recommended_tests.append(36)  # Cochran's Q
                    else:
                        recommended_tests.append(38)  # Multinomial Logistic
        else:
            recommended_tests.append(39)  # Multiple Logistic Regression
    
    # Highlight recommended tests
    if recommended_tests:
        nx.draw_networkx_nodes(G, pos, nodelist=recommended_tests, node_shape='s', 
                             node_size=3000, node_color='gold', alpha=0.8)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='Start'),
        plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='lightblue', markersize=15, label='Decision Point'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=15, label='Statistical Test'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=15, label='Recommended Test')
    ]
    
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
              fancybox=True, shadow=True, ncol=4)
    
    # Remove axis
    plt.axis('off')
    
    # Add title
    plt.title('Statistical Test Selection Flowchart', fontsize=16, y=1.05)
    
    return plt.gcf()


def plot_effect_sizes(effect_sizes: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a visualization of effect size measures and their interpretations.
    
    Parameters
    ----------
    effect_sizes : Dict[str, Any]
        Dictionary containing effect size results (from calculate_effect_size)
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
        
    Returns
    -------
    plt.Figure
        Figure object with effect size visualization
    """
    # Extract relevant information
    test_type = effect_sizes.get('test_type', 'unknown')
    effect_metrics = effect_sizes.get('effect_sizes', {})
    magnitudes = effect_sizes.get('magnitude', {})
    
    if not effect_metrics:
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No effect size metrics available", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # 1. Bar chart of effect size metrics
    metrics = []
    values = []
    
    for metric, value in effect_metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            metrics.append(metric)
            values.append(value)
    
    # Plot bars
    bars = axes[0].bar(metrics, values, color='skyblue')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add reference lines for interpretation (if applicable)
    if test_type == 't_test' and 'cohens_d' in effect_metrics:
        # Reference lines for Cohen's d
        axes[0].axhline(y=0.2, linestyle='--', color='gray', alpha=0.5)
        axes[0].axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
        axes[0].axhline(y=0.8, linestyle='--', color='gray', alpha=0.5)
        
        # Add text labels
        axes[0].text(0, 0.15, 'Small', ha='center', va='bottom', fontsize=8, alpha=0.7)
        axes[0].text(0, 0.45, 'Medium', ha='center', va='bottom', fontsize=8, alpha=0.7)
        axes[0].text(0, 0.75, 'Large', ha='center', va='bottom', fontsize=8, alpha=0.7)
        
    elif test_type == 'correlation' and 'pearsons_r' in effect_metrics:
        # Reference lines for correlation
        axes[0].axhline(y=0.1, linestyle='--', color='gray', alpha=0.5)
        axes[0].axhline(y=0.3, linestyle='--', color='gray', alpha=0.5)
        axes[0].axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
        
        # Add text labels
        axes[0].text(0, 0.05, 'Small', ha='center', va='bottom', fontsize=8, alpha=0.7)
        axes[0].text(0, 0.25, 'Medium', ha='center', va='bottom', fontsize=8, alpha=0.7)
        axes[0].text(0, 0.45, 'Large', ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Set labels
    axes[0].set_title('Effect Size Metrics')
    axes[0].set_ylabel('Effect Size')
    
    # Rotate x-axis labels for better readability
    axes[0].set_xticklabels(metrics, rotation=45, ha='right')
    
    # 2. Interpretation panel
    axes[1].axis('off')  # No actual plotting here
    
    # Create a text box with interpretation
    text_content = "Effect Size Interpretation:\n\n"
    
    for metric, magnitude in magnitudes.items():
        text_content += f"{metric}: {magnitude}\n"
    
    # Add additional interpretation if available
    if 'interpretation' in effect_sizes:
        text_content += "\nDetails:\n"
        for metric, interp in effect_sizes['interpretation'].items():
            if len(interp) < 100:  # Keep it short
                text_content += f"\n{interp}"
    
    axes[1].text(0, 0.9, text_content, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Set title
    test_name = test_type.replace('_', ' ').title()
    fig.suptitle(f'Effect Size Analysis for {test_name}', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig