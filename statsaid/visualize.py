"""Visualization functions for data exploration."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Union, Optional, Tuple, Any


def plot_missing_values(data: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap visualization of missing values in the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    plt.Figure
        Figure object with the missing values heatmap
    """
    # Create mask of missing values
    missing_mask = data.isna()
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the heatmap
    ax = sns.heatmap(
        missing_mask, 
        cmap=['#FFFFFF', '#2171b5'],
        cbar_kws={'label': 'Missing'},
        yticklabels=False
    )
    
    # Set labels and title
    plt.title('Missing Values Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Samples')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def plot_missing_bar(data: pd.DataFrame) -> plt.Figure:
    """
    Create a bar chart showing the percentage of missing values per column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    plt.Figure
        Figure object with the missing values bar chart
    """
    # Calculate percentage of missing values per column
    missing_pct = (data.isna().mean() * 100)
    
    # Sort by percentage
    missing_pct = missing_pct.sort_values(ascending=False)
    
    # Only plot columns with missing values
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) == 0:
        # No missing values
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No missing values in the dataset", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Create the figure
    plt.figure(figsize=(10, max(6, len(missing_pct) * 0.3)))
    
    # Plot the bar chart
    bars = plt.barh(missing_pct.index, missing_pct.values, color='#2171b5')
    
    # Add percentage labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center')
    
    # Set labels and title
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Missing (%)')
    plt.ylabel('Features')
    
    # Set x-axis limit
    plt.xlim(0, 100)
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def plot_distributions(data: pd.DataFrame, max_cols: int = 20) -> Dict[str, plt.Figure]:
    """
    Create distribution plots for numeric and categorical variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    max_cols : int, optional
        Maximum number of columns to plot, by default 20
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing figures for numeric and categorical variables
    """
    figures = {}
    
    # Get numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Limit the number of columns to plot
    if len(numeric_cols) > max_cols:
        numeric_cols = numeric_cols[:max_cols]
    
    if len(categorical_cols) > max_cols:
        categorical_cols = categorical_cols[:max_cols]
        
    # Plot distributions for numeric variables
    if numeric_cols:
        # Determine grid dimensions
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) - 1) // n_cols + 1
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # Plot each variable
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(data[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                # Add text with basic stats
                stats_text = (
                    f"Mean: {data[col].mean():.2f}\n"
                    f"Median: {data[col].median():.2f}\n"
                    f"Std: {data[col].std():.2f}\n"
                    f"Missing: {data[col].isna().sum()} ({data[col].isna().mean()*100:.1f}%)"
                )
                axes[i].text(0.95, 0.95, stats_text, 
                             transform=axes[i].transAxes,
                             verticalalignment='top', 
                             horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        figures['numeric'] = fig
    
    # Plot distributions for categorical variables
    if categorical_cols:
        # Determine grid dimensions
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) - 1) // n_cols + 1
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # Plot each variable
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Get value counts and limit to top categories if too many
                value_counts = data[col].value_counts()
                if len(value_counts) > 15:
                    # Keep top categories and group others
                    top_counts = value_counts.head(14)
                    other_count = value_counts.iloc[14:].sum()
                    
                    # Create new series with 'Other' category
                    value_counts = pd.concat([top_counts, pd.Series([other_count], index=['Other'])])
                
                # Bar plot
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                
                # Rotate x-axis labels for better readability
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                
                # Add text with basic stats
                stats_text = (
                    f"Unique values: {data[col].nunique()}\n"
                    f"Most common: {data[col].value_counts().index[0]}\n"
                    f"Missing: {data[col].isna().sum()} ({data[col].isna().mean()*100:.1f}%)"
                )
                axes[i].text(0.95, 0.95, stats_text, 
                             transform=axes[i].transAxes,
                             verticalalignment='top', 
                             horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        figures['categorical'] = fig
    
    return figures


def plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> plt.Figure:
    """
    Create a correlation matrix heatmap for numeric variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall'), by default 'pearson'
        
    Returns
    -------
    plt.Figure
        Figure object with the correlation matrix heatmap
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.shape[1] <= 1:
        # Not enough numeric columns for correlation
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "Not enough numeric columns to compute correlation matrix", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Compute correlation matrix
    corr_matrix = numeric_data.corr(method=method)
    
    # Determine figure size based on number of variables
    n_vars = len(corr_matrix)
    fig_size = max(8, n_vars * 0.5)
    
    # Create figure
    plt.figure(figsize=(fig_size, fig_size))
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    ax = sns.heatmap(
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
    plt.title(f'{method.capitalize()} Correlation Matrix')
    
    # Rotate y-axis labels
    plt.yticks(rotation=0)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def plot_pairplot(data: pd.DataFrame, max_cols: int = 5, hue: str = None) -> plt.Figure:
    """
    Create a pairplot for numeric variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    max_cols : int, optional
        Maximum number of columns to include, by default 5
    hue : str, optional
        Column name to use for color encoding, by default None
        
    Returns
    -------
    plt.Figure
        Figure object with the pairplot
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.shape[1] <= 1:
        # Not enough numeric columns for pairplot
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "Not enough numeric columns to create pairplot", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Limit the number of columns
    if numeric_data.shape[1] > max_cols:
        # Select columns with highest variance
        vars_to_include = numeric_data.var().nlargest(max_cols).index.tolist()
        numeric_data = numeric_data[vars_to_include]
    
    # Create pairplot
    if hue and hue in data.columns:
        g = sns.pairplot(data=data, vars=numeric_data.columns, hue=hue, 
                          diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        g = sns.pairplot(data=numeric_data, diag_kind='kde')
    
    # Set title
    g.fig.suptitle('Pairwise Relationships', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return g.fig


def plot_outliers(data: pd.DataFrame, max_cols: int = 20) -> plt.Figure:
    """
    Create boxplots to visualize outliers in numeric variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    max_cols : int, optional
        Maximum number of columns to plot, by default 20
        
    Returns
    -------
    plt.Figure
        Figure object with boxplots
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.shape[1] == 0:
        # No numeric columns
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No numeric columns to check for outliers", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        return plt.gcf()
    
    # Limit the number of columns
    if numeric_data.shape[1] > max_cols:
        numeric_data = numeric_data.iloc[:, :max_cols]
    
    # Create figure
    plt.figure(figsize=(min(12, numeric_data.shape[1] * 2), 6))
    
    # Create boxplot
    ax = sns.boxplot(data=numeric_data)
    
    # Set labels and title
    plt.title('Boxplots for Outlier Detection')
    plt.xlabel('Features')
    plt.ylabel('Values')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()


def save_figure_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object
        
    Returns
    -------
    str
        Base64-encoded string representation of the figure
    """
    # Create a bytes buffer for the image
    buf = io.BytesIO()
    
    # Save the figure to the buffer
    fig.savefig(buf, format='png', bbox_inches='tight')
    
    # Get the image bytes
    buf.seek(0)
    img_bytes = buf.getvalue()
    
    # Encode as base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64