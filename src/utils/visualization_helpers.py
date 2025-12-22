"""
Visualization Helpers for Machine Learning
===========================================

Utility functions for creating common ML visualizations.
These functions encapsulate common plotting patterns to speed up EDA and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union


def plot_distribution(
    data: pd.Series,
    title: Optional[str] = None,
    kde: bool = True,
    bins: Union[int, str] = 'auto',
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot distribution of a numeric variable with optional statistics.
    
    Parameters
    ----------
    data : pd.Series
        Numeric data to plot
    title : str, optional
        Plot title (defaults to column name)
    kde : bool, default True
        Whether to overlay KDE curve
    bins : int or str, default 'auto'
        Number of histogram bins
    figsize : tuple, default (10, 6)
        Figure size if creating new figure
    show_stats : bool, default True
        Whether to show mean, median, std on plot
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    plt.Axes
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram with optional KDE
    sns.histplot(data.dropna(), kde=kde, bins=bins, ax=ax)
    
    # Title
    title = title or data.name or 'Distribution'
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    # Add statistics
    if show_stats:
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        # Add vertical lines
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Add text box with stats
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend()
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = 'Correlation Heatmap',
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    cmap: str = 'RdBu_r',
    mask_upper: bool = True,
    threshold: Optional[float] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a correlation heatmap with optional masking and threshold highlighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns
    title : str, default 'Correlation Heatmap'
        Plot title
    figsize : tuple, default (10, 8)
        Figure size
    annot : bool, default True
        Whether to show correlation values
    cmap : str, default 'RdBu_r'
        Colormap to use
    mask_upper : bool, default True
        Whether to mask upper triangle
    threshold : float, optional
        If provided, return pairs with |correlation| > threshold
        
    Returns
    -------
    tuple
        (fig, high_corr_df) - Figure and DataFrame of high correlations
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if mask_upper else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    # Find high correlation pairs if threshold provided
    high_corr_df = pd.DataFrame()
    if threshold is not None:
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr
                    })
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
                'Correlation', key=abs, ascending=False
            )
    
    return fig, high_corr_df


def plot_missing_values(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    threshold_line: float = 5.0
) -> plt.Figure:
    """
    Visualize missing values in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    figsize : tuple, default (12, 6)
        Figure size
    threshold_line : float, default 5.0
        Percentage threshold to draw reference line
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    
    # Only keep columns with missing values
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No missing values found!', ha='center', va='center',
                fontsize=14, color='green')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars by severity
    colors = ['green' if p < 5 else 'orange' if p < 20 else 'red' for p in missing_pct.values]
    
    # Create horizontal bar chart
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor='black')
    
    # Add threshold lines
    ax.axvline(x=threshold_line, color='orange', linestyle='--', label=f'{threshold_line}% threshold')
    ax.axvline(x=20, color='red', linestyle='--', label='20% threshold')
    
    # Add value labels
    for bar, pct in zip(bars, missing_pct.values):
        ax.text(pct + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Missing Percentage', fontsize=12)
    ax.set_title('Missing Values by Feature', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


def plot_class_balance(
    y: pd.Series,
    title: str = 'Class Distribution',
    figsize: Tuple[int, int] = (10, 5),
    show_ratio: bool = True
) -> Tuple[plt.Figure, float]:
    """
    Visualize class balance for classification targets.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    title : str, default 'Class Distribution'
        Plot title
    figsize : tuple, default (10, 5)
        Figure size
    show_ratio : bool, default True
        Whether to display imbalance ratio
        
    Returns
    -------
    tuple
        (fig, imbalance_ratio)
    """
    class_counts = y.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_counts)))
    bars = axes[0].bar(class_counts.index.astype(str), class_counts.values, color=colors)
    
    # Add percentage labels
    for bar, count in zip(bars, class_counts.values):
        pct = count / len(y) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(y)*0.01,
                     f'{pct:.1f}%', ha='center', fontweight='bold')
    
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(title, fontweight='bold', fontsize=14)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index.astype(str), 
                autopct='%1.1f%%', colors=colors, startangle=90,
                explode=[0.02] * len(class_counts))
    axes[1].set_title('Class Proportions', fontweight='bold', fontsize=14)
    
    if show_ratio:
        fig.suptitle(f'Imbalance Ratio: {imbalance_ratio:.2f}:1', fontsize=12, y=1.02)
    
    plt.tight_layout()
    return fig, imbalance_ratio


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    title: str = 'Feature Importance',
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Parameters
    ----------
    importance : np.ndarray
        Feature importance values
    feature_names : list
        Names of features
    title : str, default 'Feature Importance'
        Plot title
    top_n : int, optional
        Only show top N features
    figsize : tuple, default (10, 6)
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Select top N if specified
    if top_n is not None:
        importance_df = importance_df.tail(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create gradient colors
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(importance_df)))
    
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + importance_df['importance'].max() * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlim(0, importance_df['importance'].max() * 1.15)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Create a formatted confusion matrix visualization.
    
    Parameters
    ----------
    confusion : np.ndarray
        Confusion matrix (2D array)
    labels : list, optional
        Class labels
    normalize : bool, default False
        Whether to show percentages
    title : str, default 'Confusion Matrix'
        Plot title
    figsize : tuple, default (8, 6)
        Figure size
    cmap : str, default 'Blues'
        Colormap to use
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if labels is None:
        labels = [f'Class {i}' for i in range(len(confusion))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        confusion_plot = confusion / confusion.sum() * 100
        fmt = '.1f'
        cbar_label = 'Percentage'
    else:
        confusion_plot = confusion
        fmt = 'd'
        cbar_label = 'Count'
    
    # Create heatmap
    sns.heatmap(
        confusion_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': cbar_label},
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    return fig


def create_eda_dashboard(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive EDA dashboard for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    target_col : str, optional
        Target column for classification analysis
    numeric_cols : list, optional
        Specific numeric columns to analyze
    figsize : tuple, default (16, 12)
        Figure size
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Select numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distribution of first numeric column
    ax1 = fig.add_subplot(gs[0, 0])
    if len(numeric_cols) > 0:
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax1)
        ax1.set_title(f'Distribution: {numeric_cols[0]}', fontweight='bold')
    
    # 2. Distribution of second numeric column
    ax2 = fig.add_subplot(gs[0, 1])
    if len(numeric_cols) > 1:
        sns.histplot(df[numeric_cols[1]].dropna(), kde=True, ax=ax2)
        ax2.set_title(f'Distribution: {numeric_cols[1]}', fontweight='bold')
    
    # 3. Box plots
    ax3 = fig.add_subplot(gs[0, 2])
    if len(numeric_cols) >= 2:
        df[numeric_cols[:4]].boxplot(ax=ax3)
        ax3.set_title('Box Plots', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Correlation heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    numeric_df = df[numeric_cols].dropna()
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax4)
        ax4.set_title('Correlation Heatmap', fontweight='bold')
    
    # 5. Scatter plot of two most correlated features
    ax5 = fig.add_subplot(gs[1, 2])
    if len(numeric_cols) >= 2:
        ax5.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)
        ax5.set_xlabel(numeric_cols[0])
        ax5.set_ylabel(numeric_cols[1])
        ax5.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}', fontweight='bold')
    
    # 6. Target distribution (if provided)
    ax6 = fig.add_subplot(gs[2, 0])
    if target_col and target_col in df.columns:
        df[target_col].value_counts().plot.bar(ax=ax6, color='steelblue')
        ax6.set_title(f'Target: {target_col}', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'No target specified', ha='center', va='center')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
    
    # 7. Missing values
    ax7 = fig.add_subplot(gs[2, 1])
    missing_pct = (df.isnull().sum() / len(df) * 100)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    if len(missing_pct) > 0:
        missing_pct.plot.barh(ax=ax7, color='salmon')
        ax7.set_title('Missing Values (%)', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'No missing values!', ha='center', va='center', color='green')
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
    
    # 8. Data info
    ax8 = fig.add_subplot(gs[2, 2])
    info_text = f"""
    Rows: {len(df):,}
    Columns: {len(df.columns)}
    Numeric: {len(df.select_dtypes(include=[np.number]).columns)}
    Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}
    Missing: {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / df.size * 100:.1f}%)
    """
    ax8.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
             fontfamily='monospace')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Dataset Info', fontweight='bold')
    
    fig.suptitle('EDA Dashboard', fontsize=16, fontweight='bold', y=1.02)
    
    return fig


# Convenience functions for quick plots
def quick_hist(data: pd.Series, **kwargs) -> plt.Axes:
    """Quick histogram with sensible defaults."""
    return plot_distribution(data, **kwargs)


def quick_corr(df: pd.DataFrame, **kwargs) -> Tuple[plt.Figure, pd.DataFrame]:
    """Quick correlation heatmap."""
    return plot_correlation_heatmap(df, **kwargs)


def quick_missing(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick missing value visualization."""
    return plot_missing_values(df, **kwargs)
