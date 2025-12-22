"""
Statistics Helpers for Machine Learning
=======================================

Utility functions for common statistical operations in ML workflows.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def describe_distribution(
    data: Union[pd.Series, np.ndarray], name: Optional[str] = None
) -> Dict:
    """
    Get comprehensive descriptive statistics for a distribution.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Numeric data to analyze
    name : str, optional
        Name for the data

    Returns
    -------
    dict
        Dictionary containing all descriptive statistics
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    return {
        "name": name or data.name or "data",
        "count": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "var": data.var(),
        "min": data.min(),
        "max": data.max(),
        "range": data.max() - data.min(),
        "q1": data.quantile(0.25),
        "q3": data.quantile(0.75),
        "iqr": data.quantile(0.75) - data.quantile(0.25),
        "skewness": data.skew(),
        "kurtosis": data.kurtosis(),
        "cv": (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan,
    }


def check_normality(data: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> Dict:
    """
    Test if data follows a normal distribution.

    Uses Shapiro-Wilk test (best for n < 5000).

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Data to test
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Test results including statistic, p-value, and interpretation
    """
    if isinstance(data, pd.Series):
        data = data.dropna().values

    # Use sample if too large
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    statistic, p_value = stats.shapiro(data)

    return {
        "test": "Shapiro-Wilk",
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "is_normal": p_value > alpha,
        "interpretation": (
            "Normal distribution" if p_value > alpha else "Not normal distribution"
        ),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Two groups to compare

    Returns
    -------
    float
        Cohen's d value

    Notes
    -----
    Effect size interpretation:
    - |d| < 0.2: Small
    - 0.2 <= |d| < 0.8: Medium
    - |d| >= 0.8: Large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d value

    Returns
    -------
    str
        Interpretation of effect size
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_two_groups(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    alpha: float = 0.05,
) -> Dict:
    """
    Comprehensive comparison of two groups.

    Performs t-test and calculates effect size.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Two groups to compare
    group1_name, group2_name : str
        Names for the groups
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Comprehensive comparison results
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Effect size
    d = cohens_d(group1, group2)

    return {
        "group1_name": group1_name,
        "group2_name": group2_name,
        "group1_mean": np.mean(group1),
        "group2_mean": np.mean(group2),
        "group1_std": np.std(group1, ddof=1),
        "group2_std": np.std(group2, ddof=1),
        "group1_n": len(group1),
        "group2_n": len(group2),
        "mean_difference": np.mean(group1) - np.mean(group2),
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "cohens_d": d,
        "effect_size": interpret_effect_size(d),
    }


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    confidence : float, default 0.95
        Confidence level (0.90, 0.95, 0.99)

    Returns
    -------
    tuple
        (lower_bound, upper_bound, margin_of_error)
    """
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    # Use t-distribution for small samples
    if n < 30:
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_critical * se
    else:
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * se

    return (mean - margin, mean + margin, margin)


def correlation_with_pvalue(
    x: np.ndarray, y: np.ndarray, method: str = "pearson"
) -> Dict:
    """
    Calculate correlation with p-value and interpretation.

    Parameters
    ----------
    x, y : np.ndarray
        Two variables to correlate
    method : str, default 'pearson'
        Correlation method: 'pearson' or 'spearman'

    Returns
    -------
    dict
        Correlation results
    """
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    # Interpret strength
    r_abs = abs(r)
    if r_abs < 0.3:
        strength = "weak"
    elif r_abs < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if r > 0 else "negative"

    return {
        "method": method,
        "correlation": r,
        "p_value": p,
        "strength": strength,
        "direction": direction,
        "interpretation": f"{strength} {direction} correlation",
    }


def find_outliers_zscore(
    data: np.ndarray, threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find outliers using z-score method.

    Parameters
    ----------
    data : np.ndarray
        Data to check for outliers
    threshold : float, default 3.0
        Z-score threshold for outliers

    Returns
    -------
    tuple
        (outlier_indices, outlier_values)
    """
    z_scores = np.abs(stats.zscore(data))
    outlier_mask = z_scores > threshold

    return np.where(outlier_mask)[0], data[outlier_mask]


def find_outliers_iqr(
    data: np.ndarray, multiplier: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find outliers using IQR method.

    Parameters
    ----------
    data : np.ndarray
        Data to check for outliers
    multiplier : float, default 1.5
        IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)

    Returns
    -------
    tuple
        (outlier_indices, outlier_values)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    return np.where(outlier_mask)[0], data[outlier_mask]


def bootstrap_mean(
    data: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Bootstrap estimation of mean with confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    n_iterations : int, default 1000
        Number of bootstrap iterations
    confidence : float, default 0.95
        Confidence level
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Bootstrap results including CI
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    bootstrap_means = np.array(
        [np.mean(np.random.choice(data, n, replace=True)) for _ in range(n_iterations)]
    )

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return {
        "mean": np.mean(bootstrap_means),
        "std": np.std(bootstrap_means),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_iterations": n_iterations,
    }
