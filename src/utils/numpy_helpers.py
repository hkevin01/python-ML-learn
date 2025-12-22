"""
=============================================================================
NumPy Helper Utilities
=============================================================================

Utility functions to complement NumPy for Machine Learning workflows.

This module provides:
- Common array operations
- Normalization functions
- Array validation
- Performance utilities

Usage:
------
    from src.utils.numpy_helpers import normalize, standardize, check_nan
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# =============================================================================
# Normalization Functions
# =============================================================================


def normalize(
    arr: ArrayLike, method: str = "minmax", axis: Optional[int] = None
) -> NDArray:
    """
    Normalize array values using various methods.

    Parameters:
    -----------
    arr : ArrayLike
        Input array to normalize
    method : str, default="minmax"
        Normalization method:
        - "minmax": Scale to [0, 1] range
        - "zscore": Standardize to mean=0, std=1
        - "l1": L1 normalization (sum of abs values = 1)
        - "l2": L2 normalization (Euclidean norm = 1)
    axis : int, optional
        Axis along which to normalize. If None, normalize entire array.

    Returns:
    --------
    NDArray
        Normalized array with same shape as input

    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> normalize(arr, method="minmax")
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    >>> normalize(arr, method="zscore")
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])

    Raises:
    -------
    ValueError
        If unknown normalization method is specified
    """
    arr = np.asarray(arr, dtype=np.float64)

    if method == "minmax":
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)
        range_val = max_val - min_val
        # Handle case where all values are the same
        range_val = np.where(range_val == 0, 1, range_val)
        return (arr - min_val) / range_val

    elif method == "zscore":
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        # Handle case where std is 0
        std = np.where(std == 0, 1, std)
        return (arr - mean) / std

    elif method == "l1":
        l1_norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
        l1_norm = np.where(l1_norm == 0, 1, l1_norm)
        return arr / l1_norm

    elif method == "l2":
        l2_norm = np.sqrt(np.sum(arr**2, axis=axis, keepdims=True))
        l2_norm = np.where(l2_norm == 0, 1, l2_norm)
        return arr / l2_norm

    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Valid methods are: 'minmax', 'zscore', 'l1', 'l2'"
        )


def standardize(arr: ArrayLike, axis: Optional[int] = None) -> NDArray:
    """
    Standardize array to have mean=0 and std=1.

    Shorthand for normalize(arr, method="zscore").

    Parameters:
    -----------
    arr : ArrayLike
        Input array to standardize
    axis : int, optional
        Axis along which to standardize

    Returns:
    --------
    NDArray
        Standardized array
    """
    return normalize(arr, method="zscore", axis=axis)


# =============================================================================
# Array Validation
# =============================================================================


def check_nan(arr: ArrayLike) -> Tuple[bool, int]:
    """
    Check if array contains NaN values.

    Parameters:
    -----------
    arr : ArrayLike
        Array to check

    Returns:
    --------
    Tuple[bool, int]
        (has_nan, count_of_nans)

    Examples:
    ---------
    >>> arr = np.array([1, 2, np.nan, 4, np.nan])
    >>> check_nan(arr)
    (True, 2)
    """
    arr = np.asarray(arr)
    nan_mask = np.isnan(arr)
    count = int(np.sum(nan_mask))
    return count > 0, count


def check_inf(arr: ArrayLike) -> Tuple[bool, int]:
    """
    Check if array contains infinite values.

    Parameters:
    -----------
    arr : ArrayLike
        Array to check

    Returns:
    --------
    Tuple[bool, int]
        (has_inf, count_of_infs)
    """
    arr = np.asarray(arr)
    inf_mask = np.isinf(arr)
    count = int(np.sum(inf_mask))
    return count > 0, count


def array_info(arr: ArrayLike) -> dict:
    """
    Get comprehensive information about an array.

    Parameters:
    -----------
    arr : ArrayLike
        Array to analyze

    Returns:
    --------
    dict
        Dictionary containing array information:
        - shape, dtype, size, ndim
        - min, max, mean, std
        - nan_count, inf_count

    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> info = array_info(arr)
    >>> print(info['mean'])
    3.0
    """
    arr = np.asarray(arr)

    has_nan, nan_count = check_nan(arr)
    has_inf, inf_count = check_inf(arr)

    # Calculate stats only on finite values
    finite_arr = arr[np.isfinite(arr)]

    info = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "ndim": arr.ndim,
        "min": float(np.min(finite_arr)) if finite_arr.size > 0 else None,
        "max": float(np.max(finite_arr)) if finite_arr.size > 0 else None,
        "mean": float(np.mean(finite_arr)) if finite_arr.size > 0 else None,
        "std": float(np.std(finite_arr)) if finite_arr.size > 0 else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

    return info


# =============================================================================
# Array Operations
# =============================================================================


def safe_divide(
    numerator: ArrayLike, denominator: ArrayLike, fill_value: float = 0.0
) -> NDArray:
    """
    Safely divide arrays, replacing division by zero with fill_value.

    Parameters:
    -----------
    numerator : ArrayLike
        Numerator array
    denominator : ArrayLike
        Denominator array
    fill_value : float, default=0.0
        Value to use when denominator is zero

    Returns:
    --------
    NDArray
        Result of division with zeros replaced

    Examples:
    ---------
    >>> a = np.array([10, 20, 30])
    >>> b = np.array([2, 0, 5])
    >>> safe_divide(a, b)
    array([5., 0., 6.])
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    # Use numpy's divide with 'where' parameter to avoid division by zero warning
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill_value)
    return result


def clip_outliers(
    arr: ArrayLike, lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> NDArray:
    """
    Clip outliers to percentile boundaries.

    Parameters:
    -----------
    arr : ArrayLike
        Input array
    lower_percentile : float, default=1.0
        Lower percentile boundary (0-100)
    upper_percentile : float, default=99.0
        Upper percentile boundary (0-100)

    Returns:
    --------
    NDArray
        Array with outliers clipped

    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 100, 4, 5])  # 100 is outlier
    >>> clip_outliers(arr)
    array([1., 2., 3., 5., 4., 5.])  # 100 clipped to 99th percentile
    """
    arr = np.asarray(arr, dtype=np.float64)
    lower = np.percentile(arr, lower_percentile)
    upper = np.percentile(arr, upper_percentile)
    return np.clip(arr, lower, upper)


def moving_average(arr: ArrayLike, window_size: int, mode: str = "valid") -> NDArray:
    """
    Calculate moving average of 1D array.

    Parameters:
    -----------
    arr : ArrayLike
        1D input array
    window_size : int
        Size of the moving window
    mode : str, default="valid"
        Convolution mode: "valid", "same", or "full"

    Returns:
    --------
    NDArray
        Moving average values

    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> moving_average(arr, 3)
    array([2., 3., 4., 5., 6., 7., 8., 9.])
    """
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode=mode)


# =============================================================================
# Random Utilities
# =============================================================================


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters:
    -----------
    seed : int, default=42
        Random seed value
    """
    np.random.seed(seed)


def train_test_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Generate train/test split indices.

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    test_size : float, default=0.2
        Proportion of samples for test set
    shuffle : bool, default=True
        Whether to shuffle before splitting
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Tuple[NDArray, NDArray]
        (train_indices, test_indices)

    Examples:
    ---------
    >>> train_idx, test_idx = train_test_split_indices(100, test_size=0.2, seed=42)
    >>> len(train_idx), len(test_idx)
    (80, 20)
    """
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_point = int(n_samples * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    return train_indices, test_indices


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Normalization
    "normalize",
    "standardize",
    # Validation
    "check_nan",
    "check_inf",
    "array_info",
    # Operations
    "safe_divide",
    "clip_outliers",
    "moving_average",
    # Random
    "set_seed",
    "train_test_split_indices",
]
