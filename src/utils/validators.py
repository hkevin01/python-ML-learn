"""
=============================================================================
Data Validation Utility Module
=============================================================================

Provides comprehensive data validation utilities for ML applications.
Ensures data quality and correctness before processing.

Features:
---------
- NumPy array validation
- Pandas DataFrame validation
- Type checking and coercion
- Range and boundary validation
- NaN/Inf detection and handling
- Shape validation

Usage Examples:
---------------
    from src.utils.validators import validate_array, validate_dataframe

    # Validate numpy array
    X = validate_array(X, name="features", allow_nan=False)

    # Validate DataFrame
    df = validate_dataframe(df, required_columns=['feature1', 'target'])

Author: ML Study Guide
Version: 0.1.0
"""

import logging
from typing import (
    Any, Optional, Union, List, Tuple, Dict,
    Sequence, Type, Callable
)
import warnings

# Lazy imports for numpy and pandas (may not always be installed)
np = None
pd = None


def _ensure_numpy():
    """Lazy load numpy."""
    global np
    if np is None:
        import numpy
        np = numpy
    return np


def _ensure_pandas():
    """Lazy load pandas."""
    global pd
    if pd is None:
        import pandas
        pd = pandas
    return pd


# =============================================================================
# Array Validation
# =============================================================================

def validate_array(
    array: Any,
    name: str = "array",
    dtype: Optional[Type] = None,
    ndim: Optional[int] = None,
    shape: Optional[Tuple[Optional[int], ...]] = None,
    min_samples: Optional[int] = None,
    min_features: Optional[int] = None,
    allow_nan: bool = True,
    allow_inf: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    copy: bool = False,
    ensure_2d: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1
) -> Any:
    """
    Validate and optionally convert a numpy array.

    This function performs comprehensive validation on numpy arrays
    to ensure they meet the requirements for ML operations.

    Parameters:
    -----------
    array : array-like
        Input array to validate.
    name : str
        Name of the array (for error messages).
    dtype : type, optional
        Required dtype. If set, will attempt conversion.
    ndim : int, optional
        Required number of dimensions.
    shape : tuple, optional
        Required shape. Use None for dimensions that can vary.
        Example: (None, 10) means any number of rows, exactly 10 columns.
    min_samples : int, optional
        Minimum number of samples (rows).
    min_features : int, optional
        Minimum number of features (columns).
    allow_nan : bool
        Whether to allow NaN values. Default True.
    allow_inf : bool
        Whether to allow infinite values. Default True.
    min_value : float, optional
        Minimum allowed value.
    max_value : float, optional
        Maximum allowed value.
    copy : bool
        Whether to return a copy. Default False.
    ensure_2d : bool
        Whether to ensure the array is 2D. Default False.
    ensure_min_samples : int
        Minimum samples after validation. Default 1.
    ensure_min_features : int
        Minimum features after validation. Default 1.

    Returns:
    --------
    np.ndarray
        Validated (and possibly converted) array.

    Raises:
    -------
    ValueError
        If validation fails.
    TypeError
        If input cannot be converted to array.

    Examples:
    ---------
    >>> X = validate_array(X, name="features", allow_nan=False)
    >>> y = validate_array(y, name="target", ndim=1)
    >>> X = validate_array(X, shape=(None, 10), dtype=np.float32)
    """
    np = _ensure_numpy()

    # -------------------------------------------------------------------------
    # Convert to numpy array
    # -------------------------------------------------------------------------
    try:
        if copy:
            array = np.array(array, dtype=dtype)
        else:
            array = np.asarray(array, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"'{name}' cannot be converted to numpy array: {e}"
        )

    # -------------------------------------------------------------------------
    # Check if array is empty
    # -------------------------------------------------------------------------
    if array.size == 0:
        raise ValueError(f"'{name}' is empty")

    # -------------------------------------------------------------------------
    # Ensure 2D if requested
    # -------------------------------------------------------------------------
    if ensure_2d:
        if array.ndim == 1:
            # Reshape 1D to 2D (column vector)
            array = array.reshape(-1, 1)
            logging.debug(f"'{name}' reshaped from 1D to 2D: {array.shape}")
        elif array.ndim != 2:
            raise ValueError(
                f"'{name}' must be 2D, got {array.ndim}D"
            )

    # -------------------------------------------------------------------------
    # Validate dimensions
    # -------------------------------------------------------------------------
    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"'{name}' must have {ndim} dimensions, got {array.ndim}"
        )

    # -------------------------------------------------------------------------
    # Validate shape
    # -------------------------------------------------------------------------
    if shape is not None:
        if len(shape) != array.ndim:
            raise ValueError(
                f"'{name}' shape mismatch: expected {len(shape)} dims, "
                f"got {array.ndim}"
            )
        for i, (expected, actual) in enumerate(zip(shape, array.shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"'{name}' dimension {i} must be {expected}, got {actual}"
                )

    # -------------------------------------------------------------------------
    # Validate minimum samples/features
    # -------------------------------------------------------------------------
    n_samples = array.shape[0]
    n_features = array.shape[1] if array.ndim > 1 else 1

    if min_samples is not None and n_samples < min_samples:
        raise ValueError(
            f"'{name}' must have at least {min_samples} samples, "
            f"got {n_samples}"
        )

    if min_features is not None and n_features < min_features:
        raise ValueError(
            f"'{name}' must have at least {min_features} features, "
            f"got {n_features}"
        )

    if n_samples < ensure_min_samples:
        raise ValueError(
            f"'{name}' must have at least {ensure_min_samples} sample(s), "
            f"got {n_samples}"
        )

    if n_features < ensure_min_features:
        raise ValueError(
            f"'{name}' must have at least {ensure_min_features} feature(s), "
            f"got {n_features}"
        )

    # -------------------------------------------------------------------------
    # Check for NaN values
    # -------------------------------------------------------------------------
    if not allow_nan:
        if np.issubdtype(array.dtype, np.floating):
            nan_count = np.isnan(array).sum()
            if nan_count > 0:
                raise ValueError(
                    f"'{name}' contains {nan_count} NaN value(s)"
                )

    # -------------------------------------------------------------------------
    # Check for infinite values
    # -------------------------------------------------------------------------
    if not allow_inf:
        if np.issubdtype(array.dtype, np.floating):
            inf_count = np.isinf(array).sum()
            if inf_count > 0:
                raise ValueError(
                    f"'{name}' contains {inf_count} infinite value(s)"
                )

    # -------------------------------------------------------------------------
    # Check value range
    # -------------------------------------------------------------------------
    if min_value is not None:
        if np.any(array < min_value):
            actual_min = np.nanmin(array)
            raise ValueError(
                f"'{name}' values must be >= {min_value}, "
                f"found minimum {actual_min}"
            )

    if max_value is not None:
        if np.any(array > max_value):
            actual_max = np.nanmax(array)
            raise ValueError(
                f"'{name}' values must be <= {max_value}, "
                f"found maximum {actual_max}"
            )

    return array


def validate_X_y(
    X: Any,
    y: Any,
    allow_nan: bool = False,
    multi_output: bool = False,
    y_numeric: bool = True
) -> Tuple[Any, Any]:
    """
    Validate feature matrix X and target vector y.

    Common validation for supervised learning problems.

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector or matrix.
    allow_nan : bool
        Whether to allow NaN values. Default False.
    multi_output : bool
        Whether y can be 2D (multiple outputs). Default False.
    y_numeric : bool
        Whether y must be numeric. Default True.

    Returns:
    --------
    tuple of (X, y)
        Validated arrays.

    Raises:
    -------
    ValueError
        If validation fails.

    Examples:
    ---------
    >>> X, y = validate_X_y(X_train, y_train)
    >>> X, y = validate_X_y(X, y, allow_nan=False, y_numeric=True)
    """
    np = _ensure_numpy()

    # Validate X
    X = validate_array(
        X,
        name="X",
        ensure_2d=True,
        allow_nan=allow_nan,
        allow_inf=False
    )

    # Validate y
    y_ndim = 1 if not multi_output else None
    y = validate_array(
        y,
        name="y",
        ndim=y_ndim,
        allow_nan=allow_nan,
        allow_inf=False
    )

    # Check that X and y have consistent lengths
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent number of samples: "
            f"X has {X.shape[0]}, y has {y.shape[0]}"
        )

    # Check y is numeric if required
    if y_numeric and not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            f"y must be numeric, got dtype {y.dtype}"
        )

    return X, y


# =============================================================================
# DataFrame Validation
# =============================================================================

def validate_dataframe(
    df: Any,
    name: str = "DataFrame",
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, Type]] = None,
    min_rows: int = 1,
    max_rows: Optional[int] = None,
    allow_duplicates: bool = True,
    allow_nan: bool = True,
    drop_nan: bool = False,
    copy: bool = False
) -> Any:
    """
    Validate a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Name for error messages.
    required_columns : list of str, optional
        Columns that must be present.
    optional_columns : list of str, optional
        Columns that may be present (others will be dropped if specified).
    column_types : dict, optional
        Expected types for columns {column_name: type}.
    min_rows : int
        Minimum number of rows. Default 1.
    max_rows : int, optional
        Maximum number of rows.
    allow_duplicates : bool
        Whether to allow duplicate rows. Default True.
    allow_nan : bool
        Whether to allow NaN values. Default True.
    drop_nan : bool
        Whether to drop rows with NaN. Default False.
    copy : bool
        Whether to return a copy. Default False.

    Returns:
    --------
    pd.DataFrame
        Validated DataFrame.

    Raises:
    -------
    ValueError
        If validation fails.
    TypeError
        If input is not a DataFrame.

    Examples:
    ---------
    >>> df = validate_dataframe(
    ...     df,
    ...     required_columns=['feature1', 'feature2', 'target'],
    ...     column_types={'target': int}
    ... )
    """
    pd = _ensure_pandas()

    # -------------------------------------------------------------------------
    # Check type
    # -------------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"'{name}' must be a pandas DataFrame, got {type(df).__name__}"
        )

    # Make copy if requested
    if copy:
        df = df.copy()

    # -------------------------------------------------------------------------
    # Check for empty DataFrame
    # -------------------------------------------------------------------------
    if len(df) == 0:
        raise ValueError(f"'{name}' is empty")

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------
    if required_columns is not None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"'{name}' is missing required columns: {missing}"
            )

    # -------------------------------------------------------------------------
    # Filter to allowed columns if optional_columns specified
    # -------------------------------------------------------------------------
    if optional_columns is not None:
        all_allowed = set(required_columns or []) | set(optional_columns)
        extra_cols = set(df.columns) - all_allowed
        if extra_cols:
            logging.debug(f"Dropping extra columns: {extra_cols}")
            df = df[[c for c in df.columns if c in all_allowed]]

    # -------------------------------------------------------------------------
    # Check column types
    # -------------------------------------------------------------------------
    if column_types is not None:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_dtype = df[col].dtype

                # Handle numeric types
                if expected_type in (int, float):
                    if not pd.api.types.is_numeric_dtype(actual_dtype):
                        raise ValueError(
                            f"Column '{col}' must be numeric, "
                            f"got {actual_dtype}"
                        )
                # Handle string types
                elif expected_type == str:
                    if not pd.api.types.is_string_dtype(actual_dtype):
                        # Try to convert
                        try:
                            df[col] = df[col].astype(str)
                        except Exception as e:
                            raise ValueError(
                                f"Column '{col}' cannot be converted to string: {e}"
                            )

    # -------------------------------------------------------------------------
    # Check row count
    # -------------------------------------------------------------------------
    if len(df) < min_rows:
        raise ValueError(
            f"'{name}' must have at least {min_rows} rows, got {len(df)}"
        )

    if max_rows is not None and len(df) > max_rows:
        raise ValueError(
            f"'{name}' must have at most {max_rows} rows, got {len(df)}"
        )

    # -------------------------------------------------------------------------
    # Handle NaN values
    # -------------------------------------------------------------------------
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        if not allow_nan and not drop_nan:
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(
                f"'{name}' contains NaN values in columns: {nan_cols}"
            )

        if drop_nan:
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                logging.info(
                    f"Dropped {dropped} rows with NaN values from '{name}'"
                )

    # -------------------------------------------------------------------------
    # Handle duplicates
    # -------------------------------------------------------------------------
    if not allow_duplicates:
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            raise ValueError(
                f"'{name}' contains {n_duplicates} duplicate rows"
            )

    return df


# =============================================================================
# Type Validation Utilities
# =============================================================================

def check_type(
    value: Any,
    expected_types: Union[Type, Tuple[Type, ...]],
    name: str = "value",
    allow_none: bool = False
) -> Any:
    """
    Check if a value is of expected type(s).

    Parameters:
    -----------
    value : Any
        Value to check.
    expected_types : type or tuple of types
        Expected type(s).
    name : str
        Name for error messages.
    allow_none : bool
        Whether None is allowed. Default False.

    Returns:
    --------
    Any
        The original value if valid.

    Raises:
    -------
    TypeError
        If type doesn't match.

    Examples:
    ---------
    >>> epochs = check_type(epochs, int, "epochs")
    >>> lr = check_type(lr, (int, float), "learning_rate")
    """
    if value is None:
        if allow_none:
            return value
        else:
            raise TypeError(f"'{name}' cannot be None")

    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = " or ".join(t.__name__ for t in expected_types)
        else:
            type_names = expected_types.__name__

        raise TypeError(
            f"'{name}' must be {type_names}, "
            f"got {type(value).__name__}"
        )

    return value


def check_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: str = "value",
    inclusive: Tuple[bool, bool] = (True, True)
) -> Union[int, float]:
    """
    Check if a value is within a specified range.

    Parameters:
    -----------
    value : int or float
        Value to check.
    min_value : number, optional
        Minimum value.
    max_value : number, optional
        Maximum value.
    name : str
        Name for error messages.
    inclusive : tuple of (bool, bool)
        Whether min and max are inclusive. Default (True, True).

    Returns:
    --------
    number
        The original value if valid.

    Raises:
    -------
    ValueError
        If value is out of range.

    Examples:
    ---------
    >>> lr = check_range(lr, min_value=0, max_value=1, name="learning_rate")
    >>> epochs = check_range(epochs, min_value=1, name="epochs")
    """
    min_inc, max_inc = inclusive

    if min_value is not None:
        if min_inc:
            if value < min_value:
                raise ValueError(
                    f"'{name}' must be >= {min_value}, got {value}"
                )
        else:
            if value <= min_value:
                raise ValueError(
                    f"'{name}' must be > {min_value}, got {value}"
                )

    if max_value is not None:
        if max_inc:
            if value > max_value:
                raise ValueError(
                    f"'{name}' must be <= {max_value}, got {value}"
                )
        else:
            if value >= max_value:
                raise ValueError(
                    f"'{name}' must be < {max_value}, got {value}"
                )

    return value


def check_choice(
    value: Any,
    choices: Sequence[Any],
    name: str = "value"
) -> Any:
    """
    Check if a value is one of the allowed choices.

    Parameters:
    -----------
    value : Any
        Value to check.
    choices : sequence
        Allowed values.
    name : str
        Name for error messages.

    Returns:
    --------
    Any
        The original value if valid.

    Raises:
    -------
    ValueError
        If value is not in choices.

    Examples:
    ---------
    >>> optimizer = check_choice(optimizer, ['sgd', 'adam', 'rmsprop'])
    >>> metric = check_choice(metric, ['accuracy', 'f1', 'auc'])
    """
    if value not in choices:
        raise ValueError(
            f"'{name}' must be one of {list(choices)}, got '{value}'"
        )
    return value


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Validators Module Demonstration")
    print("=" * 60)

    # Test 1: Array validation
    print("\n1. Array Validation:")
    X = np.random.randn(100, 10)
    X_valid = validate_array(X, name="features", allow_nan=False)
    print(f"   Valid array shape: {X_valid.shape}")

    # Test 2: X, y validation
    print("\n2. X, y Validation:")
    y = np.random.randint(0, 2, 100)
    X_valid, y_valid = validate_X_y(X, y)
    print(f"   X: {X_valid.shape}, y: {y_valid.shape}")

    # Test 3: DataFrame validation
    print("\n3. DataFrame Validation:")
    df = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'target': np.random.randint(0, 2, 50)
    })
    df_valid = validate_dataframe(
        df,
        required_columns=['feature1', 'target']
    )
    print(f"   Valid DataFrame: {df_valid.shape}")

    # Test 4: Type checking
    print("\n4. Type Checking:")
    epochs = check_type(10, int, "epochs")
    lr = check_type(0.001, (int, float), "learning_rate")
    print(f"   epochs: {epochs}, lr: {lr}")

    # Test 5: Range checking
    print("\n5. Range Checking:")
    lr = check_range(0.001, min_value=0, max_value=1, name="lr")
    print(f"   lr in valid range: {lr}")

    # Test 6: Choice checking
    print("\n6. Choice Checking:")
    optimizer = check_choice('adam', ['sgd', 'adam', 'rmsprop'], "optimizer")
    print(f"   Valid optimizer: {optimizer}")

    print("\n" + "=" * 60)
    print("Validators module demonstration complete!")
    print("=" * 60)
