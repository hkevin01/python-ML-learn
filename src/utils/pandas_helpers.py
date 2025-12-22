"""
Pandas helper functions for data manipulation in ML workflows.

This module provides utility functions for common data preprocessing tasks
including loading, cleaning, and transforming DataFrames.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any


def load_csv_optimized(
    filepath: str,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a CSV file with optimized settings for ML workflows.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    usecols : list of str, optional
        Columns to load. If None, loads all columns.
    dtype : dict, optional
        Data types for columns.
    parse_dates : list of str, optional
        Columns to parse as dates.
    nrows : int, optional
        Number of rows to read. If None, reads all rows.
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    
    Examples
    --------
    >>> df = load_csv_optimized('data.csv', usecols=['id', 'value'])
    """
    return pd.read_csv(
        filepath,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        nrows=nrows,
        low_memory=False
    )


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about missing values in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: column, missing_count, missing_pct, dtype.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, 6]})
    >>> get_missing_info(df)
    """
    missing_count = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    
    result = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_count.values,
        'missing_pct': missing_pct.values,
        'dtype': df.dtypes.values
    })
    
    return result.sort_values('missing_count', ascending=False).reset_index(drop=True)


def fill_missing_by_group(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    method: str = 'mean'
) -> pd.Series:
    """
    Fill missing values using group-wise statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column with missing values to fill.
    group_by : str
        Column to group by.
    method : str, default 'mean'
        Aggregation method: 'mean', 'median', 'mode'.
    
    Returns
    -------
    pd.Series
        Series with filled values.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B'],
    ...     'value': [1, np.nan, 3, 4]
    ... })
    >>> fill_missing_by_group(df, 'value', 'category', 'mean')
    """
    if method == 'mean':
        return df.groupby(group_by)[column].transform(lambda x: x.fillna(x.mean()))
    elif method == 'median':
        return df.groupby(group_by)[column].transform(lambda x: x.fillna(x.median()))
    elif method == 'mode':
        return df.groupby(group_by)[column].transform(
            lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else x)
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', or 'mode'.")


def create_dummy_variables(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
    prefix_sep: str = '_'
) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Categorical columns to convert.
    drop_first : bool, default True
        Whether to drop the first category (avoid multicollinearity).
    prefix_sep : str, default '_'
        Separator between column name and category.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dummy variables.
    
    Examples
    --------
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
    >>> create_dummy_variables(df, ['color'])
    """
    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=drop_first,
        prefix_sep=prefix_sep
    )


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using the IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Numeric column to check for outliers.
    multiplier : float, default 1.5
        IQR multiplier. Use 3.0 for extreme outliers only.
    
    Returns
    -------
    pd.Series
        Boolean Series where True indicates an outlier.
    
    Examples
    --------
    >>> df = pd.DataFrame({'value': [1, 2, 3, 100]})
    >>> detect_outliers_iqr(df, 'value')
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    test_size : float, default 0.2
        Proportion of data for test set.
    random_state : int, optional
        Random seed for reproducibility.
    stratify_column : str, optional
        Column to stratify by (for classification).
    
    Returns
    -------
    tuple of pd.DataFrame
        (train_df, test_df)
    
    Examples
    --------
    >>> df = pd.DataFrame({'X': range(100), 'y': [0, 1] * 50})
    >>> train, test = split_train_test(df, test_size=0.2)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(df)
    
    if stratify_column is not None:
        # Stratified split
        train_indices = []
        test_indices = []
        
        for _, group in df.groupby(stratify_column):
            group_indices = group.index.tolist()
            np.random.shuffle(group_indices)
            split_point = int(len(group_indices) * (1 - test_size))
            train_indices.extend(group_indices[:split_point])
            test_indices.extend(group_indices[split_point:])
        
        return df.loc[train_indices].copy(), df.loc[test_indices].copy()
    else:
        # Random split
        indices = np.arange(n)
        np.random.shuffle(indices)
        split_point = int(n * (1 - test_size))
        
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        return df.iloc[train_indices].copy(), df.iloc[test_indices].copy()


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    verbose : bool, default True
        Whether to print memory usage information.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized dtypes.
    
    Examples
    --------
    >>> df = pd.DataFrame({'id': range(10000), 'value': np.random.randn(10000)})
    >>> df_optimized = reduce_memory_usage(df)
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    result = df.copy()
    
    for col in result.columns:
        col_type = result[col].dtype
        
        if col_type != object:
            c_min = result[col].min()
            c_max = result[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    result[col] = result[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    result[col] = result[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    result[col] = result[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    result[col] = result[col].astype(np.float32)
    
    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return result


def dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    dict
        Dictionary with shape, memory, dtypes, missing, and duplicates info.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> info = dataframe_info(df)
    """
    return {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_total': df.isna().sum().sum(),
        'missing_pct': (df.isna().sum().sum() / df.size * 100),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
