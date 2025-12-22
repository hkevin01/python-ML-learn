"""
=============================================================================
Utilities Package
=============================================================================

Core utility modules for the ML study guide.

Modules:
--------
- timer: Performance timing and measurement utilities
- logger: Logging configuration and helpers
- error_handling: Custom exceptions and error handling
- memory: Memory monitoring and management
- validators: Data validation utilities
- numpy_helpers: NumPy utility functions for ML

Usage:
------
    from src.utils import Timer, setup_logging
    from src.utils import safe_execute, validate_array
    from src.utils import normalize, standardize
"""

from src.utils.error_handling import (
    DataValidationError,
    MLError,
    ModelNotFittedError,
    safe_execute,
)
from src.utils.logger import get_logger, setup_logging
from src.utils.memory import MemoryMonitor, check_memory
from src.utils.numpy_helpers import (
    array_info,
    check_inf,
    check_nan,
    clip_outliers,
    moving_average,
    normalize,
    safe_divide,
    set_seed,
    standardize,
    train_test_split_indices,
)
from src.utils.pandas_helpers import (
    create_dummy_variables,
    dataframe_info,
    detect_outliers_iqr,
    fill_missing_by_group,
    get_missing_info,
    load_csv_optimized,
    reduce_memory_usage,
    split_train_test,
)
from src.utils.timer import Timer, time_function
from src.utils.validators import validate_array, validate_dataframe

__all__ = [
    # Timer
    "Timer",
    "time_function",
    # Logger
    "setup_logging",
    "get_logger",
    # Error handling
    "MLError",
    "DataValidationError",
    "ModelNotFittedError",
    "safe_execute",
    # Memory
    "MemoryMonitor",
    "check_memory",
    # Validators
    "validate_array",
    "validate_dataframe",
    # NumPy helpers
    "normalize",
    "standardize",
    "check_nan",
    "check_inf",
    "array_info",
    "safe_divide",
    "clip_outliers",
    "moving_average",
    "set_seed",
    "train_test_split_indices",
    # Pandas helpers
    "load_csv_optimized",
    "get_missing_info",
    "fill_missing_by_group",
    "create_dummy_variables",
    "detect_outliers_iqr",
    "split_train_test",
    "reduce_memory_usage",
    "dataframe_info",
]

# Scikit-Learn helpers
from .sklearn_helpers import (
    compare_classifiers,
    create_preprocessing_pipeline,
    detect_data_leakage,
    evaluate_classifier,
    evaluate_regressor,
    get_feature_importance,
    learning_curve_data,
    model_summary,
    quick_grid_search,
    quick_train_test_split,
)

# Statistics helpers
from .stats_helpers import (
    bootstrap_mean,
    calculate_confidence_interval,
    check_normality,
    cohens_d,
    compare_two_groups,
    correlation_with_pvalue,
    describe_distribution,
    find_outliers_iqr,
    find_outliers_zscore,
    interpret_effect_size,
)

# Visualization helpers
from .visualization_helpers import (
    create_eda_dashboard,
    plot_class_balance,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_missing_values,
    quick_corr,
    quick_hist,
    quick_missing,
)
