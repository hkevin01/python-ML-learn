"""
=============================================================================
ml_core Package
=============================================================================

Part of the Python Machine Learning Study Guide.

Core ML utilities and helper functions for supervised, unsupervised,
and deep learning modules.
"""

from .supervised import (
    compare_models,
    create_baseline_models,
    evaluate_classification,
    evaluate_regression,
    get_feature_importance_df,
    plot_learning_curve,
    plot_roc_curves,
    plot_validation_curve,
)

__all__ = [
    # Supervised Learning
    "evaluate_classification",
    "evaluate_regression",
    "plot_learning_curve",
    "plot_validation_curve",
    "plot_roc_curves",
    "get_feature_importance_df",
    "compare_models",
    "create_baseline_models",
]
