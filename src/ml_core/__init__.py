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
from .unsupervised import (
    assign_cluster_to_new_data,
    compare_clustering_algorithms,
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    evaluate_clustering,
    find_optimal_clusters,
    find_optimal_dbscan_params,
    get_cluster_summary,
    get_pca_loadings,
    get_pca_variance_analysis,
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
    # Unsupervised Learning
    "find_optimal_clusters",
    "evaluate_clustering",
    "compare_clustering_algorithms",
    "find_optimal_dbscan_params",
    "get_pca_variance_analysis",
    "get_pca_loadings",
    "detect_anomalies_zscore",
    "detect_anomalies_iqr",
    "get_cluster_summary",
    "assign_cluster_to_new_data",
]
