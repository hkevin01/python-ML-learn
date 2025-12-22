"""
=============================================================================
ml_core Package
=============================================================================

Part of the Python Machine Learning Study Guide.

Core ML utilities and helper functions for supervised, unsupervised,
deep learning, and NLP modules.
"""

from .deep_learning import (
    EarlyStopping,
    TrainingHistory,
    accuracy,
    calculate_conv_output_size,
    calculate_pool_output_size,
    compute_class_weights,
    count_parameters,
    create_learning_rate_schedule,
    get_activation_function,
    get_layer_output_shapes,
)
from .nlp import (
    TextPreprocessor,
    build_vocabulary,
    clean_text,
    compute_idf,
    compute_tf,
    compute_tfidf,
    cosine_similarity,
    create_attention_mask,
    encode_texts,
    generate_ngrams,
    get_word_frequencies,
    mean_pooling,
    pad_sequences,
    simple_tokenize,
)
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
    # Deep Learning
    "calculate_conv_output_size",
    "calculate_pool_output_size",
    "count_parameters",
    "get_layer_output_shapes",
    "EarlyStopping",
    "TrainingHistory",
    "create_learning_rate_schedule",
    "compute_class_weights",
    "accuracy",
    "get_activation_function",
    # NLP
    "clean_text",
    "simple_tokenize",
    "build_vocabulary",
    "encode_texts",
    "compute_tf",
    "compute_idf",
    "compute_tfidf",
    "generate_ngrams",
    "cosine_similarity",
    "pad_sequences",
    "create_attention_mask",
    "TextPreprocessor",
    "get_word_frequencies",
    "mean_pooling",
]
