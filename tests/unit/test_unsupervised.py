"""
Unit tests for ml_core.unsupervised module.

Tests clustering, dimensionality reduction, and anomaly detection helpers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '/home/kevin/Projects/python-ML-learn/src')

from ml_core.unsupervised import (
    find_optimal_clusters,
    evaluate_clustering,
    compare_clustering_algorithms,
    find_optimal_dbscan_params,
    get_pca_variance_analysis,
    get_pca_loadings,
    detect_anomalies_zscore,
    detect_anomalies_iqr,
    get_cluster_summary,
    assign_cluster_to_new_data
)


class TestFindOptimalClusters:
    """Tests for find_optimal_clusters function."""
    
    @pytest.fixture
    def blob_data(self):
        """Generate blob data with 4 known clusters."""
        X, _ = make_blobs(n_samples=200, centers=4, random_state=42, cluster_std=0.5)
        return X
    
    def test_elbow_method_returns_dict(self, blob_data):
        """Test that elbow method returns expected dictionary keys."""
        result = find_optimal_clusters(blob_data, k_range=(2, 8), method='elbow')
        
        assert 'k_values' in result
        assert 'inertias' in result
        assert 'optimal_k' in result
        assert len(result['k_values']) == len(result['inertias'])
    
    def test_silhouette_method_returns_dict(self, blob_data):
        """Test that silhouette method returns expected dictionary keys."""
        result = find_optimal_clusters(blob_data, k_range=(2, 8), method='silhouette')
        
        assert 'k_values' in result
        assert 'silhouette_scores' in result
        assert 'optimal_k' in result
        assert len(result['k_values']) == len(result['silhouette_scores'])
    
    def test_both_method_returns_all_keys(self, blob_data):
        """Test that 'both' method returns all expected keys."""
        result = find_optimal_clusters(blob_data, k_range=(2, 6), method='both')
        
        assert 'k_values' in result
        assert 'inertias' in result
        assert 'silhouette_scores' in result
        assert 'optimal_k' in result
    
    def test_optimal_k_in_valid_range(self, blob_data):
        """Test that optimal k is within the specified range."""
        result = find_optimal_clusters(blob_data, k_range=(2, 8), method='silhouette')
        
        assert 2 <= result['optimal_k'] < 8
    
    def test_silhouette_finds_correct_clusters(self, blob_data):
        """Test that silhouette method finds approximately correct number of clusters."""
        result = find_optimal_clusters(blob_data, k_range=(2, 8), method='silhouette')
        
        # Should find close to 4 clusters (the true number)
        assert 3 <= result['optimal_k'] <= 5


class TestEvaluateClustering:
    """Tests for evaluate_clustering function."""
    
    @pytest.fixture
    def clustered_data(self):
        """Generate data with known clusters."""
        X, labels = make_blobs(n_samples=200, centers=3, random_state=42)
        return X, labels
    
    def test_returns_all_metrics(self, clustered_data):
        """Test that all expected metrics are returned."""
        X, labels = clustered_data
        metrics = evaluate_clustering(X, labels)
        
        assert 'silhouette' in metrics
        assert 'davies_bouldin' in metrics
        assert 'calinski_harabasz' in metrics
        assert 'n_clusters' in metrics
    
    def test_silhouette_in_valid_range(self, clustered_data):
        """Test that silhouette score is in valid range [-1, 1]."""
        X, labels = clustered_data
        metrics = evaluate_clustering(X, labels)
        
        assert -1 <= metrics['silhouette'] <= 1
    
    def test_correct_n_clusters(self, clustered_data):
        """Test that correct number of clusters is returned."""
        X, labels = clustered_data
        metrics = evaluate_clustering(X, labels)
        
        assert metrics['n_clusters'] == 3
    
    def test_handles_noise_labels(self):
        """Test that noise labels (-1) are handled correctly."""
        X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
        labels = np.array([0] * 40 + [1] * 40 + [-1] * 20)  # 20 noise points
        
        metrics = evaluate_clustering(X, labels)
        
        assert metrics['n_clusters'] == 2
    
    def test_handles_single_cluster(self):
        """Test that single cluster returns NaN for metrics."""
        X = np.random.randn(100, 2)
        labels = np.zeros(100, dtype=int)
        
        metrics = evaluate_clustering(X, labels)
        
        assert np.isnan(metrics['silhouette'])


class TestCompareClusteringAlgorithms:
    """Tests for compare_clustering_algorithms function."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple blob data."""
        X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
        return X
    
    def test_returns_dataframe(self, simple_data):
        """Test that function returns a DataFrame."""
        result = compare_clustering_algorithms(simple_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_default_algorithms_included(self, simple_data):
        """Test that default algorithms are included."""
        result = compare_clustering_algorithms(simple_data)
        
        assert len(result) >= 3  # At least 3 default algorithms
        assert 'Algorithm' in result.columns
    
    def test_custom_algorithms(self, simple_data):
        """Test that custom algorithms can be provided."""
        from sklearn.cluster import KMeans
        
        custom = {
            'MyKMeans': KMeans(n_clusters=3, random_state=42, n_init=10)
        }
        result = compare_clustering_algorithms(simple_data, algorithms=custom)
        
        assert len(result) == 1
        assert result['Algorithm'].iloc[0] == 'MyKMeans'


class TestFindOptimalDBSCANParams:
    """Tests for find_optimal_dbscan_params function."""
    
    @pytest.fixture
    def moon_data(self):
        """Generate moon-shaped data suitable for DBSCAN."""
        X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
        return X
    
    def test_returns_expected_keys(self, moon_data):
        """Test that result contains expected keys."""
        result = find_optimal_dbscan_params(moon_data, n_eps=5)
        
        assert 'best_eps' in result
        assert 'best_min_samples' in result
        assert 'best_silhouette' in result
        assert 'results_df' in result
    
    def test_results_df_is_dataframe(self, moon_data):
        """Test that results_df is a DataFrame."""
        result = find_optimal_dbscan_params(moon_data, n_eps=5)
        
        assert isinstance(result['results_df'], pd.DataFrame)
    
    def test_finds_reasonable_params(self, moon_data):
        """Test that reasonable parameters are found."""
        result = find_optimal_dbscan_params(moon_data, eps_range=(0.1, 0.5), n_eps=5)
        
        # Should find some reasonable eps value
        assert result['best_eps'] is not None or result['best_silhouette'] == -1


class TestPCAVarianceAnalysis:
    """Tests for get_pca_variance_analysis function."""
    
    @pytest.fixture
    def high_dim_data(self):
        """Generate high-dimensional data."""
        np.random.seed(42)
        return np.random.randn(100, 20)
    
    def test_returns_expected_keys(self, high_dim_data):
        """Test that all expected keys are returned."""
        result = get_pca_variance_analysis(high_dim_data)
        
        assert 'explained_variance_ratio' in result
        assert 'cumulative_variance' in result
        assert 'n_components_95' in result
        assert 'n_components_99' in result
        assert 'pca' in result
    
    def test_variance_sums_to_one(self, high_dim_data):
        """Test that explained variance ratios sum to approximately 1."""
        result = get_pca_variance_analysis(high_dim_data)
        
        assert np.isclose(sum(result['explained_variance_ratio']), 1.0, atol=0.01)
    
    def test_cumulative_variance_increases(self, high_dim_data):
        """Test that cumulative variance is monotonically increasing."""
        result = get_pca_variance_analysis(high_dim_data)
        
        cumulative = result['cumulative_variance']
        assert all(cumulative[i] <= cumulative[i+1] for i in range(len(cumulative)-1))
    
    def test_n_components_95_less_than_99(self, high_dim_data):
        """Test that n_components for 95% is less than or equal to 99%."""
        result = get_pca_variance_analysis(high_dim_data)
        
        assert result['n_components_95'] <= result['n_components_99']


class TestPCALoadings:
    """Tests for get_pca_loadings function."""
    
    @pytest.fixture
    def fitted_pca(self):
        """Create a fitted PCA object."""
        X = np.random.randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        return pca
    
    def test_returns_dataframe(self, fitted_pca):
        """Test that function returns a DataFrame."""
        loadings = get_pca_loadings(fitted_pca)
        
        assert isinstance(loadings, pd.DataFrame)
    
    def test_correct_dimensions(self, fitted_pca):
        """Test that DataFrame has correct dimensions."""
        loadings = get_pca_loadings(fitted_pca, n_components=3)
        
        assert loadings.shape == (5, 3)  # 5 features, 3 components
    
    def test_custom_feature_names(self, fitted_pca):
        """Test that custom feature names are used."""
        feature_names = ['A', 'B', 'C', 'D', 'E']
        loadings = get_pca_loadings(fitted_pca, feature_names=feature_names)
        
        assert list(loadings.index) == feature_names
    
    def test_component_column_names(self, fitted_pca):
        """Test that component columns are named correctly."""
        loadings = get_pca_loadings(fitted_pca, n_components=2)
        
        assert list(loadings.columns) == ['PC1', 'PC2']


class TestAnomalyDetection:
    """Tests for anomaly detection functions."""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Generate data with known outliers."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[0] = [10, 10]  # Obvious outlier
        X[1] = [-10, -10]  # Another outlier
        return X
    
    def test_zscore_detects_outliers(self, data_with_outliers):
        """Test that Z-score method detects obvious outliers."""
        anomalies = detect_anomalies_zscore(data_with_outliers)
        
        assert anomalies[0] == True  # First point is an outlier
        assert anomalies[1] == True  # Second point is an outlier
    
    def test_zscore_returns_boolean_array(self, data_with_outliers):
        """Test that Z-score returns boolean array."""
        anomalies = detect_anomalies_zscore(data_with_outliers)
        
        assert anomalies.dtype == bool
        assert len(anomalies) == len(data_with_outliers)
    
    def test_iqr_detects_outliers(self, data_with_outliers):
        """Test that IQR method detects obvious outliers."""
        anomalies = detect_anomalies_iqr(data_with_outliers)
        
        assert anomalies[0] == True  # First point is an outlier
        assert anomalies[1] == True  # Second point is an outlier
    
    def test_iqr_returns_boolean_array(self, data_with_outliers):
        """Test that IQR returns boolean array."""
        anomalies = detect_anomalies_iqr(data_with_outliers)
        
        assert anomalies.dtype == bool
        assert len(anomalies) == len(data_with_outliers)
    
    def test_zscore_threshold_affects_detection(self):
        """Test that threshold parameter affects detection."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[0] = [4, 4]  # Moderate outlier
        
        anomalies_strict = detect_anomalies_zscore(X, threshold=5.0)
        anomalies_loose = detect_anomalies_zscore(X, threshold=2.0)
        
        # Looser threshold should detect more anomalies
        assert anomalies_loose.sum() >= anomalies_strict.sum()


class TestClusterSummary:
    """Tests for get_cluster_summary function."""
    
    @pytest.fixture
    def clustered_data(self):
        """Generate clustered data."""
        X, labels = make_blobs(n_samples=150, centers=3, random_state=42)
        return X, labels
    
    def test_returns_dataframe(self, clustered_data):
        """Test that function returns a DataFrame."""
        X, labels = clustered_data
        summary = get_cluster_summary(X, labels)
        
        assert isinstance(summary, pd.DataFrame)
    
    def test_contains_size_column(self, clustered_data):
        """Test that Size column is included."""
        X, labels = clustered_data
        summary = get_cluster_summary(X, labels)
        
        assert 'Size' in summary.columns
    
    def test_correct_number_of_clusters(self, clustered_data):
        """Test that correct number of clusters in summary."""
        X, labels = clustered_data
        summary = get_cluster_summary(X, labels)
        
        assert len(summary) == 3
    
    def test_custom_feature_names(self, clustered_data):
        """Test that custom feature names are used."""
        X, labels = clustered_data
        summary = get_cluster_summary(X, labels, feature_names=['X', 'Y'])
        
        assert any('X_' in col for col in summary.columns)
        assert any('Y_' in col for col in summary.columns)


class TestAssignClusterToNewData:
    """Tests for assign_cluster_to_new_data function."""
    
    @pytest.fixture
    def kmeans_model(self):
        """Create a fitted KMeans model."""
        X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans
    
    def test_returns_array(self, kmeans_model):
        """Test that function returns an array."""
        X_new = np.array([[0, 0], [5, 5]])
        labels = assign_cluster_to_new_data(X_new, kmeans_model.cluster_centers_)
        
        assert isinstance(labels, np.ndarray)
    
    def test_correct_number_of_labels(self, kmeans_model):
        """Test that correct number of labels are returned."""
        X_new = np.array([[0, 0], [5, 5], [10, 10]])
        labels = assign_cluster_to_new_data(X_new, kmeans_model.cluster_centers_)
        
        assert len(labels) == 3
    
    def test_labels_in_valid_range(self, kmeans_model):
        """Test that labels are within valid range."""
        X_new = np.array([[0, 0], [5, 5]])
        labels = assign_cluster_to_new_data(X_new, kmeans_model.cluster_centers_)
        
        assert all(0 <= label < 3 for label in labels)
    
    def test_point_near_centroid_assigned_correctly(self, kmeans_model):
        """Test that points near centroids are assigned to correct cluster."""
        centroids = kmeans_model.cluster_centers_
        
        # Point very close to first centroid
        X_new = centroids[0:1] + 0.01
        labels = assign_cluster_to_new_data(X_new, centroids)
        
        assert labels[0] == 0
