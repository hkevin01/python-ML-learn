"""
Integration tests for the complete ML pipeline.

These tests verify that different components work together correctly.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.datasets import make_classification, make_regression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.numpy_helpers import normalize, standardize, set_seed
from src.utils.pandas_helpers import get_missing_info, detect_outliers_iqr
from src.utils.sklearn_helpers import quick_train_test_split, evaluate_classifier, evaluate_regressor
from src.ml_core.supervised import evaluate_classification, evaluate_regression as eval_reg
from src.ml_core.unsupervised import find_optimal_clusters, evaluate_clustering


# ============================================================================
# Test Data Processing Pipeline
# ============================================================================

class TestDataProcessingPipeline:
    """Integration tests for data processing pipeline."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples) * 10 + 50,
            'feature2': np.random.randn(n_samples) * 5 + 25,
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Add some missing values
        df.loc[np.random.choice(n_samples, 5), 'feature1'] = np.nan
        df.loc[np.random.choice(n_samples, 3), 'feature2'] = np.nan
        
        return df
    
    def test_complete_preprocessing_flow(self, sample_dataset):
        """Test complete data preprocessing flow."""
        df = sample_dataset
        
        # Step 1: Check for missing values
        missing_info = get_missing_info(df)
        assert isinstance(missing_info, pd.DataFrame)
        assert 'missing_count' in missing_info.columns
        
        # Step 2: Fill missing values
        df['feature1'] = df['feature1'].fillna(df['feature1'].median())
        df['feature2'] = df['feature2'].fillna(df['feature2'].median())
        
        # Verify no missing values remain in numeric columns
        assert df['feature1'].isna().sum() == 0
        assert df['feature2'].isna().sum() == 0
        
        # Step 3: Normalize numeric features
        X = df[['feature1', 'feature2']].values
        X_normalized = normalize(X, method='minmax')
        
        assert X_normalized.min() >= 0
        assert X_normalized.max() <= 1
        
        # Step 4: Standardize
        X_standardized = standardize(X)
        np.testing.assert_almost_equal(X_standardized.mean(axis=0), [0, 0], decimal=10)
    
    def test_outlier_detection_integration(self, sample_dataset):
        """Test outlier detection with preprocessing."""
        df = sample_dataset.dropna()
        
        # Detect outliers
        outliers = detect_outliers_iqr(df, 'feature1')
        
        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool
        
        # Check that outliers can be filtered
        df_clean = df[~outliers]
        assert len(df_clean) <= len(df)


# ============================================================================
# Test Classification Pipeline
# ============================================================================

class TestClassificationPipeline:
    """Integration tests for classification pipeline."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        set_seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_complete_classification_flow(self, classification_data):
        """Test complete classification workflow."""
        from sklearn.linear_model import LogisticRegression
        
        X, y = classification_data
        
        # Step 1: Split data
        X_train, X_test, y_train, y_test = quick_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 160
        assert len(X_test) == 40
        
        # Step 2: Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Step 3: Evaluate using sklearn_helpers
        results = evaluate_classifier(model, X_test, y_test)
        
        assert 'accuracy' in results
        assert 'predictions' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1
        
        # Step 4: Evaluate using ml_core.supervised
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_stratified_split_preserves_proportions(self, classification_data):
        """Test that stratified split preserves class proportions."""
        X, y = classification_data
        
        # Get original proportions
        original_props = np.bincount(y) / len(y)
        
        # Stratified split
        X_train, X_test, y_train, y_test = quick_train_test_split(
            X, y, test_size=0.2, stratify=True, random_state=42
        )
        
        train_props = np.bincount(y_train) / len(y_train)
        test_props = np.bincount(y_test) / len(y_test)
        
        # Check proportions are similar
        np.testing.assert_array_almost_equal(train_props, original_props, decimal=1)
        np.testing.assert_array_almost_equal(test_props, original_props, decimal=1)


# ============================================================================
# Test Regression Pipeline
# ============================================================================

class TestRegressionPipeline:
    """Integration tests for regression pipeline."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        set_seed(42)
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=10,
            random_state=42
        )
        return X, y
    
    def test_complete_regression_flow(self, regression_data):
        """Test complete regression workflow."""
        from sklearn.linear_model import Ridge
        
        X, y = regression_data
        
        # Step 1: Split data
        X_train, X_test, y_train, y_test = quick_train_test_split(
            X, y, test_size=0.2, stratify=False, random_state=42
        )
        
        # Step 2: Standardize features
        X_train_std = standardize(X_train)
        X_test_std = standardize(X_test)
        
        # Step 3: Train model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_std, y_train)
        
        # Step 4: Evaluate using sklearn_helpers
        results = evaluate_regressor(model, X_test_std, y_test)
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        
        # Verify RMSE is sqrt of MSE
        np.testing.assert_almost_equal(
            results['rmse'], 
            np.sqrt(results['mse'])
        )
        
        # Step 5: Evaluate using ml_core.supervised
        y_pred = model.predict(X_test_std)
        metrics = eval_reg(y_test, y_pred)
        
        assert metrics['r2'] > 0  # Model should explain some variance


# ============================================================================
# Test Clustering Pipeline
# ============================================================================

class TestClusteringPipeline:
    """Integration tests for clustering pipeline."""
    
    @pytest.fixture
    def cluster_data(self):
        """Create clustering dataset."""
        from sklearn.datasets import make_blobs
        
        set_seed(42)
        X, true_labels = make_blobs(
            n_samples=200,
            n_features=5,
            centers=3,
            cluster_std=1.0,
            random_state=42
        )
        return X, true_labels
    
    def test_complete_clustering_flow(self, cluster_data):
        """Test complete clustering workflow."""
        from sklearn.cluster import KMeans
        
        X, true_labels = cluster_data
        
        # Step 1: Standardize features
        X_std = standardize(X)
        
        # Step 2: Find optimal clusters
        optimal = find_optimal_clusters(X_std, k_range=range(2, 6), method='silhouette')
        
        assert 'optimal_k' in optimal
        assert 2 <= optimal['optimal_k'] <= 5
        
        # Step 3: Fit clustering
        k = optimal['optimal_k']
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_std)
        
        # Step 4: Evaluate clustering
        metrics = evaluate_clustering(X_std, labels)
        
        assert 'silhouette' in metrics
        assert -1 <= metrics['silhouette'] <= 1
    
    def test_clustering_reproducibility(self, cluster_data):
        """Test that clustering is reproducible with seed."""
        from sklearn.cluster import KMeans
        
        X, _ = cluster_data
        X_std = standardize(X)
        
        # Fit twice with same seed
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
        
        labels1 = kmeans1.fit_predict(X_std)
        labels2 = kmeans2.fit_predict(X_std)
        
        np.testing.assert_array_equal(labels1, labels2)


# ============================================================================
# Test End-to-End Pipeline
# ============================================================================

class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_full_ml_workflow(self):
        """Test a complete ML workflow from data to evaluation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Generate synthetic data
        set_seed(42)
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Convert to DataFrame for realistic workflow
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Check data
        missing = get_missing_info(df)
        assert missing['missing_count'].sum() == 0  # No missing values
        
        # Prepare features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = quick_train_test_split(
            X, y, test_size=0.2, stratify=True, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        results = evaluate_classifier(model, X_test_scaled, y_test)
        
        # Verify results
        assert results['accuracy'] > 0.7  # Model should perform reasonably
        assert len(results['predictions']) == len(y_test)
        
        # Get detailed metrics
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        metrics = evaluate_classification(y_test, y_pred, y_proba)
        
        # All metrics should be reasonable
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            assert 0 <= metrics[key] <= 1
    
    def test_numpy_pandas_integration(self):
        """Test that numpy and pandas utilities work together."""
        # Create data
        set_seed(42)
        data = np.random.randn(100, 5)
        
        # Normalize with numpy helpers
        normalized = normalize(data, method='minmax')
        
        # Convert to DataFrame
        df = pd.DataFrame(normalized, columns=[f'col_{i}' for i in range(5)])
        
        # Check missing values
        missing = get_missing_info(df)
        assert missing['missing_count'].sum() == 0
        
        # Detect outliers (after normalization, should have few)
        outliers = detect_outliers_iqr(df, 'col_0')
        
        # Most data should not be outliers after normalization
        assert outliers.sum() < len(df) * 0.1  # Less than 10% outliers
