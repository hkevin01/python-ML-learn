"""
Unit tests for sklearn_helpers module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils.sklearn_helpers import (
    quick_train_test_split,
    evaluate_classifier,
    evaluate_regressor,
    compare_classifiers,
    create_preprocessing_pipeline,
    quick_grid_search,
    learning_curve_data,
    get_feature_importance,
    detect_data_leakage,
    model_summary,
)


class TestQuickTrainTestSplit:
    """Tests for quick_train_test_split function."""
    
    def test_basic_split(self):
        """Test basic train/test split."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = quick_train_test_split(X, y)
        
        assert len(X_train) == 120  # 80% of 150
        assert len(X_test) == 30    # 20% of 150
        
    def test_custom_test_size(self):
        """Test with custom test size."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = quick_train_test_split(X, y, test_size=0.3)
        
        assert len(X_test) == 45  # 30% of 150
        
    def test_stratification(self):
        """Test stratified split."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = quick_train_test_split(X, y, stratify=True)
        
        # Check class proportions are similar
        train_props = np.bincount(y_train) / len(y_train)
        test_props = np.bincount(y_test) / len(y_test)
        np.testing.assert_array_almost_equal(train_props, test_props, decimal=1)
        
    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        X, y = load_iris(return_X_y=True)
        split1 = quick_train_test_split(X, y, random_state=42)
        split2 = quick_train_test_split(X, y, random_state=42)
        
        np.testing.assert_array_equal(split1[0], split2[0])


class TestEvaluateClassifier:
    """Tests for evaluate_classifier function."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = quick_train_test_split(X, y)
        
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        
        return clf, X_test, y_test
    
    def test_returns_dict(self, fitted_classifier):
        """Test that function returns a dictionary."""
        clf, X_test, y_test = fitted_classifier
        result = evaluate_classifier(clf, X_test, y_test)
        
        assert isinstance(result, dict)
        
    def test_contains_accuracy(self, fitted_classifier):
        """Test that result contains accuracy."""
        clf, X_test, y_test = fitted_classifier
        result = evaluate_classifier(clf, X_test, y_test)
        
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
        
    def test_contains_predictions(self, fitted_classifier):
        """Test that result contains predictions."""
        clf, X_test, y_test = fitted_classifier
        result = evaluate_classifier(clf, X_test, y_test)
        
        assert 'predictions' in result
        assert len(result['predictions']) == len(y_test)
        
    def test_contains_confusion_matrix(self, fitted_classifier):
        """Test that result contains confusion matrix."""
        clf, X_test, y_test = fitted_classifier
        result = evaluate_classifier(clf, X_test, y_test)
        
        assert 'confusion_matrix' in result
        assert result['confusion_matrix'].shape[0] == len(np.unique(y_test))


class TestEvaluateRegressor:
    """Tests for evaluate_regressor function."""
    
    @pytest.fixture
    def fitted_regressor(self):
        """Create a fitted regressor."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = quick_train_test_split(X, y, stratify=False)
        
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        return reg, X_test, y_test
    
    def test_returns_dict(self, fitted_regressor):
        """Test that function returns a dictionary."""
        reg, X_test, y_test = fitted_regressor
        result = evaluate_regressor(reg, X_test, y_test)
        
        assert isinstance(result, dict)
        
    def test_contains_metrics(self, fitted_regressor):
        """Test that result contains expected metrics."""
        reg, X_test, y_test = fitted_regressor
        result = evaluate_regressor(reg, X_test, y_test)
        
        expected_keys = ['mse', 'rmse', 'mae', 'r2', 'predictions']
        for key in expected_keys:
            assert key in result
            
    def test_rmse_is_sqrt_mse(self, fitted_regressor):
        """Test that RMSE is sqrt of MSE."""
        reg, X_test, y_test = fitted_regressor
        result = evaluate_regressor(reg, X_test, y_test)
        
        np.testing.assert_almost_equal(result['rmse'], np.sqrt(result['mse']))


class TestCompareClassifiers:
    """Tests for compare_classifiers function."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        X, y = load_iris(return_X_y=True)
        classifiers = {
            'KNN': KNeighborsClassifier(),
            'DT': DecisionTreeClassifier()
        }
        
        result = compare_classifiers(classifiers, X, y, cv=3)
        
        assert isinstance(result, pd.DataFrame)
        
    def test_has_all_classifiers(self):
        """Test that all classifiers are in results."""
        X, y = load_iris(return_X_y=True)
        classifiers = {
            'KNN': KNeighborsClassifier(),
            'DT': DecisionTreeClassifier()
        }
        
        result = compare_classifiers(classifiers, X, y, cv=3)
        
        assert 'KNN' in result.index
        assert 'DT' in result.index
        
    def test_sorted_by_mean(self):
        """Test that results are sorted by mean score."""
        X, y = load_iris(return_X_y=True)
        classifiers = {
            'KNN': KNeighborsClassifier(),
            'DT': DecisionTreeClassifier()
        }
        
        result = compare_classifiers(classifiers, X, y, cv=3)
        
        assert result['mean'].is_monotonic_decreasing


class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline function."""
    
    def test_returns_column_transformer(self):
        """Test that function returns a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        
        pipeline = create_preprocessing_pipeline(
            numeric_features=['age', 'income'],
            categorical_features=['education']
        )
        
        assert isinstance(pipeline, ColumnTransformer)
        
    def test_different_scalers(self):
        """Test different scaler options."""
        for scaler in ['standard', 'minmax', 'robust']:
            pipeline = create_preprocessing_pipeline(
                numeric_features=['age'],
                categorical_features=['cat'],
                numeric_scaler=scaler
            )
            assert pipeline is not None


class TestQuickGridSearch:
    """Tests for quick_grid_search function."""
    
    def test_returns_best_estimator(self):
        """Test that function returns best estimator and params."""
        X, y = load_iris(return_X_y=True)
        param_grid = {'n_neighbors': [3, 5, 7]}
        
        best_est, best_params = quick_grid_search(
            KNeighborsClassifier(), param_grid, X, y, cv=3
        )
        
        assert hasattr(best_est, 'predict')
        assert 'n_neighbors' in best_params
        
    def test_best_param_in_grid(self):
        """Test that best param is from the grid."""
        X, y = load_iris(return_X_y=True)
        param_grid = {'n_neighbors': [3, 5, 7]}
        
        best_est, best_params = quick_grid_search(
            KNeighborsClassifier(), param_grid, X, y, cv=3
        )
        
        assert best_params['n_neighbors'] in [3, 5, 7]


class TestLearningCurveData:
    """Tests for learning_curve_data function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        X, y = load_iris(return_X_y=True)
        result = learning_curve_data(KNeighborsClassifier(), X, y, cv=3)
        
        assert isinstance(result, dict)
        
    def test_contains_expected_keys(self):
        """Test that result contains expected keys."""
        X, y = load_iris(return_X_y=True)
        result = learning_curve_data(KNeighborsClassifier(), X, y, cv=3)
        
        expected_keys = ['train_sizes', 'train_scores_mean', 'test_scores_mean']
        for key in expected_keys:
            assert key in result


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        X, y = load_iris(return_X_y=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        result = get_feature_importance(rf, ['f1', 'f2', 'f3', 'f4'])
        
        assert isinstance(result, pd.DataFrame)
        
    def test_sorted_by_importance(self):
        """Test that results are sorted by importance."""
        X, y = load_iris(return_X_y=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        result = get_feature_importance(rf, ['f1', 'f2', 'f3', 'f4'])
        
        assert result['importance'].is_monotonic_decreasing
        
    def test_top_n(self):
        """Test top_n parameter."""
        X, y = load_iris(return_X_y=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        result = get_feature_importance(rf, ['f1', 'f2', 'f3', 'f4'], top_n=2)
        
        assert len(result) == 2
        
    def test_raises_on_no_importance(self):
        """Test that error is raised for models without feature importance."""
        X, y = load_iris(return_X_y=True)
        knn = KNeighborsClassifier()
        knn.fit(X, y)
        
        with pytest.raises(ValueError):
            get_feature_importance(knn, ['f1', 'f2', 'f3', 'f4'])


class TestDetectDataLeakage:
    """Tests for detect_data_leakage function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        
        result = detect_data_leakage(X_train, X_test)
        
        assert isinstance(result, dict)
        
    def test_detects_duplicates(self):
        """Test that duplicates are detected."""
        X_train = np.random.randn(100, 5)
        X_test = X_train[:5].copy()  # Copy some training samples
        
        result = detect_data_leakage(X_train, X_test)
        
        assert result['has_potential_leakage'] == True
        assert result['suspicious_samples'] > 0


class TestModelSummary:
    """Tests for model_summary function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        X, y = load_iris(return_X_y=True)
        knn = KNeighborsClassifier()
        knn.fit(X, y)
        
        result = model_summary(knn)
        
        assert isinstance(result, dict)
        
    def test_contains_type(self):
        """Test that result contains model type."""
        X, y = load_iris(return_X_y=True)
        knn = KNeighborsClassifier()
        knn.fit(X, y)
        
        result = model_summary(knn)
        
        assert result['type'] == 'KNeighborsClassifier'
        
    def test_detects_fitted(self):
        """Test that fitted status is detected."""
        X, y = load_iris(return_X_y=True)
        knn = KNeighborsClassifier()
        
        result_unfitted = model_summary(knn)
        assert result_unfitted['is_fitted'] == False
        
        knn.fit(X, y)
        result_fitted = model_summary(knn)
        assert result_fitted['is_fitted'] == True
