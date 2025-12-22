"""
Unit tests for supervised learning helpers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.ml_core.supervised import (
    evaluate_classification,
    evaluate_regression,
    get_feature_importance_df,
    compare_models,
    create_baseline_models,
)


class TestEvaluateClassification:
    """Tests for evaluate_classification function."""
    
    @pytest.fixture
    def classification_data(self):
        X, y = load_iris(return_X_y=True)
        # Binary for simpler testing
        mask = y != 2
        X, y = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test
    
    def test_returns_dict(self, classification_data):
        clf, X_test, y_test = classification_data
        y_pred = clf.predict(X_test)
        result = evaluate_classification(y_test, y_pred)
        assert isinstance(result, dict)
    
    def test_contains_basic_metrics(self, classification_data):
        clf, X_test, y_test = classification_data
        y_pred = clf.predict(X_test)
        result = evaluate_classification(y_test, y_pred)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert metric in result
            assert 0 <= result[metric] <= 1
    
    def test_with_probabilities(self, classification_data):
        clf, X_test, y_test = classification_data
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        result = evaluate_classification(y_test, y_pred, y_proba)
        
        assert 'auc_roc' in result
        assert 0 <= result['auc_roc'] <= 1


class TestEvaluateRegression:
    """Tests for evaluate_regression function."""
    
    @pytest.fixture
    def regression_data(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg, X_test, y_test
    
    def test_returns_dict(self, regression_data):
        reg, X_test, y_test = regression_data
        y_pred = reg.predict(X_test)
        result = evaluate_regression(y_test, y_pred)
        assert isinstance(result, dict)
    
    def test_contains_basic_metrics(self, regression_data):
        reg, X_test, y_test = regression_data
        y_pred = reg.predict(X_test)
        result = evaluate_regression(y_test, y_pred)
        
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            assert metric in result
    
    def test_rmse_is_sqrt_mse(self, regression_data):
        reg, X_test, y_test = regression_data
        y_pred = reg.predict(X_test)
        result = evaluate_regression(y_test, y_pred)
        
        np.testing.assert_almost_equal(result['rmse'], np.sqrt(result['mse']))
    
    def test_adjusted_r2(self, regression_data):
        reg, X_test, y_test = regression_data
        y_pred = reg.predict(X_test)
        result = evaluate_regression(y_test, y_pred, n_features=5)
        
        assert 'adjusted_r2' in result


class TestGetFeatureImportanceDF:
    """Tests for get_feature_importance_df function."""
    
    def test_with_tree_model(self):
        X, y = load_iris(return_X_y=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        result = get_feature_importance_df(rf, ['f1', 'f2', 'f3', 'f4'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'importance' in result.columns
    
    def test_with_linear_model(self):
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        lr = LinearRegression()
        lr.fit(X, y)
        
        result = get_feature_importance_df(lr, ['f1', 'f2', 'f3'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_sorted_by_importance(self):
        X, y = load_iris(return_X_y=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        result = get_feature_importance_df(rf, ['f1', 'f2', 'f3', 'f4'])
        
        assert result['importance'].is_monotonic_decreasing


class TestCompareModels:
    """Tests for compare_models function."""
    
    def test_returns_dataframe(self):
        X, y = load_iris(return_X_y=True)
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        result = compare_models(models, X, y, cv=3)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    
    def test_contains_model_names(self):
        X, y = load_iris(return_X_y=True)
        models = {
            'Model_A': LogisticRegression(max_iter=1000),
            'Model_B': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        result = compare_models(models, X, y, cv=3)
        
        assert 'Model_A' in result['Model'].values
        assert 'Model_B' in result['Model'].values


class TestCreateBaselineModels:
    """Tests for create_baseline_models function."""
    
    def test_classification_models(self):
        models = create_baseline_models('classification')
        
        assert isinstance(models, dict)
        assert len(models) >= 4
        assert 'Logistic Regression' in models
    
    def test_regression_models(self):
        models = create_baseline_models('regression')
        
        assert isinstance(models, dict)
        assert len(models) >= 4
        assert 'Linear Regression' in models
    
    def test_models_can_fit(self):
        X, y = load_iris(return_X_y=True)
        models = create_baseline_models('classification')
        
        for name, model in models.items():
            model.fit(X, y)
            assert hasattr(model, 'predict')
