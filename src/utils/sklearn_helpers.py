"""
Scikit-Learn Helper Functions
============================

Reusable utilities for common ML tasks with scikit-learn.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union


def quick_train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple:
    """
    Quick train/test split with sensible defaults.
    
    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target
    test_size : float
        Proportion for test set (default 0.2)
    stratify : bool
        Whether to stratify (default True for classification)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # Determine if stratification is appropriate (classification)
    stratify_param = y if stratify and len(np.unique(y)) < 50 else None
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive classifier evaluation.
    
    Parameters
    ----------
    model : estimator
        Fitted classifier
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    target_names : list, optional
        Class names for report
        
    Returns
    -------
    dict
        Dictionary with accuracy, predictions, and classification report
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'predictions': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=target_names
        )
    }


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive regressor evaluation.
    
    Parameters
    ----------
    model : estimator
        Fitted regressor
    X_test : array-like
        Test features
    y_test : array-like
        True values
        
    Returns
    -------
    dict
        Dictionary with various regression metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, 
        r2_score, mean_absolute_percentage_error
    )
    
    y_pred = model.predict(X_test)
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }


def compare_classifiers(
    classifiers: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare multiple classifiers using cross-validation.
    
    Parameters
    ----------
    classifiers : dict
        Dictionary of {name: estimator}
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of CV folds
    scoring : str
        Scoring metric
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    from sklearn.model_selection import cross_val_score
    
    results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    return pd.DataFrame(results).T.sort_values('mean', ascending=False)


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_imputer: str = 'median',
    numeric_scaler: str = 'standard'
) -> Any:
    """
    Create a preprocessing pipeline for mixed data types.
    
    Parameters
    ----------
    numeric_features : list
        Names of numeric columns
    categorical_features : list
        Names of categorical columns
    numeric_imputer : str
        Imputation strategy ('mean', 'median', 'most_frequent')
    numeric_scaler : str
        Scaling strategy ('standard', 'minmax', 'robust')
        
    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    # Select scaler
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scalers.get(numeric_scaler, StandardScaler())
    
    # Numeric transformer
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=numeric_imputer)),
        ('scaler', scaler)
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


def quick_grid_search(
    estimator,
    param_grid: Dict[str, List],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Quick grid search with sensible defaults.
    
    Parameters
    ----------
    estimator : estimator
        Model to tune
    param_grid : dict
        Parameter grid
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of CV folds
    scoring : str
        Scoring metric
    n_jobs : int
        Number of parallel jobs (-1 for all)
        
    Returns
    -------
    tuple
        (best_estimator, best_params)
    """
    from sklearn.model_selection import GridSearchCV
    
    grid = GridSearchCV(
        estimator, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=n_jobs, return_train_score=True
    )
    grid.fit(X, y)
    
    return grid.best_estimator_, grid.best_params_


def learning_curve_data(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Generate learning curve data.
    
    Parameters
    ----------
    estimator : estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of CV folds
    train_sizes : array-like, optional
        Relative or absolute numbers of training examples
        
    Returns
    -------
    dict
        Dictionary with train_sizes, train_scores, test_scores
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        train_sizes=train_sizes,
        n_jobs=-1
    )
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': train_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'test_scores_mean': test_scores.mean(axis=1),
        'test_scores_std': test_scores.std(axis=1)
    }


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Parameters
    ----------
    model : estimator
        Fitted model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int, optional
        Number of top features to return
        
    Returns
    -------
    pd.DataFrame
        Feature importance ranking
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    
    return importance_df


def detect_data_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Check for potential data leakage between train and test sets.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    threshold : float
        Similarity threshold for warning
        
    Returns
    -------
    dict
        Leakage detection results
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit on train, find distances to test
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    
    # Check for very close matches (potential duplicates)
    suspicious_count = (distances < 1e-10).sum()
    
    return {
        'suspicious_samples': suspicious_count,
        'min_distance': distances.min(),
        'mean_distance': distances.mean(),
        'has_potential_leakage': suspicious_count > 0,
        'warning': suspicious_count > 0 and f"{suspicious_count} test samples are nearly identical to training samples"
    }


def model_summary(model) -> Dict[str, Any]:
    """
    Get a summary of a fitted model.
    
    Parameters
    ----------
    model : estimator
        Fitted sklearn model
        
    Returns
    -------
    dict
        Model information
    """
    summary = {
        'type': type(model).__name__,
        'params': model.get_params(),
        'is_fitted': hasattr(model, 'n_features_in_')
    }
    
    if hasattr(model, 'n_features_in_'):
        summary['n_features'] = model.n_features_in_
    
    if hasattr(model, 'classes_'):
        summary['classes'] = list(model.classes_)
        summary['n_classes'] = len(model.classes_)
    
    if hasattr(model, 'feature_importances_'):
        summary['has_feature_importance'] = True
    
    if hasattr(model, 'coef_'):
        summary['n_coefficients'] = len(np.ravel(model.coef_))
    
    return summary
