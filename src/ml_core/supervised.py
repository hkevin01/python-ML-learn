"""
Supervised Learning Helper Functions
====================================

Reusable utilities for classification and regression tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Comprehensive classification evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for AUC calculation
    average : str
        Averaging method for multiclass ('weighted', 'macro', 'micro')
        
    Returns
    -------
    dict
        Dictionary with all classification metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, balanced_accuracy_score, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics['auc_roc'] = roc_auc_score(y_true, proba)
            else:
                # Multiclass
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
        except Exception:
            pass
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive regression evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_features : int, optional
        Number of features for adjusted R²
        
    Returns
    -------
    dict
        Dictionary with all regression metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error, median_absolute_error
    )
    
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    
    # Adjusted R²
    if n_features is not None:
        r2 = metrics['r2']
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        metrics['adjusted_r2'] = adj_r2
    
    return metrics


def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = 'accuracy',
    ax = None
):
    """
    Plot learning curve to diagnose bias/variance.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Cross-validation folds
    train_sizes : array-like, optional
        Training set sizes
    scoring : str
        Scoring metric
    ax : matplotlib axis, optional
        Axis to plot on
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, 
        scoring=scoring, n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(train_sizes_abs, train_mean, 'b-o', label='Training score')
    ax.plot(train_sizes_abs, test_mean, 'r-s', label='Cross-validation score')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel(f'{scoring.capitalize()} Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_validation_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy',
    ax = None
):
    """
    Plot validation curve to tune hyperparameters.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    param_name : str
        Parameter to vary
    param_range : array-like
        Parameter values to try
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    ax : matplotlib axis, optional
        Axis to plot on
    """
    from sklearn.model_selection import validation_curve
    import matplotlib.pyplot as plt
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(param_range, train_mean, 'b-o', label='Training score')
    ax.plot(param_range, test_mean, 'r-s', label='Cross-validation score')
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'{scoring.capitalize()} Score')
    ax.set_title(f'Validation Curve ({param_name})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax = None
):
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    models : dict
        Dictionary of {name: fitted_model}
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    ax : matplotlib axis, optional
        Axis to plot on
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def get_feature_importance_df(
    model,
    feature_names: List[str],
    importance_type: str = 'auto'
) -> pd.DataFrame:
    """
    Extract feature importance from various model types.
    
    Parameters
    ----------
    model : sklearn estimator
        Fitted model
    feature_names : list
        Feature names
    importance_type : str
        Type of importance ('auto', 'gini', 'coef', 'permutation')
        
    Returns
    -------
    pd.DataFrame
        Feature importance ranking
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).ravel()
        if len(importance) != len(feature_names):
            importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    df['cumulative'] = df['importance'].cumsum() / df['importance'].sum()
    df['rank'] = range(1, len(df) + 1)
    
    return df


def compare_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Union[str, List[str]] = 'accuracy',
    task: str = 'classification'
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Parameters
    ----------
    models : dict
        Dictionary of {name: estimator}
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Cross-validation folds
    scoring : str or list
        Scoring metric(s)
    task : str
        'classification' or 'regression'
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    from sklearn.model_selection import cross_validate
    
    if isinstance(scoring, str):
        scoring = [scoring]
    
    results = []
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        result = {'Model': name}
        for metric in scoring:
            result[f'{metric}_train'] = cv_results[f'train_{metric}'].mean()
            result[f'{metric}_train_std'] = cv_results[f'train_{metric}'].std()
            result[f'{metric}_test'] = cv_results[f'test_{metric}'].mean()
            result[f'{metric}_test_std'] = cv_results[f'test_{metric}'].std()
        
        result['fit_time'] = cv_results['fit_time'].mean()
        results.append(result)
    
    return pd.DataFrame(results).sort_values(f'{scoring[0]}_test', ascending=False)


def create_baseline_models(task: str = 'classification') -> Dict[str, Any]:
    """
    Create a set of baseline models for comparison.
    
    Parameters
    ----------
    task : str
        'classification' or 'regression'
        
    Returns
    -------
    dict
        Dictionary of baseline models
    """
    if task == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB()
        }
    else:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        
        return {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=10000),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR()
        }
