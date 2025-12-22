"""
Unit tests for visualization_helpers module.

These tests verify that visualization functions work correctly.
Note: We use matplotlib's non-interactive backend to avoid display issues.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.utils.visualization_helpers import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_class_balance,
    plot_feature_importance,
    plot_confusion_matrix,
    create_eda_dashboard,
    quick_hist,
    quick_corr,
    quick_missing,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'income': np.random.normal(50000, 15000, 100),
        'satisfaction': np.random.uniform(1, 10, 100),
        'performance': np.random.normal(75, 10, 100),
        'department': np.random.choice(['Sales', 'Engineering', 'HR'], 100),
    })


@pytest.fixture
def sample_series():
    """Create sample Series for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 20, 100), name='test_data')


class TestPlotDistribution:
    """Tests for plot_distribution function."""
    
    def test_basic_plot(self, sample_series):
        """Test basic distribution plot."""
        ax = plot_distribution(sample_series)
        assert ax is not None
        assert hasattr(ax, 'figure')
        plt.close('all')
    
    def test_with_title(self, sample_series):
        """Test with custom title."""
        ax = plot_distribution(sample_series, title='Custom Title')
        assert ax.get_title() == 'Custom Title'
        plt.close('all')
    
    def test_without_kde(self, sample_series):
        """Test without KDE overlay."""
        ax = plot_distribution(sample_series, kde=False)
        assert ax is not None
        plt.close('all')
    
    def test_without_stats(self, sample_series):
        """Test without statistics display."""
        ax = plot_distribution(sample_series, show_stats=False)
        assert ax is not None
        plt.close('all')
    
    def test_custom_bins(self, sample_series):
        """Test with custom bin count."""
        ax = plot_distribution(sample_series, bins=20)
        assert ax is not None
        plt.close('all')
    
    def test_with_existing_axes(self, sample_series):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_ax = plot_distribution(sample_series, ax=ax)
        assert result_ax is ax
        plt.close('all')


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""
    
    def test_basic_heatmap(self, sample_df):
        """Test basic correlation heatmap."""
        fig, high_corr = plot_correlation_heatmap(sample_df)
        assert fig is not None
        assert isinstance(high_corr, pd.DataFrame)
        plt.close('all')
    
    def test_with_threshold(self, sample_df):
        """Test with correlation threshold."""
        fig, high_corr = plot_correlation_heatmap(sample_df, threshold=0.1)
        assert fig is not None
        plt.close('all')
    
    def test_without_mask(self, sample_df):
        """Test without upper triangle mask."""
        fig, _ = plot_correlation_heatmap(sample_df, mask_upper=False)
        assert fig is not None
        plt.close('all')
    
    def test_without_annotations(self, sample_df):
        """Test without annotations."""
        fig, _ = plot_correlation_heatmap(sample_df, annot=False)
        assert fig is not None
        plt.close('all')
    
    def test_custom_colormap(self, sample_df):
        """Test with custom colormap."""
        fig, _ = plot_correlation_heatmap(sample_df, cmap='coolwarm')
        assert fig is not None
        plt.close('all')


class TestPlotMissingValues:
    """Tests for plot_missing_values function."""
    
    def test_no_missing_values(self, sample_df):
        """Test with DataFrame without missing values."""
        fig = plot_missing_values(sample_df)
        assert fig is not None
        plt.close('all')
    
    def test_with_missing_values(self, sample_df):
        """Test with DataFrame containing missing values."""
        df = sample_df.copy()
        df.loc[0:9, 'age'] = np.nan
        df.loc[0:19, 'income'] = np.nan
        fig = plot_missing_values(df)
        assert fig is not None
        plt.close('all')
    
    def test_custom_threshold(self, sample_df):
        """Test with custom threshold line."""
        df = sample_df.copy()
        df.loc[0:9, 'age'] = np.nan
        fig = plot_missing_values(df, threshold_line=10.0)
        assert fig is not None
        plt.close('all')


class TestPlotClassBalance:
    """Tests for plot_class_balance function."""
    
    def test_binary_classification(self):
        """Test with binary classification."""
        y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
        fig, ratio = plot_class_balance(y)
        assert fig is not None
        assert ratio > 0
        plt.close('all')
    
    def test_multiclass(self):
        """Test with multiple classes."""
        y = pd.Series([0, 0, 1, 1, 1, 2, 2, 2, 2])
        fig, ratio = plot_class_balance(y)
        assert fig is not None
        plt.close('all')
    
    def test_custom_title(self):
        """Test with custom title."""
        y = pd.Series([0, 0, 1, 1, 1])
        fig, _ = plot_class_balance(y, title='Custom Title')
        assert fig is not None
        plt.close('all')
    
    def test_without_ratio(self):
        """Test without showing ratio."""
        y = pd.Series([0, 0, 1, 1, 1])
        fig, _ = plot_class_balance(y, show_ratio=False)
        assert fig is not None
        plt.close('all')


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance function."""
    
    def test_basic_importance(self):
        """Test basic feature importance plot."""
        importance = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
        features = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        fig = plot_feature_importance(importance, features)
        assert fig is not None
        plt.close('all')
    
    def test_top_n(self):
        """Test showing only top N features."""
        importance = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
        features = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        fig = plot_feature_importance(importance, features, top_n=3)
        assert fig is not None
        plt.close('all')
    
    def test_custom_title(self):
        """Test with custom title."""
        importance = np.array([0.3, 0.2, 0.15])
        features = ['a', 'b', 'c']
        fig = plot_feature_importance(importance, features, title='Custom')
        assert fig is not None
        plt.close('all')


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""
    
    def test_basic_confusion_matrix(self):
        """Test basic confusion matrix plot."""
        confusion = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(confusion)
        assert fig is not None
        plt.close('all')
    
    def test_with_labels(self):
        """Test with custom labels."""
        confusion = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(confusion, labels=['Negative', 'Positive'])
        assert fig is not None
        plt.close('all')
    
    def test_normalized(self):
        """Test normalized confusion matrix."""
        confusion = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(confusion, normalize=True)
        assert fig is not None
        plt.close('all')
    
    def test_multiclass_confusion(self):
        """Test multiclass confusion matrix."""
        confusion = np.array([[30, 5, 2], [3, 40, 5], [1, 2, 35]])
        fig = plot_confusion_matrix(confusion)
        assert fig is not None
        plt.close('all')


class TestCreateEDADashboard:
    """Tests for create_eda_dashboard function."""
    
    def test_basic_dashboard(self, sample_df):
        """Test basic EDA dashboard."""
        fig = create_eda_dashboard(sample_df)
        assert fig is not None
        plt.close('all')
    
    def test_with_target(self, sample_df):
        """Test dashboard with target column."""
        fig = create_eda_dashboard(sample_df, target_col='department')
        assert fig is not None
        plt.close('all')
    
    def test_with_specific_columns(self, sample_df):
        """Test dashboard with specific numeric columns."""
        fig = create_eda_dashboard(sample_df, numeric_cols=['age', 'income'])
        assert fig is not None
        plt.close('all')


class TestQuickFunctions:
    """Tests for convenience functions."""
    
    def test_quick_hist(self, sample_series):
        """Test quick_hist function."""
        ax = quick_hist(sample_series)
        assert ax is not None
        plt.close('all')
    
    def test_quick_corr(self, sample_df):
        """Test quick_corr function."""
        fig, _ = quick_corr(sample_df)
        assert fig is not None
        plt.close('all')
    
    def test_quick_missing(self, sample_df):
        """Test quick_missing function."""
        fig = quick_missing(sample_df)
        assert fig is not None
        plt.close('all')
