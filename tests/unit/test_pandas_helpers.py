"""
Unit tests for pandas_helpers module.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.pandas_helpers import (
    get_missing_info,
    fill_missing_by_group,
    create_dummy_variables,
    detect_outliers_iqr,
    split_train_test,
    reduce_memory_usage,
    dataframe_info,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': ['x', 'y', 'x', 'y', 'x'],
        'D': [100, 200, 300, 400, 500]
    })


@pytest.fixture
def df_with_groups():
    """DataFrame with groups for testing."""
    return pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A'],
        'value': [1, np.nan, 3, 4, 5]
    })


# ============================================================================
# Test get_missing_info
# ============================================================================

class TestGetMissingInfo:
    """Tests for get_missing_info function."""
    
    def test_basic(self, sample_df):
        """Test basic missing info extraction."""
        result = get_missing_info(sample_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'column' in result.columns
        assert 'missing_count' in result.columns
        assert 'missing_pct' in result.columns
        assert 'dtype' in result.columns
    
    def test_counts_correct(self, sample_df):
        """Test that missing counts are correct."""
        result = get_missing_info(sample_df)
        
        # A has 1 missing, B has 1 missing
        a_row = result[result['column'] == 'A'].iloc[0]
        b_row = result[result['column'] == 'B'].iloc[0]
        
        assert a_row['missing_count'] == 1
        assert b_row['missing_count'] == 1
    
    def test_sorted_by_missing(self, sample_df):
        """Test that result is sorted by missing count descending."""
        result = get_missing_info(sample_df)
        
        counts = result['missing_count'].tolist()
        assert counts == sorted(counts, reverse=True)


# ============================================================================
# Test fill_missing_by_group
# ============================================================================

class TestFillMissingByGroup:
    """Tests for fill_missing_by_group function."""
    
    def test_fill_with_mean(self, df_with_groups):
        """Test filling with group mean."""
        result = fill_missing_by_group(df_with_groups, 'value', 'group', 'mean')
        
        # NaN was in group A, mean of [1, 5] = 3
        assert not result.isna().any()
        assert result.iloc[1] == 3.0
    
    def test_fill_with_median(self, df_with_groups):
        """Test filling with group median."""
        result = fill_missing_by_group(df_with_groups, 'value', 'group', 'median')
        
        assert not result.isna().any()
    
    def test_invalid_method(self, df_with_groups):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            fill_missing_by_group(df_with_groups, 'value', 'group', 'invalid')


# ============================================================================
# Test create_dummy_variables
# ============================================================================

class TestCreateDummyVariables:
    """Tests for create_dummy_variables function."""
    
    def test_basic(self):
        """Test basic dummy variable creation."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
        result = create_dummy_variables(df, ['color'])
        
        # With drop_first=True, should have 2 columns
        assert 'color_green' in result.columns or 'color_blue' in result.columns
        assert 'color' not in result.columns
    
    def test_drop_first_false(self):
        """Test without dropping first category."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        result = create_dummy_variables(df, ['color'], drop_first=False)
        
        # Should have all categories
        assert len([c for c in result.columns if 'color' in c]) == 2


# ============================================================================
# Test detect_outliers_iqr
# ============================================================================

class TestDetectOutliersIqr:
    """Tests for detect_outliers_iqr function."""
    
    def test_no_outliers(self):
        """Test with no outliers."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = detect_outliers_iqr(df, 'value')
        
        assert result.sum() == 0
    
    def test_with_outlier(self):
        """Test with obvious outlier."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        result = detect_outliers_iqr(df, 'value')
        
        assert result.iloc[-1] == True  # 100 should be outlier
    
    def test_multiplier(self):
        """Test that larger multiplier is more lenient."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 15]})
        
        result_15 = detect_outliers_iqr(df, 'value', multiplier=1.5)
        result_30 = detect_outliers_iqr(df, 'value', multiplier=3.0)
        
        # 3.0 multiplier should find fewer outliers
        assert result_30.sum() <= result_15.sum()


# ============================================================================
# Test split_train_test
# ============================================================================

class TestSplitTrainTest:
    """Tests for split_train_test function."""
    
    def test_split_proportions(self):
        """Test that split proportions are approximately correct."""
        df = pd.DataFrame({'X': range(100), 'y': range(100)})
        train, test = split_train_test(df, test_size=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
    
    def test_no_overlap(self):
        """Test that train and test have no overlapping indices."""
        df = pd.DataFrame({'X': range(50)})
        train, test = split_train_test(df, test_size=0.3)
        
        train_indices = set(train.index)
        test_indices = set(test.index)
        
        assert len(train_indices & test_indices) == 0
    
    def test_reproducibility(self):
        """Test that random_state produces reproducible splits."""
        df = pd.DataFrame({'X': range(100)})
        
        train1, test1 = split_train_test(df, test_size=0.2, random_state=42)
        train2, test2 = split_train_test(df, test_size=0.2, random_state=42)
        
        assert train1.index.tolist() == train2.index.tolist()
    
    def test_stratified(self):
        """Test stratified splitting."""
        df = pd.DataFrame({
            'X': range(100),
            'y': [0] * 80 + [1] * 20  # Imbalanced
        })
        
        train, test = split_train_test(df, test_size=0.2, stratify_column='y')
        
        # Both train and test should have similar proportions
        train_ratio = (train['y'] == 1).mean()
        test_ratio = (test['y'] == 1).mean()
        
        assert abs(train_ratio - 0.2) < 0.1
        assert abs(test_ratio - 0.2) < 0.1


# ============================================================================
# Test reduce_memory_usage
# ============================================================================

class TestReduceMemoryUsage:
    """Tests for reduce_memory_usage function."""
    
    def test_reduces_memory(self):
        """Test that memory is actually reduced."""
        df = pd.DataFrame({
            'small_int': np.random.randint(0, 100, 10000),
            'big_float': np.random.randn(10000)
        })
        
        start_mem = df.memory_usage(deep=True).sum()
        result = reduce_memory_usage(df, verbose=False)
        end_mem = result.memory_usage(deep=True).sum()
        
        assert end_mem < start_mem
    
    def test_preserves_values(self):
        """Test that values are preserved after optimization."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]})
        result = reduce_memory_usage(df, verbose=False)
        
        assert np.allclose(result['A'].values, df['A'].values)
        assert np.allclose(result['B'].values, df['B'].values, rtol=1e-5)


# ============================================================================
# Test dataframe_info
# ============================================================================

class TestDataframeInfo:
    """Tests for dataframe_info function."""
    
    def test_returns_dict(self, sample_df):
        """Test that function returns a dictionary."""
        result = dataframe_info(sample_df)
        
        assert isinstance(result, dict)
    
    def test_contains_expected_keys(self, sample_df):
        """Test that result contains all expected keys."""
        result = dataframe_info(sample_df)
        
        expected_keys = [
            'shape', 'memory_mb', 'columns', 'dtypes',
            'missing_total', 'missing_pct', 'duplicates',
            'numeric_columns', 'categorical_columns'
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_shape_correct(self, sample_df):
        """Test that shape is reported correctly."""
        result = dataframe_info(sample_df)
        
        assert result['shape'] == sample_df.shape
    
    def test_column_types(self, sample_df):
        """Test that numeric and categorical columns are identified."""
        result = dataframe_info(sample_df)
        
        assert 'A' in result['numeric_columns']
        assert 'C' in result['categorical_columns']
