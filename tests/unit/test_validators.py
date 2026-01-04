"""
Unit tests for validators module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.validators import (
    validate_array,
    validate_dataframe,
    validate_positive,
    validate_probability,
    validate_in_range,
    check_is_fitted,
)


# ============================================================================
# Test Array Validation
# ============================================================================

class TestValidateArray:
    """Tests for validate_array function."""
    
    def test_basic_array(self):
        """Test validation of basic array."""
        arr = np.array([1, 2, 3])
        result = validate_array(arr)
        np.testing.assert_array_equal(result, arr)
    
    def test_list_to_array(self):
        """Test conversion from list to array."""
        result = validate_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
    
    def test_dtype_conversion(self):
        """Test dtype conversion."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = validate_array(arr, dtype=np.float64)
        assert result.dtype == np.float64
    
    def test_ndim_validation(self):
        """Test dimension validation."""
        arr = np.array([[1, 2], [3, 4]])
        result = validate_array(arr, ndim=2)
        assert result.ndim == 2
    
    def test_ndim_validation_fails(self):
        """Test dimension validation failure."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="dimensions"):
            validate_array(arr, ndim=2)
    
    def test_shape_validation(self):
        """Test shape validation."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = validate_array(arr, shape=(2, 3))
        assert result.shape == (2, 3)
    
    def test_shape_with_none(self):
        """Test shape validation with None for flexible dimensions."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = validate_array(arr, shape=(None, 3))
        assert result.shape == (3, 3)
    
    def test_nan_not_allowed(self):
        """Test NaN detection when not allowed."""
        arr = np.array([1, np.nan, 3])
        with pytest.raises(ValueError, match="NaN|nan"):
            validate_array(arr, allow_nan=False)
    
    def test_nan_allowed(self):
        """Test that NaN is allowed by default."""
        arr = np.array([1, np.nan, 3])
        result = validate_array(arr, allow_nan=True)
        assert np.isnan(result[1])
    
    def test_inf_not_allowed(self):
        """Test infinity detection when not allowed."""
        arr = np.array([1, np.inf, 3])
        with pytest.raises(ValueError, match="[Ii]nf"):
            validate_array(arr, allow_inf=False)
    
    def test_min_value(self):
        """Test minimum value validation."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="min|Min"):
            validate_array(arr, min_value=2)
    
    def test_max_value(self):
        """Test maximum value validation."""
        arr = np.array([1, 2, 10])
        with pytest.raises(ValueError, match="max|Max"):
            validate_array(arr, max_value=5)
    
    def test_empty_array(self):
        """Test empty array rejection."""
        arr = np.array([])
        with pytest.raises(ValueError, match="empty"):
            validate_array(arr)
    
    def test_ensure_2d(self):
        """Test 1D to 2D conversion."""
        arr = np.array([1, 2, 3])
        result = validate_array(arr, ensure_2d=True)
        assert result.ndim == 2
        assert result.shape == (3, 1)
    
    def test_copy(self):
        """Test that copy creates new array."""
        arr = np.array([1, 2, 3])
        result = validate_array(arr, copy=True)
        result[0] = 999
        assert arr[0] == 1  # Original unchanged


# ============================================================================
# Test DataFrame Validation
# ============================================================================

class TestValidateDataFrame:
    """Tests for validate_dataframe function."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })
    
    def test_basic_dataframe(self, sample_df):
        """Test validation of basic DataFrame."""
        result = validate_dataframe(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)
    
    def test_required_columns(self, sample_df):
        """Test required columns validation."""
        result = validate_dataframe(sample_df, required_columns=['A', 'B'])
        assert 'A' in result.columns
        assert 'B' in result.columns
    
    def test_missing_required_column(self, sample_df):
        """Test missing required column raises error."""
        with pytest.raises(ValueError, match="column|Column"):
            validate_dataframe(sample_df, required_columns=['A', 'D'])
    
    def test_min_rows(self, sample_df):
        """Test minimum rows validation."""
        result = validate_dataframe(sample_df, min_rows=2)
        assert len(result) >= 2
    
    def test_min_rows_fails(self, sample_df):
        """Test minimum rows validation failure."""
        with pytest.raises(ValueError, match="row"):
            validate_dataframe(sample_df, min_rows=10)
    
    def test_min_columns(self, sample_df):
        """Test minimum columns validation."""
        result = validate_dataframe(sample_df, min_columns=2)
        assert len(result.columns) >= 2
    
    def test_empty_dataframe(self):
        """Test empty DataFrame rejection."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)


# ============================================================================
# Test Numeric Validators
# ============================================================================

class TestValidatePositive:
    """Tests for validate_positive function."""
    
    def test_positive_value(self):
        """Test positive value passes."""
        result = validate_positive(5, name="count")
        assert result == 5
    
    def test_zero_not_allowed(self):
        """Test zero is not positive."""
        with pytest.raises(ValueError, match="positive"):
            validate_positive(0, name="count")
    
    def test_negative_not_allowed(self):
        """Test negative value rejected."""
        with pytest.raises(ValueError, match="positive"):
            validate_positive(-1, name="count")
    
    def test_allows_zero_with_flag(self):
        """Test zero allowed with flag."""
        result = validate_positive(0, name="count", allow_zero=True)
        assert result == 0


class TestValidateProbability:
    """Tests for validate_probability function."""
    
    def test_valid_probability(self):
        """Test valid probability passes."""
        result = validate_probability(0.5, name="p")
        assert result == 0.5
    
    def test_zero_probability(self):
        """Test zero probability is valid."""
        result = validate_probability(0.0, name="p")
        assert result == 0.0
    
    def test_one_probability(self):
        """Test one probability is valid."""
        result = validate_probability(1.0, name="p")
        assert result == 1.0
    
    def test_negative_probability(self):
        """Test negative probability rejected."""
        with pytest.raises(ValueError, match="probability|0|1"):
            validate_probability(-0.1, name="p")
    
    def test_over_one_probability(self):
        """Test probability > 1 rejected."""
        with pytest.raises(ValueError, match="probability|0|1"):
            validate_probability(1.5, name="p")


class TestValidateInRange:
    """Tests for validate_in_range function."""
    
    def test_in_range(self):
        """Test value in range passes."""
        result = validate_in_range(5, 1, 10, name="value")
        assert result == 5
    
    def test_at_lower_bound(self):
        """Test value at lower bound."""
        result = validate_in_range(1, 1, 10, name="value")
        assert result == 1
    
    def test_at_upper_bound(self):
        """Test value at upper bound."""
        result = validate_in_range(10, 1, 10, name="value")
        assert result == 10
    
    def test_below_range(self):
        """Test value below range rejected."""
        with pytest.raises(ValueError, match="range"):
            validate_in_range(0, 1, 10, name="value")
    
    def test_above_range(self):
        """Test value above range rejected."""
        with pytest.raises(ValueError, match="range"):
            validate_in_range(11, 1, 10, name="value")


# ============================================================================
# Test Model Fitting Check
# ============================================================================

class TestCheckIsFitted:
    """Tests for check_is_fitted function."""
    
    def test_fitted_model(self):
        """Test fitted model passes check."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model.fit(X, y)
        
        # Should not raise
        check_is_fitted(model)
    
    def test_unfitted_model_raises(self):
        """Test unfitted model raises error."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        
        with pytest.raises((ValueError, Exception)):
            check_is_fitted(model)
    
    def test_custom_attributes(self):
        """Test checking custom fitted attributes."""
        class CustomModel:
            pass
        
        model = CustomModel()
        model.is_fitted_ = True
        
        # Should not raise if attribute exists
        check_is_fitted(model, attributes=['is_fitted_'])


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases for validators."""
    
    def test_validate_array_with_pandas_series(self):
        """Test array validation accepts pandas Series."""
        s = pd.Series([1, 2, 3])
        result = validate_array(s)
        assert isinstance(result, np.ndarray)
    
    def test_validate_array_with_nested_list(self):
        """Test array validation with nested list."""
        nested = [[1, 2], [3, 4]]
        result = validate_array(nested)
        assert result.shape == (2, 2)
    
    def test_dataframe_from_dict(self):
        """Test DataFrame created from dict."""
        data = {'A': [1, 2], 'B': [3, 4]}
        df = pd.DataFrame(data)
        result = validate_dataframe(df)
        assert len(result) == 2
