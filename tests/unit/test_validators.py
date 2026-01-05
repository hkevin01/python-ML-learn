"""
Unit tests for validators module.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.validators import (
    check_choice,
    check_range,
    check_type,
    validate_array,
    validate_dataframe,
    validate_X_y,
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
        with pytest.raises(ValueError, match="dimensions|ndim|dim"):
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
        with pytest.raises(ValueError, match="NaN|nan|contains"):
            validate_array(arr, allow_nan=False)

    def test_nan_allowed(self):
        """Test that NaN is allowed by default."""
        arr = np.array([1, np.nan, 3])
        result = validate_array(arr, allow_nan=True)
        assert np.isnan(result[1])

    def test_inf_not_allowed(self):
        """Test infinity detection when not allowed."""
        arr = np.array([1, np.inf, 3])
        with pytest.raises(ValueError, match="[Ii]nf|infinite"):
            validate_array(arr, allow_inf=False)

    def test_min_value(self):
        """Test minimum value validation."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="min|below|less"):
            validate_array(arr, min_value=2)

    def test_max_value(self):
        """Test maximum value validation."""
        arr = np.array([1, 2, 10])
        with pytest.raises(ValueError, match="max|above|greater"):
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
# Test X, y Validation
# ============================================================================


class TestValidateXy:
    """Tests for validate_X_y function."""

    def test_basic_validation(self):
        """Test basic X, y validation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        X_out, y_out = validate_X_y(X, y)

        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="length|sample|shape"):
            validate_X_y(X, y)

    def test_list_conversion(self):
        """Test that lists are converted to arrays."""
        X = [[1, 2], [3, 4]]
        y = [0, 1]

        X_out, y_out = validate_X_y(X, y)

        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)


# ============================================================================
# Test DataFrame Validation
# ============================================================================


class TestValidateDataFrame:
    """Tests for validate_dataframe function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["x", "y", "z"]})

    def test_basic_dataframe(self, sample_df):
        """Test validation of basic DataFrame."""
        result = validate_dataframe(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_required_columns(self, sample_df):
        """Test required columns validation."""
        result = validate_dataframe(sample_df, required_columns=["A", "B"])
        assert "A" in result.columns
        assert "B" in result.columns

    def test_missing_required_column(self, sample_df):
        """Test missing required column raises error."""
        with pytest.raises(ValueError, match="column|Column|missing"):
            validate_dataframe(sample_df, required_columns=["A", "D"])

    def test_min_rows(self, sample_df):
        """Test minimum rows validation."""
        result = validate_dataframe(sample_df, min_rows=2)
        assert len(result) >= 2

    def test_min_rows_fails(self, sample_df):
        """Test minimum rows validation failure."""
        with pytest.raises(ValueError, match="row|sample"):
            validate_dataframe(sample_df, min_rows=10)

    def test_empty_dataframe(self):
        """Test empty DataFrame rejection."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)


# ============================================================================
# Test Type Checking
# ============================================================================


class TestCheckType:
    """Tests for check_type function."""

    def test_valid_type(self):
        """Test valid type passes."""
        result = check_type(5, int, name="count")
        assert result == 5

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(TypeError, match="must be|got"):
            check_type("string", int, name="count")

    def test_multiple_types(self):
        """Test multiple acceptable types."""
        result = check_type(5, (int, float), name="number")
        assert result == 5

        result = check_type(5.5, (int, float), name="number")
        assert result == 5.5

    def test_numpy_types(self):
        """Test numpy type checking."""
        arr = np.array([1, 2, 3])
        result = check_type(arr, np.ndarray, name="array")
        np.testing.assert_array_equal(result, arr)


# ============================================================================
# Test Range Checking
# ============================================================================


class TestCheckRange:
    """Tests for check_range function."""

    def test_value_in_range(self):
        """Test value in range passes."""
        result = check_range(5, 1, 10, name="value")
        assert result == 5

    def test_at_lower_bound(self):
        """Test value at lower bound."""
        result = check_range(1, 1, 10, name="value")
        assert result == 1

    def test_at_upper_bound(self):
        """Test value at upper bound."""
        result = check_range(10, 1, 10, name="value")
        assert result == 10

    def test_below_range(self):
        """Test value below range rejected."""
        with pytest.raises(ValueError, match="range|between|must be"):
            check_range(0, 1, 10, name="value")

    def test_above_range(self):
        """Test value above range rejected."""
        with pytest.raises(ValueError, match="range|between|must be"):
            check_range(11, 1, 10, name="value")


# ============================================================================
# Test Choice Checking
# ============================================================================


class TestCheckChoice:
    """Tests for check_choice function."""

    def test_valid_choice(self):
        """Test valid choice passes."""
        result = check_choice("a", ["a", "b", "c"], name="option")
        assert result == "a"

    def test_invalid_choice(self):
        """Test invalid choice raises error."""
        with pytest.raises(ValueError, match="choice|must be one of|valid"):
            check_choice("d", ["a", "b", "c"], name="option")

    def test_numeric_choice(self):
        """Test numeric choices."""
        result = check_choice(2, [1, 2, 3], name="number")
        assert result == 2


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
        data = {"A": [1, 2], "B": [3, 4]}
        df = pd.DataFrame(data)
        result = validate_dataframe(df)
        assert len(result) == 2

    def test_check_type_with_none(self):
        """Test check_type with None value."""
        # Should allow None if it's an option
        with pytest.raises(TypeError):
            check_type(None, int, name="value")
