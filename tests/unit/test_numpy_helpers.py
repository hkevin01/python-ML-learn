"""
=============================================================================
NumPy Helpers Unit Tests
=============================================================================

Unit tests for the numpy_helpers utility module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.numpy_helpers import (
    normalize,
    standardize,
    check_nan,
    check_inf,
    array_info,
    safe_divide,
    clip_outliers,
    moving_average,
    set_seed,
    train_test_split_indices,
)


class TestNormalize:
    """Tests for normalization functions."""
    
    def test_minmax_basic(self):
        """Test min-max normalization."""
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize(arr, method="minmax")
        expected = np.array([0., 0.25, 0.5, 0.75, 1.])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_minmax_handles_constant_array(self):
        """Test that min-max handles constant arrays."""
        arr = np.array([5, 5, 5, 5])
        result = normalize(arr, method="minmax")
        # Should return zeros (or handle gracefully)
        assert not np.any(np.isnan(result))
    
    def test_zscore_basic(self):
        """Test z-score standardization."""
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize(arr, method="zscore")
        assert np.abs(np.mean(result)) < 1e-10  # Mean should be ~0
        assert np.abs(np.std(result) - 1.0) < 1e-10  # Std should be ~1
    
    def test_l2_normalization(self):
        """Test L2 normalization."""
        arr = np.array([3, 4])  # 3-4-5 right triangle
        result = normalize(arr, method="l2")
        expected = np.array([0.6, 0.8])  # 3/5, 4/5
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            normalize(np.array([1, 2, 3]), method="invalid")
    
    def test_2d_axis_normalization(self):
        """Test normalization along specific axis."""
        arr = np.array([[1, 2], [3, 4]])
        result = normalize(arr, method="minmax", axis=0)
        # Each column should be normalized independently
        assert result[0, 0] == 0  # Min of first column
        assert result[1, 0] == 1  # Max of first column


class TestStandardize:
    """Tests for standardize function."""
    
    def test_standardize_is_zscore(self):
        """Test that standardize is equivalent to zscore."""
        arr = np.array([10, 20, 30, 40, 50])
        result1 = standardize(arr)
        result2 = normalize(arr, method="zscore")
        np.testing.assert_array_almost_equal(result1, result2)


class TestCheckNan:
    """Tests for NaN checking."""
    
    def test_no_nan(self):
        """Test array without NaN."""
        arr = np.array([1, 2, 3, 4])
        has_nan, count = check_nan(arr)
        assert has_nan is False
        assert count == 0
    
    def test_has_nan(self):
        """Test array with NaN values."""
        arr = np.array([1, np.nan, 3, np.nan, 5])
        has_nan, count = check_nan(arr)
        assert has_nan is True
        assert count == 2


class TestCheckInf:
    """Tests for infinity checking."""
    
    def test_no_inf(self):
        """Test array without infinity."""
        arr = np.array([1, 2, 3])
        has_inf, count = check_inf(arr)
        assert has_inf is False
        assert count == 0
    
    def test_has_inf(self):
        """Test array with infinity values."""
        arr = np.array([1, np.inf, 3, -np.inf])
        has_inf, count = check_inf(arr)
        assert has_inf is True
        assert count == 2


class TestArrayInfo:
    """Tests for array_info function."""
    
    def test_basic_info(self):
        """Test basic array information."""
        arr = np.array([1, 2, 3, 4, 5])
        info = array_info(arr)
        
        assert info["shape"] == (5,)
        assert info["size"] == 5
        assert info["ndim"] == 1
        assert info["min"] == 1
        assert info["max"] == 5
        assert info["mean"] == 3.0
        assert info["nan_count"] == 0
        assert info["inf_count"] == 0


class TestSafeDivide:
    """Tests for safe_divide function."""
    
    def test_normal_division(self):
        """Test normal division."""
        a = np.array([10, 20, 30])
        b = np.array([2, 4, 5])
        result = safe_divide(a, b)
        expected = np.array([5., 5., 6.])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_division_by_zero(self):
        """Test division by zero returns fill_value."""
        a = np.array([10, 20, 30])
        b = np.array([2, 0, 5])
        result = safe_divide(a, b, fill_value=0.0)
        expected = np.array([5., 0., 6.])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_custom_fill_value(self):
        """Test custom fill value for division by zero."""
        a = np.array([10, 20])
        b = np.array([2, 0])
        result = safe_divide(a, b, fill_value=-1.0)
        assert result[1] == -1.0


class TestClipOutliers:
    """Tests for clip_outliers function."""
    
    def test_basic_clipping(self):
        """Test that outliers are clipped."""
        arr = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        result = clip_outliers(arr, lower_percentile=0, upper_percentile=90)
        # The max value should be less than or equal to the 90th percentile
        assert np.max(result) <= np.percentile(arr, 90)


class TestMovingAverage:
    """Tests for moving_average function."""
    
    def test_basic_moving_average(self):
        """Test basic moving average calculation."""
        arr = np.array([1, 2, 3, 4, 5])
        result = moving_average(arr, window_size=3)
        expected = np.array([2., 3., 4.])  # Average of [1,2,3], [2,3,4], [3,4,5]
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_window_size_one(self):
        """Test window size of 1 returns original array."""
        arr = np.array([1, 2, 3, 4, 5])
        result = moving_average(arr, window_size=1)
        np.testing.assert_array_almost_equal(result, arr)
    
    def test_invalid_ndim(self):
        """Test that 2D array raises error."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            moving_average(arr, window_size=2)


class TestTrainTestSplitIndices:
    """Tests for train_test_split_indices function."""
    
    def test_split_proportions(self):
        """Test that split proportions are correct."""
        train_idx, test_idx = train_test_split_indices(100, test_size=0.2, seed=42)
        assert len(train_idx) == 80
        assert len(test_idx) == 20
    
    def test_no_overlap(self):
        """Test that train and test indices don't overlap."""
        train_idx, test_idx = train_test_split_indices(100, seed=42)
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0
    
    def test_all_indices_used(self):
        """Test that all indices are used."""
        train_idx, test_idx = train_test_split_indices(100, seed=42)
        all_indices = set(train_idx) | set(test_idx)
        assert all_indices == set(range(100))
    
    def test_reproducibility(self):
        """Test that same seed produces same split."""
        train1, test1 = train_test_split_indices(100, seed=42)
        train2, test2 = train_test_split_indices(100, seed=42)
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)


class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_reproducibility(self):
        """Test that setting seed makes random reproducible."""
        set_seed(42)
        random1 = np.random.rand(5)
        
        set_seed(42)
        random2 = np.random.rand(5)
        
        np.testing.assert_array_equal(random1, random2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
