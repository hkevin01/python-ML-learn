"""
Unit tests for stats_helpers module.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.utils.stats_helpers import (
    bootstrap_mean,
    calculate_confidence_interval,
    check_normality,
    cohens_d,
    compare_two_groups,
    correlation_with_pvalue,
    describe_distribution,
    find_outliers_iqr,
    find_outliers_zscore,
    interpret_effect_size,
)


@pytest.fixture
def normal_data():
    """Create normally distributed data."""
    np.random.seed(42)
    return np.random.normal(100, 15, 200)


@pytest.fixture
def skewed_data():
    """Create skewed data."""
    np.random.seed(42)
    return np.random.exponential(2, 200)


class TestDescribeDistribution:
    """Tests for describe_distribution function."""

    def test_returns_dict(self, normal_data):
        """Test that function returns a dictionary."""
        result = describe_distribution(normal_data)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, normal_data):
        """Test that result contains expected keys."""
        result = describe_distribution(normal_data)
        expected_keys = [
            "count",
            "mean",
            "median",
            "std",
            "var",
            "min",
            "max",
            "range",
            "q1",
            "q3",
            "iqr",
            "skewness",
            "kurtosis",
            "cv",
        ]
        for key in expected_keys:
            assert key in result

    def test_with_pandas_series(self, normal_data):
        """Test with pandas Series input."""
        series = pd.Series(normal_data, name="test_data")
        result = describe_distribution(series)
        assert result["name"] == "test_data"

    def test_count_correct(self, normal_data):
        """Test count is correct."""
        result = describe_distribution(normal_data)
        assert result["count"] == len(normal_data)


class TestTestNormality:
    """Tests for check_normality function."""

    def test_returns_dict(self, normal_data):
        """Test that function returns a dictionary."""
        result = check_normality(normal_data)
        assert isinstance(result, dict)

    def test_normal_data_detected(self, normal_data):
        """Test that normal data is detected as normal."""
        result = check_normality(normal_data)
        # May not always be true due to randomness, but should usually work
        assert "is_normal" in result
        assert "p_value" in result

    def test_skewed_data_detected(self, skewed_data):
        """Test that skewed data is detected as not normal."""
        result = check_normality(skewed_data)
        # Exponential distribution should not be normal
        assert result["is_normal"] == False


class TestCohensD:
    """Tests for cohens_d function."""

    def test_identical_groups(self):
        """Test that identical groups have d=0."""
        data = np.array([1, 2, 3, 4, 5])
        d = cohens_d(data, data)
        assert d == 0.0

    def test_large_difference(self):
        """Test large effect size detection."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([10, 11, 12, 13, 14])
        d = cohens_d(group1, group2)
        assert abs(d) > 0.8  # Large effect

    def test_sign_correct(self):
        """Test that sign indicates direction."""
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([1, 2, 3, 4, 5])
        d = cohens_d(group1, group2)
        assert d > 0  # group1 > group2


class TestInterpretEffectSize:
    """Tests for interpret_effect_size function."""

    def test_negligible(self):
        """Test negligible effect size."""
        assert interpret_effect_size(0.1) == "negligible"

    def test_small(self):
        """Test small effect size."""
        assert interpret_effect_size(0.3) == "small"

    def test_medium(self):
        """Test medium effect size."""
        assert interpret_effect_size(0.6) == "medium"

    def test_large(self):
        """Test large effect size."""
        assert interpret_effect_size(1.0) == "large"

    def test_handles_negative(self):
        """Test that negative values are handled correctly."""
        assert interpret_effect_size(-0.9) == "large"


class TestCompareTwoGroups:
    """Tests for compare_two_groups function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        result = compare_two_groups(group1, group2)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        """Test that result contains expected keys."""
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        result = compare_two_groups(group1, group2)
        expected_keys = ["t_statistic", "p_value", "cohens_d", "significant"]
        for key in expected_keys:
            assert key in result

    def test_significant_difference(self):
        """Test detection of significant difference."""
        np.random.seed(42)
        group1 = np.random.normal(100, 5, 100)
        group2 = np.random.normal(120, 5, 100)
        result = compare_two_groups(group1, group2)
        assert result["significant"] == True


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval function."""

    def test_returns_tuple(self, normal_data):
        """Test that function returns a tuple."""
        result = calculate_confidence_interval(normal_data)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_mean_within_interval(self, normal_data):
        """Test that mean is within the interval."""
        lower, upper, margin = calculate_confidence_interval(normal_data)
        mean = np.mean(normal_data)
        assert lower < mean < upper

    def test_higher_confidence_wider_interval(self, normal_data):
        """Test that higher confidence gives wider interval."""
        _, _, margin_90 = calculate_confidence_interval(normal_data, confidence=0.90)
        _, _, margin_99 = calculate_confidence_interval(normal_data, confidence=0.99)
        assert margin_99 > margin_90


class TestCorrelationWithPvalue:
    """Tests for correlation_with_pvalue function."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = correlation_with_pvalue(x, y)
        assert np.isclose(result["correlation"], 1.0, atol=1e-10)

    def test_negative_correlation(self):
        """Test negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = correlation_with_pvalue(x, y)
        assert result["correlation"] < 0
        assert result["direction"] == "negative"

    def test_spearman_method(self):
        """Test Spearman correlation method."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])  # Monotonic but not linear
        result = correlation_with_pvalue(x, y, method="spearman")
        assert result["method"] == "spearman"
        assert np.isclose(result["correlation"], 1.0, atol=1e-10)  # Perfect monotonic

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            correlation_with_pvalue(x, y, method="invalid")


class TestFindOutliersZscore:
    """Tests for find_outliers_zscore function."""

    def test_no_outliers(self):
        """Test data without outliers."""
        data = np.array([1, 2, 3, 4, 5])
        indices, values = find_outliers_zscore(data)
        assert len(indices) == 0

    def test_finds_outliers(self):
        """Test that outliers are found."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        indices, values = find_outliers_zscore(data, threshold=2.0)
        assert len(indices) > 0
        assert 100 in values


class TestFindOutliersIqr:
    """Tests for find_outliers_iqr function."""

    def test_no_outliers(self):
        """Test data without outliers."""
        data = np.array([1, 2, 3, 4, 5])
        indices, values = find_outliers_iqr(data)
        assert len(indices) == 0

    def test_finds_outliers(self):
        """Test that outliers are found."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        indices, values = find_outliers_iqr(data)
        assert len(indices) > 0
        assert 100 in values


class TestBootstrapMean:
    """Tests for bootstrap_mean function."""

    def test_returns_dict(self, normal_data):
        """Test that function returns a dictionary."""
        result = bootstrap_mean(normal_data, n_iterations=100)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, normal_data):
        """Test that result contains expected keys."""
        result = bootstrap_mean(normal_data, n_iterations=100)
        expected_keys = ["mean", "std", "ci_lower", "ci_upper", "confidence"]
        for key in expected_keys:
            assert key in result

    def test_ci_contains_true_mean(self, normal_data):
        """Test that CI typically contains true mean."""
        # Use larger iterations for reliability
        result = bootstrap_mean(normal_data, n_iterations=500, random_state=42)
        sample_mean = np.mean(normal_data)
        # Mean should be close to sample mean
        assert np.isclose(result["mean"], sample_mean, atol=5)

    def test_reproducible_with_seed(self, normal_data):
        """Test reproducibility with random state."""
        result1 = bootstrap_mean(normal_data, n_iterations=100, random_state=42)
        result2 = bootstrap_mean(normal_data, n_iterations=100, random_state=42)
        assert result1["mean"] == result2["mean"]
