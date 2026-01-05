"""
Unit tests for memory module.
"""

import gc
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.memory import (
    MemoryMonitor,
    MemoryStats,
    check_memory,
    clear_memory_after,
    force_garbage_collection,
    get_memory_stats,
    memory_limit,
)

# ============================================================================
# Test MemoryStats
# ============================================================================


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = MemoryStats()
        assert stats.total_gb == 0.0
        assert stats.available_gb == 0.0
        assert stats.used_gb == 0.0
        assert stats.percent_used == 0.0

    def test_custom_values(self):
        """Test custom values."""
        stats = MemoryStats(
            total_gb=16.0, available_gb=8.0, used_gb=8.0, percent_used=50.0
        )
        assert stats.total_gb == 16.0
        assert stats.available_gb == 8.0
        assert stats.percent_used == 50.0

    def test_str_representation(self):
        """Test string representation."""
        stats = MemoryStats(
            total_gb=16.0, available_gb=8.0, used_gb=8.0, percent_used=50.0
        )
        s = str(stats)
        assert "8.00" in s
        assert "16.00" in s
        assert "50.0%" in s


# ============================================================================
# Test get_memory_stats
# ============================================================================


class TestGetMemoryStats:
    """Tests for get_memory_stats function."""

    def test_returns_memory_stats(self):
        """Test that function returns MemoryStats."""
        stats = get_memory_stats()
        assert isinstance(stats, MemoryStats)

    def test_values_are_reasonable(self):
        """Test that memory values are reasonable."""
        stats = get_memory_stats()

        # Total should be positive
        assert stats.total_gb > 0

        # Used should be <= total
        assert stats.used_gb <= stats.total_gb

        # Percent should be 0-100
        assert 0 <= stats.percent_used <= 100

    def test_has_timestamp(self):
        """Test that stats have timestamp."""
        stats = get_memory_stats()
        assert stats.timestamp is not None


# ============================================================================
# Test check_memory
# ============================================================================


class TestCheckMemory:
    """Tests for check_memory function."""

    def test_returns_bool(self):
        """Test that function returns boolean."""
        result = check_memory(required_gb=0.001)
        assert isinstance(result, bool)

    def test_small_requirement_passes(self):
        """Test that small memory requirement passes."""
        # 1MB should always be available
        result = check_memory(required_gb=0.001)
        assert result is True

    def test_huge_requirement_fails(self):
        """Test that huge memory requirement fails."""
        # 10TB should never be available
        result = check_memory(required_gb=10000)
        assert result is False


# ============================================================================
# Test MemoryMonitor
# ============================================================================


class TestMemoryMonitor:
    """Tests for MemoryMonitor context manager."""

    def test_context_manager(self):
        """Test basic context manager usage."""
        with MemoryMonitor() as monitor:
            # Do some work
            _ = [i**2 for i in range(1000)]

        assert monitor is not None

    def test_tracks_peak_memory(self):
        """Test that peak memory is tracked."""
        with MemoryMonitor() as monitor:
            # Allocate some memory
            data = np.zeros((1000, 1000))

        assert hasattr(monitor, "peak_memory_gb")
        assert monitor.peak_memory_gb >= 0

    def test_has_get_summary_method(self):
        """Test that get_summary method is available."""
        with MemoryMonitor() as monitor:
            import time

            time.sleep(0.01)

        # Check for get_summary method
        assert hasattr(monitor, "get_summary")
        summary = monitor.get_summary()
        assert "peak_memory_gb" in summary
        assert "start_memory_gb" in summary
        assert "end_memory_gb" in summary

    def test_tracks_start_and_end_stats(self):
        """Test start and end stats are captured."""
        with MemoryMonitor() as monitor:
            pass

        assert monitor.start_stats is not None
        assert monitor.end_stats is not None


# ============================================================================
# Test force_garbage_collection
# ============================================================================


class TestForceGarbageCollection:
    """Tests for force_garbage_collection function."""

    def test_clears_memory(self):
        """Test that memory clearing works."""
        # Create some garbage
        _ = [np.zeros((100, 100)) for _ in range(10)]

        # Clear it
        force_garbage_collection()

        # Should not raise
        gc.collect()

    def test_returns_collection_info(self):
        """Test return value."""
        result = force_garbage_collection()
        # Result should be an integer (number of objects collected) or None
        assert result is None or isinstance(result, int)


# ============================================================================
# Test clear_memory_after decorator
# ============================================================================


class TestClearMemoryAfter:
    """Tests for clear_memory_after decorator."""

    def test_decorator_works(self):
        """Test basic decorator functionality."""

        @clear_memory_after
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_preserves_return_value(self):
        """Test that return value is preserved."""

        @clear_memory_after
        def create_array():
            return np.array([1, 2, 3])

        result = create_array()
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @clear_memory_after
        def my_efficient_function():
            """Efficient docstring."""
            pass

        assert my_efficient_function.__name__ == "my_efficient_function"

    def test_with_arguments(self):
        """Test decorator with function arguments."""

        @clear_memory_after
        def multiply(a, b):
            return a * b

        result = multiply(3, 4)
        assert result == 12


# ============================================================================
# Test memory_limit decorator
# ============================================================================


class TestMemoryLimit:
    """Tests for memory_limit decorator."""

    def test_decorator_allows_small_allocation(self):
        """Test that small allocations pass."""

        @memory_limit(max_gb=10.0, action="warn")
        def small_allocation():
            return np.zeros((10, 10))

        result = small_allocation()
        assert result is not None

    def test_preserves_return_value(self):
        """Test that return value is preserved."""

        @memory_limit(max_gb=10.0)
        def get_value():
            return {"key": "value"}

        result = get_value()
        assert result == {"key": "value"}


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for memory utilities."""

    def test_memory_stats_are_current(self):
        """Test that stats reflect current state."""
        stats1 = get_memory_stats()

        # Allocate memory
        large_array = np.zeros((1000, 1000))

        stats2 = get_memory_stats()

        # Memory should have changed (or at least be measurable)
        assert stats1.timestamp != stats2.timestamp

        # Clean up
        del large_array
        gc.collect()

    def test_monitor_handles_exception(self):
        """Test that monitor handles exceptions properly."""
        try:
            with MemoryMonitor() as monitor:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Monitor should still have recorded something
        assert monitor is not None
        assert monitor.end_stats is not None

    def test_nested_monitors(self):
        """Test nested memory monitors."""
        with MemoryMonitor(name="outer") as outer:
            with MemoryMonitor(name="inner") as inner:
                _ = np.zeros((100, 100))

        assert outer is not None
        assert inner is not None
        assert outer.peak_memory_gb >= 0
        assert inner.peak_memory_gb >= 0
