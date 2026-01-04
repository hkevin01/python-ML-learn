"""
Unit tests for memory module.
"""

import pytest
import numpy as np
import sys
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.memory import (
    MemoryStats,
    get_memory_stats,
    check_memory,
    MemoryMonitor,
    memory_efficient,
    clear_memory,
    get_object_size,
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
            total_gb=16.0,
            available_gb=8.0,
            used_gb=8.0,
            percent_used=50.0
        )
        assert stats.total_gb == 16.0
        assert stats.available_gb == 8.0
        assert stats.percent_used == 50.0
    
    def test_str_representation(self):
        """Test string representation."""
        stats = MemoryStats(
            total_gb=16.0,
            available_gb=8.0,
            used_gb=8.0,
            percent_used=50.0
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
            _ = [i ** 2 for i in range(1000)]
        
        assert monitor is not None
    
    def test_tracks_peak_memory(self):
        """Test that peak memory is tracked."""
        with MemoryMonitor() as monitor:
            # Allocate some memory
            data = np.zeros((1000, 1000))
        
        assert hasattr(monitor, 'peak_memory_gb')
        assert monitor.peak_memory_gb >= 0
    
    def test_has_duration(self):
        """Test that duration is recorded."""
        with MemoryMonitor() as monitor:
            import time
            time.sleep(0.01)
        
        assert hasattr(monitor, 'duration')
        assert monitor.duration > 0
    
    def test_summary_method(self):
        """Test summary method if available."""
        with MemoryMonitor() as monitor:
            pass
        
        if hasattr(monitor, 'summary'):
            summary = monitor.summary()
            assert summary is not None


# ============================================================================
# Test memory_efficient decorator
# ============================================================================

class TestMemoryEfficient:
    """Tests for memory_efficient decorator."""
    
    def test_decorator_works(self):
        """Test basic decorator functionality."""
        @memory_efficient
        def my_function():
            return 42
        
        result = my_function()
        assert result == 42
    
    def test_preserves_return_value(self):
        """Test that return value is preserved."""
        @memory_efficient
        def create_array():
            return np.array([1, 2, 3])
        
        result = create_array()
        np.testing.assert_array_equal(result, [1, 2, 3])
    
    def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @memory_efficient
        def my_efficient_function():
            """Efficient docstring."""
            pass
        
        assert my_efficient_function.__name__ == "my_efficient_function"
    
    def test_with_arguments(self):
        """Test decorator with function arguments."""
        @memory_efficient
        def multiply(a, b):
            return a * b
        
        result = multiply(3, 4)
        assert result == 12


# ============================================================================
# Test clear_memory
# ============================================================================

class TestClearMemory:
    """Tests for clear_memory function."""
    
    def test_clears_memory(self):
        """Test that memory clearing works."""
        # Create some garbage
        _ = [np.zeros((100, 100)) for _ in range(10)]
        
        # Clear it
        clear_memory()
        
        # Should not raise
        gc.collect()
    
    def test_returns_none_or_count(self):
        """Test return value."""
        result = clear_memory()
        # Result could be None or count of collected objects
        assert result is None or isinstance(result, int)


# ============================================================================
# Test get_object_size
# ============================================================================

class TestGetObjectSize:
    """Tests for get_object_size function."""
    
    def test_basic_object(self):
        """Test size of basic object."""
        obj = [1, 2, 3]
        size = get_object_size(obj)
        assert size > 0
    
    def test_numpy_array(self):
        """Test size of numpy array."""
        arr = np.zeros((100, 100), dtype=np.float64)
        size = get_object_size(arr)
        
        # Should be approximately 80000 bytes (100*100*8)
        assert size >= 80000
    
    def test_larger_object_has_larger_size(self):
        """Test that larger objects have larger sizes."""
        small = np.zeros((10, 10))
        large = np.zeros((100, 100))
        
        small_size = get_object_size(small)
        large_size = get_object_size(large)
        
        assert large_size > small_size
    
    def test_nested_objects(self):
        """Test size of nested objects."""
        nested = {
            'array': np.zeros((10, 10)),
            'list': [1, 2, 3, 4, 5],
            'string': 'hello world'
        }
        size = get_object_size(nested)
        assert size > 0


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
    
    def test_empty_object_size(self):
        """Test size of empty objects."""
        empty_list = []
        empty_dict = {}
        
        list_size = get_object_size(empty_list)
        dict_size = get_object_size(empty_dict)
        
        assert list_size > 0
        assert dict_size > 0
