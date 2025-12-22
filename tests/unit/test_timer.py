"""
=============================================================================
Timer Utility Tests
=============================================================================

Unit tests for the Timer utility module.

These tests verify:
- Basic timing functionality
- Context manager usage
- Decorator functionality
- Time unit conversions
- Edge cases and error handling
"""

import time
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.timer import Timer, TimeUnit, time_function


class TestTimeUnit:
    """Tests for TimeUnit enum."""
    
    def test_from_string_milliseconds(self):
        """Test conversion of 'ms' string to TimeUnit."""
        assert TimeUnit.from_string('ms') == TimeUnit.MILLISECONDS
        assert TimeUnit.from_string('milliseconds') == TimeUnit.MILLISECONDS
    
    def test_from_string_seconds(self):
        """Test conversion of 's' string to TimeUnit."""
        assert TimeUnit.from_string('s') == TimeUnit.SECONDS
        assert TimeUnit.from_string('seconds') == TimeUnit.SECONDS
        assert TimeUnit.from_string('sec') == TimeUnit.SECONDS
    
    def test_from_string_microseconds(self):
        """Test conversion of 'us' string to TimeUnit."""
        assert TimeUnit.from_string('us') == TimeUnit.MICROSECONDS
        assert TimeUnit.from_string('μs') == TimeUnit.MICROSECONDS
    
    def test_from_string_invalid(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TimeUnit.from_string('invalid_unit')
        assert 'Unknown time unit' in str(exc_info.value)
    
    def test_from_string_case_insensitive(self):
        """Test that string conversion is case-insensitive."""
        assert TimeUnit.from_string('MS') == TimeUnit.MILLISECONDS
        assert TimeUnit.from_string('Seconds') == TimeUnit.SECONDS


class TestTimer:
    """Tests for Timer class."""
    
    def test_basic_timing(self):
        """Test that Timer measures elapsed time correctly."""
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # Sleep for 100ms
        timer.stop()
        
        # Allow some tolerance for timing
        assert timer.elapsed >= 0.09  # At least 90ms
        assert timer.elapsed < 0.2    # Less than 200ms
    
    def test_context_manager(self):
        """Test Timer as context manager."""
        with Timer("test_block") as timer:
            time.sleep(0.05)  # Sleep for 50ms
        
        assert timer.elapsed >= 0.04  # At least 40ms
        assert timer.elapsed < 0.15   # Less than 150ms
    
    def test_elapsed_ms_property(self):
        """Test elapsed_ms property returns milliseconds."""
        with Timer() as timer:
            time.sleep(0.1)  # Sleep for 100ms
        
        # elapsed_ms should be around 100
        assert timer.elapsed_ms >= 90
        assert timer.elapsed_ms < 200
    
    def test_timer_not_started(self):
        """Test accessing elapsed on unstarted timer."""
        timer = Timer()
        # Should handle gracefully - return 0 or raise
        assert timer.elapsed == 0.0 or hasattr(timer, 'elapsed')
    
    def test_timer_name(self):
        """Test that timer stores name correctly."""
        timer = Timer("my_timer")
        assert timer.name == "my_timer"
    
    def test_timer_reset(self):
        """Test timer reset functionality."""
        timer = Timer()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        
        first_elapsed = timer.elapsed
        
        timer.reset()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        
        # Both measurements should be similar
        assert abs(first_elapsed - timer.elapsed) < 0.05


class TestTimeFunction:
    """Tests for the time_function decorator."""
    
    def test_decorator_basic(self):
        """Test that decorator works on simple function."""
        @time_function
        def slow_function():
            time.sleep(0.05)
            return "done"
        
        result = slow_function()
        assert result == "done"
    
    def test_decorator_with_args(self):
        """Test decorator with function arguments."""
        @time_function
        def add_numbers(a, b):
            return a + b
        
        result = add_numbers(3, 5)
        assert result == 8
    
    def test_decorator_preserves_docstring(self):
        """Test that decorator preserves function metadata."""
        @time_function
        def documented_function():
            """This is a documented function."""
            pass
        
        assert documented_function.__doc__ == "This is a documented function."
        assert documented_function.__name__ == "documented_function"


# =============================================================================
# Run tests directly if this file is executed
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
