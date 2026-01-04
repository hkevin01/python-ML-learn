"""
Unit tests for error_handling module.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.error_handling import (
    MLError,
    DataValidationError,
    ModelNotFittedError,
    safe_execute,
    retry,
    fallback,
)


# ============================================================================
# Test Custom Exceptions
# ============================================================================

class TestMLError:
    """Tests for the base MLError exception."""
    
    def test_basic_creation(self):
        """Test basic exception creation."""
        error = MLError("Test error")
        assert error.message == "Test error"
        assert str(error) == "Test error"
    
    def test_with_details(self):
        """Test exception with details dict."""
        error = MLError("Test error", details={"key": "value"})
        assert error.details == {"key": "value"}
        assert "key=value" in str(error)
    
    def test_with_cause(self):
        """Test exception with cause."""
        cause = ValueError("Original error")
        error = MLError("Wrapped error", cause=cause)
        assert error.cause == cause
        assert "ValueError" in str(error)
    
    def test_timestamp_exists(self):
        """Test that timestamp is set."""
        error = MLError("Test error")
        assert hasattr(error, 'timestamp')
        assert error.timestamp is not None
    
    def test_to_dict(self):
        """Test serialization to dict."""
        error = MLError("Test error", details={"step": "training"})
        d = error.to_dict()
        
        assert d['type'] == 'MLError'
        assert d['message'] == 'Test error'
        assert d['details'] == {"step": "training"}
        assert 'timestamp' in d


class TestDataValidationError:
    """Tests for DataValidationError."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        error = DataValidationError("Invalid shape")
        assert isinstance(error, MLError)
        assert "Invalid shape" in str(error)
    
    def test_with_column_info(self):
        """Test with column info."""
        error = DataValidationError(
            "Invalid value",
            column="age",
            expected="positive",
            actual=-5
        )
        assert error.details['column'] == 'age'
        assert error.details['expected'] == 'positive'
        assert error.details['actual'] == -5
    
    def test_inherits_from_mlerror(self):
        """Test inheritance."""
        error = DataValidationError("test")
        assert isinstance(error, MLError)


class TestModelNotFittedError:
    """Tests for ModelNotFittedError."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        error = ModelNotFittedError("Model not fitted")
        assert isinstance(error, MLError)
    
    def test_raise_and_catch(self):
        """Test that we can raise and catch properly."""
        with pytest.raises(ModelNotFittedError):
            raise ModelNotFittedError("Call fit() first")


# ============================================================================
# Test Safe Execution
# ============================================================================

class TestSafeExecute:
    """Tests for safe_execute function."""
    
    def test_successful_execution(self):
        """Test successful function execution."""
        def success_func():
            return 42
        
        result = safe_execute(success_func)
        assert result == 42
    
    def test_fallback_on_error(self):
        """Test fallback value on error."""
        def fail_func():
            raise ValueError("Oops")
        
        result = safe_execute(fail_func, fallback_value="default")
        assert result == "default"
    
    def test_with_args(self):
        """Test with function arguments."""
        def add(a, b):
            return a + b
        
        result = safe_execute(add, 3, 4)
        assert result == 7
    
    def test_with_kwargs(self):
        """Test with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = safe_execute(greet, "World", greeting="Hi")
        assert result == "Hi, World!"
    
    def test_none_as_fallback(self):
        """Test None as explicit fallback."""
        def fail_func():
            raise RuntimeError("Error")
        
        result = safe_execute(fail_func, fallback_value=None)
        assert result is None


# ============================================================================
# Test Retry Decorator
# ============================================================================

class TestRetryDecorator:
    """Tests for retry decorator."""
    
    def test_succeeds_first_try(self):
        """Test function succeeding on first try."""
        call_count = [0]
        
        @retry(max_attempts=3)
        def succeed():
            call_count[0] += 1
            return "success"
        
        result = succeed()
        assert result == "success"
        assert call_count[0] == 1
    
    def test_succeeds_after_retry(self):
        """Test function succeeding after retry."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = fail_twice()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_exhausts_retries(self):
        """Test that exception is raised after max attempts."""
        @retry(max_attempts=2, delay=0)
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fail()


# ============================================================================
# Test Fallback Decorator
# ============================================================================

class TestFallbackDecorator:
    """Tests for fallback decorator."""
    
    def test_returns_normal_result(self):
        """Test normal execution returns result."""
        @fallback(default_value="fallback")
        def succeed():
            return "success"
        
        assert succeed() == "success"
    
    def test_returns_fallback_on_error(self):
        """Test fallback on error."""
        @fallback(default_value="fallback")
        def fail():
            raise RuntimeError("Error")
        
        assert fail() == "fallback"
    
    def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @fallback(default_value=None)
        def my_function():
            """My docstring."""
            pass
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases for error handling."""
    
    def test_nested_errors(self):
        """Test nested exception handling."""
        inner = ValueError("Inner")
        outer = MLError("Outer", cause=inner)
        
        assert outer.cause == inner
        assert "Inner" in str(outer.cause)
    
    def test_empty_details(self):
        """Test error with empty details."""
        error = MLError("Test", details={})
        assert error.details == {}
        # Should not include [] in output for empty details
        assert "Test" in str(error)
    
    def test_complex_details(self):
        """Test error with complex details."""
        error = MLError(
            "Complex error",
            details={
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "number": 42
            }
        )
        d = error.to_dict()
        assert d['details']['list'] == [1, 2, 3]
        assert d['details']['nested'] == {"a": 1}
