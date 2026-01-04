"""
Unit tests for logger module.
"""

import pytest
import logging
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.logger import (
    setup_logging,
    get_logger,
    LogColors,
    log_function_call,
    log_execution_time,
)


# ============================================================================
# Test Setup Logging
# ============================================================================

class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_basic_setup(self):
        """Test basic logging setup."""
        setup_logging(level='INFO')
        logger = logging.getLogger()
        assert logger.level == logging.INFO
    
    def test_debug_level(self):
        """Test debug level setup."""
        setup_logging(level='DEBUG')
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
    
    def test_with_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'test.log')
            setup_logging(level='INFO', log_file=log_file)
            
            logger = get_logger('test')
            logger.info("Test message")
            
            # Check file was created and has content
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content
    
    def test_level_string_case_insensitive(self):
        """Test that level string is case insensitive."""
        setup_logging(level='info')
        logger = logging.getLogger()
        assert logger.level == logging.INFO
        
        setup_logging(level='DEBUG')
        assert logging.getLogger().level == logging.DEBUG


# ============================================================================
# Test Get Logger
# ============================================================================

class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_by_name(self):
        """Test getting logger by name."""
        logger = get_logger('my_module')
        assert logger.name == 'my_module'
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger('test')
        assert isinstance(logger, logging.Logger)
    
    def test_same_name_returns_same_logger(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger('shared')
        logger2 = get_logger('shared')
        assert logger1 is logger2


# ============================================================================
# Test Log Colors
# ============================================================================

class TestLogColors:
    """Tests for LogColors class."""
    
    def test_color_constants_exist(self):
        """Test that color constants are defined."""
        assert hasattr(LogColors, 'RESET')
        assert hasattr(LogColors, 'RED')
        assert hasattr(LogColors, 'GREEN')
        assert hasattr(LogColors, 'YELLOW')
        assert hasattr(LogColors, 'BLUE')
    
    def test_disable_colors(self):
        """Test that colors can be disabled."""
        # Save original values
        original_reset = LogColors.RESET
        
        LogColors.disable()
        
        assert LogColors.RESET == ''
        assert LogColors.RED == ''
        assert LogColors.GREEN == ''
        
        # Restore (reinitialize class for other tests)
        LogColors.RESET = '\033[0m'
        LogColors.RED = '\033[31m'
        LogColors.GREEN = '\033[32m'


# ============================================================================
# Test Decorators
# ============================================================================

class TestLogFunctionCall:
    """Tests for log_function_call decorator."""
    
    def test_decorator_logs_function(self, caplog):
        """Test that decorator logs function call."""
        @log_function_call
        def my_function():
            return 42
        
        with caplog.at_level(logging.DEBUG):
            result = my_function()
        
        assert result == 42
    
    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @log_function_call
        def my_function():
            """My docstring."""
            pass
        
        assert my_function.__name__ == "my_function"
    
    def test_decorator_with_args(self, caplog):
        """Test decorator with function arguments."""
        @log_function_call
        def add(a, b):
            return a + b
        
        with caplog.at_level(logging.DEBUG):
            result = add(3, 4)
        
        assert result == 7


class TestLogExecutionTime:
    """Tests for log_execution_time decorator."""
    
    def test_logs_execution_time(self, caplog):
        """Test that execution time is logged."""
        @log_execution_time
        def slow_function():
            import time
            time.sleep(0.01)
            return "done"
        
        with caplog.at_level(logging.DEBUG):
            result = slow_function()
        
        assert result == "done"
    
    def test_preserves_return_value(self):
        """Test that return value is preserved."""
        @log_execution_time
        def get_value():
            return {"key": "value"}
        
        result = get_value()
        assert result == {"key": "value"}
    
    def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @log_execution_time
        def my_timed_function():
            """Timed docstring."""
            pass
        
        assert my_timed_function.__name__ == "my_timed_function"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases for logger."""
    
    def test_multiple_setup_calls(self):
        """Test that multiple setup calls don't cause issues."""
        setup_logging(level='INFO')
        setup_logging(level='DEBUG')
        setup_logging(level='WARNING')
        
        logger = logging.getLogger()
        assert logger.level == logging.WARNING
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent = get_logger('parent')
        child = get_logger('parent.child')
        
        assert child.parent == parent
    
    def test_exception_in_decorated_function(self):
        """Test that exceptions propagate through decorators."""
        @log_function_call
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
