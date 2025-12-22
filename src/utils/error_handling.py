"""
=============================================================================
Error Handling Utility Module
=============================================================================

Provides comprehensive error handling utilities for ML applications including:
- Custom exception hierarchy for ML-specific errors
- Graceful error recovery mechanisms
- Safe execution wrappers with fallbacks
- Error logging and reporting
- Crash prevention and recovery

Design Philosophy:
------------------
1. Fail gracefully - never crash without cleanup
2. Provide meaningful error messages
3. Log errors for debugging
4. Allow recovery when possible
5. Preserve data integrity

Usage Examples:
---------------
    from src.utils.error_handling import (
        safe_execute,
        MLError,
        DataValidationError,
        ModelNotFittedError
    )

    # Safe execution with fallback
    result = safe_execute(
        risky_function,
        fallback_value=None,
        log_errors=True
    )

    # Custom exception
    if not model.is_fitted:
        raise ModelNotFittedError("Call fit() before predict()")

Author: ML Study Guide
Version: 0.1.0
"""

import functools
import traceback
import sys
import logging
from typing import (
    Any, Callable, Optional, Type, TypeVar, Union,
    Tuple, Dict, List
)
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


# =============================================================================
# Type Variables for Generic Functions
# =============================================================================

T = TypeVar('T')  # Generic return type
E = TypeVar('E', bound=Exception)  # Exception type


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================
# These exceptions provide specific error types for ML operations,
# making it easier to catch and handle specific error conditions.
# =============================================================================

class MLError(Exception):
    """
    Base exception for all ML-related errors.

    All custom exceptions in this module inherit from MLError,
    allowing you to catch all ML errors with a single except clause.

    Attributes:
    -----------
    message : str
        Human-readable error message.
    details : dict
        Additional context about the error.
    timestamp : datetime
        When the error occurred.

    Examples:
    ---------
    >>> try:
    ...     raise MLError("Something went wrong", details={"step": "training"})
    ... except MLError as e:
    ...     print(e.message)
    ...     print(e.details)
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the MLError.

        Parameters:
        -----------
        message : str
            Description of what went wrong.
        details : dict, optional
            Additional context (e.g., variable values, step info).
        cause : Exception, optional
            The underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        """Format error message with details."""
        base_msg = self.message

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg += f" [{details_str}]"

        if self.cause:
            base_msg += f" (caused by: {type(self.cause).__name__}: {self.cause})"

        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'type': type(self).__name__,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None,
        }


class DataValidationError(MLError):
    """
    Raised when data validation fails.

    Use this for:
    - Invalid input shapes
    - Missing required columns
    - Invalid data types
    - Out-of-range values
    - NaN/Inf handling issues

    Examples:
    ---------
    >>> if X.shape[1] != expected_features:
    ...     raise DataValidationError(
    ...         "Feature count mismatch",
    ...         details={"expected": expected_features, "got": X.shape[1]}
    ...     )
    """

    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize DataValidationError with validation details.

        Parameters:
        -----------
        message : str
            Description of validation failure.
        column : str, optional
            Name of the column/feature that failed validation.
        expected : Any, optional
            What was expected.
        actual : Any, optional
            What was actually found.
        """
        details = kwargs.get('details', {})
        if column:
            details['column'] = column
        if expected is not None:
            details['expected'] = expected
        if actual is not None:
            details['actual'] = actual

        super().__init__(message, details=details, **kwargs)


class ModelNotFittedError(MLError):
    """
    Raised when a model method is called before fitting.

    This is similar to sklearn's NotFittedError but provides
    more context for debugging.

    Examples:
    ---------
    >>> class MyModel:
    ...     def predict(self, X):
    ...         if not hasattr(self, 'weights_'):
    ...             raise ModelNotFittedError(
    ...                 "Model not fitted",
    ...                 model_type=type(self).__name__,
    ...                 required_method="fit"
    ...             )
    """

    def __init__(
        self,
        message: str = "Model has not been fitted yet",
        model_type: Optional[str] = None,
        required_method: str = "fit",
        **kwargs
    ):
        """
        Initialize ModelNotFittedError.

        Parameters:
        -----------
        message : str
            Error message.
        model_type : str, optional
            Name of the model class.
        required_method : str
            Method that should be called first (usually 'fit').
        """
        details = kwargs.get('details', {})
        if model_type:
            details['model_type'] = model_type
        details['required_method'] = required_method

        super().__init__(message, details=details, **kwargs)


class TrainingError(MLError):
    """
    Raised when training fails.

    Examples:
    ---------
    >>> if loss != loss:  # NaN check
    ...     raise TrainingError(
    ...         "Training diverged (NaN loss)",
    ...         epoch=current_epoch,
    ...         last_valid_loss=last_loss
    ...     )
    """

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if epoch is not None:
            details['epoch'] = epoch
        if batch is not None:
            details['batch'] = batch
        if loss is not None:
            details['loss'] = loss

        super().__init__(message, details=details, **kwargs)


class ResourceError(MLError):
    """
    Raised when resource limits are exceeded.

    Use for:
    - Out of memory
    - GPU memory exhausted
    - Disk space issues
    - Timeout exceeded
    """

    def __init__(
        self,
        message: str,
        resource_type: str = "memory",
        limit: Optional[Any] = None,
        current: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['resource_type'] = resource_type
        if limit is not None:
            details['limit'] = limit
        if current is not None:
            details['current'] = current

        super().__init__(message, details=details, **kwargs)


class ConfigurationError(MLError):
    """
    Raised when configuration is invalid.

    Examples:
    ---------
    >>> if learning_rate <= 0:
    ...     raise ConfigurationError(
    ...         "Learning rate must be positive",
    ...         parameter="learning_rate",
    ...         value=learning_rate
    ...     )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        valid_range: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if parameter:
            details['parameter'] = parameter
        if value is not None:
            details['value'] = value
        if valid_range:
            details['valid_range'] = valid_range

        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Error Result Container
# =============================================================================

@dataclass
class Result:
    """
    Container for operation results that may fail.

    This implements a simple Result pattern (similar to Rust's Result type)
    for operations that may fail without using exceptions for flow control.

    Attributes:
    -----------
    success : bool
        Whether the operation succeeded.
    value : Any
        The result value (if success=True).
    error : Exception
        The error (if success=False).
    message : str
        Human-readable status message.

    Examples:
    ---------
    >>> result = safe_divide(10, 0)
    >>> if result.success:
    ...     print(result.value)
    ... else:
    ...     print(f"Error: {result.message}")
    """
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    message: str = ""

    @classmethod
    def ok(cls, value: Any, message: str = "Success") -> 'Result':
        """Create a successful result."""
        return cls(success=True, value=value, message=message)

    @classmethod
    def fail(cls, error: Exception, message: Optional[str] = None) -> 'Result':
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            message=message or str(error)
        )

    def unwrap(self) -> Any:
        """
        Get the value or raise the error.

        Returns:
        --------
        Any
            The result value.

        Raises:
        -------
        Exception
            The stored error if success=False.
        """
        if self.success:
            return self.value
        else:
            raise self.error or MLError(self.message)

    def unwrap_or(self, default: T) -> Union[Any, T]:
        """
        Get the value or return a default.

        Parameters:
        -----------
        default : T
            Value to return if the operation failed.

        Returns:
        --------
        Union[Any, T]
            The result value or the default.
        """
        return self.value if self.success else default


# =============================================================================
# Safe Execution Utilities
# =============================================================================

def safe_execute(
    func: Callable[..., T],
    *args,
    fallback_value: Optional[T] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_errors: bool = True,
    logger: Optional[logging.Logger] = None,
    reraise: bool = False,
    on_error: Optional[Callable[[Exception], None]] = None,
    **kwargs
) -> Optional[T]:
    """
    Execute a function safely, catching exceptions and returning fallback.

    This is useful for operations that might fail but shouldn't crash
    the entire program, such as:
    - Loading optional data
    - Non-critical computations
    - External API calls

    Parameters:
    -----------
    func : Callable
        The function to execute.
    *args
        Positional arguments to pass to func.
    fallback_value : T, optional
        Value to return if an exception occurs. Default None.
    exceptions : tuple of Exception types
        Which exceptions to catch. Default (Exception,).
    log_errors : bool
        Whether to log errors. Default True.
    logger : logging.Logger, optional
        Logger to use. If None, uses module logger.
    reraise : bool
        Whether to re-raise exceptions after handling. Default False.
    on_error : Callable, optional
        Callback function to call on error with the exception.
    **kwargs
        Keyword arguments to pass to func.

    Returns:
    --------
    T or None
        The function result, or fallback_value if an exception occurred.

    Examples:
    ---------
    >>> # Basic usage
    >>> result = safe_execute(risky_function, arg1, arg2)

    >>> # With custom fallback
    >>> result = safe_execute(
    ...     load_optional_data,
    ...     fallback_value={},
    ...     log_errors=True
    ... )

    >>> # Only catch specific exceptions
    >>> result = safe_execute(
    ...     parse_json,
    ...     data,
    ...     exceptions=(json.JSONDecodeError, KeyError),
    ...     fallback_value=None
    ... )
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        return func(*args, **kwargs)

    except exceptions as e:
        # Log the error
        if log_errors:
            logger.error(
                f"Error in {func.__name__}: {type(e).__name__}: {e}",
                exc_info=True
            )

        # Call error callback if provided
        if on_error is not None:
            try:
                on_error(e)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")

        # Re-raise if requested
        if reraise:
            raise

        return fallback_value


def safe_execute_with_result(
    func: Callable[..., T],
    *args,
    **kwargs
) -> Result:
    """
    Execute a function and return a Result object.

    This is useful when you need more information about what went
    wrong, not just a fallback value.

    Parameters:
    -----------
    func : Callable
        The function to execute.
    *args, **kwargs
        Arguments to pass to the function.

    Returns:
    --------
    Result
        A Result object containing either the value or the error.

    Examples:
    ---------
    >>> result = safe_execute_with_result(parse_config, "config.json")
    >>> if result.success:
    ...     config = result.value
    ... else:
    ...     print(f"Failed to load config: {result.message}")
    ...     config = default_config
    """
    try:
        value = func(*args, **kwargs)
        return Result.ok(value)
    except Exception as e:
        return Result.fail(e)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator to retry a function on failure.

    Useful for:
    - Network requests
    - Database connections
    - Flaky operations

    Parameters:
    -----------
    max_attempts : int
        Maximum number of attempts. Default 3.
    delay : float
        Initial delay between attempts in seconds. Default 1.0.
    backoff : float
        Multiplier for delay after each attempt. Default 2.0.
    exceptions : tuple
        Exceptions to catch and retry on.
    on_retry : Callable, optional
        Callback called before each retry with (exception, attempt_number).

    Returns:
    --------
    Callable
        Decorated function.

    Examples:
    ---------
    >>> @retry(max_attempts=3, delay=1.0)
    ... def fetch_data(url):
    ...     return requests.get(url).json()

    >>> @retry(exceptions=(ConnectionError,), backoff=2.0)
    ... def connect_to_db():
    ...     return database.connect()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time

            last_exception = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts:
                        # Call retry callback
                        if on_retry is not None:
                            on_retry(e, attempt)

                        # Log retry
                        logging.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): "
                            f"{e}. Retrying in {current_delay:.1f}s..."
                        )

                        # Wait before retry
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        # Final attempt failed
                        logging.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            # All attempts failed
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Error Recovery Utilities
# =============================================================================

class ErrorRecovery:
    """
    Context manager for error recovery with cleanup.

    Ensures cleanup actions are performed even if an error occurs,
    and optionally suppresses or transforms exceptions.

    Parameters:
    -----------
    cleanup : Callable, optional
        Function to call on exit (success or failure).
    on_error : Callable, optional
        Function to call on error, receives the exception.
    suppress : bool
        Whether to suppress exceptions. Default False.
    transform : Callable, optional
        Function to transform exceptions before re-raising.

    Examples:
    ---------
    >>> with ErrorRecovery(cleanup=lambda: print("Cleanup done")):
    ...     risky_operation()

    >>> with ErrorRecovery(
    ...     on_error=lambda e: save_state(),
    ...     suppress=True
    ... ):
    ...     might_fail()
    """

    def __init__(
        self,
        cleanup: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        suppress: bool = False,
        transform: Optional[Callable[[Exception], Exception]] = None
    ):
        self.cleanup = cleanup
        self.on_error = on_error
        self.suppress = suppress
        self.transform = transform
        self.exception: Optional[Exception] = None

    def __enter__(self) -> 'ErrorRecovery':
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb
    ) -> bool:
        # Store exception for later inspection
        if exc_val is not None:
            self.exception = exc_val

            # Call error handler
            if self.on_error is not None:
                try:
                    self.on_error(exc_val)
                except Exception as handler_error:
                    logging.error(f"Error in error handler: {handler_error}")

        # Always run cleanup
        if self.cleanup is not None:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                logging.error(f"Error in cleanup: {cleanup_error}")

        # Handle exception transformation
        if exc_val is not None and self.transform is not None:
            try:
                new_exception = self.transform(exc_val)
                raise new_exception from exc_val
            except Exception:
                if not self.suppress:
                    raise

        # Return True to suppress exception, False to propagate
        return self.suppress


# =============================================================================
# Memory Safety Utilities
# =============================================================================

def check_memory_available(
    required_bytes: int,
    raise_on_low: bool = True,
    threshold_percent: float = 90.0
) -> bool:
    """
    Check if enough memory is available.

    Parameters:
    -----------
    required_bytes : int
        Bytes of memory needed.
    raise_on_low : bool
        Whether to raise ResourceError if memory is low.
    threshold_percent : float
        Warning threshold as percent of total memory.

    Returns:
    --------
    bool
        True if enough memory is available.

    Raises:
    -------
    ResourceError
        If memory is insufficient and raise_on_low=True.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()

        available = mem.available
        total = mem.total
        used_percent = mem.percent

        # Check if we have enough
        has_enough = available >= required_bytes

        # Check if we're near threshold
        if used_percent >= threshold_percent:
            logging.warning(
                f"Memory usage is high: {used_percent:.1f}% "
                f"(available: {available / 1e9:.2f} GB)"
            )

        if not has_enough and raise_on_low:
            raise ResourceError(
                f"Insufficient memory: need {required_bytes / 1e9:.2f} GB, "
                f"available {available / 1e9:.2f} GB",
                resource_type="memory",
                limit=available,
                current=required_bytes
            )

        return has_enough

    except ImportError:
        # psutil not available, can't check
        logging.warning("psutil not installed, cannot check memory")
        return True


# =============================================================================
# Crash Prevention Decorator
# =============================================================================

def prevent_crash(
    fallback_value: Any = None,
    log_level: int = logging.ERROR,
    save_state: Optional[Callable[[], None]] = None
) -> Callable:
    """
    Decorator to prevent function from crashing the program.

    Catches all exceptions, logs them, optionally saves state,
    and returns a fallback value.

    Parameters:
    -----------
    fallback_value : Any
        Value to return on crash. Default None.
    log_level : int
        Logging level for error messages.
    save_state : Callable, optional
        Function to call to save state before returning.

    Returns:
    --------
    Callable
        Decorated function.

    Examples:
    ---------
    >>> @prevent_crash(fallback_value=[], save_state=lambda: save_checkpoint())
    ... def process_batch(data):
    ...     return transform(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)

            except KeyboardInterrupt:
                # Don't catch keyboard interrupt
                logging.info(f"{func.__name__} interrupted by user")
                raise

            except SystemExit:
                # Don't catch system exit
                raise

            except Exception as e:
                # Log the error
                logging.log(
                    log_level,
                    f"Prevented crash in {func.__name__}: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True
                )

                # Save state if requested
                if save_state is not None:
                    try:
                        save_state()
                        logging.info("State saved successfully")
                    except Exception as save_error:
                        logging.error(f"Failed to save state: {save_error}")

                return fallback_value

        return wrapper
    return decorator


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Error Handling Module Demonstration")
    print("=" * 60)

    # Test 1: Custom Exceptions
    print("\n1. Custom Exceptions:")
    try:
        raise DataValidationError(
            "Invalid shape",
            column="features",
            expected=(100, 10),
            actual=(100, 5)
        )
    except MLError as e:
        print(f"   Caught: {e}")
        print(f"   Details: {e.details}")

    # Test 2: Safe Execute
    print("\n2. Safe Execute:")

    def risky_divide(a, b):
        return a / b

    result = safe_execute(risky_divide, 10, 0, fallback_value=float('inf'))
    print(f"   10 / 0 with fallback = {result}")

    # Test 3: Result Pattern
    print("\n3. Result Pattern:")
    result = safe_execute_with_result(risky_divide, 10, 2)
    if result.success:
        print(f"   Success: {result.value}")
    else:
        print(f"   Failed: {result.message}")

    # Test 4: Retry Decorator
    print("\n4. Retry Decorator:")

    attempt_count = 0

    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Simulated failure")
        return "Success!"

    try:
        result = flaky_function()
        print(f"   Result: {result} (after {attempt_count} attempts)")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test 5: Error Recovery
    print("\n5. Error Recovery:")
    with ErrorRecovery(
        cleanup=lambda: print("   Cleanup executed"),
        suppress=True
    ):
        raise ValueError("This error will be suppressed")
    print("   Continued after suppressed error")

    print("\n" + "=" * 60)
    print("Error handling demonstration complete!")
    print("=" * 60)
