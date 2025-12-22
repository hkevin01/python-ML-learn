"""
=============================================================================
Timer Utility Module
=============================================================================

Provides comprehensive timing and performance measurement utilities for
machine learning experiments. Supports multiple time units, statistical
analysis, and graceful handling of edge cases.

Features:
---------
- Context manager for easy timing blocks
- Decorator for timing functions
- Multiple time unit support (ns, us, ms, s, min, hr)
- Statistical summaries for repeated measurements
- Thread-safe operation
- Memory of past measurements

Usage Examples:
---------------
    # As context manager
    with Timer("Training") as t:
        model.fit(X, y)
    print(f"Training took {t.elapsed_ms:.2f} ms")

    # As decorator
    @time_function
    def train_model(X, y):
        ...

    # Manual timing
    timer = Timer()
    timer.start()
    # ... do work ...
    timer.stop()
    print(timer.summary())

Author: ML Study Guide
Version: 0.1.0
"""

import time
import functools
import threading
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import statistics


# =============================================================================
# Time Unit Definitions
# =============================================================================

class TimeUnit(Enum):
    """
    Enumeration of supported time units with their conversion factors.

    Each unit stores the number of that unit per second.
    For example: MILLISECONDS = 1000 means 1000 ms per second.
    """
    NANOSECONDS = 1_000_000_000   # 10^9 ns per second
    MICROSECONDS = 1_000_000      # 10^6 μs per second
    MILLISECONDS = 1_000          # 10^3 ms per second
    SECONDS = 1                    # 1 s per second
    MINUTES = 1/60                 # 1/60 min per second
    HOURS = 1/3600                 # 1/3600 hr per second

    @classmethod
    def from_string(cls, unit_str: str) -> 'TimeUnit':
        """
        Convert a string representation to a TimeUnit.

        Parameters:
        -----------
        unit_str : str
            String like 'ms', 'seconds', 'μs', etc.

        Returns:
        --------
        TimeUnit
            The corresponding TimeUnit enum value.

        Raises:
        -------
        ValueError
            If the string doesn't match any known unit.

        Examples:
        ---------
        >>> TimeUnit.from_string('ms')
        TimeUnit.MILLISECONDS
        >>> TimeUnit.from_string('seconds')
        TimeUnit.SECONDS
        """
        # Mapping of common string representations to TimeUnit
        mapping = {
            # Nanoseconds
            'ns': cls.NANOSECONDS,
            'nanoseconds': cls.NANOSECONDS,
            'nanosecond': cls.NANOSECONDS,
            # Microseconds
            'us': cls.MICROSECONDS,
            'μs': cls.MICROSECONDS,
            'microseconds': cls.MICROSECONDS,
            'microsecond': cls.MICROSECONDS,
            # Milliseconds
            'ms': cls.MILLISECONDS,
            'milliseconds': cls.MILLISECONDS,
            'millisecond': cls.MILLISECONDS,
            # Seconds
            's': cls.SECONDS,
            'sec': cls.SECONDS,
            'seconds': cls.SECONDS,
            'second': cls.SECONDS,
            # Minutes
            'min': cls.MINUTES,
            'mins': cls.MINUTES,
            'minutes': cls.MINUTES,
            'minute': cls.MINUTES,
            # Hours
            'hr': cls.HOURS,
            'hrs': cls.HOURS,
            'hours': cls.HOURS,
            'hour': cls.HOURS,
        }

        # Normalize input: lowercase and strip whitespace
        normalized = unit_str.lower().strip()

        if normalized not in mapping:
            valid_units = ', '.join(sorted(set(mapping.keys())))
            raise ValueError(
                f"Unknown time unit: '{unit_str}'. "
                f"Valid units are: {valid_units}"
            )

        return mapping[normalized]


# =============================================================================
# Timer Statistics
# =============================================================================

@dataclass
class TimerStats:
    """
    Statistical summary of timer measurements.

    Attributes:
    -----------
    count : int
        Number of measurements taken.
    total : float
        Total time across all measurements (in seconds).
    mean : float
        Average time per measurement (in seconds).
    median : float
        Median time (in seconds).
    std_dev : float
        Standard deviation (in seconds), 0 if count < 2.
    min_time : float
        Minimum time recorded (in seconds).
    max_time : float
        Maximum time recorded (in seconds).
    """
    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary format."""
        return {
            'count': self.count,
            'total_seconds': self.total,
            'mean_seconds': self.mean,
            'median_seconds': self.median,
            'std_dev_seconds': self.std_dev,
            'min_seconds': self.min_time,
            'max_seconds': self.max_time,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.count == 0:
            return "No measurements recorded"

        return (
            f"Timer Stats (n={self.count}):\n"
            f"  Total:  {self.total:.4f}s\n"
            f"  Mean:   {self.mean:.4f}s\n"
            f"  Median: {self.median:.4f}s\n"
            f"  StdDev: {self.std_dev:.4f}s\n"
            f"  Range:  [{self.min_time:.4f}s, {self.max_time:.4f}s]"
        )


# =============================================================================
# Main Timer Class
# =============================================================================

class Timer:
    """
    Comprehensive timer for measuring code execution time.

    This class provides multiple ways to measure time:
    1. Context manager (with statement)
    2. Manual start/stop
    3. Decorator (via time_function)

    Features:
    - Thread-safe operation
    - Multiple time unit support
    - History of measurements for statistical analysis
    - Graceful error handling
    - Human-readable output

    Parameters:
    -----------
    name : str, optional
        A name for this timer (useful for logging).
    auto_print : bool, optional
        If True, automatically print elapsed time when stopped.
        Default is False.

    Attributes:
    -----------
    name : str
        The timer's name.
    elapsed : float
        Elapsed time in seconds (0 if not yet stopped).
    is_running : bool
        Whether the timer is currently running.
    measurements : List[float]
        History of all measurements (in seconds).

    Examples:
    ---------
    >>> # Context manager usage
    >>> with Timer("My Operation") as t:
    ...     time.sleep(0.1)
    >>> print(f"Took {t.elapsed_ms:.2f} ms")
    Took 100.xx ms

    >>> # Manual usage
    >>> t = Timer()
    >>> t.start()
    >>> time.sleep(0.1)
    >>> t.stop()
    >>> print(t.elapsed)
    0.1...

    >>> # Multiple measurements
    >>> t = Timer("Loop")
    >>> for i in range(5):
    ...     t.start()
    ...     time.sleep(0.01)
    ...     t.stop()
    >>> print(t.stats())
    Timer Stats (n=5): ...
    """

    def __init__(
        self,
        name: str = "Timer",
        auto_print: bool = False
    ):
        """
        Initialize a new Timer instance.

        Parameters:
        -----------
        name : str
            Descriptive name for this timer.
        auto_print : bool
            Whether to print elapsed time automatically on stop.
        """
        # ---------------------------------------------------------------------
        # Instance attributes
        # ---------------------------------------------------------------------
        self.name = name
        self.auto_print = auto_print

        # Internal state
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._is_running: bool = False

        # Thread safety lock
        self._lock = threading.Lock()

        # History of measurements for statistical analysis
        self._measurements: List[float] = []

    # =========================================================================
    # Core Timer Methods
    # =========================================================================

    def start(self) -> 'Timer':
        """
        Start the timer.

        Returns:
        --------
        Timer
            Self, for method chaining.

        Raises:
        -------
        RuntimeError
            If the timer is already running.

        Examples:
        ---------
        >>> t = Timer()
        >>> t.start()
        >>> # ... do work ...
        >>> t.stop()
        """
        with self._lock:
            # Boundary condition: prevent starting an already-running timer
            if self._is_running:
                raise RuntimeError(
                    f"Timer '{self.name}' is already running. "
                    "Call stop() before starting again."
                )

            # Record start time using high-resolution timer
            # time.perf_counter() is the most precise timer available
            self._start_time = time.perf_counter()
            self._is_running = True
            self._end_time = None

        return self

    def stop(self) -> float:
        """
        Stop the timer and record the elapsed time.

        Returns:
        --------
        float
            Elapsed time in seconds.

        Raises:
        -------
        RuntimeError
            If the timer is not running.

        Examples:
        ---------
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(0.1)
        >>> elapsed = t.stop()
        >>> print(f"{elapsed:.2f} seconds")
        """
        with self._lock:
            # Boundary condition: prevent stopping a non-running timer
            if not self._is_running:
                raise RuntimeError(
                    f"Timer '{self.name}' is not running. "
                    "Call start() before stop()."
                )

            # Record end time
            self._end_time = time.perf_counter()
            self._is_running = False

            # Calculate elapsed time
            # Handle potential edge case where start_time could be None
            if self._start_time is None:
                self._elapsed = 0.0
            else:
                self._elapsed = self._end_time - self._start_time

            # Store measurement for statistical analysis
            self._measurements.append(self._elapsed)

            # Auto-print if enabled
            if self.auto_print:
                print(f"[{self.name}] Elapsed: {self._format_time(self._elapsed)}")

        return self._elapsed

    def reset(self) -> 'Timer':
        """
        Reset the timer to its initial state.

        This clears the current measurement but preserves history.
        Use clear_history() to also clear the measurement history.

        Returns:
        --------
        Timer
            Self, for method chaining.
        """
        with self._lock:
            self._start_time = None
            self._end_time = None
            self._elapsed = 0.0
            self._is_running = False

        return self

    def clear_history(self) -> 'Timer':
        """
        Clear all recorded measurements.

        Returns:
        --------
        Timer
            Self, for method chaining.
        """
        with self._lock:
            self._measurements.clear()

        return self

    # =========================================================================
    # Elapsed Time Properties (Multiple Units)
    # =========================================================================

    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        If the timer is still running, returns the time elapsed so far.
        """
        with self._lock:
            if self._is_running and self._start_time is not None:
                # Timer is running - return current elapsed
                return time.perf_counter() - self._start_time
            return self._elapsed

    @property
    def elapsed_ns(self) -> float:
        """Get elapsed time in nanoseconds."""
        return self.elapsed * TimeUnit.NANOSECONDS.value

    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed * TimeUnit.MICROSECONDS.value

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * TimeUnit.MILLISECONDS.value

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds (alias for elapsed)."""
        return self.elapsed

    @property
    def elapsed_min(self) -> float:
        """Get elapsed time in minutes."""
        return self.elapsed * TimeUnit.MINUTES.value

    @property
    def elapsed_hr(self) -> float:
        """Get elapsed time in hours."""
        return self.elapsed * TimeUnit.HOURS.value

    def elapsed_in(self, unit: str) -> float:
        """
        Get elapsed time in a specified unit.

        Parameters:
        -----------
        unit : str
            Time unit string (e.g., 'ms', 'seconds', 'μs').

        Returns:
        --------
        float
            Elapsed time in the specified unit.

        Examples:
        ---------
        >>> t.elapsed_in('ms')
        123.45
        >>> t.elapsed_in('minutes')
        0.002
        """
        time_unit = TimeUnit.from_string(unit)
        return self.elapsed * time_unit.value

    @property
    def is_running(self) -> bool:
        """Check if the timer is currently running."""
        with self._lock:
            return self._is_running

    @property
    def measurements(self) -> List[float]:
        """Get a copy of all recorded measurements (in seconds)."""
        with self._lock:
            return self._measurements.copy()

    # =========================================================================
    # Statistical Analysis
    # =========================================================================

    def stats(self) -> TimerStats:
        """
        Calculate statistical summary of all measurements.

        Returns:
        --------
        TimerStats
            Dataclass containing count, total, mean, median, std_dev,
            min_time, and max_time.

        Examples:
        ---------
        >>> t = Timer()
        >>> for _ in range(10):
        ...     t.start()
        ...     time.sleep(0.01)
        ...     t.stop()
        >>> stats = t.stats()
        >>> print(f"Average: {stats.mean:.4f}s")
        """
        with self._lock:
            measurements = self._measurements.copy()

        # Handle empty measurements
        if not measurements:
            return TimerStats()

        # Handle single measurement (std_dev requires at least 2 values)
        count = len(measurements)
        total = sum(measurements)
        mean = total / count
        median = statistics.median(measurements)
        min_time = min(measurements)
        max_time = max(measurements)

        # Standard deviation requires at least 2 measurements
        if count >= 2:
            std_dev = statistics.stdev(measurements)
        else:
            std_dev = 0.0

        return TimerStats(
            count=count,
            total=total,
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_time=min_time,
            max_time=max_time,
        )

    def summary(self) -> str:
        """
        Get a human-readable summary of timer measurements.

        Returns:
        --------
        str
            Formatted summary string.
        """
        stats = self.stats()
        return f"[{self.name}] {stats}"

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> 'Timer':
        """
        Enter the context manager, starting the timer.

        Usage:
        ------
        >>> with Timer("Operation") as t:
        ...     # code to time
        ...     pass
        >>> print(t.elapsed)
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager, stopping the timer.

        Parameters:
        -----------
        exc_type : type
            Exception type if an exception occurred.
        exc_val : Exception
            Exception instance if an exception occurred.
        exc_tb : traceback
            Traceback if an exception occurred.

        Returns:
        --------
        bool
            False to propagate any exceptions (we don't suppress them).
        """
        # Stop the timer even if an exception occurred
        # This ensures we always get a measurement
        try:
            self.stop()
        except RuntimeError:
            # Timer might already be stopped in some edge cases
            pass

        # Return False to propagate any exceptions
        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format a time value in the most appropriate unit.

        Parameters:
        -----------
        seconds : float
            Time in seconds.

        Returns:
        --------
        str
            Formatted time string with appropriate unit.

        Examples:
        ---------
        >>> Timer._format_time(0.0001)
        '100.00 μs'
        >>> Timer._format_time(1.5)
        '1.50 s'
        >>> Timer._format_time(120)
        '2.00 min'
        """
        # Handle negative or zero time
        if seconds <= 0:
            return "0.00 s"

        # Choose appropriate unit based on magnitude
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.2f} μs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        elif seconds < 60:
            return f"{seconds:.2f} s"
        elif seconds < 3600:
            return f"{seconds / 60:.2f} min"
        else:
            return f"{seconds / 3600:.2f} hr"

    def __str__(self) -> str:
        """String representation of the timer."""
        status = "running" if self.is_running else "stopped"
        return f"Timer('{self.name}', {status}, elapsed={self._format_time(self.elapsed)})"

    def __repr__(self) -> str:
        """Detailed representation of the timer."""
        return (
            f"Timer(name='{self.name}', "
            f"elapsed={self.elapsed:.6f}s, "
            f"measurements={len(self._measurements)}, "
            f"is_running={self.is_running})"
        )


# =============================================================================
# Function Decorator
# =============================================================================

def time_function(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    print_result: bool = True,
    unit: str = 'auto'
) -> Callable:
    """
    Decorator to time function execution.

    Can be used with or without parentheses:
        @time_function
        def my_func(): ...

        @time_function(name="Custom Name", unit='ms')
        def my_func(): ...

    Parameters:
    -----------
    func : Callable, optional
        The function to decorate (provided automatically).
    name : str, optional
        Custom name for the timer. Defaults to function name.
    print_result : bool
        Whether to print the timing result. Default True.
    unit : str
        Time unit for output ('auto', 'ms', 's', etc.). Default 'auto'.

    Returns:
    --------
    Callable
        The decorated function.

    Examples:
    ---------
    >>> @time_function
    ... def slow_function():
    ...     time.sleep(0.1)
    >>> slow_function()
    [slow_function] Elapsed: 100.xx ms

    >>> @time_function(name="Data Processing", unit='s')
    ... def process_data(data):
    ...     return [x * 2 for x in data]
    """
    def decorator(f: Callable) -> Callable:
        # Use function name if no custom name provided
        timer_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            # Create timer for this invocation
            timer = Timer(timer_name)

            try:
                timer.start()
                result = f(*args, **kwargs)
                elapsed = timer.stop()

                if print_result:
                    if unit == 'auto':
                        formatted = timer._format_time(elapsed)
                    else:
                        value = timer.elapsed_in(unit)
                        formatted = f"{value:.2f} {unit}"
                    print(f"[{timer_name}] Elapsed: {formatted}")

                return result

            except Exception as e:
                # Stop timer even on exception
                try:
                    timer.stop()
                except RuntimeError:
                    pass
                raise  # Re-raise the original exception

        return wrapper

    # Handle both @time_function and @time_function()
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Module-level Testing
# =============================================================================

if __name__ == "__main__":
    # Basic usage demonstration
    print("=" * 60)
    print("Timer Module Demonstration")
    print("=" * 60)

    # Test 1: Context manager
    print("\n1. Context Manager Usage:")
    with Timer("Sleep Test", auto_print=True) as t:
        time.sleep(0.1)
    print(f"   Elapsed: {t.elapsed_ms:.2f} ms")

    # Test 2: Manual start/stop
    print("\n2. Manual Start/Stop:")
    timer = Timer("Manual")
    timer.start()
    time.sleep(0.05)
    elapsed = timer.stop()
    print(f"   Elapsed: {timer.elapsed_in('ms'):.2f} ms")

    # Test 3: Multiple measurements
    print("\n3. Multiple Measurements:")
    timer = Timer("Loop")
    for i in range(5):
        timer.start()
        time.sleep(0.01 + i * 0.005)
        timer.stop()
    print(timer.summary())

    # Test 4: Decorator
    print("\n4. Decorator Usage:")
    @time_function
    def example_function():
        time.sleep(0.05)
        return "done"

    result = example_function()
    print(f"   Result: {result}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
