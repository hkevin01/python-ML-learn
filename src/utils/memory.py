"""
=============================================================================
Memory Monitoring Utility Module
=============================================================================

Provides memory monitoring and management utilities for ML applications.
Helps prevent out-of-memory crashes and optimize memory usage.

Features:
---------
- Real-time memory monitoring
- Memory usage tracking over time
- Automatic garbage collection triggers
- Memory-aware decorators
- GPU memory monitoring (if available)

Usage Examples:
---------------
    from src.utils.memory import MemoryMonitor, check_memory

    # Quick memory check
    if check_memory(required_gb=4.0):
        load_large_dataset()

    # Continuous monitoring
    with MemoryMonitor() as monitor:
        train_model()
    print(monitor.peak_memory_gb)

Author: ML Study Guide
Version: 0.1.0
"""

import gc
import os
import sys
import functools
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time


# =============================================================================
# Memory Statistics Container
# =============================================================================

@dataclass
class MemoryStats:
    """
    Container for memory statistics.

    Attributes:
    -----------
    total_gb : float
        Total system memory in GB.
    available_gb : float
        Available memory in GB.
    used_gb : float
        Used memory in GB.
    percent_used : float
        Percentage of memory used.
    timestamp : datetime
        When the measurement was taken.
    """
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    percent_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"Memory: {self.used_gb:.2f}/{self.total_gb:.2f} GB "
            f"({self.percent_used:.1f}% used), "
            f"{self.available_gb:.2f} GB available"
        )


# =============================================================================
# Memory Utilities
# =============================================================================

def get_memory_stats() -> MemoryStats:
    """
    Get current memory statistics.

    Returns:
    --------
    MemoryStats
        Current memory usage statistics.

    Notes:
    ------
    Requires psutil for accurate measurements.
    Falls back to basic info if psutil is not available.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()

        return MemoryStats(
            total_gb=mem.total / (1024 ** 3),
            available_gb=mem.available / (1024 ** 3),
            used_gb=mem.used / (1024 ** 3),
            percent_used=mem.percent,
            timestamp=datetime.now()
        )
    except ImportError:
        # Fallback without psutil
        logging.warning("psutil not installed, memory stats limited")
        return MemoryStats(timestamp=datetime.now())


def check_memory(
    required_gb: float = 1.0,
    warn_threshold: float = 80.0,
    critical_threshold: float = 95.0
) -> bool:
    """
    Check if enough memory is available.

    Parameters:
    -----------
    required_gb : float
        Required memory in gigabytes.
    warn_threshold : float
        Percentage at which to log a warning.
    critical_threshold : float
        Percentage at which to return False.

    Returns:
    --------
    bool
        True if enough memory is available.

    Examples:
    ---------
    >>> if check_memory(required_gb=8.0):
    ...     load_large_model()
    ... else:
    ...     load_small_model()
    """
    stats = get_memory_stats()

    # Log warning if memory is getting low
    if stats.percent_used >= warn_threshold:
        logging.warning(
            f"Memory usage is high: {stats.percent_used:.1f}% "
            f"({stats.available_gb:.2f} GB available)"
        )

    # Check if we're at critical levels
    if stats.percent_used >= critical_threshold:
        logging.error(
            f"Memory critically low: {stats.percent_used:.1f}% used"
        )
        return False

    # Check if we have enough available
    if stats.available_gb < required_gb:
        logging.warning(
            f"Insufficient memory: need {required_gb:.2f} GB, "
            f"have {stats.available_gb:.2f} GB"
        )
        return False

    return True


def force_garbage_collection(
    generations: int = 2,
    log_freed: bool = True
) -> int:
    """
    Force garbage collection to free memory.

    Parameters:
    -----------
    generations : int
        Number of generations to collect (0, 1, or 2).
        Higher = more thorough but slower.
    log_freed : bool
        Whether to log the number of objects freed.

    Returns:
    --------
    int
        Number of unreachable objects found.

    Examples:
    ---------
    >>> # After processing a large batch
    >>> del large_data
    >>> freed = force_garbage_collection()
    >>> print(f"Freed {freed} objects")
    """
    # Get memory before
    before = get_memory_stats()

    # Run garbage collection
    gc.collect(generations)

    # Get memory after
    after = get_memory_stats()

    # Calculate freed memory
    freed_gb = before.used_gb - after.used_gb

    if log_freed and freed_gb > 0.01:  # Only log if > 10 MB freed
        logging.info(f"Garbage collection freed {freed_gb:.3f} GB")

    return gc.collect(0)  # Return count from gen 0


# =============================================================================
# Memory Monitor Class
# =============================================================================

class MemoryMonitor:
    """
    Context manager for monitoring memory usage during operations.

    Tracks peak memory usage and can log periodic updates.
    Useful for identifying memory-intensive operations.

    Parameters:
    -----------
    name : str
        Name for this monitoring session.
    log_interval : float, optional
        If set, log memory usage every N seconds.
    warn_threshold : float
        Memory percentage at which to log warnings.

    Examples:
    ---------
    >>> with MemoryMonitor("Training") as monitor:
    ...     model.fit(X, y)
    >>> print(f"Peak memory: {monitor.peak_memory_gb:.2f} GB")

    >>> # With periodic logging
    >>> with MemoryMonitor("Long Operation", log_interval=30.0):
    ...     long_running_process()
    """

    def __init__(
        self,
        name: str = "MemoryMonitor",
        log_interval: Optional[float] = None,
        warn_threshold: float = 80.0
    ):
        self.name = name
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold

        # Tracking variables
        self.start_stats: Optional[MemoryStats] = None
        self.end_stats: Optional[MemoryStats] = None
        self.peak_memory_gb: float = 0.0
        self.measurements: List[MemoryStats] = []

        # Background monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    def __enter__(self) -> 'MemoryMonitor':
        """Start monitoring."""
        self.start_stats = get_memory_stats()
        self.peak_memory_gb = self.start_stats.used_gb
        self.measurements = [self.start_stats]

        logging.debug(
            f"[{self.name}] Started - {self.start_stats}"
        )

        # Start background monitoring if interval is set
        if self.log_interval is not None:
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._background_monitor,
                daemon=True
            )
            self._monitor_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop monitoring and log summary."""
        # Stop background thread
        if self._monitor_thread is not None:
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=1.0)

        # Get final stats
        self.end_stats = get_memory_stats()
        self.measurements.append(self.end_stats)

        # Update peak if needed
        if self.end_stats.used_gb > self.peak_memory_gb:
            self.peak_memory_gb = self.end_stats.used_gb

        # Calculate memory change
        memory_change = self.end_stats.used_gb - self.start_stats.used_gb

        # Log summary
        change_str = f"+{memory_change:.3f}" if memory_change >= 0 else f"{memory_change:.3f}"
        logging.info(
            f"[{self.name}] Finished - "
            f"Peak: {self.peak_memory_gb:.2f} GB, "
            f"Change: {change_str} GB"
        )

        return False  # Don't suppress exceptions

    def _background_monitor(self) -> None:
        """Background thread for periodic monitoring."""
        while not self._stop_monitoring.wait(self.log_interval):
            stats = get_memory_stats()
            self.measurements.append(stats)

            # Update peak
            if stats.used_gb > self.peak_memory_gb:
                self.peak_memory_gb = stats.used_gb

            # Log current status
            logging.debug(f"[{self.name}] {stats}")

            # Warn if threshold exceeded
            if stats.percent_used >= self.warn_threshold:
                logging.warning(
                    f"[{self.name}] High memory: {stats.percent_used:.1f}%"
                )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory usage during monitoring.

        Returns:
        --------
        dict
            Summary statistics.
        """
        if not self.measurements:
            return {}

        used_values = [m.used_gb for m in self.measurements]

        return {
            'name': self.name,
            'start_memory_gb': self.start_stats.used_gb if self.start_stats else 0,
            'end_memory_gb': self.end_stats.used_gb if self.end_stats else 0,
            'peak_memory_gb': self.peak_memory_gb,
            'min_memory_gb': min(used_values),
            'avg_memory_gb': sum(used_values) / len(used_values),
            'num_measurements': len(self.measurements),
        }


# =============================================================================
# Memory-Aware Decorators
# =============================================================================

def memory_limit(max_gb: float, action: str = 'warn'):
    """
    Decorator to enforce memory limits on functions.

    Parameters:
    -----------
    max_gb : float
        Maximum memory usage allowed in GB.
    action : str
        What to do if limit is exceeded:
        - 'warn': Log a warning
        - 'error': Raise an exception
        - 'gc': Force garbage collection

    Returns:
    --------
    Callable
        Decorated function.

    Examples:
    ---------
    >>> @memory_limit(max_gb=8.0, action='error')
    ... def memory_intensive_operation():
    ...     # Process large data
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check before execution
            stats_before = get_memory_stats()

            # Execute function
            result = func(*args, **kwargs)

            # Check after execution
            stats_after = get_memory_stats()
            memory_used = stats_after.used_gb - stats_before.used_gb

            if memory_used > max_gb:
                msg = (
                    f"{func.__name__} used {memory_used:.2f} GB, "
                    f"exceeding limit of {max_gb:.2f} GB"
                )

                if action == 'warn':
                    logging.warning(msg)
                elif action == 'error':
                    raise MemoryError(msg)
                elif action == 'gc':
                    logging.warning(f"{msg} - triggering garbage collection")
                    force_garbage_collection()

            return result

        return wrapper
    return decorator


def clear_memory_after(func: Callable) -> Callable:
    """
    Decorator to clear memory after function execution.

    Useful for functions that create large temporary objects.

    Examples:
    ---------
    >>> @clear_memory_after
    ... def process_large_batch(data):
    ...     # Creates many temporary objects
    ...     return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            force_garbage_collection(log_freed=False)

    return wrapper


# =============================================================================
# GPU Memory Utilities (if available)
# =============================================================================

def get_gpu_memory_stats() -> Optional[Dict[str, Any]]:
    """
    Get GPU memory statistics (requires PyTorch or nvidia-ml-py).

    Returns:
    --------
    dict or None
        GPU memory stats, or None if no GPU available.
    """
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

            return {
                'device': device,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
            }
    except ImportError:
        pass

    # Try nvidia-ml-py
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return {
            'device': 0,
            'used_gb': info.used / (1024 ** 3),
            'total_gb': info.total / (1024 ** 3),
            'free_gb': info.free / (1024 ** 3),
        }
    except ImportError:
        pass

    return None


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Memory Monitoring Module Demonstration")
    print("=" * 60)

    # Test 1: Get current stats
    print("\n1. Current Memory Stats:")
    stats = get_memory_stats()
    print(f"   {stats}")

    # Test 2: Check memory
    print("\n2. Memory Check:")
    has_memory = check_memory(required_gb=1.0)
    print(f"   Has 1 GB available: {has_memory}")

    # Test 3: Memory monitor
    print("\n3. Memory Monitor:")
    with MemoryMonitor("Test Operation") as monitor:
        # Simulate some work
        data = [i ** 2 for i in range(1000000)]
        del data

    summary = monitor.get_summary()
    print(f"   Peak memory: {summary['peak_memory_gb']:.2f} GB")

    # Test 4: Garbage collection
    print("\n4. Garbage Collection:")
    force_garbage_collection()

    # Test 5: GPU memory (if available)
    print("\n5. GPU Memory:")
    gpu_stats = get_gpu_memory_stats()
    if gpu_stats:
        print(f"   GPU: {gpu_stats}")
    else:
        print("   No GPU available or libraries not installed")

    print("\n" + "=" * 60)
    print("Memory module demonstration complete!")
    print("=" * 60)
