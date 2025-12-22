"""
=============================================================================
Logger Utility Module
=============================================================================

Provides comprehensive logging utilities for ML experiments with support for:
- Console and file logging
- Colored output for readability
- Experiment tracking
- Log rotation
- Performance logging

Usage Examples:
---------------
    from src.utils.logger import setup_logging, get_logger

    # Basic setup
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    logger.info("Starting training...")

    # Advanced setup with file logging
    setup_logging(
        level='DEBUG',
        log_file='logs/experiment.log',
        rotation=True
    )

Author: ML Study Guide
Version: 0.1.0
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from logging.handlers import RotatingFileHandler
import functools


# =============================================================================
# Color Codes for Console Output
# =============================================================================

class LogColors:
    """
    ANSI color codes for terminal output.

    These codes work in most modern terminals (Linux, macOS, Windows 10+).
    """
    # Reset
    RESET = '\033[0m'

    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bold colors
    BOLD_RED = '\033[1;31m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_YELLOW = '\033[1;33m'
    BOLD_BLUE = '\033[1;34m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'

    @classmethod
    def disable(cls):
        """Disable colors (e.g., for non-TTY output)."""
        cls.RESET = ''
        cls.BLACK = ''
        cls.RED = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.BLUE = ''
        cls.MAGENTA = ''
        cls.CYAN = ''
        cls.WHITE = ''
        cls.BOLD_RED = ''
        cls.BOLD_GREEN = ''
        cls.BOLD_YELLOW = ''
        cls.BOLD_BLUE = ''
        cls.BG_RED = ''
        cls.BG_GREEN = ''
        cls.BG_YELLOW = ''


# =============================================================================
# Custom Formatter with Colors
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages based on level.

    This makes it easy to spot errors (red), warnings (yellow), and
    distinguish between different log levels at a glance.

    Attributes:
    -----------
    LEVEL_COLORS : dict
        Mapping of log levels to their color codes.
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD_RED,
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True
    ):
        """
        Initialize the colored formatter.

        Parameters:
        -----------
        fmt : str, optional
            Log message format string.
        datefmt : str, optional
            Date format string.
        use_colors : bool
            Whether to apply colors. Default True.
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Parameters:
        -----------
        record : logging.LogRecord
            The log record to format.

        Returns:
        --------
        str
            Formatted (and optionally colored) log message.
        """
        # Create a copy to avoid modifying the original record
        record = logging.makeLogRecord(record.__dict__)

        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            record.msg = f"{color}{record.msg}{LogColors.RESET}"

        return super().format(record)


# =============================================================================
# Experiment Logger
# =============================================================================

class ExperimentLogger:
    """
    Specialized logger for ML experiments.

    Tracks metrics, parameters, and provides structured logging for
    experiment reproducibility.

    Parameters:
    -----------
    name : str
        Name of the experiment.
    log_dir : str or Path, optional
        Directory for log files. Creates if doesn't exist.

    Examples:
    ---------
    >>> exp_logger = ExperimentLogger("mnist_cnn_v1")
    >>> exp_logger.log_params({"lr": 0.001, "batch_size": 32})
    >>> exp_logger.log_metric("accuracy", 0.95, step=100)
    >>> exp_logger.log_artifact("model.pkl")
    """

    def __init__(self, name: str, log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the experiment logger.

        Parameters:
        -----------
        name : str
            Experiment name (used for log file naming).
        log_dir : str or Path, optional
            Directory for log files. Defaults to 'logs/experiments'.
        """
        self.name = name
        self.start_time = datetime.now()

        # Set up log directory
        if log_dir is None:
            log_dir = Path("logs/experiments")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment-specific log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"

        # Initialize logger
        self.logger = logging.getLogger(f"experiment.{name}")
        self.logger.setLevel(logging.DEBUG)

        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)

        # Track metrics and parameters
        self.params: dict = {}
        self.metrics: dict = {}
        self.artifacts: list = []

        # Log experiment start
        self.logger.info(f"Experiment '{name}' started at {self.start_time}")

    def log_params(self, params: dict) -> None:
        """
        Log experiment parameters/hyperparameters.

        Parameters:
        -----------
        params : dict
            Dictionary of parameter names and values.
        """
        self.params.update(params)
        self.logger.info(f"Parameters: {params}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a metric value.

        Parameters:
        -----------
        name : str
            Metric name (e.g., 'accuracy', 'loss').
        value : float
            Metric value.
        step : int, optional
            Training step or epoch number.
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now()
        })

        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metric {name}{step_str}: {value}")

    def log_artifact(self, path: Union[str, Path]) -> None:
        """
        Log an artifact (model file, plot, etc.).

        Parameters:
        -----------
        path : str or Path
            Path to the artifact file.
        """
        self.artifacts.append(str(path))
        self.logger.info(f"Artifact saved: {path}")

    def finish(self) -> dict:
        """
        Finish the experiment and return summary.

        Returns:
        --------
        dict
            Summary of the experiment.
        """
        end_time = datetime.now()
        duration = end_time - self.start_time

        summary = {
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'log_file': str(self.log_file)
        }

        self.logger.info(f"Experiment finished. Duration: {duration}")
        return summary


# =============================================================================
# Main Setup Functions
# =============================================================================

def setup_logging(
    level: Union[str, int] = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    rotation: bool = False,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    This function configures the root logger with sensible defaults
    for ML development, including colored console output and optional
    file logging with rotation.

    Parameters:
    -----------
    level : str or int
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        Default is 'INFO'.
    log_file : str or Path, optional
        Path to log file. If provided, logs will also be written to file.
    log_format : str, optional
        Custom format string for log messages.
        Default: '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    date_format : str, optional
        Custom format string for timestamps.
        Default: '%Y-%m-%d %H:%M:%S'
    rotation : bool
        Enable log file rotation. Default False.
    max_bytes : int
        Maximum log file size before rotation (if rotation=True).
        Default 10 MB.
    backup_count : int
        Number of backup files to keep (if rotation=True).
        Default 5.
    use_colors : bool
        Use colored output in console. Default True.

    Returns:
    --------
    logging.Logger
        The configured root logger.

    Examples:
    ---------
    >>> # Basic setup
    >>> setup_logging(level='DEBUG')

    >>> # With file logging and rotation
    >>> setup_logging(
    ...     level='INFO',
    ...     log_file='logs/app.log',
    ...     rotation=True
    ... )
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default formats
    if log_format is None:
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # ---------------------------------------------------------------------
    # Console Handler
    # ---------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(
        fmt=log_format,
        datefmt=date_format,
        use_colors=use_colors
    ))
    root_logger.addHandler(console_handler)

    # ---------------------------------------------------------------------
    # File Handler (optional)
    # ---------------------------------------------------------------------
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_path)

        file_handler.setLevel(level)
        # No colors for file output
        file_handler.setFormatter(logging.Formatter(
            fmt=log_format,
            datefmt=date_format
        ))
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Parameters:
    -----------
    name : str, optional
        Logger name. If None, returns root logger.
        Typically use __name__ to get a module-specific logger.

    Returns:
    --------
    logging.Logger
        Logger instance.

    Examples:
    ---------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing data...")
    >>> logger.error("Something went wrong!")
    """
    return logging.getLogger(name)


# =============================================================================
# Logging Decorators
# =============================================================================

def log_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    log_args: bool = True,
    log_result: bool = False
):
    """
    Decorator to log function execution.

    Parameters:
    -----------
    logger : logging.Logger, optional
        Logger to use. If None, creates one from function's module.
    level : int
        Log level for messages. Default INFO.
    log_args : bool
        Whether to log function arguments. Default True.
    log_result : bool
        Whether to log function return value. Default False.

    Examples:
    ---------
    >>> @log_execution()
    ... def train_model(epochs, lr):
    ...     return "trained"
    >>> train_model(10, 0.01)
    # Logs: "Calling train_model(epochs=10, lr=0.01)"
    # Logs: "train_model completed in X.XXs"
    """
    def decorator(func):
        func_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            if log_args:
                args_str = ', '.join(
                    [repr(a) for a in args] +
                    [f"{k}={v!r}" for k, v in kwargs.items()]
                )
                func_logger.log(level, f"Calling {func.__name__}({args_str})")
            else:
                func_logger.log(level, f"Calling {func.__name__}()")

            # Execute function
            import time
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                if log_result:
                    func_logger.log(
                        level,
                        f"{func.__name__} completed in {elapsed:.3f}s, "
                        f"returned: {result!r}"
                    )
                else:
                    func_logger.log(
                        level,
                        f"{func.__name__} completed in {elapsed:.3f}s"
                    )

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                func_logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    # Demonstrate logging capabilities
    print("=" * 60)
    print("Logger Module Demonstration")
    print("=" * 60)

    # Set up logging
    setup_logging(level='DEBUG')
    logger = get_logger(__name__)

    # Test different log levels
    print("\n1. Log Levels:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test decorator
    print("\n2. Decorated Function:")

    @log_execution(log_result=True)
    def example_function(x, y):
        return x + y

    result = example_function(5, 3)

    print("\n" + "=" * 60)
    print("Logger demonstration complete!")
    print("=" * 60)
