"""
=============================================================================
Python Machine Learning Study Guide - Source Package
=============================================================================

This package contains all the core modules for the ML study guide:

Subpackages:
-----------
- data_processing: Data loading, cleaning, feature engineering
- ml_core: Core ML algorithms implemented from scratch with explanations
- models: Pre-built model classes for various ML tasks
- utils: Utility functions for timing, logging, error handling
- visualization: Plotting and visualization utilities

Usage:
------
    from src.utils import Timer, setup_logging
    from src.data_processing import DataLoader
    from src.ml_core import LinearRegressionFromScratch

Author: ML Study Guide
Version: 0.1.0
"""

# =============================================================================
# Package Metadata
# =============================================================================
__version__ = "0.1.0"
__author__ = "ML Study Guide"
__description__ = "Comprehensive Machine Learning Study Guide with Python"

# =============================================================================
# Convenience Imports
# =============================================================================
# Import commonly used classes/functions for easier access
# Example: from src import Timer, DataLoader

from src.utils.timer import Timer
from src.utils.logger import setup_logging, get_logger
from src.utils.error_handling import (
    MLError,
    DataValidationError,
    ModelNotFittedError,
    safe_execute
)

# =============================================================================
# Package-level Configuration
# =============================================================================
import logging

# Set up default logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())
