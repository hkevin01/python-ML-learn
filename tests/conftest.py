"""
=============================================================================
Pytest Configuration and Fixtures
=============================================================================

Central configuration for all tests in the ML Study Guide project.

This file is automatically loaded by pytest and provides:
- Common fixtures available to all tests
- Test configuration settings
- Path utilities for test data
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# =============================================================================
# Path Configuration
# =============================================================================

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"

# Add src to path for imports
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Fixtures - Reusable Test Components
# =============================================================================

@pytest.fixture
def sample_array() -> np.ndarray:
    """
    Provide a simple numpy array for testing.
    
    Returns:
        np.ndarray: 1D array with values [1, 2, 3, 4, 5]
    """
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_2d_array() -> np.ndarray:
    """
    Provide a 2D numpy array for testing.
    
    Returns:
        np.ndarray: 3x3 array
    """
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])


@pytest.fixture
def random_seed():
    """
    Set a fixed random seed for reproducible tests.
    
    Yields:
        int: The seed value used (42)
    """
    np.random.seed(42)
    yield 42


@pytest.fixture
def project_root() -> Path:
    """
    Provide the project root path.
    
    Returns:
        Path: Absolute path to project root
    """
    return PROJECT_ROOT


@pytest.fixture
def test_data_dir() -> Path:
    """
    Provide the test data directory path.
    
    Returns:
        Path: Absolute path to test data directory
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """
    Configure custom pytest markers.
    """
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
