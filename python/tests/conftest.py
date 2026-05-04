"""pytest configuration for the python/ test suite.

Adds the parent directory (python/) to sys.path so tests can import
modules like `tigphrates_env`, `train`, `reset_pool` directly.
"""
import os
import sys

# Add python/ to sys.path so tests can import siblings.
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)


def pytest_configure(config):
    """Register custom markers used in this test suite."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring external resources (e.g. the Node bridge)",
    )
