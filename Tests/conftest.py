import os
import sys
import pytest

# Add root directory to PYTHONPATH before imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Now we can import from Utils
from Utils.logger import Logger, LogLevel


def pytest_configure(config):
    """Initial pytest configuration"""
    pass


# Define test suites
test_suites = {
    'logger': [os.path.join('Tests', 'UnitTests', 'test_logger.py')],
    'data_loader': [os.path.join('Tests', 'UnitTests', 'test_data_loader.py')],
    'all': [
        os.path.join('Tests', 'UnitTests', 'test_logger.py'),
        os.path.join('Tests', 'UnitTests', 'test_data_loader.py')
    ]
}


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--suite",
        action="store",
        default="all",
        help="Name of the test suite to run: " + ", ".join(test_suites.keys())
    )


@pytest.fixture(scope='session')
def test_suite(request):
    """Fixture to get the selected test suite"""
    suite_name = request.config.getoption("--suite")
    if suite_name not in test_suites:
        raise ValueError(f"Unknown test suite: {suite_name}")
    return test_suites[suite_name]


# Common fixtures that can be used across test files
@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.DEBUG)
