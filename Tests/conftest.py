import os
import pytest
from Utils.logger import Logger, LogLevel


def pytest_configure(config):
    """Initial pytest configuration"""
    pass


# Define test suites
test_suites = {
    'logger': [os.path.join('Tests', 'UnitTests', 'test_logger.py')],
    'data_loader': [os.path.join('Tests', 'UnitTests', 'test_data_loader.py')],
    'features': [
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py')
    ],
    'all': [
        os.path.join('Tests', 'UnitTests', 'test_logger.py'),
        os.path.join('Tests', 'UnitTests', 'test_data_loader.py'),
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py')
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
