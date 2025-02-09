import os
import sys
import pytest


# Add root directory to PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


def pytest_configure(config):
    """Initial pytest configuration"""
    config.option.cov_config = '.coveragerc'
    config.option.cov_branch = True


# Define test suites
test_suites = {
    'logger': [os.path.join('Tests', 'UnitTests', 'test_logger.py')],
    'all': [os.path.join('Tests', 'UnitTests', 'test_logger.py')]
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
