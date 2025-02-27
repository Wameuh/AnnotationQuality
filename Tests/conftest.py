import os
import pytest
from Utils.logger import Logger, LogLevel
from src.dataPreparation import DataLoader
from src.raw_agreement import RawAgreement
from src.cohen_kappa import CohenKappa
from Utils.confident_interval import ConfidenceIntervalCalculator


def pytest_configure(config):
    """Initial pytest configuration"""
    pass


# Define test suites
test_suites = {
    'logger': [os.path.join('Tests', 'UnitTests', 'test_logger.py')],
    'data_loader': [os.path.join('Tests', 'UnitTests', 'test_data_loader.py')],
    'raw_agreement': [
        os.path.join('Tests', 'UnitTests', 'test_raw_agreement.py'),
        os.path.join('Tests', 'FeatureTests', 'test_raw_agreement_feature.py')
    ],
    'cohen_kappa': [
        os.path.join('Tests', 'UnitTests', 'test_cohen_kappa.py'),
        os.path.join('Tests', 'FeatureTests', 'test_cohen_kappa_feature.py')
    ],
    'pretty_print': [os.path.join('Tests', 'UnitTests', 'test_pretty_print.py')],
    'confidence_interval': [os.path.join('Tests', 'UnitTests', 'test_confidence_interval.py')],
    'features': [
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py'),
        os.path.join('Tests', 'FeatureTests', 'test_raw_agreement_feature.py'),
        os.path.join('Tests', 'FeatureTests', 'test_cohen_kappa_feature.py')
    ],
    'all': [
        os.path.join('Tests', 'UnitTests', 'test_logger.py'),
        os.path.join('Tests', 'UnitTests', 'test_data_loader.py'),
        os.path.join('Tests', 'UnitTests', 'test_raw_agreement.py'),
        os.path.join('Tests', 'UnitTests', 'test_cohen_kappa.py'),
        os.path.join('Tests', 'UnitTests', 'test_confidence_interval.py'),
        os.path.join('Tests', 'UnitTests', 'test_pretty_print.py'),
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py'),
        os.path.join('Tests', 'FeatureTests', 'test_raw_agreement_feature.py'),
        os.path.join('Tests', 'FeatureTests', 'test_cohen_kappa_feature.py')
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


@pytest.fixture
def data_loader(logger):
    """Fixture providing a DataLoader instance."""
    return DataLoader(logger)


@pytest.fixture
def agreement_calc(logger):
    """Fixture providing a RawAgreement instance."""
    return RawAgreement(logger)


@pytest.fixture
def kappa_calc(logger):
    """Fixture providing a CohenKappa instance."""
    return CohenKappa(logger)


@pytest.fixture
def ci_calc():
    """Fixture providing a ConfidenceIntervalCalculator instance."""
    return ConfidenceIntervalCalculator(confidence=0.95)
