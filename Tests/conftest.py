import os
import pytest
from Utils.logger import Logger, LogLevel
from src.dataPreparation import DataLoader
from src.raw_agreement import RawAgreement
from src.cohen_kappa import CohenKappa
from src.fleiss_kappa import FleissKappa
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
    'fleiss_kappa': [
        os.path.join('Tests', 'UnitTests', 'test_fleiss_kappa.py'),
        os.path.join('Tests', 'FeatureTests', 'test_fleiss_kappa_real_data.py')
    ],
    'krippendorff_alpha': [
        os.path.join('Tests', 'UnitTests', 'test_krippendorff_alpha.py'),
        os.path.join('Tests',
                     'FeatureTests',
                     'test_krippendorff_alpha_real_data.py')
    ],
    'f_measure': [
        os.path.join('Tests', 'UnitTests', 'test_f_measure.py'),
        os.path.join('Tests', 'FeatureTests', 'test_f_measure_real_data.py')
    ],
    'pretty_print': [os.path.join('Tests',
                                  'UnitTests',
                                  'test_pretty_print.py')],
    'confidence_interval': [os.path.join('Tests',
                                         'UnitTests',
                                         'test_confidence_interval.py')],
    'features': [
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py'),
        os.path.join('Tests', 'FeatureTests', 'test_raw_agreement_feature.py'),
        os.path.join('Tests', 'FeatureTests', 'test_cohen_kappa_feature.py'),
        os.path.join('Tests',
                     'FeatureTests',
                     'test_fleiss_kappa_real_data.py'),
        os.path.join('Tests',
                     'FeatureTests',
                     'test_krippendorff_alpha_real_data.py'),
        os.path.join('Tests', 'FeatureTests', 'test_f_measure_real_data.py')
    ],
    'boundary_weighted_fleiss_kappa': [
        os.path.join('Tests',
                     'UnitTests',
                     'test_boundary_weighted_fleiss_kappa.py')
    ],
    'distance_based_cell_agreement': [
        os.path.join('Tests',
                     'UnitTests',
                     'test_distance_based_cell_agreement.py')
    ],
    'all': [
        os.path.join('Tests', 'UnitTests', 'test_logger.py'),
        os.path.join('Tests', 'UnitTests', 'test_data_loader.py'),
        os.path.join('Tests', 'UnitTests', 'test_raw_agreement.py'),
        os.path.join('Tests', 'UnitTests', 'test_cohen_kappa.py'),
        os.path.join('Tests', 'UnitTests', 'test_fleiss_kappa.py'),
        os.path.join('Tests', 'UnitTests', 'test_krippendorff_alpha.py'),
        os.path.join('Tests', 'UnitTests', 'test_f_measure.py'),
        os.path.join('Tests', 'UnitTests', 'test_confidence_interval.py'),
        os.path.join('Tests', 'UnitTests', 'test_pretty_print.py'),
        os.path.join('Tests', 'FeatureTests', 'test_reviews_loading.py'),
        os.path.join('Tests', 'FeatureTests', 'test_raw_agreement_feature.py'),
        os.path.join('Tests', 'FeatureTests', 'test_cohen_kappa_feature.py'),
        os.path.join('Tests',
                     'FeatureTests',
                     'test_fleiss_kappa_real_data.py'),
        os.path.join('Tests',
                     'FeatureTests',
                     'test_krippendorff_alpha_real_data.py'),
        os.path.join('Tests', 'FeatureTests', 'test_f_measure_real_data.py'),
        os.path.join('Tests',
                     'UnitTests',
                     'test_boundary_weighted_fleiss_kappa.py'),
        os.path.join('Tests',
                     'UnitTests',
                     'test_distance_based_cell_agreement.py')
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
def fleiss_kappa_calc(logger):
    """Fixture providing a FleissKappa instance."""
    return FleissKappa(logger)


@pytest.fixture
def ci_calc():
    """Fixture providing a ConfidenceIntervalCalculator instance."""
    return ConfidenceIntervalCalculator(confidence=0.95)


@pytest.fixture
def real_data(data_loader):
    """Fixture providing the real data from the Reviews_annotated.csv file."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir,
                             "Tests",
                             "Assets",
                             "Reviews_annotated.csv")

    # Load the data
    df = data_loader.load_data(data_file)
    return df


@pytest.fixture
def f_measure_calc(logger):
    """Fixture providing a FMeasure instance."""
    from src.f_measure import FMeasure
    return FMeasure(logger)


@pytest.fixture
def bwfk_calc(logger):
    """Fixture providing a BoundaryWeightedFleissKappa instance."""
    from src.boundary_weighted_fleiss_kappa import BoundaryWeightedFleissKappa
    return BoundaryWeightedFleissKappa(logger)


@pytest.fixture
def dbcaa_calc(logger):
    """Fixture providing a DistanceBasedCellAgreement instance."""
    from src.distance_based_cell_agreement import DistanceBasedCellAgreement
    return DistanceBasedCellAgreement(logger)


# Ajoutez cette ligne pour inclure explicitement le répertoire des tests
pytest_plugins = [
    "Tests.UnitTests.test_raw_agreement",
    "Tests.UnitTests.test_cohen_kappa",
    "Tests.UnitTests.test_fleiss_kappa",
    "Tests.UnitTests.test_krippendorff_alpha",
    "Tests.UnitTests.test_f_measure",
    "Tests.UnitTests.test_data_loader",
    "Tests.UnitTests.test_logger",
    "Tests.FeatureTests.test_krippendorff_alpha_real_data",
    "Tests.FeatureTests.test_f_measure_real_data",
    "Tests.UnitTests.test_boundary_weighted_fleiss_kappa",
    "Tests.UnitTests.test_distance_based_cell_agreement"
]
