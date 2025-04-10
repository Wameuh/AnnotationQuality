import os
import pytest
from src.dataPreparation import DataLoader
from src.fleiss_kappa import FleissKappa
from Utils.logger import Logger, LogLevel


@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.DEBUG)


@pytest.fixture
def data_loader(logger):
    """Fixture providing a DataLoader instance."""
    return DataLoader(logger)


@pytest.fixture
def fleiss_kappa_calc(logger):
    """Fixture providing a FleissKappa instance."""
    return FleissKappa(logger)


@pytest.fixture
def real_data(data_loader):
    """Fixture providing the real data from the Reviews_annotated.csv file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(base_dir,
                             "Tests",
                             "Assets",
                             "Reviews_annotated.csv")

    # Load the data
    df = data_loader.load_data(data_file)
    return df


def test_fleiss_kappa_overall(fleiss_kappa_calc, real_data):
    """Test that the overall Fleiss' Kappa matches the expected value."""
    # Calculate overall Fleiss' Kappa
    kappa = fleiss_kappa_calc.calculate(real_data)

    # Check that the overall Kappa matches the expected value
    expected_kappa = 0.7309
    assert round(kappa, 4) == expected_kappa

    # Check the interpretation
    interpretation = fleiss_kappa_calc.interpret_kappa(kappa)
    assert "Substantial agreement" in interpretation


def test_fleiss_kappa_by_category(fleiss_kappa_calc, real_data):
    """Test that the Fleiss' Kappa by category matches the expected values."""
    # Calculate Fleiss' Kappa by category
    kappas_by_category = fleiss_kappa_calc.calculate_by_category(real_data)

    # Expected values for each category
    expected_kappas = {
        1: 0.7768,  # Substantial agreement
        2: 0.5916,  # Moderate agreement
        3: 0.6594,  # Substantial agreement
        4: 0.6515,  # Substantial agreement
        5: 0.8324,  # Almost perfect agreement
    }

    # Check that we have the expected number of categories
    assert len(kappas_by_category) == len(expected_kappas)

    # Check that each category's Kappa matches the expected value
    for category, expected_kappa in expected_kappas.items():
        assert category in kappas_by_category
        assert round(kappas_by_category[category], 4) == expected_kappa

        # Check the interpretation
        interpretation = fleiss_kappa_calc.interpret_kappa(
            kappas_by_category[category])
        if category == 1:
            assert "Substantial agreement" in interpretation
        elif category == 2:
            assert "Moderate agreement" in interpretation
        elif category == 3:
            assert "Substantial agreement" in interpretation
        elif category == 4:
            assert "Substantial agreement" in interpretation
        elif category == 5:
            assert "Almost perfect agreement" in interpretation


def test_fleiss_kappa_results_consistency(fleiss_kappa_calc, real_data):
    """
    Test that the overall Kappa and category Kappas match expected values.
    """
    # Calculate overall Fleiss' Kappa
    overall_kappa = fleiss_kappa_calc.calculate(real_data)

    # Calculate Fleiss' Kappa by category
    kappas_by_category = fleiss_kappa_calc.calculate_by_category(real_data)

    # Check that the overall Kappa matches the expected value
    assert round(overall_kappa, 4) == 0.7309

    # Check that the category 5 Kappa matches the expected value
    assert round(kappas_by_category[5], 4) == 0.8324

    # The overall Kappa is different from the category 5 Kappa
    # This is expected because the overall Kappa considers all categories
    # together
