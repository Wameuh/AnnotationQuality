import pytest
import pandas as pd
import numpy as np
from src.icc import ICC
from Utils.logger import Logger, LogLevel


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the logger singleton before each test."""
    # Reset the singleton instance
    Logger._instance = None
    yield
    # Clean up after test
    Logger._instance = None


@pytest.fixture
def icc_calculator():
    """Fixture providing an ICC instance."""
    return ICC(level=LogLevel.DEBUG)


@pytest.fixture
def continuous_data():
    """Fixture providing continuous test data."""
    # Create a DataFrame with continuous data
    data = {
        'Annotator1': [4.5, 3.2, 5.0, 2.8, 4.1],
        'Annotator2': [4.2, 3.0, 4.8, 3.0, 3.9],
        'Annotator3': [4.7, 3.5, 5.2, 2.5, 4.3]
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_data():
    """Fixture providing data with missing values."""
    # Create a DataFrame with missing values
    data = {
        'Annotator1': [4.5, 3.2, np.nan, 2.8, 4.1],
        'Annotator2': [4.2, np.nan, 4.8, 3.0, 3.9],
        'Annotator3': [np.nan, 3.5, 5.2, 2.5, 4.3]
    }
    return pd.DataFrame(data)


def test_calculate_with_continuous_data(icc_calculator, continuous_data):
    """Test calculate method with continuous data."""
    # Calculate ICC with default form (2,1)
    icc = icc_calculator.calculate(continuous_data)

    # Check that result is a float between 0 and 1
    assert isinstance(icc, float)
    assert 0.0 <= icc <= 1.0

    # For this specific data, ICC should be high (good agreement)
    assert icc > 0.7


def test_calculate_with_different_forms(icc_calculator, continuous_data):
    """Test calculate method with different ICC forms."""
    # Calculate ICC with different forms
    icc_1_1 = icc_calculator.calculate(continuous_data, form='1,1')
    icc_2_1 = icc_calculator.calculate(continuous_data, form='2,1')
    icc_3_1 = icc_calculator.calculate(continuous_data, form='3,1')
    icc_1_k = icc_calculator.calculate(continuous_data, form='1,k')
    icc_2_k = icc_calculator.calculate(continuous_data, form='2,k')
    icc_3_k = icc_calculator.calculate(continuous_data, form='3,k')

    # Check that all results are floats between 0 and 1
    for icc in [icc_1_1, icc_2_1, icc_3_1, icc_1_k, icc_2_k, icc_3_k]:
        assert isinstance(icc, float)
        assert 0.0 <= icc <= 1.0

    # Average measures (k) should generally be higher than single measures (1)
    assert icc_1_k >= icc_1_1
    assert icc_2_k >= icc_2_1
    assert icc_3_k >= icc_3_1


def test_calculate_with_missing_data(icc_calculator, missing_data):
    """Test calculate method with missing data."""
    # Calculate ICC
    icc = icc_calculator.calculate(missing_data)

    # Check that result is a float between 0 and 1
    assert isinstance(icc, float)
    assert 0.0 <= icc <= 1.0


def test_calculate_with_empty_data(icc_calculator):
    """Test calculate method with empty data."""
    # Create an empty DataFrame
    empty_df = pd.DataFrame()

    # Calculate ICC
    icc = icc_calculator.calculate(empty_df)

    # Should return 0.0 for empty data
    assert icc == 0.0


def test_calculate_with_insufficient_subjects(icc_calculator):
    """Test calculate method with insufficient subjects."""
    # Create a DataFrame with only one subject
    data = {
        'Annotator1': [4.5],
        'Annotator2': [4.2],
        'Annotator3': [4.7]
    }
    df = pd.DataFrame(data)

    # Calculate ICC
    icc = icc_calculator.calculate(df)

    # Should return 0.0 for insufficient subjects
    assert icc == 0.0


def test_calculate_with_insufficient_raters(icc_calculator):
    """Test calculate method with insufficient raters."""
    # Create a DataFrame with only one rater
    data = {
        'Annotator1': [4.5, 3.2, 5.0, 2.8, 4.1]
    }
    df = pd.DataFrame(data)

    # Calculate ICC
    icc = icc_calculator.calculate(df)

    # Should return 0.0 for insufficient raters
    assert icc == 0.0


def test_interpret_icc(icc_calculator):
    """Test interpret_icc method."""
    # Test different ranges
    assert "Slight agreement" in icc_calculator.interpret_icc(0.1)
    assert "Fair agreement" in icc_calculator.interpret_icc(0.3)
    assert "Moderate agreement" in icc_calculator.interpret_icc(0.5)
    assert "Substantial agreement" in icc_calculator.interpret_icc(0.7)
    assert "Almost perfect agreement" in icc_calculator.interpret_icc(0.9)


def test_get_icc_statistics(icc_calculator, continuous_data):
    """Test get_icc_statistics method."""
    # Get statistics
    stats = icc_calculator.get_icc_statistics(continuous_data)

    # Check that result is a dictionary with the expected keys
    assert isinstance(stats, dict)
    assert 'icc' in stats
    assert 'interpretation' in stats

    # Check that ICC value is a float between 0 and 1
    assert isinstance(stats['icc'], float)
    assert 0.0 <= stats['icc'] <= 1.0

    # Check that interpretation is a string
    assert isinstance(stats['interpretation'], str)


def test_get_icc_statistics_with_form(icc_calculator, continuous_data):
    """Test get_icc_statistics method with specific form."""
    # Get statistics with form='3,k'
    stats = icc_calculator.get_icc_statistics(continuous_data, form='3,k')

    # Check that result is a dictionary with the expected keys
    assert isinstance(stats, dict)
    assert 'icc' in stats
    assert 'interpretation' in stats

    # Check that ICC value is a float between 0 and 1
    assert isinstance(stats['icc'], float)
    assert 0.0 <= stats['icc'] <= 1.0

    # Check that interpretation is a string
    assert isinstance(stats['interpretation'], str)


def test_calculate_with_all_missing_rows(icc_calculator):
    """Test calculate method when all rows have at least one missing value."""
    # Create a DataFrame where every row has at least one missing value
    # and no row is complete
    data = {
        'Annotator1': [4.5, np.nan, 5.0],
        'Annotator2': [np.nan, 3.0, np.nan],
        'Annotator3': [4.7, np.nan, np.nan]
    }
    df = pd.DataFrame(data)

    # Calculate ICC
    icc = icc_calculator.calculate(df)

    # Should return 0.0 when no complete cases are available
    assert icc == 0.0


def test_calculate_with_unknown_form(icc_calculator, continuous_data):
    """Test calculate method with an unknown ICC form."""
    # Calculate ICC with an invalid form
    icc = icc_calculator.calculate(continuous_data, form='invalid_form')

    # Should return 0.0 for unknown form
    assert icc == 0.0


def test_calculate_pairwise(icc_calculator, continuous_data):
    """Test calculate_pairwise method."""
    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(continuous_data)

    # Check that result is a dictionary
    assert isinstance(pairwise_iccs, dict)

    # Check that all pairs are present
    expected_pairs = [
        ('Annotator1', 'Annotator2'),
        ('Annotator1', 'Annotator3'),
        ('Annotator2', 'Annotator3')
    ]
    for pair in expected_pairs:
        assert pair in pairwise_iccs

    # Check that ICC values are between 0 and 1
    for pair, icc in pairwise_iccs.items():
        assert 0.0 <= icc <= 1.0


def test_calculate_pairwise_with_missing_data(icc_calculator, missing_data):
    """Test calculate_pairwise method with missing data."""
    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(missing_data)

    # Check that result is a dictionary
    assert isinstance(pairwise_iccs, dict)

    # Check that all pairs are present (even with missing data)
    expected_pairs = [
        ('Annotator1', 'Annotator2'),
        ('Annotator1', 'Annotator3'),
        ('Annotator2', 'Annotator3')
    ]
    for pair in expected_pairs:
        assert pair in pairwise_iccs

    # Check that ICC values are between 0 and 1
    for pair, icc in pairwise_iccs.items():
        assert 0.0 <= icc <= 1.0


def test_calculate_pairwise_with_empty_data(icc_calculator):
    """Test calculate_pairwise method with empty data."""
    # Create an empty DataFrame
    empty_df = pd.DataFrame()

    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(empty_df)

    # Should return an empty dictionary for empty data
    assert isinstance(pairwise_iccs, dict)
    assert len(pairwise_iccs) == 0


def test_calculate_pairwise_with_single_annotator(icc_calculator):
    """Test calculate_pairwise method with a single annotator."""
    # Create a DataFrame with only one annotator
    data = {
        'Annotator1': [4.5, 3.2, 5.0, 2.8, 4.1]
    }
    df = pd.DataFrame(data)

    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(df)

    # Should return an empty dictionary (no pairs possible)
    assert isinstance(pairwise_iccs, dict)
    assert len(pairwise_iccs) == 0


def test_calculate_pairwise_with_different_forms(icc_calculator,
                                                 continuous_data):
    """Test calculate_pairwise method with different ICC forms."""
    # Calculate pairwise ICCs with different forms
    forms = ['1,1', '2,1', '3,1', '1,k', '2,k', '3,k']

    for form in forms:
        pairwise_iccs = icc_calculator.calculate_pairwise(continuous_data,
                                                          form=form)

        # Check that result is a dictionary
        assert isinstance(pairwise_iccs, dict)

        # Check that all pairs are present
        expected_pairs = [
            ('Annotator1', 'Annotator2'),
            ('Annotator1', 'Annotator3'),
            ('Annotator2', 'Annotator3')
        ]
        for pair in expected_pairs:
            assert pair in pairwise_iccs

        # Check that ICC values are between 0 and 1
        for pair, icc in pairwise_iccs.items():
            assert 0.0 <= icc <= 1.0


def test_calculate_pairwise_with_all_missing_data(icc_calculator):
    """
    Test calculate_pairwise method with data where all values are missing
    for a pair.
    """
    # Create a DataFrame with missing values for one pair
    data = {
        'Annotator1': [1, 2, 3, 4, 5],
        'Annotator2': [1, 2, 3, 4, 5],
        'Annotator3': [np.nan, np.nan, np.nan, np.nan, np.nan]
        # All values are NaN
    }
    df = pd.DataFrame(data)

    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_iccs, dict)

    # Check that pairs with valid data are present
    assert ('Annotator1', 'Annotator2') in pairwise_iccs

    # Check that pairs with all missing data are not present
    assert ('Annotator1', 'Annotator3') not in pairwise_iccs
    assert ('Annotator2', 'Annotator3') not in pairwise_iccs


def test_calculate_pairwise_with_exception(icc_calculator, monkeypatch):
    """Test calculate_pairwise method when calculate raises an exception."""
    # Create a DataFrame with valid data
    data = {
        'Annotator1': [1, 2, 3, 4, 5],
        'Annotator2': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    # Define a mock calculate method that raises an exception
    def mock_calculate(*args, **kwargs):
        raise ValueError("Forced error in calculate method")

    # Replace the calculate method with our mock
    monkeypatch.setattr(icc_calculator, "calculate", mock_calculate)

    # Calculate pairwise ICCs
    pairwise_iccs = icc_calculator.calculate_pairwise(df)

    # Check that result is an empty dictionary (no successful calculations)
    assert isinstance(pairwise_iccs, dict)
    assert len(pairwise_iccs) == 0

    # Restore the original calculate method (monkeypatch does this
    # automatically)
