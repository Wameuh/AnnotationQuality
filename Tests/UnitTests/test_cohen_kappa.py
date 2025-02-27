import pytest
import pandas as pd
import numpy as np
from src.cohen_kappa import CohenKappa
from Utils.logger import Logger, LogLevel


@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.DEBUG)


@pytest.fixture
def kappa_calc(logger):
    """Fixture providing a CohenKappa instance."""
    return CohenKappa(logger)


@pytest.fixture
def sample_df():
    """Fixture providing a sample DataFrame with annotator scores."""
    # Create a sample DataFrame with 5 reviews and 3 annotators
    data = {
        'Annotator1_score': [5, 4, 3, 2, 1],
        'Annotator2_score': [5, 4, 3, 2, 1],  # Perfect agreement with Annotator1
        'Annotator3_score': [5, 3, 3, 1, 1],  # Partial agreement with others
    }
    return pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])


@pytest.fixture
def sample_df_with_missing():
    """Fixture providing a sample DataFrame with missing values."""
    # Create a sample DataFrame with missing values
    data = {
        'Annotator1_score': [5, 4, 3, np.nan, 1],
        'Annotator2_score': [5, 4, np.nan, 2, 1],
        'Annotator3_score': [np.nan, 3, 3, 1, 1],
    }
    return pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])


def test_calculate_kappa_perfect_agreement(kappa_calc):
    """Test _calculate_kappa with perfect agreement."""
    # Create two arrays with perfect agreement
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4, 5])

    # Calculate kappa
    kappa = kappa_calc._calculate_kappa(a, b)

    # With perfect agreement, kappa should be 1.0
    assert kappa == 1.0


def test_calculate_kappa_chance_agreement(kappa_calc):
    """Test _calculate_kappa with agreement by chance."""
    # Create two arrays with no correlation (simulating chance agreement)
    # For simplicity, we'll use a contrived example where expected agreement equals observed
    a = np.array([1, 2, 1, 2, 1])
    b = np.array([1, 1, 2, 2, 1])

    # Calculate kappa
    kappa = kappa_calc._calculate_kappa(a, b)

    # Kappa should be close to 0 for chance agreement
    assert -0.25 <= kappa <= 0.25


def test_calculate_kappa_partial_agreement(kappa_calc):
    """Test _calculate_kappa with partial agreement."""
    # Create two arrays with partial agreement
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 3, 4])  # 3/5 exact matches

    # Calculate kappa
    kappa = kappa_calc._calculate_kappa(a, b)

    # Kappa should be between 0 and 1
    assert 0.0 < kappa < 1.0


def test_calculate_kappa_different_lengths(kappa_calc):
    """Test _calculate_kappa with arrays of different lengths."""
    # Create two arrays with different lengths
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4])

    # This should raise a ValueError
    with pytest.raises(ValueError):
        kappa_calc._calculate_kappa(a, b)


def test_calculate_pairwise(kappa_calc, sample_df):
    """Test calculate_pairwise with sample data."""
    # Calculate pairwise kappas
    kappas = kappa_calc.calculate_pairwise(sample_df)

    # Check that all pairs are in the result
    assert ('Annotator1', 'Annotator2') in kappas
    assert ('Annotator1', 'Annotator3') in kappas
    assert ('Annotator2', 'Annotator3') in kappas

    # Check kappa values
    assert kappas[('Annotator1', 'Annotator2')] == 1.0  # Perfect agreement
    assert 0.0 < kappas[('Annotator1', 'Annotator3')] < 1.0  # Partial agreement
    assert 0.0 < kappas[('Annotator2', 'Annotator3')] < 1.0  # Partial agreement


def test_calculate_pairwise_with_missing(kappa_calc, sample_df_with_missing):
    """Test calculate_pairwise with missing values."""
    # Calculate pairwise kappas
    kappas = kappa_calc.calculate_pairwise(sample_df_with_missing)

    # Check that all valid pairs are in the result
    assert ('Annotator1', 'Annotator2') in kappas
    assert ('Annotator1', 'Annotator3') in kappas
    assert ('Annotator2', 'Annotator3') in kappas

    # Check that missing values are handled correctly
    # For Annotator1 and Annotator2, there are 3 complete reviews (two have missing values)
    # For Annotator1 and Annotator3, there are 3 complete reviews (two have missing values)
    # For Annotator2 and Annotator3, there are 2 complete reviews (three have missing values)
    assert kappas[('Annotator1', 'Annotator2')] == 1.0  # Still perfect agreement


def test_get_kappa_statistics(kappa_calc, sample_df):
    """Test get_kappa_statistics."""
    # Calculate kappa statistics
    stats = kappa_calc.get_kappa_statistics(sample_df)

    # Check that all expected statistics are in the result
    assert 'average_kappa' in stats
    assert 'min_kappa' in stats
    assert 'max_kappa' in stats

    # Check that values are between -1 and 1 (theoretical range of kappa)
    assert -1.0 <= stats['average_kappa'] <= 1.0
    assert -1.0 <= stats['min_kappa'] <= 1.0
    assert -1.0 <= stats['max_kappa'] <= 1.0

    # Check that min <= average <= max
    assert stats['min_kappa'] <= stats['average_kappa'] <= stats['max_kappa']


def test_get_kappa_statistics_no_valid_pairs(kappa_calc):
    """Test get_kappa_statistics when there are no valid pairs."""
    # Create a DataFrame where no pair has complete reviews
    data = {
        'Annotator1_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Annotator2_score': [5, 4, 3, np.nan, np.nan],
        'Annotator3_score': [np.nan, np.nan, np.nan, 1, 1],
    }
    df = pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])

    # Calculate kappa statistics
    stats = kappa_calc.get_kappa_statistics(df)

    # Check that default values are returned
    assert stats['average_kappa'] == 0.0
    assert stats['min_kappa'] == 0.0
    assert stats['max_kappa'] == 0.0


def test_interpret_kappa(kappa_calc):
    """Test interpret_kappa with various kappa values."""
    # Test interpretation for each range
    assert "Poor agreement" in kappa_calc.interpret_kappa(-0.1)
    assert "Slight agreement" in kappa_calc.interpret_kappa(0.1)
    assert "Fair agreement" in kappa_calc.interpret_kappa(0.3)
    assert "Moderate agreement" in kappa_calc.interpret_kappa(0.5)
    assert "Substantial agreement" in kappa_calc.interpret_kappa(0.7)
    assert "Almost perfect agreement" in kappa_calc.interpret_kappa(0.9)


def test_calculate_kappa_edge_case(kappa_calc):
    """Test _calculate_kappa with edge case where expected agreement is 1."""
    # Create a case where all annotators use only one category
    a = np.array([1, 1, 1, 1, 1])
    b = np.array([1, 1, 1, 1, 1])

    # Calculate kappa
    kappa = kappa_calc._calculate_kappa(a, b)

    # When expected agreement is 1, kappa is undefined
    # Our implementation returns 1.0 if observed agreement is also 1
    assert kappa == 1.0

    # Now test with different values but same category
    a = np.array([1, 1, 1, 1, 1])
    b = np.array([2, 2, 2, 2, 2])

    # Calculate kappa
    kappa = kappa_calc._calculate_kappa(a, b)

    # When expected agreement is 1 but observed is 0, should return 0
    assert kappa == 0.0