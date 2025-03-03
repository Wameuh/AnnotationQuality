import pytest
import pandas as pd
import numpy as np
from src.krippendorff_alpha import KrippendorffAlpha
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
def alpha_calculator():
    """Fixture providing a KrippendorffAlpha instance."""
    return KrippendorffAlpha(level=LogLevel.DEBUG)


@pytest.fixture
def perfect_agreement_df():
    """Fixture providing a DataFrame with perfect agreement."""
    data = {
        'Annotator1': [5, 4, 3, 2, 1],
        'Annotator2': [5, 4, 3, 2, 1],
        'Annotator3': [5, 4, 3, 2, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def high_agreement_df():
    """Fixture providing a DataFrame with high agreement."""
    data = {
        'Annotator1': [5, 4, 3, 2, 1],
        'Annotator2': [5, 4, 3, 2, 1],
        'Annotator3': [5, 4, 3, 1, 1]  # One disagreement
    }
    return pd.DataFrame(data)


@pytest.fixture
def moderate_agreement_df():
    """Fixture providing a DataFrame with moderate agreement."""
    data = {
        'Annotator1': [5, 4, 3, 2, 1],
        'Annotator2': [5, 4, 2, 2, 1],
        'Annotator3': [4, 4, 3, 1, 1]  # Several disagreements
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_values_df():
    """Fixture providing a DataFrame with missing values."""
    data = {
        'Annotator1': [5, 4, 3, np.nan, 1],
        'Annotator2': [5, 4, np.nan, 2, 1],
        'Annotator3': [np.nan, 4, 3, 2, 1]
    }
    return pd.DataFrame(data)


def test_prepare_data(alpha_calculator, high_agreement_df):
    """Test _prepare_data method."""
    # Call the method
    result = alpha_calculator._prepare_data(high_agreement_df)

    # Check that result is a numpy array
    assert isinstance(result, np.ndarray)

    # Check dimensions
    assert result.shape == high_agreement_df.shape

    # Create a DataFrame with NaN values
    df_with_nan = high_agreement_df.copy()
    df_with_nan.iloc[0, 0] = np.nan

    # Call the method with NaN values
    result_with_nan = alpha_calculator._prepare_data(df_with_nan)

    # Check that NaN is replaced with None
    assert result_with_nan[0, 0] is None


def test_coincidence_matrix(alpha_calculator):
    """Test _coincidence_matrix method."""
    # Create a simple test data matrix
    data = np.array([
        [1, 1, 1],  # All annotators agree on value 1
        [2, 2, 2],  # All annotators agree on value 2
        [1, 2, 3],  # All annotators disagree
        [None, 1, 1]  # Two annotators agree, one missing
    ])

    # Call the method
    result = alpha_calculator._coincidence_matrix(data)

    # Check that result is a numpy array
    assert isinstance(result, np.ndarray)

    # Check dimensions (should be n_unique_values x n_unique_values)
    assert result.shape == (3, 3)  # 3 unique values: 1, 2, 3

    # Check specific values
    # Diagonal elements should be higher (agreements)
    assert result[0, 0] > 0  # Agreements on value 1
    assert result[1, 1] > 0  # Agreements on value 2

    # Off-diagonal elements should represent disagreements
    assert result[0, 1] > 0  # Disagreements between values 1 and 2
    assert result[0, 2] > 0  # Disagreements between values 1 and 3
    assert result[1, 2] > 0  # Disagreements between values 2 and 3


def test_calculate_distance(alpha_calculator):
    """Test _calculate_distance method."""
    # Test nominal metric
    assert alpha_calculator._calculate_distance(1, 1, 'nominal') == 0.0
    # Same values
    assert alpha_calculator._calculate_distance(1, 2, 'nominal') == 1.0
    # Different values

    # Test ordinal metric
    assert alpha_calculator._calculate_distance(1, 1, 'ordinal') == 0.0
    # Same values
    assert alpha_calculator._calculate_distance(1, 3, 'ordinal') == 4.0
    # |1-3|^2 = 4

    # Test interval/ratio metrics
    assert alpha_calculator._calculate_distance(1, 1, 'interval') == 0.0
    # Same values
    assert alpha_calculator._calculate_distance(1, 3, 'interval') == 4.0
    # (1-3)^2 = 4
    assert alpha_calculator._calculate_distance(1, 3, 'ratio') == 4.0
    # Same as interval

    # Test unknown metric (should default to nominal)
    assert alpha_calculator._calculate_distance(1, 1, 'unknown') == 0.0
    assert alpha_calculator._calculate_distance(1, 2, 'unknown') == 1.0


def test_observed_disagreement(alpha_calculator):
    """Test _observed_disagreement method."""
    # Create a simple test data matrix
    data = np.array([
        [1, 1, 1],  # All annotators agree on value 1
        [2, 2, 2],  # All annotators agree on value 2
        [1, 2, 3],  # All annotators disagree
    ])

    # Call the method with nominal metric
    result = alpha_calculator._observed_disagreement(data, 'nominal')

    # Check that result is a float
    assert isinstance(result, float)

    # Check that result is between 0 and 1
    assert 0.0 <= result <= 1.0

    # For perfect agreement data, observed disagreement should be 0
    perfect_data = np.array([
        [1, 1, 1],
        [2, 2, 2],
    ])
    perfect_result = alpha_calculator._observed_disagreement(perfect_data,
                                                             'nominal')
    assert perfect_result == 0.0


def test_observed_disagreement_no_coincidences(alpha_calculator):
    """Test _observed_disagreement method when no coincidences are found."""
    # Create a data matrix where each item has only one annotation
    # This will result in no coincidences
    data = np.array([
        [1, None, None],
        [None, 2, None],
        [None, None, 3]
    ])

    # Call the method
    result = alpha_calculator._observed_disagreement(data, 'nominal')

    # When no coincidences are found, observed disagreement should be 0.0
    assert result == 0.0


def test_expected_disagreement(alpha_calculator):
    """Test _expected_disagreement method."""
    # Create a simple test data matrix
    data = np.array([
        [1, 1, 1],  # All annotators agree on value 1
        [2, 2, 2],  # All annotators agree on value 2
        [1, 2, 3],  # All annotators disagree
    ])

    # Call the method with nominal metric
    result = alpha_calculator._expected_disagreement(data, 'nominal')

    # Check that result is a float
    assert isinstance(result, float)

    # Check that result is between 0 and 1
    assert 0.0 <= result <= 1.0

    # For data with only one value, expected disagreement should be 0
    single_value_data = np.array([
        [1, 1, 1],
        [1, 1, 1],
    ])
    single_value_result = alpha_calculator._expected_disagreement(
        single_value_data, 'nominal')
    assert single_value_result == 0.0


def test_expected_disagreement_no_valid_data(alpha_calculator):
    """Test _expected_disagreement method when no valid data is found."""
    # Create a data matrix with only None values
    # This will result in no valid data for expected disagreement
    data = np.array([
        [None, None, None],
        [None, None, None],
        [None, None, None]
    ])

    # Call the method
    result = alpha_calculator._expected_disagreement(data, 'nominal')

    # When no valid data is found, expected disagreement should be 0.0
    assert result == 0.0


def test_calculate_perfect_agreement(alpha_calculator, perfect_agreement_df):
    """Test calculate method with perfect agreement."""
    # Calculate alpha with nominal metric
    alpha = alpha_calculator.calculate(perfect_agreement_df, 'nominal')

    # For perfect agreement, alpha should be 1.0
    assert alpha == 1.0

    # Test with other metrics
    assert alpha_calculator.calculate(perfect_agreement_df, 'ordinal') == 1.0
    assert alpha_calculator.calculate(perfect_agreement_df, 'interval') == 1.0
    assert alpha_calculator.calculate(perfect_agreement_df, 'ratio') == 1.0


def test_calculate_high_agreement(alpha_calculator, high_agreement_df):
    """Test calculate method with high agreement."""
    # Calculate alpha with nominal metric
    alpha = alpha_calculator.calculate(high_agreement_df, 'nominal')

    # For high agreement, alpha should be close to 1.0
    assert 0.8 <= alpha <= 1.0

    # Test with other metrics
    ordinal_alpha = alpha_calculator.calculate(high_agreement_df, 'ordinal')
    assert 0.8 <= ordinal_alpha <= 1.0


def test_calculate_moderate_agreement(alpha_calculator, moderate_agreement_df):
    """Test calculate method with moderate agreement."""
    # Calculate alpha with nominal metric
    alpha = alpha_calculator.calculate(moderate_agreement_df, 'nominal')

    # For moderate agreement, alpha should be between 0.4 and 0.8
    assert 0.4 <= alpha <= 0.8


def test_calculate_with_missing_values(alpha_calculator, missing_values_df):
    """Test calculate method with missing values."""
    # Calculate alpha with nominal metric
    alpha = alpha_calculator.calculate(missing_values_df, 'nominal')

    # Alpha should be a float between -1 and 1
    assert isinstance(alpha, float)
    assert -1.0 <= alpha <= 1.0


def test_interpret_alpha(alpha_calculator):
    """Test interpret_alpha method."""
    # Test different ranges
    assert "Poor agreement" in alpha_calculator.interpret_alpha(-0.1)
    assert "Slight agreement" in alpha_calculator.interpret_alpha(0.1)
    assert "Fair agreement" in alpha_calculator.interpret_alpha(0.3)
    assert "Moderate agreement" in alpha_calculator.interpret_alpha(0.5)
    assert "Substantial agreement" in alpha_calculator.interpret_alpha(0.7)
    assert "Almost perfect agreement" in alpha_calculator.interpret_alpha(0.9)


def test_get_alpha_statistics(alpha_calculator, high_agreement_df):
    """Test get_alpha_statistics method."""
    # Get statistics
    stats = alpha_calculator.get_alpha_statistics(high_agreement_df)

    # Check that all metrics are included
    assert 'alpha_nominal' in stats
    assert 'alpha_ordinal' in stats
    assert 'alpha_interval' in stats
    assert 'alpha_ratio' in stats

    # Check that interpretations are included
    assert 'interpretation_nominal' in stats
    assert 'interpretation_ordinal' in stats
    assert 'interpretation_interval' in stats
    assert 'interpretation_ratio' in stats

    # Check that values are in the expected range
    for metric in ['nominal', 'ordinal', 'interval', 'ratio']:
        assert -1.0 <= stats[f'alpha_{metric}'] <= 1.0
        assert isinstance(stats[f'interpretation_{metric}'], str)


def test_calculate_with_zero_expected_disagreement(alpha_calculator):
    """Test calculate method when expected disagreement is zero."""
    # Create a DataFrame where all annotators use the same value for all items
    # This will result in expected_disagreement = 0
    data = {
        'Annotator1': [3, 3, 3, 3, 3],
        'Annotator2': [3, 3, 3, 3, 3],
        'Annotator3': [3, 3, 3, 3, 3]
    }
    df = pd.DataFrame(data)

    # Calculate alpha
    alpha = alpha_calculator.calculate(df, 'nominal')

    # When expected disagreement is zero, alpha should be 1.0
    assert alpha == 1.0

    # Test with other metrics as well
    assert alpha_calculator.calculate(df, 'ordinal') == 1.0
    assert alpha_calculator.calculate(df, 'interval') == 1.0
    assert alpha_calculator.calculate(df, 'ratio') == 1.0
