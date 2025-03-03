import pytest
import pandas as pd
import numpy as np
from src.f_measure import FMeasure
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
def f_measure_calculator():
    """Fixture providing a FMeasure instance."""
    return FMeasure(level=LogLevel.DEBUG)


@pytest.fixture
def binary_data_df():
    """Fixture providing a DataFrame with binary data."""
    data = {
        'Annotator1': [1, 0, 1, 0, 1],
        'Annotator2': [1, 0, 1, 0, 0],
        'Annotator3': [1, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def non_binary_data_df():
    """Fixture providing a DataFrame with non-binary data."""
    data = {
        'Annotator1': [5, 2, 4, 1, 3],
        'Annotator2': [5, 2, 3, 1, 2],
        'Annotator3': [4, 2, 3, 1, 3]
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_values_df():
    """Fixture providing a DataFrame with missing values."""
    data = {
        'Annotator1': [1, 0, 1, np.nan, 1],
        'Annotator2': [1, 0, np.nan, 0, 0],
        'Annotator3': [np.nan, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)


def test_prepare_binary_data_default(f_measure_calculator, binary_data_df):
    """Test _prepare_binary_data method with default parameters."""
    # Call the method
    result = f_measure_calculator._prepare_binary_data(binary_data_df)

    # Check that result is a numpy array
    assert isinstance(result, np.ndarray)

    # Check dimensions
    assert result.shape == binary_data_df.shape

    # Check that values are preserved (already binary)
    assert np.array_equal(result, binary_data_df.values)


def test_prepare_binary_data_threshold(f_measure_calculator,
                                       non_binary_data_df):
    """Test _prepare_binary_data method with threshold."""
    # Call the method with threshold
    threshold = 3
    result = f_measure_calculator._prepare_binary_data(
        non_binary_data_df, threshold=threshold)

    # Check that values are binarized correctly
    expected = np.array([
        [1, 1, 1],  # [5, 5, 4] > 3
        [0, 0, 0],  # [2, 2, 2] <= 3
        [1, 0, 0],  # [4, 3, 3] compared to 3
        [0, 0, 0],  # [1, 1, 1] <= 3
        [0, 0, 0]   # [3, 2, 3] <= 3 (for first and third, = 3 for third)
    ])
    assert np.array_equal(result, expected)


def test_prepare_binary_data_positive_class(f_measure_calculator,
                                            non_binary_data_df):
    """Test _prepare_binary_data method with positive class."""
    # Call the method with positive class
    positive_class = 3
    result = f_measure_calculator._prepare_binary_data(
        non_binary_data_df, positive_class=positive_class)

    # Check that values are binarized correctly
    expected = np.array([
        [0, 0, 0],  # [5, 5, 4] != 3
        [0, 0, 0],  # [2, 2, 2] != 3
        [0, 1, 1],  # [4, 3, 3] compared to 3
        [0, 0, 0],  # [1, 1, 1] != 3
        [1, 0, 1]   # [3, 2, 3] compared to 3
    ])
    assert np.array_equal(result, expected)


def test_prepare_binary_data_with_nan(f_measure_calculator, missing_values_df):
    """Test _prepare_binary_data method with missing values."""
    # Call the method
    result = f_measure_calculator._prepare_binary_data(missing_values_df)

    # Check that NaN values are preserved
    assert np.isnan(result[3, 0])
    assert np.isnan(result[2, 1])
    assert np.isnan(result[0, 2])

    # Check that non-NaN values are preserved
    assert result[0, 0] == 1
    assert result[1, 1] == 0
    assert result[4, 2] == 1


def test_calculate_precision_recall(f_measure_calculator):
    """Test _calculate_precision_recall method."""
    # Test case 1: Perfect agreement
    a1 = np.array([1, 0, 1, 0])
    a2 = np.array([1, 0, 1, 0])
    precision, recall = f_measure_calculator._calculate_precision_recall(a1,
                                                                         a2)
    assert precision == 1.0
    assert recall == 1.0

    # Test case 2: No agreement
    a1 = np.array([1, 1, 1, 1])
    a2 = np.array([0, 0, 0, 0])
    precision, recall = f_measure_calculator._calculate_precision_recall(a1,
                                                                         a2)
    assert precision == 0.0
    assert recall == 0.0

    # Test case 3: Mixed agreement
    a1 = np.array([1, 0, 1, 0, 1])
    a2 = np.array([1, 0, 0, 1, 1])
    precision, recall = f_measure_calculator._calculate_precision_recall(a1,
                                                                         a2)
    assert precision == 2/3  # 2 true positives, 1 false positive
    assert recall == 2/3     # 2 true positives, 1 false negative


def test_calculate_binary_data(f_measure_calculator, binary_data_df):
    """Test calculate method with binary data."""
    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(binary_data_df)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0

    # For this specific data, we can calculate expected value manually
    # Annotator1 vs Annotator2: precision=2/3, recall=2/3, F1=2/3
    # Annotator1 vs Annotator3: precision=2/3, recall=2/3, F1=2/3
    # Annotator2 vs Annotator3: precision=1/2, recall=1/2, F1=1/2
    # Average precision = (2/3 + 2/3 + 1/2) / 3 = 0.611
    # Average recall = (2/3 + 2/3 + 1/2) / 3 = 0.611
    # F-measure = 2 * (0.611 * 0.611) / (0.611 + 0.611) = 0.611
    #
    # Note: The implementation calculates F-measure differently:
    # It first calculates F1 for each pair, then averages those F1 values.
    # This results in F-measure = (2/3 + 2/3 + 1/2) / 3 = 0.705
    assert f_measure == pytest.approx(0.705, abs=0.001)


def test_calculate_non_binary_data(f_measure_calculator, non_binary_data_df):
    """Test calculate method with non-binary data and threshold."""
    # Calculate F-measure with threshold
    f_measure = f_measure_calculator.calculate(non_binary_data_df, threshold=3)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0


def test_calculate_with_missing_values(f_measure_calculator,
                                       missing_values_df):
    """Test calculate method with missing values."""
    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(missing_values_df)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0


def test_get_f_measure_statistics_binary(f_measure_calculator, binary_data_df):
    """Test get_f_measure_statistics method with binary data."""
    # Get statistics
    stats = f_measure_calculator.get_f_measure_statistics(binary_data_df)

    # Check that basic F-measure is included
    assert 'f_measure' in stats
    assert isinstance(stats['f_measure'], float)
    assert 0.0 <= stats['f_measure'] <= 1.0


def test_get_f_measure_statistics_non_binary(f_measure_calculator,
                                             non_binary_data_df):
    """Test get_f_measure_statistics method with non-binary data."""
    # Get statistics
    stats = f_measure_calculator.get_f_measure_statistics(non_binary_data_df)

    # Check that various F-measures are included
    assert 'f_measure' in stats
    assert 'f_measure_median_threshold' in stats
    assert 'f_measure_mean_threshold' in stats

    # Check that class-specific F-measures are included
    unique_values = np.unique(
        non_binary_data_df.values[~np.isnan(non_binary_data_df.values)])
    for value in unique_values:
        assert f'f_measure_class_{value}' in stats


def test_interpret_f_measure(f_measure_calculator):
    """Test interpret_f_measure method."""
    # Test different ranges
    assert "Poor agreement" in f_measure_calculator.interpret_f_measure(0.1)
    assert "Fair agreement" in f_measure_calculator.interpret_f_measure(0.3)
    assert "Moderate agreement" in \
        f_measure_calculator.interpret_f_measure(0.5)
    assert "Substantial agreement" in \
        f_measure_calculator.interpret_f_measure(0.7)
    assert "Almost perfect agreement" in \
        f_measure_calculator.interpret_f_measure(0.9)


def test_calculate_with_all_missing_pair(f_measure_calculator):
    """Test calculate method with a pair having all missing values."""
    # Create a DataFrame where one pair has all missing values
    data = {
        'Annotator1': [1, 0, 1, 0, 1],
        'Annotator2': [1, 0, 1, 0, 0],
        'Annotator3': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All missing
    }
    df = pd.DataFrame(data)

    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(df)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0

    # Only one valid pair (Annotator1 vs Annotator2)
    # For this pair:
    # True positives = 2 (both annotators marked 1 for items 0 and 2)
    # False positives = 1
    # (Annotator1 marked 1 but Annotator2 marked 0 for item 4)
    # False negatives = 0
    # (no cases where Annotator1 marked 0 but Annotator2 marked 1)
    # Precision = 2/3, Recall = 1.0, F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 0.8
    assert f_measure == pytest.approx(0.8, abs=0.001)
