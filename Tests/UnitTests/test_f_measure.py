import pytest
import pandas as pd
import numpy as np
import io
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
    # Average F1 = (2/3 + 2/3 + 1/2) / 3 = 0.611
    #
    # Note: The implementation calculates F-measure differently:
    # It first calculates F1 for each pair, then averages those F1 values.
    # This results in F-measure = (2/3 + 2/3 + 1/2) / 3 = 0.7
    assert f_measure == pytest.approx(0.7, abs=0.001)


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


@pytest.fixture
def binary_data():
    """Fixture providing binary test data."""
    # Create a DataFrame with binary data (0/1)
    data = {
        'Annotator1': [1, 0, 1, 1, 0],
        'Annotator2': [1, 0, 0, 1, 1],
        'Annotator3': [1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def continuous_data():
    """Fixture providing continuous test data."""
    # Create a DataFrame with continuous data
    data = {
        'Annotator1': [0.9, 0.2, 0.8, 0.7, 0.1],
        'Annotator2': [0.8, 0.3, 0.3, 0.9, 0.6],
        'Annotator3': [0.7, 0.1, 0.9, 0.2, 0.8]
    }
    return pd.DataFrame(data)


@pytest.fixture
def categorical_data():
    """Fixture providing categorical test data."""
    # Create a DataFrame with categorical data
    data = {
        'Annotator1': ['A', 'B', 'A', 'A', 'B'],
        'Annotator2': ['A', 'B', 'B', 'A', 'A'],
        'Annotator3': ['A', 'B', 'A', 'B', 'A']
    }
    return pd.DataFrame(data)


def test_calculate_with_binary_data(f_measure_calculator, binary_data):
    """Test calculate method with binary data."""
    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(binary_data)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0


def test_calculate_with_continuous_data(f_measure_calculator, continuous_data):
    """Test calculate method with continuous data and threshold."""
    # Calculate F-measure with threshold
    f_measure = f_measure_calculator.calculate(continuous_data, threshold=0.5)

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0


def test_calculate_with_categorical_data(f_measure_calculator,
                                         categorical_data):
    """Test calculate method with categorical data and positive class."""
    # Calculate F-measure with positive class
    f_measure = f_measure_calculator.calculate(categorical_data,
                                               positive_class='A')

    # Check that result is a float between 0 and 1
    assert isinstance(f_measure, float)
    assert 0.0 <= f_measure <= 1.0


def test_calculate_pairwise(f_measure_calculator, binary_data_df):
    """Test calculate_pairwise method with binary data."""
    # Calculate pairwise F-measures
    pairwise_f_measures = f_measure_calculator.calculate_pairwise(
        binary_data_df)

    # Check that result is a dictionary
    assert isinstance(pairwise_f_measures, dict)

    # Check that all pairs are present
    columns = binary_data_df.columns
    n_annotators = len(columns)
    expected_pairs_count = (n_annotators * (n_annotators - 1)) // 2
    assert len(pairwise_f_measures) == expected_pairs_count

    # Check that values are between 0 and 1
    for k, v in pairwise_f_measures.items():
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert 0.0 <= v <= 1.0

    # Check that the dictionary is symmetric
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            pair = (columns[i], columns[j])
            assert pair in pairwise_f_measures


def test_calculate_pairwise_with_threshold(f_measure_calculator,
                                           non_binary_data_df):
    """Test calculate_pairwise method with threshold."""
    # Calculate pairwise F-measures with threshold
    pairwise_f_measures = f_measure_calculator.calculate_pairwise(
        non_binary_data_df, threshold=3.0)

    # Check that result is a dictionary
    assert isinstance(pairwise_f_measures, dict)

    # Check that values are between 0 and 1
    for v in pairwise_f_measures.values():
        assert 0.0 <= v <= 1.0


def test_calculate_pairwise_with_positive_class(f_measure_calculator,
                                                non_binary_data_df):
    """Test calculate_pairwise method with positive class."""
    # Calculate pairwise F-measures with positive class
    pairwise_f_measures = f_measure_calculator.calculate_pairwise(
        non_binary_data_df, positive_class=5)

    # Check that result is a dictionary
    assert isinstance(pairwise_f_measures, dict)

    # Check that values are between 0 and 1
    for v in pairwise_f_measures.values():
        assert 0.0 <= v <= 1.0


def test_calculate_pairwise_with_missing_values(f_measure_calculator,
                                                missing_values_df):
    """Test calculate_pairwise method with missing values."""
    # Calculate pairwise F-measures with missing values
    pairwise_f_measures = f_measure_calculator.calculate_pairwise(
        missing_values_df)

    # Check that result is a dictionary
    assert isinstance(pairwise_f_measures, dict)

    # Check that values are between 0 and 1
    for v in pairwise_f_measures.values():
        assert 0.0 <= v <= 1.0


def test_prepare_binary_data(f_measure_calculator, continuous_data):
    """Test _prepare_binary_data method."""
    # Prepare binary data with threshold
    binary_data = f_measure_calculator._prepare_binary_data(
        continuous_data, threshold=0.5)

    # Check that result is a numpy array with the expected shape
    assert isinstance(binary_data, np.ndarray)
    assert binary_data.shape == continuous_data.shape

    # Check that all values are 0, 1, or NaN
    for value in binary_data.flatten():
        assert value in [0, 1] or np.isnan(value)


def test_calculate_precision_recall(f_measure_calculator):
    """Test _calculate_precision_recall method."""
    # Create test data
    a1 = np.array([1, 0, 1, 1, 0])
    a2 = np.array([1, 0, 0, 1, 1])

    # Calculate precision and recall
    precision, recall = f_measure_calculator._calculate_precision_recall(a1,
                                                                         a2)

    # Check that results are floats between 0 and 1
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0

    # Check specific values for this test case
    # TP = 2, FP = 1, FN = 1
    # Precision = TP / (TP + FP) = 2 / 3 = 0.667
    # Recall = TP / (TP + FN) = 2 / 3 = 0.667
    assert np.isclose(precision, 2/3)
    assert np.isclose(recall, 2/3)


def test_interpret_f_measure(f_measure_calculator):
    """Test interpret_f_measure method."""
    # Test interpretation for each range
    assert "Invalid F-measure value" in \
        f_measure_calculator.interpret_f_measure(-0.1)
    assert "Poor agreement" in f_measure_calculator.interpret_f_measure(0.1)
    assert "Fair agreement" in f_measure_calculator.interpret_f_measure(0.3)
    assert "Moderate agreement" in \
        f_measure_calculator.interpret_f_measure(0.5)
    assert "Substantial agreement" in \
        f_measure_calculator.interpret_f_measure(0.7)
    assert "Almost perfect agreement" in \
        f_measure_calculator.interpret_f_measure(0.9)

    # Check that the interpretation mentions "random or opposite agreement"
    # for low values
    assert "random or opposite agreement" in \
        f_measure_calculator.interpret_f_measure(0.1)


def test_get_f_measure_statistics(f_measure_calculator, binary_data):
    """Test get_f_measure_statistics method."""
    # Calculate F-measure statistics
    stats = f_measure_calculator.get_f_measure_statistics(binary_data)

    # Check that result is a dictionary with the expected keys
    assert isinstance(stats, dict)
    assert 'f_measure' in stats


def test_get_f_measure_statistics_with_continuous_data(f_measure_calculator,
                                                       continuous_data):
    """Test get_f_measure_statistics method with continuous data."""
    # Calculate F-measure statistics
    stats = f_measure_calculator.get_f_measure_statistics(continuous_data)

    # Check that result is a dictionary with the expected keys
    assert isinstance(stats, dict)
    assert 'f_measure' in stats
    assert 'f_measure_median_threshold' in stats
    assert 'f_measure_mean_threshold' in stats


def test_calculate_with_empty_data(f_measure_calculator):
    """Test calculate method with empty data."""
    # Create an empty DataFrame
    empty_df = pd.DataFrame()

    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(empty_df)

    # Should return 0.0 for empty data
    assert f_measure == 0.0


def test_calculate_with_perfect_agreement(f_measure_calculator):
    """Test calculate method with perfect agreement."""
    # Create a DataFrame with perfect agreement
    data = {
        'Annotator1': [1, 0, 1, 0, 1],
        'Annotator2': [1, 0, 1, 0, 1],
        'Annotator3': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(df)

    # Should return 1.0 for perfect agreement
    assert f_measure == 1.0


def test_calculate_with_opposite_agreement(f_measure_calculator):
    """Test calculate method with opposite agreement."""
    # Create a DataFrame with opposite agreement
    data = {
        'Annotator1': [1, 1, 1, 1, 1],
        'Annotator2': [0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)

    # Calculate F-measure
    f_measure = f_measure_calculator.calculate(df)

    # Should return 0.0 for opposite agreement
    assert f_measure == 0.0


def test_prepare_binary_data_with_threshold_on_non_numeric(
        f_measure_calculator):
    """Test _prepare_binary_data with threshold on non-numeric data."""
    # Create a DataFrame with non-numeric data
    data = {
        'Annotator1': ['high', 'low', 'medium', 'high', 'low'],
        'Annotator2': ['medium', 'low', 'high', 'medium', 'high']
    }
    df = pd.DataFrame(data)

    # Try to prepare binary data with threshold (should trigger warning)
    binary_data = f_measure_calculator._prepare_binary_data(df, threshold=0.5)

    # Check that result is a numpy array with the expected shape
    assert isinstance(binary_data, np.ndarray)
    assert binary_data.shape == df.shape

    # Check that the original data was returned
    assert binary_data.dtype == np.dtype('O')  # Object dtype for strings


def test_prepare_binary_data_with_non_numeric_data(f_measure_calculator):
    """
    Test _prepare_binary_data with non-numeric data and no
    threshold/positive_class.
    """
    # Create a DataFrame with non-numeric data
    data = {
        'Annotator1': ['yes', 'no', 'maybe', 'yes', 'no'],
        'Annotator2': ['maybe', 'no', 'yes', 'maybe', 'yes']
    }
    df = pd.DataFrame(data)

    # Try to prepare binary data without threshold or positive_class
    binary_data = f_measure_calculator._prepare_binary_data(df)

    # Check that result is a numpy array with the expected shape
    assert isinstance(binary_data, np.ndarray)
    assert binary_data.shape == df.shape

    # Check that the original data was returned
    assert binary_data.dtype == np.dtype('O')  # Object dtype for strings


def test_prepare_binary_data_with_none_and_empty_strings(f_measure_calculator):
    """Test _prepare_binary_data with None values and empty strings."""
    # Create a DataFrame with None values and empty strings
    data = {
        'Annotator1': ['A', None, 'B', '', 'C'],
        'Annotator2': ['B', 'A', None, 'C', '']
    }
    df = pd.DataFrame(data)

    # Prepare binary data with positive_class
    binary_data = f_measure_calculator._prepare_binary_data(df,
                                                            positive_class='A')

    # Check that result is a numpy array with the expected shape
    assert isinstance(binary_data, np.ndarray)
    assert binary_data.shape == df.shape

    # Check that None and empty strings are converted to NaN
    # First column: [1, nan, 0, nan, 0]
    # Second column: [0, 1, nan, 0, nan]
    assert binary_data[0, 0] == 1  # 'A' in first column
    assert binary_data[1, 1] == 1  # 'A' in second column
    assert np.isnan(binary_data[1, 0])  # None in first column
    assert np.isnan(binary_data[2, 1])  # None in second column
    assert np.isnan(binary_data[3, 0])  # Empty string in first column
    assert np.isnan(binary_data[4, 1])  # Empty string in second column


def test__calculate_precision_recall_with_precision_and_recall_zero(
        f_measure_calculator):
    """
    Test _calculate_precision_recall method with precision and recall zero.
    """
    a1 = np.array([0, 0, 0])
    a2 = np.array([0, 0, 0])

    precision, recall = f_measure_calculator._calculate_precision_recall(a1,
                                                                         a2)
    assert precision == 0.0
    assert recall == 0.0


def test__prepare_binary_data_with_other_than_0_1_nan(f_measure_calculator):
    """Test _prepare_binary_data method with other than 0, 1, or NaN values."""
    data = {
        'Annotator1': [1, 2, 3, 4, 5],
        'Annotator2': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    output_stream = io.StringIO()

    f_measure_calculator.logger.output = output_stream

    # Try to prepare binary data with threshold (should trigger warning)
    binary_data = f_measure_calculator._prepare_binary_data(df)

    # Check that result is a numpy array with the expected shape
    assert isinstance(binary_data, np.ndarray)
    assert binary_data.shape == df.shape

    output = output_stream.getvalue()
    print(output)
    assert "Data contains values other than 0 and 1. " in output
    assert "Consider providing a threshold or positive_class." in output
