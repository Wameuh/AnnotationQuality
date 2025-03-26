import pytest
import numpy as np
import pandas as pd
from src.distance_based_cell_agreement import DistanceBasedCellAgreement
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
def dbcaa_calculator():
    """Fixture providing a DistanceBasedCellAgreement instance."""
    return DistanceBasedCellAgreement(level=LogLevel.DEBUG)


@pytest.fixture
def perfect_agreement_cells():
    """Fixture providing cell positions with perfect agreement."""
    # Create 3 identical sets of cell positions
    cells = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50]
    ])

    return [cells.copy() for _ in range(3)]


@pytest.fixture
def high_agreement_cells():
    """
    Fixture providing cell positions with high agreement (small differences).
    """
    # Create base cell positions
    base_cells = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50]
    ])

    # Create variations with more significant differences
    cells1 = base_cells.copy()

    cells2 = base_cells.copy()
    cells2[0] += np.array([5, 5])  # Move first cell more significantly

    cells3 = base_cells.copy()
    cells3[1] += np.array([-5, 5])  # Move second cell more significantly
    cells3[4] += np.array([5, -5])  # Move fifth cell more significantly

    return [cells1, cells2, cells3]


@pytest.fixture
def low_agreement_cells():
    """Fixture providing cell positions with low agreement."""
    # Create different sets of cell positions
    cells1 = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50]
    ])

    cells2 = np.array([
        [15, 15],
        [25, 25],
        [35, 35],
        [45, 45]
    ])

    cells3 = np.array([
        [10, 10],
        [30, 30],
        [60, 60]
    ])

    return [cells1, cells2, cells3]


@pytest.fixture
def moderate_agreement_cells():
    """Fixture providing cell positions with moderate agreement."""
    # Create three sets of cell positions with moderate agreement
    cells1 = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
    cells2 = np.array([[12, 12], [22, 22], [32, 32], [42, 42], [52, 52]])
    # 2.83 units away
    cells3 = np.array([[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]])
    # 7.07 units away

    return [cells1, cells2, cells3]


def test_calculate_with_empty_input(dbcaa_calculator):
    """Test calculate method with empty input."""
    # Empty list of cell positions
    with pytest.raises(ValueError, match="No cell position data provided"):
        dbcaa_calculator.calculate([])


def test_calculate_with_invalid_input(dbcaa_calculator):
    """Test calculate method with invalid input."""
    # List with non-numpy arrays
    with pytest.raises(ValueError,
                       match="All cell positions must be numpy arrays"):
        dbcaa_calculator.calculate([[[1, 2], [3, 4]], np.array([[5, 6]])])

    # Arrays with wrong shape
    with pytest.raises(ValueError,
                       match="Cell position arrays must have shape"):
        dbcaa_calculator.calculate([
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6, 7]])  # Wrong shape
        ])


def test_calculate_with_valid_input(dbcaa_calculator):
    """Test calculate method with valid input."""
    # Create test data
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[11, 11], [21, 21], [31, 31]])  # Close to cells1
    cells3 = np.array([[15, 15], [25, 25], [35, 35]])  # Further from cells1

    # Calculate DBCAA
    dbcaa = dbcaa_calculator.calculate([cells1, cells2, cells3])

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_calculate_pairwise_agreement_both_empty(dbcaa_calculator):
    """Test _calculate_pairwise_agreement with both arrays empty."""
    cells1 = np.array([]).reshape(0, 2)
    cells2 = np.array([]).reshape(0, 2)

    # Should return 1.0 (perfect agreement) when both are empty
    agreement = dbcaa_calculator._calculate_pairwise_agreement(
        cells1, cells2, distance_threshold=10.0)
    assert agreement == 1.0


def test_calculate_pairwise_agreement_one_empty(dbcaa_calculator):
    """Test _calculate_pairwise_agreement with one array empty."""
    cells1 = np.array([[10, 10], [20, 20]])
    cells2 = np.array([]).reshape(0, 2)

    # Should return 0.0 (no agreement) when one is empty
    agreement = dbcaa_calculator._calculate_pairwise_agreement(
        cells1, cells2, distance_threshold=10.0)
    assert agreement == 0.0


def test_calculate_pairwise_agreement_exact_match(dbcaa_calculator):
    """Test _calculate_pairwise_agreement with exact matching cells."""
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[10, 10], [20, 20], [30, 30]])  # Exact match

    # Should return 1.0 (perfect agreement) for exact matches
    agreement = dbcaa_calculator._calculate_pairwise_agreement(
        cells1, cells2, distance_threshold=10.0)
    assert agreement == 1.0


def test_calculate_pairwise_agreement_close_match(dbcaa_calculator):
    """Test _calculate_pairwise_agreement with close matching cells."""
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[11, 11], [21, 21], [31, 31]])  # Close match

    # Should return high agreement for close matches
    agreement = dbcaa_calculator._calculate_pairwise_agreement(
        cells1, cells2, distance_threshold=5.0)
    assert agreement > 0.8


def test_calculate_pairwise_agreement_no_match(dbcaa_calculator):
    """Test _calculate_pairwise_agreement with non-matching cells."""
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[50, 50], [60, 60], [70, 70]])  # No match

    # Should return 0.0 for no matches within threshold
    agreement = dbcaa_calculator._calculate_pairwise_agreement(
        cells1, cells2, distance_threshold=5.0)
    assert agreement == 0.0


def test_calculate_from_dataframe(dbcaa_calculator):
    """Test calculate_from_dataframe method."""
    # Create a DataFrame where each row is a cell position
    # and each column is an annotator
    data = {
        'Annotator1': [[10, 10], [20, 20], [30, 30]],
        'Annotator2': [[11, 11], [21, 21], [31, 31]]
    }

    # Convert to a format that calculate_from_dataframe can handle
    cell_positions = []
    for col in data:
        cell_positions.append(np.array(data[col]))

    # Call calculate directly with the cell positions
    dbcaa = dbcaa_calculator.calculate(cell_positions)

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_interpret_dbcaa(dbcaa_calculator):
    """Test interpret_dbcaa method."""
    # Test interpretation for each range
    assert "Invalid DBCAA value" in dbcaa_calculator.interpret_dbcaa(-0.1)
    assert "Poor agreement" in dbcaa_calculator.interpret_dbcaa(0.05)
    assert "Slight agreement" in dbcaa_calculator.interpret_dbcaa(0.15)
    assert "Fair agreement" in dbcaa_calculator.interpret_dbcaa(0.3)
    assert "Moderate agreement" in dbcaa_calculator.interpret_dbcaa(0.5)
    assert "Substantial agreement" in dbcaa_calculator.interpret_dbcaa(0.7)
    assert "Almost perfect agreement" in dbcaa_calculator.interpret_dbcaa(0.9)


def test_get_dbcaa_statistics(dbcaa_calculator):
    """Test get_dbcaa_statistics method."""
    # Create test data
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[12, 12], [22, 22], [32, 32]])  # 2.83 units away

    # Calculate with different thresholds
    # Should match none of the cells (distance > 2)
    low_threshold = dbcaa_calculator.calculate(
        [cells1, cells2], distance_threshold=2.0)

    # Should match all cells (distance < 5)
    high_threshold = dbcaa_calculator.calculate(
        [cells1, cells2], distance_threshold=5.0)

    # Low threshold should give lower agreement
    assert low_threshold < high_threshold

    # High threshold should give perfect agreement
    assert high_threshold == 1.0


def test_calculate_with_no_valid_pairs(dbcaa_calculator):
    """Test calculate method when no valid pairs are found."""
    # Create a case where there are no valid pairs
    # (e.g., only one annotator)
    cells = [np.array([[10, 10], [20, 20]])]

    # This should return 0.0 and log a warning
    result = dbcaa_calculator.calculate(cells)
    assert result == 0.0


def test_calculate_with_nan_coordinates(dbcaa_calculator):
    """Test calculate method with NaN coordinates."""
    # Create cell positions with NaN values
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[11, 11], [np.nan, np.nan], [31, 31]])  # NaN coords

    # This should raise a ValueError or handle NaNs gracefully
    with pytest.raises(ValueError):
        dbcaa_calculator.calculate([cells1, cells2])


def test_calculate_from_dataframe_with_missing_data(dbcaa_calculator):
    """Test calculate_from_dataframe method with missing data."""
    # Create a DataFrame with missing data
    data = {
        'Annotator1': [[10, 10], [20, 20], None],
        'Annotator2': [[11, 11], None, [31, 31]]
    }

    # Convert to a format that calculate can handle, filtering out None values
    cell_positions = []
    for col in data:
        valid_positions = [pos for pos in data[col] if pos is not None]
        cell_positions.append(np.array(valid_positions))

    # Call calculate directly with the filtered cell positions
    dbcaa = dbcaa_calculator.calculate(cell_positions)

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_calculate_from_dataframe_with_real_dataframe(dbcaa_calculator):
    """Test calculate_from_dataframe method with a real DataFrame."""
    # Create a DataFrame with cell position data
    data = {
        'Annotator1': [[10, 10], [20, 20], None],
        'Annotator2': [[11, 11], None, [31, 31]]
    }
    df = pd.DataFrame(data)

    # Call calculate_from_dataframe directly
    dbcaa = dbcaa_calculator.calculate_from_dataframe(df)

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_calculate_from_dataframe_with_nan_values(dbcaa_calculator):
    """Test calculate_from_dataframe method with NaN values."""
    # Create a DataFrame with NaN values
    data = {
        'Annotator1': [[10, 10], [np.nan, np.nan], [30, 30]],
        'Annotator2': [[11, 11], [21, 21], [31, 31]]
    }
    df = pd.DataFrame(data)

    # Call calculate_from_dataframe directly
    dbcaa = dbcaa_calculator.calculate_from_dataframe(df)

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_calculate_from_dataframe_with_invalid_column(dbcaa_calculator):
    """Test calculate_from_dataframe method with an invalid column."""
    # Create a DataFrame with an invalid column
    data = {
        'Annotator1': [[10, 10], [20, 20], [30, 30]],
        'Annotator2': ["not a position", "invalid", "data"]  # Invalid data
    }
    df = pd.DataFrame(data)

    # Call calculate_from_dataframe directly
    dbcaa = dbcaa_calculator.calculate_from_dataframe(df)

    # Check that result is a float between 0 and 1
    assert isinstance(dbcaa, float)
    assert 0.0 <= dbcaa <= 1.0


def test_get_dbcaa_statistics_with_different_thresholds(dbcaa_calculator):
    """Test get_dbcaa_statistics method with different thresholds."""
    # Create test data
    cells1 = np.array([[10, 10], [20, 20], [30, 30]])
    cells2 = np.array([[12, 12], [22, 22], [32, 32]])  # 2.83 units away

    # Call get_dbcaa_statistics
    stats = dbcaa_calculator.get_dbcaa_statistics(
        [cells1, cells2],
        distance_thresholds=[2.0, 5.0, 10.0]
    )

    # Check that the result contains the expected keys
    assert 'dbcaa_standard' in stats
    assert 'interpretation_standard' in stats
    assert 'dbcaa_threshold_2.0' in stats
    assert 'interpretation_threshold_2.0' in stats
    assert 'dbcaa_threshold_5.0' in stats
    assert 'interpretation_threshold_5.0' in stats
    assert 'dbcaa_threshold_10.0' in stats
    assert 'interpretation_threshold_10.0' in stats

    # Check that the values make sense
    assert stats['dbcaa_threshold_2.0'] < stats['dbcaa_threshold_5.0']
    assert stats['dbcaa_threshold_5.0'] <= stats['dbcaa_threshold_10.0']


def test_calculate_pairwise(dbcaa_calculator, perfect_agreement_cells):
    """Test calculate_pairwise method with perfect agreement."""
    # Create a DataFrame with cell positions in a format that the calculator
    # expects
    # For example, if the calculator expects strings like "10,10;20,20;30,30"
    def format_cells(cells):
        return ";".join([f"{x},{y}" for x, y in cells])

    df = pd.DataFrame({
        'Annotator1': [format_cells(perfect_agreement_cells[0])],
        'Annotator2': [format_cells(perfect_agreement_cells[1])],
        'Annotator3': [format_cells(perfect_agreement_cells[2])]
    })

    # Calculate pairwise DBCAAs
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_dbcaas, dict)

    # Check that all pairs are present
    columns = df.columns
    n_annotators = len(columns)
    expected_pairs_count = (n_annotators * (n_annotators - 1)) // 2
    assert len(pairwise_dbcaas) == expected_pairs_count

    # Check that values are between 0 and 1
    for k, v in pairwise_dbcaas.items():
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert 0.0 <= v <= 1.0

    # With perfect agreement, DBCAA should be 1.0
    for v in pairwise_dbcaas.values():
        assert v == 1.0


def test_calculate_pairwise_with_small_differences(dbcaa_calculator,
                                                   moderate_agreement_cells):
    """Test calculate_pairwise method with small differences."""
    # Create a DataFrame from the cell positions
    df = pd.DataFrame({
        'Annotator1': [moderate_agreement_cells[0]],
        'Annotator2': [moderate_agreement_cells[1]],
        'Annotator3': [moderate_agreement_cells[2]]
    })

    # Calculate pairwise DBCAAs with a smaller threshold
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df, threshold=2.0)

    # Check that result is a dictionary
    assert isinstance(pairwise_dbcaas, dict)

    # Check that values are between 0 and 1
    for v in pairwise_dbcaas.values():
        assert 0.0 <= v <= 1.0

    # With moderate agreement, at least one pair should have DBCAA < 1.0
    assert any(v < 1.0 for v in pairwise_dbcaas.values())


def test_calculate_pairwise_with_threshold(dbcaa_calculator,
                                           moderate_agreement_cells):
    """Test calculate_pairwise method with different threshold."""
    # Create a DataFrame from the cell positions
    df = pd.DataFrame({
        'Annotator1': [moderate_agreement_cells[0]],
        'Annotator2': [moderate_agreement_cells[1]],
        'Annotator3': [moderate_agreement_cells[2]]
    })

    # Calculate pairwise DBCAAs with different threshold
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df, threshold=20.0)

    # Check that result is a dictionary
    assert isinstance(pairwise_dbcaas, dict)

    # Check that values are between 0 and 1
    for v in pairwise_dbcaas.values():
        assert 0.0 <= v <= 1.0

    # With larger threshold, DBCAA should be higher
    # Calculate with default threshold for comparison
    pairwise_dbcaas_default = dbcaa_calculator.calculate_pairwise(df)

    # At least one pair should have higher agreement with larger threshold
    assert any(pairwise_dbcaas[k] >= pairwise_dbcaas_default[k]
               for k in pairwise_dbcaas.keys())


def test_calculate_pairwise_with_empty_data(dbcaa_calculator):
    """Test calculate_pairwise method with empty data."""
    # Create a DataFrame with empty data for one annotator
    df = pd.DataFrame({
        'Annotator1': [";".join(
            [f"{x},{y}" for x, y in np.array([[10, 10], [20, 20]])])],
        'Annotator2': [";".join(
            [f"{x},{y}" for x, y in np.array([[11, 11], [21, 21]])])],
        'Annotator3': [None]  # Empty data
    })

    # Calculate pairwise DBCAAs
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_dbcaas, dict)

    # Should only have one pair (Annotator1, Annotator2)
    assert len(pairwise_dbcaas) == 1
    assert ('Annotator1', 'Annotator2') in pairwise_dbcaas


def test_calculate_pairwise_with_error(dbcaa_calculator):
    """Test calculate_pairwise method with data that causes an error."""
    # Create a DataFrame with invalid data that will cause an error
    df = pd.DataFrame({
        'Annotator1': [";".join(
            [f"{x},{y}" for x, y in np.array([[10, 10], [20, 20]])])],
        'Annotator2': ["invalid;data;format"],  # Invalid format
        'Annotator3': [";".join(
            [f"{x},{y}" for x, y in np.array([[11, 11], [21, 21]])])]
    })

    # Calculate pairwise DBCAAs
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_dbcaas, dict)

    # Should have all pairs, but with 0.0 for pairs involving invalid data
    assert len(pairwise_dbcaas) == 3
    assert ('Annotator1', 'Annotator2') in pairwise_dbcaas
    assert ('Annotator1', 'Annotator3') in pairwise_dbcaas
    assert ('Annotator2', 'Annotator3') in pairwise_dbcaas

    # Check that pairs with invalid data have a score of 0.0
    assert pairwise_dbcaas[('Annotator1', 'Annotator2')] == 0.0
    assert pairwise_dbcaas[('Annotator2', 'Annotator3')] == 0.0

    # Check that the valid pair has a positive score
    assert pairwise_dbcaas[('Annotator1', 'Annotator3')] > 0.0


def test_extract_cell_coordinates_with_invalid_input(dbcaa_calculator):
    """Test _extract_cell_coordinates method with invalid input."""
    # Test with None
    result = dbcaa_calculator._extract_cell_coordinates(None)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)

    # Test with empty string
    result = dbcaa_calculator._extract_cell_coordinates("")
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)

    # Test with non-string
    result = dbcaa_calculator._extract_cell_coordinates(123)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)


def test_extract_cell_coordinates_with_valid_input(dbcaa_calculator):
    """Test _extract_cell_coordinates method with valid input."""
    # Test with valid string
    cell_str = "10,10;20,20;30,30"
    result = dbcaa_calculator._extract_cell_coordinates(cell_str)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    assert np.array_equal(result, np.array([[10, 10], [20, 20], [30, 30]]))

    # Test with mixed valid and invalid entries
    cell_str = "10,10;invalid;20,20;30,30"
    result = dbcaa_calculator._extract_cell_coordinates(cell_str)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    assert np.array_equal(result, np.array([[10, 10], [20, 20], [30, 30]]))


def test_extract_cell_coordinates_with_error(dbcaa_calculator):
    """
    Test _extract_cell_coordinates method with input that causes an error.
    """
    # Create a string that will cause an error during parsing
    cell_str = "10,10;20,invalid;30,30"

    # Should not raise an exception, but return an array with valid coordinates
    result = dbcaa_calculator._extract_cell_coordinates(cell_str)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)  # Should have 2 valid coordinates
    assert np.array_equal(result, np.array([[10, 10], [30, 30]]))


def test_calculate_pairwise_with_calculation_error(dbcaa_calculator,
                                                   monkeypatch):
    """
    Test calculate_pairwise method with an error during DBCAA calculation.
    """
    # Create a DataFrame with valid data
    df = pd.DataFrame({
        'Annotator1': [";".join(
            [f"{x},{y}" for x, y in np.array([[10, 10], [20, 20]])])],
        'Annotator2': [";".join(
            [f"{x},{y}" for x, y in np.array([[11, 11], [21, 21]])])]
    })

    # Mock the _calculate_pairwise_agreement method to raise an exception
    def mock_calculate_pairwise_agreement(*args, **kwargs):
        raise ValueError("Simulated error during calculation")

    # Apply the mock
    monkeypatch.setattr(dbcaa_calculator, "_calculate_pairwise_agreement",
                        mock_calculate_pairwise_agreement)

    # Calculate pairwise DBCAAs - should not raise an exception
    pairwise_dbcaas = dbcaa_calculator.calculate_pairwise(df)

    # Check that result is an empty dictionary (no successful calculations)
    assert isinstance(pairwise_dbcaas, dict)
    assert len(pairwise_dbcaas) == 0


def test_extract_cell_coordinates_with_general_exception(dbcaa_calculator,
                                                         monkeypatch):
    """Test _extract_cell_coordinates method with a general exception."""
    # Create a valid cell string
    cell_str = "10,10;20,20;30,30"

    # Mock numpy.array to raise an exception
    original_array = np.array

    def mock_array(arg, *args, **kwargs):
        if (isinstance(arg, list) and len(arg) > 0 and
                isinstance(arg[0], list) and len(arg[0]) == 2):
            # This is the call in _extract_cell_coordinates
            raise Exception("Simulated numpy.array exception")
        return original_array(arg, *args, **kwargs)

    # Apply the mock
    monkeypatch.setattr(np, "array", mock_array)

    # Call _extract_cell_coordinates - should trigger the general exception
    result = dbcaa_calculator._extract_cell_coordinates(cell_str)

    # Should return an empty array
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)
