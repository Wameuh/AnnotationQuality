import os
import pytest
import numpy as np
from src.dataPreparation import DataLoader
from src.krippendorff_alpha import KrippendorffAlpha
from Utils.logger import LogLevel


@pytest.fixture
def data_loader():
    """Fixture providing a DataLoader instance."""
    return DataLoader(level=LogLevel.DEBUG)


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


@pytest.fixture
def alpha_calculator():
    """Fixture providing a KrippendorffAlpha instance."""
    return KrippendorffAlpha(level=LogLevel.DEBUG)


def test_krippendorff_alpha_real_data(real_data, alpha_calculator):
    """
    Test that Krippendorff's Alpha calculations on real data match expected
    values.

    This test ensures that future changes to the codebase don't inadvertently
    change the results of the Alpha calculations on our reference dataset.
    """
    # Calculate alpha statistics
    alpha_stats = alpha_calculator.get_alpha_statistics(real_data)

    # Expected values based on current implementation
    expected_values = {
        'alpha_nominal': 0.7982,
        'alpha_ordinal': 0.9582,
        'alpha_interval': 0.9582,
        'alpha_ratio': 0.9582
    }

    # Check that each alpha value is close to the expected value
    for metric in ['nominal', 'ordinal', 'interval', 'ratio']:
        alpha_key = f'alpha_{metric}'
        assert alpha_key in alpha_stats, f"Missing {alpha_key} in results"

        # Use pytest's approx to allow for small floating-point differences
        assert alpha_stats[alpha_key] == pytest.approx(
            expected_values[alpha_key], abs=0.001
        ), (f"Alpha for {metric} metric changed: expected"
            f" {expected_values[alpha_key]}, got {alpha_stats[alpha_key]}")

        # Also check that interpretations are present
        interp_key = f'interpretation_{metric}'
        assert interp_key in alpha_stats, f"Missing {interp_key} in results"

        # Verify interpretations match expected values
        if metric == 'nominal':
            assert "Substantial agreement" in alpha_stats[interp_key]
        else:
            assert "Almost perfect agreement" in alpha_stats[interp_key]


def test_krippendorff_alpha_individual_metrics(real_data, alpha_calculator):
    """
    Test individual metric calculations for Krippendorff's Alpha on real data.

    This test checks each metric separately to isolate potential issues.
    """
    # Expected values for each metric
    expected = {
        'nominal': 0.7982,
        'ordinal': 0.9582,
        'interval': 0.9582,
        'ratio': 0.9582
    }

    # Test each metric individually
    for metric, expected_value in expected.items():
        alpha = alpha_calculator.calculate(real_data, metric=metric)
        assert alpha == pytest.approx(
            expected_value, abs=0.001
        ), (f"Alpha for {metric} metric changed: "
            f"expected {expected_value}, got {alpha}")


def test_krippendorff_alpha_with_subset(real_data, alpha_calculator):
    """
    Test Krippendorff's Alpha on a subset of the data to ensure stability.

    This test checks that the algorithm works correctly with different
    data sizes.
    """
    # Create a subset of the data (first 10 rows)
    subset_data = real_data.iloc[:10, :]

    # Calculate alpha for the subset
    alpha_nominal = alpha_calculator.calculate(subset_data, metric='nominal')

    # We don't check exact values here, just that the calculation completes
    # and returns a reasonable value
    assert -1.0 <= alpha_nominal <= 1.0, \
        f"Alpha value out of expected range: {alpha_nominal}"

    # The interpretation should be a non-empty string
    interpretation = alpha_calculator.interpret_alpha(alpha_nominal)
    assert isinstance(interpretation, str) and len(interpretation) > 0


def test_krippendorff_alpha_with_missing_values(real_data, alpha_calculator):
    """
    Test Krippendorff's Alpha with artificially introduced missing values.

    This test ensures the algorithm correctly handles missing data.
    """
    # Create a copy of the data with some values set to NaN
    data_with_missing = real_data.copy()

    # Set ~10% of values to NaN
    rows, cols = data_with_missing.shape
    for _ in range(int(rows * cols * 0.1)):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        data_with_missing.iloc[i, j] = np.nan

    # Calculate alpha with missing values
    alpha_nominal = alpha_calculator.calculate(data_with_missing,
                                               metric='nominal')

    # The value should be within the valid range
    assert -1.0 <= alpha_nominal <= 1.0, \
        f"Alpha value with missing data out of expected range: {alpha_nominal}"
