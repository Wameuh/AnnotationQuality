import os
import pytest
import pandas as pd
import numpy as np
from src.dataPreparation import DataLoader
from src.f_measure import FMeasure
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
def f_measure_calculator():
    """Fixture providing a FMeasure instance."""
    return FMeasure(level=LogLevel.DEBUG)


def test_f_measure_real_data(real_data, f_measure_calculator):
    """
    Test that F-measure calculations on real data match expected values.

    This test ensures that future changes to the codebase don't inadvertently
    change the results of the F-measure calculations on our reference dataset.
    """
    # Calculate F-measure statistics
    f_measure_stats = f_measure_calculator.get_f_measure_statistics(real_data)

    # Expected values based on current implementation
    expected_values = {
        'f_measure': 1.0,
        'f_measure_median_threshold': 0.0,
        'f_measure_mean_threshold': 0.9235,
        'f_measure_class_1.0': 0.8089,
        'f_measure_class_2.0': 0.6257,
        'f_measure_class_3.0': 0.6922,
        'f_measure_class_4.0': 0.7222,
        'f_measure_class_5.0': 0.9235
    }

    # Check that each F-measure value is close to the expected value
    for key, expected_value in expected_values.items():
        assert key in f_measure_stats, f"Missing {key} in results"

        # Use pytest's approx to allow for small floating-point differences
        assert f_measure_stats[key] == pytest.approx(
            expected_value, abs=0.001
        ), (f"F-measure for {key} changed: "
            f"expected {expected_value}, got {f_measure_stats[key]}")

        # Also check that interpretations are consistent
        if key == 'f_measure':
            interpretation = f_measure_calculator.interpret_f_measure(
                f_measure_stats[key])
            assert "Almost perfect agreement" in interpretation


def test_f_measure_individual_metrics(real_data, f_measure_calculator):
    """
    Test individual F-measure calculations on real data.

    This test checks each F-measure calculation separately to isolate
    potential issues.
    """
    # Test standard F-measure
    f_measure = f_measure_calculator.calculate(real_data)
    assert f_measure == pytest.approx(1.0, abs=0.001)

    # Test with median threshold
    median_threshold = np.median(real_data.values[~np.isnan(real_data.values)])
    f_measure_median = f_measure_calculator.calculate(
        real_data, threshold=median_threshold)
    assert f_measure_median == pytest.approx(0.0, abs=0.001)

    # Test with mean threshold
    mean_threshold = np.mean(real_data.values[~np.isnan(real_data.values)])
    f_measure_mean = f_measure_calculator.calculate(
        real_data, threshold=mean_threshold)
    assert f_measure_mean == pytest.approx(0.9235, abs=0.001)


def test_f_measure_with_subset(real_data, f_measure_calculator):
    """
    Test F-measure on a subset of the data to ensure stability.

    This test checks that the algorithm works correctly with different
    data sizes.
    """
    # Create a subset of the data (first 10 rows)
    subset_data = real_data.iloc[:10, :]

    # Calculate F-measure for the subset
    f_measure = f_measure_calculator.calculate(subset_data)

    # We don't check exact values here, just that the calculation completes
    # and returns a reasonable value
    assert 0.0 <= f_measure <= 1.0

    # The interpretation should be a non-empty string
    interpretation = f_measure_calculator.interpret_f_measure(f_measure)
    assert isinstance(interpretation, str) and len(interpretation) > 0


def test_f_measure_class_specific(real_data, f_measure_calculator):
    """
    Test class-specific F-measure calculations.

    This test ensures that F-measure calculations for specific classes
    remain consistent.
    """
    # Expected values for each class
    expected = {
        1.0: 0.8089,
        2.0: 0.6257,
        3.0: 0.6922,
        4.0: 0.7222,
        5.0: 0.9235
    }

    # Test each class individually
    for class_value, expected_value in expected.items():
        f_measure = f_measure_calculator.calculate(
            real_data, positive_class=class_value)
        assert f_measure == pytest.approx(
            expected_value, abs=0.001
        ), (f"F-measure for class {class_value} changed: "
            f"expected {expected_value}, got {f_measure}")