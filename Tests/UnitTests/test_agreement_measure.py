import pytest
import pandas as pd
from src.agreement_measure import AgreementMeasure
from Utils.logger import LogLevel


class MockAgreementMeasure(AgreementMeasure):
    """
    Mock implementation of AgreementMeasure that only implements calculate.
    """

    def calculate(self, df):
        """Implement the abstract calculate method."""
        return 0.5  # Return a dummy value


def test_calculate_pairwise_not_implemented():
    """
    Test that calculate_pairwise raises NotImplementedError when not
    implemented.
    """
    # Create an instance of our test class
    measure = MockAgreementMeasure(level=LogLevel.DEBUG)

    # Create a dummy DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Call calculate_pairwise and check that it raises NotImplementedError
    with pytest.raises(NotImplementedError) as excinfo:
        measure.calculate_pairwise(df)

    # Check that the error message contains the class name
    assert "MockAgreementMeasure" in str(excinfo.value)
    assert "does not implement calculate_pairwise" in str(excinfo.value)


def test_calculate_from_dataframe():
    """Test that calculate_from_dataframe calls calculate."""
    # Create an instance of our test class
    measure = MockAgreementMeasure(level=LogLevel.DEBUG)

    # Create a dummy DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Call calculate_from_dataframe
    result = measure.calculate_from_dataframe(df)

    # Check that it returns the same value as calculate
    assert result == 0.5


def test_interpret():
    """Test the interpret method."""
    measure = MockAgreementMeasure(level=LogLevel.DEBUG)

    # Test different agreement values
    assert "Poor agreement" in measure.interpret(-0.1)
    assert "Slight agreement" in measure.interpret(0.1)
    assert "Fair agreement" in measure.interpret(0.3)
    assert "Moderate agreement" in measure.interpret(0.5)
    assert "Substantial agreement" in measure.interpret(0.7)
    assert "Almost perfect agreement" in measure.interpret(0.9)
