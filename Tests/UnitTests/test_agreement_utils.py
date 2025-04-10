import pytest
from Utils.agreement_utils import (
    get_unique_annotators, create_agreement_matrix,
    format_with_confidence_interval, get_confidence_method_display,
    get_confidence_level
)


@pytest.fixture
def sample_agreements():
    """Fixture providing sample agreement data."""
    return {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }


@pytest.fixture
def sample_confidence_intervals():
    """Fixture providing sample confidence interval data."""
    return {
        ('Annotator1', 'Annotator2'): {
            'ci_lower': 0.70,
            'ci_upper': 0.80,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        ('Annotator1', 'Annotator3'): {
            'ci_lower': 0.75,
            'ci_upper': 0.85,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        ('Annotator2', 'Annotator3'): {
            'ci_lower': 0.80,
            'ci_upper': 0.90,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
    }


def test_get_unique_annotators(sample_agreements):
    """Test get_unique_annotators function."""
    annotators = get_unique_annotators(sample_agreements)
    assert len(annotators) == 3
    assert 'Annotator1' in annotators
    assert 'Annotator2' in annotators
    assert 'Annotator3' in annotators
    assert annotators == sorted(annotators)  # Check that they're sorted


def test_create_agreement_matrix(sample_agreements,
                                 sample_confidence_intervals):
    """Test create_agreement_matrix function."""
    annotators = get_unique_annotators(sample_agreements)

    # Test without confidence intervals
    matrix = create_agreement_matrix(sample_agreements, annotators)
    assert len(matrix) == 3  # 3 rows
    assert len(matrix[0]) == 3  # 3 columns

    # Check diagonal elements
    assert matrix[0][0] == "---"
    assert matrix[1][1] == "---"
    assert matrix[2][2] == "---"

    # Check some values
    assert "75.0%" in matrix[0][1] or "75.0%" in matrix[1][0]
    assert "80.0%" in matrix[0][2] or "80.0%" in matrix[2][0]
    assert "85.0%" in matrix[1][2] or "85.0%" in matrix[2][1]

    # Test with confidence intervals
    matrix_with_ci = create_agreement_matrix(
        sample_agreements, annotators, sample_confidence_intervals)

    # Check that confidence intervals are included
    for i in range(3):
        for j in range(3):
            if i != j:  # Skip diagonal
                cell = matrix_with_ci[i][j]
                if cell != "N/A":
                    assert "(" in cell and ")" in cell


def test_format_with_confidence_interval():
    """Test format_with_confidence_interval function."""
    ci = {'ci_lower': 0.7, 'ci_upper': 0.8}

    # Test with default format function
    result = format_with_confidence_interval(0.75, ci)
    assert result == "75.0% (70.0% - 80.0%)"

    # Test with custom format function
    result = format_with_confidence_interval(0.75,
                                             ci,
                                             lambda val: f"{val:.2f}")
    assert result == "0.75 (0.70 - 0.80)"


def test_get_confidence_method_display(sample_confidence_intervals):
    """Test get_confidence_method_display function."""
    method_display = get_confidence_method_display(sample_confidence_intervals)
    assert method_display == "Wilson score"

    # Test with different method
    modified_ci = sample_confidence_intervals.copy()
    first_key = next(iter(modified_ci.keys()))
    modified_ci[first_key] = modified_ci[first_key].copy()
    modified_ci[first_key]['method'] = 'bootstrap'

    method_display = get_confidence_method_display(modified_ci)
    assert method_display == "Bootstrap"

    # Test with unknown method
    modified_ci[first_key]['method'] = 'unknown_method'
    method_display = get_confidence_method_display(modified_ci)
    assert method_display == "unknown_method"


def test_get_confidence_level(sample_confidence_intervals):
    """Test get_confidence_level function."""
    confidence_level = get_confidence_level(sample_confidence_intervals)
    assert confidence_level == 0.95

    # Test with different confidence level
    modified_ci = sample_confidence_intervals.copy()
    first_key = next(iter(modified_ci.keys()))
    modified_ci[first_key] = modified_ci[first_key].copy()
    modified_ci[first_key]['confidence_level'] = 0.99

    confidence_level = get_confidence_level(modified_ci)
    assert confidence_level == 0.99

    # Test with missing confidence level
    del modified_ci[first_key]['confidence_level']
    confidence_level = get_confidence_level(modified_ci)
    assert confidence_level == 0.95  # Default value


def test_get_confidence_level_with_empty_dict():
    """Test get_confidence_level function with empty dictionary."""
    # Test with empty dictionary
    empty_dict = {}
    confidence_level = get_confidence_level(empty_dict)
    assert confidence_level == 0.95  # Default value

    # Test with None
    confidence_level = get_confidence_level(None)
    assert confidence_level == 0.95  # Default value
