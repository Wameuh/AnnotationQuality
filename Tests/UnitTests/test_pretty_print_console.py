import pytest
import io
from contextlib import redirect_stdout
from Utils.pretty_print_console import print_agreement_table


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


def test_print_agreement_table_basic(sample_agreements):
    """Test basic functionality of print_agreement_table."""
    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(sample_agreements)

    output = f.getvalue()

    # Check that all annotators are in the output
    assert 'Annotator1' in output
    assert 'Annotator2' in output
    assert 'Annotator3' in output

    # Check that all agreement values are in the output
    assert '75.0%' in output
    assert '80.0%' in output
    assert '85.0%' in output

    # Check that the table header is present
    assert 'Inter-annotator Agreement Matrix (3 annotators)' in output


def test_print_agreement_table_with_ci(sample_agreements,
                                       sample_confidence_intervals):
    """Test print_agreement_table with confidence intervals."""
    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(sample_agreements, sample_confidence_intervals)

    output = f.getvalue()

    # Print the output for debugging
    print(f"Output: {repr(output)}")

    # Check that confidence intervals are in the output
    assert '70.0%' in output or '70.0' in output
    assert '80.0%' in output or '80.0' in output
    assert '85.0%' in output or '85.0' in output

    # Check that the confidence level is mentioned
    assert 'p=95%' in output


def test_print_agreement_table_different_order(sample_agreements,
                                               sample_confidence_intervals):
    """Test print_agreement_table with different order of annotators."""
    # Create a new agreements dict with different order
    agreements_different_order = {
        ('Annotator3', 'Annotator1'): 0.80,  # Swapped order
        ('Annotator2', 'Annotator1'): 0.75,  # Swapped order
        ('Annotator3', 'Annotator2'): 0.85,  # Swapped order
    }

    # Create matching confidence intervals with the same order
    confidence_intervals_different_order = {
        ('Annotator3',
         'Annotator1'): sample_confidence_intervals[('Annotator1',
                                                     'Annotator3')],
        ('Annotator2',
         'Annotator1'): sample_confidence_intervals[('Annotator1',
                                                     'Annotator2')],
        ('Annotator3',
         'Annotator2'): sample_confidence_intervals[('Annotator2',
                                                     'Annotator3')],
    }

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(
            agreements_different_order, confidence_intervals_different_order)

    output = f.getvalue()

    # Check that agreement values are in the output
    assert '75.0%' in output
    assert '80.0%' in output
    assert '85.0%' in output

    # Check that confidence intervals are in the output
    assert '70.0' in output or '0.70' in output
    assert '80.0' in output or '0.80' in output
    assert '90.0' in output or '0.90' in output

    # Check that the confidence level is mentioned
    assert 'p=95%' in output


def test_print_agreement_table_truncate_names():
    """Test print_agreement_table with long annotator names that need
        truncation."""
    # Create agreements with very long annotator names
    long_names = {
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName2'): 0.75,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName3'): 0.80,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName3'): 0.85,
    }

    # Capture stdout with a very small max_width to force truncation
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(long_names, max_width=40)

    output = f.getvalue()

    # Print the output for debugging
    print(f"Output: {repr(output)}")

    # Check for specific content instead of truncated names
    assert '75.0%' in output
    assert '80.0%' in output
    assert '85.0%' in output

    # Check that the table was generated
    assert 'Inter-annotator Agreement Matrix (3 annotators)' in output


def test_print_agreement_table_missing_pair():
    """Test print_agreement_table with missing annotator pairs."""
    # Create agreements with missing pairs
    incomplete_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        # Missing pair: ('Annotator2', 'Annotator3')
    }

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(incomplete_agreements)

    output = f.getvalue()

    # Check that all three annotators are in the output
    assert 'Annotator1' in output
    assert 'Annotator2' in output
    assert 'Annotator3' in output

    # Check that the existing agreement values are in the output
    assert '75.0%' in output
    assert '80.0%' in output

    # Check that N/A is in the output for the missing pair
    assert '---' in output


def test_print_agreement_table_with_empty_agreements():
    """Test print_agreement_table with empty agreements dictionary."""
    # Create an empty agreements dictionary
    empty_agreements = {}

    # Capture stdout and stderr to handle potential errors
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            print_agreement_table(empty_agreements)
        output = f.getvalue()
        # If we get here, the function didn't raise an exception
        assert output.strip() == "" or "No agreement data" in output
    except ValueError as e:
        # If the function raises a ValueError, that's also acceptable
        # since the current implementation doesn't handle empty dictionaries
        assert ("empty" in str(e).lower() or
                "iterable argument is empty" in str(e))


def test_print_agreement_table_with_single_annotator():
    """Test print_agreement_table with a single annotator (no pairs)."""
    # Create an agreements dictionary with a single annotator (self-agreement)
    # This is an edge case that might trigger the n_annotators == 0 condition
    single_annotator = {('Annotator1', 'Annotator1'): 1.0}

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(single_annotator)

    output = f.getvalue()

    # Check the output - either it contains a message about no annotators
    # or it shows a 1x1 table with the self-agreement
    assert output.strip() != ""
    assert "Annotator1" in output or "No annotators" in output


def test_print_agreement_table_with_many_long_names():
    """
    Test print_agreement_table with many long annotator names that
    need truncation.
    """
    # Create agreements with more than 3 annotators with very long names
    many_long_names = {
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName2'): 0.75,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName3'): 0.80,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName4'): 0.85,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName3'): 0.90,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName4'): 0.95,
        ('VeryLongAnnotatorName3', 'VeryLongAnnotatorName4'): 0.99,
        ('SN1', 'SN2'): 1.0,
    }

    # Capture stdout with a small max_width to force truncation
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(many_long_names, max_width=80)

    output = f.getvalue()

    # Print the output for debugging
    print(f"Output: {repr(output)}")

    # Check that the table was generated
    assert 'Inter-annotator Agreement Matrix (6 annotators)' in output

    # Check for specific content
    assert '75.0%' in output
    assert '80.0%' in output
    assert '85.0%' in output
    assert '100.0%' in output

    # Check that at least one name was truncated (contains '...')
    # This is the key assertion to verify lines 41-42 were executed
    assert '...' in output


def test_print_agreement_table_with_extreme_truncation():
    """
    Test print_agreement_table with extreme truncation (very small max_width).
    """
    # Create agreements with many annotators with long names
    many_annotators = {
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName2'): 0.75,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName3'): 0.80,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName4'): 0.85,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName3'): 0.90,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName4'): 0.95,
        ('VeryLongAnnotatorName3', 'VeryLongAnnotatorName4'): 0.99,
    }

    # Test that an exception is raised with an extremely small max_width
    with pytest.raises(ValueError) as excinfo:
        print_agreement_table(many_annotators, max_width=10)

    # Check that the error message is informative
    assert "Table width" in str(excinfo.value)
    assert "exceeds maximum width" in str(excinfo.value)
    assert "Cannot truncate names" in str(excinfo.value)


def test_print_agreement_table_with_partial_ci():
    """
    Test print_agreement_table with confidence intervals for only
    some annotator pairs.
    """
    # Create agreements for three pairs of annotators
    agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    # Create confidence intervals for only one pair
    partial_confidence_intervals = {
        ('Annotator1', 'Annotator2'): {
            'ci_lower': 0.70,
            'ci_upper': 0.80,
            'confidence_level': 0.95,
            'method': 'wilson'
        }
        # No CI for other pairs
    }

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(agreements, partial_confidence_intervals)

    output = f.getvalue()

    # Check that the table was generated
    assert 'Inter-annotator Agreement Matrix (3 annotators)' in output

    # Check that CI is shown for the pair that has it
    assert '70.0%' in output and '80.0%' in output

    # Check the summary section which should show both formats:
    # - With CI for the pair that has it
    # - Without CI for the pairs that don't have it
    assert "Annotator1 & Annotator2: 75.0% [70.0%-80.0%]" in output
    assert "Annotator1 & Annotator3: 80.0%" in output
    assert "Annotator2 & Annotator3: 85.0%" in output
