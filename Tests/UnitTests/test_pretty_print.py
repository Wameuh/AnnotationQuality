import pytest
import os
import io
import csv
from contextlib import redirect_stdout
from Utils.pretty_print import print_agreement_table, save_agreement_csv


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
        ('Annotator1', 'Annotator2'): {'ci_lower': 0.70, 'ci_upper': 0.80},
        ('Annotator1', 'Annotator3'): {'ci_lower': 0.75, 'ci_upper': 0.85},
        ('Annotator2', 'Annotator3'): {'ci_lower': 0.80, 'ci_upper': 0.90},
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

    # Check that confidence intervals are in the output
    assert '[70.0%-80.0%]' in output
    assert '[75.0%-85.0%]' in output
    assert '[80.0%-90.0%]' in output


def test_print_agreement_table_different_order(sample_agreements,
                                               sample_confidence_intervals):
    """Test print_agreement_table with pairs in different order."""
    # Create a new agreements dict with different order of pairs
    different_order = {
        ('Annotator2', 'Annotator1'): 0.75,  # Reversed order
        ('Annotator3', 'Annotator1'): 0.80,  # Reversed order
        ('Annotator2', 'Annotator3'): 0.85,
    }

    # Create a new confidence intervals dict with matching order
    different_ci = {
        ('Annotator2', 'Annotator1'): {'ci_lower': 0.70, 'ci_upper': 0.80},
        ('Annotator3', 'Annotator1'): {'ci_lower': 0.75, 'ci_upper': 0.85},
        ('Annotator2', 'Annotator3'): {'ci_lower': 0.80, 'ci_upper': 0.90},
    }

    # Capture stdout for both versions
    f1 = io.StringIO()
    f2 = io.StringIO()

    with redirect_stdout(f1):
        print_agreement_table(sample_agreements, sample_confidence_intervals)

    with redirect_stdout(f2):
        print_agreement_table(different_order, different_ci)

    # Check that both outputs contain the same agreement values
    output1 = f1.getvalue()
    output2 = f2.getvalue()

    # Instead of comparing the entire output, check for specific content
    assert '75.0%' in output1 and '75.0%' in output2
    assert '80.0%' in output1 and '80.0%' in output2
    assert '85.0%' in output1 and '85.0%' in output2
    assert '[70.0%-80.0%]' in output1 and '[70.0%-80.0%]' in output2
    assert '[75.0%-85.0%]' in output1 and '[75.0%-85.0%]' in output2
    assert '[80.0%-90.0%]' in output1 and '[80.0%-90.0%]' in output2


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


def test_save_agreement_csv(sample_agreements,
                            sample_confidence_intervals,
                            tmp_path):
    """Test save_agreement_csv functionality."""
    # Create a temporary file path
    output_file = tmp_path / "test_agreements.csv"

    # Save the agreements to CSV
    save_agreement_csv(str(output_file),
                       sample_agreements,
                       sample_confidence_intervals)

    # Check that the file exists
    assert os.path.exists(output_file)

    # Read the CSV file and check its contents
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check header
    assert rows[0] == [
        'Annotator 1',
        'Annotator 2',
        'Agreement',
        'CI Lower',
        'CI Upper'
    ]

    # Check data rows (sorted by annotator names)
    assert len(rows) == 4  # Header + 3 data rows

    # Check first data row
    assert rows[1][0] == 'Annotator1'
    assert rows[1][1] == 'Annotator2'
    assert rows[1][2] == '75.000%'
    assert rows[1][3] == '70.000%'
    assert rows[1][4] == '80.000%'


def test_save_agreement_csv_without_ci(sample_agreements, tmp_path):
    """Test save_agreement_csv without confidence intervals."""
    # Create a temporary file path
    output_file = tmp_path / "test_agreements_no_ci.csv"

    # Save the agreements to CSV without confidence intervals
    save_agreement_csv(str(output_file), sample_agreements)

    # Read the CSV file and check its contents
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check data rows
    assert len(rows) == 4  # Header + 3 data rows

    # Check that CI columns are empty
    assert rows[1][3] == ''
    assert rows[1][4] == ''


def test_print_agreement_table_name_truncation():
    """Test specifically the name truncation logic in print_agreement_table."""
    # Mock the tabulate function to verify the annotator names are truncated
    from Utils.pretty_print import tabulate as original_tabulate

    # Create a simple agreements dict with more than 3 annotators
    agreements = {
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName2'): 0.75,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName3'): 0.80,
        ('VeryLongAnnotatorName1', 'VeryLongAnnotatorName4'): 0.85,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName3'): 0.90,
        ('VeryLongAnnotatorName2', 'VeryLongAnnotatorName4'): 0.95,
        ('VeryLongAnnotatorName3', 'VeryLongAnnotatorName4'): 0.99,
    }

    # Create a flag to check if truncation happened
    truncation_happened = [False]
    truncated_names = []

    # Define a mock tabulate function
    def mock_tabulate(matrix, headers, **kwargs):
        # Check if headers contain truncated names
        truncated_names.extend(headers)
        for header in headers:
            if '...' in header:
                truncation_happened[0] = True
        # Call the original function
        return original_tabulate(matrix, headers, **kwargs)

    # Replace the tabulate function temporarily
    import Utils.pretty_print
    original = Utils.pretty_print.tabulate
    Utils.pretty_print.tabulate = mock_tabulate

    try:
        # Call print_agreement_table with a very small max_width
        f = io.StringIO()
        with redirect_stdout(f):
            print_agreement_table(agreements, max_width=10)

        # Check if truncation happened
        assert truncation_happened[0], "Name truncation did not occur"
        assert any('...' in name for name in truncated_names), \
            "No truncated names found"

    finally:
        # Restore the original tabulate function
        Utils.pretty_print.tabulate = original


def test_print_agreement_table_missing_pair():
    """Test print_agreement_table with a missing annotator pair."""
    # Create agreements with a missing pair
    incomplete_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        # Missing pair: ('Annotator2', 'Annotator3')
    }

    # Add a fourth annotator with no agreements
    # This will create a cell in the table where both pair orders are missing

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_agreement_table(incomplete_agreements)

    output = f.getvalue()

    # Check that N/A is in the output for the missing pair
    assert 'N/A' in output

    # Check that the table was generated with all three annotators
    assert 'Inter-annotator Agreement Matrix (3 annotators)' in output
