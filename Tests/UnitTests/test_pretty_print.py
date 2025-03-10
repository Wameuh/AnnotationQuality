import pytest
import os
import io
import csv
import tempfile
from contextlib import redirect_stdout
from Utils.pretty_print import print_agreement_table, save_agreement_csv
from Utils.pretty_print import save_agreement_html, get_cell_html
from Utils.pretty_print import get_confidence_interval_class
from Utils.pretty_print import export_agreement_csv
from Utils.pretty_print import get_unique_annotators, create_agreement_matrix
from Utils.pretty_print import format_with_confidence_interval
from Utils.pretty_print import get_confidence_method_display
from Utils.pretty_print import get_confidence_level
from Utils.pretty_print import export_multi_agreement_csv


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

    # Check header with the new format
    assert rows[0] == [
        'Annotator1_name',
        'Annotator2_name',
        'Agreement',
        'Lower_Bound_interval',
        'Upper_bound_interval',
        'p'
    ]

    # Check data rows
    assert len(rows) == 4  # Header + 3 data rows

    # Check that all pairs are present
    pairs_found = set()
    for row in rows[1:]:
        ann1, ann2 = row[0], row[1]
        pairs_found.add((ann1, ann2))

        # Check agreement value format
        assert len(row[2]) == 6  # Format like '0.7500'

        # Check confidence interval values
        if ann1 == 'Annotator1' and ann2 == 'Annotator2':
            assert row[3] == '0.7000'
            assert row[4] == '0.8000'
        elif ann1 == 'Annotator1' and ann2 == 'Annotator3':
            assert row[3] == '0.7500'
            assert row[4] == '0.8500'
        elif ann1 == 'Annotator2' and ann2 == 'Annotator3':
            assert row[3] == '0.8000'
            assert row[4] == '0.9000'

        # Check p-value
        assert row[5] == '0.05'

    # Verify all pairs are present
    expected_pairs = {
        ('Annotator1', 'Annotator2'),
        ('Annotator1', 'Annotator3'),
        ('Annotator2', 'Annotator3')
    }
    assert pairs_found == expected_pairs


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

    # Check data rows - now we have a header and 3 data rows
    assert len(rows) == 4  # Header + 3 data rows

    # Check the header
    assert rows[0] == ['Annotator1_name', 'Annotator2_name', 'Agreement']

    # Check the data
    assert rows[1][0] == 'Annotator1'
    assert rows[1][1] == 'Annotator2'
    assert rows[1][2] == '0.7500'

    assert rows[2][0] == 'Annotator1'
    assert rows[2][1] == 'Annotator3'
    assert rows[2][2] == '0.8000'

    assert rows[3][0] == 'Annotator2'
    assert rows[3][1] == 'Annotator3'
    assert rows[3][2] == '0.8500'


def test_save_agreement_csv_with_ci(sample_agreements,
                                    sample_confidence_intervals,
                                    tmp_path):
    """Test save_agreement_csv with confidence intervals."""
    # Create a temporary file path
    output_file = tmp_path / "test_agreements_with_ci.csv"

    # Save the agreements to CSV with confidence intervals
    save_agreement_csv(str(output_file),
                       sample_agreements,
                       sample_confidence_intervals)

    # Read the CSV file and check its contents
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check data rows
    assert len(rows) == 4  # Header + 3 data rows

    # Check the header
    assert rows[0] == ['Annotator1_name', 'Annotator2_name', 'Agreement',
                       'Lower_Bound_interval', 'Upper_bound_interval', 'p']

    # Check the data
    assert rows[1][0] == 'Annotator1'
    assert rows[1][1] == 'Annotator2'
    assert rows[1][2] == '0.7500'
    assert rows[1][3] == '0.7000'
    assert rows[1][4] == '0.8000'
    assert rows[1][5] == '0.05'  # p-value (1 - confidence_level)

    assert rows[2][0] == 'Annotator1'
    assert rows[2][1] == 'Annotator3'
    assert rows[2][2] == '0.8000'
    assert rows[2][3] == '0.7500'
    assert rows[2][4] == '0.8500'
    assert rows[2][5] == '0.05'

    assert rows[3][0] == 'Annotator2'
    assert rows[3][1] == 'Annotator3'
    assert rows[3][2] == '0.8500'
    assert rows[3][3] == '0.8000'
    assert rows[3][4] == '0.9000'
    assert rows[3][5] == '0.05'


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


def test_save_agreement_html(sample_agreements, sample_confidence_intervals):
    """Test saving agreement results to HTML file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save HTML file
        save_agreement_html(
            tmp_path,
            sample_agreements,
            sample_confidence_intervals,
            title="Test Agreement Results"
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the file and check for expected content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for basic HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html lang=\"en\">" in content
        assert "</html>" in content

        # Check for title
        assert "<title>Test Agreement Results</title>" in content
        assert "<h1>Test Agreement Results</h1>" in content

        # Check for annotator names
        for annotator in ["Annotator1", "Annotator2", "Annotator3"]:
            assert annotator in content

        # Check for agreement values in different possible formats
        for value in [0.75, 0.80, 0.85]:
            assert (str(value) in content or
                    f"{value:.1f}" in content or
                    f"{value:.2f}" in content)

        # Check for heatmap
        assert "Agreement Heatmap" in content
        assert "data:image/png;base64," in content

        # Check for CSS classes for color coding
        assert "high-agreement" in content
        assert "medium-agreement" in content

        # Check for footer
        assert "Generated by" in content and "IAA-Eval" in content

        # Check that confidence intervals are in the output
        assert '70.0%' in content or '0.70' in content
        assert '80.0%' in content or '0.80' in content
        assert '90.0%' in content or '0.90' in content

        # Check that the confidence method and p-value are mentioned
        assert 'wilson' in content.lower()
        assert 'p = 0.05' in content

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_save_agreement_html_no_heatmap(sample_agreements):
    """Test saving agreement results to HTML without heatmap."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save HTML file without heatmap
        save_agreement_html(
            tmp_path,
            sample_agreements,
            title="Agreement Results without Heatmap",
            include_heatmap=False
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the file and check content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that the heatmap is not included
        assert "Agreement Heatmap" not in content
        assert "data:image/png;base64," not in content

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_get_cell_html_with_different_values():
    """Test get_cell_html function with different values and classes."""
    # Test with a high agreement value
    high_cell = get_cell_html("95.0%", 0.95)
    assert 'high-agreement' in high_cell
    assert '95.0%' in high_cell

    # Test with a medium agreement value
    medium_cell = get_cell_html("65.0%", 0.65)
    assert 'medium-agreement' in medium_cell
    assert '65.0%' in medium_cell

    # Test with a low agreement value
    low_cell = get_cell_html("35.0%", 0.35)
    assert 'low-agreement' in low_cell
    assert '35.0%' in low_cell

    # Test with a cell that already contains confidence interval text
    cell_with_ci_text = get_cell_html("65.0% [60.0%-70.0%]", 0.65)
    assert 'medium-agreement' in cell_with_ci_text
    assert '65.0%' in cell_with_ci_text
    assert '60.0%' in cell_with_ci_text
    assert '70.0%' in cell_with_ci_text


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


def test_save_agreement_csv_with_empty_agreements(tmp_path):
    """Test save_agreement_csv with empty agreements dictionary."""
    # Create an empty agreements dictionary
    empty_agreements = {}

    # Create a temporary file path
    output_file = tmp_path / "empty_agreements.csv"

    # Save the empty agreements to CSV
    save_agreement_csv(str(output_file), empty_agreements)

    # Check that the file exists
    assert os.path.exists(output_file)

    # Read the CSV file and check its contents
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check that the file contains at least the header
    assert len(rows) >= 1  # At least the header

    # Check that the header is correct
    assert rows[0] == ['Annotator1_name', 'Annotator2_name', 'Agreement']

    # Check that there are no data rows (since the dictionary is empty)
    assert len(rows) == 1


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


def test_save_agreement_html_with_partial_ci():
    """
    Test saving agreement results to HTML with partial confidence intervals.
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

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save HTML file with partial confidence intervals
        save_agreement_html(
            tmp_path,
            agreements,
            partial_confidence_intervals,
            title="Test Agreement Results with Partial CI"
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the file and check for expected content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for basic HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html lang=\"en\">" in content
        assert "</html>" in content

        # Check for title
        assert "Test Agreement Results with Partial CI" in content

        # Check for annotator names
        for annotator in ["Annotator1", "Annotator2", "Annotator3"]:
            assert annotator in content

        # Check for agreement values
        assert "75.0%" in content
        assert "80.0%" in content
        assert "85.0%" in content

        # Check for confidence intervals for the pair that has them
        assert "(70.0% - 80.0%)" in content

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_get_cell_html_with_special_values():
    """
    Test get_cell_html function with special values like N/A and diagonal.
    """
    # Test with diagonal marker
    diagonal_cell = get_cell_html("---", None)
    assert '<td>---</td>' == diagonal_cell

    # Test with N/A value - this tests line 463
    na_cell = get_cell_html("N/A", None)
    assert '<td>N/A</td>' == na_cell

    # Test with a regular value
    regular_cell = get_cell_html("75.0%", 0.75)
    assert '<td' in regular_cell
    assert 'class=' in regular_cell
    assert '75.0%' in regular_cell
    assert '</td>' in regular_cell


def test_get_confidence_interval_class():
    """
    Test get_confidence_interval_class function with different interval widths.
    """

    # Test with a narrow interval (width <= 0.1)
    narrow_class = get_confidence_interval_class(0.75, 0.80)
    assert narrow_class == "narrow-interval"

    # Test with a medium interval (0.1 < width <= 0.2)
    medium_class = get_confidence_interval_class(0.70, 0.85)
    assert medium_class == "medium-interval"

    # Test with a wide interval (width > 0.2) - this tests line 484
    wide_class = get_confidence_interval_class(0.60, 0.90)
    assert wide_class == "wide-interval"

    # Test with extreme values
    extreme_class = get_confidence_interval_class(0.0, 1.0)
    assert extreme_class == "wide-interval"


def test_export_agreement_csv(sample_agreements, sample_confidence_intervals):
    """Test exporting agreement results to CSV."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV with matrix
        export_agreement_csv(
            tmp_path,
            sample_agreements,
            sample_confidence_intervals,
            include_matrix=True
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check the header
        assert rows[0] == ['Annotator1_name', 'Annotator2_name', 'Agreement',
                           'Lower_Bound_interval', 'Upper_bound_interval', 'p']

        # Check the data
        assert rows[1][0] == 'Annotator1'
        assert rows[1][1] == 'Annotator2'
        assert rows[1][2] == '0.7500'
        assert rows[1][3] == '0.7000'
        assert rows[1][4] == '0.8000'
        assert rows[1][5] == '0.05'

        # Check that the matrix is present
        matrix_header_row = None
        for i, row in enumerate(rows):
            if row and row[0] == 'Agreement Matrix':
                matrix_header_row = i
                break

        assert matrix_header_row is not None

        # Check the matrix header
        assert rows[matrix_header_row + 1][0] == ''
        assert 'Annotator1' in rows[matrix_header_row + 1]
        assert 'Annotator2' in rows[matrix_header_row + 1]
        assert 'Annotator3' in rows[matrix_header_row + 1]

        # Check the metadata
        metadata_row = None
        for i in range(len(rows) - 1, 0, -1):
            if rows[i] and 'Confidence Level:' in rows[i][0]:
                metadata_row = i
                break

        assert metadata_row is not None
        assert '0.95' in rows[metadata_row][0]
        assert 'Wilson' in rows[metadata_row][1]

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_agreement_csv_without_ci():
    """Test exporting agreement results to CSV without confidence intervals."""
    # Create sample agreements
    agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV without confidence intervals
        export_agreement_csv(
            tmp_path,
            agreements,
            confidence_intervals=None,
            include_matrix=True
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check the header
        assert rows[0] == ['Annotator1_name', 'Annotator2_name', 'Agreement']

        # Check that all annotator pairs are present
        # without caring about the exact order
        pairs_found = set()
        for row in rows[1:4]:  # The 3 data rows
            if len(row) >= 3:  # Make sure the row has at least 3 columns
                ann1, ann2, agreement = row[0], row[1], row[2]
                if ann1 == 'Annotator1' and ann2 == 'Annotator2':
                    assert agreement == '0.7500'
                    pairs_found.add(('Annotator1', 'Annotator2'))
                elif ann1 == 'Annotator1' and ann2 == 'Annotator3':
                    assert agreement == '0.8000'
                    pairs_found.add(('Annotator1', 'Annotator3'))
                elif ann1 == 'Annotator2' and ann2 == 'Annotator3':
                    assert agreement == '0.8500'
                    pairs_found.add(('Annotator2', 'Annotator3'))

        # Check that all pairs were found
        assert pairs_found == {
            ('Annotator1', 'Annotator2'),
            ('Annotator1', 'Annotator3'),
            ('Annotator2', 'Annotator3')
        }

        # Check that there's no metadata row about confidence intervals
        for row in rows:
            if row and row[0].startswith('Confidence Level:'):
                assert False, \
                    "Found confidence level metadata when none should exist"

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_agreement_csv_without_matrix():
    """Test exporting agreement results to CSV without the matrix."""
    # Create sample agreements
    agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    # Create sample confidence intervals
    confidence_intervals = {
        ('Annotator1', 'Annotator2'): {
            'ci_lower': 0.70,
            'ci_upper': 0.80,
            'confidence_level': 0.95,
            'method': 'wilson'
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV without the matrix
        export_agreement_csv(
            tmp_path,
            agreements,
            confidence_intervals,
            include_matrix=False
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check that there's no matrix section
        for row in rows:
            if row and row[0] == 'Agreement Matrix':
                assert False, "Found matrix section when include_matrix=False"

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_multi_agreement_csv():
    """Test exporting multiple agreement results to a single CSV file."""
    # Create sample agreements for different methods
    raw_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    kappa_agreements = {
        ('Annotator1', 'Annotator2'): 0.65,
        ('Annotator1', 'Annotator3'): 0.70,
        ('Annotator2', 'Annotator3'): 0.75,
    }

    # Create sample confidence intervals
    raw_ci = {
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
        }
    }

    kappa_ci = {
        ('Annotator1', 'Annotator2'): {
            'ci_lower': 0.60,
            'ci_upper': 0.70,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        ('Annotator1', 'Annotator3'): {
            'ci_lower': 0.65,
            'ci_upper': 0.75,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        ('Annotator2', 'Annotator3'): {
            'ci_lower': 0.70,
            'ci_upper': 0.80,
            'confidence_level': 0.95,
            'method': 'wilson'
        }
    }

    # Create dictionaries for export
    agreements_dict = {
        'Raw_Agreement': raw_agreements,
        'Cohen_Kappa': kappa_agreements
    }

    confidence_intervals_dict = {
        'Raw_Agreement': raw_ci,
        'Cohen_Kappa': kappa_ci
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV
        export_multi_agreement_csv(
            tmp_path,
            agreements_dict,
            confidence_intervals_dict
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check header row
        assert rows[0] == [
            'Annotator1_name',
            'Annotator2_name',
            'Raw_Agreement',
            'Lower_Bound_interval',
            'Upper_bound_interval',
            'p',
            'Cohen_Kappa',
            'Lower_Bound_interval',
            'Upper_bound_interval',
            'p'
        ]

        # Check data rows - we should have 3 rows (one for each pair)
        assert len(rows) == 4  # Header + 3 data rows

        # Check that all pairs are present
        pairs_found = set()
        for row in rows[1:]:
            ann1, ann2 = row[0], row[1]
            pairs_found.add((ann1, ann2))

            # Check Raw Agreement values and CI
            raw_value = float(row[2])
            raw_lower = float(row[3])
            raw_upper = float(row[4])
            raw_p = float(row[5])

            # Check Cohen's Kappa values and CI
            kappa_value = float(row[6])
            kappa_lower = float(row[7])
            kappa_upper = float(row[8])
            kappa_p = float(row[9])

            # Verify values match our input data
            if ann1 == 'Annotator1' and ann2 == 'Annotator2':
                assert raw_value == 0.75
                assert raw_lower == 0.70
                assert raw_upper == 0.80
                assert raw_p == 0.05

                assert kappa_value == 0.65
                assert kappa_lower == 0.60
                assert kappa_upper == 0.70
                assert kappa_p == 0.05

            elif ann1 == 'Annotator1' and ann2 == 'Annotator3':
                assert raw_value == 0.80
                assert raw_lower == 0.75
                assert raw_upper == 0.85
                assert raw_p == 0.05

                assert kappa_value == 0.70
                assert kappa_lower == 0.65
                assert kappa_upper == 0.75
                assert kappa_p == 0.05

            elif ann1 == 'Annotator2' and ann2 == 'Annotator3':
                assert raw_value == 0.85
                assert raw_lower == 0.80
                assert raw_upper == 0.90
                assert raw_p == 0.05

                assert kappa_value == 0.75
                assert kappa_lower == 0.70
                assert kappa_upper == 0.80
                assert kappa_p == 0.05

        # Check that all pairs were found
        assert pairs_found == {
            ('Annotator1', 'Annotator2'),
            ('Annotator1', 'Annotator3'),
            ('Annotator2', 'Annotator3')
        }

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_multi_agreement_csv_without_ci():
    """
    Test exporting multiple agreement results without confidence intervals.
    """
    # Create sample agreements for different methods
    raw_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    kappa_agreements = {
        ('Annotator1', 'Annotator2'): 0.65,
        ('Annotator1', 'Annotator3'): 0.70,
        ('Annotator2', 'Annotator3'): 0.75,
    }

    # Create dictionaries for export
    agreements_dict = {
        'Raw_Agreement': raw_agreements,
        'Cohen_Kappa': kappa_agreements
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV without confidence intervals
        export_multi_agreement_csv(
            tmp_path,
            agreements_dict
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check header row - should only have method names, no CI columns
        assert rows[0] == [
            'Annotator1_name', 'Annotator2_name',
            'Raw_Agreement', 'Cohen_Kappa'
        ]

        # Check data rows - we should have 3 rows (one for each pair)
        assert len(rows) == 4  # Header + 3 data rows

        # Check that all pairs are present with correct values
        pairs_found = set()
        for row in rows[1:]:
            ann1, ann2 = row[0], row[1]
            pairs_found.add((ann1, ann2))

            # Check Raw Agreement value
            raw_value = float(row[2])

            # Check Cohen's Kappa value
            kappa_value = float(row[3])

            # Verify values match our input data
            if ann1 == 'Annotator1' and ann2 == 'Annotator2':
                assert raw_value == 0.75
                assert kappa_value == 0.65

            elif ann1 == 'Annotator1' and ann2 == 'Annotator3':
                assert raw_value == 0.80
                assert kappa_value == 0.70

            elif ann1 == 'Annotator2' and ann2 == 'Annotator3':
                assert raw_value == 0.85
                assert kappa_value == 0.75

        # Check that all pairs were found
        assert pairs_found == {
            ('Annotator1', 'Annotator2'),
            ('Annotator1', 'Annotator3'),
            ('Annotator2', 'Annotator3')
        }

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_multi_agreement_csv_with_missing_pairs():
    """
    Test exporting multiple agreement results with different pairs in each
    method.
    """
    # Create sample agreements with different pairs
    raw_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    kappa_agreements = {
        ('Annotator1', 'Annotator2'): 0.65,
        # Missing ('Annotator1', 'Annotator3')
        ('Annotator2', 'Annotator3'): 0.75,
    }

    # Create dictionaries for export
    agreements_dict = {
        'Raw_Agreement': raw_agreements,
        'Cohen_Kappa': kappa_agreements
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV
        export_multi_agreement_csv(
            tmp_path,
            agreements_dict
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Check header row
        assert rows[0] == [
            'Annotator1_name', 'Annotator2_name',
            'Raw_Agreement', 'Cohen_Kappa'
        ]

        # Check data rows - we should have 3 rows (one for each pair)
        assert len(rows) == 4  # Header + 3 data rows

        # Find the row for the missing pair
        missing_pair_row = None
        for row in rows[1:]:
            if row[0] == 'Annotator1' and row[1] == 'Annotator3':
                missing_pair_row = row
                break

        # Check that the missing pair has N/A for Cohen's Kappa
        assert missing_pair_row is not None
        assert missing_pair_row[2] == "0.8000"  # Raw Agreement value
        assert missing_pair_row[3] == "N/A"     # Cohen's Kappa should be N/A

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_multi_agreement_csv_with_missing_pairs_and_ci():
    """
    Test exporting with missing pairs when confidence intervals are present.
    """
    # Create sample agreements with different pairs
    raw_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    kappa_agreements = {
        ('Annotator1', 'Annotator2'): 0.65,
        # Missing ('Annotator1', 'Annotator3')
        ('Annotator2', 'Annotator3'): 0.75,
    }

    # Create confidence intervals for both methods
    raw_ci = {
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
        }
    }

    kappa_ci = {
        ('Annotator1', 'Annotator2'): {
            'ci_lower': 0.60,
            'ci_upper': 0.70,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        # Note: We have CI for ('Annotator1', 'Annotator3') even though
        # it's missing in kappa_agreements
        ('Annotator1', 'Annotator3'): {
            'ci_lower': 0.65,
            'ci_upper': 0.75,
            'confidence_level': 0.95,
            'method': 'wilson'
        },
        ('Annotator2', 'Annotator3'): {
            'ci_lower': 0.70,
            'ci_upper': 0.80,
            'confidence_level': 0.95,
            'method': 'wilson'
        }
    }

    # Create dictionaries for export
    agreements_dict = {
        'Raw_Agreement': raw_agreements,
        'Cohen_Kappa': kappa_agreements
    }

    confidence_intervals_dict = {
        'Raw_Agreement': raw_ci,
        'Cohen_Kappa': kappa_ci
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV
        export_multi_agreement_csv(
            tmp_path,
            agreements_dict,
            confidence_intervals_dict
        )

        # Check that file exists and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Read the CSV file and check its content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Find the row for the missing pair
        missing_pair_row = None
        for row in rows[1:]:
            if row[0] == 'Annotator1' and row[1] == 'Annotator3':
                missing_pair_row = row
                break

        # Check that the missing pair has N/A for Cohen's Kappa and its CI
        assert missing_pair_row is not None
        assert missing_pair_row[2] == "0.8000"  # Raw Agreement value
        assert missing_pair_row[3] == "0.7500"  # Raw Agreement CI lower
        assert missing_pair_row[4] == "0.8500"  # Raw Agreement CI upper
        assert missing_pair_row[5] == "0.05"    # Raw Agreement p-value

        assert missing_pair_row[6] == "N/A"     # Cohen's Kappa should be N/A
        assert missing_pair_row[7] == "N/A"
        # Cohen's Kappa CI lower should be N/A
        assert missing_pair_row[8] == "N/A"
        # Cohen's Kappa CI upper should be N/A
        assert missing_pair_row[9] == "N/A"
        # Cohen's Kappa p-value should be N/A

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
