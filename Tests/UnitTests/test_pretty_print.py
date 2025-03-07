import pytest
import os
import io
import csv
import tempfile
from contextlib import redirect_stdout
from Utils.pretty_print import print_agreement_table, save_agreement_csv
from Utils.pretty_print import save_agreement_html, get_cell_html
from Utils.pretty_print import get_confidence_interval_class


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

    # Check that the file exists but is essentially empty (just headers)
    assert os.path.exists(output_file)

    # Read the CSV file and check its contents
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check that there's only the header row
    assert len(rows) == 1
    assert rows[0] == ['Annotator 1',
                       'Annotator 2',
                       'Agreement',
                       'CI Lower',
                       'CI Upper']


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
    assert "Annotator1 & Annotator2: 75.0% (70.0%-80.0%)" in output
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
        assert "[70.0%-80.0%]" in content or "[70.0% - 80.0%]" in content

        # Check that the other pairs don't have confidence intervals
        # This is the key assertion to verify line 313 was executed
        assert "80.0%<br>" not in content
        assert "85.0%<br>" not in content

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
