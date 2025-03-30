import pytest
import os
import csv
import tempfile
from Utils.pretty_print_csv import (
    save_agreement_csv, export_agreement_csv, export_multi_agreement_csv
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

            # Check values for specific pairs
            if ann1 == 'Annotator1' and ann2 == 'Annotator2':
                assert float(row[2]) == 0.75  # Raw Agreement value
                assert float(row[3]) == 0.70  # Raw CI lower
                assert float(row[4]) == 0.80  # Raw CI upper
                assert float(row[5]) == 0.05  # Raw p-value

                assert float(row[6]) == 0.65  # Kappa value
                assert float(row[7]) == 0.60  # Kappa CI lower
                assert float(row[8]) == 0.70  # Kappa CI upper
                assert float(row[9]) == 0.05  # Kappa p-value

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

        # Check data for specific pairs
        for row in rows[1:]:
            ann1, ann2 = row[0], row[1]
            if ann1 == 'Annotator1' and ann2 == 'Annotator2':
                assert float(row[2]) == 0.75  # Raw Agreement value
                assert float(row[3]) == 0.65  # Kappa value

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
    Test exporting multiple agreement results with missing pairs and
    confidence intervals.
    This specifically tests line 231 in pretty_print_csv.py where N/A values
    are added for confidence intervals when a method doesn't have data for
    a specific pair.
    """
    # Create sample agreements with different pairs
    raw_agreements = {
        ('Annotator1', 'Annotator2'): 0.75,
        ('Annotator1', 'Annotator3'): 0.80,
        ('Annotator2', 'Annotator3'): 0.85,
    }

    # Kappa is missing a pair
    kappa_agreements = {
        ('Annotator1', 'Annotator2'): 0.65,
        # Missing ('Annotator1', 'Annotator3')
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
        # Missing CI for ('Annotator1', 'Annotator3')
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
        # Export to CSV with confidence intervals
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

        # Find the row for the missing pair
        missing_pair_row = None
        for row in rows[1:]:
            if row[0] == 'Annotator1' and row[1] == 'Annotator3':
                missing_pair_row = row
                break

        # Check that the missing pair has N/A for Cohen's Kappa and CI values
        assert missing_pair_row is not None

        # Raw Agreement values should be present
        assert missing_pair_row[2] == "0.8000"  # Raw Agreement value
        assert missing_pair_row[3] == "0.7500"  # Raw CI lower
        assert missing_pair_row[4] == "0.8500"  # Raw CI upper
        assert missing_pair_row[5] == "0.05"    # Raw p-value

        # Cohen's Kappa values should all be N/A
        assert missing_pair_row[6] == "N/A"     # Cohen's Kappa should be N/A
        assert missing_pair_row[7] == "N/A"     # Kappa CI lower should be N/A
        assert missing_pair_row[8] == "N/A"     # Kappa CI upper should be N/A
        assert missing_pair_row[9] == "N/A"     # Kappa p-value should be N/A

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
