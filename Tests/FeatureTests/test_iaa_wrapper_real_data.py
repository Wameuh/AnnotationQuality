import os
import pytest
import json
import tempfile
from src.iaa_wrapper import IAAWrapper


@pytest.fixture
def test_data_path():
    """Fixture providing path to test data file."""
    base_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "Tests", "Assets", "Reviews_annotated.csv")


@pytest.fixture
def iaa_args(test_data_path):
    """Fixture providing arguments for IAAWrapper."""
    return {
        'input_file': test_data_path,
        'output': None,  # Will be set in individual tests
        'output_format': 'console',  # Will be set in individual tests
        'log_level': 'info',
        'all': False,
        'raw': True,
        'cohen_kappa': True,
        'fleiss_kappa': True,
        'krippendorff_alpha': True,
        'f_measure': True,
        'icc': True,
        'bwfk': False,
        'dbcaa': False,
        'iou': False,
        'confidence_interval': 0.95,
        'confidence_method': 'wilson',
        'bootstrap_samples': 1000,
        'positive_class': None,
        'distance_threshold': 10.0,
        'bwfk_width': 5,
        'icc_form': '2,1',
        'pairwise': True
    }


def test_iaa_wrapper_html_output(test_data_path, iaa_args):
    """Test IAA wrapper with HTML output format."""
    # Create a temporary file for HTML output
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Configure wrapper for HTML output
        iaa_args['output'] = tmp_path
        iaa_args['output_format'] = 'html'

        # Create and run wrapper
        wrapper = IAAWrapper(iaa_args)
        wrapper.run()

        # Verify HTML output
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check HTML structure
            assert '<html' in content
            assert '<body' in content
            assert '</html>' in content
            # Check content
            assert 'IAA-Eval Results' in content
            assert 'raw' in content.lower()
            assert 'cohen_kappa' in content.lower()
            assert 'fleiss_kappa' in content.lower()
            # Check for confidence intervals
            assert 'Confidence' in content
            assert '95%' in content

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_iaa_wrapper_json_output(test_data_path, iaa_args):
    """Test IAA wrapper with JSON output format."""
    # Create a temporary file for JSON output
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Configure wrapper for JSON output
        iaa_args['output'] = tmp_path
        iaa_args['output_format'] = 'json'

        # Create and run wrapper
        wrapper = IAAWrapper(iaa_args)
        wrapper.run()

        # Verify JSON output
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Check structure
            assert isinstance(data, dict)
            assert 'raw' in data
            assert 'cohen_kappa' in data
            assert 'fleiss_kappa' in data
            # Check confidence intervals
            assert 'confidence_intervals' in data
            # Check interpretations
            assert 'interpretations' in data

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_iaa_wrapper_csv_output(test_data_path, iaa_args):
    """Test IAA wrapper with CSV output format."""
    # Create a temporary file for CSV output
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Configure wrapper for CSV output
        iaa_args['output'] = tmp_path
        iaa_args['output_format'] = 'csv'

        # Create and run wrapper
        wrapper = IAAWrapper(iaa_args)
        wrapper.run()

        # Verify CSV output
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check headers
            assert 'Annotator1_name' in content
            assert 'Annotator2_name' in content
            assert 'Agreement' in content
            # Check confidence interval columns
            assert 'Lower_Bound_interval' in content
            assert 'Upper_bound_interval' in content
            assert 'p' in content
            # Check data presence
            lines = content.split('\n')
            assert len(lines) > 1  # At least header and one data row

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_iaa_wrapper_results_consistency(test_data_path, iaa_args):
    """Test consistency of results across different output formats."""
    results = {}

    # Test each format and collect results
    for fmt in ['json', 'csv', 'html']:
        with tempfile.NamedTemporaryFile(suffix=f'.{fmt}',
                                         delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Configure wrapper
            iaa_args['output'] = tmp_path
            iaa_args['output_format'] = fmt

            # Create and run wrapper
            wrapper = IAAWrapper(iaa_args)
            wrapper.run()

            # Store results for comparison
            results[fmt] = {
                'path': tmp_path,
                'wrapper': wrapper
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Compare results across formats
    base_wrapper = results['json']['wrapper']
    for fmt, data in results.items():
        if fmt != 'json':
            current_wrapper = data['wrapper']
            # Compare raw results
            assert base_wrapper.results == current_wrapper.results
            # Compare confidence intervals
            assert (base_wrapper.confidence_intervals ==
                   current_wrapper.confidence_intervals)


def test_iaa_wrapper_error_handling(test_data_path, iaa_args):
    """Test error handling with invalid input/configuration."""
    # Test with non-existent input file
    iaa_args['input_file'] = 'non_existent_file.csv'
    wrapper = IAAWrapper(iaa_args)
    with pytest.raises(Exception):
        wrapper.run()

    # Test with invalid output format
    iaa_args['input_file'] = test_data_path
    iaa_args['output_format'] = 'invalid_format'
    wrapper = IAAWrapper(iaa_args)
    wrapper.run()  # Should fall back to console output without error
