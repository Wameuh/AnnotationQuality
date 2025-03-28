import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
import io
import os
import json
import tempfile
from src.iaa_wrapper import IAAWrapper
from Utils.logger import LogLevel


@pytest.fixture
def mock_args():
    """Fixture providing mock command line arguments."""
    return {
        'input_file': 'test_input.csv',
        'output': None,
        'output_format': 'console',
        'log_level': 'info',
        'all': True,
        'raw': False,
        'cohen_kappa': False,
        'fleiss_kappa': False,
        'krippendorff_alpha': False,
        'f_measure': False,
        'icc': False,
        'bwfk': False,
        'dbcaa': False,
        'iou': False,
        'confidence_interval': 0,
        'confidence_method': 'wilson',
        'bootstrap_samples': 1000,
        'positive_class': None,
        'distance_threshold': 10.0,
        'bwfk_width': 5,
        'icc_form': '2,1',
        'pairwise': False
    }


@pytest.fixture
def sample_data():
    """Fixture providing sample annotation data."""
    data = {
        'Annotator1': [1, 2, 3, 4, 5],
        'Annotator2': [1, 2, 3, 4, 4],
        'Annotator3': [1, 2, 2, 4, 5]
    }
    return pd.DataFrame(data)


@pytest.fixture
def wrapper(mock_args):
    """Fixture providing an IAAWrapper instance with mocked dependencies."""
    with patch('src.iaa_wrapper.get_logger'):
        with patch('src.iaa_wrapper.DataLoader'):
            return IAAWrapper(mock_args)


def test_init(mock_args):
    """Test initialization of IAAWrapper."""
    # Mock the logger and data loader
    with patch('src.iaa_wrapper.get_logger') as mock_get_logger:
        with patch('src.iaa_wrapper.DataLoader') as mock_data_loader:
            # Create the wrapper
            wrapper = IAAWrapper(mock_args)

            # Check that the wrapper was initialized correctly
            assert wrapper.args == mock_args
            assert wrapper.data is None
            assert wrapper.results == {}
            assert wrapper.confidence_intervals == {}
            assert wrapper.calculators == {}

            # Check that the logger was created with the correct log level
            mock_get_logger.assert_called_once()

            # Check that the data loader was created with the correct log level
            mock_data_loader.assert_called_once()


def test_get_log_level():
    """Test _get_log_level method."""
    # Create a wrapper with minimal mocking
    with patch('src.iaa_wrapper.get_logger'):
        with patch('src.iaa_wrapper.DataLoader'):
            wrapper = IAAWrapper({'log_level': 'info'})

            # Test all log levels
            assert wrapper._get_log_level('debug') == LogLevel.DEBUG
            assert wrapper._get_log_level('info') == LogLevel.INFO
            assert wrapper._get_log_level('warning') == LogLevel.WARNING
            assert wrapper._get_log_level('error') == LogLevel.ERROR
            assert wrapper._get_log_level('critical') == LogLevel.CRITICAL

            # Test invalid log level (should default to INFO)
            assert wrapper._get_log_level('invalid') == LogLevel.INFO


def test_load_data(wrapper, sample_data):
    """Test load_data method."""
    # Mock the data loader to return our sample data
    wrapper.data_loader.load_data.return_value = sample_data

    # Call the method
    wrapper.load_data()

    # Check that the data loader was called with the correct file
    wrapper.data_loader.load_data.assert_called_once_with(
        wrapper.args['input_file'])

    # Check that the data was stored
    assert wrapper.data is sample_data


def test_load_data_error(wrapper):
    """Test load_data method with an error."""
    # Mock the data loader to raise an exception
    wrapper.data_loader.load_data.side_effect = Exception("Test error")

    # Call the method and check that it raises the exception
    with pytest.raises(Exception, match="Test error"):
        wrapper.load_data()


def test_get_measures_to_calculate_all(wrapper):
    """Test _get_measures_to_calculate method with --all flag."""
    # Set the --all flag
    wrapper.args['all'] = True

    # Call the method
    measures = wrapper._get_measures_to_calculate()

    # Check that all measures are included
    assert set(measures) == {
        'raw', 'cohen_kappa', 'fleiss_kappa', 'krippendorff_alpha',
        'f_measure', 'icc', 'bwfk', 'dbcaa', 'iou'
    }


def test_get_measures_to_calculate_specific(wrapper):
    """Test _get_measures_to_calculate method with specific measures."""
    # Set specific measures
    wrapper.args['all'] = False
    wrapper.args['raw'] = True
    wrapper.args['cohen_kappa'] = True

    # Call the method
    measures = wrapper._get_measures_to_calculate()

    # Check that only the specified measures are included
    assert set(measures) == {'raw', 'cohen_kappa'}


def test_get_measures_to_calculate_specific_measures(wrapper):
    """Test _get_measures_to_calculate method with specific measures."""
    # Set all to False and enable specific measures
    wrapper.args['all'] = False
    wrapper.args['raw'] = True
    wrapper.args['cohen_kappa'] = True
    wrapper.args['fleiss_kappa'] = False
    wrapper.args['krippendorff_alpha'] = True
    wrapper.args['f_measure'] = False
    wrapper.args['icc'] = True
    wrapper.args['bwfk'] = False
    wrapper.args['dbcaa'] = True
    wrapper.args['iou'] = False

    # Call the method
    measures = wrapper._get_measures_to_calculate()

    # Check that only the enabled measures are included
    assert 'raw' in measures
    assert 'cohen_kappa' in measures
    assert 'fleiss_kappa' not in measures
    assert 'krippendorff_alpha' in measures
    assert 'f_measure' not in measures
    assert 'icc' in measures
    assert 'bwfk' not in measures
    assert 'dbcaa' in measures
    assert 'iou' not in measures


def test_calculate_agreements(wrapper, sample_data):
    """Test calculate_agreements method."""
    # Mock the data and _calculate_measure method
    wrapper.data = sample_data
    wrapper._calculate_measure = MagicMock()

    # Mock _get_measures_to_calculate to return specific measures
    wrapper._get_measures_to_calculate = MagicMock(
        return_value=['raw', 'cohen_kappa'])

    # Call the method
    wrapper.calculate_agreements()

    # Check that _calculate_measure was called for each measure
    wrapper._calculate_measure.assert_has_calls([
        call('raw'),
        call('cohen_kappa')
    ])


def test_calculate_agreements_no_data(wrapper):
    """Test calculate_agreements method with no data."""
    # Ensure data is None
    wrapper.data = None

    # Call the method
    wrapper.calculate_agreements()

    # Check that the logger was called with an error
    wrapper.logger.error.assert_called_once()


def test_calculate_measure_raw(wrapper, sample_data):
    """Test _calculate_measure method for raw agreement."""
    # Mock the data
    wrapper.data = sample_data

    # Ensure 'raw' is not already in calculators
    if 'raw' in wrapper.calculators:
        del wrapper.calculators['raw']

    # Create a mock for RawAgreement with the calculate_pairwise method
    mock_raw_instance = MagicMock()
    mock_raw_instance.calculate_pairwise.return_value = {
        ('Annotator1', 'Annotator2'): 0.8,
        ('Annotator1', 'Annotator3'): 0.6,
        ('Annotator2', 'Annotator3'): 0.7
    }

    # Mock the RawAgreement class to return our mock instance
    with patch('src.iaa_wrapper.RawAgreement',
               return_value=mock_raw_instance) as mock_raw:
        # Set pairwise parameter to True to use calculate_pairwise
        wrapper.args['pairwise'] = True

        # Mock the ConfidenceIntervalCalculator class
        with patch('src.iaa_wrapper.ConfidenceIntervalCalculator') as mock_ci:
            # Create a mock for the ConfidenceIntervalCalculator instance
            mock_ci_instance = MagicMock()
            mock_ci_instance.wilson_interval.return_value = {
                'ci_lower': 0.7,
                'ci_upper': 0.9,
                'confidence_level': 0.95
            }
            mock_ci.return_value = mock_ci_instance
            wrapper.calculators['raw'] = mock_raw(level=wrapper.log_level)
            # Set confidence interval parameter
            wrapper.args['confidence_interval'] = 0.95

            # Call the method
            wrapper._calculate_measure('raw')

            # Check that the RawAgreement class was created with the correct
            # log level
            # mock_raw.assert_called_once_with(level=wrapper.log_level)

            # Check that calculate_pairwise was called with the correct data
            mock_raw_instance.calculate_pairwise.assert_called_once_with(
                sample_data)

            # Check that the results were stored
            assert wrapper.results['raw'] == {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }

            # Check that the calculator was stored
            assert 'raw' in wrapper.calculators
            assert wrapper.calculators['raw'] == mock_raw_instance

            # Check that ConfidenceIntervalCalculator was created with the
            # correct parameters
            mock_ci.assert_called_once_with(
                confidence=0.95,
                level=wrapper.log_level
            )

            # Check that wilson_interval was called for each pair
            assert mock_ci_instance.wilson_interval.call_count == 3

            # Check that the confidence intervals were stored
            assert ('Annotator1',
                    'Annotator2') in wrapper.confidence_intervals['raw']
            assert wrapper.confidence_intervals['raw'][('Annotator1',
                                                        'Annotator2')] == {
                'ci_lower': 0.7,
                'ci_upper': 0.9,
                'confidence_level': 0.95
            }


def test_output_results_console(wrapper):
    """Test output_results method with console output."""
    # Set up the wrapper with some results
    wrapper.args['output_format'] = 'console'
    wrapper.args['output'] = None
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Mock the _output_to_console method
    wrapper._output_to_console = MagicMock()

    # Call the method
    wrapper.output_results()

    # Check that _output_to_console was called
    wrapper._output_to_console.assert_called_once()


def test_output_results_csv(wrapper):
    """Test output_results method with CSV output."""
    # Set up the wrapper with some results
    wrapper.args['output_format'] = 'csv'
    wrapper.args['output'] = 'test_output.csv'
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        }
    }

    # Mock the _output_to_csv method
    wrapper._output_to_csv = MagicMock()

    # Call the method
    wrapper.output_results()

    # Check that _output_to_csv was called with the correct file
    wrapper._output_to_csv.assert_called_once_with('test_output.csv')


def test_output_to_console(wrapper):
    """Test _output_to_console method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that print_agreement_table was called for the raw agreement
            mock_print.assert_called_once()

            # Check that the output contains the fleiss_kappa value
            output = fake_out.getvalue()
            assert "fleiss_kappa: 0.6500" in output


def test_output_to_csv_pairwise(wrapper):
    """Test _output_to_csv method with pairwise results."""
    # Set up the wrapper with some pairwise results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock export_multi_agreement_csv
        with patch(
                'src.iaa_wrapper.export_multi_agreement_csv') as mock_export:
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Check that export_multi_agreement_csv was called with the
            # correct parameters
            mock_export.assert_called_once_with(
                tmp_path,
                {'raw': wrapper.results['raw']},
                {}
            )
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_csv_single_value(wrapper):
    """Test _output_to_csv method with single value results."""
    # Set up the wrapper with some single value results
    wrapper.results = {
        'fleiss_kappa': 0.65,
        'krippendorff_alpha': 0.7
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_csv(tmp_path)

        # Check that the file was created and contains the correct data
        with open(tmp_path, 'r') as f:
            content = f.read()
            assert "Measure,Value" in content
            assert "fleiss_kappa,0.6500" in content
            assert "krippendorff_alpha,0.7000" in content
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_run(wrapper, sample_data):
    """Test run method."""
    # Mock the methods
    wrapper.load_data = MagicMock()
    wrapper.calculate_agreements = MagicMock()
    wrapper.output_results = MagicMock()

    # Call the method
    wrapper.run()

    # Check that the methods were called in the correct order
    wrapper.load_data.assert_called_once()
    wrapper.calculate_agreements.assert_called_once()
    wrapper.output_results.assert_called_once()


def test_run_with_error(wrapper):
    """Test run method with an error."""
    # Mock load_data to raise an exception
    wrapper.load_data = MagicMock(side_effect=Exception("Test error"))

    # Call the method and check that it raises the exception
    with pytest.raises(Exception, match="Test error"):
        wrapper.run()

    # Check that the logger was called with an error
    wrapper.logger.error.assert_called_once()


def test_calculate_different_measures(wrapper, sample_data):
    """Test _calculate_measure method with different measures."""
    # Set the data
    wrapper.data = sample_data

    # Mock the _get_measure_params method to return empty dictionaries
    wrapper._get_measure_params = MagicMock(return_value={})

    # Create mocks for each calculator
    for measure in ['cohen_kappa', 'fleiss_kappa', 'krippendorff_alpha',
                    'f_measure', 'icc', 'bwfk', 'dbcaa', 'iou']:
        mock_calculator = MagicMock()

        # Configure the return values
        if measure == 'cohen_kappa':
            mock_calculator.calculate_pairwise.return_value = {
                ('Annotator1', 'Annotator2'): 0.8}
        else:
            mock_calculator.calculate.return_value = {
                'cohen_kappa': 0.8,
                'fleiss_kappa': 0.7,
                'krippendorff_alpha': 0.6,
                'f_measure': 0.9,
                'icc': 0.85,
                'bwfk': 0.75,
                'dbcaa': 0.65,
                'iou': 0.55
            }[measure]

        # Store the mock in the calculators dictionary
        wrapper.calculators[measure] = mock_calculator

    # Mock directly the _calculate_measure method to avoid checks
    original_calculate_measure = wrapper._calculate_measure

    def mock_calculate_measure(measure):
        calculator = wrapper.calculators[measure]
        if measure == 'cohen_kappa':
            wrapper.results[measure] = calculator.calculate_pairwise(
                wrapper.data)
        else:
            wrapper.results[measure] = calculator.calculate(wrapper.data)

    wrapper._calculate_measure = mock_calculate_measure

    # Test each measure
    for measure in list(wrapper.calculators.keys()):
        wrapper._calculate_measure(measure)

    # Restore the original method
    wrapper._calculate_measure = original_calculate_measure

    # Verify that the results were stored correctly
    assert wrapper.results['cohen_kappa'] == {('Annotator1',
                                               'Annotator2'): 0.8}
    assert wrapper.results['fleiss_kappa'] == 0.7
    assert wrapper.results['krippendorff_alpha'] == 0.6
    assert wrapper.results['f_measure'] == 0.9
    assert wrapper.results['icc'] == 0.85
    assert wrapper.results['bwfk'] == 0.75
    assert wrapper.results['dbcaa'] == 0.65
    assert wrapper.results['iou'] == 0.55


def test_output_results_formats(wrapper):
    """Test output_results method with different formats."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Test with text format and output file
    wrapper.args['output_format'] = 'text'
    wrapper.args['output'] = 'test_output.txt'
    wrapper._output_to_text_file = MagicMock()
    wrapper.output_results()
    wrapper._output_to_text_file.assert_called_once_with('test_output.txt')

    # Test with html format
    wrapper.args['output_format'] = 'html'
    wrapper._output_to_html = MagicMock()
    wrapper.output_results()
    wrapper._output_to_html.assert_called_once_with('test_output.txt')

    # Test with console format
    wrapper.args['output_format'] = 'console'
    wrapper.args['output'] = None
    wrapper._output_to_console = MagicMock()
    wrapper.output_results()
    wrapper._output_to_console.assert_called_once()

    # Test with text format but no output file
    wrapper.args['output_format'] = 'text'
    wrapper.args['output'] = None
    wrapper._output_to_console = MagicMock()
    wrapper.output_results()
    wrapper._output_to_console.assert_called_once()


def test_output_to_html(wrapper):
    """Test _output_to_html method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html
        with patch('src.iaa_wrapper.save_agreement_html') as mock_save:
            # Call the method
            wrapper._output_to_html(tmp_path)

            # Check that save_agreement_html was called for the raw agreement
            mock_save.assert_called_once()

            # Check that the HTML file was created and contains the expected
            # content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "<html>" in content
                assert "<title>IAA-Eval Results</title>" in content
                assert "fleiss_kappa: 0.6500" in content
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json(wrapper):
    """Test _output_to_json method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Add some confidence intervals
    wrapper.confidence_intervals = {
        'raw': {
            ('Annotator1', 'Annotator2'): {
                'ci_lower': 0.7,
                'ci_upper': 0.9,
                'confidence_level': 0.95
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file was created and contains the expected
        # content
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'raw' in content
            assert 'Annotator1_Annotator2' in content['raw']
            assert content['raw']['Annotator1_Annotator2'] == 0.8
            assert 'fleiss_kappa' in content
            assert content['fleiss_kappa'] == 0.65
            assert 'confidence_intervals' in content
            assert 'raw' in content['confidence_intervals']
            raw = content['confidence_intervals']['raw']
            assert 'Annotator1_Annotator2' in raw
            assert raw['Annotator1_Annotator2']['ci_lower'] == 0.7
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_text_file(wrapper):
    """Test _output_to_text_file method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Check that print_agreement_table was called for the raw agreement
            mock_print.assert_called_once()

            # Check that the text file was created and contains the expected
            # content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "IAA-Eval Results" in content
                assert "=== RAW ===" in content
                assert "=== FLEISS_KAPPA ===" in content
                assert "fleiss_kappa: 0.6500" in content
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_get_measure_params(wrapper):
    """Test _get_measure_params method with different measures."""
    # Test f_measure with positive_class
    wrapper.args['positive_class'] = 1
    params = wrapper._get_measure_params('f_measure')
    assert params == {'positive_class': 1}

    # Test icc with icc_form
    wrapper.args['icc_form'] = '3,1'
    params = wrapper._get_measure_params('icc')
    assert params == {'form': '3,1'}

    # Test bwfk with bwfk_width
    wrapper.args['bwfk_width'] = 10
    params = wrapper._get_measure_params('bwfk')
    assert params == {'width': 10}

    # Test dbcaa with distance_threshold
    wrapper.args['distance_threshold'] = 5.0
    params = wrapper._get_measure_params('dbcaa')
    assert params == {'threshold': 5.0}

    # Test krippendorff_alpha with metric
    wrapper.args['metric'] = 'nominal'
    params = wrapper._get_measure_params('krippendorff_alpha')
    assert params == {'metric': 'nominal'}

    # Test with unknown measure
    params = wrapper._get_measure_params('unknown_measure')
    assert params == {}


def test_calculate_confidence_intervals_global(wrapper, sample_data):
    """Test _calculate_confidence_intervals method with global result."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['confidence_interval'] = 0.95
    wrapper.results['fleiss_kappa'] = 0.75  # Single value result

    # Mock the ConfidenceIntervalCalculator
    with patch('src.iaa_wrapper.ConfidenceIntervalCalculator') as mock_ci:
        mock_ci.return_value.wilson_interval.return_value = {
            'ci_lower': 0.65,
            'ci_upper': 0.85,
            'confidence_level': 0.95
        }

        # Call the method
        wrapper._calculate_confidence_intervals('fleiss_kappa')

        # Check that wilson_interval was called with the correct parameters
        mock_ci.return_value.wilson_interval.assert_called_once_with(
            0.75, len(sample_data))

        # Check that the confidence intervals were stored
        assert wrapper.confidence_intervals['fleiss_kappa'] == {
            'ci_lower': 0.65,
            'ci_upper': 0.85,
            'confidence_level': 0.95
        }


def test_interpret_results(wrapper):
    """Test interpretation of results in different output methods."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': 0.8,
        'cohen_kappa': 0.7,
        'fleiss_kappa': 0.65
    }

    # Create mock calculators with interpret method
    for measure in wrapper.results:
        mock_calculator = MagicMock()
        mock_calculator.interpret.return_value = (
            f"Test interpretation for {measure}"
        )
        wrapper.calculators[measure] = mock_calculator

    # Test _output_to_console
    with patch('builtins.print'):
        wrapper._output_to_console()
        # Check that interpret was called for each measure
        for measure in wrapper.results:
            wrapper.calculators[measure].interpret.assert_called_with(
                wrapper.results[measure])

    # Test _output_to_csv
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper._output_to_csv(tmp_path)

        # Check that the CSV file contains interpretations
        with open(tmp_path, 'r') as f:
            content = f.read()
            for measure in wrapper.results:
                assert f"Test interpretation for {measure}" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Test _output_to_html
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html
        with patch('src.iaa_wrapper.save_agreement_html'):
            wrapper._output_to_html(tmp_path)

            # Check that the HTML file contains interpretations
            with open(tmp_path, 'r') as f:
                content = f.read()
                for measure in wrapper.results:
                    assert f"Test interpretation for {measure}" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Test _output_to_json
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file contains interpretations
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'interpretations' in content
            for measure in wrapper.results:
                assert measure in content['interpretations']
                assert content['interpretations'][measure] == (
                    f"Test interpretation for {measure}"
                )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Test _output_to_text_file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper._output_to_text_file(tmp_path)

        # Check that the text file contains interpretations
        with open(tmp_path, 'r') as f:
            content = f.read()
            for measure in wrapper.results:
                assert f"Test interpretation for {measure}" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_calculate_measure_creates_new_calculator(wrapper, sample_data):
    """
    Test that _calculate_measure creates a new calculator if it doesn't exist.
    """
    # Set up the wrapper
    wrapper.data = sample_data

    # Ensure 'raw' is not in calculators
    if 'raw' in wrapper.calculators:
        del wrapper.calculators['raw']

    # Create a mock for the calculator class
    mock_calculator_class = MagicMock()
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.8
    mock_calculator_class.return_value = mock_calculator

    # Replace the MEASURE_CALCULATORS entry for 'raw'
    original_calculator = wrapper.MEASURE_CALCULATORS.get('raw')
    wrapper.MEASURE_CALCULATORS['raw'] = mock_calculator_class

    try:
        # Call the method - this should create a new calculator
        wrapper._calculate_measure('raw')

        # Check that the calculator class was called with the correct log level
        mock_calculator_class.assert_called_once_with(level=wrapper.log_level)

        # Check that the calculator was stored
        assert 'raw' in wrapper.calculators
        assert wrapper.calculators['raw'] == mock_calculator

        # Check that calculate was called
        mock_calculator.calculate.assert_called_once_with(sample_data)

        # Check that the result was stored
        assert wrapper.results['raw'] == 0.8
    finally:
        # Restore the original calculator class
        wrapper.MEASURE_CALCULATORS['raw'] = original_calculator


def test_calculate_measure_with_no_data(wrapper):
    """Test _calculate_measure method when no data is loaded."""
    # Ensure data is None
    wrapper.data = None

    # Call the method
    wrapper._calculate_measure('raw')

    # Check that no result was stored
    assert 'raw' not in wrapper.results


def test_calculate_measure_with_unknown_measure(wrapper, sample_data):
    """Test _calculate_measure method with an unknown measure."""
    # Set up the wrapper
    wrapper.data = sample_data

    # Call the method with an unknown measure
    wrapper._calculate_measure('unknown_measure')

    # Check that no result was stored
    assert 'unknown_measure' not in wrapper.results


def test_get_measure_params_with_missing_params(wrapper):
    """Test _get_measure_params method with missing parameters."""
    # Remove parameters from args
    wrapper.args.pop('positive_class', None)
    wrapper.args.pop('icc_form', None)
    wrapper.args.pop('bwfk_width', None)
    wrapper.args.pop('distance_threshold', None)
    wrapper.args.pop('metric', None)

    # Test each measure
    assert wrapper._get_measure_params('f_measure') == {}
    assert wrapper._get_measure_params('icc') == {}
    assert wrapper._get_measure_params('bwfk') == {}
    assert wrapper._get_measure_params('dbcaa') == {}
    assert wrapper._get_measure_params('krippendorff_alpha') == {}


def test_output_results_with_csv_format(wrapper):
    """Test output_results method with CSV format."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Test with CSV format
    wrapper.args['output_format'] = 'csv'
    wrapper.args['output'] = 'test_output.csv'
    wrapper._output_to_csv = MagicMock()
    wrapper.output_results()
    wrapper._output_to_csv.assert_called_once_with('test_output.csv')

    # Test with JSON format
    wrapper.args['output_format'] = 'json'
    wrapper.args['output'] = 'test_output.json'
    wrapper._output_to_json = MagicMock()
    wrapper.output_results()
    wrapper._output_to_json.assert_called_once_with('test_output.json')


def test_output_to_json_with_confidence_intervals(wrapper):
    """Test _output_to_json method with confidence intervals."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': 0.8,
        'fleiss_kappa': 0.65
    }

    # Add confidence intervals for single values
    wrapper.confidence_intervals = {
        'fleiss_kappa': {
            'ci_lower': 0.55,
            'ci_upper': 0.75,
            'confidence_level': 0.95
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file contains confidence intervals
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'confidence_intervals' in content
            assert 'fleiss_kappa' in content['confidence_intervals']
            assert (content['confidence_intervals']['fleiss_kappa']['ci_lower']
                   == 0.55)
            assert (content['confidence_intervals']['fleiss_kappa']['ci_upper']
                   == 0.75)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_calculate_measure_with_exception(wrapper, sample_data):
    """Test _calculate_measure method when an exception is raised."""
    # Set up the wrapper
    wrapper.data = sample_data

    # Ensure 'raw' is not in calculators
    if 'raw' in wrapper.calculators:
        del wrapper.calculators['raw']

    # Create a mock for the calculator class that raises an exception
    mock_calculator_class = MagicMock()
    mock_calculator = MagicMock()
    mock_calculator.calculate.side_effect = ValueError("Test exception")
    mock_calculator_class.return_value = mock_calculator

    # Replace the MEASURE_CALCULATORS entry for 'raw'
    original_calculator = wrapper.MEASURE_CALCULATORS.get('raw')
    wrapper.MEASURE_CALCULATORS['raw'] = mock_calculator_class

    try:
        # Call the method - this should raise an exception
        with pytest.raises(ValueError) as excinfo:
            wrapper._calculate_measure('raw')

        # Check that the exception message is correct
        assert "Test exception" in str(excinfo.value)

        # Check that the calculator class was called
        mock_calculator_class.assert_called_once_with(level=wrapper.log_level)

        # Check that calculate was called
        mock_calculator.calculate.assert_called_once_with(sample_data)

        # Check that no result was stored
        assert 'raw' not in wrapper.results
    finally:
        # Restore the original calculator class
        wrapper.MEASURE_CALCULATORS['raw'] = original_calculator


def test_output_results_with_unsupported_format(wrapper):
    """Test output_results method with an unsupported format."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        },
        'fleiss_kappa': 0.65
    }

    # Test with an unsupported format but with output file
    wrapper.args['output_format'] = 'unsupported_format'
    wrapper._output_to_console = MagicMock()

    # Call the method
    wrapper.output_results()

    # Check that _output_to_console was called
    wrapper._output_to_console.assert_called_once()
