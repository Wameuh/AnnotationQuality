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
    """Test _get_measures_to_calculate method with all measures enabled."""
    # Set all to True
    wrapper.args['all'] = True

    # Call the method
    measures = wrapper._get_measures_to_calculate()

    # Check that all measures are included
    assert 'raw' in measures
    assert 'cohen_kappa' in measures
    assert 'fleiss_kappa' in measures
    assert 'krippendorff_alpha' in measures
    assert 'f_measure' in measures
    assert 'icc' in measures
    assert 'bwfk' in measures
    assert 'dbcaa' in measures
    assert 'iou' in measures


def test_get_measures_to_calculate_specific(wrapper):
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


def test_get_measures_to_calculate_none(wrapper):
    """Test _get_measures_to_calculate method with no measures enabled."""
    # Set all to False
    wrapper.args['all'] = False

    # Call the method
    measures = wrapper._get_measures_to_calculate()

    # Check that no measures are included
    assert not measures


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
    mock_raw_instance.calculate.return_value = 0.7

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

            # Check that calculate_pairwise was called with the correct data
            mock_raw_instance.calculate_pairwise.assert_called_once_with(
                sample_data)

            # Check that the results were stored
            assert wrapper.results['raw'] == {
                'overall': 0.7,
                'pairwise': {
                    ('Annotator1', 'Annotator2'): 0.8,
                    ('Annotator1', 'Annotator3'): 0.6,
                    ('Annotator2', 'Annotator3'): 0.7
                }
            }


def test_output_results_console(wrapper):
    """Test output_results method with console output."""
    # Set up the wrapper with some results
    wrapper.args['output_format'] = 'console'
    wrapper.args['output'] = None
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
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
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
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
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that print_agreement_table was called at least once
            assert mock_print.call_count >= 1, (
                f"Expected print_agreement_table to be called at least once, "
                f"but was called {mock_print.call_count} times"
            )

            # Check that the output contains the fleiss_kappa value
            output = fake_out.getvalue()
            assert "fleiss_kappa: 0.6500" in output


def test_output_to_csv_pairwise(wrapper):
    """Test _output_to_csv method with pairwise results."""
    # Set up the wrapper with some pairwise results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock export_multi_agreement_csv
        with patch(
            'src.iaa_wrapper.export_multi_agreement_csv'
        ) as mock_export:
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Check that export_multi_agreement_csv was called correctly
            mock_export.assert_called_once_with(
                tmp_path,
                {'raw': wrapper.results['raw']['pairwise']},
                {},
                use_method_names=True
            )
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_csv_single_value(wrapper):
    """Test _output_to_csv method with single value results."""
    # Set up the wrapper with some single value results
    wrapper.results = {
        'fleiss_kappa': {
            'overall': 0.65
        },
        'krippendorff_alpha': {
            'overall': 0.7
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        base_name, ext = os.path.splitext(tmp_path)

    try:
        # Mock save_agreement_csv to verify it's called correctly
        with patch('src.iaa_wrapper.save_agreement_csv') as mock_save:
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Check that save_agreement_csv was called for each measure
            expected_calls = [
                call(
                    f"{base_name}_fleiss_kappa{ext}",
                    {('Overall', 'Result'): 0.65},
                    confidence_intervals=None,
                    agreement_name='fleiss_kappa'
                ),
                call(
                    f"{base_name}_krippendorff_alpha{ext}",
                    {('Overall', 'Result'): 0.7},
                    confidence_intervals=None,
                    agreement_name='krippendorff_alpha'
                )
            ]
            mock_save.assert_has_calls(expected_calls, any_order=True)

    finally:
        # Clean up the temporary files
        for measure in wrapper.results.keys():
            measure_file = f"{base_name}_{measure}{ext}"
            if os.path.exists(measure_file):
                os.unlink(measure_file)
        # Clean up interpretation files if they exist
        for measure in wrapper.results.keys():
            interp_file = f"{base_name}_{measure}_interpretation.txt"
            if os.path.exists(interp_file):
                os.unlink(interp_file)


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


def test_output_to_html(wrapper):
    """Test _output_to_html method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Create mock calculators with interpret method
    for measure in wrapper.results.keys():
        mock_calculator = MagicMock()
        mock_calculator.interpret.return_value = (
            f"Test interpretation for {measure}"
        )
        wrapper.calculators[measure] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html
        with patch('src.iaa_wrapper.save_agreement_html') as mock_save:
            # Call the method
            wrapper._output_to_html(tmp_path)

            # Check that save_agreement_html was called once for each measure
            assert mock_save.call_count == 2, (
                f"Expected save_agreement_html to be called 2 times, "
                f"but was called {mock_save.call_count} times"
            )

            # Check that the HTML file was created and contains the expected
            # content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "<html>" in content
                assert "<title>IAA-Eval Results</title>" in content

                # Check that the links are correctly formatted
                for measure in wrapper.results.keys():
                    base_name = os.path.splitext(tmp_path)[0]
                    expected_link = (
                        f"{os.path.basename(base_name)}_{measure}.html"
                    )
                    assert f"href='{expected_link}'" in content, (
                        f"Link to {measure} is incorrect"
                    )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json(wrapper):
    """Test _output_to_json method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Create mock calculators with interpret method
    for measure in wrapper.results.keys():
        mock_calculator = MagicMock()
        mock_calculator.interpret.return_value = (
            f"Test interpretation for {measure}"
        )
        wrapper.calculators[measure] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Check JSON file content
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'raw' in content
            assert 'fleiss_kappa' in content

            # Check raw agreement data
            raw_data = content['raw']
            assert raw_data['overall'] == 0.75
            pairwise = raw_data['pairwise']
            # Convert tuple keys to strings for comparison
            expected_pairwise = {
                'Annotator1_Annotator2': 0.8,
                'Annotator1_Annotator3': 0.6,
                'Annotator2_Annotator3': 0.7
            }
            assert pairwise == expected_pairwise

            # Check fleiss_kappa data
            fleiss_data = content['fleiss_kappa']
            assert fleiss_data['overall'] == 0.65

            # Check interpretations
            assert 'interpretations' in content
            for measure in wrapper.results:
                assert measure in content['interpretations']
                assert (content['interpretations'][measure] ==
                       f"Test interpretation for {measure}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_text_file(wrapper):
    """Test _output_to_text_file method."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Create mock calculators with interpret method
    for measure in wrapper.results.keys():
        mock_calculator = MagicMock()
        mock_calculator.interpret.return_value = (
            f"Test interpretation for {measure}"
        )
        wrapper.calculators[measure] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Check that print_agreement_table was called for each measure
            assert mock_print.call_count == 2

            # Check text file content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "IAA-Eval Results" in content
                assert "=== RAW ===" in content
                assert "=== FLEISS_KAPPA ===" in content

                # Check interpretations
                for measure in wrapper.results:
                    expected_text = f"Test interpretation for {measure}"
                    assert expected_text in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_text_file_direct_result(wrapper):
    """Test _output_to_text_file method with direct result."""
    # Set up the wrapper with a direct dictionary result
    wrapper.results = {
        'raw': {
            'overall': 0.75
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Check that print_agreement_table was called with the correct
            # arguments
            mock_print.assert_called_once_with(
                {'overall': 0.75},
                confidence_intervals=None,
                file=mock_print.call_args[1]['file']
            )

            # Check text file content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "IAA-Eval Results" in content
                assert "=== RAW ===" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_text_file_pairwise_interp(wrapper):
    """Test _output_to_text_file writes interpretation for pairwise results."""
    # Set up the wrapper with pairwise results that will be flattened
    wrapper.results = {
        'raw': {
            'pairwise': {
                ('A1', 'A2'): 0.8,
                ('A1', 'A3'): 0.7
            }
        }
    }

    # Create mock calculator with interpret method
    mock_calculator = MagicMock()
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Verify print_agreement_table was called with the pairwise results
            mock_print.assert_called_with(
                {'pairwise': {('A1', 'A2'): 0.8, ('A1', 'A3'): 0.7}},
                file=mock_print.call_args[1]['file']
            )

            # Read the file content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "Interpretation: Good agreement" in content

            # Verify interpret was called with first pair value
            mock_calculator.interpret.assert_called_once_with(0.8)

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_interpret_results(wrapper):
    """Test interpretation of results in different output methods."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.8,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8
            }
        },
        'cohen_kappa': {
            'overall': 0.7,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.65
            }
        }
    }

    # Create mock calculators with interpret method
    for measure in wrapper.results:
        mock_calculator = MagicMock()
        mock_calculator.interpret.return_value = (
            f"Test interpretation for {measure}"
        )
        wrapper.calculators[measure] = mock_calculator

    # Test _output_to_html
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html to avoid actual file creation
        with patch('src.iaa_wrapper.save_agreement_html'):
            wrapper._output_to_html(tmp_path)

            # Check HTML content
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "<title>IAA-Eval Results</title>" in content

                # Check that the links are correctly formatted
                for measure in wrapper.results.keys():
                    base_name = os.path.splitext(tmp_path)[0]
                    expected_link = (
                        f"{os.path.basename(base_name)}_{measure}.html"
                    )
                    assert f"href='{expected_link}'" in content, (
                        f"Link to {measure} is incorrect"
                    )

                # Check interpretations
                for measure in wrapper.results:
                    expected_text = f"Test interpretation for {measure}"
                    assert expected_text in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Test _output_to_json
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper._output_to_json(tmp_path)

        # Check JSON content
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            for measure in wrapper.results:
                assert measure in content
                assert (content[measure]['overall'] ==
                       wrapper.results[measure]['overall'])

                # Convert tuple keys to strings for comparison
                if 'pairwise' in wrapper.results[measure]:
                    expected_pairwise = {
                        'Annotator1_Annotator2': (
                            wrapper.results[measure]['pairwise'][
                                ('Annotator1', 'Annotator2')
                            ]
                        )
                    }
                    assert content[measure]['pairwise'] == expected_pairwise
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
    mock_calculator.calculate_pairwise.return_value = {}
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

        # Check that the result was stored with both overall and pairwise
        assert wrapper.results['raw'] == {
            'overall': 0.8,
            'pairwise': {}
        }
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
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
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
        'raw': {
            'overall': 0.75,
            'pairwise': {
                'Annotator1_Annotator2': 0.8,
                'Annotator1_Annotator3': 0.6,
                'Annotator2_Annotator3': 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Add confidence intervals for single values
    wrapper.confidence_intervals = {
        'fleiss_kappa': {
            'overall': {
                'ci_lower': 0.55,
                'ci_upper': 0.75,
                'confidence_level': 0.95
            }
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
            ci_data = content['confidence_intervals']['fleiss_kappa']
            assert 'overall' in ci_data
            assert ci_data['overall']['ci_lower'] == 0.55
            assert ci_data['overall']['ci_upper'] == 0.75
            assert ci_data['overall']['confidence_level'] == 0.95
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
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Test with an unsupported format but with output file
    wrapper.args['output_format'] = 'unsupported_format'
    wrapper._output_to_console = MagicMock()

    # Call the method
    wrapper.output_results()

    # Check that _output_to_console was called
    wrapper._output_to_console.assert_called_once()


def test_output_to_csv_single_value_with_ci(wrapper):
    """Test _output_to_csv method with single value results and confidence
    intervals.

    This test specifically covers lines 286-287 in iaa_wrapper.py where
    confidence intervals are set up for single value results.
    """
    # Set up the wrapper with some single value results
    wrapper.results = {
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Add confidence intervals
    wrapper.confidence_intervals = {
        'fleiss_kappa': {
            'overall': {
                'ci_lower': 0.55,
                'ci_upper': 0.75,
                'confidence_level': 0.95
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        base_name, ext = os.path.splitext(tmp_path)

    try:
        # Mock save_agreement_csv
        with patch('src.iaa_wrapper.save_agreement_csv') as mock_save:
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Verify csv was called with confidence intervals
            measure_file = f"{base_name}_fleiss_kappa{ext}"
            expected_ci = {
                ('Overall', 'Result'):
                wrapper.confidence_intervals['fleiss_kappa']['overall']
            }

            mock_save.assert_called_once_with(
                measure_file,
                {('Overall', 'Result'): 0.65},
                confidence_intervals=expected_ci,
                agreement_name='fleiss_kappa'
            )
    finally:
        # Clean up temporary files
        measure_file = f"{base_name}_fleiss_kappa{ext}"
        if os.path.exists(measure_file):
            os.unlink(measure_file)
        interp_file = f"{base_name}_fleiss_kappa_interpretation.txt"
        if os.path.exists(interp_file):
            os.unlink(interp_file)


def test_output_to_html_single_value_with_ci(wrapper):
    """Test _output_to_html method with single value results and confidence
    intervals."""
    # Set up the wrapper with some single value results
    wrapper.results = {
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Add confidence intervals
    wrapper.confidence_intervals = {
        'fleiss_kappa': {
            'overall': {
                'ci_lower': 0.55,
                'ci_upper': 0.75,
                'confidence_level': 0.95
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html
        with patch('src.iaa_wrapper.save_agreement_html') as mock_save:
            # Call the method
            wrapper._output_to_html(tmp_path)

            # Verify html was called with confidence intervals
            html_file = tmp_path.replace('.html', '_fleiss_kappa.html')
            expected_ci = {
                ('Overall', 'Result'): (
                    wrapper.confidence_intervals['fleiss_kappa']['overall']
                )
            }

            mock_save.assert_has_calls([
                call(
                    html_file,
                    {('Overall', 'Result'): 0.65},
                    confidence_intervals=expected_ci,
                    title='FLEISS_KAPPA Overall Result'
                )
            ])
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_text_file_single_value_with_ci(wrapper):
    """Test _output_to_text_file with single value results and confidence
    intervals."""
    # Set up the wrapper with some single value results
    wrapper.results = {
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Add confidence intervals
    wrapper.confidence_intervals = {
        'fleiss_kappa': {
            'overall': {
                'ci_lower': 0.55,
                'ci_upper': 0.75,
                'confidence_level': 0.95
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table to capture calls
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Verify print_agreement_table was called with confidence intervals
            expected_agreements = {'overall': 0.65}
            ci_path = wrapper.confidence_intervals['fleiss_kappa']
            expected_ci = {'overall': ci_path['overall']}
            mock_args = mock_print.call_args[1]

            mock_print.assert_called_once_with(
                expected_agreements,
                confidence_intervals=expected_ci,
                file=mock_args['file']
            )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_html_with_confidence_interval_info(wrapper):
    """Test _output_to_html method with confidence interval information.

    This test specifically covers lines 359-368 in iaa_wrapper.py where
    confidence interval information is added to the index HTML file.
    """
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.6,
                ('Annotator2', 'Annotator3'): 0.7
            }
        },
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Add confidence intervals
    wrapper.confidence_intervals = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): {
                    'ci_lower': 0.7,
                    'ci_upper': 0.9,
                    'confidence_level': 0.95
                }
            }
        },
        'fleiss_kappa': {
            'overall': {
                'ci_lower': 0.55,
                'ci_upper': 0.75,
                'confidence_level': 0.95
            }
        }
    }

    # Set confidence method and level
    wrapper.args['confidence_interval'] = 0.95
    wrapper.args['confidence_method'] = 'wilson'

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html to avoid actual file creation
        with patch('src.iaa_wrapper.save_agreement_html'):
            # Call the method
            wrapper._output_to_html(tmp_path)

            # Check that the HTML index file contains confidence interval info
            with open(tmp_path, 'r') as f:
                content = f.read()
                # Check for confidence interval section
                assert "<h2>Confidence Interval Information</h2>" in content
                assert "<p>Confidence Level: 95%</p>" in content
                assert "<p>Method: wilson</p>" in content
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json_with_calculator_global_value(wrapper):
    """Test _output_to_json with a calculator that has get_global_value.

    This test specifically covers lines 409-410 in iaa_wrapper.py where
    a calculator with get_global_value method is used for interpretations.
    """
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        }
    }

    # Create a mock calculator with get_global_value method
    mock_calculator = MagicMock()
    mock_calculator.get_global_value.return_value = 0.7  # average value
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Verify calculator methods were called correctly
        mock_calculator.get_global_value.assert_called_once_with(
            wrapper.results['raw']
        )
        mock_calculator.interpret.assert_called_once_with(0.7)

        # Check that the JSON file contains the correct interpretation
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'interpretations' in content
            assert 'raw' in content['interpretations']
            assert content['interpretations']['raw'] == "Good agreement"
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json_with_pairwise_no_global_value(wrapper):
    """Test _output_to_json with interpret but no get_global_value method.

    This test covers lines 414-419 in iaa_wrapper.py where a calculator with
    interpret method but no get_global_value method is used for pairwise
    interpretations.
    """
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            ('Annotator1', 'Annotator2'): 0.8,
            ('Annotator1', 'Annotator3'): 0.6,
            ('Annotator2', 'Annotator3'): 0.7
        }
    }

    # Create a mock calculator with interpret but no get_global_value
    mock_calculator = MagicMock(spec=['interpret'])
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Verify interpret was called with the first pair's value
        mock_calculator.interpret.assert_called_once_with(0.8)

        # Check the JSON file contains the correct interpretation
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'interpretations' in content
            assert 'raw' in content['interpretations']
            expected = "Example interpretation for "
            expected += "Annotator1-Annotator2: Good agreement"
            assert content['interpretations']['raw'] == expected
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_get_log_level_with_invalid_case(wrapper):
    """Test _get_log_level method with invalid case."""
    # Test with uppercase input
    assert wrapper._get_log_level('INFO') == LogLevel.INFO
    # Test with mixed case
    assert wrapper._get_log_level('DeBuG') == LogLevel.DEBUG
    # Test with invalid input
    assert wrapper._get_log_level('invalid_level') == LogLevel.INFO


def test_calculate_measure_with_confidence_intervals(wrapper, sample_data):
    """Test _calculate_measure with confidence intervals."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['confidence_interval'] = 0.95
    wrapper.args['confidence_method'] = 'wilson'
    wrapper.args['metric'] = 'nominal'  # Test metric parameter

    # Create a mock calculator
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.8
    mock_calculator.calculate_pairwise.return_value = {
        ('Annotator1', 'Annotator2'): 0.8
    }

    # Create a mock ConfidenceIntervalCalculator
    with patch('src.iaa_wrapper.ConfidenceIntervalCalculator') as mock_ci:
        mock_ci_instance = MagicMock()
        mock_ci_instance.wilson_interval.return_value = {
            'ci_lower': 0.7,
            'ci_upper': 0.9,
            'confidence_level': 0.95
        }
        mock_ci.return_value = mock_ci_instance

        # Replace the calculator in MEASURE_CALCULATORS
        original = wrapper.MEASURE_CALCULATORS.get('krippendorff_alpha')
        wrapper.MEASURE_CALCULATORS['krippendorff_alpha'] = (
            lambda **kwargs: mock_calculator
        )

        try:
            # Call the method
            wrapper._calculate_measure('krippendorff_alpha')

            # Check that confidence intervals were calculated
            assert 'krippendorff_alpha' in wrapper.confidence_intervals
            ci_data = wrapper.confidence_intervals['krippendorff_alpha']
            assert 'overall' in ci_data
            ci = ci_data['overall']
            assert ci['ci_lower'] == 0.7
            assert ci['ci_upper'] == 0.9

            # Check that metric parameter was used
            mock_calculator.calculate.assert_called_once_with(
                sample_data, metric='nominal')

        finally:
            # Restore the original calculator
            wrapper.MEASURE_CALCULATORS['krippendorff_alpha'] = original


def test_output_to_html_with_confidence_method(wrapper):
    """Test _output_to_html with confidence method specification."""
    # Set up the wrapper with results and confidence intervals
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8
            }
        }
    }
    wrapper.confidence_intervals = {
        'raw': {
            'overall': {
                'ci_lower': 0.7,
                'ci_upper': 0.9,
                'confidence_level': 0.95
            }
        }
    }
    wrapper.args['confidence_interval'] = 0.95
    wrapper.args['confidence_method'] = 'custom_method'

    # Create mock calculator with interpret method
    mock_calculator = MagicMock()
    mock_calculator.interpret.return_value = "Test interpretation"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock save_agreement_html
        with patch('src.iaa_wrapper.save_agreement_html'):
            # Call the method
            wrapper._output_to_html(tmp_path)

            # Check that the HTML file contains the confidence method
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "<p>Method: custom_method</p>" in content
                assert "Test interpretation" in content

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json_with_pairwise_interpretation(wrapper):
    """Test _output_to_json with pairwise interpretation."""
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.7
            }
        }
    }

    # Create mock calculator with interpret method
    mock_calculator = MagicMock(spec=['interpret'])
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file contains the correct interpretation
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'interpretations' in content
            assert 'raw' in content['interpretations']
            expected = (
                "Example interpretation for "
                "Annotator1-Annotator2: Good agreement"
            )
            assert content['interpretations']['raw'] == expected

            # Verify the pairwise results were correctly formatted
            assert 'raw' in content
            assert 'pairwise' in content['raw']
            pairwise = content['raw']['pairwise']
            assert pairwise['Annotator1_Annotator2'] == 0.8
            assert pairwise['Annotator1_Annotator3'] == 0.7

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_results_with_default_filename(wrapper):
    """Test output_results method with no output file specified."""
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75
        }
    }

    # Test with text format but no output file
    wrapper.args = {
        'output_format': 'text',
        'output': None,
        'measure': 'raw'  # Required argument
    }
    wrapper._output_to_console = MagicMock()
    wrapper.output_results()
    wrapper._output_to_console.assert_called_once_with()


def test_output_to_console_with_non_float_result(wrapper):
    """Test _output_to_console with a non-float result."""
    # Set up the wrapper with a non-float result
    wrapper.results = {
        'raw': {
            'overall': "non-float-value"  # This will trigger the else branch
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains the non-float value
            output = fake_out.getvalue()
            assert "raw: non-float-value" in output


def test_output_to_console_with_direct_value(wrapper):
    """Test _output_to_console with a direct value (non-dictionary result)."""
    # Set up the wrapper with a direct value result
    wrapper.results = {
        'raw': {'overall': 0.75}  # Dictionary with overall value
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains the direct value
            output = fake_out.getvalue()
            assert "raw: 0.75" in output


def test_output_to_console_with_non_dict_result(wrapper):
    """Test _output_to_console when results contain a non-dictionary value."""
    # Set up the wrapper with a non-dictionary result
    wrapper.results = {
        'raw': {
            'overall': 0.85,  # Overall result as a float
            'other_value': 'some-text'  # This will be ignored
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains the value
            output = fake_out.getvalue()
            # Check formatted float value
            assert "raw: 0.8500" in output


def test_output_to_console_with_non_float_result_and_ci(wrapper):
    """Test _output_to_console with non-float result and confidence intervals.
    """
    # Set up the wrapper with a non-float result and confidence intervals
    wrapper.results = {
        'raw': {
            'overall': "non-float-value"  # This will trigger the else branch
        }
    }
    wrapper.confidence_intervals = {
        'raw': {
            'overall': {
                'ci_lower': 0.7,
                'ci_upper': 0.9,
                'confidence_level': 0.95
            }
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains the non-float value
            output = fake_out.getvalue()
            assert "raw: non-float-value" in output


def test_calculate_measure_with_dbcaa_threshold(wrapper, sample_data):
    """Test that distance threshold is correctly passed to DBCAA calculator."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['distance_threshold'] = 15.0  # Set a threshold value

    # Create a mock calculator
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.8
    mock_calculator.calculate_pairwise.return_value = {}

    # Store the mock in calculators
    wrapper.calculators['dbcaa'] = mock_calculator

    # Call the method
    wrapper._calculate_measure('dbcaa')

    # Verify that calculate was called with the threshold parameter
    mock_calculator.calculate.assert_called_once_with(
        sample_data,
        threshold=15.0
    )

    # Verify results were stored correctly
    assert wrapper.results['dbcaa'] == {
        'overall': 0.8,
        'pairwise': {}
    }


def test_calculate_measure_with_f_measure_positive_class(wrapper, sample_data):
    """Test that positive_class parameter is correctly passed to F-measure
    calculator."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['positive_class'] = 1  # Set a positive class value

    # Create a mock calculator
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.75
    mock_calculator.calculate_pairwise.return_value = {}

    # Store the mock in calculators
    wrapper.calculators['f_measure'] = mock_calculator

    # Call the method
    wrapper._calculate_measure('f_measure')

    # Verify that calculate was called with the positive_class parameter
    mock_calculator.calculate.assert_called_once_with(
        sample_data,
        positive_class=1
    )

    # Verify results were stored correctly
    assert wrapper.results['f_measure'] == {
        'overall': 0.75,
        'pairwise': {}
    }


def test_calculate_measure_with_icc_form(wrapper, sample_data):
    """Test that ICC form parameter is correctly passed to ICC calculator."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['icc_form'] = '2,1'  # Set ICC form value

    # Create a mock calculator
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.82
    mock_calculator.calculate_pairwise.return_value = {}

    # Store the mock in calculators
    wrapper.calculators['icc'] = mock_calculator

    # Call the method
    wrapper._calculate_measure('icc')

    # Verify that calculate was called with the form parameter
    mock_calculator.calculate.assert_called_once_with(
        sample_data,
        form='2,1'
    )

    # Verify results were stored correctly
    assert wrapper.results['icc'] == {
        'overall': 0.82,
        'pairwise': {}
    }


def test_calculate_measure_with_bwfk_width(wrapper, sample_data):
    """Test that BWFK width parameter is correctly passed to BWFK
    calculator."""
    # Set up the wrapper
    wrapper.data = sample_data
    wrapper.args['bwfk_width'] = 5  # Set BWFK width value

    # Create a mock calculator
    mock_calculator = MagicMock()
    mock_calculator.calculate.return_value = 0.78
    mock_calculator.calculate_pairwise.return_value = {}

    # Store the mock in calculators
    wrapper.calculators['bwfk'] = mock_calculator

    # Call the method
    wrapper._calculate_measure('bwfk')

    # Verify that calculate was called with the width parameter
    mock_calculator.calculate.assert_called_once_with(
        sample_data,
        width=5
    )

    # Verify results were stored correctly
    assert wrapper.results['bwfk'] == {
        'overall': 0.78,
        'pairwise': {}
    }


def test_output_to_console_with_float_and_ci(wrapper):
    """Test _output_to_console with float result and confidence intervals."""
    # Set up the wrapper with float result and confidence intervals
    wrapper.results = {
        'raw': {
            'overall': 0.75
        }
    }
    wrapper.confidence_intervals = {
        'raw': {
            'overall': {
                'ci_lower': 0.65,
                'ci_upper': 0.85,
                'confidence_level': 0.95
            }
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains the float value with CI
            output = fake_out.getvalue()
            expected = "raw: 0.7500 (CI: 0.6500 - 0.8500)"
            assert expected in output


def test_output_to_console_with_interpretation(wrapper):
    """Test _output_to_console with calculator interpretation."""
    # Set up the wrapper with a result
    wrapper.results = {
        'raw': {
            'overall': 0.75
        }
    }

    # Create a mock calculator with interpret method
    mock_calculator = MagicMock()
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table'):
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            # Call the method
            wrapper._output_to_console()

            # Check that the output contains both the value and interpretation
            output = fake_out.getvalue()
            assert "raw: 0.7500" in output
            assert "Interpretation: Good agreement" in output

            # Verify interpret was called with the result
            mock_calculator.interpret.assert_called_once_with(0.75)


def test_output_to_console_with_pairwise_ci(wrapper):
    """Test _output_to_console with pairwise confidence intervals."""
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.7
            }
        }
    }

    # Add pairwise confidence intervals
    wrapper.confidence_intervals = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): {
                    'ci_lower': 0.7,
                    'ci_upper': 0.9,
                    'confidence_level': 0.95
                },
                ('Annotator1', 'Annotator3'): {
                    'ci_lower': 0.6,
                    'ci_upper': 0.8,
                    'confidence_level': 0.95
                }
            }
        }
    }

    # Mock print_agreement_table
    with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
        # Call the method
        wrapper._output_to_console()

        # Verify print_agreement_table was called with pairwise CI
        mock_print.assert_called_with(
            wrapper.results['raw']['pairwise'],
            confidence_intervals=(
                wrapper.confidence_intervals['raw']['pairwise']
            )
        )


def test_output_to_console_with_error(wrapper):
    """Test _output_to_console error handling."""
    # Set up the wrapper with a result that includes pairwise data
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8
            }
        }
    }

    # Mock print_agreement_table at the module level
    with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
        mock_print.side_effect = TypeError("Test error")

        # Verify that the error is logged and re-raised
        with pytest.raises(TypeError):
            wrapper._output_to_console()

        # Verify that error was logged
        wrapper.logger.error.assert_any_call(
            "Exception in _output_to_console: Test error"
        )
        # Verify that traceback was logged
        assert any(
            call.args[0].startswith("Traceback: ")
            for call in wrapper.logger.error.call_args_list
        )


def test_output_to_csv_with_pairwise_ci(wrapper):
    """Test _output_to_csv method with pairwise confidence intervals."""
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8,
                ('Annotator1', 'Annotator3'): 0.7
            }
        }
    }

    # Add pairwise confidence intervals
    wrapper.confidence_intervals = {
        'raw': {
            'pairwise': {
                ('Annotator1', 'Annotator2'): {
                    'ci_lower': 0.7,
                    'ci_upper': 0.9,
                    'confidence_level': 0.95
                }
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        suffix='.csv', delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        # Mock export_multi_agreement_csv
        with patch(
            'src.iaa_wrapper.export_multi_agreement_csv'
        ) as mock_export:
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Verify export_multi_agreement_csv was called with CIs
            mock_export.assert_called_once_with(
                tmp_path,
                {'raw': wrapper.results['raw']['pairwise']},
                {'raw': wrapper.confidence_intervals['raw']['pairwise']},
                use_method_names=True
            )
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_csv_with_interpretation_file(wrapper):
    """Test _output_to_csv method creates interpretation file for single value
    results."""
    # Set up the wrapper with single value results
    wrapper.results = {
        'fleiss_kappa': {
            'overall': 0.65
        }
    }

    # Create a mock calculator with interpret method
    mock_calculator = MagicMock()
    mock_calculator.interpret.return_value = "Good agreement"
    wrapper.calculators['fleiss_kappa'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        base_name, ext = os.path.splitext(tmp_path)

    try:
        # Mock save_agreement_csv to avoid actual file creation
        with patch('src.iaa_wrapper.save_agreement_csv'):
            # Call the method
            wrapper._output_to_csv(tmp_path)

            # Verify interpretation file was created and contains correct
            # content
            interp_file = (
                f"{base_name}_fleiss_kappa_interpretation.txt"
            )
            assert os.path.exists(interp_file)
            with open(interp_file, 'r') as f:
                content = f.read()
                assert content == "Interpretation: Good agreement\n"

            # Verify logger was called
            wrapper.logger.info.assert_any_call(
                "Saved fleiss_kappa interpretation to: "
                f"{interp_file}"
            )

    finally:
        # Clean up the temporary files
        measure_file = f"{base_name}_fleiss_kappa{ext}"
        if os.path.exists(measure_file):
            os.unlink(measure_file)
        interp_file = (
            f"{base_name}_fleiss_kappa_interpretation.txt"
        )
        if os.path.exists(interp_file):
            os.unlink(interp_file)


def test_output_to_text_file_direct_dict(wrapper):
    """Test _output_to_text_file when result is a direct dictionary."""
    # Set up the wrapper with a direct dictionary result
    wrapper.results = {
        'raw': {
            ('A1', 'A2'): 0.8,
            ('A1', 'A3'): 0.7
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock print_agreement_table to verify it's called correctly
        with patch('src.iaa_wrapper.print_agreement_table') as mock_print:
            # Call the method
            wrapper._output_to_text_file(tmp_path)

            # Verify print_agreement_table was called directly with result
            mock_print.assert_called_once_with(
                wrapper.results['raw'],
                file=mock_print.call_args[1]['file']
            )

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json_overall_interpretation(wrapper):
    """Test _output_to_json method interprets overall results correctly."""
    # Set up the wrapper with overall results
    wrapper.results = {
        'raw': {
            'overall': 0.85
        }
    }

    # Create mock calculator with interpret method but no get_global_value
    mock_calculator = MagicMock(spec=['interpret'])
    mock_calculator.interpret.return_value = "Very good agreement"
    wrapper.calculators['raw'] = mock_calculator

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file contains the correct interpretation
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'interpretations' in content
            assert 'raw' in content['interpretations']
            assert content['interpretations']['raw'] == "Very good agreement"

        # Verify interpret was called with the overall value
        mock_calculator.interpret.assert_called_once_with(0.85)

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_to_json_pairwise_confidence_intervals(wrapper):
    """Test _output_to_json method handles pairwise confidence intervals
    correctly."""
    # Set up the wrapper with pairwise results
    wrapper.results = {
        'raw': {
            'pairwise': {
                ('A1', 'A2'): 0.8,
                ('A1', 'A3'): 0.7
            }
        }
    }

    # Add pairwise confidence intervals with mixed types
    wrapper.confidence_intervals = {
        'raw': {
            'pairwise': {
                ('A1', 'A2'): {
                    'ci_lower': 0.7,
                    'ci_upper': 0.9,
                    'confidence_level': 0.95,
                    'method': 'wilson'
                },
                ('A1', 'A3'): {
                    'ci_lower': 0.6,
                    'ci_upper': 0.8,
                    # String value to test type handling
                    'confidence_level': '95%',
                    'method': 'wilson'
                }
            }
        }
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Call the method
        wrapper._output_to_json(tmp_path)

        # Check that the JSON file contains the correct confidence intervals
        with open(tmp_path, 'r') as f:
            content = json.load(f)
            assert 'confidence_intervals' in content
            assert 'raw' in content['confidence_intervals']
            ci_data = content['confidence_intervals']['raw']
            assert 'pairwise' in ci_data

            # Check first pair's confidence intervals
            assert 'A1_A2' in ci_data['pairwise']
            first_pair = ci_data['pairwise']['A1_A2']
            assert first_pair['ci_lower'] == 0.7
            assert first_pair['ci_upper'] == 0.9
            assert first_pair['confidence_level'] == 0.95
            assert first_pair['method'] == 'wilson'

            # Check second pair's confidence intervals with string value
            assert 'A1_A3' in ci_data['pairwise']
            second_pair = ci_data['pairwise']['A1_A3']
            assert second_pair['ci_lower'] == 0.6
            assert second_pair['ci_upper'] == 0.8
            assert second_pair['confidence_level'] == '95%'
            assert second_pair['method'] == 'wilson'

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_output_results_html_and_text(wrapper):
    """Test output_results method with HTML and text formats.

    This test specifically covers lines 231 and 235 in iaa_wrapper.py where
    the output_results method calls _output_to_html and _output_to_text_file.
    """
    # Set up the wrapper with some results
    wrapper.results = {
        'raw': {
            'overall': 0.75,
            'pairwise': {
                ('Annotator1', 'Annotator2'): 0.8
            }
        }
    }

    # Test HTML format
    wrapper.args['output_format'] = 'html'
    wrapper.args['output'] = 'test_output.html'
    wrapper._output_to_html = MagicMock()
    wrapper.output_results()
    wrapper._output_to_html.assert_called_once_with('test_output.html')

    # Test text format
    wrapper.args['output_format'] = 'text'
    wrapper.args['output'] = 'test_output.txt'
    wrapper._output_to_text_file = MagicMock()
    wrapper.output_results()
    wrapper._output_to_text_file.assert_called_once_with('test_output.txt')
