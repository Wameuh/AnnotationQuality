import pytest
from unittest.mock import patch
from io import StringIO
from src.argparser import parse_arguments, print_available_options


@pytest.fixture
def mock_input_file(tmp_path):
    """Create a temporary input file for testing."""
    input_file = tmp_path / "test_input.csv"
    input_file.write_text("test data")
    return str(input_file)


def test_parse_arguments_with_input_file(mock_input_file):
    """Test parsing arguments with a valid input file."""
    # Mock sys.argv
    with patch('sys.argv', ['iaa-eval', mock_input_file]):
        args = parse_arguments()

        # Check that the input file is correctly parsed
        assert args['input_file'] == mock_input_file

        # Check default values
        assert args['output'] is None
        assert args['output_format'] == 'text'
        assert args['log_level'] == 'error'  # Default with no -v
        assert args['all'] is True  # Default when no measures are specified


def test_parse_arguments_with_specific_measures(mock_input_file):
    """Test parsing arguments with specific measures."""
    # Mock sys.argv
    with patch('sys.argv',
               ['iaa-eval',
                mock_input_file,
                '--raw',
                '--cohen-kappa']):
        args = parse_arguments()

        # Check that the specified measures are enabled
        assert args['raw'] is True
        assert args['cohen_kappa'] is True

        # Check that other measures are disabled
        assert args['fleiss_kappa'] is False
        assert args['krippendorff_alpha'] is False
        assert args['all'] is False


def test_parse_arguments_with_advanced_options(mock_input_file):
    """Test parsing arguments with advanced options."""
    # Mock sys.argv
    with patch('sys.argv', [
        'iaa-eval', mock_input_file,
        '--icc', '--icc-form', '3,1',
        '--confidence-interval', '0.9',
        '--bootstrap-samples', '500'
    ]):
        args = parse_arguments()

        # Check that the advanced options are correctly parsed
        assert args['icc'] is True
        assert args['icc_form'] == '3,1'
        assert args['confidence_interval'] == 0.9
        assert args['bootstrap_samples'] == 500


def test_parse_arguments_with_output_options(mock_input_file):
    """Test parsing arguments with output options."""
    # Mock sys.argv
    with patch('sys.argv', [
        'iaa-eval', mock_input_file,
        '--output', 'results.json',
        '--output-format', 'json'
    ]):
        args = parse_arguments()

        # Check that the output options are correctly parsed
        assert args['output'] == 'results.json'
        assert args['output_format'] == 'json'


def test_parse_arguments_with_invalid_input_file():
    """Test parsing arguments with an invalid input file."""
    # Mock sys.argv
    with patch('sys.argv', ['iaa-eval', 'nonexistent_file.csv']):
        # The parser should exit with an error
        with pytest.raises(SystemExit):
            parse_arguments()


def test_parse_arguments_with_invalid_confidence_interval(mock_input_file):
    """Test parsing arguments with an invalid confidence interval."""
    # Mock sys.argv
    with patch('sys.argv', [
        'iaa-eval', mock_input_file,
        '--confidence-interval', '1.5'
    ]):
        # The parser should exit with an error
        with pytest.raises(SystemExit):
            parse_arguments()


def test_parse_arguments_with_invalid_bootstrap_samples(mock_input_file):
    """Test parsing arguments with an invalid number of bootstrap samples."""
    # Mock sys.argv
    with patch('sys.argv', [
        'iaa-eval', mock_input_file,
        '--bootstrap-samples', '-10'
    ]):
        # The parser should exit with an error
        with pytest.raises(SystemExit):
            parse_arguments()


def test_show_options():
    """Test the --show-options argument."""
    # Mock sys.argv
    with patch('sys.argv', ['iaa-eval', '--show-options']):
        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            # The parser should exit after printing options
            with pytest.raises(SystemExit):
                parse_arguments()

            # Check that options were printed
            output = fake_out.getvalue()
            assert "IAA-EVAL: AVAILABLE OPTIONS" in output
            assert "BASIC USAGE:" in output
            assert "EXAMPLES:" in output


def test_help_option():
    """Test the --help argument."""
    # Mock sys.argv
    with patch('sys.argv', ['iaa-eval', '--help']):
        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            # The parser should exit after printing help
            with pytest.raises(SystemExit):
                parse_arguments()

            # Check that help was printed
            output = fake_out.getvalue()
            assert "usage:" in output
            assert "IAA-EVAL: AVAILABLE OPTIONS" in output


def test_print_available_options():
    """Test the print_available_options function."""
    # Capture stdout
    with patch('sys.stdout', new=StringIO()) as fake_out:
        print_available_options()

        # Check that options were printed
        output = fake_out.getvalue()
        assert "IAA-EVAL: AVAILABLE OPTIONS" in output
        assert "BASIC USAGE:" in output
        assert "EXAMPLES:" in output


def test_no_input_file():
    """Test parsing arguments without an input file."""
    # Mock sys.argv
    with patch('sys.argv', ['iaa-eval']):
        # The parser should exit with an error
        with pytest.raises(SystemExit) as excinfo:
            parse_arguments()

        # Check that the error message is about missing input file
        assert ("Input file is required" in str(excinfo.value)
                or excinfo.value.code != 0)


def test_no_measures_specified(mock_input_file):
    """Test parsing arguments without specifying any measures."""
    # Mock sys.argv with only the input file and no measure options
    with patch('sys.argv', ['iaa-eval', mock_input_file]):
        args = parse_arguments()

        # Check that --all is automatically enabled
        assert args['all'] is True

        # Check that other measures are disabled
        assert args['raw'] is False
        assert args['cohen_kappa'] is False
        assert args['fleiss_kappa'] is False
        assert args['krippendorff_alpha'] is False
        assert args['f_measure'] is False
        assert args['icc'] is False


def test_parse_arguments_with_verbosity(mock_input_file):
    """Test parsing arguments with different verbosity levels."""
    # Test with -v 0
    with patch('sys.argv', ['iaa-eval', mock_input_file, '-v', '0']):
        args = parse_arguments()
        assert args['log_level'] == 'error'

    # Test with -v 1
    with patch('sys.argv', ['iaa-eval', mock_input_file, '-v', '1']):
        args = parse_arguments()
        assert args['log_level'] == 'warning'

    # Test with -v 2
    with patch('sys.argv', ['iaa-eval', mock_input_file, '-v', '2']):
        args = parse_arguments()
        assert args['log_level'] == 'info'

    # Test with -v 3
    with patch('sys.argv', ['iaa-eval', mock_input_file, '-v', '3']):
        args = parse_arguments()
        assert args['log_level'] == 'debug'


def test_parse_arguments_with_confidence_method(mock_input_file):
    """Test parsing arguments with confidence method option."""
    # Mock sys.argv
    with patch('sys.argv', [
        'iaa-eval', mock_input_file,
        '--confidence-method', 'wilson'
    ]):
        args = parse_arguments()

        # Check that the confidence method is correctly parsed
        assert args['confidence_method'] == 'wilson'

        # Check default values
        assert args['confidence_interval'] == 0.95
        assert args['bootstrap_samples'] == 1000
