"""Unit tests for iaa_eval.py."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the project root to the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)
sys.path.insert(0, project_root)

from iaa_eval import main  # noqa: E402
from Utils.logger import LogLevel  # noqa: E402


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
        'confidence_interval': 0.95,
        'confidence_method': 'wilson',
        'bootstrap_samples': 1000,
        'positive_class': None,
        'distance_threshold': 10.0,
        'bwfk_width': 5,
        'icc_form': '2,1'
    }


@pytest.fixture
def mock_wrapper(monkeypatch):
    """Fixture providing a mock IAAWrapper."""
    mock = MagicMock()
    monkeypatch.setattr('iaa_eval.IAAWrapper', mock)
    return mock


def test_main_success(mock_args, mock_wrapper):
    """Test that main function runs successfully with valid arguments."""
    mock_logger = MagicMock()
    with patch('iaa_eval.parse_arguments', return_value=mock_args):
        with patch('iaa_eval.get_logger', return_value=mock_logger):
            result = main()
            assert result == 0
            mock_wrapper.assert_called_once_with(mock_args)
            mock_wrapper.return_value.run.assert_called_once()
            mock_logger.info.assert_any_call("Starting IAA-Eval")
            mock_logger.info.assert_any_call("IAA-Eval completed successfully")


def test_main_error_in_wrapper(mock_args, mock_wrapper):
    """Test that main function handles errors from IAAWrapper properly."""
    error_msg = "Test error"
    mock_wrapper.return_value.run.side_effect = Exception(error_msg)
    mock_logger = MagicMock()

    with patch('iaa_eval.parse_arguments', return_value=mock_args):
        with patch('iaa_eval.get_logger', return_value=mock_logger):
            result = main()
            assert result == 1
            mock_logger.error.assert_called_once_with(
                f"Error in IAA-Eval: {error_msg}"
            )


def test_main_keyboard_interrupt(mock_args, mock_wrapper):
    """Test that main function handles keyboard interrupts properly."""
    mock_wrapper.return_value.run.side_effect = KeyboardInterrupt()

    with patch('iaa_eval.parse_arguments', return_value=mock_args):
        with patch('sys.stderr', new=StringIO()) as fake_stderr:
            result = main()
            assert result == 130
            assert "Operation cancelled by user" in fake_stderr.getvalue()


def test_main_fatal_error(mock_args):
    """Test that main function handles fatal errors in argument parsing."""
    error_msg = "Fatal parsing error"
    with patch(
        'iaa_eval.parse_arguments',
        side_effect=Exception(error_msg)
    ):
        with patch('sys.stderr', new=StringIO()) as fake_stderr:
            result = main()
            assert result == 1
            assert f"Fatal error: {error_msg}" in fake_stderr.getvalue()


@pytest.mark.parametrize("log_level,expected_enum", [
    ('debug', 'DEBUG'),
    ('info', 'INFO'),
    ('warning', 'WARNING'),
    ('error', 'ERROR'),
    ('critical', 'CRITICAL'),
    ('invalid', 'INFO'),  # Test default fallback
])
def test_main_log_levels(mock_args, mock_wrapper, log_level, expected_enum):
    """Test that main function sets up logging with different log levels."""
    mock_args['log_level'] = log_level
    mock_logger = MagicMock()

    with patch('iaa_eval.parse_arguments', return_value=mock_args):
        with patch(
            'iaa_eval.get_logger',
            return_value=mock_logger
        ) as mock_get_logger:
            main()

            # Verify the logger was created with correct log level
            expected_level = getattr(LogLevel, expected_enum)
            mock_get_logger.assert_called_once_with(expected_level)

            # Verify logger was used
            mock_logger.info.assert_any_call("Starting IAA-Eval")
            if not mock_wrapper.return_value.run.side_effect:
                mock_logger.info.assert_any_call(
                    "IAA-Eval completed successfully"
                )
