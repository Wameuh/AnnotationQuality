import pytest
import io
import sys
from Utils.logger import Logger, LogLevel, ContextLogger, get_logger
from Utils.logger import init_logger


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the logger singleton before each test."""
    # Reset the singleton instance
    Logger._instance = None
    yield
    # Clean up after test
    Logger._instance = None


@pytest.fixture
def string_io():
    return io.StringIO()


@pytest.fixture
def logger(string_io):
    """Create a logger instance with StringIO output."""
    return Logger(level=LogLevel.DEBUG, output=string_io)


def test_log_levels(logger, string_io):
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    output = string_io.getvalue()
    assert "DEBUG" in output
    assert "Info message" in output
    assert "Warning message" in output
    assert "Error message" in output
    assert "Critical message" in output


def test_log_level_filtering(string_io):
    logger = Logger(level=LogLevel.WARNING, output=string_io)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    output = string_io.getvalue()
    assert "Debug message" not in output
    assert "Info message" not in output
    assert "Warning message" in output
    assert "Error message" in output


def test_log_format(logger, string_io):
    logger.info("Test message")

    output = string_io.getvalue()
    assert "[INFO]" in output
    assert "Test message" in output
    assert "TID:" in output


def test_log_scope_decorator(logger, string_io):
    @logger.log_scope
    def test_function():
        return "test result"

    result = test_function()

    output = string_io.getvalue()
    assert "++ test_function() ++" in output
    assert "-- test_function() --" in output
    assert result == "test result"


def test_log_scope_exception(logger, string_io):
    @logger.log_scope
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        failing_function()

    output = string_io.getvalue()
    assert "++ failing_function() ++" in output
    assert "Exception in failing_function: Test error" in output
    assert "-- failing_function() --" in output


def test_context_logger_enter_exception():
    """Test ContextLogger when __enter__ raises an exception"""
    class BrokenLogger:
        level = LogLevel.DEBUG

        def debug(self, _):
            raise Exception("Debug error")

        def warning(self, message):
            pass

    context = ContextLogger(BrokenLogger(), lambda: None)
    context.__enter__()  # Should catch exception and continue


def test_set_level():
    logger = Logger(level=LogLevel.DEBUG)
    logger.set_level(LogLevel.INFO)
    assert logger.level == LogLevel.INFO


def test_logger_attribute_error(capsys):
    """Test Logger when output has no write attribute"""
    class NoWriteOutput:
        def flush(self):
            pass

    # Redirect stderr to capture error messages
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        logger = Logger(output=NoWriteOutput())
        logger.error("Test message")  # Should handle AttributeError

        error_output = sys.stderr.getvalue()
        assert "Logger error: Output is not writable" in error_output
    finally:
        sys.stderr = old_stderr


def test_write_attribute_error(capsys):
    """Test handling of AttributeError in __write__"""
    class BrokenOutput:
        def flush(self):
            pass
        # No write() method -> will raise AttributeError

    # Redirect stderr to capture error messages
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        logger = Logger(output=BrokenOutput())
        logger.info("Test message")

        error_output = sys.stderr.getvalue()
        assert "Logger error: Output is not writable" in error_output
    finally:
        sys.stderr = old_stderr


def test_write_error(capsys):
    """Test handling of general exceptions in __write__"""
    class BrokenOutput:
        def write(self, _):
            raise Exception("Write error")

        def flush(self):
            pass

    # Redirect stderr to capture error messages
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        logger = Logger(output=BrokenOutput())
        logger.info("Test message")

        error_output = sys.stderr.getvalue()
        assert "Logger error: Failed to write message" in error_output
    finally:
        sys.stderr = old_stderr


def test_level_invalid_type():
    """Test setting level with invalid type raises ValueError"""
    with pytest.raises(ValueError,
                       match="Level must be a valid LogLevel enum value"):
        Logger(level="INVALID")


def test_set_level_with_invalid_type():
    """Test setting level with invalid type raises ValueError"""
    logger = Logger()

    invalid_levels = [
        None,
        42,
        "DEBUG",  # String instead of LogLevel enum
        object(),  # Random object
    ]

    for invalid_level in invalid_levels:
        with pytest.raises(ValueError,
                           match="Level must be a valid LogLevel enum value"):
            logger.set_level(invalid_level)


def test_context_logger_enter():
    """Test the __enter__ method of ContextLogger."""
    # Create a logger with DEBUG level
    logger = Logger(level=LogLevel.DEBUG)

    # Define a simple function to use with ContextLogger
    def test_function():
        pass

    # Create a ContextLogger instance
    context = ContextLogger(logger, test_function)

    # Call __enter__ and verify it returns self
    result = context.__enter__()
    assert result is context

    # Test with a non-DEBUG level to ensure the branch is covered
    logger.set_level(LogLevel.INFO)
    context = ContextLogger(logger, test_function)
    result = context.__enter__()
    assert result is context

    # Test exception handling
    class BrokenLogger:
        level = LogLevel.DEBUG

        def debug(self, message):
            raise Exception("Test exception")

        def warning(self, message):
            # Ne fait rien
            pass

    broken_context = ContextLogger(BrokenLogger(), test_function)
    # This should not raise an exception, but might return None
    result = broken_context.__enter__()
    # Ne vérifie pas que result is broken_context car cela peut être None
    assert result is None or result is broken_context


def test_get_logger_singleton():
    """Test that get_logger returns the same instance."""
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_get_logger_with_params():
    """Test get_logger with parameters."""
    # Reset singleton first
    Logger._instance = None

    # First call should create a new instance with the specified level
    logger1 = get_logger(level=LogLevel.ERROR)
    assert logger1.level == LogLevel.ERROR

    # Second call should return the same instance, ignoring the new level
    logger2 = get_logger(level=LogLevel.DEBUG)
    assert logger2 is logger1
    assert logger2.level == LogLevel.ERROR  # Level should not change


def test_context_logger_exit_missing_attributes():
    """Test ContextLogger.__exit__ when attributes are missing."""
    # Create a minimal ContextLogger without required attributes
    context = ContextLogger.__new__(ContextLogger)

    # This should not raise an exception
    context.__exit__(None, None, None)

    # Create a ContextLogger with logger but no function_name
    logger = Logger()
    context = ContextLogger.__new__(ContextLogger)
    context.logger = logger  # Set logger attribute
    # Intentionally NOT setting function_name attribute

    # This should not raise an exception and should return early
    context.__exit__(None, None, None)

    # Create a ContextLogger with all required attributes but DEBUG level
    context = ContextLogger.__new__(ContextLogger)
    logger = Logger(level=LogLevel.INFO)  # Not DEBUG level
    context.logger = logger
    context.function_name = "test_function"
    context.function_file = "test_file.py"
    context.function_line = 42

    # This should not log anything because level is not DEBUG
    context.__exit__(None, None, None)


def test_init_logger():
    """
    Test that init_logger properly resets and initializes the logger singleton.
    """
    # First, create a logger with INFO level
    initial_logger = get_logger(level=LogLevel.INFO)
    assert initial_logger.level == LogLevel.INFO

    # Now initialize a new logger with DEBUG level
    debug_logger = init_logger(level=LogLevel.DEBUG)

    # Check that the new logger has DEBUG level
    assert debug_logger.level == LogLevel.DEBUG

    # Check that get_logger now returns the new instance
    singleton = get_logger()
    assert singleton is debug_logger
    assert singleton.level == LogLevel.DEBUG

    # Test with a custom output stream
    output_stream = io.StringIO()
    custom_logger = init_logger(level=LogLevel.ERROR, output=output_stream)

    # Check that the new logger has the custom settings
    assert custom_logger.level == LogLevel.ERROR
    assert custom_logger.output is output_stream

    # Verify that the singleton was updated
    assert get_logger() is custom_logger

    # Test that logging works with the new output stream
    custom_logger.error("Test error message")
    output = output_stream.getvalue()
    assert "Test error message" in output
    assert "[ERROR]" in output
