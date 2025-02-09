import pytest
import io
from Utils.logger import Logger, LogLevel, ContextLogger


@pytest.fixture
def string_io():
    return io.StringIO()


@pytest.fixture
def logger(string_io):
    return Logger(level=LogLevel.DEBUG, output=string_io)


def test_log_levels(logger, string_io):
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    output = string_io.getvalue()
    assert "DEBUG" in output
    assert "Info message" in output
    assert "Warning message" in output
    assert "Error message" in output


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

        def warning(self, _):
            pass

    context = ContextLogger(BrokenLogger(), lambda: None)
    context.__enter__()  # Should catch exception and continue


def test_logger_attribute_error():
    """Test Logger when output has no write attribute"""
    class NoWriteOutput:
        def flush(self):
            pass

    logger = Logger(output=NoWriteOutput())
    logger.error("Test message")  # Should handle AttributeError


def test_set_level_invalid():
    """Test setting an invalid log level"""
    logger = Logger()
    with pytest.raises(ValueError, match="Level must be a valid LogLevel enum value"):
        logger.set_level("INVALID") 