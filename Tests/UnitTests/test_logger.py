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

def test_set_level(logger, string_io):
    logger.set_level(LogLevel.ERROR)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    output = string_io.getvalue()
    assert "Debug message" not in output
    assert "Info message" not in output
    assert "Warning message" not in output
    assert "Error message" in output

def test_invalid_log_level():
    with pytest.raises(ValueError):
        Logger(level="INVALID")

def test_context_logger(logger, string_io):
    @logger.log_scope
    def test_function():
        logger.info("Inside function")
    
    test_function()
    output = string_io.getvalue()
    
    assert "++ test_function() ++" in output
    assert "Inside function" in output
    assert "-- test_function() --" in output

def test_context_logger_with_exception(logger, string_io):
    @logger.log_scope
    def failing_function():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        failing_function()
    
    output = string_io.getvalue()
    assert "++ failing_function() ++" in output
    assert "Exception in failing_function: Test error" in output
    assert "-- failing_function() --" in output

def test_write_error_handling(capsys):
    class BrokenOutput:
        def write(self, _):
            raise IOError("Write error")
        
        def flush(self):
            pass
    
    logger = Logger(output=BrokenOutput())
    logger.error("Test message")
    
    captured = capsys.readouterr()
    assert "Logger error: Failed to write message" in captured.err 

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