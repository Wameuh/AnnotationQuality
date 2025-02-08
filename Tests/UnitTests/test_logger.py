import unittest
import sys
from io import StringIO
from Utils.logger import Logger, LogLevel

class TestLogger(unittest.TestCase):
    """Test suite for the Logger class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.stdout = StringIO()  # Create a string buffer to capture output
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout  # Redirect stdout to our buffer

    def tearDown(self):
        """Clean up after each test method."""
        sys.stdout = self.original_stdout  # Restore original stdout
        self.stdout.close()

    def test_info_message_at_info_level(self):
        """Test info message when level is INFO."""
        logger = Logger(level=LogLevel.INFO, output=self.stdout)
        test_message = "Test info message"
        logger.info(test_message)
        output = self.stdout.getvalue()
        
        self.assertIn(test_message, output)
        self.assertIn("[INFO]", output)

    def test_info_message_at_warning_level(self):
        """Test info message when level is WARNING."""
        logger = Logger(level=LogLevel.WARNING, output=self.stdout)
        test_message = "Test info message"
        logger.info(test_message)
        output = self.stdout.getvalue()
        
        self.assertEqual(output.strip(), "")  # Nothing should be printed

    def test_warning_message(self):
        """Test warning message (should print at WARNING level and below)."""
        logger = Logger(level=LogLevel.WARNING, output=self.stdout)
        test_message = "Test warning message"
        logger.warning(test_message)
        output = self.stdout.getvalue()
        
        self.assertIn(test_message, output)
        self.assertIn("[WARNING]", output)

    def test_error_message(self):
        """Test error message (should print at all levels)."""
        logger = Logger(level=LogLevel.ERROR, output=self.stdout)
        test_message = "Test error message"
        logger.error(test_message)
        output = self.stdout.getvalue()
        
        self.assertIn(test_message, output)
        self.assertIn("[ERROR]", output)

    def test_debug_message_at_debug_level(self):
        """Test debug message when level is DEBUG."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        test_message = "Test debug message"
        logger.debug(test_message)
        output = self.stdout.getvalue()
        
        self.assertIn(test_message, output)
        self.assertIn("[DEBUG]", output)

    def test_debug_message_at_info_level(self):
        """Test debug message when level is INFO."""
        logger = Logger(level=LogLevel.INFO, output=self.stdout)
        test_message = "Test debug message"
        logger.debug(test_message)
        output = self.stdout.getvalue()
        
        self.assertEqual(output.strip(), "")  # Nothing should be printed

    def test_multiple_messages(self):
        """Test multiple messages in sequence."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        messages = ["First message", "Second message", "Third message"]
        
        for msg in messages:
            logger.info(msg)
        
        output = self.stdout.getvalue()
        for msg in messages:
            self.assertIn(msg, output)
            self.assertIn("[INFO]", output)

    def test_log_level_names(self):
        """Test that log level names are correctly displayed."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        
        # Test each log level
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        output = self.stdout.getvalue()
        
        # Verify level names are correct
        self.assertIn("[DEBUG]", output)
        self.assertIn("[INFO]", output)
        self.assertIn("[WARNING]", output)
        self.assertIn("[ERROR]", output)
        
        # Verify no numeric values are present
        for level_value in ["0", "1", "2", "3"]:
            self.assertNotIn(f"[{level_value}]", output)

    def test_context_logger_debug_level(self):
        """Test context logger when level is DEBUG."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        
        def test_function():
            ctxLog = logger.logScope()  # Log entry
            # Function body
            # Log exit will happen when ctxLog is destroyed
        
        test_function()
        output = self.stdout.getvalue()
        
        # Check for function entry and exit markers
        self.assertIn("++ test_function() ++", output)
        self.assertIn("-- test_function() --", output)
        self.assertIn("[DEBUG]", output)
        self.assertIn("File:", output)
        self.assertIn("Line:", output)

    def test_context_logger_info_level(self):
        """Test context logger when level is INFO."""
        logger = Logger(level=LogLevel.INFO, output=self.stdout)
        
        def test_function():
            ctxLog = logger.logScope()  # Should not log at INFO level
            # Function body
        
        test_function()
        output = self.stdout.getvalue()
        
        # Nothing should be logged at INFO level
        self.assertEqual(output.strip(), "")

    def test_error_message_always_shows(self):
        """Test that error messages show at all levels."""
        for level in LogLevel:
            logger = Logger(level=level, output=self.stdout)
            test_message = f"Test error message at {level.name}"
            logger.error(test_message)
            output = self.stdout.getvalue()
            self.assertIn(test_message, output)
            self.assertIn("[ERROR]", output)
            self.stdout = StringIO()  # Reset output buffer

if __name__ == '__main__':
    unittest.main() 