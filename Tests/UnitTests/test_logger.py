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

    def test_function_decorator_debug_level(self):
        """Test the function decorator when level is DEBUG."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        
        @logger.logScope
        def test_function():
            logger.info("Inside function")
        
        test_function()
        output = self.stdout.getvalue()
        
        # Check for function entry and exit markers
        self.assertIn("++ test_function() ++", output)
        self.assertIn("-- test_function() --", output)
        self.assertIn("[DEBUG]", output)
        self.assertIn("File:", output)
        self.assertIn("Line:", output)
        self.assertIn("Inside function", output)

    def test_function_decorator_info_level(self):
        """Test the function decorator when level is INFO."""
        logger = Logger(level=LogLevel.INFO, output=self.stdout)
        
        @logger.logScope
        def test_function():
            logger.info("Inside function")
        
        test_function()
        output = self.stdout.getvalue()
        
        # Only the INFO message should be logged, not the entry/exit
        self.assertNotIn("++ test_function() ++", output)
        self.assertNotIn("-- test_function() --", output)
        self.assertIn("Inside function", output)

    def test_function_decorator_with_args(self):
        """Test the function decorator with arguments."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        
        @logger.logScope
        def test_function(x, y, name="test"):
            logger.info(f"Processing {x}, {y}, {name}")
            return x + y
        
        result = test_function(5, 3, name="example")
        output = self.stdout.getvalue()
        
        # Check function execution
        self.assertEqual(result, 8)
        # Check logs
        self.assertIn("++ test_function() ++", output)
        self.assertIn("Processing 5, 3, example", output)
        self.assertIn("-- test_function() --", output)

    def test_function_decorator_exception(self):
        """Test the function decorator with exception handling."""
        logger = Logger(level=LogLevel.DEBUG, output=self.stdout)
        
        @logger.logScope
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()
        
        output = self.stdout.getvalue()
        
        # Check all logging occurred
        self.assertIn("++ failing_function() ++", output)
        self.assertIn("-- failing_function() --", output)
        self.assertIn("[ERROR]", output)
        self.assertIn("Exception in failing_function: Test error", output)

    def test_set_level(self):
        """Test changing log level dynamically."""
        logger = Logger(level=LogLevel.ERROR, output=self.stdout)
        test_message = "Test message"
        
        logger.info(test_message)
        self.assertEqual(self.stdout.getvalue().strip(), "")  # Nothing should be printed
        
        logger.set_level(LogLevel.INFO)
        logger.info(test_message)
        self.assertIn(test_message, self.stdout.getvalue())

    def test_invalid_output_handler(self):
        """Test logger behavior with invalid output handler."""
        class InvalidOutput:
            pass
        
        logger = Logger(level=LogLevel.INFO, output=InvalidOutput())
        test_message = "Test message"
        
        # Should not raise exception but fall back to stderr
        logger.info(test_message)

if __name__ == '__main__':
    unittest.main() 