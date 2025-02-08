import sys
import inspect
import threading
from datetime import datetime
from typing import TextIO
import os
import io
from enum import Enum
from functools import wraps
class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = 0  # Most verbose
    INFO = 1
    WARNING = 2
    ERROR = 3  # Least verbose

class ContextLogger:
    """
    A logger for function entry and exit points.
    This class automatically logs when a function is entered (at creation)
    and exited (at destruction), including file name, line number, and thread information.
    """

    def __init__(self, logger: 'Logger', func):
        """
        Initialize the context logger and log the function entry.

        Args:
            logger (Logger): The logger instance to use for logging.
        """
        self.logger = logger
        self.function_name = func.__name__
        self.function_file = os.path.basename(func.__code__.co_filename)
        self.function_line = func.__code__.co_firstlineno
        try:
            if self.logger.level.value <= LogLevel.DEBUG.value:
                entry_message = (f"File: {self.function_file} | "
                            f"Line: {self.function_line} | "
                            f"++ {self.function_name}() ++")
                self.logger.debug(entry_message)
        except Exception as e:
            self.logger.warning(f"Failed to initialize logging context: {str(e)}")


    def __del__(self):
        """Log function exit when the object is destroyed."""
        if hasattr(self, 'logger') and hasattr(self, 'function_name'):  # Check if initialization was successful
            if self.logger.level.value <= LogLevel.DEBUG.value:
                exit_message = f"File: {self.function_file} | Line: {self.function_line} | -- {self.function_name}() --"
                self.logger.debug(exit_message)


class Logger:
    """
    A logging utility class that provides different levels of logging
    with optional verbosity control and function call tracing.
    Supports context-based logging for function entry/exit points.
    """

    def __init__(self, level: LogLevel = LogLevel.DEBUG, output: TextIO = sys.stdout):
        """
        Initialize the Logger.

        Args:
            level (LogLevel): The logging level. Defaults to LogLevel.DEBUG.
            output (TextIO): Output stream to write logs to. Defaults to sys.stdout.
            
        Raises:
            ValueError: If level is not a valid LogLevel
        """
        if not isinstance(level, LogLevel):
            raise ValueError("Level must be a valid LogLevel enum value")
        self.level = level
        self.output = output

    def __write__(self, message: str) -> None:
        """Write a message to the output stream."""
        try:
            if isinstance(self.output, io.TextIOBase):
                self.output.write(message)
                self.output.flush()
            elif hasattr(self.output, 'write'):
                # Support any object with a write method
                self.output.write(message)
                if hasattr(self.output, 'flush'):
                    self.output.flush()
        except Exception as e:
            # Fallback to stderr in case of writing errors
            sys.stderr.write(f"Logger error: Failed to write message: {str(e)}\n")
            sys.stderr.flush()

    def _format_message(self, level: LogLevel, message: str) -> str:
        """
        Format a log message according to the configured format.
        
        Args:
            level (LogLevel): The log level (DEBUG, INFO, WARNING, ERROR)
            message (str): The message to log
            
        Returns:
            str: The formatted message with timestamp, thread ID, and level name
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thread_id = threading.get_ident()
        return f"[{timestamp}] [TID: {thread_id}] [{level.name}] {message}\n"

    def _log(self, level: LogLevel, message: str) -> None:
        """
        Log a message if its level is greater than or equal to the logger's level.

        Args:
            level (LogLevel): The level of the message.
            message (str): The message to log.
        """
        if level.value >= self.level.value:
            formatted_message = self._format_message(level, message)
            self.__write__(formatted_message)

    def info(self, message: str) -> None:
        """Log an informational message if level is INFO or lower."""
        self._log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        """Log a warning message if level is WARNING or lower."""
        self._log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        """Log an error message if level is ERROR or lower."""
        self._log(LogLevel.ERROR, message)

    def debug(self, message: str) -> None:
        """Log a debug message if level is DEBUG."""
        self._log(LogLevel.DEBUG, message)

    def set_level(self, level: LogLevel) -> None:
        """
        Change the logging level.
        
        Args:
            level (LogLevel): The new logging level
            
        Raises:
            ValueError: If level is not a valid LogLevel
        """
        if not isinstance(level, LogLevel):
            raise ValueError("Level must be a valid LogLevel enum value")
        self.level = level

    def logScope(self, func):
        """
        Decorator for logging function entry and exit.
        
        Args:
            func: The function to be decorated
            
        Returns:
            The wrapped function with logging
        
        Usage:
            @logger.logScope
            def my_function():
                # Function code here
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a context logger   
            logscope = ContextLogger(self, func)
            try:
                # Execute the function
                result = func(*args, **kwargs)

                return result
            except Exception as e:
                # Log any exception that occurs
                self.error(f"Exception in {func.__name__}: {str(e)}")
                raise
            finally:
                # Context logger will be destroyed here, logging the exit
                del logscope
        return wrapper




