import sys
import threading
from datetime import datetime
from typing import TextIO
import os
from enum import Enum
from functools import wraps


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = 0    # Most verbose
    INFO = 1
    WARNING = 2
    ERROR = 3    # Least verbose


class ContextLogger:
    """
    A logger for function entry and exit points.
    This class automatically logs when a function is entered (at creation)
    and exited (at destruction), including file name, line number,
    and thread information.
    """
    def __init__(self, logger: 'Logger', func):
        """
        Initialize the context logger and log the function entry.

        Args:
            logger (Logger): The logger instance to use for logging.
            func: The function being logged.
        """
        self.logger = logger
        self.function_name = func.__name__
        self.function_file = os.path.basename(func.__code__.co_filename)
        self.function_line = func.__code__.co_firstlineno

    def __enter__(self):
        try:
            if self.logger.level.value <= LogLevel.DEBUG.value:
                entry_message = (
                    f"File: {self.function_file} | "
                    f"Line: {self.function_line} | "
                    f"++ {self.function_name}() ++"
                )
                self.logger.debug(entry_message)
            return self
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize logging context: {str(e)}"
            )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Log function exit when the context is exited.

        Args:
            exc_type: The type of the exception that occurred, if any
            exc_value: The instance of the exception that occurred, if any
            traceback: The traceback of the exception that occurred, if any
        """
        if hasattr(self, 'logger') and hasattr(self, 'function_name'):
            if self.logger.level.value <= LogLevel.DEBUG.value:
                exit_message = (
                    f"File: {self.function_file} | "
                    f"Line: {self.function_line} | "
                    f"-- {self.function_name}() --"
                )
                self.logger.debug(exit_message)


class Logger:
    """
    A logging utility class that provides different levels of logging
    with optional verbosity control and function call tracing.
    Supports context-based logging for function entry/exit points.
    """
    def __init__(self, level: LogLevel = LogLevel.DEBUG,
                 output: TextIO = sys.stdout):
        """
        Initialize the Logger.

        Args:
            level (LogLevel): The logging level. Defaults to LogLevel.DEBUG.
            output (TextIO): Output stream to write logs to. Defaults to stdout.

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
            self.output.write(message)
            self.output.flush()
        except AttributeError as e:
            sys.stderr.write("Logger error: Output is not writable\n")
            err_msg = (f"Logger error: Failed to write message: "
                      f"message {message}, error {str(e)}\n")
            sys.stderr.write(err_msg)
            sys.stderr.flush()
        except Exception as e:
            err_msg = (f"Logger error: Failed to write message: "
                      f"message {message}, error {str(e)}\n")
            sys.stderr.write(err_msg)
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
        return (f"[{timestamp}] [TID: {thread_id}] "
                f"[{level.name}] {message}\n")

    def _log(self, level: LogLevel, message: str) -> None:
        """
        Log a message if its level is greater than or equal to logger's level.

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

    def log_scope(self, func):
        """
        Decorator for logging function entry and exit.

        Args:
            func: The function to be decorated

        Returns:
            The wrapped function with logging

        Usage:
            @logger.log_scope
            def my_function():
                # Function code here
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ContextLogger(self, func):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error(f"Exception in {func.__name__}: {str(e)}")
                    raise
        return wrapper




