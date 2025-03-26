from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union, Tuple
from Utils.logger import LogLevel, get_logger


class AgreementMeasure(ABC):
    """
    Base class for all agreement measures.

    This abstract class defines the common interface that all agreement
    measures should implement, ensuring consistency across different
    implementations.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the agreement measure.

        Args:
            level (LogLevel, optional):
                Log level for the logger. Defaults to LogLevel.INFO.
        """
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[float, Dict[Any, Any]]:
        """
        Calculate the agreement measure for the given data.

        This is the main method that all agreement measures must implement.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
                Expected format: Each column represents an annotator's scores.

        Returns:
            Union[float, Dict[Any, Any]]: The calculated agreement value.
                This can be a single float value or a dictionary of values
                (e.g., for pairwise measures).
        """
        pass

    def calculate_pairwise(self,
                           df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate agreement between all pairs of annotators.

        This method is optional and only relevant for measures that can be
        calculated between pairs of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and agreement values as values.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement calculate_pairwise")

    def calculate_from_dataframe(self,
                                 df: pd.DataFrame
                                 ) -> Union[float, Dict[Any, Any]]:
        """
        Alias for calculate() to ensure backward compatibility.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Union[float, Dict[Any, Any]]: The calculated agreement value.
        """
        return self.calculate(df)

    def interpret(self, value: float) -> str:
        """
        Interpret the agreement value according to standard scales.

        Args:
            value (float): The agreement value to interpret.

        Returns:
            str: Interpretation of the agreement value.
        """
        if value < 0:
            return "Poor agreement (less than chance)"
        elif value < 0.2:
            return "Slight agreement"
        elif value < 0.4:
            return "Fair agreement"
        elif value < 0.6:
            return "Moderate agreement"
        elif value < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
