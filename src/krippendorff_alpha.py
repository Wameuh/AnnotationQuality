import numpy as np
import pandas as pd
from typing import Dict, Union
from Utils.logger import get_logger, LogLevel


class KrippendorffAlpha:
    """
    A class to calculate Krippendorff's Alpha coefficient.

    Krippendorff's Alpha is a reliability coefficient that measures the
    agreement among multiple annotators, taking into account chance agreement
    and handling missing data appropriately.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the KrippendorffAlpha calculator.

        Args:
            level (LogLevel, optional):
                Logging level. Defaults to LogLevel.INFO.
        """
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    def calculate(self, df: pd.DataFrame, metric: str = 'nominal') -> float:
        """
        Calculate Krippendorff's Alpha for the given data.

        Args:
            df (pd.DataFrame):
                DataFrame with annotator scores as columns and items as rows.
            metric (str, optional): Distance metric to use. Options:
                - 'nominal': For categorical data (default)
                - 'ordinal': For ordinal data
                - 'interval': For interval data
                - 'ratio': For ratio data

        Returns:
            float: Krippendorff's Alpha coefficient (-1 to 1)
        """
        self.logger.info(
            f"Calculating Krippendorff's Alpha with {metric} metric")

        # Convert DataFrame to reliability data matrix
        reliability_data = self._prepare_data(df)

        # Calculate observed disagreement
        self.logger.debug("Calculating observed disagreement")
        observed_disagreement = self._observed_disagreement(
            reliability_data, metric)

        # Calculate expected disagreement
        self.logger.debug("Calculating expected disagreement")
        expected_disagreement = self._expected_disagreement(
            reliability_data, metric)

        # Calculate alpha
        if expected_disagreement == 0:
            self.logger.warning(
                "Expected disagreement is zero, perfect agreement by chance")
            return 1.0

        alpha = 1.0 - (observed_disagreement / expected_disagreement)
        self.logger.info(f"Krippendorff's Alpha: {alpha:.4f}")

        return alpha

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert DataFrame to reliability data matrix.

        Args:
            df (pd.DataFrame):
                DataFrame with annotator scores as columns and items as rows.

        Returns:
            np.ndarray:
                Reliability data matrix with shape (n_items, n_annotators)
        """
        # Convert to numpy array
        reliability_data = df.values

        # Replace NaN with None for proper handling in the algorithm
        reliability_data = np.where(np.isnan(reliability_data),
                                    None,
                                    reliability_data)

        self.logger.debug(
            f"Prepared reliability data with shape {reliability_data.shape}")
        return reliability_data

    def _observed_disagreement(self, data: np.ndarray, metric: str) -> float:
        """
        Calculate the observed disagreement.

        Args:
            data (np.ndarray): Reliability data matrix.
            metric (str): Distance metric to use.

        Returns:
            float: Observed disagreement value.
        """
        n_items, n_annotators = data.shape

        # Count valid values per item
        value_counts = np.zeros(n_items)
        for i in range(n_items):
            value_counts[i] = np.sum([1 for v in data[i] if v is not None])

        # Calculate coincidence matrix
        coincidence_matrix = self._coincidence_matrix(data)

        # Calculate observed disagreement using the distance metric
        disagreement = 0.0
        total_coincidences = np.sum(coincidence_matrix)

        if total_coincidences == 0:
            self.logger.warning(
                "No coincidences found, cannot calculate observed disagreement"
                )
            return 0.0

        for i in range(coincidence_matrix.shape[0]):
            for j in range(coincidence_matrix.shape[1]):
                if i != j:  # Only count disagreements
                    distance = self._calculate_distance(i, j, metric)
                    disagreement += coincidence_matrix[i, j] * distance

        return disagreement / total_coincidences

    def _expected_disagreement(self, data: np.ndarray, metric: str) -> float:
        """
        Calculate the expected disagreement by chance.

        Args:
            data (np.ndarray): Reliability data matrix.
            metric (str): Distance metric to use.

        Returns:
            float: Expected disagreement value.
        """
        # Calculate coincidence matrix
        coincidence_matrix = self._coincidence_matrix(data)

        # Calculate value frequencies
        value_frequencies = np.sum(coincidence_matrix, axis=1)
        total_frequency = np.sum(value_frequencies)

        if total_frequency == 0:
            self.logger.warning(
                "No valid data found, cannot calculate expected disagreement")
            return 0.0

        # Calculate expected disagreement
        disagreement = 0.0
        for i in range(len(value_frequencies)):
            for j in range(len(value_frequencies)):
                if i != j:  # Only count disagreements
                    distance = self._calculate_distance(i, j, metric)
                    disagreement += (
                        value_frequencies[i] * value_frequencies[j] * distance)

        return disagreement / (total_frequency * (total_frequency - 1))

    def _coincidence_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the coincidence matrix from the reliability data.

        Args:
            data (np.ndarray): Reliability data matrix.

        Returns:
            np.ndarray: Coincidence matrix.
        """
        # Get unique values (excluding None)
        all_values = [v for row in data for v in row if v is not None]
        unique_values = sorted(set(all_values))
        value_to_index = {value: i for i, value in enumerate(unique_values)}

        # Initialize coincidence matrix
        n_values = len(unique_values)
        coincidence_matrix = np.zeros((n_values, n_values))

        # Fill coincidence matrix
        for i in range(data.shape[0]):  # For each item
            # Get values for this item (excluding None)
            item_values = [v for v in data[i] if v is not None]
            n_values_item = len(item_values)

            if n_values_item <= 1:
                continue  # Skip items with 0 or 1 annotation

            # Update coincidence matrix
            for v1 in item_values:
                for v2 in item_values:
                    i1 = value_to_index[v1]
                    i2 = value_to_index[v2]
                    coincidence_matrix[i1, i2] += 1.0 / (n_values_item - 1)

        return coincidence_matrix

    def _calculate_distance(self,
                            value1_idx: int,
                            value2_idx: int,
                            metric: str) -> float:
        """
        Calculate the distance between two values based on the metric.

        Args:
            value1_idx (int): Index of first value.
            value2_idx (int): Index of second value.
            metric (str): Distance metric to use.

        Returns:
            float: Distance between the values.
        """
        if metric == 'nominal':
            # For nominal data, distance is 0 if same, 1 if different
            return 0.0 if value1_idx == value2_idx else 1.0

        elif metric == 'ordinal':
            # For ordinal data, distance is based on rank difference
            return abs(value1_idx - value2_idx) ** 2

        elif metric in ['interval', 'ratio']:
            # For interval/ratio data, distance is squared difference
            return (value1_idx - value2_idx) ** 2

        else:
            self.logger.warning(f"Unknown metric: {metric}, using nominal")
            return 0.0 if value1_idx == value2_idx else 1.0

    def interpret_alpha(self, alpha: float) -> str:
        """
        Interpret the Krippendorff's Alpha value.

        Args:
            alpha (float): Krippendorff's Alpha coefficient.

        Returns:
            str: Interpretation of the alpha value.
        """
        if alpha < 0:
            return "Poor agreement (worse than chance)"
        elif alpha < 0.2:
            return "Slight agreement"
        elif alpha < 0.4:
            return "Fair agreement"
        elif alpha < 0.6:
            return "Moderate agreement"
        elif alpha < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def get_alpha_statistics(self,
                             df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        Calculate Krippendorff's Alpha with different metrics and provide
        interpretations.

        Args:
            df (pd.DataFrame):
                DataFrame with annotator scores as columns and items as rows.

        Returns:
            Dict[str, Union[float, str]]:
                Dictionary with alpha values and interpretations.
        """
        metrics = ['nominal', 'ordinal', 'interval', 'ratio']
        results = {}

        for metric in metrics:
            alpha = self.calculate(df, metric=metric)
            results[f'alpha_{metric}'] = alpha
            results[f'interpretation_{metric}'] = self.interpret_alpha(alpha)

        return results
