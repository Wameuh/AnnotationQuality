import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from Utils.logger import get_logger, LogLevel


class FMeasure:
    """
    A class to calculate F-measure for inter-annotator agreement.

    F-measure is the harmonic mean of precision and recall, commonly used
    in information retrieval and classification tasks. In the context of
    inter-annotator agreement, it can be used to measure how well annotators
    agree on positive classifications.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the FMeasure calculator.

        Args:
            level (LogLevel, optional):
                    Logging level. Defaults to LogLevel.INFO.
        """
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    def calculate(self, df: pd.DataFrame,
                  threshold: Optional[float] = None,
                  positive_class: Optional[Union[int, str]] = None) -> float:
        """
        Calculate the F-measure for the given data.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns and
                items as rows.
            threshold (float, optional): Threshold for binary classification.
                Values above threshold are considered positive. If None and
                positive_class is None, assumes binary data (0/1).
            positive_class (Union[int, str], optional): Value that represents
                the positive class. If provided, all other values are
                considered negative. Takes precedence over threshold.

        Returns:
            float: F-measure value (0 to 1)
        """
        self.logger.info("Calculating F-measure")

        # Convert DataFrame to binary matrix based on threshold or positive
        # class
        binary_data = self._prepare_binary_data(df, threshold, positive_class)

        # Calculate precision and recall for each pair of annotators
        precisions = []
        recalls = []

        n_annotators = binary_data.shape[1]
        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                # Get annotations from both annotators
                a1 = binary_data[:, i]
                a2 = binary_data[:, j]

                # Skip pairs with missing values
                valid_indices = ~np.isnan(a1) & ~np.isnan(a2)
                if not np.any(valid_indices):
                    continue

                a1 = a1[valid_indices]
                a2 = a2[valid_indices]

                # Calculate precision and recall
                precision, recall = self._calculate_precision_recall(a1, a2)

                precisions.append(precision)
                recalls.append(recall)

                self.logger.debug(f"Annotators {i+1} and {j+1}: "
                                  f"Precision={precision:.4f}, "
                                  f"Recall={recall:.4f}")

        # Calculate average precision and recall
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0

        # Calculate F-measure
        if avg_precision + avg_recall == 0:
            f_measure = 0.0
        else:
            f_measure = 2 * (avg_precision * avg_recall) / (
                avg_precision + avg_recall)

        self.logger.info(f"F-measure: {f_measure:.4f} "
                         f"(Precision={avg_precision:.4f}, "
                         f"Recall={avg_recall:.4f})")

        return f_measure

    def _prepare_binary_data(self, df: pd.DataFrame,
                             threshold: Optional[float] = None,
                             positive_class: Optional[Union[int, str]] = None
                             ) -> np.ndarray:
        """
        Convert DataFrame to binary matrix.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores.
            threshold (float, optional): Threshold for binary classification.
            positive_class (Union[int, str], optional): Value representing
            positive class.

        Returns:
            np.ndarray: Binary matrix where 1 represents positive class.
        """
        # Convert to numpy array
        data = df.values.copy()

        if positive_class is not None:
            # Use positive_class to binarize data
            binary_data = np.where(data == positive_class, 1, 0)
            self.logger.debug(
                f"Binarized data using positive class: {positive_class}")
        elif threshold is not None:
            # Use threshold to binarize data
            binary_data = np.where(data > threshold, 1, 0)
            self.logger.debug(f"Binarized data using threshold: {threshold}")
        else:
            # Assume data is already binary (0/1)
            # Check if data contains only 0s and 1s (and NaNs)
            unique_values = np.unique(data[~np.isnan(data)])
            if not np.all(np.isin(unique_values, [0, 1])):
                self.logger.warning(
                    "Data contains values other than 0 and 1. "
                    "Consider providing a threshold or positive_class.")
            binary_data = data
            self.logger.debug("Using data as-is, assuming binary values (0/1)")

        # Replace NaN with NaN (to preserve missing values)
        binary_data = np.where(np.isnan(data), np.nan, binary_data)

        return binary_data

    def _calculate_precision_recall(self,
                                    a1: np.ndarray,
                                    a2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate precision and recall between two annotators.

        Args:
            a1 (np.ndarray): Binary annotations from first annotator.
            a2 (np.ndarray): Binary annotations from second annotator.

        Returns:
            Tuple[float, float]: Precision and recall values.
        """
        # Calculate true positives, false positives, false negatives
        true_positives = np.sum((a1 == 1) & (a2 == 1))
        false_positives = np.sum((a1 == 1) & (a2 == 0))
        false_negatives = np.sum((a1 == 0) & (a2 == 1))

        # Calculate precision and recall
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        return precision, recall

    def get_f_measure_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate F-measure with different thresholds and provide statistics.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores.

        Returns:
            Dict[str, float]: Dictionary with F-measure values and statistics.
        """
        results = {}

        # Calculate F-measure with default settings (assuming binary data)
        results['f_measure'] = self.calculate(df)

        # If data appears to be non-binary, calculate with different thresholds
        unique_values = np.unique(df.values[~np.isnan(df.values)])
        if len(unique_values) > 2:
            # Try with median as threshold
            median_threshold = np.median(df.values[~np.isnan(df.values)])
            results['f_measure_median_threshold'] = self.calculate(
                df, threshold=median_threshold)

            # Try with mean as threshold
            mean_threshold = np.mean(df.values[~np.isnan(df.values)])
            results['f_measure_mean_threshold'] = self.calculate(
                df, threshold=mean_threshold)

            # Try with each unique value as positive class
            for value in unique_values:
                results[f'f_measure_class_{value}'] = self.calculate(
                    df, positive_class=value)

        return results

    def interpret_f_measure(self, f_measure: float) -> str:
        """
        Interpret the F-measure value.

        Args:
            f_measure (float): F-measure value.

        Returns:
            str: Interpretation of the F-measure value.
        """
        if f_measure < 0.2:
            return "Poor agreement"
        elif f_measure < 0.4:
            return "Fair agreement"
        elif f_measure < 0.6:
            return "Moderate agreement"
        elif f_measure < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
