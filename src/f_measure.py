import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional
from Utils.logger import get_logger
from src.agreement_measure import AgreementMeasure


class FMeasure(AgreementMeasure):
    """
    A class to calculate F-measure for evaluating agreement between annotators.

    F-measure is the harmonic mean of precision and recall, commonly used in
    information retrieval and binary classification tasks. In the context of
    annotator agreement, it measures how well annotators agree on positive
    classifications.
    """

    @get_logger().log_scope
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

        # Calculate pairwise F-measures
        pairwise_f_measures = self.calculate_pairwise(df,
                                                      threshold,
                                                      positive_class)

        # Calculate average F-measure
        if not pairwise_f_measures:
            self.logger.warning("No valid pairs for F-measure calculation")
            return 0.0

        avg_f_measure = np.mean(list(pairwise_f_measures.values()))
        self.logger.info(f"Average F-measure: {avg_f_measure:.4f}")

        return avg_f_measure

    @get_logger().log_scope
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
            # Create a mask for missing values (None or empty string)
            is_missing = np.vectorize(lambda x: x is None or x == '')
            missing_mask = is_missing(data)

            # Use positive_class to binarize data
            binary_data = np.where(data == positive_class, 1, 0)

            # Set missing values to NaN
            binary_data = np.where(missing_mask, np.nan, binary_data)

            self.logger.debug(
                f"Binarized data using positive class: {positive_class}")
        elif threshold is not None:
            # Use threshold to binarize data
            # Convert to numeric if possible
            try:
                numeric_data = data.astype(float)
                binary_data = np.where(numeric_data > threshold, 1, 0)
                self.logger.debug(
                    f"Binarized data using threshold: {threshold}")
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not convert data to numeric for threshold "
                    "comparison. Consider using positive_class instead.")
                binary_data = data
        else:
            # Assume data is already binary (0/1)
            # Try to convert to numeric
            try:
                numeric_data = data.astype(float)
                # Check if data contains only 0s and 1s (and NaNs)
                unique_values = np.unique(
                    numeric_data[~np.isnan(numeric_data)])
                if not np.all(np.isin(unique_values, [0, 1])):
                    self.logger.warning(
                        "Data contains values other than 0 and 1. "
                        "Consider providing a threshold or positive_class.")
                binary_data = numeric_data
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not convert data to numeric. "
                    "Assuming non-numeric data is already properly encoded.")
                binary_data = data
            self.logger.debug("Using data as-is, assuming binary values (0/1)")

        # Handle missing values
        # For numeric data, use np.isnan
        if np.issubdtype(binary_data.dtype, np.number):
            binary_data = np.where(np.isnan(binary_data), np.nan, binary_data)
        # For object data (strings, etc.), check for None or empty string
        else:
            is_missing = np.vectorize(lambda x: x is None or x == '')
            binary_data = np.where(is_missing(data), np.nan, binary_data)

        return binary_data

    @get_logger().log_scope
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

    @get_logger().log_scope
    def interpret_f_measure(self, f_measure: float) -> str:
        """
        Interpret the F-measure value in the context of inter-annotator
        agreement.

        Args:
            f_measure (float): F-measure value (typically between 0 and 1).

        Returns:
            str: Interpretation of the F-measure value.
        """
        # F-measure values are normally between 0 and 1
        # 0 represents complete disagreement or opposite agreement
        # Values close to 0 represent random or near-random agreement
        if f_measure < 0:
            return "Invalid F-measure value (should be non-negative)"
        elif f_measure < 0.2:
            return "Poor agreement (close to random or opposite agreement)"
        elif f_measure < 0.4:
            return "Fair agreement"
        elif f_measure < 0.6:
            return "Moderate agreement"
        elif f_measure < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @get_logger().log_scope
    def calculate_pairwise(self, df: pd.DataFrame,
                           threshold: Optional[float] = None,
                           positive_class: Optional[Union[int, str]] = None
                           ) -> Dict[Tuple[str, str], float]:
        """
        Calculate F-measure for each pair of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
            positive_class: The class to consider as positive for binary
                classification. If None, all non-zero values are considered
                positive.
            threshold: Threshold value for converting scores to binary. If
                None, no thresholding is applied.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and F-measure values as values.
        """
        self.logger.info("Calculating pairwise F-measures")

        pairwise_f_measures = {}
        columns = df.columns
        n_annotators = len(columns)

        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                # Select only the two annotators
                pair_df = df[[columns[i], columns[j]]].copy()

                # Drop rows with missing values
                pair_df = pair_df.dropna()

                if len(pair_df) == 0:
                    self.logger.warning(
                        f"No valid data for pair {columns[i]} and "
                        f"{columns[j]}")
                    continue

                # Convert to binary if threshold is provided
                if threshold is not None:
                    pair_df[columns[i]] = (
                        pair_df[columns[i]] > threshold).astype(int)
                    pair_df[columns[j]] = (
                        pair_df[columns[j]] > threshold).astype(int)

                # Convert to binary based on positive_class
                if positive_class is not None:
                    pair_df[columns[i]] = (
                        pair_df[columns[i]] == positive_class).astype(int)
                    pair_df[columns[j]] = (
                        pair_df[columns[j]] == positive_class).astype(int)

                # Calculate precision and recall
                # Treat first annotator as ground truth and second as
                # prediction
                true_positives = ((pair_df[columns[i]] == 1) &
                                  (pair_df[columns[j]] == 1)).sum()
                false_positives = ((pair_df[columns[i]] == 0) &
                                   (pair_df[columns[j]] == 1)).sum()
                false_negatives = ((pair_df[columns[i]] == 1) &
                                   (pair_df[columns[j]] == 0)).sum()

                # Calculate precision, recall, and F-measure
                positives = true_positives + false_positives
                negatives = true_positives + false_negatives
                if positives > 0:
                    precision = true_positives / positives
                else:
                    precision = 0.0
                if negatives > 0:
                    recall = true_positives / negatives
                else:
                    recall = 0.0

                if precision + recall > 0:
                    f_measure = 2 * (precision * recall) / (precision + recall)
                else:
                    f_measure = 0.0

                # Store result with annotator names as key
                pair_key = (columns[i], columns[j])
                pairwise_f_measures[pair_key] = f_measure

                self.logger.debug(
                    f"F-measure between {columns[i]} and {columns[j]}: "
                    f"{f_measure:.4f}")

        return pairwise_f_measures
