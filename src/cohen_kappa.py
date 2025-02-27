import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from Utils.logger import Logger, LogLevel


class CohenKappa:
    """
    Calculate Cohen's Kappa agreement between annotators.

    Cohen's Kappa measures inter-annotator agreement for categorical items,
    taking into account the agreement that would be expected by chance.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize CohenKappa calculator.

        Args:
            logger (Logger, optional): Logger instance for tracking operations.
                If None, creates a new logger.
        """
        self._logger = logger or Logger(level=LogLevel.INFO)

    def calculate_pairwise(self,
                           df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate Cohen's Kappa between all pairs of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and kappa values as values.
        """
        # Get score columns and remove '_score' suffix for annotator names
        score_cols = [col for col in df.columns if col.endswith('_score')]
        annotators = [col.replace('_score', '') for col in score_cols]

        kappas = {}
        for i, ann1 in enumerate(annotators[:-1]):
            for ann2 in annotators[i + 1:]:
                # Get complete reviews for this pair
                pair_df = df[[f"{ann1}_score", f"{ann2}_score"]].dropna()

                if len(pair_df) == 0:
                    self._logger.warning(
                        f"No complete reviews found for {ann1} and {ann2}")
                    continue

                # Calculate Cohen's Kappa
                kappa = self._calculate_kappa(
                    pair_df[f"{ann1}_score"].values,
                    pair_df[f"{ann2}_score"].values
                )

                kappas[(ann1, ann2)] = kappa

                self._logger.info(
                    f"Cohen's Kappa between {ann1} and {ann2}: "
                    f"{kappa:.2f}"
                )

        return kappas

    def _calculate_kappa(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Cohen's Kappa for two annotators.

        Args:
            a (np.ndarray): Scores from first annotator.
            b (np.ndarray): Scores from second annotator.

        Returns:
            float: Cohen's Kappa value.
        """
        if len(a) != len(b):
            raise ValueError("Arrays must have the same length")

        # Get unique categories from both annotators
        categories = sorted(set(np.concatenate((a, b))))
        n_categories = len(categories)

        # Create confusion matrix
        confusion_matrix = np.zeros((n_categories, n_categories))
        for i, j in zip(a, b):
            i_idx = categories.index(i)
            j_idx = categories.index(j)
            confusion_matrix[i_idx, j_idx] += 1

        # Calculate observed agreement
        observed_agreement = (
            np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        )

        # Calculate expected agreement
        row_sums = np.sum(confusion_matrix, axis=1)
        col_sums = np.sum(confusion_matrix, axis=0)
        expected_agreement = (
            np.sum(row_sums * col_sums) / (np.sum(confusion_matrix) ** 2)
        )

        # Calculate Cohen's Kappa
        if expected_agreement == 1:
            # If expected agreement is 1, kappa is undefined
            # In this case, return 1 if observed agreement is 1, else 0
            return 1.0 if observed_agreement == 1 else 0.0

        kappa = (
            (observed_agreement - expected_agreement)
            / (1 - expected_agreement)
        )

        return kappa

    def get_kappa_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive Cohen's Kappa statistics.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[str, float]: Dictionary containing:
                - average_kappa: Mean of all pairwise kappas
                - min_kappa: Lowest pairwise kappa
                - max_kappa: Highest pairwise kappa
        """
        pairwise_kappas = self.calculate_pairwise(df)
        kappa_values = list(pairwise_kappas.values())

        if not kappa_values:
            self._logger.warning("No valid kappa values calculated")
            return {
                'average_kappa': 0.0,
                'min_kappa': 0.0,
                'max_kappa': 0.0
            }

        stats_data = {
            'average_kappa': np.mean(kappa_values),
            'min_kappa': min(kappa_values),
            'max_kappa': max(kappa_values)
        }

        self._logger.info("Cohen's Kappa Statistics:")
        for metric, value in stats_data.items():
            self._logger.info(f"{metric}: {value:.2f}")

        return stats_data

    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Cohen's Kappa value according to common guidelines.

        Args:
            kappa (float): Cohen's Kappa value.

        Returns:
            str: Interpretation of the kappa value.
        """
        if kappa < 0:
            return "Poor agreement (less than chance)"
        elif kappa < 0.2:
            return "Slight agreement"
        elif kappa < 0.4:
            return "Fair agreement"
        elif kappa < 0.6:
            return "Moderate agreement"
        elif kappa < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
