import pandas as pd
import numpy as np
from typing import Dict, Tuple
from Utils.logger import get_logger
from src.agreement_measure import AgreementMeasure


class CohenKappa(AgreementMeasure):
    """
    Calculate Cohen's Kappa agreement between annotators.

    Cohen's Kappa measures inter-annotator agreement for categorical items,
    taking into account the agreement that would be expected by chance.
    """

    @get_logger().log_scope
    def calculate(self, df: pd.DataFrame) -> float:
        """
        Calculate average Cohen's Kappa across all annotator pairs.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            float: Average Cohen's Kappa value.
        """
        # Calculate pairwise kappas
        pairwise_kappas = self.calculate_pairwise(df)

        # Return average kappa
        if not pairwise_kappas:
            self.logger.warning("No valid annotator pairs found")
            return 0.0

        avg_kappa = np.mean(list(pairwise_kappas.values()))
        self.logger.info(f"Average Cohen's Kappa: {avg_kappa:.4f}")
        return avg_kappa

    @get_logger().log_scope
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
                    self.logger.warning(
                        f"No complete reviews found for {ann1} and {ann2}")
                    continue

                # Calculate Cohen's Kappa
                kappa = self._calculate_kappa(
                    pair_df[f"{ann1}_score"].values,
                    pair_df[f"{ann2}_score"].values
                )

                kappas[(ann1, ann2)] = kappa
                self.logger.info(
                    f"Cohen's Kappa for {ann1} and {ann2}: {kappa:.4f}")

        return kappas

    @get_logger().log_scope
    def _calculate_kappa(self, ratings1: np.ndarray,
                         ratings2: np.ndarray) -> float:
        """
        Calculate Cohen's Kappa for two sets of ratings.

        Args:
            ratings1 (np.ndarray): First set of ratings.
            ratings2 (np.ndarray): Second set of ratings.

        Returns:
            float: Cohen's Kappa value.
        """
        if len(ratings1) != len(ratings2):
            raise ValueError("Rating arrays must have the same length")

        if len(ratings1) == 0:
            self.logger.warning("No ratings provided")
            return 0.0

        # Get unique categories
        categories = sorted(set(np.concatenate([ratings1, ratings2])))
        n_categories = len(categories)

        # Create confusion matrix
        confusion_matrix = np.zeros((n_categories, n_categories))
        for i in range(len(ratings1)):
            # Find indices for the categories
            idx1 = categories.index(ratings1[i])
            idx2 = categories.index(ratings2[i])
            confusion_matrix[idx1, idx2] += 1

        # Calculate observed agreement
        observed_agreement = np.sum(np.diag(confusion_matrix)) / np.sum(
            confusion_matrix)

        # Calculate expected agreement
        row_sums = np.sum(confusion_matrix, axis=1)
        col_sums = np.sum(confusion_matrix, axis=0)
        expected_agreement = np.sum(
            row_sums * col_sums) / (np.sum(confusion_matrix) ** 2)

        # Calculate Cohen's Kappa
        if expected_agreement == 1.0:
            # If expected agreement is 1, kappa is undefined
            # In this case, return 1 if observed agreement is 1, else 0
            return 1.0 if observed_agreement == 1 else 0.0

        kappa = (observed_agreement - expected_agreement) / (
            1.0 - expected_agreement)

        return kappa

    @get_logger().log_scope
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
            self.logger.warning("No valid kappa values calculated")
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

        self.logger.info("Cohen's Kappa Statistics:")
        for metric, value in stats_data.items():
            self.logger.info(f"{metric}: {value:.2f}")

        return stats_data

    @get_logger().log_scope
    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Cohen's Kappa value according to common guidelines.

        Args:
            kappa (float): Cohen's Kappa value.

        Returns:
            str: Interpretation of the kappa value.
        """
        return self.interpret(kappa)
