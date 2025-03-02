import pandas as pd
import numpy as np
from typing import Tuple, Dict
from Utils.logger import LogLevel, get_logger


class RawAgreement:
    """
    Calculate raw agreement between annotators.

    Raw agreement is the percentage of items where annotators agree on the
    score.
    For multiple annotators, agreement means all annotators gave the same
    score.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize RawAgreement calculator.

        Args:
            logger (Logger, optional): Logger instance for tracking operations.
                If None, creates a new logger.
        """
        # Use get_logger() to obtain the singleton instance
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    @get_logger().log_scope
    def calculate_pairwise(self,
                           df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate raw agreement between all pairs of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and agreement values as values.
        """
        # Get score columns and remove '_score' suffix for annotator names
        score_cols = [col for col in df.columns if col.endswith('_score')]
        annotators = [col.replace('_score', '') for col in score_cols]

        agreements = {}
        for i, ann1 in enumerate(annotators[:-1]):
            for ann2 in annotators[i + 1:]:
                # Get complete reviews for this pair
                pair_df = df[[f"{ann1}_score", f"{ann2}_score"]].dropna()

                if len(pair_df) == 0:
                    self._logger.warning(
                        f"No complete reviews found for {ann1} and {ann2}")
                    continue

                # Calculate agreement
                agreements[(ann1, ann2)] = (
                    pair_df[f"{ann1}_score"] == pair_df[f"{ann2}_score"]
                ).mean()

                self._logger.info(
                    f"Agreement between {ann1} and {ann2}: "
                    f"{agreements[(ann1, ann2)]:.1%}")

        return agreements

    @get_logger().log_scope
    def calculate_overall(self, df: pd.DataFrame) -> float:
        """
        Calculate overall raw agreement across all annotators.

        Agreement is counted only for reviews where all annotators provided
        scores.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
                Expected format: review_id as index, annotator scores in
                columns.

        Returns:
            float: Overall agreement score (0-1).
        """
        self._logger.info("Calculating overall raw agreement")

        # Get score columns
        score_cols = [col for col in df.columns if col.endswith('_score')]

        # Get reviews where all annotators provided scores
        complete_reviews = df[score_cols].dropna()

        if len(complete_reviews) == 0:
            msg = "No reviews found with scores from all annotators"
            self._logger.warning(msg)
            return 0.0

        # Calculate agreement (all annotators gave same score)
        agreements = complete_reviews.apply(
            lambda row: len(set(row)) == 1, axis=1
        )
        overall_agreement = agreements.mean()

        self._logger.info(
            f"Overall agreement across {len(score_cols)} annotators: "
            f"{overall_agreement:.2%}"
        )

        return overall_agreement

    @get_logger().log_scope
    def get_agreement_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive agreement statistics.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
                Expected format: review_id as index, annotator scores in
                columns.

        Returns:
            Dict[str, float]: Dictionary containing:
                - overall_agreement: Agreement across all annotators
                - average_pairwise: Mean of all pairwise agreements
                - min_pairwise: Lowest pairwise agreement
                - max_pairwise: Highest pairwise agreement
        """
        pairwise_agreements = self.calculate_pairwise(df)
        pairwise_values = list(pairwise_agreements.values())

        stats_data = {
            'overall_agreement': self.calculate_overall(df),
            'average_pairwise': np.mean(pairwise_values),
            'min_pairwise': min(pairwise_values),
            'max_pairwise': max(pairwise_values)
        }

        self._logger.info("Agreement Statistics:")
        for metric, value in stats_data.items():
            self._logger.info(f"{metric}: {value:.2%}")

        return stats_data
