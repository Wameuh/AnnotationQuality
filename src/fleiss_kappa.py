import numpy as np
import pandas as pd
from typing import Dict


class FleissKappa:
    """
    Class for calculating Fleiss' Kappa, a statistical measure of inter-rater
    reliability for categorical items.

    Fleiss' Kappa is used when multiple raters assign categorical ratings to
    items. It measures the degree of agreement among raters beyond what would
    be expected by chance.
    """

    def __init__(self, logger):
        """
        Initialize the FleissKappa calculator.

        Args:
            logger: Logger instance for logging messages.
        """
        self._logger = logger

    def calculate(self, df: pd.DataFrame) -> float:
        """
        Calculate Fleiss' Kappa for all annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            float: Fleiss' Kappa value.
        """
        # Get score columns
        score_cols = [col for col in df.columns if col.endswith('_score')]

        if len(score_cols) < 2:
            self._logger.warning(
                "At least 2 annotators are required for Fleiss' Kappa"
            )
            return 0.0

        # Convert scores to a matrix where each row is an item and each column
        # is an annotator
        scores_matrix = df[score_cols].values

        # Get unique categories (scores)
        all_scores = np.unique(scores_matrix[~np.isnan(scores_matrix)])
        categories = sorted(all_scores)
        n_categories = len(categories)

        self._logger.info(
            f"Found {n_categories} unique categories: {categories}"
        )

        # Create a matrix where each row is an item and each column is a
        # category. Each cell contains the number of annotators who assigned
        # that category to that item
        n_items = len(df)

        # Initialize the category count matrix
        category_counts = np.zeros((n_items, n_categories))

        # Count how many annotators assigned each category to each item
        for i in range(n_items):
            item_scores = scores_matrix[i]
            valid_scores = item_scores[~np.isnan(item_scores)]
            n_valid = len(valid_scores)

            if n_valid == 0:
                continue

            for j, category in enumerate(categories):
                category_counts[i, j] = np.sum(valid_scores == category)

        # Calculate the number of valid ratings per item
        n_ratings_per_item = np.sum(category_counts, axis=1)

        # Remove items with no valid ratings
        valid_items = n_ratings_per_item > 0
        category_counts = category_counts[valid_items]
        n_ratings_per_item = n_ratings_per_item[valid_items]

        if len(category_counts) == 0:
            self._logger.warning(
                "No valid items found for Fleiss' Kappa calculation"
            )
            return 0.0

        # Calculate observed agreement for each item
        p_i = np.zeros(len(category_counts))
        for i in range(len(category_counts)):
            if n_ratings_per_item[i] <= 1:
                p_i[i] = 0  # Agreement is undefined for items with only one
                # rating
            else:
                p_i[i] = np.sum(
                    category_counts[i] * (category_counts[i] - 1)
                ) / (n_ratings_per_item[i] * (n_ratings_per_item[i] - 1))

        # Calculate mean observed agreement across all items
        P_o = np.mean(p_i)

        # Calculate expected agreement
        p_j = np.sum(category_counts, axis=0) / np.sum(n_ratings_per_item)
        P_e = np.sum(p_j ** 2)

        # Calculate Fleiss' Kappa
        # Handle special case: if P_e is 1, then all annotators used the same
        # category
        if P_e >= 0.9999:  # Use a threshold close to 1 for floating-point
            # precision
            if P_o >= 0.9999:  # Perfect agreement
                return 1.0
            else:
                return 0.0  # No agreement beyond chance

        kappa = (P_o - P_e) / (1 - P_e)

        self._logger.info(f"Fleiss' Kappa: {kappa:.4f}")
        self._logger.info(f"Observed agreement (P_o): {P_o:.4f}")
        self._logger.info(f"Expected agreement (P_e): {P_e:.4f}")

        return kappa

    def calculate_by_category(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate Fleiss' Kappa for each category separately.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[int, float]: Dictionary with categories as keys and kappa
                values as values.
        """
        # Get score columns
        score_cols = [col for col in df.columns if col.endswith('_score')]

        if len(score_cols) < 2:
            self._logger.warning(
                "At least 2 annotators are required for Fleiss' Kappa"
            )
            return {}

        # Convert scores to a matrix where each row is an item and each column
        # is an annotator
        scores_matrix = df[score_cols].values

        # Get unique categories (scores)
        all_scores = np.unique(scores_matrix[~np.isnan(scores_matrix)])
        categories = sorted(all_scores)
        n_categories = len(categories)

        # Create a matrix where each row is an item and each column is a
        # category. Each cell contains the number of annotators who assigned
        # that category to that item
        n_items = len(df)

        # Initialize the category count matrix
        category_counts = np.zeros((n_items, n_categories))

        # Count how many annotators assigned each category to each item
        for i in range(n_items):
            item_scores = scores_matrix[i]
            valid_scores = item_scores[~np.isnan(item_scores)]
            n_valid = len(valid_scores)

            if n_valid == 0:
                continue

            for j, category in enumerate(categories):
                category_counts[i, j] = np.sum(valid_scores == category)

        # Calculate the number of valid ratings per item
        n_ratings_per_item = np.sum(category_counts, axis=1)

        # Remove items with no valid ratings
        valid_items = n_ratings_per_item > 0
        category_counts = category_counts[valid_items]
        n_ratings_per_item = n_ratings_per_item[valid_items]

        if len(category_counts) == 0:
            self._logger.warning(
                "No valid items found for Fleiss' Kappa calculation"
            )
            return {}

        # Calculate kappa for each category
        kappas = {}
        for j, category in enumerate(categories):
            # Create a new DataFrame for this category
            category_df = df.copy()

            # Convert to binary problem: this category (1) vs. not this
            # category (0)
            for col in score_cols:
                # Convert non-NaN values to 1 if they match the category,
                # 0 otherwise
                mask = ~df[col].isna()
                category_df.loc[mask, col] = np.where(
                    df.loc[mask, col] == category, 1, 0
                )

            # Calculate Fleiss' Kappa for this binary problem
            kappa_j = self.calculate(category_df)

            kappas[int(category)] = kappa_j
            self._logger.info(
                f"Fleiss' Kappa for category {int(category)}: {kappa_j:.4f}"
            )

        return kappas

    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret the Fleiss' Kappa value according to Landis and Koch's scale.

        Args:
            kappa (float): Fleiss' Kappa value.

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
