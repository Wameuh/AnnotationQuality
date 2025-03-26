import numpy as np
import pandas as pd
from typing import Dict, Tuple
from Utils.logger import get_logger
from src.agreement_measure import AgreementMeasure


class FleissKappa(AgreementMeasure):
    """
    Class for calculating Fleiss' Kappa, a statistical measure of inter-rater
    reliability for categorical items.

    Fleiss' Kappa is used when multiple raters assign categorical ratings to
    items. It measures the degree of agreement among raters beyond what would
    be expected by chance.
    """

    @get_logger().log_scope
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
            self.logger.warning(
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

        self.logger.info(
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
            self.logger.warning(
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

        self.logger.info(f"Fleiss' Kappa: {kappa:.4f}")
        self.logger.info(f"Observed agreement (P_o): {P_o:.4f}")
        self.logger.info(f"Expected agreement (P_e): {P_e:.4f}")

        return kappa

    @get_logger().log_scope
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
            self.logger.warning(
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
            self.logger.warning(
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
            self.logger.info(
                f"Fleiss' Kappa for category {int(category)}: {kappa_j:.4f}"
            )

        return kappas

    @get_logger().log_scope
    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret the Fleiss' Kappa value according to Landis and Koch's scale.

        Args:
            kappa (float): Fleiss' Kappa value.

        Returns:
            str: Interpretation of the kappa value.
        """
        return self.interpret(kappa)

    @get_logger().log_scope
    def get_kappa_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various Fleiss' Kappa statistics.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[str, float]: Dictionary with various kappa statistics.
        """
        self.logger.info("Calculating Fleiss' Kappa statistics")

        # Calculate overall kappa
        kappa = self.calculate(df)

        # Calculate kappa by category
        kappas_by_category = self.calculate_by_category(df)

        # Compile statistics
        stats = {
            'kappa': kappa,
            'interpretation': self.interpret_kappa(kappa)
        }

        # Add category-specific kappas
        for category, k in kappas_by_category.items():
            stats[f'kappa_category_{category}'] = k
            stats[
                f'interpretation_category_{category}'
                ] = self.interpret_kappa(k)

        # Add min, max, and average category kappa if there are categories
        if kappas_by_category:
            stats['min_category_kappa'] = min(kappas_by_category.values())
            stats['max_category_kappa'] = max(kappas_by_category.values())
            stats['avg_category_kappa'] = np.mean(
                list(kappas_by_category.values()))

        return stats

    @get_logger().log_scope
    def calculate_pairwise(self,
                           df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate Fleiss' Kappa for each pair of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and kappa values as values.
        """
        self.logger.info("Calculating pairwise Fleiss' Kappa values")

        pairwise_kappas = {}
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

                # Calculate kappa for this pair
                # For two annotators, Fleiss' Kappa is equivalent to Cohen's
                # Kappa. So we can use the same calculation approach.

                # Get unique categories
                all_categories = sorted(set(pair_df[columns[i]].unique()) |
                                        set(pair_df[columns[j]].unique()))

                # Create contingency table
                contingency_table = pd.crosstab(
                    pair_df[columns[i]], pair_df[columns[j]])

                # Ensure all categories are in the table
                for cat in all_categories:
                    if cat not in contingency_table.index:
                        contingency_table.loc[cat] = 0
                    if cat not in contingency_table.columns:
                        contingency_table[cat] = 0

                # Sort the table by category
                contingency_table = (contingency_table
                                     .sort_index(axis=0)
                                     .sort_index(axis=1))

                # Calculate observed agreement
                n = contingency_table.sum().sum()
                observed_agreement = (contingency_table.values.diagonal()
                                      .sum() / n)

                # Calculate expected agreement
                row_sums = contingency_table.sum(axis=1)
                col_sums = contingency_table.sum(axis=0)

                # Use the categories as indices, not numeric indices
                expected_agreement = sum(
                    (row_sums[cat] * col_sums[cat]) / (n * n)
                    for cat in all_categories)

                # Calculate kappa
                if expected_agreement == 1:
                    # Perfect expected agreement, kappa is undefined
                    kappa = 1.0 if observed_agreement == 1 else 0.0
                else:
                    difference = observed_agreement - expected_agreement
                    kappa = (difference) / (1 - expected_agreement)

                # Store result with annotator names as key
                pair_key = (columns[i], columns[j])
                pairwise_kappas[pair_key] = kappa

                self.logger.debug(
                    f"Kappa between {columns[i]} and {columns[j]}: "
                    f"{kappa:.4f}")

        return pairwise_kappas
