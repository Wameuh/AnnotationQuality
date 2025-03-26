import numpy as np
import pandas as pd
from typing import Dict, Union, Literal, Tuple
from Utils.logger import get_logger
from src.agreement_measure import AgreementMeasure


class ICC(AgreementMeasure):
    """
    A class to calculate Intraclass Correlation Coefficient (ICC).

    ICC measures the reliability of ratings by comparing the variability of
    different ratings of the same subject to the total variation across all
    ratings and subjects.
    """

    @get_logger().log_scope
    def calculate(self,
                  df: pd.DataFrame,
                  form: Literal['1,1',
                                '2,1',
                                '3,1',
                                '1,k',
                                '2,k',
                                '3,k'] = '2,1') -> float:
        """
        Calculate ICC for the given data.

        Args:
            df (pd.DataFrame):
                DataFrame with annotator scores as columns and items as rows.
            form (str, optional): ICC form to calculate. Defaults to '2,1'.
                '1,1': One-way random effects, single rater/measurement
                '2,1': Two-way random effects, single rater/measurement
                '3,1': Two-way mixed effects, single rater/measurement
                '1,k': One-way random effects, average of k raters/measurements
                '2,k': Two-way random effects, average of k raters/measurements
                '3,k': Two-way mixed effects, average of k raters/measurements

        Returns:
            float: ICC value between 0 and 1
        """
        self.logger.info(f"Calculating ICC({form})")

        # Validate input
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return 0.0

        # Convert to numpy array and handle missing values
        data = df.values.copy()

        # Check if data contains missing values
        if np.isnan(data).any():
            self.logger.warning(
                "Data contains missing values. Using available data only.")
            # For ICC, we need complete cases, so we'll drop rows with any
            # missing values
            valid_rows = ~np.isnan(data).any(axis=1)
            if not np.any(valid_rows):
                self.logger.error(
                    "No complete cases available for ICC calculation")
                return 0.0
            data = data[valid_rows]

        # Get dimensions
        n_subjects = data.shape[0]  # Number of subjects/items
        n_raters = data.shape[1]    # Number of raters/annotators

        # Check if we have enough data
        if n_subjects < 2:
            self.logger.error(
                "At least 2 subjects are required for ICC calculation")
            return 0.0
        if n_raters < 2:
            self.logger.error(
                "At least 2 raters are required for ICC calculation")
            return 0.0

        # Calculate mean per subject (row means)
        subject_means = np.mean(data, axis=1)

        # Calculate mean per rater (column means)
        rater_means = np.mean(data, axis=0)

        # Calculate grand mean
        grand_mean = np.mean(data)

        # Calculate sum of squares
        SS_total = np.sum((data - grand_mean) ** 2)
        SS_subjects = n_raters * np.sum((subject_means - grand_mean) ** 2)
        SS_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
        SS_residual = SS_total - SS_subjects - SS_raters
        SS_error = SS_raters + SS_residual

        # Calculate degrees of freedom
        df_subjects = n_subjects - 1
        df_raters = n_raters - 1
        df_residual = df_subjects * df_raters
        df_error = df_raters + df_residual

        # Calculate mean squares
        MS_subjects = SS_subjects / df_subjects
        MS_raters = SS_raters / df_raters
        MS_residual = SS_residual / df_residual
        MS_error = SS_error / df_error

        # Calculate ICC based on form
        if form == '1,1':
            # One-way random effects, single rater
            icc = (MS_subjects - MS_error) / (
                MS_subjects + (n_raters - 1) * MS_error)
        elif form == '2,1':
            # Two-way random effects, single rater
            icc = (
                MS_subjects - MS_residual) / (
                    MS_subjects + (
                        n_raters - 1) * MS_residual + n_raters * (
                            MS_raters - MS_residual) / n_subjects)
        elif form == '3,1':
            # Two-way mixed effects, single rater
            icc = (MS_subjects - MS_residual) / (
                MS_subjects + (n_raters - 1) * MS_residual)
        elif form == '1,k':
            # One-way random effects, average of k raters
            icc = (MS_subjects - MS_error) / MS_subjects
        elif form == '2,k':
            # Two-way random effects, average of k raters
            icc = (MS_subjects - MS_residual) / (
                MS_subjects + (MS_raters - MS_residual) / n_subjects)
        elif form == '3,k':
            # Two-way mixed effects, average of k raters
            icc = (MS_subjects - MS_residual) / MS_subjects
        else:
            self.logger.error(f"Unknown ICC form: {form}")
            return 0.0

        # Ensure ICC is between 0 and 1
        icc = max(0.0, min(1.0, icc))

        self.logger.info(f"ICC({form}) = {icc:.4f}")
        return icc

    @get_logger().log_scope
    def interpret_icc(self, icc: float) -> str:
        """
        Interpret the ICC value.

        Args:
            icc (float): ICC value (typically between 0 and 1).

        Returns:
            str: Interpretation of the ICC value.
        """
        return self.interpret(icc)

    @get_logger().log_scope
    def get_icc_statistics(self,
                           df: pd.DataFrame,
                           form: Literal['1,1',
                                         '2,1',
                                         '3,1',
                                         '1,k',
                                         '2,k',
                                         '3,k'] = '2,1'
                           ) -> Dict[str, Union[float, str]]:
        """
        Calculate ICC and provide statistics.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores.
            form (str, optional): ICC form to calculate. Defaults to '2,1'.

        Returns:
            Dict[str, Union[float, str]]:
                    Dictionary with ICC values and interpretation.
        """
        results = {}

        # Calculate ICC for the specified form (default)
        icc = self.calculate(df, form)
        results['icc'] = icc
        results['interpretation'] = self.interpret_icc(icc)

        # Calculate ICC for all forms
        results['icc_1,1'] = self.calculate(df, '1,1')
        results['icc_2,1'] = self.calculate(df, '2,1')
        results['icc_3,1'] = self.calculate(df, '3,1')
        results['icc_1,k'] = self.calculate(df, '1,k')
        results['icc_2,k'] = self.calculate(df, '2,k')
        results['icc_3,k'] = self.calculate(df, '3,k')

        return results

    @get_logger().log_scope
    def calculate_pairwise(self,
                           df: pd.DataFrame,
                           form='2,1') -> Dict[Tuple[str, str], float]:
        """
        Calculate ICC for each pair of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
            form (str): ICC form to calculate, in the format 'k,l' where k
                is the model (1, 2, or 3) and l is the type (1 for single,
                k for average). Default is '2,1' (two-way random effects,
                single rater).

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and ICC values as values.
        """
        self.logger.info(f"Calculating pairwise ICC values (form {form})")

        pairwise_iccs = {}
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

                # For two raters, we need to adjust the form
                # Only form '1,1' (one-way random effects, single rater)
                # makes sense for two raters
                pair_form = '1,1'

                # Calculate ICC for this pair
                try:
                    # Calculate ICC directly on the DataFrame
                    icc_value = self.calculate(pair_df, pair_form)

                    # Store result with annotator names as key
                    pair_key = (columns[i], columns[j])
                    pairwise_iccs[pair_key] = icc_value

                    self.logger.debug(
                        f"ICC between {columns[i]} and {columns[j]}: "
                        f"{icc_value:.4f}")
                except Exception as e:
                    self.logger.warning(
                        f"Error calculating ICC for pair {columns[i]} and "
                        f"{columns[j]}: {str(e)}")
                    continue

        return pairwise_iccs
