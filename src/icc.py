import numpy as np
import pandas as pd
from typing import Dict, Union, Literal
from Utils.logger import get_logger, LogLevel


class ICC:
    """
    A class to calculate Intraclass Correlation Coefficient (ICC).

    ICC is a reliability measure used to assess the agreement between
    continuous or ordinal annotations from multiple annotators. It compares
    the variance between annotators to the total variance, estimating the
    proportion of variance attributable to agreement.

    This implementation supports different ICC forms:
    - ICC(1,1): One-way random effects, single rater/measurement
    - ICC(2,1): Two-way random effects, single rater/measurement
    - ICC(3,1): Two-way mixed effects, single rater/measurement
    - ICC(1,k): One-way random effects, average of k raters/measurements
    - ICC(2,k): Two-way random effects, average of k raters/measurements
    - ICC(3,k): Two-way mixed effects, average of k raters/measurements
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the ICC calculator.

        Args:
            level (LogLevel, optional):
                Logging level. Defaults to LogLevel.INFO.
        """
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

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

        # Convert DataFrame to numpy array, handling missing values
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

    def get_icc_statistics(self,
                           df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        Calculate ICC with different forms and provide statistics.

        Args:
            df (pd.DataFrame):
                DataFrame with annotator scores as columns and items as rows.

        Returns:
            Dict[str, Union[float, str]]:
                Dictionary with ICC values and interpretations.
        """
        results = {}

        # Calculate ICC with different forms
        forms = ['1,1', '2,1', '3,1', '1,k', '2,k', '3,k']
        for form in forms:
            icc = self.calculate(df, form=form)
            results[f'icc_{form}'] = icc
            results[f'interpretation_{form}'] = self.interpret_icc(icc)

        # Add recommended ICC (2,1) as the main result
        results['icc'] = results['icc_2,1']
        results['interpretation'] = results['interpretation_2,1']

        return results

    def interpret_icc(self, icc: float) -> str:
        """
        Interpret the ICC value.

        Args:
            icc (float): ICC value between 0 and 1.

        Returns:
            str: Interpretation of the ICC value.
        """
        if icc < 0.40:
            return "Poor reliability"
        elif icc < 0.60:
            return "Fair reliability"
        elif icc < 0.75:
            return "Good reliability"
        else:
            return "Excellent reliability"
