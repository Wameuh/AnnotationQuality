import numpy as np
import pandas as pd
from typing import Dict, List
from Utils.logger import get_logger, LogLevel
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class DistanceBasedCellAgreement:
    """
    A class to calculate Distance-Based Cell Agreement Algorithm (DBCAA).

    DBCAA is an algorithm that measures agreement in cell detection without
    relying on ground truth. It calculates agreement based on the spatial
    distances between cells detected by different annotators.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the DistanceBasedCellAgreement calculator.

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
                  cell_positions: List[np.ndarray],
                  distance_threshold: float = 10.0) -> float:
        """
        Calculate the Distance-Based Cell Agreement score.

        Args:
            cell_positions (List[np.ndarray]): List of arrays containing cell
                positions from different annotators. Each array should have
                shape (n_cells, 2) where each row is (x, y) coordinates.
            distance_threshold (float, optional): Maximum distance (in pixels)
                for two cells to be considered matching. Defaults to 10.0.

        Returns:
            float: DBCAA score between 0 and 1
        """
        self.logger.info("Calculating Distance-Based Cell Agreement")

        # Validate inputs
        if not cell_positions:
            raise ValueError("No cell positions provided")

        if len(cell_positions) < 2:
            raise ValueError(
                "At least two sets of cell positions are required")

        for i, positions in enumerate(cell_positions):
            if not isinstance(positions, np.ndarray):
                raise ValueError(f"Cell positions {i} must be a numpy array")

            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError(
                    f"Cell positions {i} must have shape (n_cells, 2)")

        # Calculate pairwise agreement scores
        n_annotators = len(cell_positions)
        agreement_scores = []

        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                score = self._calculate_pairwise_agreement(
                    cell_positions[i],
                    cell_positions[j],
                    distance_threshold
                )
                agreement_scores.append(score)

        # Return average agreement score
        return np.mean(agreement_scores)

    def _calculate_pairwise_agreement(self,
                                      cells1: np.ndarray,
                                      cells2: np.ndarray,
                                      distance_threshold: float) -> float:
        """
        Calculate agreement score between two sets of cell positions.

        Args:
            cells1 (np.ndarray): First set of cell positions, shape (n1, 2)
            cells2 (np.ndarray): Second set of cell positions, shape (n2, 2)
            distance_threshold (float): Maximum distance for matching

        Returns:
            float: Agreement score between 0 and 1
        """
        # Handle empty arrays
        if cells1.shape[0] == 0 and cells2.shape[0] == 0:
            return 1.0  # Perfect agreement if both are empty

        if cells1.shape[0] == 0 or cells2.shape[0] == 0:
            return 0.0  # No agreement if one is empty

        # Calculate distance matrix between all pairs of cells
        distance_matrix = cdist(cells1, cells2, metric='euclidean')

        # Find optimal assignment (Hungarian algorithm)
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        # Count matches (cells within threshold distance)
        matches = sum(
            distance_matrix[row_indices, col_indices] <= distance_threshold)

        # Calculate F1-score (harmonic mean of precision and recall)
        precision = matches / cells1.shape[0]
        recall = matches / cells2.shape[0]

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def calculate_from_dataframe(self,
                                 df: pd.DataFrame,
                                 distance_threshold: float = 10.0) -> float:
        """
        Calculate DBCAA from a DataFrame containing cell positions.

        Args:
            df (pd.DataFrame): DataFrame where each column is an annotator and
                each row contains cell position as "x,y" string or tuple.
            distance_threshold (float, optional): Maximum distance for
                matching.
                Defaults to 10.0.

        Returns:
            float: DBCAA score between 0 and 1
        """
        self.logger.info("Calculating DBCAA from DataFrame")

        # Convert DataFrame to list of cell position arrays
        cell_positions = []
        invalid_count = 0

        for col in df.columns:
            # Extract non-null positions
            positions = df[col].dropna().values

            # Convert string positions to coordinates if needed
            coords = []
            col_invalid_count = 0

            for pos in positions:
                try:
                    if isinstance(pos, str) and ',' in pos:
                        # Assume format "x,y"
                        x, y = map(float, pos.split(','))
                        coords.append([x, y])
                    elif isinstance(pos, tuple) and len(pos) == 2:
                        coords.append([float(pos[0]), float(pos[1])])
                    else:
                        col_invalid_count += 1
                except (ValueError, TypeError):
                    col_invalid_count += 1

            if col_invalid_count > 0:
                self.logger.warning(
                    f"Skipped {col_invalid_count} invalid"
                    f" positions in column {col}")
                invalid_count += col_invalid_count

            if coords:  # Only add if there are valid coordinates
                cell_positions.append(np.array(coords))

        if invalid_count > 0:
            self.logger.warning(
                f"Skipped a total of {invalid_count} invalid "
                f"positions across all columns")

        # Calculate DBCAA
        if len(cell_positions) < 2:
            self.logger.warning(
                "Not enough valid annotators with cell positions")
            return 0.0

        return self.calculate(cell_positions, distance_threshold)

    def interpret_dbcaa(self, dbcaa: float) -> str:
        """
        Interpret the DBCAA value.

        Args:
            dbcaa (float): DBCAA value.

        Returns:
            str: Interpretation of the DBCAA value.
        """
        if dbcaa < 0.2:
            return "Poor agreement"
        elif dbcaa < 0.4:
            return "Fair agreement"
        elif dbcaa < 0.6:
            return "Moderate agreement"
        elif dbcaa < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def get_dbcaa_statistics(self,
                             cell_positions: List[np.ndarray],
                             distance_thresholds:
                             List[float] = [5.0, 10.0, 15.0, 20.0]
                             ) -> Dict[str, float]:
        """
        Calculate DBCAA with different distance thresholds and
        provide statistics.

        Args:
            cell_positions (List[np.ndarray]): List of arrays containing cell
                positions from different annotators.
            distance_thresholds (List[float], optional): List of distance
                thresholds to try.

        Returns:
            Dict[str, float]: Dictionary with DBCAA values and statistics.
        """
        results = {}

        # Calculate standard DBCAA (default threshold)
        standard_dbcaa = self.calculate(cell_positions)
        results['dbcaa_standard'] = standard_dbcaa
        results['interpretation_standard'] = \
            self.interpret_dbcaa(standard_dbcaa)

        # Calculate DBCAA with different thresholds
        for threshold in distance_thresholds:
            dbcaa = self.calculate(cell_positions,
                                   distance_threshold=threshold)
            results[f'dbcaa_threshold_{threshold}'] = dbcaa
            results[f'interpretation_threshold_{threshold}'] = \
                self.interpret_dbcaa(dbcaa)

        return results
