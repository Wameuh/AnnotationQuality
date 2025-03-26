import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from Utils.logger import get_logger
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from src.agreement_measure import AgreementMeasure


class DistanceBasedCellAgreement(AgreementMeasure):
    """
    A class to calculate Distance-Based Cell Agreement Algorithm (DBCAA).

    DBCAA is an algorithm that measures agreement in cell detection without
    relying on ground truth. It calculates agreement based on the spatial
    distances between cells detected by different annotators.
    """

    @get_logger().log_scope
    def calculate(self,
                  cell_positions: List[np.ndarray],
                  distance_threshold: float = 10.0) -> float:
        """
        Calculate DBCAA for cell detection data.

        Args:
            cell_positions (List[np.ndarray]): List of arrays containing cell
                positions from different annotators. Each array should have
                shape (n_cells, 2) where each row is (x, y) coordinates.
            distance_threshold (float, optional): Maximum distance (in pixels)
                for two cells to be considered matching. Defaults to 10.0.

        Returns:
            float: DBCAA value (0 to 1)
        """
        self.logger.info("Calculating Distance-Based Cell Agreement")

        # Validate inputs
        if not cell_positions:
            raise ValueError("No cell position data provided")

        if not all(isinstance(pos, np.ndarray) for pos in cell_positions):
            raise ValueError("All cell positions must be numpy arrays")

        if not all(
            pos.shape[1] == 2 for pos in cell_positions if len(pos) > 0
                   ):
            raise ValueError(
                "Cell position arrays must have shape (n_cells, 2)")

        # Calculate pairwise agreement
        n_annotators = len(cell_positions)
        pairwise_agreements = []

        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                agreement = self._calculate_pairwise_agreement(
                    cell_positions[i], cell_positions[j], distance_threshold)
                pairwise_agreements.append(agreement)
                self.logger.debug(
                    f"Agreement between annotator {i} and {j}:"
                    f" {agreement:.4f}")

        # Calculate overall agreement as average of pairwise agreements
        if not pairwise_agreements:
            self.logger.warning("No valid pairs for agreement calculation")
            return 0.0

        overall_agreement = np.mean(pairwise_agreements)
        self.logger.info(f"Overall DBCAA: {overall_agreement:.4f}")

        return overall_agreement

    @get_logger().log_scope
    def _calculate_pairwise_agreement(self,
                                      cells1,
                                      cells2,
                                      distance_threshold=10.0):
        """
        Calculate agreement between two sets of cell positions.

        Args:
            cells1 (np.ndarray): First set of cell positions,
                                    shape (n_cells, 2)
            cells2 (np.ndarray): Second set of cell positions,
                                    shape (m_cells, 2)
            distance_threshold (float): Maximum distance for cells to be
            considered matching

        Returns:
            float: Agreement score between 0 and 1
        """
        # Handle empty arrays
        if len(cells1) == 0 and len(cells2) == 0:
            return 1.0  # Perfect agreement if both are empty
        if len(cells1) == 0 or len(cells2) == 0:
            return 0.0  # No agreement if one is empty

        # Calculate pairwise distances between all cells
        distances = cdist(cells1, cells2)

        # Find optimal assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(distances)

        # Count matches (cells within threshold)
        matches = sum(
            distances[row_ind[i], col_ind[i]] <= distance_threshold
            for i in range(len(row_ind)))

        # Calculate agreement as proportion of matches to total cells
        total_cells = max(len(cells1), len(cells2))
        agreement = matches / total_cells

        return agreement

    @get_logger().log_scope
    def calculate_from_dataframe(self,
                                 df: pd.DataFrame,
                                 distance_threshold: float = 10.0) -> float:
        """
        Calculate DBCAA from a DataFrame containing cell position data.

        Args:
            df (pd.DataFrame): DataFrame with annotator cell positions.
                Expected format: Each column represents an annotator's data,
                with each cell containing a list of (x, y) coordinates.
            distance_threshold (float, optional): Maximum distance for
                matching. Defaults to 10.0.

        Returns:
            float: DBCAA value (0 to 1)
        """
        self.logger.info("Calculating DBCAA from DataFrame")

        # Convert DataFrame to list of cell position arrays
        cell_positions = []
        invalid_count = 0

        for col in df.columns:
            try:
                # Filter out None values and convert to numpy array
                valid_positions = [
                    pos for pos in df[col].tolist() if pos is not None]
                positions = np.array(valid_positions)

                # Check for NaN values
                if np.isnan(positions).any():
                    self.logger.warning(
                        f"NaN values found in column {col}, "
                        "filtering them out")
                    positions = positions[~np.isnan(positions).any(axis=1)]

                cell_positions.append(positions)
            except Exception as e:
                self.logger.warning(f"Error processing column {col}: {e}")
                invalid_count += 1
                # Add an empty array for this annotator
                cell_positions.append(np.array([]).reshape(0, 2))

        if invalid_count > 0:
            self.logger.warning(f"Skipped {invalid_count} invalid columns")

        # Calculate DBCAA
        return self.calculate(cell_positions, distance_threshold)

    @get_logger().log_scope
    def interpret_dbcaa(self, dbcaa: float) -> str:
        """
        Interpret the DBCAA value according to guidelines specific to this
            measure.

        Args:
            dbcaa (float): DBCAA value (typically between 0 and 1).

        Returns:
            str: Interpretation of the DBCAA value.
        """
        # DBCAA values are normally between 0 and 1
        # 0 represents random agreement or no agreement
        if dbcaa < 0:
            return "Invalid DBCAA value (should be non-negative)"
        elif dbcaa < 0.1:
            return "Poor agreement (close to random)"
        elif dbcaa < 0.2:
            return "Slight agreement"
        elif dbcaa < 0.4:
            return "Fair agreement"
        elif dbcaa < 0.6:
            return "Moderate agreement"
        elif dbcaa < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @get_logger().log_scope
    def get_dbcaa_statistics(self,
                             cell_positions: List[np.ndarray],
                             distance_thresholds:
                             List[float] = [5.0, 10.0, 15.0]
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

    @get_logger().log_scope
    def calculate_pairwise(self,
                           df: pd.DataFrame,
                           threshold=10.0) -> Dict[Tuple[str, str], float]:
        """
        Calculate Distance-Based Cell Agreement for each pair of annotators.

        Args:
            df (pd.DataFrame): DataFrame with annotator scores as columns.
            threshold (float): Distance threshold for matching cells.

        Returns:
            Dict[Tuple[str, str], float]: Dictionary with annotator pairs as
                keys and DBCAA values as values.
        """
        self.logger.info(
            "Calculating pairwise Distance-Based Cell Agreement values")

        pairwise_dbcaas = {}
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

                try:
                    # Extract cell coordinates
                    if isinstance(pair_df[columns[i]].iloc[0], str):
                        # If the data is stored as strings, parse them
                        cells1 = self._extract_cell_coordinates(
                            pair_df[columns[i]].iloc[0])
                        cells2 = self._extract_cell_coordinates(
                            pair_df[columns[j]].iloc[0])
                    else:
                        # If the data is already numpy arrays,use them directly
                        cells1 = pair_df[columns[i]].iloc[0]
                        cells2 = pair_df[columns[j]].iloc[0]

                    # Calculate agreement using the correct method
                    dbcaa = self._calculate_pairwise_agreement(cells1,
                                                               cells2,
                                                               threshold)

                    # Store result with annotator names as key
                    pair_key = (columns[i], columns[j])
                    pairwise_dbcaas[pair_key] = dbcaa

                    self.logger.debug(
                        f"DBCAA between {columns[i]} and {columns[j]}: "
                        f"{dbcaa:.4f}")
                except Exception as e:
                    self.logger.warning(
                        f"Error calculating DBCAA for pair {columns[i]} and "
                        f"{columns[j]}: {str(e)}")
                    continue

        return pairwise_dbcaas

    @get_logger().log_scope
    def _extract_cell_coordinates(self, cell_str):
        """
        Extract cell coordinates from a string representation.

        Args:
            cell_str (str): String representation of cell coordinates,
                e.g., "10,10;20,20;30,30"

        Returns:
            np.ndarray: Array of cell coordinates with shape (n_cells, 2)
        """
        if not cell_str or not isinstance(cell_str, str):
            return np.array([]).reshape(0, 2)

        try:
            # Split the string by semicolons to get individual cell coordinates
            cells = cell_str.split(';')

            # Parse each cell coordinate
            coordinates = []
            for cell in cells:
                if ',' in cell:
                    try:
                        x, y = cell.split(',')
                        coordinates.append([float(x), float(y)])
                    except ValueError:
                        # Skip invalid coordinates
                        continue

            return np.array(coordinates)
        except Exception as e:
            self.logger.warning(f"Error extracting cell coordinates: {str(e)}")
            return np.array([]).reshape(0, 2)
