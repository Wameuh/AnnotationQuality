import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from Utils.logger import get_logger, LogLevel
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure


class BoundaryWeightedFleissKappa:
    """
    A class to calculate Boundary-Weighted Fleiss' Kappa (BWFK).

    BWFK is an extension of Fleiss' Kappa that reduces the impact of minor
    disagreements that often occur along boundaries in segmentation tasks.
    It is particularly useful for evaluating agreement in tissue segmentation
    or other spatial annotation tasks.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the BoundaryWeightedFleissKappa calculator.

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
                  segmentations: List[np.ndarray],
                  boundary_width: int = 2,
                  weight_factor: float = 0.5) -> float:
        """
        Calculate Boundary-Weighted Fleiss' Kappa for segmentation data.

        Args:
            segmentations (List[np.ndarray]): List of segmentation masks from
                different annotators. Each mask should be a binary numpy array
                with the same shape.
            boundary_width (int, optional): Width of the boundary region in
                pixels. Defaults to 2.
            weight_factor (float, optional): Weight factor for boundary
                regions. Should be between 0 and 1. Lower values reduce the
                impact of disagreements in boundary regions. Defaults to 0.5.

        Returns:
            float: BWFK value (-1 to 1)
        """
        self.logger.info("Calculating Boundary-Weighted Fleiss' Kappa")

        # Validate inputs
        if not segmentations:
            raise ValueError("No segmentation masks provided")

        if not all(isinstance(seg, np.ndarray) for seg in segmentations):
            raise ValueError("All segmentations must be numpy arrays")

        # Get shape from first segmentation
        reference_shape = segmentations[0].shape

        # Check that all segmentations have the same shape
        if not all(seg.shape == reference_shape for seg in segmentations):
            raise ValueError("All segmentations must have the same shape")

        # Check that all segmentations are binary
        for i, seg in enumerate(segmentations):
            unique_values = np.unique(seg)
            if not np.all(np.isin(unique_values, [0, 1])):
                self.logger.warning(
                    f"Segmentation {i} is not binary. Converting to binary.")
                segmentations[i] = (seg > 0).astype(np.int32)

        # Identify boundary regions in each segmentation
        boundary_masks = [self._get_boundary_mask(seg, boundary_width)
                          for seg in segmentations]

        # Combine boundary masks to get overall boundary region
        combined_boundary = np.logical_or.reduce(
            boundary_masks).astype(np.int32)

        # Create weight matrix
        # (1.0 for non-boundary, weight_factor for boundary)
        weights = np.where(combined_boundary, weight_factor, 1.0)

        # Calculate weighted observed agreement
        observed_agreement = self._calculate_weighted_observed_agreement(
            segmentations, weights)

        # Calculate weighted chance agreement
        chance_agreement = self._calculate_weighted_chance_agreement(
            segmentations, weights)

        # Calculate kappa
        if chance_agreement == 1.0:
            self.logger.warning(
                "Chance agreement is 1.0, perfect agreement by chance")
            return 0.0

        kappa = (observed_agreement - chance_agreement) / (
            1.0 - chance_agreement)

        self.logger.info(f"Boundary-Weighted Fleiss' Kappa: {kappa:.4f}")
        self.logger.debug(f"Observed agreement: {observed_agreement:.4f}")
        self.logger.debug(f"Chance agreement: {chance_agreement:.4f}")

        return kappa

    def _get_boundary_mask(self,
                           segmentation: np.ndarray,
                           width: int) -> np.ndarray:
        """
        Get boundary mask for a segmentation.

        Args:
            segmentation (np.ndarray): Binary segmentation mask.
            width (int): Width of the boundary in pixels.

        Returns:
            np.ndarray: Binary mask of boundary regions.
        """
        # Create structuring element for dilation/erosion
        struct = generate_binary_structure(segmentation.ndim, 1)

        # Dilate and erode to get inner and outer boundaries
        dilated = binary_dilation(segmentation, struct, iterations=width)
        eroded = binary_erosion(segmentation, struct, iterations=width)

        # Boundary is the difference between dilated and eroded
        boundary = np.logical_xor(dilated, eroded)

        return boundary

    def _calculate_weighted_observed_agreement(self,
                                               segmentations: List[np.ndarray],
                                               weights: np.ndarray) -> float:
        """
        Calculate weighted observed agreement.

        Args:
            segmentations (List[np.ndarray]): List of segmentation masks.
            weights (np.ndarray): Weight matrix.

        Returns:
            float: Weighted observed agreement.
        """
        n_annotators = len(segmentations)
        if n_annotators < 2:
            raise ValueError(
                "Need at least 2 annotators to calculate agreement")

        # Stack segmentations for easier processing
        stacked_segs = np.stack(segmentations, axis=0)

        # Calculate agreement for each pixel
        total_agreement = 0.0
        total_weight = np.sum(weights)

        # For each pair of annotators
        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                # Agreement is 1 where both annotators agree, 0 otherwise
                agreement = (
                    stacked_segs[i] == stacked_segs[j]).astype(np.float32)

                # Weight the agreement
                weighted_agreement = np.sum(agreement * weights)

                # Add to total
                total_agreement += weighted_agreement

        # Normalize by number of pairs and total weight
        n_pairs = (n_annotators * (n_annotators - 1)) / 2
        normalized_agreement = total_agreement / (n_pairs * total_weight)

        return normalized_agreement

    def _calculate_weighted_chance_agreement(self,
                                             segmentations: List[np.ndarray],
                                             weights: np.ndarray) -> float:
        """
        Calculate weighted chance agreement.

        Args:
            segmentations (List[np.ndarray]): List of segmentation masks.
            weights (np.ndarray): Weight matrix.

        Returns:
            float: Weighted chance agreement.
        """

        # Calculate class probabilities (foreground and background)
        foreground_probs = []
        for seg in segmentations:
            foreground_prob = np.mean(seg)
            foreground_probs.append(foreground_prob)

        # Average probability across annotators
        avg_foreground_prob = np.mean(foreground_probs)
        avg_background_prob = 1.0 - avg_foreground_prob

        # Calculate chance agreement
        chance_agreement = (
            avg_foreground_prob ** 2) + (avg_background_prob ** 2)

        # Apply weights
        weighted_chance = np.sum(weights) * chance_agreement / np.sum(weights)

        return weighted_chance

    def calculate_from_dataframe(self,
                                 df: pd.DataFrame,
                                 image_shape: Tuple[int, int],
                                 boundary_width: int = 2,
                                 weight_factor: float = 0.5) -> float:
        """
        Calculate BWFK from a DataFrame containing flattened segmentation data.

        Args:
            df (pd.DataFrame): DataFrame with annotator segmentations as
                columns. Each row represents a pixel, and each column an
                annotator.
            image_shape (Tuple[int, int]): Original shape of the segmentation
                images.
            boundary_width (int, optional): Width of boundary region.
                Defaults to 2.
            weight_factor (float, optional): Weight for boundary regions.
                Defaults to 0.5.

        Returns:
            float: BWFK value (-1 to 1)
        """
        self.logger.info("Calculating BWFK from DataFrame")

        # Convert DataFrame to list of segmentation masks
        segmentations = []
        for col in df.columns:
            # Reshape flattened data back to 2D
            seg = df[col].values.reshape(image_shape)
            segmentations.append(seg)

        # Calculate BWFK
        return self.calculate(segmentations, boundary_width, weight_factor)

    def interpret_bwfk(self, bwfk: float) -> str:
        """
        Interpret the BWFK value.

        Args:
            bwfk (float): BWFK value.

        Returns:
            str: Interpretation of the BWFK value.
        """
        if bwfk < 0:
            return "Poor agreement (worse than chance)"
        elif bwfk < 0.2:
            return "Slight agreement"
        elif bwfk < 0.4:
            return "Fair agreement"
        elif bwfk < 0.6:
            return "Moderate agreement"
        elif bwfk < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def get_bwfk_statistics(self,
                            segmentations: List[np.ndarray],
                            boundary_widths: List[int] = [1, 2, 3, 5],
                            weight_factors: List[float] = [0.25, 0.5, 0.75]
                            ) -> Dict[str, float]:
        """
        Calculate BWFK with different parameters and provide statistics.

        Args:
            segmentations (List[np.ndarray]): List of segmentation masks.
            boundary_widths (List[int], optional): List of boundary widths to
                try.
            weight_factors (List[float], optional): List of weight factors to
                try.

        Returns:
            Dict[str, float]: Dictionary with BWFK values and statistics.
        """
        results = {}

        # Calculate standard BWFK (default parameters)
        standard_bwfk = self.calculate(segmentations)
        results['bwfk_standard'] = standard_bwfk
        results['interpretation_standard'] = self.interpret_bwfk(standard_bwfk)

        # Calculate BWFK with different boundary widths
        for width in boundary_widths:
            bwfk = self.calculate(segmentations,
                                  boundary_width=width,
                                  weight_factor=0.5)
            results[f'bwfk_width_{width}'] = bwfk
            results[f'interpretation_width_{width}'] = \
                self.interpret_bwfk(bwfk)

        # Calculate BWFK with different weight factors
        for factor in weight_factors:
            bwfk = self.calculate(segmentations,
                                  boundary_width=2,
                                  weight_factor=factor)
            results[f'bwfk_factor_{factor}'] = bwfk
            results[f'interpretation_factor_{factor}'] = \
                self.interpret_bwfk(bwfk)

        return results
