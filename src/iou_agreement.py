import numpy as np
import pandas as pd
from typing import Dict, List, Union
from Utils.logger import get_logger, LogLevel


class IoUAgreement:
    """
    A class to calculate Intersection over Union (IoU) based agreement
    measures.

    IoU is commonly used for evaluating segmentation tasks. This class provides
    methods to calculate IoU between multiple annotators and interpret the
    results.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize the IoUAgreement calculator.

        Args:
            level (LogLevel, optional):
            Logging level. Defaults to LogLevel.INFO.
        """
        self._logger = get_logger(level)

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    def calculate_pairwise_iou(self,
                               mask1: np.ndarray,
                               mask2: np.ndarray) -> float:
        """
        Calculate IoU between two binary masks.

        Args:
            mask1 (np.ndarray): First binary mask
            mask2 (np.ndarray): Second binary mask

        Returns:
            float: IoU score between 0 and 1
        """
        if mask1.shape != mask2.shape:
            raise ValueError("Masks must have the same shape")

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        # Handle edge case where both masks are empty
        if union == 0:
            return 1.0  # Perfect agreement if both masks are empty

        # Calculate IoU
        iou = intersection / union
        return iou

    def calculate_mean_iou(self, masks: List[np.ndarray]) -> float:
        """
        Calculate mean IoU across all pairs of masks.

        Args:
            masks (List[np.ndarray]): List of binary masks from different
            annotators

        Returns:
            float: Mean IoU score between 0 and 1
        """
        if not masks:
            raise ValueError("No masks provided")

        if len(masks) < 2:
            raise ValueError("At least two masks are required")

        # Validate masks
        for i, mask in enumerate(masks):
            if not isinstance(mask, np.ndarray):
                raise ValueError(f"Mask {i} must be a numpy array")

        # Now that we know all masks are numpy arrays, we can check their
        # shapes
        shape = masks[0].shape
        for i, mask in enumerate(masks):
            if mask.shape != shape:
                raise ValueError(f"All masks must have the same shape. "
                                 f"Mask {i} has different shape")

        # Calculate pairwise IoU for all pairs
        n_annotators = len(masks)
        iou_scores = []

        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                iou = self.calculate_pairwise_iou(masks[i], masks[j])
                iou_scores.append(iou)

        # Return mean IoU
        return np.mean(iou_scores)

    def calculate_from_dataframe(self, df: pd.DataFrame) -> float:
        """
        Calculate mean IoU from a DataFrame containing binary annotations.

        Args:
            df (pd.DataFrame): DataFrame where each column is an annotator and
                each row contains binary values (0 or 1)

        Returns:
            float: Mean IoU score between 0 and 1
        """
        self.logger.info("Calculating IoU from DataFrame")

        # Check if data is binary
        unique_values = np.unique(df.values[~np.isnan(df.values)])
        if not np.all(np.isin(unique_values, [0, 1])):
            self.logger.warning("Data contains non-binary values. IoU is "
                                "designed for binary data.")

        # Convert DataFrame to list of masks
        masks = []
        for col in df.columns:
            mask = df[col].values
            masks.append(mask)

        # Calculate mean IoU
        if len(masks) < 2:
            self.logger.warning("Not enough annotators for IoU calculation")
            return 0.0

        return self.calculate_mean_iou(masks)

    def interpret_iou(self, iou: float) -> str:
        """
        Interpret the IoU value.

        Args:
            iou (float): IoU value

        Returns:
            str: Interpretation of the IoU value
        """
        if iou < 0.2:
            return "Poor agreement"
        elif iou < 0.4:
            return "Fair agreement"
        elif iou < 0.6:
            return "Moderate agreement"
        elif iou < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def get_iou_statistics(self,
                           masks: List[np.ndarray]) -> Dict[str,
                                                            Union[float, str]]:
        """
        Calculate IoU statistics for a list of masks.

        Args:
            masks (List[np.ndarray]):
                List of binary masks from different annotators

        Returns:
            Dict[str, Union[float, str]]: Dictionary with IoU statistics
        """
        results = {}

        # Calculate mean IoU
        mean_iou = self.calculate_mean_iou(masks)
        results['mean_iou'] = mean_iou
        results['interpretation'] = self.interpret_iou(mean_iou)

        # Calculate pairwise IoUs
        n_annotators = len(masks)
        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                iou = self.calculate_pairwise_iou(masks[i], masks[j])
                results[f'iou_{i+1}_{j+1}'] = iou

        # Calculate min and max IoU
        pairwise_ious = [v for k, v in results.items() if k.startswith('iou_')]
        results['min_iou'] = min(pairwise_ious)
        results['max_iou'] = max(pairwise_ious)

        return results
