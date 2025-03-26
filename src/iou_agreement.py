import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple
from Utils.logger import get_logger
from src.agreement_measure import AgreementMeasure


class IoUAgreement(AgreementMeasure):
    """
    A class to calculate Intersection over Union (IoU) agreement between
    annotators.

    IoU measures the overlap between annotations by comparing the ratio of
    intersection area to union area.
    """

    @get_logger().log_scope
    def calculate(self, annotations: List[np.ndarray]) -> float:
        """
        Calculate IoU agreement for the given annotations.

        Args:
            annotations (List[np.ndarray]):
                List of binary masks or bounding boxes.

        Returns:
            float: IoU agreement value (between 0 and 1).
        """
        self.logger.info("Calculating IoU agreement")
        # Validate input
        if not annotations or len(annotations) < 2:
            self.logger.warning("Insufficient annotations provided")
            return 0.0

        # Calculate pairwise IoU values
        pairwise_ious = self.calculate_pairwise(annotations)

        # Calculate average IoU
        if not pairwise_ious:
            self.logger.warning("No valid IoU values calculated")
            return 0.0

        avg_iou = sum(pairwise_ious.values()) / len(pairwise_ious)

        self.logger.info(f"Average IoU: {avg_iou:.4f}")

        return avg_iou

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

    @get_logger().log_scope
    def interpret_iou(self, iou: float) -> str:
        """
        Interpret the IoU value.

        Args:
            iou (float): IoU value (between 0 and 1).

        Returns:
            str: Interpretation of the IoU value.
        """
        # Use custom interpretation for IoU
        if iou < 0 or iou > 1:
            return "Invalid IoU value"
        elif iou < 0.2:
            return "Poor agreement"
        elif iou < 0.4:
            return "Fair agreement"
        elif iou < 0.6:
            return "Moderate agreement"
        elif iou < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @get_logger().log_scope
    def get_iou_statistics(self,
                           annotations: List[np.ndarray]
                           ) -> Dict[str, Union[float, str]]:
        """
        Calculate IoU agreement and provide statistics.

        Args:
            annotations (List[np.ndarray]):
                List of binary masks or bounding boxes.

        Returns:
            Dict[str, Union[float, str]]:
                Dictionary with IoU value and interpretation.
        """
        results = {}

        # Calculate IoU
        iou = self.calculate(annotations)
        results['iou'] = iou
        results['interpretation'] = self.interpret_iou(iou)

        # Calculate pairwise IoU values
        pairwise_ious = self.calculate_pairwise(annotations)

        # Add individual pairwise IoUs to results
        for (i, j), iou_value in pairwise_ious.items():
            results[f'iou_{i+1}_{j+1}'] = iou_value

        # Calculate min, max, and std of pairwise IoU values
        if pairwise_ious:
            iou_values = list(pairwise_ious.values())
            results['min_iou'] = min(iou_values)
            results['max_iou'] = max(iou_values)
            results['std_iou'] = np.std(iou_values)
        else:
            results['min_iou'] = 0.0
            results['max_iou'] = 0.0
            results['std_iou'] = 0.0

        return results

    @get_logger().log_scope
    def calculate_pairwise(self,
                           annotations: List[np.ndarray]
                           ) -> Dict[Tuple[int, int], float]:
        """
        Calculate pairwise IoU values between all annotators.

        Args:
            annotations (List[np.ndarray]):
                List of binary masks or bounding boxes.

        Returns:
            Dict[Tuple[int, int], float]: Dictionary mapping pairs of annotator
                indices to their IoU values.
        """
        self.logger.info("Calculating pairwise IoU values")

        pairwise_ious = {}
        n_annotators = len(annotations)

        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                # Calculate IoU between annotator i and j
                iou = self.calculate_pairwise_iou(annotations[i],
                                                  annotations[j])

                # Store result with annotator indices as key
                pair_key = (i, j)
                pairwise_ious[pair_key] = iou

                self.logger.debug(
                    f"IoU between annotator {i} and {j}: {iou:.4f}")

        return pairwise_ious
