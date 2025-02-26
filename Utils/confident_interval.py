import numpy as np
import scipy.stats
from scipy.stats import norm


class ConfidenceIntervalCalculator:
    """
    A class to calculate different confidence intervals:
     * Wilson confidence interval
     * Standard confidence interval
    """

    def __init__(self, confidence: float = 0.95):
        """
        Initialize the ConfidenceIntervalCalculator.

        Args:
            confidence (float): Confidence level (0-1). Defaults to 0.95.

        Raises:
            ValueError: If confidence is not between 0 and 1.
        """
        if not 0 < confidence < 1:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence}"
            )
        self.confidence = confidence

    def wilson_interval(self, p_hat: float, n: int) -> dict[str, float]:
        """
        Calculate Wilson confidence interval for a proportion.

        Args:
            p_hat (float): Proportion estimate.
            n (int): Sample size.

        Raises:
            ValueError: If p_hat not in [0,1] or n <= 0.

        Returns:
            dict[str, float]: Dictionary containing:
                - estimate: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        if not 0 <= p_hat <= 1:
            raise ValueError(
                f"Proportion must be between 0 and 1, got {p_hat}")
        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")
        z = norm.ppf(1 - (1 - self.confidence) / 2)
        z2 = z * z

        denominator = 1 + z2 / n
        center = (p_hat + z2 / (2 * n)) / denominator

        variance = p_hat * (1 - p_hat) / n + z2 / (4 * n * n)
        interval = z * np.sqrt(variance) / denominator

        ci_lower = max(0, center - interval)
        ci_upper = min(1, center + interval)

        return {
            'estimate': p_hat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    def clopper_pearson_interval(self,
                                 p_hat: float,
                                 n: int) -> dict[str, float]:
        """
        Calculate Clopper-Pearson (Exact) confidence interval for a proportion.

        Args:
            p_hat (float): Proportion estimate.
            n (int): Sample size.

        Raises:
            ValueError: If p_hat not in [0,1] or n <= 0.

        Returns:
            dict[str, float]: Dictionary containing:
                - estimate: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        if not 0 <= p_hat <= 1:
            raise ValueError(
                f"Proportion must be between 0 and 1, got {p_hat}")
        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")

        alpha = 1 - self.confidence
        lower_bound = 0 if p_hat == 0 else \
            scipy.stats.beta.ppf(alpha / 2, p_hat * n, (1 - p_hat) * n + 1)
        upper_bound = 1 if p_hat == 1 else \
            scipy.stats.beta.ppf(1 - alpha / 2, p_hat * n + 1, (1 - p_hat) * n)

        return {
            'estimate': p_hat,
            'ci_lower': lower_bound,
            'ci_upper': upper_bound
        }

    def standard_interval(self, p_hat: float, n: int) -> dict[str, float]:
        """
        Calculate standard confidence interval for a proportion.

        Args:
            p_hat (float): Proportion estimate.
            n (int): Sample size.

        Raises:
            ValueError: If p_hat not in [0,1] or n <= 0.

        Returns:
            Dict containing:
                - estimate: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        # Validate inputs
        if not 0 <= p_hat <= 1:
            raise ValueError(
                f"Proportion must be between 0 and 1, got {p_hat}")
        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")

        z = norm.ppf(1 - (1 - self.confidence) / 2)
        standard_error = np.sqrt(p_hat * (1 - p_hat) / n)

        ci_lower = max(0, p_hat - z * standard_error)
        ci_upper = min(1, p_hat + z * standard_error)

        return {
            'estimate': p_hat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
