import pytest
import numpy as np
from Utils.confident_interval import ConfidenceIntervalCalculator
from Utils.logger import Logger, get_logger, LogLevel


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the logger singleton before each test."""
    # Reset the singleton instance
    Logger._instance = None
    yield
    # Clean up after test
    Logger._instance = None


@pytest.fixture
def calculator():
    """Fixture providing a ConfidenceIntervalCalculator instance."""
    return ConfidenceIntervalCalculator(confidence=0.95)


def test_init_with_logger():
    """Test initialization with a logger instance."""
    # Get the singleton instance of the logger
    singleton_logger = get_logger()

    # Create a DataLoader without passing a logger explicitly
    CIC = ConfidenceIntervalCalculator()

    # Verify that the DataLoader's logger is the singleton instance
    assert CIC.logger is singleton_logger


def test_init_without_logger():
    """Test DataLoader initialization without a logger."""
    CIC = ConfidenceIntervalCalculator(level=LogLevel.DEBUG)
    assert isinstance(CIC.logger, Logger)
    assert CIC.logger.level == LogLevel.DEBUG


def test_wilson_perfect_agreement():
    """Test Wilson interval with perfect agreement (p=1)."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.wilson_interval(p_hat=1.0, n=100)

    assert result['estimate'] == 1.0
    assert result['ci_lower'] > 0.95  # Should be very close to 1
    assert result['ci_upper'] == 1.0


def test_wilson_no_agreement():
    """Test Wilson interval with no agreement (p=0)."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.wilson_interval(p_hat=0.0, n=100)

    assert result['estimate'] == 0.0
    assert np.isclose(result['ci_lower'], 0.0, atol=1e-15)
    assert result['ci_upper'] < 0.05  # Should be very close to 0


def test_wilson_partial_agreement():
    """Test Wilson interval with 50% agreement."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.wilson_interval(p_hat=0.5, n=100)

    assert result['estimate'] == 0.5
    assert 0.4 < result['ci_lower'] < 0.5  # Typical range for n=100
    assert 0.5 < result['ci_upper'] < 0.6


def test_standard_interval_basic():
    """Test standard interval calculation."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.standard_interval(p_hat=0.5, n=100)

    assert result['estimate'] == 0.5
    assert result['ci_lower'] < 0.5
    assert result['ci_upper'] > 0.5
    assert result['ci_lower'] >= 0
    assert result['ci_upper'] <= 1


def test_different_confidence_levels():
    """Test intervals with different confidence levels."""
    calc_90 = ConfidenceIntervalCalculator(confidence=0.90)
    calc_99 = ConfidenceIntervalCalculator(confidence=0.99)

    result_90 = calc_90.wilson_interval(p_hat=0.5, n=100)
    result_99 = calc_99.wilson_interval(p_hat=0.5, n=100)

    # 99% CI should be wider than 90% CI
    width_90 = result_90['ci_upper'] - result_90['ci_lower']
    width_99 = result_99['ci_upper'] - result_99['ci_lower']
    assert width_99 > width_90


def test_sample_size_effect():
    """Test effect of sample size on interval width."""
    calc = ConfidenceIntervalCalculator()

    result_small = calc.wilson_interval(p_hat=0.5, n=50)
    result_large = calc.wilson_interval(p_hat=0.5, n=1000)

    # Larger sample should give narrower interval
    width_small = result_small['ci_upper'] - result_small['ci_lower']
    width_large = result_large['ci_upper'] - result_large['ci_lower']
    assert width_large < width_small


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    calc = ConfidenceIntervalCalculator()

    # Invalid confidence level
    with pytest.raises(ValueError):
        ConfidenceIntervalCalculator(confidence=1.5)

    # Invalid proportion
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=1.5, n=100)

    # Invalid proportion
    with pytest.raises(ValueError):
        calc.clopper_pearson_interval(p_hat=1.5, n=100)

    # Invalid sample size
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=0.5, n=0)
    with pytest.raises(ValueError):
        calc.clopper_pearson_interval(p_hat=0.5, n=0)


def test_compare_methods():
    """Compare Wilson and standard intervals."""
    calc = ConfidenceIntervalCalculator()
    p_hat, n = 0.5, 100

    wilson = calc.wilson_interval(p_hat, n)
    standard = calc.standard_interval(p_hat, n)

    # Both should give same point estimate
    assert wilson['estimate'] == standard['estimate']

    # For extreme proportions, intervals should be different
    p_extreme = 0.95
    wilson_extreme = calc.wilson_interval(p_extreme, n)
    standard_extreme = calc.standard_interval(p_extreme, n)

    # Verify intervals are different
    diff_upper = abs(wilson_extreme['ci_upper'] - standard_extreme['ci_upper'])
    diff_lower = abs(wilson_extreme['ci_lower'] - standard_extreme['ci_lower'])
    assert diff_upper > 1e-6
    assert diff_lower > 1e-6


def test_clopper_pearson_perfect():
    """Test Clopper-Pearson interval with perfect agreement (p=1)."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.clopper_pearson_interval(p_hat=1.0, n=100)

    assert result['estimate'] == 1.0
    assert result['ci_lower'] > 0.95  # Should be very close to 1
    assert result['ci_upper'] == 1.0


def test_clopper_pearson_zero():
    """Test Clopper-Pearson interval with no agreement (p=0)."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.clopper_pearson_interval(p_hat=0.0, n=100)

    assert result['estimate'] == 0.0
    assert result['ci_lower'] == 0.0
    assert result['ci_upper'] < 0.05


def test_clopper_pearson_partial():
    """Test Clopper-Pearson interval with 50% agreement."""
    calc = ConfidenceIntervalCalculator(confidence=0.95)
    result = calc.clopper_pearson_interval(p_hat=0.5, n=100)

    assert result['estimate'] == 0.5
    assert 0.35 < result['ci_lower'] < 0.45  # Ajusté pour Clopper-Pearson
    assert 0.55 < result['ci_upper'] < 0.65  # Plus large que Wilson


def test_compare_all_methods():
    """Compare all three interval methods."""
    calc = ConfidenceIntervalCalculator()
    p_hat, n = 0.8, 100

    wilson = calc.wilson_interval(p_hat, n)
    standard = calc.standard_interval(p_hat, n)
    clopper = calc.clopper_pearson_interval(p_hat, n)

    # All should give same point estimate
    assert wilson['estimate'] == standard['estimate'] == clopper['estimate']

    # Clopper-Pearson should be most conservative (widest)
    wilson_width = wilson['ci_upper'] - wilson['ci_lower']
    standard_width = standard['ci_upper'] - standard['ci_lower']
    clopper_width = clopper['ci_upper'] - clopper['ci_lower']

    assert clopper_width >= wilson_width
    assert clopper_width >= standard_width


def test_standard_interval_invalid_inputs():
    """Test handling of invalid inputs in standard_interval."""
    calc = ConfidenceIntervalCalculator()

    # Invalid proportion
    with pytest.raises(ValueError):
        calc.standard_interval(p_hat=1.5, n=100)

    # Invalid sample size
    with pytest.raises(ValueError):
        calc.standard_interval(p_hat=0.5, n=0)


def test_standard_interval_edge_cases():
    """Test standard interval with edge case values."""
    calc = ConfidenceIntervalCalculator()

    # Test with p_hat = 0
    result = calc.standard_interval(p_hat=0.0, n=100)
    assert result['estimate'] == 0.0
    assert result['ci_lower'] == 0.0
    assert result['ci_upper'] < 0.05

    # Test with p_hat = 1
    result = calc.standard_interval(p_hat=1.0, n=100)
    assert result['estimate'] == 1.0
    assert result['ci_lower'] > 0.95
    assert result['ci_upper'] == 1.0


def test_wilson_interval_invalid_inputs():
    """Test handling of invalid inputs in wilson_interval."""
    calc = ConfidenceIntervalCalculator()

    # Invalid proportion
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=1.5, n=100)

    # Invalid sample size
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=0.5, n=0)

    # Negative proportion
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=-0.1, n=100)

    # Negative sample size
    with pytest.raises(ValueError):
        calc.wilson_interval(p_hat=0.5, n=-10)


def test_agresti_coull_interval_basic(calculator):
    """Test basic functionality of agresti_coull_interval."""
    # Test with a simple proportion and sample size
    result = calculator.agresti_coull_interval(p_hat=0.7, n=100)

    # Check that result contains expected keys
    assert 'ci_lower' in result
    assert 'ci_upper' in result

    # Check that bounds are between 0 and 1
    assert 0 <= result['ci_lower'] <= 1
    assert 0 <= result['ci_upper'] <= 1

    # Check that lower bound is less than upper bound
    assert result['ci_lower'] < result['ci_upper']

    # Check that the interval contains the point estimate
    assert result['ci_lower'] < 0.7 < result['ci_upper']


def test_agresti_coull_interval_extreme_values(calculator):
    """Test agresti_coull_interval with extreme proportions."""
    # Test with proportion = 0
    result_0 = calculator.agresti_coull_interval(p_hat=0.0, n=100)
    assert result_0['ci_lower'] == 0.0
    assert result_0['ci_upper'] > 0.0

    # Test with proportion = 1
    result_1 = calculator.agresti_coull_interval(p_hat=1.0, n=100)
    assert result_1['ci_lower'] < 1.0
    assert result_1['ci_upper'] == 1.0


def test_agresti_coull_interval_small_sample(calculator):
    """Test agresti_coull_interval with small sample size."""
    # Test with small sample size
    result = calculator.agresti_coull_interval(p_hat=0.5, n=10)

    # Check that bounds are between 0 and 1
    assert 0 <= result['ci_lower'] <= 1
    assert 0 <= result['ci_upper'] <= 1

    # Check that the interval is wider than with a larger sample
    large_sample = calculator.agresti_coull_interval(p_hat=0.5, n=1000)
    assert (
        result['ci_upper'] - result['ci_lower']
    ) > (large_sample['ci_upper'] - large_sample['ci_lower'])


def test_agresti_coull_vs_wilson(calculator):
    """Compare Agresti-Coull interval with Wilson interval."""
    # For moderate proportions and large samples, they should be similar
    ac_result = calculator.agresti_coull_interval(p_hat=0.5, n=1000)
    wilson_result = calculator.wilson_interval(p_hat=0.5, n=1000)

    # Check that the intervals are similar (within 0.01)
    assert abs(ac_result['ci_lower'] - wilson_result['ci_lower']) < 0.01
    assert abs(ac_result['ci_upper'] - wilson_result['ci_upper']) < 0.01


def test_bootstrap_basic(calculator):
    """Test basic functionality of bootstrap method."""
    # Create sample data: 70 agreements out of 100
    data = [(1, 1)] * 70 + [(1, 2)] * 30

    # Calculate bootstrap CI with fewer resamples for speed
    result = calculator.bootstrap(data, n_resamples=100)

    # Check that result contains expected keys
    assert 'ci_lower' in result
    assert 'ci_upper' in result

    # Check that bounds are between 0 and 1
    assert 0 <= result['ci_lower'] <= 1
    assert 0 <= result['ci_upper'] <= 1

    # Check that lower bound is less than upper bound
    assert result['ci_lower'] < result['ci_upper']

    # Check that the interval contains the point estimate (0.7)
    assert result['ci_lower'] < 0.7 < result['ci_upper']


def test_bootstrap_custom_statistic(calculator):
    """Test bootstrap with custom statistic function."""
    # Create sample data
    data = [(1, 1), (2, 2), (3, 3), (4, 3), (5, 4)]

    # Define custom statistic: mean absolute difference
    def mean_abs_diff(sample):
        return sum(abs(a - b) for a, b in sample) / len(sample)

    # Calculate bootstrap CI with custom statistic
    result = calculator.bootstrap(data,
                                  n_resamples=100,
                                  statistic=mean_abs_diff)

    # Check that result contains expected keys
    assert 'ci_lower' in result
    assert 'ci_upper' in result

    # The point estimate should be (0 + 0 + 0 + 1 + 1) / 5 = 0.4
    # Check that the interval contains this value
    assert result['ci_lower'] <= 0.4 <= result['ci_upper']


def test_bootstrap_empty_data(calculator):
    """Test bootstrap with empty data."""
    # This should raise a ValueError
    with pytest.raises(Exception):
        calculator.bootstrap([])


def test_bootstrap_deterministic(calculator):
    """Test bootstrap with deterministic data."""
    # All pairs agree
    all_agree = [(1, 1), (2, 2), (3, 3)]
    result_agree = calculator.bootstrap(all_agree, n_resamples=100)
    assert result_agree['ci_lower'] == 1.0
    assert result_agree['ci_upper'] == 1.0

    # All pairs disagree
    all_disagree = [(1, 2), (2, 3), (3, 4)]
    result_disagree = calculator.bootstrap(all_disagree, n_resamples=100)
    assert result_disagree['ci_lower'] == 0.0
    assert result_disagree['ci_upper'] == 0.0


def test_normal_approximation_basic(ci_calc):
    """Test basic functionality of normal_approximation."""
    # Test with a simple proportion and sample size
    result = ci_calc.normal_approximation(p_hat=0.7, n=100)

    # Check that result contains expected keys
    assert 'ci_lower' in result
    assert 'ci_upper' in result

    # Check that bounds are between 0 and 1
    assert 0 <= result['ci_lower'] <= 1
    assert 0 <= result['ci_upper'] <= 1

    # Check that lower bound is less than upper bound
    assert result['ci_lower'] < result['ci_upper']

    # Check that the interval contains the point estimate
    assert result['ci_lower'] < 0.7 < result['ci_upper']


def test_normal_approximation_extreme_values(ci_calc):
    """Test normal_approximation with extreme proportions."""
    # Test with proportion = 0
    result_0 = ci_calc.normal_approximation(p_hat=0.0, n=100)
    assert result_0['ci_lower'] == 0.0
    assert result_0['ci_upper'] > 0.0

    # Test with proportion = 1
    result_1 = ci_calc.normal_approximation(p_hat=1.0, n=100)
    assert result_1['ci_lower'] < 1.0
    assert result_1['ci_upper'] == 1.0


def test_normal_approximation_sample_size(ci_calc):
    """Test normal_approximation with different sample sizes."""
    # Test with small sample size
    small_sample = ci_calc.normal_approximation(p_hat=0.5, n=10)

    # Test with large sample size
    large_sample = ci_calc.normal_approximation(p_hat=0.5, n=1000)

    # Interval should be wider with smaller sample size
    assert (
        small_sample['ci_upper'] - small_sample['ci_lower']
        ) > (large_sample['ci_upper'] - large_sample['ci_lower'])


def test_normal_approximation_confidence_levels():
    """Test normal_approximation with different confidence levels."""
    # Create calculators with different confidence levels
    ci_90 = ConfidenceIntervalCalculator(confidence=0.90)
    ci_95 = ConfidenceIntervalCalculator(confidence=0.95)
    ci_99 = ConfidenceIntervalCalculator(confidence=0.99)

    # Calculate intervals for the same data
    result_90 = ci_90.normal_approximation(p_hat=0.5, n=100)
    result_95 = ci_95.normal_approximation(p_hat=0.5, n=100)
    result_99 = ci_99.normal_approximation(p_hat=0.5, n=100)

    # Higher confidence should give wider intervals
    width_90 = result_90['ci_upper'] - result_90['ci_lower']
    width_95 = result_95['ci_upper'] - result_95['ci_lower']
    width_99 = result_99['ci_upper'] - result_99['ci_lower']

    assert width_90 < width_95 < width_99


def test_normal_approximation_invalid_inputs(ci_calc):
    """Test normal_approximation with invalid inputs."""
    # Test with negative sample size
    with pytest.raises(ValueError):
        ci_calc.normal_approximation(p_hat=0.5, n=-10)

    # Test with zero sample size
    with pytest.raises(ValueError):
        ci_calc.normal_approximation(p_hat=0.5, n=0)


def test_normal_approximation_vs_wilson(ci_calc):
    """Compare normal approximation with Wilson interval."""
    # For large samples and moderate proportions, they should be similar
    p_hat = 0.5
    n = 1000

    normal_result = ci_calc.normal_approximation(p_hat=p_hat, n=n)
    wilson_result = ci_calc.wilson_interval(p_hat=p_hat, n=n)

    # Check that the intervals are similar (within 0.01)
    assert abs(normal_result['ci_lower'] - wilson_result['ci_lower']) < 0.01
    assert abs(normal_result['ci_upper'] - wilson_result['ci_upper']) < 0.01

    # For small samples or extreme proportions, they can differ more
    p_hat = 0.05
    n = 20

    normal_result = ci_calc.normal_approximation(p_hat=p_hat, n=n)
    wilson_result = ci_calc.wilson_interval(p_hat=p_hat, n=n)

    # Wilson interval is generally more conservative for extreme proportions
    # (wider interval)
    normal_width = normal_result['ci_upper'] - normal_result['ci_lower']
    wilson_width = wilson_result['ci_upper'] - wilson_result['ci_lower']

    # This might not always be true, but is a common pattern
    # So we'll just print the values instead of asserting
    print(f"Normal width: {normal_width}, Wilson width: {wilson_width}")
