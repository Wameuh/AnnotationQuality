import pytest
import numpy as np
from Utils.confident_interval import ConfidenceIntervalCalculator


@pytest.fixture
def calculator():
    """Fixture providing a ConfidenceIntervalCalculator instance."""
    return ConfidenceIntervalCalculator(confidence=0.95)


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
