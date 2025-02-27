import os
import pytest
import numpy as np
from src.dataPreparation import DataLoader
from src.cohen_kappa import CohenKappa
from Utils.logger import Logger, LogLevel


@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.INFO)


@pytest.fixture
def data_loader(logger):
    """Fixture providing a DataLoader instance."""
    return DataLoader(logger)


@pytest.fixture
def kappa_calc(logger):
    """Fixture providing a CohenKappa instance."""
    return CohenKappa(logger)


@pytest.fixture
def annotated_reviews_path():
    """Fixture providing the path to the annotated reviews file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "Tests", "Assets", "Reviews_annotated.csv")


def test_cohen_kappa_on_real_data(logger,
                                  data_loader,
                                  kappa_calc,
                                  annotated_reviews_path):
    """Test Cohen's Kappa calculation on real annotated reviews data."""
    # Load the data
    logger.info(f"Loading data from {annotated_reviews_path}")
    df = data_loader.load_data(annotated_reviews_path)
    logger.info(f"Loaded {len(df)} reviews")

    # Calculate pairwise kappas
    logger.info("Calculating pairwise Cohen's Kappa...")
    pairwise_kappas = kappa_calc.calculate_pairwise(df)

    # Check that we have the expected number of pairs
    # With 4 annotators, we should have 6 pairs (n*(n-1)/2)
    assert len(pairwise_kappas) == 6, \
        f"Expected 6 annotator pairs, got {len(pairwise_kappas)}"

    # Get kappa statistics
    logger.info("Calculating kappa statistics...")
    stats = kappa_calc.get_kappa_statistics(df)

    # Check that all expected statistics are present
    assert 'average_kappa' in stats, "Missing 'average_kappa' in statistics"
    assert 'min_kappa' in stats, "Missing 'min_kappa' in statistics"
    assert 'max_kappa' in stats, "Missing 'max_kappa' in statistics"

    # Check that statistics are consistent with pairwise values
    kappa_values = list(pairwise_kappas.values())
    assert abs(stats['average_kappa'] - np.mean(kappa_values)) < 1e-10, \
        "Average kappa in statistics doesn't match the mean of all " \
        "pairwise kappas"
    assert abs(stats['min_kappa'] - min(kappa_values)) < 1e-10, \
        "Min kappa in statistics doesn't match the min of all " \
        "pairwise kappas"
    assert abs(stats['max_kappa'] - max(kappa_values)) < 1e-10, \
        "Max kappa in statistics doesn't match the max of all " \
        "pairwise kappas"

    # Check specific kappa values based on the actual results from test.py
    expected_kappas = {
        ('Gemini_1', 'Gemini_2'): 0.885,
        ('Gemini_1', 'Mistral_1'): 0.689,
        ('Gemini_1', 'Mistral_2'): 0.677,
        ('Gemini_2', 'Mistral_1'): 0.668,
        ('Gemini_2', 'Mistral_2'): 0.678,
        ('Mistral_1', 'Mistral_2'): 0.786
    }

    # Check that all expected pairs are present with correct values
    # (within tolerance)
    for pair, expected_value in expected_kappas.items():
        assert pair in pairwise_kappas or (pair[1], pair[0]) \
            in pairwise_kappas, \
            f"Expected pair {pair} not found in results"

        # Get the actual value, handling both possible orders of the pair
        if pair in pairwise_kappas:
            actual_value = pairwise_kappas[pair]
        else:
            actual_value = pairwise_kappas[(pair[1], pair[0])]

        # Check that the value is close to the expected value (within 1%)
        assert abs(actual_value - expected_value) < 0.01, \
            f"Kappa for {pair} expected to be {expected_value:.3f} " \
            f"but got {actual_value:.3f}"

    # Test interpretation of kappa values
    # Based on the output from test.py, all pairs should have
    #  "Substantial agreement"
    # except for Gemini_1-Gemini_2 which should have "Almost perfect agreement"
    for pair, kappa_val in pairwise_kappas.items():
        interpretation = kappa_calc.interpret_kappa(kappa_val)
        logger.info(f"Kappa for {pair}: {kappa_val:.3f} ({interpretation})")

        if (
                pair == ('Gemini_1', 'Gemini_2') or
                pair == ('Gemini_2', 'Gemini_1')
                ):
            assert "Almost perfect" in interpretation, \
                f"Expected 'Almost perfect agreement' for {pair} " \
                f"with kappa {kappa_val:.3f}, got '{interpretation}'"
        else:
            assert "Substantial" in interpretation, \
                f"Expected 'Substantial agreement' for {pair} " \
                f"with kappa {kappa_val:.3f}, got '{interpretation}'"


def test_cohen_kappa_vs_raw_agreement(logger,
                                      data_loader,
                                      kappa_calc,
                                      annotated_reviews_path):
    """Test relationship between Cohen's Kappa and raw agreement on real """ \
        """data."""
    # Load the data
    df = data_loader.load_data(annotated_reviews_path)

    # Calculate Cohen's Kappa for all pairs
    kappas = kappa_calc.calculate_pairwise(df)

    # Calculate raw agreement for all pairs
    raw_agreements = {}
    for pair in kappas.keys():
        ann1, ann2 = pair
        col1 = f"{ann1}_score" if f"{ann1}_score" in df.columns else ann1
        col2 = f"{ann2}_score" if f"{ann2}_score" in df.columns else ann2

        # Get complete reviews for this pair
        pair_df = df[[col1, col2]].dropna()

        # Calculate raw agreement
        raw_agreement = (pair_df[col1] == pair_df[col2]).mean()
        raw_agreements[pair] = raw_agreement

    # Check that kappa is always <= raw agreement
    # This is because kappa corrects for chance agreement
    for pair, kappa_val in kappas.items():
        raw_agreement = raw_agreements[pair]

        logger.info(f"Pair {pair}: Kappa = {kappa_val:.3f}, "
                    f"Raw Agreement = {raw_agreement:.3f}")

        # Kappa should always be less than or equal to raw agreement
        # (except in rare cases with negative kappa)
        if kappa_val >= 0:
            assert kappa_val <= raw_agreement + 1e-10, \
                f"Kappa ({kappa_val}) should be <= raw agreement" \
                f" ({raw_agreement})"

    # Check specific pairs based on the output from test.py
    # Gemini_1-Gemini_2 should have the highest kappa and raw agreement
    gemini_pair = ('Gemini_1', 'Gemini_2')
    if gemini_pair not in kappas:
        gemini_pair = ('Gemini_2', 'Gemini_1')

    # Mistral_1-Mistral_2 should have the second highest kappa
    mistral_pair = ('Mistral_1', 'Mistral_2')
    if mistral_pair not in kappas:
        mistral_pair = ('Mistral_2', 'Mistral_1')

    # Check that Gemini pair has higher kappa than Mistral pair
    assert kappas[gemini_pair] > kappas[mistral_pair], \
        f"Expected Gemini pair to have higher kappa than Mistral pair, " \
        f"but got {kappas[gemini_pair]:.3f} vs {kappas[mistral_pair]:.3f}"

    # Check that the difference between kappa and raw agreement is reasonable
    # For each pair, calculate the difference
    for pair, kappa_val in kappas.items():
        raw_agreement = raw_agreements[pair]
        difference = raw_agreement - kappa_val

        # The difference should be positive (raw agreement >= kappa)
        assert difference >= -1e-10, \
            f"Raw agreement should be >= kappa, but got " \
            f"raw={raw_agreement:.3f}, kappa={kappa_val:.3f}"

        # Log the difference for each pair
        logger.info(
            f"Pair {pair}: Difference (Raw - Kappa) = {difference:.3f}")
