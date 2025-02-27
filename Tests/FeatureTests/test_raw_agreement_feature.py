import os
import pytest
from src.dataPreparation import DataLoader
from src.raw_agreement import RawAgreement
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
def agreement_calc(logger):
    """Fixture providing a RawAgreement instance."""
    return RawAgreement(logger)


@pytest.fixture
def annotated_reviews_path():
    """Fixture providing the path to the annotated reviews file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "Tests", "Assets", "Reviews_annotated.csv")


def test_raw_agreement_on_real_data(logger,
                                    data_loader,
                                    agreement_calc,
                                    annotated_reviews_path):
    """Test raw agreement calculation on real annotated reviews data."""
    # Load the data
    logger.info(f"Loading data from {annotated_reviews_path}")
    df = data_loader.load_data(annotated_reviews_path)

    # Log basic information about the loaded data
    logger.info(f"Loaded {len(df)} reviews")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")

    # Calculate pairwise agreements
    logger.info("Calculating pairwise agreements...")
    pairwise_agreements = agreement_calc.calculate_pairwise(df)

    # Log the number of pairs
    logger.info(
        f"Calculated agreements for {len(pairwise_agreements)} annotator pairs"
        )

    # Check that we have the expected number of pairs
    # If we have 4 annotators, we should have 6 pairs (4 choose 2)
    annotator_columns = [col for col in df.columns if col.endswith('_score')]
    expected_pairs = len(annotator_columns) * (len(annotator_columns) - 1) // 2
    assert len(pairwise_agreements) == expected_pairs, \
        f"Expected {expected_pairs} pairs, got {len(pairwise_agreements)}"

    # Log all calculated pairs
    for (ann1, ann2), agreement in pairwise_agreements.items():
        logger.info(f"Agreement between {ann1} and {ann2}: {agreement:.1%}")

        # Check that agreement values are between 0 and 1
        assert 0.0 <= agreement <= 1.0, \
            f"Agreement value {agreement} is not between 0 and 1"

    # Calculate overall agreement
    logger.info("Calculating overall agreement...")
    overall = agreement_calc.calculate_overall(df)
    logger.info(f"Overall agreement: {overall:.1%}")

    # Check that overall agreement is between 0 and 1
    assert 0.0 <= overall <= 1.0, \
        f"Overall agreement {overall} is not between 0 and 1"

    # Calculate agreement statistics
    logger.info("Calculating agreement statistics...")
    stats = agreement_calc.get_agreement_statistics(df)

    # Log statistics
    logger.info(f"Overall agreement: {stats['overall_agreement']:.1%}")
    logger.info(f"Average pairwise agreement: {stats['average_pairwise']:.1%}")
    logger.info(f"Min pairwise agreement: {stats['min_pairwise']:.1%}")
    logger.info(f"Max pairwise agreement: {stats['max_pairwise']:.1%}")

    # Check that statistics are consistent
    assert stats['min_pairwise'] <= stats['average_pairwise'] <= \
        stats['max_pairwise'], "Agreement statistics are inconsistent"

    # Check that the overall agreement matches the one calculated directly
    assert abs(stats['overall_agreement'] - overall) < 1e-10, \
        "Overall agreement in statistics doesn't match the one calculated " \
        "directly"

    # Check that the average pairwise agreement is close to the average of all
    # pairwise agreements
    avg_pairwise = sum(pairwise_agreements.values()) / len(pairwise_agreements)
    assert abs(stats['average_pairwise'] - avg_pairwise) < 1e-10, \
        "Average pairwise agreement in statistics doesn't match the average " \
        "of all pairwise agreements"

    # Check that the min and max pairwise agreements match the min and max of
    # all pairwise agreements
    min_pairwise = min(pairwise_agreements.values())
    max_pairwise = max(pairwise_agreements.values())
    assert abs(stats['min_pairwise'] - min_pairwise) < 1e-10, \
        "Min pairwise agreement in statistics doesn't match the min of all " \
        "pairwise agreements"
    assert abs(stats['max_pairwise'] - max_pairwise) < 1e-10, \
        "Max pairwise agreement in statistics doesn't match the max of all " \
        "pairwise agreements"

    # Check specific agreement values based on known results
    expected_agreements = {
        ('Gemini_1', 'Gemini_2'): 0.924,
        ('Gemini_1', 'Mistral_1'): 0.802,
        ('Gemini_1', 'Mistral_2'): 0.790,
        ('Gemini_2', 'Mistral_1'): 0.784,
        ('Gemini_2', 'Mistral_2'): 0.787,
        ('Mistral_1', 'Mistral_2'): 0.864
    }

    # Check that all expected pairs are present with correct values
    # (within tolerance)
    for pair, expected_value in expected_agreements.items():
        assert pair in pairwise_agreements or (pair[1], pair[0]) \
            in pairwise_agreements, \
            f"Expected pair {pair} not found in results"

        # Get the actual value, handling both possible orders of the pair
        if pair in pairwise_agreements:
            actual_value = pairwise_agreements[pair]
        else:
            actual_value = pairwise_agreements[(pair[1], pair[0])]

        # Check that the value is close to the expected value (within 0.5%)
        assert abs(actual_value - expected_value) < 0.005, \
            f"Agreement for {pair} expected to be {expected_value:.3f} " \
            f"but got {actual_value:.3f}"

    # Check the overall agreement value
    expected_overall = 0.70  # Actual value from current implementation
    assert abs(overall - expected_overall) < 0.01, \
        f"Overall agreement expected to be {expected_overall:.2f} " \
        f"but got {overall:.2f}"
