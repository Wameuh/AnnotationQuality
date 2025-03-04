import pytest
import numpy as np
import pandas as pd
from src.iou_agreement import IoUAgreement
from Utils.logger import Logger, LogLevel


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the logger singleton before each test."""
    # Reset the singleton instance
    Logger._instance = None
    yield
    # Clean up after test
    Logger._instance = None


@pytest.fixture
def iou_calculator():
    """Fixture providing an IoUAgreement instance."""
    return IoUAgreement(level=LogLevel.DEBUG)


@pytest.fixture
def perfect_agreement_masks():
    """Fixture providing masks with perfect agreement."""
    # Create 3 identical masks
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True  # A square in the middle

    return [mask.copy() for _ in range(3)]


@pytest.fixture
def high_agreement_masks():
    """Fixture providing masks with high agreement."""
    # Create base mask
    base_mask = np.zeros((10, 10), dtype=bool)
    base_mask[2:5, 2:5] = True  # A square in the middle

    # Create variations with small differences
    mask1 = base_mask.copy()

    mask2 = base_mask.copy()
    mask2[5, 2:5] = True  # Add a row

    mask3 = base_mask.copy()
    mask3[2:5, 5] = True  # Add a column

    return [mask1, mask2, mask3]


@pytest.fixture
def low_agreement_masks():
    """Fixture providing masks with low agreement."""
    # Create different masks
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[2:5, 2:5] = True  # A square in the middle

    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[6:9, 6:9] = True  # A square in the bottom right

    mask3 = np.zeros((10, 10), dtype=bool)
    mask3[0:3, 0:3] = True  # A square in the top left

    return [mask1, mask2, mask3]


def test_calculate_pairwise_iou(iou_calculator):
    """Test calculate_pairwise_iou method."""
    # Create two masks
    mask1 = np.zeros((5, 5), dtype=bool)
    mask1[1:4, 1:4] = True  # 3x3 square = 9 pixels

    mask2 = np.zeros((5, 5), dtype=bool)
    mask2[2:5, 2:5] = True  # 3x3 square = 9 pixels

    # Calculate IoU
    iou = iou_calculator.calculate_pairwise_iou(mask1, mask2)

    # Expected: intersection = 4 pixels, union = 14 pixels, IoU = 4/14 = 0.2857
    assert iou == pytest.approx(4/14, abs=0.001)


def test_calculate_pairwise_iou_perfect_agreement(iou_calculator):
    """Test calculate_pairwise_iou method with perfect agreement."""
    # Create two identical masks
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    # Calculate IoU
    iou = iou_calculator.calculate_pairwise_iou(mask, mask)

    # Expected: IoU = 1.0 (perfect agreement)
    assert iou == 1.0


def test_calculate_pairwise_iou_no_agreement(iou_calculator):
    """Test calculate_pairwise_iou method with no agreement."""
    # Create two non-overlapping masks
    mask1 = np.zeros((5, 5), dtype=bool)
    mask1[0:2, 0:2] = True

    mask2 = np.zeros((5, 5), dtype=bool)
    mask2[3:5, 3:5] = True

    # Calculate IoU
    iou = iou_calculator.calculate_pairwise_iou(mask1, mask2)

    # Expected: IoU = 0.0 (no agreement)
    assert iou == 0.0


def test_calculate_pairwise_iou_empty_masks(iou_calculator):
    """Test calculate_pairwise_iou method with empty masks."""
    # Create two empty masks
    mask1 = np.zeros((5, 5), dtype=bool)
    mask2 = np.zeros((5, 5), dtype=bool)

    # Calculate IoU
    iou = iou_calculator.calculate_pairwise_iou(mask1, mask2)

    # Expected: IoU = 1.0 (perfect agreement on empty masks)
    assert iou == 1.0


def test_calculate_pairwise_iou_one_empty_mask(iou_calculator):
    """Test calculate_pairwise_iou method with one empty mask."""
    # Create one empty and one non-empty mask
    mask1 = np.zeros((5, 5), dtype=bool)

    mask2 = np.zeros((5, 5), dtype=bool)
    mask2[1:4, 1:4] = True

    # Calculate IoU
    iou = iou_calculator.calculate_pairwise_iou(mask1, mask2)

    # Expected: IoU = 0.0 (no agreement when one mask is empty)
    assert iou == 0.0


def test_calculate_mean_iou(iou_calculator, high_agreement_masks):
    """Test calculate_mean_iou method."""
    # Calculate mean IoU
    mean_iou = iou_calculator.calculate_mean_iou(high_agreement_masks)

    # Should be between 0 and 1
    assert 0 <= mean_iou <= 1


def test_calculate_mean_iou_perfect_agreement(iou_calculator,
                                              perfect_agreement_masks):
    """Test calculate_mean_iou method with perfect agreement."""
    # Calculate mean IoU
    mean_iou = iou_calculator.calculate_mean_iou(perfect_agreement_masks)

    # Should be 1.0 (perfect agreement)
    assert mean_iou == 1.0


def test_calculate_mean_iou_low_agreement(iou_calculator, low_agreement_masks):
    """Test calculate_mean_iou method with low agreement."""
    # Calculate mean IoU
    mean_iou = iou_calculator.calculate_mean_iou(low_agreement_masks)

    # Should be low (close to 0)
    assert mean_iou < 0.2


def test_calculate_mean_iou_invalid_inputs(iou_calculator):
    """Test calculate_mean_iou method with invalid inputs."""
    # Empty list
    with pytest.raises(ValueError):
        iou_calculator.calculate_mean_iou([])

    # Only one mask
    with pytest.raises(ValueError):
        iou_calculator.calculate_mean_iou([np.zeros((5, 5), dtype=bool)])

    # Different shapes
    with pytest.raises(ValueError):
        iou_calculator.calculate_mean_iou([
            np.zeros((5, 5), dtype=bool),
            np.zeros((6, 6), dtype=bool)
        ])

    # Non-numpy arrays
    with pytest.raises(ValueError):
        iou_calculator.calculate_mean_iou([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])


def test_calculate_from_dataframe(iou_calculator):
    """Test calculate_from_dataframe method."""
    # Create a DataFrame with binary values
    data = {
        'Annotator1': [0, 0, 1, 1, 0],
        'Annotator2': [0, 1, 1, 0, 0],
        'Annotator3': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    # Calculate IoU
    iou = iou_calculator.calculate_from_dataframe(df)

    # Should be between 0 and 1
    assert 0 <= iou <= 1


def test_calculate_from_dataframe_non_binary(iou_calculator):
    """Test calculate_from_dataframe method with non-binary values."""
    # Create a DataFrame with non-binary values
    data = {
        'Annotator1': [0, 2, 1, 3, 0],
        'Annotator2': [0, 1, 2, 0, 3]
    }
    df = pd.DataFrame(data)

    # Calculate IoU (should log a warning but still calculate)
    iou = iou_calculator.calculate_from_dataframe(df)

    # Should be between 0 and 1
    assert 0 <= iou <= 1


def test_calculate_from_dataframe_insufficient_annotators(iou_calculator):
    """Test calculate_from_dataframe method with insufficient annotators."""
    # Create a DataFrame with only one annotator
    data = {
        'Annotator1': [0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

    # Calculate IoU
    iou = iou_calculator.calculate_from_dataframe(df)

    # Should return 0.0 when there are not enough annotators
    assert iou == 0.0


def test_interpret_iou(iou_calculator):
    """Test interpret_iou method."""
    # Test different ranges
    assert iou_calculator.interpret_iou(0.1) == "Poor agreement"
    assert iou_calculator.interpret_iou(0.3) == "Fair agreement"
    assert iou_calculator.interpret_iou(0.5) == "Moderate agreement"
    assert iou_calculator.interpret_iou(0.7) == "Substantial agreement"
    assert iou_calculator.interpret_iou(0.9) == "Almost perfect agreement"


def test_get_iou_statistics(iou_calculator, high_agreement_masks):
    """Test get_iou_statistics method."""
    # Calculate statistics
    stats = iou_calculator.get_iou_statistics(high_agreement_masks)

    # Check that all expected keys are present
    assert 'mean_iou' in stats
    assert 'interpretation' in stats
    assert 'min_iou' in stats
    assert 'max_iou' in stats

    # Check pairwise IoUs
    assert 'iou_1_2' in stats
    assert 'iou_1_3' in stats
    assert 'iou_2_3' in stats

    # Check that values are in expected range
    assert 0 <= stats['mean_iou'] <= 1
    assert 0 <= stats['min_iou'] <= 1
    assert 0 <= stats['max_iou'] <= 1
    assert isinstance(stats['interpretation'], str)


def test_calculate_pairwise_iou_different_shapes(iou_calculator):
    """Test calculate_pairwise_iou method with masks of different shapes."""
    # Create two masks with different shapes
    mask1 = np.zeros((5, 5), dtype=bool)
    mask2 = np.zeros((6, 6), dtype=bool)

    # Calculate IoU should raise ValueError
    with pytest.raises(ValueError, match="Masks must have the same shape"):
        iou_calculator.calculate_pairwise_iou(mask1, mask2)


def test_iou_agreement_init_with_custom_level():
    """Test IoUAgreement initialization with custom log level."""
    # Create IoUAgreement with custom log level
    iou_calc = IoUAgreement(level=LogLevel.DEBUG)

    # Check that logger has correct level
    assert iou_calc.logger.level == LogLevel.DEBUG
