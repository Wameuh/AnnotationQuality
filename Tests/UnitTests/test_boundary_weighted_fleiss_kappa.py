import pytest
import numpy as np
import pandas as pd
from src.boundary_weighted_fleiss_kappa import BoundaryWeightedFleissKappa
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
def bwfk_calculator():
    """Fixture providing a BoundaryWeightedFleissKappa instance."""
    return BoundaryWeightedFleissKappa(level=LogLevel.DEBUG)


@pytest.fixture
def perfect_agreement_segmentations():
    """Fixture providing segmentations with perfect agreement."""
    # Create 3 identical 10x10 segmentations
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[3:7, 3:7] = 1  # 4x4 square in the middle

    return [seg.copy() for _ in range(3)]


@pytest.fixture
def high_agreement_segmentations():
    """
    Fixture providing segmentations with high agreement (boundary differences).
    """
    # Create base segmentation
    base_seg = np.zeros((10, 10), dtype=np.int32)
    base_seg[3:7, 3:7] = 1  # 4x4 square in the middle

    # Create variations with small differences at boundaries
    seg1 = base_seg.copy()

    seg2 = base_seg.copy()
    seg2[3, 3:5] = 0  # Remove top-left corner

    seg3 = base_seg.copy()
    seg3[6, 5:7] = 0  # Remove bottom-right corner

    return [seg1, seg2, seg3]


@pytest.fixture
def low_agreement_segmentations():
    """Fixture providing segmentations with low agreement."""
    # Create different segmentations
    seg1 = np.zeros((10, 10), dtype=np.int32)
    seg1[3:7, 3:7] = 1  # 4x4 square in the middle

    seg2 = np.zeros((10, 10), dtype=np.int32)
    seg2[1:5, 1:5] = 1  # 4x4 square in top-left

    seg3 = np.zeros((10, 10), dtype=np.int32)
    seg3[5:9, 5:9] = 1  # 4x4 square in bottom-right

    return [seg1, seg2, seg3]


def test_get_boundary_mask(bwfk_calculator):
    """Test _get_boundary_mask method."""
    # Create a simple segmentation
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[3:7, 3:7] = 1  # 4x4 square in the middle

    # Get boundary with width 1
    boundary = bwfk_calculator._get_boundary_mask(seg, width=1)

    # Check that boundary is a binary mask
    assert np.all(np.isin(boundary, [0, 1]))

    # Check that boundary surrounds the square
    # The boundary should be around the 4x4 square
    assert np.sum(boundary) > 0

    # Check that boundary width increases with parameter
    boundary2 = bwfk_calculator._get_boundary_mask(seg, width=2)
    assert np.sum(boundary2) > np.sum(boundary)


def test_calculate_weighted_observed_agreement(bwfk_calculator,
                                               high_agreement_segmentations):
    """Test _calculate_weighted_observed_agreement method."""
    # Create uniform weights
    weights = np.ones((10, 10))

    # Calculate agreement
    agreement = bwfk_calculator._calculate_weighted_observed_agreement(
        high_agreement_segmentations, weights)

    # Check that agreement is a float between 0 and 1
    assert isinstance(agreement, float)
    assert 0.0 <= agreement <= 1.0

    # With perfect agreement, should be 1.0
    perfect_agreement = bwfk_calculator._calculate_weighted_observed_agreement(
        [high_agreement_segmentations[0]] * 3, weights)
    assert perfect_agreement == 1.0

    # With different weights, agreement should change
    boundary_weights = np.ones((10, 10))
    boundary_weights[3:7, 3:7] = 0.5  # Lower weight in the middle

    weighted_agreement = \
        bwfk_calculator._calculate_weighted_observed_agreement(
            high_agreement_segmentations, boundary_weights)

    # Agreement should be different with different weights
    assert weighted_agreement != agreement


def test_calculate_weighted_chance_agreement(bwfk_calculator,
                                             high_agreement_segmentations):
    """Test _calculate_weighted_chance_agreement method."""
    # Create uniform weights
    weights = np.ones((10, 10))

    # Calculate chance agreement
    chance = bwfk_calculator._calculate_weighted_chance_agreement(
        high_agreement_segmentations, weights)

    # Check that chance is a float between 0 and 1
    assert isinstance(chance, float)
    assert 0.0 <= chance <= 1.0

    # With all 0s or all 1s, chance should be 1.0
    all_zeros = [np.zeros((10, 10)) for _ in range(3)]
    all_ones = [np.ones((10, 10)) for _ in range(3)]

    chance_zeros = bwfk_calculator._calculate_weighted_chance_agreement(
        all_zeros, weights)
    chance_ones = bwfk_calculator._calculate_weighted_chance_agreement(
        all_ones, weights)

    assert chance_zeros == 1.0
    assert chance_ones == 1.0


def test_calculate_perfect_agreement(bwfk_calculator,
                                     perfect_agreement_segmentations):
    """Test calculate method with perfect agreement."""
    # Calculate BWFK
    bwfk = bwfk_calculator.calculate(perfect_agreement_segmentations)

    # With perfect agreement, BWFK should be 1.0
    assert bwfk == 1.0

    # Test with different parameters
    assert bwfk_calculator.calculate(perfect_agreement_segmentations,
                                     boundary_width=3,
                                     weight_factor=0.25) == 1.0


def test_calculate_high_agreement(bwfk_calculator,
                                  high_agreement_segmentations):
    """Test calculate method with high agreement (boundary differences)."""
    # Calculate BWFK with default parameters
    bwfk_default = bwfk_calculator.calculate(high_agreement_segmentations)

    # BWFK should be high but less than 1.0
    assert 0.8 <= bwfk_default < 1.0

    # Calculate with lower weight factor
    # (reduces impact of boundary disagreements)
    bwfk_low_weight = bwfk_calculator.calculate(
        high_agreement_segmentations, weight_factor=0.25)

    # BWFK should be higher with lower weight factor
    assert bwfk_low_weight > bwfk_default


def test_calculate_low_agreement(bwfk_calculator, low_agreement_segmentations):
    """Test calculate method with low agreement."""
    # Calculate BWFK
    bwfk = bwfk_calculator.calculate(low_agreement_segmentations)

    # BWFK should be low
    assert bwfk < 0.5


def test_calculate_from_dataframe(bwfk_calculator,
                                  high_agreement_segmentations):
    """Test calculate_from_dataframe method."""
    # Convert segmentations to DataFrame
    flat_data = {}
    for i, seg in enumerate(high_agreement_segmentations):
        flat_data[f'Annotator{i+1}'] = seg.flatten()

    df = pd.DataFrame(flat_data)

    # Calculate BWFK from DataFrame
    bwfk = bwfk_calculator.calculate_from_dataframe(df, (10, 10))

    # Compare with direct calculation
    direct_bwfk = bwfk_calculator.calculate(high_agreement_segmentations)

    assert bwfk == direct_bwfk


def test_interpret_bwfk(bwfk_calculator):
    """Test interpret_bwfk method."""
    # Test different ranges
    assert "Poor agreement" in bwfk_calculator.interpret_bwfk(-0.1)
    assert "Slight agreement" in bwfk_calculator.interpret_bwfk(0.1)
    assert "Fair agreement" in bwfk_calculator.interpret_bwfk(0.3)
    assert "Moderate agreement" in bwfk_calculator.interpret_bwfk(0.5)
    assert "Substantial agreement" in bwfk_calculator.interpret_bwfk(0.7)
    assert "Almost perfect agreement" in bwfk_calculator.interpret_bwfk(0.9)


def test_get_bwfk_statistics(bwfk_calculator, high_agreement_segmentations):
    """Test get_bwfk_statistics method."""
    # Get statistics
    stats = bwfk_calculator.get_bwfk_statistics(
        high_agreement_segmentations,
        boundary_widths=[1, 2],
        weight_factors=[0.5, 0.75]
    )

    # Check that standard BWFK is included
    assert 'bwfk_standard' in stats
    assert 'interpretation_standard' in stats

    # Check that width variations are included
    assert 'bwfk_width_1' in stats
    assert 'bwfk_width_2' in stats

    # Check that factor variations are included
    assert 'bwfk_factor_0.5' in stats
    assert 'bwfk_factor_0.75' in stats

    # Check that all values are in expected range
    for key, value in stats.items():
        if key.startswith('bwfk_'):
            assert -1.0 <= value <= 1.0


def test_calculate_with_invalid_inputs(bwfk_calculator):
    """Test calculate method with invalid inputs."""
    # Empty list
    with pytest.raises(ValueError):
        bwfk_calculator.calculate([])

    # Non-numpy arrays
    with pytest.raises(ValueError):
        bwfk_calculator.calculate([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])

    # Different shapes
    seg1 = np.zeros((10, 10))
    seg2 = np.zeros((8, 8))
    with pytest.raises(ValueError):
        bwfk_calculator.calculate([seg1, seg2])

    # Non-binary values (should convert automatically)
    seg1 = np.zeros((5, 5))
    seg1[1:4, 1:4] = 1

    seg2 = np.zeros((5, 5))
    seg2[1:4, 1:4] = 2  # Non-binary values

    # Should not raise error, but log a warning
    bwfk = bwfk_calculator.calculate([seg1, seg2])
    assert -1.0 <= bwfk <= 1.0


def test_calculate_with_single_annotator(bwfk_calculator):
    """Test calculate method with only one annotator."""
    seg = np.zeros((5, 5))
    seg[1:4, 1:4] = 1

    # Should raise error with only one annotator
    with pytest.raises(ValueError):
        bwfk_calculator._calculate_weighted_observed_agreement([seg],
                                                               np.ones((5, 5)))


def test_calculate_with_chance_agreement_one(bwfk_calculator):
    """Test calculate method when chance agreement is 1.0."""
    # Create segmentations where all annotators use the same value for all
    # pixels
    # This will result in chance_agreement = 1.0
    seg1 = np.ones((5, 5))  # All ones
    seg2 = np.ones((5, 5))  # All ones
    seg3 = np.ones((5, 5))  # All ones

    # Calculate BWFK
    bwfk = bwfk_calculator.calculate([seg1, seg2, seg3])

    # When chance agreement is 1.0, BWFK should be 0.0
    assert bwfk == 0.0

    # Test with all zeros as well
    seg1 = np.zeros((5, 5))  # All zeros
    seg2 = np.zeros((5, 5))  # All zeros
    seg3 = np.zeros((5, 5))  # All zeros

    # Calculate BWFK
    bwfk = bwfk_calculator.calculate([seg1, seg2, seg3])

    # When chance agreement is 1.0, BWFK should be 0.0
    assert bwfk == 0.0


def test_calculate_pairwise(bwfk_calculator, perfect_agreement_segmentations):
    """Test calculate_pairwise method with perfect agreement."""
    # Create a DataFrame from the segmentations
    df = pd.DataFrame({
        'Annotator1': [perfect_agreement_segmentations[0]],
        'Annotator2': [perfect_agreement_segmentations[1]],
        'Annotator3': [perfect_agreement_segmentations[2]]
    })

    # Calculate pairwise BWFKs
    pairwise_bwfks = bwfk_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_bwfks, dict)

    # Check that all pairs are present
    columns = df.columns
    n_annotators = len(columns)
    expected_pairs_count = (n_annotators * (n_annotators - 1)) // 2
    assert len(pairwise_bwfks) == expected_pairs_count

    # Check that values are between -1 and 1
    for k, v in pairwise_bwfks.items():
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert -1.0 <= v <= 1.0

    # With perfect agreement, BWFK should be 1.0
    for v in pairwise_bwfks.values():
        assert v == 1.0


def test_calculate_pairwise_with_boundary_differences(
        bwfk_calculator, high_agreement_segmentations):
    """Test calculate_pairwise method with boundary differences."""
    # Create a DataFrame from the segmentations
    df = pd.DataFrame({
        'Annotator1': [high_agreement_segmentations[0]],
        'Annotator2': [high_agreement_segmentations[1]],
        'Annotator3': [high_agreement_segmentations[2]]
    })

    # Calculate pairwise BWFKs
    pairwise_bwfks = bwfk_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_bwfks, dict)

    # Check that values are between -1 and 1
    for v in pairwise_bwfks.values():
        assert -1.0 <= v <= 1.0

    # With high agreement (but not perfect), BWFK should be positive but less '
    # than 1.0
    for v in pairwise_bwfks.values():
        assert 0.0 < v < 1.0


def test_calculate_pairwise_with_missing_data(bwfk_calculator):
    """Test calculate_pairwise method with missing data."""
    # Create a DataFrame with missing values
    data = {
        'Annotator1': [np.zeros((5, 5))],
        'Annotator2': [np.ones((5, 5))],
        'Annotator3': [None]  # Missing data
    }
    df = pd.DataFrame(data)

    # Calculate pairwise BWFKs
    pairwise_bwfks = bwfk_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_bwfks, dict)

    # Check that pairs with valid data are present
    assert ('Annotator1', 'Annotator2') in pairwise_bwfks

    # Check that pairs with missing data are not present
    assert ('Annotator1', 'Annotator3') not in pairwise_bwfks
    assert ('Annotator2', 'Annotator3') not in pairwise_bwfks


def test_calculate_pairwise_with_non_binary_masks(bwfk_calculator):
    """Test calculate_pairwise method with non-binary masks."""
    # Create a DataFrame with non-binary masks
    data = {
        'Annotator1': [np.zeros((5, 5))],  # Binary mask (all 0s)
        'Annotator2': [np.ones((5, 5)) * 2]  # Non-binary mask (all 2s)
    }
    df = pd.DataFrame(data)

    # Calculate pairwise BWFKs
    pairwise_bwfks = bwfk_calculator.calculate_pairwise(df)

    # Check that result is a dictionary
    assert isinstance(pairwise_bwfks, dict)

    # Check that no pairs are present (all were skipped due to non-binary
    # masks)
    assert len(pairwise_bwfks) == 0


def test_calculate_pairwise_with_exception(bwfk_calculator, monkeypatch):
    """Test calculate_pairwise method when calculate raises an exception."""
    # Create a DataFrame with valid data
    data = {
        'Annotator1': [np.zeros((5, 5))],
        'Annotator2': [np.ones((5, 5))]
    }
    df = pd.DataFrame(data)

    # Define a mock calculate method that raises an exception
    def mock_calculate(*args, **kwargs):
        raise ValueError("Forced error in calculate method")

    # Replace the calculate method with our mock
    monkeypatch.setattr(bwfk_calculator, "calculate", mock_calculate)

    # Calculate pairwise BWFKs
    pairwise_bwfks = bwfk_calculator.calculate_pairwise(df)

    # Check that result is an empty dictionary (no successful calculations)
    assert isinstance(pairwise_bwfks, dict)
    assert len(pairwise_bwfks) == 0
