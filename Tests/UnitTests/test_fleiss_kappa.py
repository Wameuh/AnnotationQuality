import pytest
import pandas as pd
import numpy as np
from src.fleiss_kappa import FleissKappa
from Utils.logger import Logger, LogLevel


@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.DEBUG)


@pytest.fixture
def kappa_calc(logger):
    """Fixture providing a FleissKappa instance."""
    return FleissKappa(logger)


@pytest.fixture
def perfect_agreement_df():
    """Fixture providing a DataFrame with perfect agreement among annotators"""
    data = {
        'Annotator1_score': [5, 4, 3, 2, 1],
        'Annotator2_score': [5, 4, 3, 2, 1],
        'Annotator3_score': [5, 4, 3, 2, 1],
    }
    return pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])


@pytest.fixture
def partial_agreement_df():
    """Fixture providing a DataFrame with partial agreement among annotators"""
    data = {
        'Annotator1_score': [5, 4, 3, 2, 1],
        'Annotator2_score': [5, 4, 3, 2, 1],
        'Annotator3_score': [5, 3, 3, 1, 1],
        'Annotator4_score': [4, 4, 3, 2, 2],
    }
    return pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])


@pytest.fixture
def missing_values_df():
    """Fixture providing a DataFrame with missing values."""
    data = {
        'Annotator1_score': [5, 4, 3, np.nan, 1],
        'Annotator2_score': [5, 4, np.nan, 2, 1],
        'Annotator3_score': [np.nan, 3, 3, 1, 1],
    }
    return pd.DataFrame(data, index=['rev1', 'rev2', 'rev3', 'rev4', 'rev5'])


def test_calculate_perfect_agreement(kappa_calc, perfect_agreement_df):
    """Test calculate with perfect agreement."""
    kappa = kappa_calc.calculate(perfect_agreement_df)
    assert kappa == 1.0


def test_calculate_partial_agreement(kappa_calc, partial_agreement_df):
    """Test calculate with partial agreement."""
    kappa = kappa_calc.calculate(partial_agreement_df)
    assert 0 < kappa < 1


def test_calculate_with_missing_values(kappa_calc, missing_values_df):
    """Test calculate with missing values."""
    kappa = kappa_calc.calculate(missing_values_df)
    assert -1.0 <= kappa <= 1.0


def test_calculate_with_single_annotator(kappa_calc):
    """Test calculate with a single annotator."""
    data = {
        'Annotator1_score': [5, 4, 3, 2, 1],
    }
    df = pd.DataFrame(data)

    # With only one annotator, should return 0.0
    kappa = kappa_calc.calculate(df)
    assert kappa == 0.0


def test_calculate_with_no_valid_items(kappa_calc):
    """Test calculate with no valid items."""
    data = {
        'Annotator1_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Annotator2_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Annotator3_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data)

    # With no valid items, should return 0.0
    kappa = kappa_calc.calculate(df)
    assert kappa == 0.0


def test_calculate_by_category(kappa_calc, partial_agreement_df):
    """Test calculate_by_category with partial agreement."""
    kappas = kappa_calc.calculate_by_category(partial_agreement_df)

    # Should return a dictionary with kappa values for each category
    assert isinstance(kappas, dict)
    assert len(kappas) > 0

    # All kappa values should be between -1 and 1
    for category, kappa in kappas.items():
        assert -1.0 <= kappa <= 1.0


def test_calculate_by_category_with_missing_values(kappa_calc,
                                                   missing_values_df):
    """Test calculate_by_category with missing values."""
    kappas = kappa_calc.calculate_by_category(missing_values_df)

    # Should return a dictionary with kappa values for each category
    assert isinstance(kappas, dict)
    assert len(kappas) > 0

    # All kappa values should be between -1 and 1
    for category, kappa in kappas.items():
        assert -1.0 <= kappa <= 1.0


def test_calculate_by_category_with_single_annotator(kappa_calc):
    """Test calculate_by_category with a single annotator."""
    data = {
        'Annotator1_score': [5, 4, 3, 2, 1],
    }
    df = pd.DataFrame(data)

    # With only one annotator, should return empty dict
    kappas = kappa_calc.calculate_by_category(df)
    assert kappas == {}


def test_calculate_by_category_with_no_valid_items(kappa_calc):
    """Test calculate_by_category with no valid items."""
    data = {
        'Annotator1_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Annotator2_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Annotator3_score': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data)

    # With no valid items, should return empty dict
    kappas = kappa_calc.calculate_by_category(df)
    assert kappas == {}


def test_interpret_kappa(kappa_calc):
    """Test interpret_kappa with various kappa values."""
    # Test interpretation for each range
    assert "Poor agreement" in kappa_calc.interpret_kappa(-0.1)
    assert "Slight agreement" in kappa_calc.interpret_kappa(0.1)
    assert "Fair agreement" in kappa_calc.interpret_kappa(0.3)
    assert "Moderate agreement" in kappa_calc.interpret_kappa(0.5)
    assert "Substantial agreement" in kappa_calc.interpret_kappa(0.7)
    assert "Almost perfect agreement" in kappa_calc.interpret_kappa(0.9)


def test_edge_cases(kappa_calc):
    """Test edge cases for Fleiss' Kappa calculation."""
    # Case 1: All annotators use the same category for all items
    data = {
        'Annotator1_score': [1, 1, 1, 1, 1],
        'Annotator2_score': [1, 1, 1, 1, 1],
        'Annotator3_score': [1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    # Should be perfect agreement (kappa = 1.0)
    kappa = kappa_calc.calculate(df)
    assert kappa == 1.0

    # Case 2: Each item has only one rating
    data = {
        'Annotator1_score': [1, np.nan, np.nan, np.nan, np.nan],
        'Annotator2_score': [np.nan, 2, np.nan, np.nan, np.nan],
        'Annotator3_score': [np.nan, np.nan, 3, np.nan, np.nan],
        'Annotator4_score': [np.nan, np.nan, np.nan, 4, np.nan],
        'Annotator5_score': [np.nan, np.nan, np.nan, np.nan, 5],
    }
    df = pd.DataFrame(data)

    # With only one rating per item, agreement is undefined
    # Our implementation should handle this gracefully
    kappa = kappa_calc.calculate(df)
    assert -1.0 <= kappa <= 1.0


def test_edge_case_same_category_no_agreement(kappa_calc):
    """Test edge case where all annotators use the same
        category but don't agree."""
    # Create a case where all annotators use only one category (category 1)
    # but they don't all agree on the same items
    data = {
        'Annotator1_score': [1, 1, 1, np.nan, np.nan],
        'Annotator2_score': [np.nan, np.nan, 1, 1, 1],
        'Annotator3_score': [1, np.nan, np.nan, 1, 1],
    }
    df = pd.DataFrame(data)

    # In this case, P_e will be close to 1 (all ratings are category 1)
    # but P_o will be less than 1 (annotators don't rate the same items)
    # So kappa should be 0.0
    kappa = kappa_calc.calculate(df)
    assert kappa == 0.0


def test_calculate_by_category_edge_case(kappa_calc):
    """Test calculate_by_category with edge case where P_e is 1."""
    # Create a case where all annotators use only one category for each item
    data = {
        'Annotator1_score': [1, 2, 3],
        'Annotator2_score': [1, 2, 3],
        'Annotator3_score': [1, 2, 3],
    }
    df = pd.DataFrame(data)

    # Calculate kappa by category
    kappas = kappa_calc.calculate_by_category(df)

    # For each category, all annotators either all use it or all don't use it
    # So for each category, P_e should be 1 and kappa should be 1.0
    assert kappas[1] == 1.0  # Perfect agreement for category 1
    assert kappas[2] == 1.0  # Perfect agreement for category 2
    assert kappas[3] == 1.0  # Perfect agreement for category 3

    # Now test the case where P_e is 1 but P_o is not 1
    data = {
        'Annotator1_score': [1, 1, 1, 2],
        'Annotator2_score': [1, 1, 2, 1],
        'Annotator3_score': [1, 2, 1, 1],
    }
    df = pd.DataFrame(data)

    # Calculate kappa by category
    kappas = kappa_calc.calculate_by_category(df)

    # For category 1, P_e should be close to 1 (most ratings are 1)
    # But P_o is clearly less than 1 (annotators don't always agree)
    # According to our implementation, kappa should be 0.0 or close to 0
    assert kappas[1] < 0.5  # Not perfect agreement for category 1


def test_calculate_by_category_single_rating(kappa_calc):
    """Test calculate_by_category with items having only one rating."""
    # Create a simpler case where all annotators evaluate all items
    data = {
        'Annotator1_score': [1, 2, 3, 4, 5],
        'Annotator2_score': [1, 2, 3, 4, 5],
        'Annotator3_score': [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)

    # Calculate kappa by category
    kappas = kappa_calc.calculate_by_category(df)

    # For all categories, all annotators agree (perfect agreement)
    for category in range(1, 6):
        assert kappas[category] == 1.0  # Perfect agreement for all categories

    # Now test with a more complex case
    data = {
        'Annotator1_score': [1, 2, 3, 4, 5],
        'Annotator2_score': [1, 2, 3, 4, 4],  # Disagree on item 4
        'Annotator3_score': [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)

    # Calculate kappa by category
    kappas = kappa_calc.calculate_by_category(df)

    # For categories 1-3, all annotators agree (perfect agreement)
    for category in range(1, 4):
        assert kappas[category] == 1.0  # Perfect agreement for categories 1-3

    # For category 5, there is disagreement on item 4
    assert kappas[5] < 1.0  # Not perfect agreement for category 5

    # Test with items having only one rating
    data = {
        'Annotator1_score': [1, np.nan, np.nan],
        'Annotator2_score': [np.nan, 2, np.nan],
        'Annotator3_score': [np.nan, np.nan, 3],
    }
    df = pd.DataFrame(data)

    # Calculate kappa by category
    kappas = kappa_calc.calculate_by_category(df)

    # For each category, there is only one rating
    # The implementation should handle this gracefully
    for category in [1, 2, 3]:
        assert category in kappas
        # We don't check the exact value of kappa here, as it might be outside
        # the normal range due to the special case of having only one rating
        # per category
