import pytest
import os
import pandas as pd
from dataPreparation import DataLoader
from Utils.logger import Logger, LogLevel


@pytest.fixture
def logger():
    """Fixture providing a logger instance."""
    return Logger(level=LogLevel.DEBUG)


@pytest.fixture
def data_loader(logger):
    """Fixture providing a DataLoader instance."""
    return DataLoader(logger)


@pytest.fixture
def reviews_file():
    """Fixture providing the path to the Reviews_annotated.csv file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "Tests", "Assets", "Reviews_annotated.csv")


def test_reviews_loading(data_loader, reviews_file):
    """Test that Reviews_annotated.csv is loaded correctly."""
    # Check file exists
    assert os.path.exists(reviews_file), "Reviews_annotated.csv file not found"

    # Load data
    df = data_loader.load_data(reviews_file)

    # Check total number of reviews
    assert len(df) == 27900, f"Expected 27900 reviews, got {len(df)}"

    # Check number of annotators
    score_cols = [col for col in df.columns
                  if col.endswith('_score')]
    assert len(score_cols) == 4, \
        f"Expected 4 annotators, got {len(score_cols)}"

    # Check random 5% of reviews have valid data
    sample_size = int(len(df) * 0.05)  # 5% of reviews
    sample_df = df.sample(n=sample_size)

    for idx, row in sample_df.iterrows():
        scores = [score for score in row[score_cols] if pd.notna(score)]
        for score in scores:
            assert isinstance(score, (int, float)), \
                f"Invalid score type in review {idx}"
            assert 1 <= score <= 5, \
                f"Score {score} out of range (1-5) in review {idx}"

    # Print summary
    print("\nTest Summary:")
    print(f"Total reviews: {len(df)}")
    print(f"Number of annotators: {len(score_cols)}")
    print(f"Score columns: {score_cols}")
    print(f"Checked {sample_size} random reviews")

    # Print annotation statistics
    scores_per_review = df[score_cols].notna().sum(axis=1)
    print("\nAnnotation Statistics:")
    print("Scores per review:")
    print(scores_per_review.value_counts().sort_index())
