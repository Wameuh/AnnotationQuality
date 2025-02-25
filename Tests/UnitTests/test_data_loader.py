import pytest
import tempfile
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
def temp_csv_standard():
    """Fixture creating a temporary CSV file in standard format."""
    data = ("review_id,AnnotatorName,Score\n" +
            "rev1,Alice,1\n" +
            "rev1,Bob,2\n" +
            "rev2,Alice,2\n" +
            "rev2,Bob,2")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_wide():
    """Fixture creating a temporary CSV file in wide format."""
    data = ("review_id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score\n" +
            "rev1,Alice,1,Bob,2\n" +
            "rev2,Alice,2,Bob,2")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)
    yield f.name
    os.unlink(f.name)


def test_init_with_logger(logger):
    """Test DataLoader initialization with a logger."""
    loader = DataLoader(logger)
    assert loader.logger == logger


def test_init_without_logger():
    """Test DataLoader initialization without a logger."""
    loader = DataLoader()
    assert isinstance(loader.logger, Logger)
    assert loader.logger.level == LogLevel.INFO


def test_logger_setter_validation_raise_error():
    """Test logger setter validation."""
    loader = DataLoader()
    with pytest.raises(ValueError,
                       match="logger must be an instance of Logger"):
        loader.logger = "not a logger"


def test_logger_setter_validation_without_raise_error(logger):
    """Test logger setter validation."""
    loader = DataLoader()
    loader.logger = logger
    assert loader.logger == logger


def test_load_standard_format(data_loader, temp_csv_standard):
    """Test loading data in standard format."""
    df = data_loader.load_data(temp_csv_standard)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert len(df.index.unique()) == 2  # Two unique review_ids

    # Check data content
    assert 'Alice' in df.columns
    assert 'Bob' in df.columns
    assert df.loc['rev1', 'Alice'] == 1
    assert df.loc['rev1', 'Bob'] == 2
    assert df.loc['rev2', 'Alice'] == 2
    assert df.loc['rev2', 'Bob'] == 2


def test_load_wide_format(data_loader, temp_csv_wide):
    """Test loading data in wide format."""
    df = data_loader.load_data(temp_csv_wide)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert len(df.index.unique()) == 2

    # Check data content
    score_cols = [col for col in df.columns if col.endswith('_score')]
    name_cols = [col for col in df.columns if col.endswith('_name')]

    assert len(score_cols) == 2
    assert len(name_cols) == 2
    assert df.loc['rev1', 'Annotator1_score'] == 1
    assert df.loc['rev1', 'Annotator2_score'] == 2


def test_load_missing_file(data_loader):
    """Test loading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        data_loader.load_data("nonexistent.csv")


def test_load_empty_file(data_loader):
    """Test loading an empty CSV file."""
    with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.csv'
    ) as f:
        f.write("")

    error_msg = "Error loading CSV file: No columns to parse from file"
    with pytest.raises(ValueError, match=error_msg):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_load_headers_only(data_loader):
    """Test loading a CSV file with only headers."""
    data = "review_id,AnnotatorName,Score\n"
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError, match="CSV file is empty"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_load_invalid_format(data_loader):
    """Test loading a CSV file with invalid format."""
    data = "column1,column2\nvalue1,value2"
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_handle_missing_data_standard(data_loader):
    """Test handling of missing data in standard format."""
    data = ("review_id,AnnotatorName,Score\n" +
            "rev1,Alice,1\n" +
            "rev1,,2\n" +
            "rev2,Bob,")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)
    assert pd.isna(df.loc['rev2', 'Bob'])
    assert df.loc['rev1', 'Alice'] == 1

    os.unlink(f.name)


def test_handle_missing_data_wide(data_loader):
    """Test handling of missing data in wide format."""
    data = ("review_id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score\n" +
            "rev1,Alice,,Bob,2\n" +
            "rev2,,1,Bob,")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)
    assert pd.isna(df.loc['rev1', 'Annotator1_score'])
    assert df.loc['rev1', 'Annotator2_score'] == 2
    assert pd.isna(df.loc['rev2', 'Annotator2_score'])

    os.unlink(f.name)


def test_wide_format_invalid_column_order(data_loader):
    """Test wide format with incorrect column order."""
    data = ("review_id,Annotator1_score,Annotator1_name," +
            "Annotator2_name,Annotator2_score\n" +
            "rev1,1,Alice,Bob,2")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_wide_format_missing_review_id(data_loader):
    """Test wide format without review_id column."""
    data = ("id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score\n" +
            "rev1,Alice,1,Bob,2")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_wide_format_odd_columns(data_loader):
    """Test wide format with odd number of annotator columns."""
    data = ("review_id,Annotator1_name,Annotator1_score,Annotator2_name\n" +
            "rev1,Alice,1,Bob")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_standard_format_missing_columns(data_loader):
    """Test standard format with missing required columns."""
    data = ("review_id,Score\n" +
            "rev1,1")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_handle_invalid_score_values(data_loader):
    """Test handling of invalid score values."""
    data = ("review_id,AnnotatorName,Score\n" +
            "rev1,Alice,invalid\n" +  # Non-numeric -> should be NaN
            "rev2,Bob,2.5")          # Float -> should be accepted
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)
    assert pd.isna(df.loc['rev1', 'Alice'])  # Invalid score -> NaN
    assert df.loc['rev2', 'Bob'] == 2.5      # Float score -> kept as is

    os.unlink(f.name)


def test_empty_review_id_standard(data_loader):
    """Test handling of empty review_id in standard format."""
    data = ("review_id,AnnotatorName,Score\n" +
            ",Alice,1\n" +
            ",Bob,2\n" +
            "nan,Charlie,3")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    result = data_loader.load_data(f.name)
    assert len(result) == 0  # No data should be loaded for empty review_ids

    os.unlink(f.name)


def test_empty_review_id_wide(data_loader):
    """Test handling of empty review_id in wide format."""
    data = ("review_id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score\n" +
            ",Alice,1,Bob,2\n" +
            " ,Charlie,3,Dave,4")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    result = data_loader.load_data(f.name)
    assert len(result) == 0  # No data should be loaded for empty review_ids

    os.unlink(f.name)


def test_standard_format_exact_columns(data_loader):
    """Test standard format with exact column order."""
    data = ("AnnotatorName,Score,review_id\n" +
            "Alice,1,rev1")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Should work regardless of column order
    df = data_loader.load_data(f.name)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert 'rev1' in df.index
    assert 'Alice' in df.columns
    assert df.loc['rev1', 'Alice'] == 1

    os.unlink(f.name)


def test_partial_annotations_standard(data_loader):
    """Test handling reviews with missing annotators in standard format."""
    data = ("review_id,AnnotatorName,Score\n" +
            "rev1,Alice,1\n" +
            "rev1,Bob,2\n" +
            "rev1,Charlie,3\n" +  # rev1 has 3 annotators
            "rev2,Alice,2\n" +
            "rev2,Bob,2\n" +
            "rev2,Charlie,3\n" +
            "rev2,Dave,4")        # rev2 has all 4 annotators
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)

    # Check all annotators are columns
    names = ['Alice', 'Bob', 'Charlie', 'Dave']
    assert all(name in df.columns for name in names)

    # Check rev1 has NaN for Dave
    assert pd.isna(df.loc['rev1', 'Dave'])

    # Check other scores are present
    assert df.loc['rev1', 'Alice'] == 1
    assert df.loc['rev1', 'Bob'] == 2
    assert df.loc['rev1', 'Charlie'] == 3

    # Check rev2 has all scores
    assert not pd.isna(df.loc['rev2', 'Dave'])

    os.unlink(f.name)


def test_partial_annotations_wide(data_loader):
    """Test handling reviews with missing annotators in wide format."""
    data = ("review_id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score," +
            "Annotator3_name,Annotator3_score\n" +
            "rev1,Alice,1,Bob,2,Charlie,3\n" +  # Only 3 annotators
            "rev2,Alice,2,Bob,2,Charlie,3")     # Only 3 annotators

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)

    # Check we have all expected columns
    score_cols = [col for col in df.columns if col.endswith('_score')]
    name_cols = [col for col in df.columns if col.endswith('_name')]

    assert len(score_cols) == 3  # Only 3 score columns
    assert len(name_cols) == 3   # Only 3 name columns

    # Check scores are present
    assert df.loc['rev1', 'Annotator1_score'] == 1
    assert df.loc['rev1', 'Annotator2_score'] == 2
    assert df.loc['rev1', 'Annotator3_score'] == 3

    os.unlink(f.name)
