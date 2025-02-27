import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from src.dataPreparation import DataLoader
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


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_content = (
        "review_id,Annotator1_name,Annotator1_score,"
        "Annotator2_name,Annotator2_score\n"
        "1,Gemini_1,5,Mistral_1,5\n"
        "2,Gemini_1,1,Mistral_1,2\n"
        "3,Gemini_1,5,Mistral_1,5\n"
    )
    csv_file = tmp_path / "test_reviews.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


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

    # Notre implémentation ne conserve que les colonnes de scores
    assert len(score_cols) == 2

    # Vérifier que les scores sont corrects
    assert df.loc['rev1', 'Alice_score'] == 1
    assert df.loc['rev1', 'Bob_score'] == 2


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
                       match="No ID column found in the CSV file"):
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
    assert pd.isna(df.loc['rev1', 'Alice_score'])
    assert df.loc['rev1', 'Bob_score'] == 2
    assert pd.isna(df.loc['rev2', 'Bob_score'])

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
    print(result)
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
            "rev1,Alice,1,Bob,2,Charlie,3\n" +  # All 3 annotators
            "rev2,Alice,2,Bob,2,Charlie,3")     # All 3 annotators

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)

    # Check we have all expected columns with real names
    assert len(df.columns) == 3  # 3 score columns with real names

    # Check that columns have the correct names
    assert 'Alice_score' in df.columns
    assert 'Bob_score' in df.columns
    assert 'Charlie_score' in df.columns

    # Check scores are present
    assert df.loc['rev1', 'Alice_score'] == 1
    assert df.loc['rev1', 'Bob_score'] == 2
    assert df.loc['rev1', 'Charlie_score'] == 3

    os.unlink(f.name)


def test_load_data_basic(data_loader, sample_csv):
    """Test basic data loading functionality."""
    df = data_loader.load_data(sample_csv)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert len(df.columns) == 2  # Two annotators

    # Check column names are correctly transformed
    assert 'Gemini_1_score' in df.columns
    assert 'Mistral_1_score' in df.columns


def test_load_data_content(data_loader, sample_csv):
    """Test that data content is correctly loaded."""
    df = data_loader.load_data(sample_csv)

    # Check specific values
    assert df.loc['1', 'Gemini_1_score'] == 5
    assert df.loc['1', 'Mistral_1_score'] == 5
    assert df.loc['2', 'Gemini_1_score'] == 1
    assert df.loc['2', 'Mistral_1_score'] == 2


def test_load_data_missing_file(data_loader):
    """Test handling of missing file."""
    with pytest.raises(FileNotFoundError):
        data_loader.load_data("nonexistent.csv")


def test_load_data_invalid_format(tmp_path):
    """Test handling of invalid CSV format."""
    # Create CSV with wrong format
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("col1,col2\n1,2\n3,4\n")

    with pytest.raises(Exception):
        DataLoader().load_data(str(invalid_csv))


def test_load_data_empty_file(tmp_path):
    """Test handling of empty CSV file."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    with pytest.raises(Exception):
        DataLoader().load_data(str(empty_csv))


def test_load_data_with_logger(sample_csv):
    """Test data loading with custom logger."""
    logger = Logger()
    loader = DataLoader(logger)
    df = loader.load_data(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert loader.logger == logger


def test_load_data_consistent_names(data_loader, tmp_path):
    """Test that annotator names are consistently mapped."""
    # Create CSV with same annotator having different names
    csv_content = (
        "review_id,Annotator1_name,Annotator1_score,"
        "Annotator2_name,Annotator2_score\n"
        "1,Gemini_1,5,Mistral_1,5\n"
        "2,Gemini_1,1,Mistral_1,2\n"
        "3,Gemini_1,5,Mistral_1,5\n"
    )
    csv_file = tmp_path / "test_consistent.csv"
    csv_file.write_text(csv_content)

    df = data_loader.load_data(str(csv_file))

    # Check that column names are consistent
    assert all(col.endswith('_score') for col in df.columns)
    assert 'Gemini_1_score' in df.columns
    assert 'Mistral_1_score' in df.columns


def test_is_standard_format(data_loader):
    """Test _is_standard_format method."""
    # Valid standard format
    columns = ['review_id', 'AnnotatorName', 'Score']
    assert data_loader._is_standard_format(columns) is True

    # Missing required column
    columns = ['review_id', 'AnnotatorName']
    assert data_loader._is_standard_format(columns) is False

    # Extra columns should still be valid
    columns = ['review_id', 'AnnotatorName', 'Score', 'ExtraColumn']
    assert data_loader._is_standard_format(columns) is True


def test_is_wide_format(data_loader):
    """Test _is_wide_format method."""
    # Valid wide format
    columns = ['review_id', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True

    # Missing review_id
    columns = ['id', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True

    # No score columns
    columns = ['review_id', 'Annotator1', 'Annotator2']
    assert data_loader._is_wide_format(columns) is True

    # Mixed columns should still be valid
    columns = ['review_id', 'Annotator1_score', 'OtherColumn']
    assert data_loader._is_wide_format(columns) is True


def test_is_name_score_format(data_loader):
    """Test _is_name_score_format method."""
    # Valid name-score format
    columns = ['review_id', 'Annotator1_name', 'Annotator1_score',
               'Annotator2_name', 'Annotator2_score']
    assert data_loader._is_name_score_format(columns) is True

    # Missing review_id
    columns = ['Annotator1_name', 'Annotator1_score']
    with pytest.raises(ValueError):
        data_loader._is_name_score_format(columns)

    # Unequal number of name and score columns
    columns = ['review_id', 'Annotator1_name', 'Annotator1_score',
               'Annotator2_name']
    with pytest.raises(ValueError):
        data_loader._is_name_score_format(columns)

    # Missing corresponding score column
    columns = ['review_id', 'Annotator1_name', 'Annotator2_score']
    with pytest.raises(ValueError):
        data_loader._is_name_score_format(columns)


def test_load_csv_method(data_loader):
    """Test the legacy load_csv method."""
    # Create a temporary CSV file in standard format
    data = ("review_id,AnnotatorName,Score\n" +
            "rev1,Alice,1\n" +
            "rev2,Bob,2")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Test loading with load_csv
    df = data_loader.load_csv(f.name)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert 'Alice' in df.columns
    assert 'Bob' in df.columns

    # Check values
    assert df.loc['rev1', 'Alice'] == 1
    assert df.loc['rev2', 'Bob'] == 2

    os.unlink(f.name)


def test_load_csv_empty_file(data_loader):
    """Test load_csv with empty file."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write("")

    with pytest.raises(ValueError, match="Error loading CSV file"):
        data_loader.load_csv(f.name)

    os.unlink(f.name)


def test_load_csv_invalid_format(data_loader):
    """Test load_csv with invalid format."""
    data = "column1,column2\nvalue1,value2"

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    with pytest.raises(ValueError,
                       match="No ID column found in the CSV file"):
        data_loader.load_csv(f.name)

    os.unlink(f.name)


def test_consistent_column_naming(data_loader):
    """Test that column naming is consistent across different formats."""
    # Create a temporary CSV file with name-score format
    data = ("review_id,Annotator1_name,Annotator1_score," +
            "Annotator2_name,Annotator2_score\n" +
            "1,Gemini_1,5,Mistral_1,5\n" +
            "2,Gemini_1,1,Mistral_1,2\n" +
            "3,Gemini_1,3,Mistral_1,4")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)

    # Check that column names are consistent
    assert all(col.endswith('_score') for col in df.columns)
    assert 'Gemini_1_score' in df.columns
    assert 'Mistral_1_score' in df.columns

    os.unlink(f.name)


def test_identify_id_column(data_loader):
    """Test the _identify_id_column method."""
    # Test with standard review_id
    columns = ['review_id', 'AnnotatorName', 'Score']
    assert data_loader._identify_id_column(columns) == 'review_id'

    # Test with alternative id column names
    columns = ['id', 'AnnotatorName', 'Score']
    assert data_loader._identify_id_column(columns) == 'id'

    columns = ['item', 'AnnotatorName', 'Score']
    assert data_loader._identify_id_column(columns) == 'item'

    columns = ['key', 'AnnotatorName', 'Score']
    assert data_loader._identify_id_column(columns) == 'key'

    # Test with no id column
    columns = ['column1', 'column2', 'column3']
    assert data_loader._identify_id_column(columns) is None


def test_is_standard_format_with_different_id_columns(data_loader):
    """Test _is_standard_format with different ID column names."""
    # With review_id
    columns = ['review_id', 'AnnotatorName', 'Score']
    assert data_loader._is_standard_format(columns) is True

    # With id
    columns = ['id', 'AnnotatorName', 'Score']
    assert data_loader._is_standard_format(columns) is True

    # With item
    columns = ['item', 'AnnotatorName', 'Score']
    assert data_loader._is_standard_format(columns) is True

    # With key
    columns = ['key', 'AnnotatorName', 'Score']
    assert data_loader._is_standard_format(columns) is True


def test_is_name_score_format_with_different_id_columns(data_loader):
    """Test _is_name_score_format with different ID column names."""
    # With review_id
    columns = ['review_id', 'Annotator1_name', 'Annotator1_score']
    assert data_loader._is_name_score_format(columns) is True

    # With id
    columns = ['id', 'Annotator1_name', 'Annotator1_score']
    assert data_loader._is_name_score_format(columns) is True

    # With item
    columns = ['item', 'Annotator1_name', 'Annotator1_score']
    assert data_loader._is_name_score_format(columns) is True

    # With key
    columns = ['key', 'Annotator1_name', 'Annotator1_score']
    assert data_loader._is_name_score_format(columns) is True


def test_is_wide_format_with_different_id_columns(data_loader):
    """Test _is_wide_format with different ID column names."""
    # With review_id
    columns = ['review_id', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True

    # With id
    columns = ['id', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True

    # With item
    columns = ['item', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True

    # With key
    columns = ['key', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_wide_format(columns) is True


def test_load_data_with_different_id_columns(data_loader):
    """Test loading data with different ID column names."""
    # Test with 'id' instead of 'review_id'
    data = ("id,AnnotatorName,Score\n" +
            "rev1,Alice,1\n" +
            "rev2,Bob,2")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    df = data_loader.load_data(f.name)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'id'  # Index name should match the ID column
    assert 'Alice' in df.columns
    assert 'Bob' in df.columns

    os.unlink(f.name)


def test_load_csv_file_not_found(data_loader):
    """Test load_csv with a non-existent file."""
    non_existent_file = "non_existent_file.csv"

    # Vérifier que FileNotFoundError est propagée sans être convertie
    with pytest.raises(FileNotFoundError):
        data_loader.load_csv(non_existent_file)


def test_process_wide_format_directly(data_loader):
    """Test _process_wide_format method directly."""
    # Create a test DataFrame
    data = {
        'review_id': ['1', '2', '3'],
        'Annotator1': [5, 4, 3],
        'Annotator2': [4, 3, 2],
        'Annotator3': [3, 2, 1]
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_wide_format(df)

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'review_id'
    assert len(result.columns) == 3

    # Check content
    assert result.loc['1', 'Annotator1'] == 5
    assert result.loc['2', 'Annotator2'] == 3
    assert result.loc['3', 'Annotator3'] == 1

    # Test with different ID column
    data = {
        'id': ['1', '2', '3'],
        'Annotator1': [5, 4, 3],
        'Annotator2': [4, 3, 2]
    }
    df = pd.DataFrame(data)

    result = data_loader._process_wide_format(df)

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'id'
    assert len(result.columns) == 2

    # Check content
    assert result.loc['1', 'Annotator1'] == 5
    assert result.loc['2', 'Annotator2'] == 3


def test_process_wide_format_with_invalid_values(data_loader):
    """Test _process_wide_format with invalid values."""
    # Create a test DataFrame with non-numeric values
    data = {
        'review_id': ['1', '2', '3'],
        'Annotator1': ['5', 'invalid', '3'],
        'Annotator2': ['4', '3', 'nan']
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_wide_format(df)

    # Check that invalid values are converted to NaN
    assert result.loc['1', 'Annotator1'] == 5
    assert pd.isna(result.loc['2', 'Annotator1'])
    assert pd.isna(result.loc['3', 'Annotator2'])


def test_process_wide_format_with_empty_ids(data_loader):
    """Test _process_wide_format with empty IDs."""
    # Create a test DataFrame with empty IDs
    data = {
        'review_id': ['1', '', 'nan'],
        'Annotator1': [5, 4, 3],
        'Annotator2': [4, 3, 2]
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_wide_format(df)

    # Check that rows with empty IDs are removed
    assert len(result) == 1
    assert '1' in result.index
    assert '' not in result.index
    assert 'nan' not in result.index


def test_load_data_general_exception(data_loader, monkeypatch):
    """Test handling of general exceptions in load_data."""
    # Mock pd.read_csv to raise a general exception
    def mock_read_csv(*args, **kwargs):
        raise Exception("General test exception")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    # Test that the exception is caught and converted to ValueError
    with pytest.raises(ValueError,
                       match="Error loading CSV file: General test exception"):
        data_loader.load_data("any_file.csv")


def test_process_standard_format_with_nan_values(data_loader):
    """Test _process_standard_format with NaN values."""
    # Create a DataFrame with NaN values
    data = {
        'review_id': ['1', '2', np.nan, '4', ''],
        'AnnotatorName': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
        'Score': [1, 2, 3, np.nan, 5]
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_standard_format(df)

    # Check that rows with NaN review_ids are removed
    assert len(result) == 3  # Only rows with valid review_ids
    assert '1' in result.index
    assert '2' in result.index
    assert '4' in result.index

    # Check that NaN scores are preserved
    assert pd.isna(result.loc['4', 'Dave'])


def test_process_name_score_format_with_nan_values(data_loader):
    """Test _process_name_score_format with NaN values."""
    # Create a DataFrame with NaN values in name and score columns
    data = {
        'review_id': ['1', '2', '3'],
        'Annotator1_name': ['Alice', np.nan, 'Charlie'],
        'Annotator1_score': [1, 2, np.nan],
        'Annotator2_name': ['Bob', 'Dave', np.nan],
        'Annotator2_score': [np.nan, 4, 6]
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_name_score_format(df)

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'review_id'

    # Check that the first non-NaN name is used for the column
    assert 'Alice_score' in result.columns
    assert 'Bob_score' in result.columns

    # Check that NaN scores are preserved

    # NaN score for Annotator1 in row 3
    assert pd.isna(result.loc['3', 'Alice_score'])
    # NaN score for Annotator2 in row 1
    assert pd.isna(result.loc['1', 'Bob_score'])


def test_process_name_score_format_all_nan_names(data_loader):
    """Test _process_name_score_format with all NaN names."""
    # Create a DataFrame with all NaN values in name columns
    data = {
        'review_id': ['1', '2'],
        'Annotator1_name': [np.nan, np.nan],
        'Annotator1_score': [1, 2],
        'Annotator2_name': [np.nan, np.nan],
        'Annotator2_score': [3, 4]
    }
    df = pd.DataFrame(data)

    # Process the DataFrame
    result = data_loader._process_name_score_format(df)

    # Check that generic names are used
    assert 'Annotator1_score' in result.columns
    assert 'Annotator2_score' in result.columns

    # Check values
    assert result.loc['1', 'Annotator1_score'] == 1
    assert result.loc['2', 'Annotator2_score'] == 4


def test_is_wide_format_no_id_column(data_loader):
    """Test _is_wide_format with no ID column."""
    # Create a list of columns without any ID column
    columns = ['Annotator1', 'Annotator2', 'Annotator3']

    # Test that the method raises ValueError when no ID column is found
    with pytest.raises(ValueError, match="No ID column found in the CSV file"):
        data_loader._is_wide_format(columns)


def test_is_name_score_format_no_name_score_columns(data_loader):
    """Test _is_name_score_format with no name/score columns."""
    # Create a list of columns with an ID but no name/score columns
    columns = ['review_id', 'column1', 'column2']

    # Test that the method returns False when no name/score columns are found
    assert data_loader._is_name_score_format(columns) is False

    # Test with only name columns but no score columns
    columns = ['review_id', 'Annotator1_name', 'Annotator2_name']
    assert data_loader._is_name_score_format(columns) is False

    # Test with only score columns but no name columns
    columns = ['review_id', 'Annotator1_score', 'Annotator2_score']
    assert data_loader._is_name_score_format(columns) is False


def test_is_standard_format_no_id_column(data_loader):
    """Test _is_standard_format with no ID column."""
    # Create a list of columns without any ID column
    columns = ['AnnotatorName', 'Score', 'OtherColumn']

    # Test that the method raises ValueError when no ID column is found
    with pytest.raises(ValueError, match="No ID column found in the CSV file"):
        data_loader._is_standard_format(columns)


def test_load_data_value_error_propagation(data_loader, monkeypatch):
    """Test that ValueError is propagated correctly in load_data."""
    # Mock _is_standard_format to raise a ValueError
    def mock_is_standard_format(*args, **kwargs):
        raise ValueError("Custom value error")

    monkeypatch.setattr(data_loader,
                        "_is_standard_format",
                        mock_is_standard_format)

    # Create a simple CSV file
    data = "review_id,AnnotatorName,Score\n1,Alice,5"
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Test that the ValueError is propagated
    with pytest.raises(ValueError, match="Custom value error"):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_load_data_no_recognized_format(data_loader, monkeypatch):
    """Test load_data when no format is recognized."""
    # Mock all format detection methods to return False
    monkeypatch.setattr(data_loader, "_is_standard_format", lambda x: False)
    monkeypatch.setattr(data_loader, "_is_name_score_format", lambda x: False)
    monkeypatch.setattr(data_loader, "_is_wide_format", lambda x: False)

    # Create a simple CSV file
    data = "review_id,column1,column2\n1,value1,value2"
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Test that the appropriate error is raised
    with pytest.raises(ValueError,
                       match="The CSV file is not in the expected format."):
        data_loader.load_data(f.name)

    os.unlink(f.name)


def test_load_data_score_columns_only(data_loader):
    """Test loading data with score columns only (no name columns)."""
    # Create a CSV file with only score columns
    data = ("review_id,Annotator1_score,Annotator2_score\n" +
            "rev1,5,4\n" +
            "rev2,3,2")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Load the data
    df = data_loader.load_data(f.name)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert 'Annotator1_score' in df.columns
    assert 'Annotator2_score' in df.columns

    # Check content
    assert df.loc['rev1', 'Annotator1_score'] == 5
    assert df.loc['rev1', 'Annotator2_score'] == 4
    assert df.loc['rev2', 'Annotator1_score'] == 3
    assert df.loc['rev2', 'Annotator2_score'] == 2

    os.unlink(f.name)


def test_load_data_with_regular_columns(data_loader):
    """Test loading data with regular column names (not ending with _score)."""
    # Create a CSV file with regular column names
    data = ("review_id,Annotator1,Annotator2\n" +
            "rev1,5,4\n" +
            "rev2,3,2")

    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    # Load the data
    df = data_loader.load_data(f.name)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'review_id'
    assert 'Annotator1' in df.columns
    assert 'Annotator2' in df.columns

    # Check content
    assert df.loc['rev1', 'Annotator1'] == 5
    assert df.loc['rev1', 'Annotator2'] == 4
    assert df.loc['rev2', 'Annotator1'] == 3
    assert df.loc['rev2', 'Annotator2'] == 2

    os.unlink(f.name)
