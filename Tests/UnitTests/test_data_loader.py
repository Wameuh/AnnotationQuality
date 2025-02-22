import pytest
import tempfile
import os
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
    data = data_loader.load_data(temp_csv_standard)

    assert 'rev1' in data
    assert 'rev2' in data
    assert len(data['rev1']) == 2
    assert len(data['rev2']) == 2

    # Check first review annotations
    annotations = data['rev1']
    assert any(a['name'] == 'Alice' and a['score'] == 1 for a in annotations)
    assert any(a['name'] == 'Bob' and a['score'] == 2 for a in annotations)


def test_load_wide_format(data_loader, temp_csv_wide):
    """Test loading data in wide format."""
    data = data_loader.load_data(temp_csv_wide)

    assert 'rev1' in data
    assert 'rev2' in data

    # Check first review
    rev1 = data['rev1']
    assert rev1['Annotator1']['name'] == 'Alice'
    assert rev1['Annotator1']['score'] == 1
    assert rev1['Annotator2']['name'] == 'Bob'
    assert rev1['Annotator2']['score'] == 2


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

    result = data_loader.load_data(f.name)
    assert len(result['rev1']) == 2
    assert result['rev1'][0]['name'] == 'Alice'
    assert result['rev1'][0]['score'] == 1
    assert result['rev1'][1]['name'] == ''
    assert result['rev1'][1]['score'] == 2
    assert result['rev2'][0]['name'] == 'Bob'
    assert result['rev2'][0]['score'] is None

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

    result = data_loader.load_data(f.name)
    assert result['rev1']['Annotator1']['name'] == 'Alice'
    assert result['rev1']['Annotator1']['score'] is None
    assert result['rev1']['Annotator2']['name'] == 'Bob'
    assert result['rev1']['Annotator2']['score'] == 2
    assert result['rev2']['Annotator1']['score'] == 1
    assert result['rev2']['Annotator2']['name'] == 'Bob'
    assert result['rev2']['Annotator2']['score'] is None

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
            "rev1,Alice,invalid\n" +
            "rev2,Bob,2.5")
    with tempfile.NamedTemporaryFile(mode='w',
                                     delete=False,
                                     suffix='.csv') as f:
        f.write(data)

    result = data_loader.load_data(f.name)
    assert result['rev1'][0]['name'] == 'Alice'
    assert result['rev1'][0]['score'] is None
    assert result['rev2'][0]['name'] == 'Bob'
    assert result['rev2'][0]['score'] is None

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
    result = data_loader.load_data(f.name)
    assert 'rev1' in result
    assert result['rev1'][0]['name'] == 'Alice'
    assert result['rev1'][0]['score'] == 1

    os.unlink(f.name)
