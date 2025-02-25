import pandas as pd
from Utils.logger import Logger, LogLevel


class DataLoader:
    """
    A class to load annotation data from CSV files using pandas.

    This class provides methods to load data from CSV files,
    ensuring that the data is correctly structured for further processing.
    It also includes basic data validation and handles missing data.
    """

    def __init__(self, logger: Logger = None):
        """
        Initializes the DataLoader.

        Args:
            logger (Logger, optional): Logger instance for tracking operations.
                If None, creates a new logger.
        """
        self._logger = logger or Logger(level=LogLevel.INFO)

    @property
    def logger(self) -> Logger:
        """Get the logger instance."""
        return self._logger

    @logger.setter
    def logger(self, logger: Logger):
        """Set the logger instance."""
        if not isinstance(logger, Logger):
            raise ValueError("logger must be an instance of Logger")
        self._logger = logger

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the annotation data.
        Raises:
            ValueError: If the CSV format is not supported.
        """
        self.logger.info(f"Loading data from {file_path}")
        try:
            return self.load_csv(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file using pandas.

        The file can be in two formats:
        1. Standard: columns: review_id, AnnotatorName, Score
        2. Wide: columns: review_id, Annotator1_name, Annotator1_score, etc.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the annotation data.

        Raises:
            ValueError: If required columns are missing or CSV is malformed.
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if file is empty (no data rows)
            if len(df.index) == 0:
                raise ValueError("CSV file is empty")

            self.logger.debug(f"CSV headers: {df.columns.tolist()}")

            # Determine format and process accordingly
            if self._is_wide_format(df.columns):
                return self._process_wide_format(df)
            elif self._is_standard_format(df.columns):
                return self._process_standard_format(df)
            else:
                raise ValueError("The CSV file is not in the expected format.")

        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty CSV file: {str(e)}")
            raise ValueError(f"Error loading CSV file: {str(e)}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise ValueError(f"Error loading CSV file: {e}")

    def _is_wide_format(self, columns: pd.Index) -> bool:
        """Check if DataFrame columns match the wide format."""
        columns = list(columns)
        if not columns or columns[0] != 'review_id':
            return False

        # Get non-review_id columns
        remaining = columns[1:]  # Use slicing to preserve order
        if len(remaining) % 2 != 0:  # Must have pairs of columns
            return False

        # Check column pairs in order
        for i in range(0, len(remaining), 2):
            annotator_num = (i // 2) + 1
            expected_name = f'Annotator{annotator_num}_name'
            expected_score = f'Annotator{annotator_num}_score'

            # Check both name and score columns are in correct order
            if (i >= len(remaining) or
                    remaining[i] != expected_name or
                    remaining[i + 1] != expected_score):
                return False

        return True

    def _is_standard_format(self, columns: pd.Index) -> bool:
        """Check if DataFrame columns match the standard format."""
        required = ['review_id', 'AnnotatorName', 'Score']
        return all(col in columns for col in required)

    def _process_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame in wide format."""
        # Créer une copie explicite du DataFrame
        df = df.copy()

        # Remove rows with empty or NaN review_ids
        df = df.dropna(subset=['review_id'])

        # Convert review_id to string and clean
        df['review_id'] = df['review_id'].astype(str)
        df = df[df['review_id'].str.strip() != '']
        df = df[df['review_id'].str.lower() != 'nan']

        # Validate scores
        score_cols = [col for col in df.columns if col.endswith('_score')]
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')

        # Set review_id as index
        df = df.set_index('review_id')

        return df

    def _process_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame in standard format."""
        # Créer une copie explicite du DataFrame
        df = df.copy()

        # Remove rows with empty or NaN review_ids
        df = df.dropna(subset=['review_id'])
        # Convert review_id to string and clean
        df['review_id'] = df['review_id'].astype(str)
        df = df[df['review_id'].str.strip() != '']
        df = df[df['review_id'].str.lower() != 'nan']

        # Convert scores to numeric
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

        # Pivot the data to wide format
        df_wide = df.pivot(
            index='review_id',
            columns='AnnotatorName',
            values='Score'
        )

        return df_wide
