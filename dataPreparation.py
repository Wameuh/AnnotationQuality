import pandas as pd
from typing import Dict, Any, Optional
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

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """
        Loads data from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            Dict[str, Any]: A dictionary representing the data.

        Raises:
            ValueError: If the CSV format is not supported.
        """
        self.logger.info(f"Loading data from {file_path}")
        try:
            return self.load_csv(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def load_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Loads data from a CSV file using pandas.

        The file can be in two formats:
        1. Standard: columns: review_id, AnnotatorName, Score
        2. Wide: columns: review_id, Annotator1_name, Annotator1_score, etc.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            Dict[str, Any]: A dictionary representing the data.

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
        """
        Check if DataFrame columns match the wide format.
        Wide format should have:
        - First column as 'review_id'
        - Pairs of columns named 'Annotator{N}_name' and 'Annotator{N}_score'
        in correct order
        """
        columns = list(columns)  # Convert to list for easier indexing
        if not columns or columns[0] != 'review_id':
            return False

        # Get non-review_id columns
        remaining = columns[1:]  # Use slicing to preserve order
        if len(remaining) % 2 != 0:  # Must have pairs of columns
            return False

        # Check column pairs in order
        for i in range(0, len(remaining), 2):
            annotator_num = (i // 2) + 1
            name_col = f'Annotator{annotator_num}_name'
            score_col = f'Annotator{annotator_num}_score'

            # Check both name and score columns are in correct order
            if (i >= len(remaining)
                    or remaining[i] != name_col
                    or remaining[i + 1] != score_col):
                return False

        return True

    def _is_standard_format(self, columns: pd.Index) -> bool:
        """Check if DataFrame columns match the standard format."""
        required = ['review_id', 'AnnotatorName', 'Score']
        return all(col in columns for col in required)

    def _safe_convert_to_int(self, value: Any) -> Optional[int]:
        """
        Safely convert a value to integer.

        Args:
            value: The value to convert.

        Returns:
            Optional[int]: The converted integer or None if conversion fails.
        """
        if pd.isna(value):
            return None
        try:
            # Convert to float first to handle both string numbers and floats
            float_val = float(str(value).strip())
            # Check if it's a whole number
            if float_val.is_integer():
                return int(float_val)
            return None
        except (ValueError, TypeError):
            return None

    def _process_standard_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process DataFrame in standard format.
        Handles missing or invalid data gracefully.
        """
        data = {}

        for _, row in df.iterrows():
            review_id = str(row['review_id']).strip()
            # Skip empty, whitespace, or 'nan' review_ids
            is_invalid = (not review_id
                          or pd.isna(row['review_id'])
                          or review_id.lower() == 'nan')
            if is_invalid:
                self.logger.debug(
                    f"Skipping row with invalid review_id: {review_id}"
                )
                continue

            name = (str(row['AnnotatorName']).strip()
                    if not pd.isna(row['AnnotatorName']) else "")
            score = self._safe_convert_to_int(row['Score'])

            if name or score is not None:
                if review_id not in data:
                    data[review_id] = []
                data[review_id].append({
                    'name': name,
                    'score': score
                })

        return data

    def _process_wide_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process DataFrame in wide format.
        Handles missing or invalid data gracefully.
        """
        data = {}
        annotator_count = (len(df.columns) - 1) // 2

        self.logger.info(
            f"Processing wide format with {annotator_count} annotators"
        )

        for _, row in df.iterrows():
            review_id = str(row['review_id']).strip()
            # Skip empty, whitespace, or 'nan' review_ids
            is_invalid = (not review_id
                          or pd.isna(row['review_id'])
                          or review_id.lower() == 'nan')
            if is_invalid:
                self.logger.debug(
                    f"Skipping row with invalid review_id: {review_id}"
                )
                continue

            data[review_id] = {}
            for i in range(annotator_count):
                name_col = f'Annotator{i+1}_name'
                score_col = f'Annotator{i+1}_score'

                name = (str(row[name_col]).strip()
                        if not pd.isna(row[name_col]) else "")
                score = self._safe_convert_to_int(row[score_col])

                if name or score is not None:
                    data[review_id][f'Annotator{i+1}'] = {
                        'name': name,
                        'score': score
                    }

        return data
