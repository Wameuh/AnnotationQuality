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
        Load and prepare data from CSV file.

        Supports multiple formats:
        1. Standard format: columns = [review_id, AnnotatorName, Score]
           - Each row contains one review and one annotation
           - Multiple rows per review (one per annotator)

        2. Wide format with name and score columns:
           - columns = [review_id,
                        Annotator1_name, Annotator1_score,
                        Annotator2_name, Annotator2_score, ...]
           - Each row contains one review and all annotations
           - Name and score columns are paired

        3. Wide format with score columns only:
           - columns = [review_id, Annotator1_score, Annotator2_score, ...]
           - Each row contains one review and all annotations
           - Only score columns, no name columns

        Args:
            file_path (str): Path to CSV file.

        Returns:
            pd.DataFrame: DataFrame in wide format with annotator scores as
                            columns.
        """
        try:
            # Read CSV file
            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError as e:
                # Format error message to match test expectations
                raise ValueError(f"Error loading CSV file: {str(e)}")
            except FileNotFoundError:
                # Propagate file not found error without converting it
                self._logger.error(f"File not found: {file_path}")
                raise

            self._logger.debug(f"Loaded {len(df)} rows from {file_path}")

            # Check if file is empty
            if len(df) == 0:
                raise ValueError("CSV file is empty")

            # Identify the ID column (review_id or similar)
            id_column = self._identify_id_column(df.columns)
            if id_column is None:
                raise ValueError("No ID column found in the CSV file")

            # Determine format and process accordingly
            if self._is_standard_format(df.columns):
                self._logger.debug("Detected standard format")
                return self._process_standard_format(df)
            elif self._is_name_score_format(df.columns):
                self._logger.debug(
                    "Detected wide format with name and score columns")
                return self._process_name_score_format(df)
            elif self._is_wide_format(df.columns):
                self._logger.debug(
                    "Detected wide format with score columns only")
                return self._process_wide_format(df)
            else:
                raise ValueError("The CSV file is not in the expected format.")

        except ValueError as e:
            self._logger.error(f"Error loading data: {str(e)}")
            raise
        except FileNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def _identify_id_column(self, columns):
        """
        Identify the ID column in the CSV file.

        Looks for columns named 'review_id', 'id', or similar.

        Args:
            columns: List of column names

        Returns:
            str: Name of the ID column, or None if not found
        """
        id_candidates = ['review_id', 'id', 'item', 'key']
        for candidate in id_candidates:
            if candidate in columns:
                return candidate
        return None

    def _is_standard_format(self, columns):
        """
        Check if CSV is in standard format.

        Standard format has columns:
        - An ID column (review_id or similar)
        - AnnotatorName
        - Score
        """
        id_column = self._identify_id_column(columns)
        if id_column is None:
            raise ValueError("No ID column found in the CSV file")

        return 'AnnotatorName' in columns and 'Score' in columns

    def _is_name_score_format(self, columns):
        """
        Check if CSV is in wide format with name and score columns.

        This format has:
        - An ID column (review_id or similar)
        - Pairs of name/score columns (Annotator1_name, Annotator1_score, etc.)
        """
        id_column = self._identify_id_column(columns)
        if id_column is None:
            raise ValueError("No ID column found in the CSV file")

        name_cols = [col for col in columns if col.endswith('_name')]
        score_cols = [col for col in columns if col.endswith('_score')]

        # Check if we have pairs of name/score columns
        if len(name_cols) == 0 or len(score_cols) == 0:
            return False

        # Check if we have the same number of name and score columns
        if len(name_cols) != len(score_cols):
            raise ValueError(
                "The CSV file is not in the expected format: "
                "unequal number of name and score columns")

        # Check that each name column has a corresponding score column
        for name_col in name_cols:
            prefix = name_col.replace('_name', '')
            score_col = f"{prefix}_score"
            if score_col not in columns:
                raise ValueError(
                    f"The CSV file is not in the expected format: "
                    f"missing score column for {name_col}")

        return True

    def _is_wide_format(self, columns):
        """
        Check if CSV is in wide format with score columns only.

        This format has:
        - An ID column (review_id or similar)
        - Score columns (Annotator1_score, Annotator2_score, etc.)
        - Or any other columns that could be annotator names
        """
        id_column = self._identify_id_column(columns)
        if id_column is None:
            raise ValueError("No ID column found in the CSV file")

        # If there's at least one column besides the ID column, consider it
        # wide format
        return len(columns) > 1

    def _process_standard_format(self, df):
        """Process CSV in standard format."""
        # Create an explicit copy of the DataFrame
        df = df.copy()

        # Identify the ID column
        id_column = self._identify_id_column(df.columns)

        # Remove rows with empty or NaN IDs
        df = df.dropna(subset=[id_column])

        # Convert ID to string and clean
        df[id_column] = df[id_column].astype(str)
        df = df[df[id_column].str.strip() != '']
        df = df[df[id_column].str.lower() != 'nan']

        # Convert scores to numeric
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

        # Pivot to wide format
        wide_df = df.pivot(
            index=id_column,
            columns='AnnotatorName',
            values='Score'
        )
        return wide_df

    def _process_wide_format(self, df):
        """Process CSV in wide format with score columns only."""
        # Create an explicit copy of the DataFrame
        df = df.copy()

        # Identify the ID column
        id_column = self._identify_id_column(df.columns)

        # Remove rows with empty or NaN IDs
        df = df.dropna(subset=[id_column])

        # Convert ID to string and clean
        df[id_column] = df[id_column].astype(str)
        df = df[df[id_column].str.strip() != '']
        df = df[df[id_column].str.lower() != 'nan']

        # Get all columns except the ID column
        annotator_cols = [col for col in df.columns if col != id_column]

        # Validate scores - try to convert all columns to numeric
        df[annotator_cols] = df[annotator_cols].apply(pd.to_numeric,
                                                      errors='coerce')

        # Set ID as index
        return df.set_index(id_column)[annotator_cols]

    def _process_name_score_format(self, df):
        """Process CSV in wide format with name and score columns."""
        # Create an explicit copy of the DataFrame
        df = df.copy()

        # Identify the ID column
        id_column = self._identify_id_column(df.columns)

        # Remove rows with empty or NaN IDs
        df = df.dropna(subset=[id_column])

        # Convert ID to string and clean
        df[id_column] = df[id_column].astype(str)
        df = df[df[id_column].str.strip() != '']
        df = df[df[id_column].str.lower() != 'nan']

        # Get annotator names and scores
        name_cols = [col for col in df.columns if col.endswith('_name')]
        score_cols = [col for col in df.columns if col.endswith('_score')]

        # Validate scores
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')

        # Create mapping of annotator numbers to real names
        annotator_names = {}
        for name_col in name_cols:
            prefix = name_col.replace('_name', '')
            score_col = f"{prefix}_score"
            if score_col in score_cols:
                # Find the first non-NaN name for this annotator
                non_nan_names = df[name_col].dropna()
                if not non_nan_names.empty:
                    first_valid_name = non_nan_names.iloc[0]
                else:
                    first_valid_name = prefix
                annotator_names[prefix] = first_valid_name

        # Store real names in a class attribute for test.py
        self._annotator_real_names = annotator_names

        # Create a new DataFrame with real names as column headers
        result_data = {}
        for prefix, real_name in annotator_names.items():
            score_col = f"{prefix}_score"
            if score_col in score_cols:
                # Use the original score column directly
                result_data[f"{real_name}_score"] = df[score_col].values

        # Create the result DataFrame with the original index
        result_df = pd.DataFrame(result_data, index=df[id_column])

        # Log the columns for debugging
        self._logger.debug(f"Annotator names: {annotator_names}")
        self._logger.debug(f"Result columns: {list(result_df.columns)}")

        return result_df

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Legacy method to load data from a CSV file.

        This method is maintained for backward compatibility.
        It delegates to load_data() for actual processing.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the annotation data.

        Raises:
            ValueError: If required columns are missing or CSV is malformed.
        """
        try:
            # Simply delegate to the main load_data method
            return self.load_data(file_path)
        except Exception as e:
            # Keep the original error message format for backward compatibility
            self.logger.error(f"Error loading CSV: {str(e)}")
            if isinstance(e, FileNotFoundError):
                raise
            else:
                raise ValueError(f"Error loading CSV file: {e}")
