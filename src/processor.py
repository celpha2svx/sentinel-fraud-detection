import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("data_processor")


class DataProcessor:
    """
    Handles the initial ingestion and cleaning of transaction data.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads dataset and logs the initial state."""
        try:
            logger.info(f"Attempting to load data from {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded {len(self.df)} records.")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def basic_clean(self):
        """Removes duplicates and handles nulls to ensure a stable baseline."""
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return

        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.fillna({'amount': 0, 'device_id': 'unknown'}, inplace=True)
        logger.info(f"Cleaned {initial_count - len(self.df)} duplicate rows.")
        return self.df