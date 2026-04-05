"""
CapstoneData module — provides the DataExploration class
used by all sentiment model scripts.
Extracted from notebooks/DataExploration.ipynb.
"""

import pandas as pd
from typing import List, Optional
import os


class DataExploration:
    """
    A class for performing Exploratory Data Analysis (EDA) and data cleaning
    on E-commerce sentiment datasets.
    """

    def __init__(self, data_source: Optional[object] = None):
        self.df: Optional[pd.DataFrame] = None
        if isinstance(data_source, str):
            self.file_path = data_source
            self._load_data()
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
            print(f"DataFrame provided directly with {len(self.df)} rows.")
        else:
            print("No valid data source provided.")

    def _load_data(self) -> None:
        if not hasattr(self, 'file_path') or not os.path.exists(self.file_path):
            print(f"Error: File not found.")
            return
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully with {len(self.df)} rows and {len(self.df.columns)} columns.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def get_summary(self) -> None:
        if self.df is not None:
            print("\n" + "=" * 50)
            print("DATASET SUMMARY")
            print("=" * 50)
            print("\n--- First 5 rows ---")
            print(self.df.head())
            print("\n--- Column Info ---")
            print(self.df.info())
            print("\n--- Missing Values Count ---")
            print(self.df.isnull().sum())
            print("=" * 50 + "\n")
        else:
            print("No data available to summarize.")

    def get_sentiment_distribution(self) -> None:
        if self.df is not None and "sentiment" in self.df.columns:
            print("\n--- Sentiment Distribution ---")
            print(self.df["sentiment"].value_counts())
        else:
            print("Data not loaded or 'sentiment' column not found.")

    def remove_nulls(self, columns: Optional[List[str]] = None) -> None:
        if self.df is not None:
            initial_count = len(self.df)
            self.df.dropna(subset=columns, inplace=True)
            if 'reviews.text' in self.df.columns:
                self.df.drop_duplicates(subset=['reviews.text'], inplace=True)
            removed_count = initial_count - len(self.df)
            print(f"\nCleaning Complete: Removed {removed_count} rows with null values.")
            print(f"New row count: {len(self.df)}")
        else:
            print("No data available to clean.")
