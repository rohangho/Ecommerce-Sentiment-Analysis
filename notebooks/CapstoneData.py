import pandas as pd
from typing import List, Optional
import os

class DataExploration:
    """
    A class for performing Exploratory Data Analysis (EDA) and data cleaning
    on E-commerce sentiment datasets.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataExploration class and loads the dataset.

        :param file_path: Path to the CSV file to be explored.
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Loads the dataset from the specified file path."""
        if not os.path.exists(self.file_path):
            print(f"Error: File not found at {os.path.abspath(self.file_path)}")
            return
            
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully with {len(self.df)} rows and {len(self.df.columns)} columns.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def get_summary(self) -> None:
        """Prints a summary of the dataset including head, info, and missing values."""
        if self.df is not None:
            print("\n" + "="*50)
            print("DATASET SUMMARY")
            print("="*50)
            print("\n--- First 5 rows ---")
            print(self.df.head())
            print("\n--- Column Info ---")
            print(self.df.info())
            print("\n--- Missing Values Count ---")
            print(self.df.isnull().sum())
            print("="*50 + "\n")
        else:
            print("No data available to summarize. Please check if the file was loaded correctly.")

    def get_sentiment_distribution(self) -> None:
        """Prints the value counts for the 'sentiment' column."""
        if self.df is not None:
            if 'sentiment' in self.df.columns:
                print("\n--- Sentiment Distribution ---")
                print(self.df['sentiment'].value_counts())
            else:
                print("\nWarning: 'sentiment' column not found in the dataset.")
        else:
            print("Data not loaded.")

    def remove_nulls(self, columns: Optional[List[str]] = None) -> None:
        """
        Removes rows with null values from specified columns.

        :param columns: List of columns to check for nulls. If None, checks all columns.
        """
        if self.df is not None:
            initial_count = len(self.df)
            self.df.dropna(subset=columns, inplace=True)
            removed_count = initial_count - len(self.df)
            print(f"\nCleaning Complete: Removed {removed_count} rows with null values.")
            print(f"New row count: {len(self.df)}")
        else:
            print("No data available to clean.")