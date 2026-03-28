import pandas as pd
import os
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import CapstoneData

class SentimentAnalysis:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def train(self, train_path: str) -> float:
        self.train_data = CapstoneData.DataExploration(train_path)
        self.preprocess_data()

        # Features and target
        X = self.train_data.df[['brand', 'categories', 'primaryCategories', 'reviews.text', 'reviews.title']]
        y = self.train_data.df['sentiment']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_features = ['brand', 'categories', 'primaryCategories']

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=10000), 'reviews.text'),
                ('title', TfidfVectorizer(max_features=3000), 'reviews.title'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Pipeline with Logistic Regression
        self.model = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ])

        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def persistModel(self, model_path: str) -> bool:
        if self.model is not None:
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
            return True
        else:
            print("No model to save. Please train the model first.")
            return False

    def load_model(self, model_path: str) -> bool:
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file not found at {model_path}")
            return False

    def preprocess_data(self) -> None:
        if self.train_data.df is not None:
            self.train_data.remove_nulls(columns=['name', 'reviews.title', 'reviews.text', 'sentiment'])
            self.train_data.get_summary()
        else:
            print("Data not loaded. Cannot preprocess.")

    def analyze_sentiments(self) -> None:
        if hasattr(self, "train_data") and self.train_data.df is not None:
            print("\n--- Sentiment Distribution After Preprocessing ---")
            print(self.train_data.df['sentiment'].value_counts())
        else:
            print("Data not loaded. Cannot analyze sentiments.")

    def predict(
            self,
            reviewText: str,
            reviewTitle: str = "",
            productBrand: str = "Unknown",
            productCategory: str = "General",
            primaryCategory: str = "General"
        ) -> str:
            if self.model is not None:
                sample = pd.DataFrame([{
                    "reviews.text": reviewText,
                    "reviews.title": reviewTitle,
                    "brand": productBrand,
                    "categories": productCategory,
                    "primaryCategories": productCategory
                }])
                return self.model.predict(sample)[0]
            else:
                print("Model not loaded. Please load or train the model first.")
                return "Model not available"

# Example usage
#sa = SentimentAnalysis()
#acc = sa.train('../Capstone/Ecommerce-Sentiment-Analysis/Ecommerce_dataset/train_data.csv')
#print("Accuracy:", acc)
#sa.persistModel('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_TFD_LR.pkl')

#sample = pd.DataFrame([{
#    "reviews.text": "It's a great product for a thrift store, not for someone who wants a quality product.",
#    "reviews.title": "Review of Amazon Echo Show Alexa-enabled Bluetooth Speaker with 7\" Screen - Charcoal",
#    "brand": "Unknown",
#    "categories": "General",
#    "primaryCategories": "General"
#}])

#loaded_model = SentimentAnalysis('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_TFD_LR.pkl')
#print(loaded_model.model.predict(sample))