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

    def train(self, data) -> float:
        self.train_data = CapstoneData.DataExploration(data)
        self.preprocess_data()

        # Features and target
        X = self.train_data.df[['brand', 'categories', 'primaryCategories', 'reviews.text', 'reviews.title']]
        y = self.train_data.df['sentiment']

        # Split data (stratify — dataset is imbalanced toward Positive)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        categorical_features = ['brand', 'categories', 'primaryCategories']

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'text',
                    TfidfVectorizer(
                        max_features=14000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                    'reviews.text',
                ),
                (
                    'title',
                    TfidfVectorizer(
                        max_features=4500,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                    'reviews.title',
                ),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ]
        )

        # Pipeline with Logistic Regression
        self.model = Pipeline([
            ('preprocess', preprocessor),
            (
                'classifier',
                LogisticRegression(
                    max_iter=4000,
                    random_state=42,
                    class_weight={'Negative': 15.0, 'Neutral': 5.0, 'Positive': 1.0},
                    C=2.5,
                    solver='saga',
                    n_jobs=-1,
                ),
            ),
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
            self.train_data.remove_nulls(columns=['reviews.title', 'reviews.text', 'sentiment'])
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

if __name__ == "__main__":
    # Example: How to use the model WITHOUT retraining
    # 1. Provide the path to your saved .pkl model file
    model_path = 'sentiment_model_TFD_LR.pkl'
    
    if os.path.exists(model_path):
        sa = SentimentAnalysis(model_path=model_path)
        print("Model loaded successfully!")
        
        # 2. Predict sentiment for a new review
        review_text = "Good product but the battery life is quite short."
        review_title = "Mixed feelings"
        
        prediction = sa.predict(review_text, review_title)
        print(f"Review: {review_text}")
        print(f"Prediction: {prediction}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        
        # Example Training (if starting from scratch):
        # sa = SentimentAnalysis()
        # sa.train('Ecommerce_dataset/train_data.csv')
        # sa.persistModel('sentiment_model_TFD_LR.pkl')