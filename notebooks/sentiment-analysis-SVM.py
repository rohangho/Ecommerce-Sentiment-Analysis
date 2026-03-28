import pandas as pd
import os
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

import CapstoneData

class SentimentAnalysisSVM:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def train(self, train_path: str, kernel: str = 'rbf', C: float = 1.0) -> float:
        """
        Train SVM model for sentiment analysis
        
        Args:
            train_path: Path to training data CSV
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter (lower = stronger regularization)
        
        Returns:
            Accuracy score on test set
        """
        self.train_data = CapstoneData.DataExploration(train_path)
        self.preprocess_data()

        # Features and target
        X = self.train_data.df[['brand', 'categories', 'primaryCategories', 'reviews.text', 'reviews.title']]
        y = self.train_data.df['sentiment']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"SVM kernel: {kernel}, C: {C}")

        categorical_features = ['brand', 'categories', 'primaryCategories']

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=10000, ngram_range=(1, 2)), 'reviews.text'),
                ('title', TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), 'reviews.title'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Pipeline with SVM
        self.model = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', SVC(
                kernel=kernel,
                C=C,
                gamma='scale',
                class_weight='balanced',
                probability=True,  # Enable probability estimates
                random_state=42,
                verbose=1  # Show training progress
            ))
        ])

        print("Training SVM model...")
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"Training complete! Test Accuracy: {accuracy:.4f}")
        
        return accuracy

    def persistModel(self, model_path: str) -> bool:
        """Save trained model to disk"""
        if self.model is not None:
            try:
                joblib.dump(self.model, model_path)
                print(f"Model saved to {model_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        else:
            print("No model to save. Please train the model first.")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load trained model from disk"""
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found at {model_path}")
            return False

    def preprocess_data(self) -> None:
        """Preprocess training data"""
        if self.train_data.df is not None:
            self.train_data.remove_nulls(columns=['reviews.title', 'reviews.text', 'sentiment'])
            self.train_data.get_summary()
        else:
            print("Data not loaded. Cannot preprocess.")

    def analyze_sentiments(self) -> None:
        """Analyze sentiment distribution in training data"""
        if hasattr(self, "train_data") and self.train_data.df is not None:
            print("\n--- Sentiment Distribution After Preprocessing ---")
            print(self.train_data.df['sentiment'].value_counts())
            print("\nSentiment Proportions:")
            print(self.train_data.df['sentiment'].value_counts(normalize=True))
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
        """
        Predict sentiment for a single review
        
        Args:
            reviewText: The review text to analyze
            reviewTitle: Title of the review
            productBrand: Product brand
            productCategory: Product category
            primaryCategory: Primary category
            
        Returns:
            Predicted sentiment label
        """
        if self.model is not None:
            try:
                sample = pd.DataFrame([{
                    "reviews.text": reviewText,
                    "reviews.title": reviewTitle,
                    "brand": productBrand,
                    "categories": productCategory,
                    "primaryCategories": primaryCategory
                }])
                prediction = self.model.predict(sample)[0]
                
                # Get prediction probability if available
                try:
                    probabilities = self.model.predict_proba(sample)[0]
                    classes = self.model.classes_
                    confidence = max(probabilities)
                    print(f"Prediction confidence: {confidence:.4f}")
                except:
                    pass
                
                return prediction
            except Exception as e:
                print(f"Error during prediction: {e}")
                return "Error"
        else:
            print("Model not loaded. Please load or train the model first.")
            return "Model not available"

# Example usage
#sa = SentimentAnalysisSVM()
#acc = sa.train(
#    '../Capstone/Ecommerce-Sentiment-Analysis/Ecommerce_dataset/train_data.csv',
#    kernel='rbf',
#    C=1.0
#)
#print(f"Final Accuracy: {acc:.4f}")
#sa.persistModel('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_SVM.pkl')

#prediction = sa.predict(
#    "It's a great product for a thrift store, not for someone who wants a quality product.",
#    "Review of Amazon Echo Show",
#    "Amazon",
#    "Electronics",
#    "Smart Speakers"
#)
#print(f"Prediction: {prediction}")


loaded_model = SentimentAnalysisSVM('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_SVM.pkl')
print(loaded_model.predict("It's a great product for a thrift store, not for someone who wants a quality product.",
    "Review of Amazon Echo Show",
    "Amazon",
    "Electronics",
    "Smart Speakers"
))