import pandas as pd
import os
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

import CapstoneData

class SentimentAnalysisRNN:
    def __init__(self, model_path: Optional[str] = None):
        # TensorFlow Metal configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(e)
        else:
            print("GPU not found, using CPU.")

        self.model = None
        self.tokenizer = None
        self.max_len = 200
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.categorical_encoders = {}
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.label_encoder = LabelEncoder()
        
        if model_path:
            self.load_model(model_path)

    def train(self, data, epochs: int = 10, batch_size: int = 32) -> float:
        self.train_data = CapstoneData.DataExploration(data)
        self.preprocess_data()

        # Features and target
        X = self.train_data.df[['brand', 'categories', 'primaryCategories', 'reviews.text', 'reviews.title']]
        y = self.train_data.df['sentiment']

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=3)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Prepare text data
        text_data = X_train['reviews.text'].fillna('').astype(str) + ' ' + X_train['reviews.title'].fillna('').astype(str)
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(text_data)

        X_train_seq = self.tokenizer.texts_to_sequences(text_data)
        X_train_padded = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')

        # Prepare test text data
        text_data_test = X_test['reviews.text'].fillna('').astype(str) + ' ' + X_test['reviews.title'].fillna('').astype(str)
        X_test_seq = self.tokenizer.texts_to_sequences(text_data_test)
        X_test_padded = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')

        # Encode categorical features
        categorical_features = ['brand', 'categories', 'primaryCategories']
        cat_features_train = []
        cat_features_test = []

        for i, feature in enumerate(categorical_features):
            if feature not in self.categorical_encoders:
                self.categorical_encoders[feature] = LabelEncoder()
                X_train[feature] = X_train[feature].fillna('Unknown')
                self.categorical_encoders[feature].fit(X_train[feature])

            X_test[feature] = X_test[feature].fillna('Unknown')
            cat_train_encoded = self.categorical_encoders[feature].transform(X_train[feature])
            cat_test_encoded = self.categorical_encoders[feature].transform(X_test[feature])

            cat_features_train.append(cat_train_encoded)
            cat_features_test.append(cat_test_encoded)

        cat_features_train = np.column_stack(cat_features_train)
        cat_features_test = np.column_stack(cat_features_test)

        # Build RNN model
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')  # 3 classes: Negative, Neutral, Positive
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        self.model.fit(
            X_train_padded, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

        # Evaluate on test set
        _, test_accuracy = self.model.evaluate(X_test_padded, y_test, verbose=0)
        return test_accuracy

    def persistModel(self, model_path: str) -> bool:
        if self.model is not None:
            # Save model
            self.model.save(model_path.replace('.pkl', '.keras'))
            
            # Save tokenizer and encoders
            joblib.dump({
                'tokenizer': self.tokenizer,
                'categorical_encoders': self.categorical_encoders,
                'label_encoder': self.label_encoder,
                'max_len': self.max_len,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'class_names': self.class_names
            }, model_path)
            
            print(f"Model saved to {model_path}")
            return True
        else:
            print("No model to save. Please train the model first.")
            return False

    def load_model(self, model_path: str) -> bool:
        try:
            # Load metadata
            if os.path.exists(model_path):
                metadata = joblib.load(model_path)
                self.tokenizer = metadata['tokenizer']
                self.categorical_encoders = metadata['categorical_encoders']
                self.label_encoder = metadata['label_encoder']
                self.max_len = metadata['max_len']
                self.vocab_size = metadata['vocab_size']
                self.embedding_dim = metadata['embedding_dim']
                self.class_names = metadata['class_names']

            # Load keras model
            keras_model_path = model_path.replace('.pkl', '.keras')
            if os.path.exists(keras_model_path):
                self.model = load_model(keras_model_path)
                print(f"Model loaded from {keras_model_path}")
                return True
            else:
                print(f"Keras model file not found at {keras_model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
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
            if self.model is not None and self.tokenizer is not None:
                try:
                    # Combine and tokenize text
                    combined_text = reviewText + ' ' + reviewTitle
                    text_seq = self.tokenizer.texts_to_sequences([combined_text])
                    text_padded = pad_sequences(text_seq, maxlen=self.max_len, padding='post')

                    # Make prediction
                    prediction = self.model.predict(text_padded, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class])

                    return self.class_names[predicted_class]
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    return "Error"
            else:
                print("Model not loaded. Please load or train the model first.")
                return "Model not available"

if __name__ == "__main__":
    # Example: How to use the model WITHOUT retraining
    # 1. Path to your saved metadata .pkl file
    model_path = 'sentiment_model_RNN.pkl'
    
    if os.path.exists(model_path):
        sa = SentimentAnalysisRNN(model_path=model_path)
        print("Model loaded successfully!")
        
        # 2. Predict sentiment for a new review
        review_text = "It's a great product for a thrift store, but not for quality lovers."
        review_title = "Average quality"
        
        prediction = sa.predict(review_text, review_title)
        print(f"Review: {review_text}")
        print(f"Prediction: {prediction}")
    else:
        print(f"Model file {model_path} not found. Train first using sa.train().")
        
        # Example Training (if starting from scratch):
        # sa = SentimentAnalysisRNN()
        # sa.train('Ecommerce_dataset/train_data.csv', epochs=10)
        # sa.persistModel('sentiment_model_RNN.pkl')