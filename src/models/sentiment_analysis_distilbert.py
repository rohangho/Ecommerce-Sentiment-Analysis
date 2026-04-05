import os
import re
import ssl
import json
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import tensorflow as tf
from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    create_optimizer,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

from src.data import CapstoneData

class SentimentAnalysisDistilBERT:
    def __init__(self, model_path: Optional[str] = None):
        # TensorFlow Metal configuration for Apple Silicon
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU (Metal acceleration): {gpus}")
            except RuntimeError as e:
                print(f"GPU Configuration Error: {e}")
        else:
            print("GPU not found, using CPU.")

        self.model_name = 'distilbert-base-uncased'
        self.max_length = 256
        self.batch_size = 12
        self.label_names = ['Negative', 'Neutral', 'Positive']
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = None

        if model_path:
            self.load_model(model_path)

    def clean_text(self, text: str) -> str:
        """Light cleaning for BERT input."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)   # URLs
        text = re.sub(r'<.*?>', '', text)               # HTML tags
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # non-letters
        text = re.sub(r'\s+', ' ', text).strip()       # extra whitespace
        return text

    def get_tf_dataset(self, texts, labels=None, shuffle=False):
        """Convert texts and labels to a batched tf.data.Dataset."""
        encodings = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        dataset_dict = {
            'input_ids':      encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
        }
        
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dataset_dict, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
            
        if shuffle:
            dataset = dataset.shuffle(len(texts))
        
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def train(self, data, epochs: int = 4, learning_rate: float = 2e-5, warmup_ratio: float = 0.12) -> dict:
        """Full training pipeline for DistilBERT."""
        self.train_data = CapstoneData.DataExploration(data)
        self.preprocess_data()
        df = self.train_data.df

        df['clean_text'] = df['reviews.text'].apply(self.clean_text)
        df['reviews.title'] = df['reviews.title'].fillna('')
        df['combined_text'] = (df['reviews.title'].apply(self.clean_text) + ' ' + df['clean_text']).str.strip()
        df['label'] = df['sentiment'].map(self.label_map)

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['combined_text'].values,
            df['label'].values,
            test_size=0.15,
            random_state=42,
            stratify=df['label'].values
        )

        # Datasets
        train_ds = self.get_tf_dataset(train_texts, train_labels, shuffle=True)
        val_ds   = self.get_tf_dataset(val_texts,   val_labels)

        # from_pt=True: Hub ships safetensors; PyTorch+safetensors needed to materialize TF weights
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3, from_pt=True
        )

        # Optimizer
        num_train_steps = len(train_ds) * epochs
        optimizer, lr_schedule = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=max(1, int(warmup_ratio * num_train_steps)),
            weight_decay_rate=0.02,
        )

        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['label'].values),
            y=df['label'].values
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Train
        print(f"Starting training for {epochs} epochs...")
        history = self.model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs,
            class_weight=class_weight_dict
        )

        # Evaluate
        loss, accuracy = self.model.evaluate(val_ds)
        print(f"Final Validation Accuracy: {accuracy:.4f}")
        
        # Build training metadata package
        metadata = {
            "validation_accuracy": float(accuracy),
            "training_samples": len(train_texts),
            "validation_samples": len(val_texts),
            "epochs": epochs,
            "batch_size": self.batch_size,
            "learning_rate": learning_rate,
            "max_length": self.max_length,
            "class_weights": {self.label_names[k]: float(v) for k, v in class_weight_dict.items()},
            "training_history": history.history
        }
        return metadata

    def preprocess_data(self) -> None:
        """Preprocess training data — matches Logistic/SVM/BERT/RNN interface."""
        if self.train_data.df is not None:
            self.train_data.remove_nulls(columns=['reviews.title', 'reviews.text', 'sentiment'])
            self.train_data.get_summary()
        else:
            print("Data not loaded. Cannot preprocess.")

    def analyze_sentiments(self) -> None:
        """Analyze sentiment distribution in training data."""
        if hasattr(self, "train_data") and self.train_data.df is not None:
            print("\n--- Sentiment Distribution ---")
            print(self.train_data.df['sentiment'].value_counts())
        else:
            print("Data not loaded.")

    def predict(self, reviewText: str, reviewTitle: str = "") -> str:
        """Predict sentiment for a single review."""
        if self.model is None:
            return "Model not loaded"

        combined_text = (self.clean_text(reviewTitle) + ' ' + self.clean_text(reviewText)).strip()
        encodings = self.tokenizer(
            [combined_text],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        dataset_dict = {
            'input_ids':      encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
        }

        logits = self.model.predict(dataset_dict, verbose=0).logits
        predicted_class = tf.argmax(logits, axis=1).numpy()[0]
        return self.label_names[predicted_class]

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[str]:
        """Predict sentiments for a list of texts in batches."""
        if self.model is None or self.tokenizer is None:
            return ["Model not loaded"] * len(texts)

        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
            logits = self.model.predict(
                {
                    'input_ids':      encodings['input_ids'],
                    'attention_mask': encodings['attention_mask'],
                },
                verbose=0
            ).logits
            preds = tf.argmax(logits, axis=1).numpy()
            all_preds.extend([self.label_names[p] for p in preds])
        return all_preds

    def persistModel(self, model_path: str) -> bool:
        """Save model, tokenizer, and metadata."""
        if self.model is not None:
            try:
                os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
                model_dir = model_path.replace('.pkl', '_distilbert_tf')
                self.model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)
                
                metadata = {
                    'model_dir': model_dir,
                    'max_length': self.max_length,
                    'label_names': self.label_names
                }
                joblib.dump(metadata, model_path)
                print(f"Model saved to {model_dir}")
                return True
            except Exception as e:
                print(f"Save Error: {e}")
                return False
        return False

    def load_model(self, model_path: str) -> bool:
        """Load model from disk."""
        try:
            if os.path.exists(model_path):
                metadata = joblib.load(model_path)
                model_dir = metadata['model_dir']
                self.max_length = metadata.get('max_length', 256)
                self.label_names = metadata.get('label_names', ['Negative', 'Neutral', 'Positive'])

                self.model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
                print(f"Model loaded from {model_dir}")
                return True
            return False
        except Exception as e:
            print(f"Load Error: {e}")
            return False

if __name__ == "__main__":
    # Example: How to use the model WITHOUT retraining
    # 1. Initialize with the path to the saved metadata .pkl file
    model_path = 'sentiment_model_distilbert.pkl'
    
    if os.path.exists(model_path):
        sa = SentimentAnalysisDistilBERT(model_path=model_path)
        print("Model loaded successfully!")
        
        # 2. Make predictions
        text = "This product is absolutely amazing, I love it!"
        title = "Best purchase ever"
        prediction = sa.predict(text, title)
        print(f"Review: {text}")
        print(f"Prediction: {prediction}")
    else:
        print(f"Model file {model_path} not found. You may need to train it first using sa.train().")
        
        # Example Training (if you want to train from scratch):
        # sa = SentimentAnalysisDistilBERT()
        # sa.train('Ecommerce_dataset/train_data.csv', epochs=1)
        # sa.persistModel('sentiment_model_distilbert.pkl')

