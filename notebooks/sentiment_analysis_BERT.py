import pandas as pd
import os
import numpy as np
from typing import Optional, Tuple
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from . import CapstoneData

class SentimentAnalysisBERT:
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
        self.label_encoder = LabelEncoder()
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.max_len = 256
        self.batch_size = 12
        
        if model_path:
            self.load_model(model_path)

    def train(
        self,
        data,
        epochs: int = 4,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.12,
    ) -> float:
        """Train BERT model for sentiment analysis using TensorFlow"""
        self.train_data = CapstoneData.DataExploration(data)
        self.preprocess_data()

        # Features and target
        X = self.train_data.df[['reviews.text', 'reviews.title']]
        y = self.train_data.df['sentiment']

        # Combine text fields
        texts = X['reviews.text'].fillna('').astype(str) + ' ' + X['reviews.title'].fillna('').astype(str)

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        texts_train, texts_test, y_train, y_test = train_test_split(
            texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training samples: {len(texts_train)}, Test samples: {len(texts_test)}")
        print("Loading BERT model...")

        # Load pre-trained BERT tokenizer and model (TF)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            from_pt=True,
        )
        print("BERT model loaded!")

        def get_tf_dataset(texts, labels, shuffle=False):
            encodings = self.tokenizer(
                list(texts),
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
            dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask']
                },
                labels
            ))
            if shuffle:
                dataset = dataset.shuffle(len(texts))
            return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset = get_tf_dataset(texts_train.values, y_train, shuffle=True)
        test_dataset = get_tf_dataset(texts_test.values, y_test)

        # Setup optimizer
        num_train_steps = len(train_dataset) * epochs
        optimizer, lr_schedule = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=max(1, int(warmup_ratio * num_train_steps)),
            weight_decay_rate=0.02,
        )

        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Training
        print("Starting training...")
        self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            class_weight=class_weight_dict
        )

        # Final evaluation
        print("Evaluating...")
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

    def persistModel(self, model_path: str) -> bool:
        """Save model, tokenizer, and metadata"""
        if self.model is not None:
            try:
                os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
                model_dir = model_path.replace('.pkl', '_bert_tf')
                self.model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)
                
                metadata = {
                    'label_encoder': self.label_encoder,
                    'class_names': self.class_names,
                    'max_len': self.max_len,
                    'model_dir': model_dir
                }
                joblib.dump(metadata, model_path)
                print(f"Model and metadata saved to {model_dir} and {model_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        else:
            print("No model to save.")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load model, tokenizer, and metadata"""
        try:
            if os.path.exists(model_path):
                metadata = joblib.load(model_path)
                self.label_encoder = metadata['label_encoder']
                self.class_names = metadata['class_names']
                self.max_len = metadata['max_len']
                model_dir = metadata['model_dir']

                self.model = TFBertForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
                print(f"Model loaded from {model_dir}")
                return True
            else:
                print(f"Model metadata file not found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_data(self) -> None:
        """Preprocess training data"""
        if self.train_data.df is not None:
            self.train_data.remove_nulls(columns=['reviews.title', 'reviews.text', 'sentiment'])
            self.train_data.get_summary()
        else:
            print("Data not loaded.")

    def analyze_sentiments(self) -> None:
        """Analyze sentiment distribution"""
        if hasattr(self, "train_data") and self.train_data.df is not None:
            print("\n--- Sentiment Distribution ---")
            print(self.train_data.df['sentiment'].value_counts())
        else:
            print("Data not loaded.")

    def predict(
            self,
            reviewText: str,
            reviewTitle: str = "",
            productBrand: str = "Unknown",
            productCategory: str = "General",
            primaryCategory: str = "General"
        ) -> str:
        """Predict sentiment for a single review using TensorFlow"""
        if self.model is not None and self.tokenizer is not None:
            try:
                combined_text = reviewText + ' ' + reviewTitle
                encoding = self.tokenizer(
                    combined_text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='tf'
                )

                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_class = tf.argmax(logits, axis=1).numpy()[0]

                return self.class_names[predicted_class]
            except Exception as e:
                print(f"Error during prediction: {e}")
                return "Error"
        else:
            return "Model not available"

if __name__ == "__main__":
    # Example: How to use the model WITHOUT retraining
    # 1. Provide the path to your metadata .pkl file
    model_path = 'sentiment_model_BERT.pkl'
    
    if os.path.exists(model_path):
        sa = SentimentAnalysisBERT(model_path=model_path)
        print("Model loaded successfully!")
        
        # 2. Predict sentiment for a new review
        review_text = "The product arrived broken and customer service was not helpful."
        review_title = "Very disappointed"
        
        prediction = sa.predict(review_text, review_title)
        print(f"Review: {review_text}")
        print(f"Prediction: {prediction}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        
        # Example Training (if starting from scratch):
        # sa = SentimentAnalysisBERT()
        # sa.train('Ecommerce_dataset/train_data.csv', epochs=3)
        # sa.persistModel('sentiment_model_BERT.pkl')

