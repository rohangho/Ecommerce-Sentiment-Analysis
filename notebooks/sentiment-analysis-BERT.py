import pandas as pd
import os
import numpy as np
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import CapstoneData

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis with BERT"""
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalysisBERT:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.max_len = 256
        self.batch_size = 16
        
        if model_path:
            self.load_model(model_path)

    def train(self, train_path: str, epochs: int = 3, learning_rate: float = 2e-5) -> float:
        """Train BERT model for sentiment analysis"""
        print(f"Using device: {self.device}")
        
        self.train_data = CapstoneData.DataExploration(train_path)
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

        # Load pre-trained BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3,  # 3 classes: Negative, Neutral, Positive
            output_loading_info=False
        )
        self.model.to(self.device)
        print("BERT model loaded!")

        print("Creating datasets...")
        # Create datasets
        train_dataset = SentimentDataset(texts_train.values, y_train, self.tokenizer, self.max_len)
        test_dataset = SentimentDataset(texts_test.values, y_test, self.tokenizer, self.max_len)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)
        print(f"Dataloaders created! Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        # Setup optimizer and scheduler
        total_steps = len(train_loader) * epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc="Training", unit="batch")
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': correct / total
                })

            train_accuracy = correct / total
            train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

            # Evaluation
            print("Evaluating...")
            test_accuracy = self.evaluate(test_loader)
            print(f"Test Accuracy: {test_accuracy:.4f}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

        return best_accuracy

    def evaluate(self, data_loader) -> float:
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0

        progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix({'accuracy': correct / total})

        return correct / total

    def persistModel(self, model_path: str) -> bool:
        """Save model, tokenizer, and metadata"""
        if self.model is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

                # Save model
                model_dir = model_path.replace('.pkl', '_bert')
                self.model.save_pretrained(model_dir)
                print(f"Model saved to {model_dir}")

                # Save tokenizer
                self.tokenizer.save_pretrained(model_dir)
                print(f"Tokenizer saved to {model_dir}")

                # Save metadata
                metadata = {
                    'label_encoder': self.label_encoder,
                    'class_names': self.class_names,
                    'max_len': self.max_len,
                    'model_dir': model_dir
                }
                joblib.dump(metadata, model_path)
                print(f"Metadata saved to {model_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        else:
            print("No model to save. Please train the model first.")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load model, tokenizer, and metadata"""
        try:
            # Load metadata
            if os.path.exists(model_path):
                metadata = joblib.load(model_path)
                self.label_encoder = metadata['label_encoder']
                self.class_names = metadata['class_names']
                self.max_len = metadata['max_len']
                model_dir = metadata['model_dir']

                # Load model and tokenizer
                self.model = BertForSequenceClassification.from_pretrained(
                    model_dir,
                    output_loading_info=False
                )
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
                self.model.to(self.device)

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
            print("Data not loaded. Cannot preprocess.")

    def analyze_sentiments(self) -> None:
        """Analyze sentiment distribution"""
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
        """Predict sentiment for a single review"""
        if self.model is not None and self.tokenizer is not None:
            try:
                # Combine text
                combined_text = reviewText + ' ' + reviewTitle
                
                # Tokenize and encode
                encoding = self.tokenizer(
                    combined_text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Make prediction
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=1).item()

                return self.class_names[predicted_class]
            except Exception as e:
                print(f"Error during prediction: {e}")
                return "Error"
        else:
            print("Model not loaded. Please load or train the model first.")
            return "Model not available"

# Example usage
#sa = SentimentAnalysisBERT()
#acc = sa.train('../Capstone/Ecommerce-Sentiment-Analysis/Ecommerce_dataset/train_data.csv', epochs=3)
#print(f"Best Test Accuracy: {acc:.4f}")
#sa.persistModel('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_BERT.pkl')

#prediction = sa.predict(
#    "It's a great product for a thrift store, not for someone who wants a quality product.",
#    "Review of Amazon Echo Show",
#    "Amazon",
#    "Electronics",
#    "Smart Speakers"
#)
#print(f"Prediction: {prediction}")


loaded_model = SentimentAnalysisBERT('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_BERT.pkl')
print(loaded_model.predict("The product didn't work.", "Review of Amazon Echo Show", "Amazon", "Electronics", "Smart Speakers"))