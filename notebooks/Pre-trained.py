import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import joblib

class SentimentAnalysisHF:
    def __init__(self, model_path: str = None):
        model_name="siebert/sentiment-roberta-large-english" 
        device=None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_path:
            self.load_model(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def predict(self, text: str) -> str:
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        # Map to label
        return self.model.config.id2label[predicted_class_id]

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

    def predict(self, reviewText: str, reviewTitle: str = "", productBrand: str = "Unknown",
                productCategory: str = "General", primaryCategory: str = "General") -> str:
        # Combine text and title if provided
        full_text = reviewText
        if reviewTitle:
            full_text = reviewTitle + " " + reviewText

        # Tokenize input
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        # Map to label (POSITIVE/NEGATIVE)
        return self.model.config.id2label[predicted_class_id]


# Example usage
#os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
#sa = SentimentAnalysisHF()
#sa.persistModel('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_pre_trained_model.pkl')
#sa = SentimentAnalysisHF('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_pre_trained_model.pkl')
#print(sa.predict("Worst product ever!"))   # → NEGATIVE
#print(sa.predict("This is a great product for a thrift shop, not for someone who wants a good quality product."))  # → NEGATIVE
#print(sa.predict("Product is okay."))   # → NEUTRAL

