import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os
import joblib

class SentimentAnalysisHF:
    def __init__(self, model_path: str = None):
        model_name = "siebert/sentiment-roberta-large-english" 
        
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_path:
            self.load_model(model_path)
        else:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, reviewText: str, reviewTitle: str = "", productBrand: str = "Unknown",
                productCategory: str = "General", primaryCategory: str = "General") -> str:
        # Combine text and title if provided
        full_text = reviewText
        if reviewTitle:
            full_text = reviewTitle + " " + reviewText

        # Tokenize input
        inputs = self.tokenizer(full_text, return_tensors="tf", truncation=True, padding=True)

        # Run model
        outputs = self.model(inputs)
        logits = outputs.logits
        predicted_class_id = tf.argmax(logits, axis=1).numpy()[0]

        # Map to label (POSITIVE/NEGATIVE)
        return self.model.config.id2label[predicted_class_id]

    def persistModel(self, model_path: str) -> bool:
        if self.model is not None:
            # For TF models in transformers, save_pretrained is preferred over joblib
            model_dir = model_path.replace('.pkl', '_hf_tf')
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            
            # Save metadata
            metadata = {'model_dir': model_dir}
            joblib.dump(metadata, model_path)
            print(f"Model saved to {model_dir} and metadata to {model_path}")
            return True
        else:
            print("No model to save.")
            return False

    def load_model(self, model_path: str) -> bool:
        try:
            if os.path.exists(model_path):
                metadata = joblib.load(model_path)
                model_dir = metadata['model_dir']
                self.model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                print(f"Model loaded from {model_dir}")
                return True
            else:
                print(f"Model file not found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Example usage (commented out as in original)
# sa = SentimentAnalysisHF()
# sa.persistModel('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_pre_trained_model.pkl')
