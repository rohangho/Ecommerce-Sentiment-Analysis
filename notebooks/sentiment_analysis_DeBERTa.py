from __future__ import annotations

import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, create_optimizer

import CapstoneData


class SentimentAnalysisDeBERTa:
    """
    DeBERTa (microsoft/deberta-v3-base) sequence classifier fine-tuned on the project
    3-way labels: Negative, Neutral, Positive. Fresh 3-class head (ignore_mismatched_sizes).
    """

    def __init__(self, model_path: str | None = None):
        self.model_name = "microsoft/deberta-v3-base"
        self.max_len = 256
        self.batch_size = 4
        self.label_encoder = LabelEncoder()
        self.class_names: list[str] = ["Negative", "Neutral", "Positive"]
        self.model = None
        self.tokenizer = None
        self.train_data = None

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(e)
        else:
            print("GPU not found, using CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if model_path:
            self.load_model(model_path)

    def preprocess_data(self) -> None:
        if self.train_data is not None and self.train_data.df is not None:
            self.train_data.remove_nulls(
                columns=["reviews.title", "reviews.text", "sentiment"]
            )
            self.train_data.get_summary()
        else:
            print("Data not loaded.")

    def train(
        self,
        data,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.12,
        weight_decay: float = 0.02,
    ) -> float:
        """Fine-tune on downstream 3-class sentiment (reviews + titles)."""
        self.train_data = CapstoneData.DataExploration(data)
        self.preprocess_data()

        X = self.train_data.df[["reviews.text", "reviews.title"]]
        y = self.train_data.df["sentiment"]
        texts = X["reviews.text"].fillna("").astype(str) + " " + X[
            "reviews.title"
        ].fillna("").astype(str)

        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = list(self.label_encoder.classes_)

        texts_train, texts_val, y_train, y_val = train_test_split(
            texts,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )

        print(
            f"Fine-tuning DeBERTa: train={len(texts_train)}, val={len(texts_val)}, "
            f"labels={self.class_names}"
        )

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.class_names),
            from_pt=True,
            ignore_mismatched_sizes=True,
        )
        self.model.config.id2label = {
            i: name for i, name in enumerate(self.class_names)
        }
        self.model.config.label2id = {
            name: i for i, name in enumerate(self.class_names)
        }

        def get_tf_dataset(texts_series, labels, shuffle=False):
            enc = self.tokenizer(
                list(texts_series),
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="tf",
            )
            ds = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                    },
                    labels,
                )
            )
            if shuffle:
                ds = ds.shuffle(len(texts_series))
            return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = get_tf_dataset(texts_train.values, y_train, shuffle=True)
        val_ds = get_tf_dataset(texts_val.values, y_val)

        num_train_steps = len(train_ds) * epochs
        optimizer, _ = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=max(1, int(warmup_ratio * num_train_steps)),
            weight_decay_rate=weight_decay,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))

        print(f"Starting fine-tuning ({epochs} epochs)...")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weight_dict
        )
        _, acc = self.model.evaluate(val_ds, verbose=0)
        print(f"Validation accuracy: {acc:.4f}")
        return float(acc)

    def predict(
        self,
        reviewText: str,
        reviewTitle: str = "",
        productBrand: str = "Unknown",
        productCategory: str = "General",
        primaryCategory: str = "General",
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return "Model not available"

        full_text = (reviewTitle + " " + reviewText).strip() if reviewTitle else reviewText
        enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        out = self.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )
        idx = int(tf.argmax(out.logits, axis=1).numpy()[0])
        if idx < len(self.class_names):
            return self.class_names[idx]
        return self.model.config.id2label.get(idx, str(idx))

    def persistModel(self, model_path: str) -> bool:
        if self.model is None:
            print("No model to save.")
            return False
        try:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            model_dir = model_path.replace(".pkl", "_deberta_tf")
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            metadata = {
                "model_dir": model_dir,
                "class_names": self.class_names,
                "max_len": self.max_len,
                "label_encoder": self.label_encoder,
            }
            joblib.dump(metadata, model_path)
            print(f"Model saved to {model_dir} and metadata to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found at {model_path}")
                return False
            metadata = joblib.load(model_path)
            model_dir = metadata["model_dir"]
            self.class_names = metadata.get("class_names", ["Negative", "Neutral", "Positive"])
            self.max_len = metadata.get("max_len", 256)
            self.label_encoder = metadata.get("label_encoder", LabelEncoder())
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
