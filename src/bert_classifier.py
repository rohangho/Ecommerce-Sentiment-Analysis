import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score,
                              precision_score, recall_score)
import pandas as pd


CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_MAP   = {"Negative": 0, "Neutral": 1, "Positive": 2}


class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by linear decay to zero."""
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int):
        self.peak_lr      = float(peak_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / self.warmup_steps)
        decay  = self.peak_lr * tf.maximum(
            0.0,
            1.0 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        )
        return tf.where(step < self.warmup_steps, warmup, decay)

    def get_config(self):
        return {
            "peak_lr":      self.peak_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


class BertSentimentClassifier:
    """
    BERT-base-uncased fine-tuned for 3-class sentiment classification.

    Usage:
        clf = BertSentimentClassifier()
        clf.build(total_steps=625, warmup_steps=62)
        clf.train(dataset, epochs=5, class_weight={0:14.3, 1:8.4, 2:0.36})
        clf.save()                    # saves weights to models/bert/
        # ── Later ──────────────────────────────────────────────────────────
        clf = BertSentimentClassifier()
        clf.load()                    # rebuilds arch + loads weights
        results = clf.evaluate(tokenizer, test_df)
    """

    MODEL_ID   = "bert-base-uncased"
    MAX_LEN    = 248
    BATCH_SIZE = 32
    WEIGHTS_DIR = "models/bert"

    def __init__(self):
        self.model     = None
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_ID)

    # ── Build ──────────────────────────────────────────────────────────────
    def build(self, total_steps: int, warmup_steps: int, peak_lr: float = 2e-5):
        """Construct the Keras model graph."""
        lr_schedule = WarmupLinearDecay(peak_lr, total_steps, warmup_steps)

        with tf.device("/GPU:0"):
            bert = TFBertModel.from_pretrained(self.MODEL_ID, use_safetensors=False)

            input_ids_in      = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_ids")
            attention_mask_in = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="attention_mask")

            bert_out   = bert({"input_ids": input_ids_in, "attention_mask": attention_mask_in})
            cls_vector = bert_out.pooler_output           # (batch, 768)

            x      = tf.keras.layers.Dense(64,  activation="relu")(cls_vector)
            x      = tf.keras.layers.Dropout(0.3)(x)
            x      = tf.keras.layers.Dense(128, activation="relu")(x)
            x      = tf.keras.layers.Dropout(0.3)(x)
            output = tf.keras.layers.Dense(3,   activation="softmax")(x)

            self.model = tf.keras.Model(
                inputs=[input_ids_in, attention_mask_in], outputs=output
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        print(f"[BertSentimentClassifier] Model built — {self.model.count_params():,} params")
        return self

    # ── Tokenise ───────────────────────────────────────────────────────────
    def tokenize(self, texts: list) -> dict:
        return self.tokenizer(
            texts,
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, dataset, epochs: int, class_weight: dict,
              early_stop_patience: int = 2):
        assert self.model is not None, "Call build() before train()"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=early_stop_patience,
                restore_best_weights=True, verbose=1
            )
        ]
        history = self.model.fit(
            dataset, epochs=epochs,
            class_weight=class_weight,
            callbacks=callbacks
        )
        return history

    # ── Save / Load ────────────────────────────────────────────────────────
    def save(self, path: str = None):
        save_path = path or self.WEIGHTS_DIR
        os.makedirs(save_path, exist_ok=True)
        self.model.save_weights(os.path.join(save_path, "weights.h5"))
        print(f"[BertSentimentClassifier] Weights saved → {save_path}/weights.h5")

    def load(self, path: str = None, total_steps: int = 625, warmup_steps: int = 62):
        """Rebuild the architecture then load saved weights."""
        load_path = path or self.WEIGHTS_DIR
        weights_file = os.path.join(load_path, "weights.h5")
        assert os.path.exists(weights_file), \
            f"No saved weights found at {weights_file}. Run train() and save() first."
        self.build(total_steps=total_steps, warmup_steps=warmup_steps)
        self.model.load_weights(weights_file)
        print(f"[BertSentimentClassifier] Weights loaded ← {weights_file}")
        return self

    # ── Predict & Evaluate ─────────────────────────────────────────────────
    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 32) -> dict:
        """Tokenise, predict, and return a metrics dict + print a report."""
        assert self.model is not None, "Model not built/loaded yet."

        y_true = test_df["sentiment"].map(LABEL_MAP).values
        enc    = self.tokenize(test_df["reviews.text"].tolist())

        ds = tf.data.Dataset.from_tensor_slices({
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        probs  = self.model.predict(ds)
        y_pred = np.argmax(probs, axis=1)

        acc         = accuracy_score(y_true, y_pred)
        macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        per_f1      = f1_score(y_true, y_pred, average=None,       zero_division=0)
        per_prec    = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_rec     = recall_score(y_true, y_pred, average=None,   zero_division=0)

        print(f"\n{'='*55}")
        print(f"  BERT — Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"{'='*55}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        return {
            "model":        "BERT",
            "accuracy":     acc,
            "macro_f1":     macro_f1,
            "weighted_f1":  weighted_f1,
            "per_class_f1": dict(zip(CLASS_NAMES, per_f1)),
            "per_class_precision": dict(zip(CLASS_NAMES, per_prec)),
            "per_class_recall":    dict(zip(CLASS_NAMES, per_rec)),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    # ── Prepare tf.data pipeline from DataFrame ────────────────────────────
    @staticmethod
    def make_dataset(texts: list, labels: list,
                     tokenizer, max_len: int = 248,
                     batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        enc           = tokenizer(texts, max_length=max_len, padding="max_length",
                                  truncation=True, return_tensors="tf")
        labels_tensor = tf.cast(tf.convert_to_tensor(labels), tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
            labels_tensor
        ))
        if shuffle:
            ds = ds.shuffle(1000, seed=42)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
