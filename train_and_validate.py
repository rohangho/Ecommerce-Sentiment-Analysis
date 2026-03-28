#!/usr/bin/env python3
"""
Train all sentiment models on the same cleaned DataFrame as MasterRunner (eda.df),
save deployable artifacts under deploy_models/, then validate on test_data_hidden.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

os.chdir(ROOT)

from CapstoneData import DataExploration
from sentiment_analysis_Logistic import SentimentAnalysis
from sentiment_analysis_SVM import SentimentAnalysisSVM
from sentiment_analysis_RNN import SentimentAnalysisRNN
from sentiment_analysis_distilbert import SentimentAnalysisDistilBERT
from sentiment_analysis_BERT import SentimentAnalysisBERT
from sentiment_analysis_Roberta import SentimentAnalysisHF
from sentiment_analysis_DeBERTa import SentimentAnalysisDeBERTa

DEPLOY_DIR = ROOT / "deploy_models"
VALIDATION_OUTPUT_DIR = ROOT / "validation_outputs"
TRAIN_CSV = ROOT / "Ecommerce_dataset" / "train_data.csv"
TEST_CSV = ROOT / "Ecommerce_dataset" / "test_data_hidden.csv"

# Fixed label order for matrices and per-class metrics (matches dataset strings).
CLASS_LABELS: tuple[str, ...] = ("Negative", "Neutral", "Positive")

MODEL_ORDER: tuple[str, ...] = (
    "Logistic",
    "SVM",
    "RNN",
    "DistilBERT",
    "BERT",
    "RoBERTa",
    "DeBERTa",
)


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print("GPUs available:", gpus)


def build_eda_df() -> pd.DataFrame:
    eda = DataExploration(str(TRAIN_CSV))
    eda.remove_nulls(columns=["reviews.text", "sentiment"])
    assert eda.df is not None
    print(f"Cleaned training DataFrame (eda.df equivalent): {len(eda.df)} rows.")
    return eda.df.copy()


def train_all_models(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # print("\n--- Training Logistic Regression ---")
    # lr = SentimentAnalysis()
    # results["Logistic"] = lr.train(df.copy())
    # lr.persistModel(str(out_dir / "sentiment_model_TFD_LR.pkl"))

    # print("\n--- Training SVM ---")
    # svm = SentimentAnalysisSVM()
    # results["SVM"] = svm.train(df.copy())
    # svm.persistModel(str(out_dir / "sentiment_model_SVM.pkl"))

    # print("\n--- Training RNN ---")
    # rnn = SentimentAnalysisRNN()
    # results["RNN"] = rnn.train(df.copy(), epochs=24, batch_size=32)
    # rnn.persistModel(str(out_dir / "sentiment_model_RNN.pkl"))

    # print("\n--- Training DistilBERT ---")
    # db = SentimentAnalysisDistilBERT()
    # results["DistilBERT"] = db.train(df.copy(), epochs=3, learning_rate=2e-5)
    # db.persistModel(str(out_dir / "sentiment_model_distilbert.pkl"))

    # print("\n--- Training BERT ---")
    # bert = SentimentAnalysisBERT()
    # results["BERT"] = bert.train(df.copy(), epochs=3, learning_rate=2e-5)
    # bert.persistModel(str(out_dir / "sentiment_model_BERT.pkl"))

    # print("\n--- Fine-tuning RoBERTa (siebert backbone, 3-class head) ---")
    # rob = SentimentAnalysisHF()
    # results["RoBERTa"] = rob.train(df.copy(), epochs=3, learning_rate=1.7e-5)
    # rob.persistModel(str(out_dir / "sentiment_model_roberta.pkl"))

    print("\n--- Fine-tuning DeBERTa (microsoft/deberta-v3-base, 3-class head) ---")
    deb = SentimentAnalysisDeBERTa()
    results["DeBERTa"] = deb.train(df.copy(), epochs=3, learning_rate=2e-5)
    deb.persistModel(str(out_dir / "sentiment_model_deberta.pkl"))

    print("\nTraining summary (validation split metrics where applicable):", results)
    return results


def prepare_test_df(path: Path) -> pd.DataFrame:
    test = pd.read_csv(path)
    test = test.dropna(subset=["reviews.text", "reviews.title", "sentiment"])
    for col in ["brand", "categories", "primaryCategories"]:
        if col in test.columns:
            test[col] = test[col].fillna("Unknown")
    test["reviews.title"] = test["reviews.title"].fillna("")
    test["reviews.text"] = test["reviews.text"].fillna("")
    return test


def predict_distilbert_batch(sa: SentimentAnalysisDistilBERT, df: pd.DataFrame) -> list:
    combined = (
        df["reviews.title"].fillna("").astype(str)
        + " "
        + df["reviews.text"].fillna("").astype(str)
    )
    texts = [sa.clean_text(t) for t in combined]
    labels = []
    bs = sa.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        enc = sa.tokenizer(
            chunk,
            max_length=sa.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        logits = sa.model.predict(
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
            verbose=0,
        ).logits
        pred = tf.argmax(logits, axis=1).numpy()
        labels.extend(sa.label_names[j] for j in pred)
    return labels


def predict_bert_batch(sa: SentimentAnalysisBERT, df: pd.DataFrame) -> list:
    combined = df["reviews.text"].fillna("").astype(str) + " " + df["reviews.title"].fillna("").astype(str)
    texts = list(combined)
    labels = []
    bs = sa.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        enc = sa.tokenizer(
            list(chunk),
            add_special_tokens=True,
            max_length=sa.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        logits = sa.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits
        pred = tf.argmax(logits, axis=1).numpy()
        labels.extend(sa.class_names[j] for j in pred)
    return labels


def predict_hf_transformer_batch(sa, df: pd.DataFrame) -> list:
    combined = (
        df["reviews.text"].fillna("").astype(str)
        + " "
        + df["reviews.title"].fillna("").astype(str)
    )
    texts = list(combined)
    labels: list = []
    bs = sa.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        enc = sa.tokenizer(
            list(chunk),
            add_special_tokens=True,
            max_length=sa.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        logits = sa.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits
        pred = tf.argmax(logits, axis=1).numpy()
        labels.extend(sa.class_names[int(j)] for j in pred)
    return labels


def gather_predictions(test_df: pd.DataFrame, out_dir: Path) -> dict[str, list[str]]:
    """Load each artifact under ``out_dir`` if present and return predicted labels."""
    feature_cols = ["brand", "categories", "primaryCategories", "reviews.text", "reviews.title"]
    X = test_df[feature_cols]
    preds: dict[str, list[str]] = {}

    lr_path = out_dir / "sentiment_model_TFD_LR.pkl"
    if lr_path.exists():
        lr = SentimentAnalysis(model_path=str(lr_path))
        preds["Logistic"] = list(lr.model.predict(X))

    svm_path = out_dir / "sentiment_model_SVM.pkl"
    if svm_path.exists():
        svm = SentimentAnalysisSVM(model_path=str(svm_path))
        preds["SVM"] = list(svm.model.predict(X))

    rnn_path = out_dir / "sentiment_model_RNN.pkl"
    if rnn_path.exists():
        rnn = SentimentAnalysisRNN(model_path=str(rnn_path))
        preds["RNN"] = [
            rnn.predict(
                str(row["reviews.text"]),
                str(row["reviews.title"]),
                str(row.get("brand", "Unknown")),
                str(row.get("categories", "General")),
                str(row.get("primaryCategories", "General")),
            )
            for _, row in test_df.iterrows()
        ]

    db_meta = out_dir / "sentiment_model_distilbert.pkl"
    if db_meta.exists():
        db = SentimentAnalysisDistilBERT(model_path=str(db_meta))
        preds["DistilBERT"] = predict_distilbert_batch(db, test_df)

    bert_meta = out_dir / "sentiment_model_BERT.pkl"
    if bert_meta.exists():
        bert = SentimentAnalysisBERT(model_path=str(bert_meta))
        preds["BERT"] = predict_bert_batch(bert, test_df)

    rob_meta = out_dir / "sentiment_model_roberta.pkl"
    if rob_meta.exists():
        rob = SentimentAnalysisHF(model_path=str(rob_meta))
        preds["RoBERTa"] = predict_hf_transformer_batch(rob, test_df)

    deb_meta = out_dir / "sentiment_model_deberta.pkl"
    if deb_meta.exists():
        deb = SentimentAnalysisDeBERTa(model_path=str(deb_meta))
        preds["DeBERTa"] = predict_hf_transformer_batch(deb, test_df)

    return preds


def _ordered_model_names(preds_by_model: dict[str, list[str]]) -> list[str]:
    return [m for m in MODEL_ORDER if m in preds_by_model]


def save_validation_plots_and_tables(
    y_true: np.ndarray,
    preds_by_model: dict[str, list[str]],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Write metrics CSV, confusion-matrix grid, macro metric comparison, and per-class F1 chart.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(CLASS_LABELS)
    rows: list[dict] = []
    model_names = _ordered_model_names(preds_by_model)

    for name in model_names:
        y_pred = np.array(preds_by_model[name])
        acc = accuracy_score(y_true, y_pred)
        p_macro = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        r_macro = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        f_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        f_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        p_c = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        r_c = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        f_c = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        row: dict = {
            "model": name,
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f_macro,
            "f1_weighted": f_weighted,
        }
        for j, lab in enumerate(labels):
            row[f"precision_{lab}"] = p_c[j]
            row[f"recall_{lab}"] = r_c[j]
            row[f"f1_{lab}"] = f_c[j]
        rows.append(row)

    summary = pd.DataFrame(rows)
    csv_path = output_dir / "metrics_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved metrics table: {csv_path.relative_to(ROOT)}")

    # --- Confusion matrices (counts + row-normalized) ---
    n = len(model_names)
    if n > 0:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
        axes1 = np.atleast_1d(axes1).ravel()
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
        axes2 = np.atleast_1d(axes2).ravel()

        for idx, name in enumerate(model_names):
            y_pred = np.array(preds_by_model[name])
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=axes1[idx],
                cbar=False,
            )
            axes1[idx].set_title(name)
            axes1[idx].set_ylabel("True")
            axes1[idx].set_xlabel("Predicted")

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Greens",
                xticklabels=labels,
                yticklabels=labels,
                ax=axes2[idx],
                vmin=0,
                vmax=1,
                cbar=False,
            )
            axes2[idx].set_title(f"{name} (row norm)")
            axes2[idx].set_ylabel("True")
            axes2[idx].set_xlabel("Predicted")

        for j in range(len(model_names), len(axes1)):
            axes1[j].set_visible(False)
            axes2[j].set_visible(False)

        fig1.tight_layout()
        p1 = output_dir / "confusion_matrices_counts.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig1)

        fig2.tight_layout()
        p2 = output_dir / "confusion_matrices_normalized.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {p1.relative_to(ROOT)}")
        print(f"Saved {p2.relative_to(ROOT)}")

        # --- Macro precision / recall / F1 bar comparison ---
        fig3, ax3 = plt.subplots(figsize=(max(8, 1.2 * n), 5))
        x = np.arange(n)
        w = 0.25
        ax3.bar(x - w, summary["precision_macro"], width=w, label="Precision (macro)")
        ax3.bar(x, summary["recall_macro"], width=w, label="Recall (macro)")
        ax3.bar(x + w, summary["f1_macro"], width=w, label="F1 (macro)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(summary["model"], rotation=25, ha="right")
        ax3.set_ylim(0, 1.05)
        ax3.legend()
        ax3.set_title("Model comparison — macro precision, recall, F1")
        ax3.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        p3 = output_dir / "macro_precision_recall_f1.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved {p3.relative_to(ROOT)}")

        # --- Per-class F1 heatmap (models × class) ---
        f1_cols = [f"f1_{lab}" for lab in labels]
        heat = summary.set_index("model")[f1_cols]
        heat.columns = labels
        fig4, ax4 = plt.subplots(figsize=(6, max(3, 0.45 * n)))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax4)
        ax4.set_title("F1 score by model and class")
        fig4.tight_layout()
        p4 = output_dir / "f1_per_class_heatmap.png"
        fig4.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved {p4.relative_to(ROOT)}")

        for metric, fname in (
            ("precision", "precision_per_class_heatmap.png"),
            ("recall", "recall_per_class_heatmap.png"),
        ):
            cols = [f"{metric}_{lab}" for lab in labels]
            h2 = summary.set_index("model")[cols]
            h2.columns = labels
            fig_h, ax_h = plt.subplots(figsize=(6, max(3, 0.45 * n)))
            sns.heatmap(h2, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax_h)
            ax_h.set_title(f"{metric.capitalize()} by model and class")
            fig_h.tight_layout()
            ph = output_dir / fname
            fig_h.savefig(ph, dpi=150, bbox_inches="tight")
            plt.close(fig_h)
            print(f"Saved {ph.relative_to(ROOT)}")

        # --- Accuracy bar chart ---
        fig5, ax5 = plt.subplots(figsize=(max(7, n * 1.0), 4.5))
        ax5.bar(summary["model"], summary["accuracy"], color="steelblue")
        ax5.set_ylim(0, 1.05)
        ax5.set_ylabel("Accuracy")
        ax5.set_title("Accuracy on hidden test set")
        ax5.tick_params(axis="x", rotation=25)
        ax5.grid(axis="y", alpha=0.3)
        fig5.tight_layout()
        p5 = output_dir / "accuracy_by_model.png"
        fig5.savefig(p5, dpi=150, bbox_inches="tight")
        plt.close(fig5)
        print(f"Saved {p5.relative_to(ROOT)}")

    return summary


def validate_models(
    test_df: pd.DataFrame,
    out_dir: Path,
    plot_output_dir: Path | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame | None]:
    """
    Evaluate all available deploy models, print reports, and save comparison plots.
    """
    y_true = test_df["sentiment"].values
    plot_dir = plot_output_dir if plot_output_dir is not None else VALIDATION_OUTPUT_DIR

    print("\n========== Validation on test_data_hidden.csv ==========")
    print(f"Rows: {len(test_df)}")

    preds_by_model = gather_predictions(test_df, out_dir)
    if not preds_by_model:
        print("No model artifacts found under", out_dir)
        return {}, None

    for name in _ordered_model_names(preds_by_model):
        y_pred = preds_by_model[name]
        print(f"\n[{name}]")
        print(classification_report(y_true, y_pred, labels=list(CLASS_LABELS), zero_division=0))
        print("Accuracy:", accuracy_score(y_true, y_pred))

    summary = save_validation_plots_and_tables(y_true, preds_by_model, plot_dir)
    return preds_by_model, summary


def validate_deploy_models_only(
    test_csv: Path | None = None,
    deploy_dir: Path | None = None,
    plot_output_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Load hidden test CSV and ``deploy_models`` artifacts; print metrics and write plots."""
    configure_gpu()
    path = test_csv or TEST_CSV
    dep = deploy_dir or DEPLOY_DIR
    test_df = prepare_test_df(path)
    _, summary = validate_models(test_df, dep, plot_output_dir=plot_output_dir)
    return summary


def run_pipeline(training_df: pd.DataFrame | None = None) -> dict:
    """
    Train all models, validate on the hidden test CSV, print artifact list.

    Parameters
    ----------
    training_df
        Cleaned training data (e.g. notebook ``eda.df``). If None, loads and
        cleans ``TRAIN_CSV`` the same way as the EDA section.
    """
    configure_gpu()
    df_train = training_df.copy() if training_df is not None else build_eda_df()
    results = train_all_models(df_train, DEPLOY_DIR)
    test_df = prepare_test_df(TEST_CSV)
    validate_models(test_df, DEPLOY_DIR)
    print("\nDeployable artifacts under:", DEPLOY_DIR)
    for p in sorted(DEPLOY_DIR.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(ROOT))
    return results


def main() -> None:
    run_pipeline(None)


if __name__ == "__main__":
    main()
