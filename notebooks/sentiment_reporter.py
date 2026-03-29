import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from wordcloud import WordCloud
try:
    from IPython.display import Image, display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

# Import all model classes
from sentiment_analysis_Logistic import SentimentAnalysis
from sentiment_analysis_SVM import SentimentAnalysisSVM
from sentiment_analysis_RNN import SentimentAnalysisRNN
from sentiment_analysis_distilbert import SentimentAnalysisDistilBERT
from sentiment_analysis_BERT import SentimentAnalysisBERT
from sentiment_analysis_Roberta import SentimentAnalysisHF
from sentiment_analysis_DeBERTa import SentimentAnalysisDeBERTa

class SentimentReporter:
    def __init__(self, output_dir="personal_update/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = ['Negative', 'Neutral', 'Positive']
        self.model_order = ["Logistic", "SVM", "RNN", "DistilBERT", "BERT", "RoBERTa", "DeBERTa"]

    def _predict_bert_batch(self, sa, df):
        combined = df["reviews.text"].fillna("").astype(str) + " " + df["reviews.title"].fillna("").astype(str)
        texts = list(combined)
        labels = []
        bs = sa.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = sa.tokenizer(list(chunk), add_special_tokens=True, max_length=sa.max_len, padding="max_length", truncation=True, return_tensors="tf")
            logits = sa.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            pred = tf.argmax(logits, axis=1).numpy()
            labels.extend(sa.class_names[j] for j in pred)
        return labels

    def _predict_hf_batch(self, sa, df):
        combined = df["reviews.text"].fillna("").astype(str) + " " + df["reviews.title"].fillna("").astype(str)
        texts = list(combined)
        labels = []
        bs = sa.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = sa.tokenizer(list(chunk), add_special_tokens=True, max_length=sa.max_len, padding="max_length", truncation=True, return_tensors="tf")
            logits = sa.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            pred = tf.argmax(logits, axis=1).numpy()
            labels.extend(sa.class_names[int(j)] for j in pred)
        return labels

    def gather_all_predictions(self, df, model_dir="deploy_models"):
        """Scans for available models and returns predicted labels for each."""
        p = Path(model_dir)
        preds = {}
        X_df = df[["reviews.text", "reviews.title"]].copy().fillna("Unknown")

        # 1. Logistic
        lr_path = p / "sentiment_model_TFD_LR.pkl"
        if lr_path.exists():
            print("Gathering predictions: Logistic...")
            lr = SentimentAnalysis(model_path=str(lr_path))
            # Pipeline expects a DataFrame with reviews.text, reviews.title, etc.
            preds["Logistic"] = lr.model.predict(df).tolist()

        # 2. SVM
        svm_path = p / "sentiment_model_SVM.pkl"
        if svm_path.exists():
            print("Gathering predictions: SVM...")
            svm = SentimentAnalysisSVM(model_path=str(svm_path))
            preds["SVM"] = svm.model.predict(df).tolist()

        # 3. RNN
        rnn_path = p / "sentiment_model_RNN.pkl"
        if rnn_path.exists():
            print("Gathering predictions: RNN...")
            rnn = SentimentAnalysisRNN(model_path=str(rnn_path))
            # RNN predict expects individual row fields
            # We'll do a batch-like approach
            results = []
            for _, r in df.iterrows():
                results.append(rnn.predict(str(r.get("reviews.text", "")), str(r.get("reviews.title", ""))))
            preds["RNN"] = results

        # 4. DistilBERT
        db_path = p / "sentiment_model_distilbert.pkl"
        if db_path.exists():
            print("Gathering predictions: DistilBERT...")
            db = SentimentAnalysisDistilBERT(model_path=str(db_path))
            combined = (df['reviews.title'].fillna('') + ' ' + df['reviews.text'].fillna('')).tolist()
            preds["DistilBERT"] = db.predict_batch(combined)

        # 5. BERT
        bert_path = p / "sentiment_model_BERT.pkl"
        if bert_path.exists():
            print("Gathering predictions: BERT...")
            bert = SentimentAnalysisBERT(model_path=str(bert_path))
            preds["BERT"] = self._predict_bert_batch(bert, df)

        # 6. RoBERTa
        rob_path = p / "sentiment_model_roberta.pkl"
        if rob_path.exists():
            print("Gathering predictions: RoBERTa...")
            rob = SentimentAnalysisHF(model_path=str(rob_path))
            preds["RoBERTa"] = self._predict_hf_batch(rob, df)

        # 7. DeBERTa
        deb_path = p / "sentiment_model_deberta.pkl"
        if deb_path.exists():
            print("Gathering predictions: DeBERTa...")
            deb = SentimentAnalysisDeBERTa(model_path=str(deb_path))
            preds["DeBERTa"] = self._predict_hf_batch(deb, df)

        return preds

    def generate_comparison_reports(self, y_true, preds_by_model):
        """Generates unified multi-model performance reports."""
        model_names = [m for m in self.model_order if m in preds_by_model]
        if not model_names: return

        rows = []
        for name in model_names:
            y_p = np.array(preds_by_model[name])
            rows.append({
                "Model": name,
                "Accuracy": (y_true == y_p).mean(),
                "Weighted F1": f1_score(y_true, y_p, average="weighted", zero_division=0),
                "Negative F1": f1_score(y_true, y_p, labels=["Negative"], average="macro", zero_division=0),
                "Neutral F1": f1_score(y_true, y_p, labels=["Neutral"], average="macro", zero_division=0),
                "Positive F1": f1_score(y_true, y_p, labels=["Positive"], average="macro", zero_division=0)
            })
        summary_df = pd.DataFrame(rows)

        # 1. Heatmap
        heat_data = summary_df.set_index("Model")[["Negative F1", "Neutral F1", "Positive F1"]]
        heat_data.columns = ["Negative", "Neutral", "Positive"]
        plt.figure(figsize=(10, len(model_names) * 0.6 + 2))
        sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1)
        plt.title("F1 Score Heatmap — Models vs Classes")
        plt.tight_layout()
        plt.savefig(self.output_dir / "report_multi_f1_heatmap.png", dpi=150)
        plt.close()

        # 2. Accuracy Bars
        plt.figure(figsize=(max(8, len(model_names) * 1.2), 6))
        sns.barplot(data=summary_df, x="Model", y="Weighted F1", palette="Blues_d")
        plt.title("Model Comparison — Weighted F1 Score")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / "report_multi_accuracy_bars.png", dpi=150)
        plt.close()

        # 3. Confusion Matrix Grid
        n = len(model_names)
        cols = 3
        rows_grid = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 4, rows_grid * 3.5))
        axes = np.atleast_1d(axes).ravel()
        for idx, name in enumerate(model_names):
            cm = confusion_matrix(y_true, preds_by_model[name], labels=self.label_names)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=self.label_names, yticklabels=self.label_names, ax=axes[idx], cbar=False)
            axes[idx].set_title(f"{name}")
        for midx in range(idx + 1, len(axes)): axes[midx].axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "report_multi_confusion_grid.png", dpi=150)
        plt.close()

        summary_df.to_csv(self.output_dir / "metrics_summary.csv", index=False)

    def plot_wordclouds(self, df):
        """Generates word clouds for each sentiment class."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, sentiment in enumerate(self.label_names):
            text = " ".join(df[df['sentiment'] == sentiment]['reviews.text'].fillna('').astype(str))
            if text.strip():
                wc = WordCloud(width=400, height=400, background_color='white').generate(text)
                axes[i].imshow(wc, interpolation='bilinear')
                axes[i].set_title(f"{sentiment} Reviews")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "report_wordclouds.png", dpi=150)
        plt.close()

    def plot_aspect_analysis(self, df):
        """Simulated aspect-based analysis (e.g., Performance, Service, Value)."""
        aspects = {
            'Performance': ['fast', 'slow', 'battery', 'power', 'speed', 'quality'],
            'Service': ['shipping', 'delivery', 'customer', 'support', 'help', 'service'],
            'Value': ['price', 'cheap', 'expensive', 'worth', 'money', 'value']
        }
        results = []
        df_text = df['reviews.text'].fillna('').str.lower()
        for aspect, keywords in aspects.items():
            mask = df_text.apply(lambda x: any(k in x for k in keywords))
            sub = df[mask]
            if not sub.empty:
                counts = sub['sentiment'].value_counts(normalize=True).to_dict()
                for s in self.label_names:
                    results.append({'Aspect': aspect, 'Sentiment': s, 'Percentage': counts.get(s, 0)})
        
        if results:
            aspect_df = pd.DataFrame(results)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=aspect_df, x='Aspect', y='Percentage', hue='Sentiment', palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
            plt.title("Aspect-Based Sentiment Analysis")
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(self.output_dir / "report_aspect_analysis.png", dpi=150)
            plt.close()

    def generate_visualizations(self, df, model_dir="deploy_models"):
        """The main entry point called from notebooks."""
        print("\n--- Generating Comprehensive Sentiment Reports ---")
        
        # Ensure text columns are strings and non-null for ML pipelines and WordClouds
        df = df.copy()
        if 'reviews.text' in df.columns:
            df['reviews.text'] = df['reviews.text'].fillna('').astype(str)
        if 'reviews.title' in df.columns:
            df['reviews.title'] = df['reviews.title'].fillna('').astype(str)

        y_true = df["sentiment"].values if "sentiment" in df.columns else None
        
        # 1. Prediction comparison
        preds = self.gather_all_predictions(df, model_dir)
        if y_true is not None:
            self.generate_comparison_reports(y_true, preds)
        
        # 2. Exploratory Analytics
        self.plot_wordclouds(df)
        self.plot_aspect_analysis(df)
        
        # 3. Display in Notebook
        self._display_all_reports()

    def _display_all_reports(self):
        """Displays saved PNG files in a Jupyter Notebook."""
        if not _HAS_IPYTHON:
            print(f"Visualizations saved to {self.output_dir}")
            return

        image_files = [
            "report_multi_f1_heatmap.png",
            "report_multi_accuracy_bars.png",
            "report_multi_confusion_grid.png",
            "report_wordclouds.png",
            "report_aspect_analysis.png"
        ]
        
        for img in image_files:
            path = self.output_dir / img
            if path.exists():
                print(f"\nDisplaying: {img}")
                display(Image(filename=str(path)))

    def run_analysis(self, csv_path, model_dir="deploy_models"):
        df = pd.read_csv(csv_path)
        self.generate_visualizations(df, model_dir=model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--multi_model_dir", default="deploy_models")
    parser.add_argument("--output", default="personal_update/outputs")
    args = parser.parse_args()
    
    reporter = SentimentReporter(output_dir=args.output)
    reporter.run_analysis(args.csv, model_dir=args.multi_model_dir)
