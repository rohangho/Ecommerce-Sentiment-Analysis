import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Import our model class
from notebooks.sentiment_analysis_distilbert import SentimentAnalysisDistilBERT

class SentimentReporter:
    def __init__(self, output_dir="personal_update/outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Aspect Keywords from Section 12
        self.ASPECT_KEYWORDS = {
            'Price/Value':       ['price','cost','expensive','cheap','value','money','worth','deal','affordable','budget','sale','bargain'],
            'Screen/Display':    ['screen','display','resolution','bright','hd','visual','color','picture','pixel','view'],
            'Battery/Power':     ['battery','charge','charging','power','last','outlet','plug','cord','usb'],
            'Sound/Audio':       ['sound','speaker','audio','volume','music','loud','bass','hear','listen','noise'],
            'Speed/Performance': ['speed','fast','slow','lag','performance','quick','responsive','processor','ram','memory'],
            'Build/Design':      ['build','quality','durable','sturdy','design','weight','light','heavy','size','compact','portable'],
            'Ease of Use':       ['easy','simple','intuitive','user-friendly','setup','navigate','interface','learn','beginner','convenient'],
            'Apps/Software':     ['app','apps','software','store','download','install','update','google','play','alexa','skill'],
            'Camera':            ['camera','photo','picture','video','record','selfie'],
            'Kids/Family':       ['kid','kids','child','children','son','daughter','grandkid','family','parent','parental','toddler'],
        }

    def categorize_product(self, name):
        """Map product names to categories."""
        n = str(name).lower()
        if 'echo show' in n:   return 'Echo Show'
        if 'echo plus' in n:   return 'Echo Plus'
        if 'tap' in n:         return 'Amazon Tap'
        if 'fire kids' in n:   return 'Fire Kids Tablet'
        if 'fire hd 10' in n:  return 'Fire HD 10'
        if 'fire hd 8' in n or 'fire hd8' in n: return 'Fire HD 8'
        if 'fire' in n and 'tablet' in n:        return 'Fire 7 Tablet'
        if 'oasis' in n:       return 'Kindle Oasis'
        if 'voyage' in n:      return 'Kindle Voyage'
        if 'kindle' in n:      return 'Kindle E-reader'
        if 'fire tv' in n:     return 'Fire TV'
        return 'Other'

    def get_product_type(self, cat):
        """Group categories into types."""
        if cat in ['Echo Show','Echo Plus','Amazon Tap']: return 'Smart Speakers'
        if cat in ['Fire HD 8','Fire HD 10','Fire 7 Tablet','Fire Kids Tablet']: return 'Tablets'
        if cat in ['Kindle E-reader','Kindle Voyage','Kindle Oasis']: return 'E-Readers'
        return 'Other'

    def extract_aspects(self, text):
        """Identify aspects in text."""
        t = str(text).lower()
        return [asp for asp, kws in self.ASPECT_KEYWORDS.items() if any(kw in t for kw in kws)]

    def generate_visualizations(self, df):
        """Run sections 14, 15, and 16 logic."""
        
        # --- Section 14: Overall Sentiment Visualizations ---
        print("Generating Section 14: Overall Sentiment Plots...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        sentiment_counts = df['sentiment'].value_counts()
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                    autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Overall Sentiment Distribution')

        df['text_length'] = df['reviews.text'].str.len()
        for sent, color in zip(['Positive','Neutral','Negative'], colors):
            if sent in df['sentiment'].values:
                subset = df[df['sentiment'] == sent]['text_length']
                axes[1].hist(subset, bins=50, alpha=0.6, label=sent, color=color)
        axes[1].set_title('Review Length Distribution')
        axes[1].set_xlabel('Length'); axes[1].set_ylabel('Count'); axes[1].legend()
        axes[1].set_xlim(0, 1000)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'report_sentiment_overview.png'), dpi=150)
        plt.close()

        # --- Section 15: Word Clouds ---
        print("Generating Section 15: Word Clouds...")
        try:
            from wordcloud import WordCloud
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            for idx, (sent, cmap) in enumerate([('Positive','Greens'), ('Neutral','Oranges'), ('Negative','Reds')]):
                if sent in df['sentiment'].values:
                    text_data = ' '.join(df[df['sentiment'] == sent]['reviews.text'].fillna('').astype(str))
                    if text_data.strip():
                        wc = WordCloud(width=600, height=300, background_color='white', colormap=cmap, max_words=80).generate(text_data)
                        axes[idx].imshow(wc, interpolation='bilinear')
                        axes[idx].set_title(f'{sent} Word Cloud')
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'report_word_clouds.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Word cloud error: {e}")

        # --- Section 16: Aspect × Product Type Heatmap ---
        print("Generating Section 16: Aspect Heatmap...")
        df['aspects'] = df['reviews.text'].apply(self.extract_aspects)
        df['product_category'] = df['name'].apply(self.categorize_product)
        df['product_type'] = df['product_category'].apply(self.get_product_type)
        
        ptypes = ['Tablets', 'Smart Speakers', 'E-Readers']
        aspect_data = []
        for pt in ptypes:
            subset = df[df['product_type'] == pt]
            for asp in self.ASPECT_KEYWORDS:
                mask = subset['aspects'].apply(lambda x: asp in x)
                asp_subset = subset[mask]
                pos_pct = (asp_subset['sentiment'] == 'Positive').mean() * 100 if len(asp_subset) > 0 else np.nan
                aspect_data.append({'Product Type': pt, 'Aspect': asp, 'Positive %': pos_pct})
        
        ap_df = pd.DataFrame(aspect_data)
        if not ap_df.empty:
            ap_pivot = ap_df.pivot(index='Aspect', columns='Product Type', values='Positive %')
            plt.figure(figsize=(10, 7))
            sns.heatmap(ap_pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=90, linewidths=0.5)
            plt.title('Positive Sentiment % — Aspect × Product Type')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'report_aspect_heatmap.png'), dpi=150)
            plt.close()

    def run_analysis(self, csv_path, model_path=None):
        """Load data, predict if needed, and plot."""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if 'sentiment' not in df.columns and model_path:
            print(f"Predicting sentiments using model {model_path}...")
            sa = SentimentAnalysisDistilBERT(model_path=model_path)
            df['sentiment'] = df.apply(lambda row: sa.predict(row['reviews.text'], row.get('reviews.title', '')), axis=1)
        elif 'sentiment' not in df.columns:
            print("Error: 'sentiment' column missing and no model provided for prediction.")
            return

        self.generate_visualizations(df)
        print(f"Analysis complete. Plots saved to {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sentiment Analysis Report")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--model", help="Path to saved model .pkl file (optional if CSV has labels)")
    parser.add_argument("--output", default="personal_update/outputs", help="Directory to save plots")
    
    args = parser.parse_args()
    
    reporter = SentimentReporter(output_dir=args.output)
    reporter.run_analysis(args.csv, args.model)
