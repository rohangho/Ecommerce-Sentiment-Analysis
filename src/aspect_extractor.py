import pandas as pd
from textblob import TextBlob
from collections import Counter, defaultdict
import logging

from src.config import STOP_ASPECTS

logger = logging.getLogger(__name__)

class AspectExtractor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df.dropna(subset=['reviews.text', 'sentiment', 'name'], inplace=True)
        logger.info(f"Data loaded, {len(self.df)} valid records found.")

    def _extract_noun_phrases(self, text: str) -> list:
        if not isinstance(text, str):
            return []
        blob = TextBlob(text)
        return [np.string.lower() for np in blob.noun_phrases if len(np.split()) <= 3]

    def extract_top_aspects(self, top_n: int = 20) -> dict:
        if self.df is None:
            self.load_data()

        logger.info("Extracting aspects from reviews... This might take a few moments.")
        
        # Structure: { sentiment: { aspect: {'count': int, 'products': { product_name: [review_texts] } } } }
        raw_aspects = {
            'Positive': defaultdict(lambda: {'count': 0, 'products': defaultdict(list)}),
            'Neutral': defaultdict(lambda: {'count': 0, 'products': defaultdict(list)}),
            'Negative': defaultdict(lambda: {'count': 0, 'products': defaultdict(list)}),
        }

        for _, row in self.df.iterrows():
            sentiment = row['sentiment']
            text = row['reviews.text']
            product_name = row['name']
            
            if pd.isna(text) or pd.isna(product_name):
                continue
                
            if sentiment in raw_aspects:
                aspects = self._extract_noun_phrases(text)
                for aspect in aspects:
                    raw_aspects[sentiment][aspect]['count'] += 1
                    # Store the review text (limit to avoid huge data)
                    if len(raw_aspects[sentiment][aspect]['products'][product_name]) < 3:
                        raw_aspects[sentiment][aspect]['products'][product_name].append(str(text)[:500])

        # Filter and structure the output
        filtered_aspects = {'Positive': [], 'Neutral': [], 'Negative': []}
        
        for sentiment in raw_aspects:
            for aspect, data in raw_aspects[sentiment].items():
                if aspect not in STOP_ASPECTS and len(aspect) > 2:
                    # Build product list with reviews (top 3 products)
                    products_with_reviews = []
                    sorted_products = sorted(data['products'].items(), key=lambda x: len(x[1]), reverse=True)[:3]
                    for prod_name, reviews in sorted_products:
                        products_with_reviews.append({
                            'name': prod_name,
                            'reviews': reviews
                        })
                    
                    filtered_aspects[sentiment].append({
                        'aspect': aspect,
                        'count': data['count'],
                        'products': products_with_reviews
                    })
            
            # Sort by frequency and get the top N
            filtered_aspects[sentiment] = sorted(
                filtered_aspects[sentiment], 
                key=lambda x: x['count'], 
                reverse=True
            )[:top_n]
            
        return filtered_aspects
