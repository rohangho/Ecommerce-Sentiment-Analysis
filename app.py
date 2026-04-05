from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import re
import json
import numpy as np

# Fix for hdbscan hanging on macOS (Apple Silicon especially)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import tensorflow as tf
from pathlib import Path
from bertopic import BERTopic

# Add src to sys.path for model class imports
# PYTHONPATH=/app exposes src.
from src.models.sentiment_analysis_DeBERTa import SentimentAnalysisDeBERTa

app = Flask(__name__)

# --- Configuration ---
DATA_PATH = "Ecommerce_dataset/combined_data_with_topics.csv"
TOPIC_INFO_PATH = "deploy_models/bertopic_info.csv"
MODEL_PKL = "deploy_models/sentiment_model_deberta.pkl"
BERTOPIC_MODEL_PATH = "deploy_models/bertopic_model"
CARDIFF_MODEL_PATH = "deploy_models/cardiffnlp_fallback"  # local cache — avoids HuggingFace download on startup

# Global objects for the app
sentiment_model = None
topic_model = None
df_data = None
df_topics = None
fallback_sentiment = None  # Cardiff RoBERTa fallback for low-confidence DeBERTa predictions
aspect_lookup = []  # loaded from seed_topics.json for multi-aspect extraction

SEED_TOPICS_FILE = "seed_topics.json"

def load_seed_aspects():
    """Load seed_topics.json and build lookup table for keyword → aspect name.
    Returns a list of dicts: [{name, keywords, pattern}, ...]
    """
    aspects = []
    if not os.path.exists(SEED_TOPICS_FILE):
        print(f"WARNING: {SEED_TOPICS_FILE} not found — aspect extraction will use BERTopic only.")
        return aspects
    with open(SEED_TOPICS_FILE, "r") as f:
        data = json.load(f)
    for entry in data["aspects"]:
        # Sort keywords longest-first so "battery life" matches before "battery"
        kws = sorted(entry["keywords"], key=len, reverse=True)
        # Build a single regex pattern: \b(keyword1|keyword2|...)\b
        escaped = [re.escape(k) for k in kws]
        pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
        aspects.append({
            'name': entry['name'],
            'keywords': kws,
            'pattern': pattern,
        })
    print(f"Loaded {len(aspects)} aspect definitions from {SEED_TOPICS_FILE}")
    return aspects


def extract_aspects(text: str) -> list:
    """Scan text for ALL mentioned aspects using keyword matching.
    Returns a list of aspect names found in the text.
    Falls back to empty list if nothing matches (caller should use BERTopic then).
    """
    found = []
    text_lower = text.lower()
    for aspect in aspect_lookup:
        if aspect['pattern'].search(text_lower):
            found.append(aspect['name'])
    return found

# DeBERTa label order: class_names = ['Negative', 'Neutral', 'Positive']
# Cardiff label mapping: LABEL_0 -> Negative, LABEL_1 -> Neutral, LABEL_2 -> Positive

# Words that are common in reviews but don't describe a product feature/aspect
_TOPIC_FILLER_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'is', 'it', 'to', 'of', 'for', 'in', 'on',
    'my', 'we', 'our', 'they', 'he', 'she', 'her', 'his', 'them', 'with',
    'bought', 'get', 'got', 'buy', 'purchase', 'purchased', 'love', 'loves',
    'loved', 'great', 'good', 'nice', 'awesome', 'excellent', 'best', 'use',
    'used', 'using', 'works', 'work', 'would', 'will', 'can', 'also', 'just',
    'very', 'really', 'so', 'this', 'that', 'these', 'those', 'then', 'now',
    'teenage', 'teenager', 'daughter', 'son', 'wife', 'husband', 'boyfriend',
    'girlfriend', 'mother', 'father', 'mom', 'dad', 'grandson', 'granddaughter',
    'kid', 'kids', 'child', 'children', 'toddler', 'year', 'old', 'age',
    'christmas', 'gift', 'birthday', 'present', 'gave', 'give',
    'beautifully', 'perfectly', 'absolutely', 'definitely', 'totally',
    'product', 'item', 'thing', 'device', 'which', 'what', 'has', 'have',
    'had', 'not', 'like', 'as', 'at', 'from', 'did', 'do', 'does',
}

def clean_topic_name(raw_name: str) -> str:
    """Strip filler/demographic words from a BERTopic auto-generated name.
    E.g. 'battery, beautifully, teenage, life' -> 'battery, life'
    Falls back to the raw name if all words are filtered out.
    """
    if not raw_name or raw_name == 'General/Mixed':
        return raw_name
    parts = [w.strip() for w in raw_name.split(',')]
    kept = [w for w in parts if w.lower() not in _TOPIC_FILLER_WORDS and len(w) > 2]
    return ', '.join(kept) if kept else raw_name
CARDIFF_LABEL_MAP = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
DEBERTA_CONFIDENCE_THRESHOLD = 0.65  # Use fallback if DeBERTa top-class prob < this

def load_resources():
    global sentiment_model, topic_model, df_data, df_topics, fallback_sentiment, aspect_lookup
    print("Loading resources... this might take a minute...")
    
    # 0. Aspect lookup table (for multi-aspect extraction)
    aspect_lookup = load_seed_aspects()
    
    # 1. Sentiment Model (DeBERTa fine-tuned)
    sentiment_model = SentimentAnalysisDeBERTa(model_path=MODEL_PKL)
    
    # 2. Fallback Sentiment Model (Cardiff RoBERTa — local copy, no internet needed)
    try:
        from transformers import pipeline
        # Prefer local path; fall back to HuggingFace Hub if not yet downloaded
        cardiff_source = CARDIFF_MODEL_PATH if os.path.exists(CARDIFF_MODEL_PATH) else "cardiffnlp/twitter-roberta-base-sentiment"
        print(f"Loading Cardiff RoBERTa fallback from: {cardiff_source}")
        fallback_sentiment = pipeline(
            "sentiment-analysis",
            model=cardiff_source,
            tokenizer=cardiff_source,
            top_k=None,  # return all class scores
        )
        print("Fallback model loaded.")
    except Exception as e:
        print(f"Warning: Could not load fallback sentiment model: {e}")
        fallback_sentiment = None
    
    # 3. Topic Model
    topic_model = BERTopic.load(BERTOPIC_MODEL_PATH, embedding_model="all-MiniLM-L6-v2")
    
    # 4. Data
    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        df_data['reviews.text'] = df_data['reviews.text'].fillna('').astype(str)
        df_data['reviews.title'] = df_data['reviews.title'].fillna('').astype(str)
    
    if os.path.exists(TOPIC_INFO_PATH):
        df_topics = pd.read_csv(TOPIC_INFO_PATH)

# Routes
@app.route('/training-status')
def get_training_status():
    status_file = "Ecommerce_dataset/training_status.json"
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({"status": "idle", "progress": 0, "message": ""})

@app.route('/')
def index():
    if df_topics is None:
        return "Resources not loaded correctly. Run the pipeline first."
    
    # Group by Topic Name and count sentiments
    topic_summary = []
    for _, row in df_topics.iterrows():
        t_id = row['Topic']
        t_name = row['Name']
        
        # Get subset of data for this topic
        topic_subset = df_data[df_data['topic'] == t_id]
        if topic_subset.empty: continue
        
        sentiments = topic_subset['sentiment'].value_counts().to_dict()
        
        # Get unique product names for this topic
        products = topic_subset['name'].unique().tolist()
        
        topic_summary.append({
            'id': int(t_id),
            'name': t_name,
            'count': int(row['Count']),
            'positive': sentiments.get('Positive', 0),
            'neutral': sentiments.get('Neutral', 0),
            'negative': sentiments.get('Negative', 0),
            'products': products[:15] # limit to top 15 products for display
        })
        
    return render_template('index.html', topics=topic_summary)

@app.route('/get_reviews', methods=['POST'])
def get_reviews():
    topic_id = request.json.get('topic_id')
    product_name = request.json.get('product_name')
    
    subset = df_data[(df_data['topic'] == int(topic_id)) & (df_data['name'] == product_name)]
    reviews = subset[['reviews.title', 'reviews.text', 'sentiment']].to_dict('records')
    return jsonify(reviews)

@app.route('/products')
def products():
    if df_data is None:
        return "Resources not loaded correctly. Run the pipeline first."
    
    # Get top 50 products by review count for performance
    top_products = df_data['name'].value_counts().head(50).index.tolist()
    
    product_summary = []
    for prod in top_products:
        if not isinstance(prod, str) or not prod.strip() or prod == 'Unknown':
            continue
            
        subset = df_data[df_data['name'] == prod]
        sentiments = subset['sentiment'].value_counts().to_dict()
        
        product_summary.append({
            'name': prod,
            'total': len(subset),
            'positive': sentiments.get('Positive', 0),
            'neutral': sentiments.get('Neutral', 0),
            'negative': sentiments.get('Negative', 0)
        })
        
    return render_template('products.html', products=product_summary)

@app.route('/api/product-reviews', methods=['POST'])
def get_product_reviews():
    product_name = request.json.get('product_name')
    sentiment = request.json.get('sentiment')
    
    subset = df_data[(df_data['name'] == product_name) & (df_data['sentiment'] == sentiment)]
    reviews = subset[['reviews.title', 'reviews.text', 'sentiment']].to_dict('records')
    return jsonify(reviews)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'})
    
    # 1. Sentiment — DeBERTa with confidence-gated fallback
    sentiment = 'Unknown'
    model_used = 'DeBERTa'

    if sentiment_model.model is not None and sentiment_model.tokenizer is not None:
        enc = sentiment_model.tokenizer(
            [text],
            add_special_tokens=True,
            max_length=sentiment_model.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        logits = sentiment_model.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits.numpy()[0]

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        max_prob = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        deberta_sentiment = sentiment_model.class_names[pred_idx]

        if max_prob >= DEBERTA_CONFIDENCE_THRESHOLD or fallback_sentiment is None:
            # DeBERTa is confident enough — trust it
            sentiment = deberta_sentiment
        else:
            # Low confidence — defer to Cardiff RoBERTa fallback
            model_used = 'Cardiff-RoBERTa (fallback)'
            try:
                results = fallback_sentiment(text[:512])  # Cardiff max token limit
                # results is a list of lists: [[{label, score}, ...]]
                scores = results[0] if isinstance(results[0], list) else results
                best = max(scores, key=lambda x: x['score'])
                sentiment = CARDIFF_LABEL_MAP.get(best['label'], deberta_sentiment)
            except Exception as e:
                print(f"Fallback model error: {e}")
                sentiment = deberta_sentiment  # graceful degradation
    elif fallback_sentiment is not None:
        # DeBERTa not loaded at all — use fallback directly
        model_used = 'Cardiff-RoBERTa (fallback)'
        try:
            results = fallback_sentiment(text[:512])
            scores = results[0] if isinstance(results[0], list) else results
            best = max(scores, key=lambda x: x['score'])
            sentiment = CARDIFF_LABEL_MAP.get(best['label'], 'Neutral')
        except Exception as e:
            print(f"Fallback model error: {e}")
            sentiment = 'Neutral'

    print(f"[predict] text='{text[:60]}' model={model_used} sentiment={sentiment}")

    # 2. Multi-Aspect Extraction
    #    First try keyword matching from seed_topics.json (returns ALL aspects)
    #    Fall back to BERTopic single-topic if no keywords match
    aspects = extract_aspects(text)
    
    if not aspects:
        # No keyword hits — fall back to BERTopic
        topics, probs = topic_model.transform([text])
        topic_id = topics[0]
        topic_name = "General/Mixed"
        if df_topics is not None:
            match = df_topics[df_topics['Topic'] == topic_id]
            if not match.empty:
                topic_name = clean_topic_name(match.iloc[0]['Name'])
        aspects = [topic_name]

    print(f"[predict] aspects={aspects}")
            
    return jsonify({
        'sentiment': sentiment,
        'aspect': ', '.join(aspects),
        'aspects': aspects,
    })

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, host='0.0.0.0', port=5001)
