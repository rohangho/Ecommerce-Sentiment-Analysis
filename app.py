from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

# Fix for hdbscan hanging on macOS (Apple Silicon especially)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import tensorflow as tf
from pathlib import Path
from bertopic import BERTopic

# Add notebooks to sys.path for model class imports
sys.path.insert(0, os.path.abspath('notebooks'))
from sentiment_analysis_DeBERTa import SentimentAnalysisDeBERTa

app = Flask(__name__)

# --- Configuration ---
DATA_PATH = "Ecommerce_dataset/combined_data_with_topics.csv"
TOPIC_INFO_PATH = "deploy_models/bertopic_info.csv"
MODEL_PKL = "deploy_models/sentiment_model_deberta.pkl"
BERTOPIC_MODEL_PATH = "deploy_models/bertopic_model"

# Global objects for the app
sentiment_model = None
topic_model = None
df_data = None
df_topics = None

def load_resources():
    global sentiment_model, topic_model, df_data, df_topics
    print("Loading resources... this might take a minute...")
    
    # 1. Sentiment Model
    sentiment_model = SentimentAnalysisDeBERTa(model_path=MODEL_PKL)
    
    # 2. Topic Model
    topic_model = BERTopic.load(BERTOPIC_MODEL_PATH)
    
    # 3. Data
    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        df_data['reviews.text'] = df_data['reviews.text'].fillna('').astype(str)
        df_data['reviews.title'] = df_data['reviews.title'].fillna('').astype(str)
    
    if os.path.exists(TOPIC_INFO_PATH):
        df_topics = pd.read_csv(TOPIC_INFO_PATH)

# Routes
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

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'})
    
    # 1. Sentiment
    # Predict using DeBERTa
    enc = sentiment_model.tokenizer(
        [text], 
        add_special_tokens=True, 
        max_length=sentiment_model.max_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="tf"
    )
    logits = sentiment_model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    pred_idx = tf.argmax(logits, axis=1).numpy()[0]
    sentiment = sentiment_model.class_names[int(pred_idx)]
    
    # 2. Aspect (Topic)
    topics, probs = topic_model.transform([text])
    topic_id = topics[0]
    
    # Find topic name
    topic_name = "General/Mixed"
    if df_topics is not None:
        match = df_topics[df_topics['Topic'] == topic_id]
        if not match.empty:
            topic_name = match.iloc[0]['Name']
            
    return jsonify({
        'sentiment': sentiment,
        'aspect': topic_name
    })

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5001)
