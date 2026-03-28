from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import os

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add notebooks to path to import the model classes
notebooks_path = os.path.join(os.path.dirname(__file__), 'notebooks')
sys.path.insert(0, notebooks_path)

# Import the model classes using importlib to handle filenames with dashes
import importlib.util

# Import SentimentAnalysis from sentiment-analysis.py
spec_sa = importlib.util.spec_from_file_location("sentiment_analysis_module", os.path.join(notebooks_path, "sentiment-analysis.py"))
sentiment_analysis_module = importlib.util.module_from_spec(spec_sa)
spec_sa.loader.exec_module(sentiment_analysis_module)
SentimentAnalysis = sentiment_analysis_module.SentimentAnalysis

# Import SentimentAnalysisHF from Pre-trained.py
spec_hf = importlib.util.spec_from_file_location("pretrained_module", os.path.join(notebooks_path, "Pre-trained.py"))
pretrained_module = importlib.util.module_from_spec(spec_hf)
spec_hf.loader.exec_module(pretrained_module)
SentimentAnalysisHF = pretrained_module.SentimentAnalysisHF

# Import SentimentAnalysisSVM from sentiment-analysis-SVM.py
spec_svm = importlib.util.spec_from_file_location("svm_module", os.path.join(notebooks_path, "sentiment-analysis-SVM.py"))
svm_module = importlib.util.module_from_spec(spec_svm)
spec_svm.loader.exec_module(svm_module)
SentimentAnalysisSVM = svm_module.SentimentAnalysisSVM

# Import SentimentAnalysisRNN from sentiment-analysis-RNN.py
spec_rnn = importlib.util.spec_from_file_location("rnn_module", os.path.join(notebooks_path, "sentiment-analysis-RNN.py"))
rnn_module = importlib.util.module_from_spec(spec_rnn)
spec_rnn.loader.exec_module(rnn_module)
SentimentAnalysisRNN = rnn_module.SentimentAnalysisRNN

# Import SentimentAnalysisBERT from sentiment-analysis-BERT.py
spec_bert = importlib.util.spec_from_file_location("bert_module", os.path.join(notebooks_path, "sentiment-analysis-BERT.py"))
bert_module = importlib.util.module_from_spec(spec_bert)
spec_bert.loader.exec_module(bert_module)
SentimentAnalysisBERT = bert_module.SentimentAnalysisBERT

app = Flask(__name__)

# Load all models
print("="*60)
print("Loading Sentiment Analysis Models")
print("="*60)

models = {}

try:
    print("Loading Logistic Regression model...")
    logistic_model = SentimentAnalysis('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_TFD_LR.pkl')
    models['logistic'] = logistic_model
    print("✓ Logistic Regression model loaded successfully")
except Exception as e:
    print(f"✗ Error loading Logistic Regression model: {e}")
    models['logistic'] = None

try:
    print("Loading RoBERTa (HuggingFace) model...")
    hf_model = SentimentAnalysisHF('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_pre_trained_model.pkl')
    models['hf'] = hf_model
    print("✓ RoBERTa model loaded successfully")
except Exception as e:
    print(f"✗ Error loading RoBERTa model: {e}")
    models['hf'] = None

try:
    print("Loading SVM model...")
    svm_model = SentimentAnalysisSVM('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_SVM.pkl')
    models['svm'] = svm_model
    print("✓ SVM model loaded successfully")
except Exception as e:
    print(f"✗ Error loading SVM model: {e}")
    models['svm'] = None

try:
    print("Loading RNN model...")
    rnn_model = SentimentAnalysisRNN('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_RNN.pkl')
    models['rnn'] = rnn_model
    print("✓ RNN model loaded successfully")
except Exception as e:
    print(f"✗ Error loading RNN model: {e}")
    models['rnn'] = None

try:
    print("Loading BERT model...")
    bert_model = SentimentAnalysisBERT('../Capstone/Ecommerce-Sentiment-Analysis/sentiment_model_BERT.pkl')
    models['bert'] = bert_model
    print("✓ BERT model loaded successfully")
except Exception as e:
    print(f"✗ Error loading BERT model: {e}")
    models['bert'] = None

print("="*60 + "\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        product = data.get('product', 'Unknown').strip()
        category = data.get('category', 'General').strip() or 'General'
        primary_category = data.get('primaryCategory', 'General').strip() or 'General'
        title = data.get('title', '').strip()
        review = data.get('review', '').strip()
        model_type = data.get('model_type', 'logistic')

        if not review:
            return jsonify({'error': 'Please enter a review to analyze'}), 400

        if len(review) < 3:
            return jsonify({'error': 'Review must be at least 3 characters'}), 400

        model_info = {
            'logistic': {'name': 'Logistic Regression', 'desc': 'Fast, trained on your dataset'},
            'hf': {'name': 'Pre-trained RoBERTa', 'desc': 'Advanced transformer from HuggingFace'},
            'svm': {'name': 'Support Vector Machine', 'desc': 'High-dimensional text classification'},
            'rnn': {'name': 'Recurrent Neural Network', 'desc': 'LSTM-based sequence learning'},
            'bert': {'name': 'BERT Transformer', 'desc': 'Pre-trained bidirectional transformer'}
        }

        if model_type not in models:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400

        model = models[model_type]
        if model is None:
            return jsonify({'error': f'{model_info[model_type]["name"]} model not loaded'}), 500

        # Make prediction based on model type
        if model_type == 'logistic':
            sample = pd.DataFrame([{
                "reviews.text": review,
                "reviews.title": title,
                "brand": product,
                "categories": category,
                "primaryCategories": primary_category
            }])
            prediction = model.model.predict(sample)[0]

        elif model_type == 'hf':
            prediction = model.predict(review)

        elif model_type == 'svm':
            prediction = model.predict(review, title, product, category, primary_category)

        elif model_type == 'rnn':
            prediction = model.predict(review, title, product, category, primary_category)

        elif model_type == 'bert':
            prediction = model.predict(review, title, product, category, primary_category)

        return jsonify({
            'success': True,
            'model': model_info[model_type]['name'],
            'prediction': prediction,
            'confidence': "N/A",
            'product': product,
            'category': category,
            'primaryCategory': primary_category,
            'title': title,
            'review': review
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Sentiment Analysis Web App")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
