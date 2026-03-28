# Sentiment Analysis Web Application

A modern web interface for sentiment analysis using two different models:
1. **Pure Logistic Regression** - Fast, trained on your dataset
2. **Pre-trained RoBERTa Model** - Advanced transformer model from HuggingFace

## Features

- 🎯 Easy-to-use web interface
- 📊 Two sentiment analysis models to choose from
- ⚡ Real-time sentiment prediction
- 🎨 Beautiful, responsive design
- 📱 Works on desktop and mobile devices

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- pandas (data manipulation)
- scikit-learn (machine learning)
- torch (PyTorch)
- transformers (HuggingFace models)
- joblib (model serialization)

### 2. Prepare Models

Make sure you have the trained model file:
- `sentiment_model_TFD_LR.pkl` - Logistic Regression model

The HuggingFace model will be downloaded automatically on first run.

**Note**: If you need to train the logistic regression model first, run:
```bash
python notebooks/sentiment-analysis.py
```

### 3. Run the Application

From the `Ecommerce-Sentiment-Analysis` directory, run:

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## How to Use

1. **Select a Model**: Choose between "Pure Logistic Regression" or "Pre-trained RoBERTa Model"
2. **Enter Text**: Type your product review or comment in the text area
3. **Analyze**: Click the "Analyze Sentiment" button
4. **View Results**: The sentiment prediction will be displayed below

## Project Structure

```
Ecommerce-Sentiment-Analysis/
├── app.py                           # Flask application main file
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── templates/
│   └── index.html                  # Web interface HTML
├── static/
│   └── style.css                   # Web interface styling
├── notebooks/
│   ├── sentiment-analysis.py       # Logistic Regression model
│   ├── Pre-trained.py              # HuggingFace model wrapper
│   ├── CapstoneData.py             # Data handling utilities
│   └── ...
├── Ecommerce_dataset/
│   ├── train_data.csv
│   ├── test_data.csv
│   └── test_data_hidden.csv
└── sentiment_model_TFD_LR.pkl      # Trained logistic regression model
```

## Model Details

### Logistic Regression Model
- Uses TF-IDF vectorization for text features
- Processes product reviews and titles
- Also considers brand and category information
- Lightweight and fast predictions

### HuggingFace RoBERTa Model
- Pre-trained "siebert/sentiment-roberta-large-english" model
- Transformer-based architecture
- More nuanced sentiment understanding
- Requires GPU for optimal performance (falls back to CPU)

## Troubleshooting

### Model loading fails
- Ensure the `sentiment_model_TFD_LR.pkl` file exists in the app directory
- Check that all required packages are installed

### HuggingFace model doesn't download
- Set your HuggingFace token in environment or in `Pre-trained.py`
- Ensure you have internet connectivity
- The model will be cached after first download

### Flask not starting
- Make sure no other service is using port 5000
- Check that Flask is properly installed: `pip install Flask`
- Run from the correct directory containing `app.py`

## API Endpoint

The application exposes a POST API endpoint for predictions:

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
    "text": "Your review text here",
    "model_type": "logistic"  // or "hf"
}
```

**Response**:
```json
{
    "success": true,
    "model": "Logistic Regression",
    "prediction": "positive",
    "confidence": "N/A",
    "text": "Your review text here"
}
```

## Development

To run in development mode with debug enabled:
```bash
python app.py
```

The application will automatically reload when you make changes to the code.

## Notes

- The logistic regression model uses brand, categories, and text features
- All text is truncated to fit the model's input requirements
- Predictions are cached in the browser for performance
- The application runs on a local development server (not for production)

## License

This project is part of an ecommerce sentiment analysis capstone project.
