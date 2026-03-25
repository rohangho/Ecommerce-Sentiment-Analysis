import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, 'Ecommerce_dataset', 'train_data.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

REPORT_OUTPUT_PATH = os.path.join(REPORT_DIR, 'aspect_sentiment_report.html')

STOP_ASPECTS = {
    'i', "it's", 'this', 'that', 'there', 'they', 
    'amazon', 'great', 'good', 'kindle', 'tablet', 
    'echo', 'alexa', 'very', 'not', 'just', 'too'
}
