import sys
import os
import logging

# Ensure src in PYTHON PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_PATH
from src.aspect_extractor import AspectExtractor
from src.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting EDA Aspect Sentiment Pipeline...")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    try:
        # Step 1: Extract Aspects
        extractor = AspectExtractor(data_path=DATA_PATH)
        top_aspects = extractor.extract_top_aspects(top_n=20)
        
        # Step 2: Generate Report
        generator = ReportGenerator()
        output_path = generator.generate_report(top_aspects)
        
        logger.info(f"Pipeline finished successfully. Please open {output_path} to view the report.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
