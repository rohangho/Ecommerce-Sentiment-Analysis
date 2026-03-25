import os
import logging
from jinja2 import Environment, FileSystemLoader

from src.config import TEMPLATE_DIR, REPORT_OUTPUT_PATH

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, template_dir: str = TEMPLATE_DIR, output_path: str = REPORT_OUTPUT_PATH):
        self.template_dir = template_dir
        self.output_path = output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def generate_report(self, top_aspects: dict) -> str:
        logger.info("Generating HTML report...")
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template('report_template.html')
        
        html_output = template.render(top_aspects=top_aspects)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
            
        logger.info(f"Report successfully saved to {self.output_path}")
        return self.output_path
