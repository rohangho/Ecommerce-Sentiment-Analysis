import os
import time
import shutil
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

INCOMING_DIR = "Ecommerce_dataset/incoming"
TRAIN_DATA_PATH = "Ecommerce_dataset/train_data.csv"

class RetrainTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.is_training = False

    def on_created(self, event):
        if self.is_training:
            return
            
        if not event.is_directory and event.src_path.endswith(".csv"):
            print(f"Detected new data file: {event.src_path}", flush=True)
            self.trigger_pipeline(event.src_path)

    def trigger_pipeline(self, new_csv_path):
        self.is_training = True
        print("\n=== STARTING AUTOMATED RETRAINING PIPELINE ===", flush=True)
        try:
            # 1. Merge the new data into train_data.csv
            print(f"Merging {new_csv_path} into {TRAIN_DATA_PATH}...", flush=True)
            new_df = pd.read_csv(new_csv_path)
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            
            combined = pd.concat([train_df, new_df], ignore_index=True)
            
            # Create a backup before overwriting
            backup_path = TRAIN_DATA_PATH.replace(".csv", "_backup.csv")
            shutil.copy(TRAIN_DATA_PATH, backup_path)
            
            combined.to_csv(TRAIN_DATA_PATH, index=False)
            print(f"Merged successfully. Train data now has {len(combined)} rows.", flush=True)

            # 2. Run the main sentiment model training pipeline
            print("Running train_and_validate.py...", flush=True)
            # Using subprocess so it runs in its own memory space
            subprocess.run(["python", "train_and_validate.py"], check=True)

            # 3. Run the BERTopic training script
            print("Running train_bertopic.py...", flush=True)
            subprocess.run(["python", "train_bertopic.py"], check=True)

            # 4. Cleanup the incoming file
            processed_path = new_csv_path.replace(".csv", "_processed.csv")
            os.rename(new_csv_path, processed_path)
            print("Pipeline finished successfully! The Flask app will use the new models on its next reload.", flush=True)
            
        except Exception as e:
            print(f"Pipeline failed: {e}", flush=True)
            # Rollback if failed during merge
            if os.path.exists(TRAIN_DATA_PATH.replace(".csv", "_backup.csv")):
                shutil.copy(TRAIN_DATA_PATH.replace(".csv", "_backup.csv"), TRAIN_DATA_PATH)
                print("Rolled back train_data.csv due to error", flush=True)
        finally:
            self.is_training = False

def start_watcher():
    os.makedirs(INCOMING_DIR, exist_ok=True)
    
    event_handler = RetrainTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, path=INCOMING_DIR, recursive=False)
    observer.start()
    
    print(f"Started watcher on {INCOMING_DIR}. Waiting for new .csv files...", flush=True)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()
