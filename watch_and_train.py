import os
import time
import shutil
import pandas as pd
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

INCOMING_DIR = "Ecommerce_dataset/incoming"
TRAIN_DATA_PATH = "Ecommerce_dataset/train_data.csv"
STATUS_FILE = "Ecommerce_dataset/training_status.json"

def update_status(progress, status, message):
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "progress": progress,
            "status": status,
            "message": message
        }, f)

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
            update_status(10, "training", "Merging incoming data with training dataset...")
            print(f"Merging {new_csv_path} into {TRAIN_DATA_PATH}...", flush=True)
            new_df = pd.read_csv(new_csv_path)
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            
            combined = pd.concat([train_df, new_df], ignore_index=True)
            
            backup_path = TRAIN_DATA_PATH.replace(".csv", "_backup.csv")
            shutil.copy(TRAIN_DATA_PATH, backup_path)
            
            combined.to_csv(TRAIN_DATA_PATH, index=False)
            print(f"Merged successfully. Train data now has {len(combined)} rows.", flush=True)

            update_status(40, "training", "Retraining DeBERTa sentiment model...")
            print("Running train_and_validate.py...", flush=True)
            subprocess.run(["python", "train_and_validate.py"], check=True)

            update_status(80, "training", "Rebuilding BERTopic clusters...")
            print("Running train_bertopic.py...", flush=True)
            subprocess.run(["python", "train_bertopic.py"], check=True)

            update_status(100, "completed", "Retraining finished successfully! Reloading UI...")
            os.remove(new_csv_path)
            print("Pipeline finished successfully! The Flask app will use the new models on its next reload.", flush=True)
            time.sleep(3) # Hold 100% for brief UI visibility
            update_status(0, "idle", "")
            
        except Exception as e:
            print(f"Pipeline failed: {e}", flush=True)
            update_status(0, "error", f"Pipeline failed: {e}")
            if os.path.exists(TRAIN_DATA_PATH.replace(".csv", "_backup.csv")):
                shutil.copy(TRAIN_DATA_PATH.replace(".csv", "_backup.csv"), TRAIN_DATA_PATH)
                print("Rolled back train_data.csv due to error", flush=True)
        finally:
            self.is_training = False
            time.sleep(5)
            update_status(0, "idle", "")

def start_watcher():
    os.makedirs(INCOMING_DIR, exist_ok=True)
    
    event_handler = RetrainTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, path=INCOMING_DIR, recursive=False)
    observer.start()
    
    print(f"Started watcher on {INCOMING_DIR}. Waiting for new .csv files...", flush=True)
    
    # Check for any lingering files that failed in the last crash and rerun them sequentially
    for file in os.listdir(INCOMING_DIR):
        if file.endswith(".csv"):
            pending_path = os.path.join(INCOMING_DIR, file)
            print(f"Found existing pending file {pending_path} on startup! Resuming pipeline...", flush=True)
            event_handler.trigger_pipeline(pending_path)
            
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()
