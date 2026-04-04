# Docker Retraining Pipeline Guide

This repository has been updated to fully support Docker deployment, including an autonomous background worker that automatically retrains your models when new data arrives.

## 1. What Was Done

To achieve a seamless Dockerized workflow, the following components were built into the codebase:

- **Dockerization (`Dockerfile`, `docker-compose.yml`, `requirements.docker.txt`)**: Set up a multi-container Docker application. Since Macs use Metal processing that Docker doesn't support natively inside Linux containers, the `requirements.docker.txt` was specially tailored to run standard Linux TensorFlow on the CPU.
- **Port Mapping Fix**: Updated Flask (`app.py`) to bind to `0.0.0.0` so that it seamlessly tunnels out of the Docker container onto your host machine (`localhost:5001`).
- **Data Watcher (`watch_and_train.py`)**: A persistent Python observer that actively guards the `Ecommerce_dataset/incoming` folder for new `.csv` files.
- **BERTopic Automation (`train_bertopic.py`)**: Extracted and modularized the topic modeling code originally trapped inside the `MasterRunner.ipynb` Jupyter Notebook. Now, the system can rebuild topic distributions autonomously on the fly. 

## 2. How to Use It

### Step 1: Start Docker Desktop
Because Docker runs via a daemon service, **you must open the "Docker" application** on your Mac (Docker Desktop). Wait for the Docker icon in your top menu bar to show that the Docker Engine is fully started.

*(If you get a `failed to connect to the docker API at unix:///...docker.sock` error, this means Docker Desktop is closed!)*

### Step 2: Boot the Pipeline
Open your terminal inside this project folder and run:
```bash
docker-compose up -d --build
```
> **Note**: The very first time you run this, it will take several minutes to download the multi-gigabyte PyTorch and TensorFlow Linux wheels. Subsequent boots will take seconds.

### Step 3: Use the Web UI
Open your web browser and go to:
**http://localhost:5001**

### Step 4: How to View Logs (Important)
Since the containers run in the background (detached mode), you can monitor the web server and training progress with these commands:

```bash
# View combined logs for both web and trainer
docker-compose logs -f

# View ONLY training/retraining logs
docker-compose logs -f trainer

# View ONLY web server logs
docker-compose logs -f web
```

### Step 5: Automate Retraining
1. Create a directory named `incoming` inside your dataset folder if it does not yet exist: 
   `mkdir -p Ecommerce_dataset/incoming/`
2. Drop your new data file into `Ecommerce_dataset/incoming/` and name it **exactly** `new.data.csv`.
3. Monitor the logs (`docker-compose logs -f trainer`)! The `trainer` container will grab the file, merge it, and start retraining.
4. Once completed, the containers will automatically serve the fresh models.

## Stopping the Server
To shut down the web server and the background watcher, simply run:
```bash
docker-compose down
```

## 3. Windows & Non-Mac Support

The project is optimized for both macOS and Windows/Linux. 

### Local (Non-Docker) Environment
If you are on **Windows** and want to run the project locally (without Docker):
1.  Use `requirements.windows.txt` instead of the default `requirements.txt`.
2.  Install with: `pip install -r requirements.windows.txt`
3.  **Note:** Standard `tensorflow==2.16.2` replaces the macOS-specific components (`tensorflow-macos` and `tensorflow-metal`).

### Docker Environment (Recommended)
Docker Compose automatically handles the environment differences. The `requirements.docker.txt` already uses standard `tensorflow==2.16.2`, which is compatible with Windows, Mac, and Linux. Just run `docker-compose up -d --build` as usual.
