FROM python:3.11-slim

# Install necessary OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install the linux-compatible requirements
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.docker.txt

# Copy the entire workspace into the container image
COPY . /app

# Make training scripts executable
RUN chmod +x train_and_validate.py
RUN chmod +x train_bertopic.py
RUN chmod +x watch_and_train.py

# Expose the Flask dev server port
EXPOSE 5001

# The default command runs the web app if not overridden by docker-compose
CMD ["python", "app.py"]
