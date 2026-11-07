# Use Python 3.12 slim image
FROM python:3.12.2-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD HuggingFace model (1.3GB - saves 30-60s on first request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Copy application code
COPY . .

# Create directory for audio examples
RUN mkdir -p static/audio

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PORT=8080

# Expose port
EXPOSE 8080

# Use gunicorn for production with optimal settings
CMD exec gunicorn --bind :$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 60 \
    --worker-class gthread \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app