# Use Python 3.11 slim for better compatibility
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy requirements first for better caching (FIXED FILENAME)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# PRE-DOWNLOAD MODELS (CRITICAL FIX - This prevents startup timeout)
RUN python -c "
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
print('Downloading models...')
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
print('Loading SentenceTransformer...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Loading CrossEncoder...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
print('Loading HuggingFaceEmbeddings...')
HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
print('All models downloaded successfully!')
"

# Copy application code (FIXED FILENAME)
COPY main.py ./main.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p uploads chroma_db

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]
