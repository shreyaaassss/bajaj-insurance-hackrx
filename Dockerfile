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

# CRITICAL FIX: Set model cache environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Set working directory
WORKDIR /app

# Create cache directories for better model caching
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers /app/.cache/transformers

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# CRITICAL FIX: Install compatible PyTorch version FIRST
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# FIXED PRE-DOWNLOAD MODELS (Proper syntax and error handling)
RUN python -c "\
import os; \
os.environ['ANONYMIZED_TELEMETRY'] = 'False'; \
os.environ['CHROMA_TELEMETRY'] = 'False'; \
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'; \
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; \
print('🚀 Downloading optimized models...'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import time; \
start = time.time(); \
print('📥 Loading SentenceTransformer...'); \
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', cache_folder='/app/.cache/sentence_transformers'); \
print('📥 Loading CrossEncoder...'); \
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', max_length=512); \
print(f'✅ All models downloaded successfully in {time.time()-start:.1f}s'); \
"

# Copy application code
COPY main.py ./main.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p uploads

# FIXED Health check endpoint (your app uses "/" not "/health")
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Expose port
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]
