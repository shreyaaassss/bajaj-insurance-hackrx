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

# Set model cache environment
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Set working directory
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/transformers

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# FIXED: Install compatible PyTorch versions
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# PRE-DOWNLOAD MODELS (UPDATED VERSION)
RUN python -c "\
import os; \
os.environ['ANONYMIZED_TELEMETRY'] = 'False'; \
os.environ['CHROMA_TELEMETRY'] = 'False'; \
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'; \
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; \
os.environ['TRANSFORMERS_OFFLINE'] = '0'; \
print('🚀 Downloading optimized models...'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import time; \
start = time.time(); \
print('📥 Loading SentenceTransformer with cache optimization...'); \
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); \
model.save('/app/.cache/sentence_transformers/all-MiniLM-L6-v2'); \
print('📥 Loading CrossEncoder with cache optimization...'); \
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', max_length=256); \
print(f'✅ All models downloaded in {time.time()-start:.1f}s'); \
"

# Copy application code (FIXED: Use correct filename)
COPY main.py ./main.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p uploads

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Expose port
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]
