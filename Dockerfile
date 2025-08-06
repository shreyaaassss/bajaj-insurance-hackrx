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
ENV HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/transformers /app/.cache/huggingface

# Copy requirements first for better caching
COPY requirements.txt ./

# FIX: Install pip and compatible PyTorch version first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch FIRST with specific compatible version
RUN pip install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD MODELS with error handling
RUN python -c "\
import os; \
os.environ['ANONYMIZED_TELEMETRY'] = 'False'; \
os.environ['CHROMA_TELEMETRY'] = 'False'; \
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'; \
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; \
os.environ['TRANSFORMERS_OFFLINE'] = '0'; \
print('🚀 Downloading optimized models...'); \
try: \
    from sentence_transformers import SentenceTransformer, CrossEncoder; \
    import time; \
    start = time.time(); \
    print('📥 Loading SentenceTransformer...'); \
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); \
    print('💾 Saving SentenceTransformer to cache...'); \
    model.save('/app/.cache/sentence_transformers/all-MiniLM-L6-v2'); \
    print('📥 Loading CrossEncoder...'); \
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', max_length=256); \
    print(f'✅ All models downloaded in {time.time()-start:.1f}s'); \
except Exception as e: \
    print(f'❌ Model download failed: {e}'); \
    import sys; \
    sys.exit(1); \
"

# Copy application code
COPY main.py ./

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
