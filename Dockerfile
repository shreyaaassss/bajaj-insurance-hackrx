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

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# PRE-DOWNLOAD MODELS (UPDATED for hackrx6.py compatibility)
RUN python -c "\
import os; \
os.environ['ANONYMIZED_TELEMETRY'] = 'False'; \
os.environ['CHROMA_TELEMETRY'] = 'False'; \
print('Downloading models...'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Loading SentenceTransformer...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Loading CrossEncoder...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('All models downloaded successfully!'); \
"

# Copy application code (UPDATED filename)
COPY main.py ./main.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p uploads

# Health check with proper timeout (UPDATED endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Expose port
EXPOSE 8080

# Start the application (UPDATED to use hackrx6.py)
CMD ["python", "main.py"]
