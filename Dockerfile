# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    libmagic-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

# Copy requirements and install Python dependencies
COPY requirements.txt ./requirements.txt

# Install PyTorch first with CPU support
RUN pip install --no-cache-dir torch>=1.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set proper ownership
RUN mkdir -p /app/uploads /app/cache /app/models /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Pre-download compatible models (single-line format to avoid parsing issues)
RUN python -c "import os; from transformers import AutoTokenizer; os.makedirs('/app/models', exist_ok=True); tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='/app/models'); print('BERT tokenizer downloaded successfully')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
