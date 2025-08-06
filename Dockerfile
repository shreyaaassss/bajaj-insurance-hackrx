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

# Install PyTorch with compatible version
RUN pip install --no-cache-dir torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

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
