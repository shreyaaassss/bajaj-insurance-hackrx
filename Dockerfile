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

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Add local bin to PATH BEFORE installing packages
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy requirements and install dependencies
COPY --chown=app:app requirements.txt ./
RUN pip install --user --no-cache-dir --upgrade -r requirements.txt

# Copy application code - Updated filename
COPY --chown=app:app main.py ./

# Create necessary directories
RUN mkdir -p uploads chroma_db

# Health check with correct port variable
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port (Cloud Run will set this)
EXPOSE ${PORT}

# Updated CMD to use main.py and PORT environment variable correctly
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --timeout-keep-alive 300 \
    --log-level info
