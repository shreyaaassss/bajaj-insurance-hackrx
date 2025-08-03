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

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Add local bin to PATH BEFORE installing packages
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy requirements and install dependencies - FIXED filename
COPY --chown=app:app requirements.txt ./requirements.txt
RUN pip install --user --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=app:app main.py ./

# Create necessary directories
RUN mkdir -p uploads chroma_db

# Health check with fixed port
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# FIXED CMD to use Python directly
CMD ["python", "main.py"]
