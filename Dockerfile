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

# Install pip and compatible PyTorch version first
RUN pip install --no-cache-dir --upgrade pip

# CRITICAL FIX: Install PyTorch 1.13.1 explicitly first
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (excluding torch to avoid conflicts)
RUN pip install --no-cache-dir -r requirements.txt

# Create a Python script for model downloading with better error handling
RUN echo 'import os\n\
import sys\n\
\n\
# Set environment variables\n\
os.environ["ANONYMIZED_TELEMETRY"] = "False"\n\
os.environ["CHROMA_TELEMETRY"] = "False"\n\
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\n\
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"\n\
os.environ["TRANSFORMERS_OFFLINE"] = "0"\n\
\n\
print("🚀 Downloading optimized models...")\n\
\n\
try:\n\
    # Test PyTorch first\n\
    import torch\n\
    print(f"✅ PyTorch version: {torch.__version__}")\n\
    \n\
    # Test transformers\n\
    import transformers\n\
    print(f"✅ Transformers version: {transformers.__version__}")\n\
    \n\
    # Import sentence transformers\n\
    from sentence_transformers import SentenceTransformer, CrossEncoder\n\
    import time\n\
    \n\
    start = time.time()\n\
    print("📥 Loading SentenceTransformer...")\n\
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")\n\
    print("💾 Saving SentenceTransformer to cache...")\n\
    model.save("/app/.cache/sentence_transformers/all-MiniLM-L6-v2")\n\
    \n\
    print("📥 Loading CrossEncoder...")\n\
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=256)\n\
    \n\
    print(f"✅ All models downloaded successfully in {time.time()-start:.1f}s")\n\
    \n\
except ImportError as e:\n\
    print(f"❌ Import error: {e}")\n\
    print("Available packages:")\n\
    import pkg_resources\n\
    installed_packages = [d.project_name for d in pkg_resources.working_set]\n\
    for pkg in ["torch", "transformers", "sentence-transformers"]:\n\
        status = "✅" if pkg in installed_packages else "❌"\n\
        print(f"  {status} {pkg}")\n\
    sys.exit(1)\n\
    \n\
except Exception as e:\n\
    print(f"❌ Model download failed: {e}")\n\
    print(f"Error type: {type(e).__name__}")\n\
    import traceback\n\
    traceback.print_exc()\n\
    sys.exit(1)\n' > download_models.py

# Run the model download script with better error reporting
RUN python download_models.py && rm download_models.py

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
