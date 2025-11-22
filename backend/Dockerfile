# QuantEase Backend - Dockerfile
# Updated to Python 3.11 for llama-cpp-python compatibility

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - build-essential + cmake → required for llama-cpp-python
# - libopenblas-dev → CPU optimization backend for GGUF
# - libpq-dev → for psycopg2/PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python separately (clean install)
RUN pip install --no-cache-dir llama-cpp-python

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models/quantized && \
    mkdir -p /app/data/uploads && \
    chmod -R 755 /app/data

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== BUILD NOTES =====
#
# For ARM/M1 Macs (Metal GPU support):
# Before building, run:
#   export CMAKE_ARGS="-DLLAMA_METAL=on"
#
# For NVIDIA GPUs (CUDA support):
#   export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
#
# Build image:
#   docker build -t quantease-backend .
#
# Run container:
#   docker run -p 8000:8000 quantease-backend
