# ===== STAGE 1: Builder =====
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ===== STAGE 2: Runtime =====
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies - INCLUDING libgomp1!
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]