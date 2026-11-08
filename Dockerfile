# Production-Ready Multi-Stage Python Container
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies needed for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy application source code
COPY . .

# Generate sample test images
RUN python generate_samples.py

# Run unit tests during container build to verify integrity
RUN python -m pytest tests/

# Expose web service port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "run.py"]
