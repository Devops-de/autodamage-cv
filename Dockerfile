# Production Multi-Stage Python Container for ClaimScope
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Filter out developer and test tooling (pytest, ruff) for lean runtime build
RUN grep -v -E "^(pytest|ruff)" requirements.txt > requirements-prod.txt && \
    pip install --no-cache-dir --user -r requirements-prod.txt

FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLAIMSCOPE_HOST=0.0.0.0 \
    PATH="/home/claimscope/.local/bin:$PATH"

WORKDIR /app

# Install minimal OS runtime libraries required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root service user for container security
RUN groupadd -r claimscope && useradd -r -g claimscope -d /home/claimscope -m claimscope

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/claimscope/.local

# Copy application source tree
COPY --chown=claimscope:claimscope . .

USER claimscope

# Pre-generate sample images for immediate local verification
RUN python generate_samples.py

EXPOSE 8000

CMD ["python", "run.py"]
