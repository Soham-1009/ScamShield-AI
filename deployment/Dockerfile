# ScamShield AI — Google Cloud Run / Container Deployment Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    SCAMSHIELD_HOST=0.0.0.0

WORKDIR /app

# Install Tesseract OCR binary and minimal system libraries required by Pillow/Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app.py train_model.py train_distilbert.py ./
COPY scamshield/ ./scamshield/
COPY data/.gitkeep ./data/.gitkeep
COPY models/ ./models/

# Run as a non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Start Flask backend via Gunicorn on $PORT (Cloud Run standard)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "60", "app:app"]
