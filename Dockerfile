FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies for OCR & PyMuPDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    gcc \
    g++ \
    libgl1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

# Create upload directory
RUN mkdir -p /data/uploads

# Optional (ignored by Cloud Run)
EXPOSE 8080

# Use Cloud Run PORT
CMD ["sh", "-c", "uvicorn smartbvb:app --host 0.0.0.0 --port ${PORT}"]
