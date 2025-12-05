FROM python:3.11-slim

# Ensures logs flush immediately
ENV PYTHONUNBUFFERED=1
# Cloud Run provides PORT dynamically
ENV PORT=8080

# ----------------------------------------------------
# Install system dependencies for OCR, Tesseract, PyMuPDF
# ----------------------------------------------------
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

# ----------------------------------------------------
# Set working directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ----------------------------------------------------
# Copy project files
# ----------------------------------------------------
COPY . /app

# ----------------------------------------------------
# Create upload directory
# ----------------------------------------------------
RUN mkdir -p /data/uploads

# ----------------------------------------------------
# Expose port (ignored by Cloud Run but useful locally)
# ----------------------------------------------------
EXPOSE 8080

# ----------------------------------------------------
# Start FastAPI app using Cloud Run's provided PORT
# MUST use $PORT (NOT ${PORT})
# ----------------------------------------------------
CMD ["sh", "-c", "uvicorn smartbvb:app --host 0.0.0.0 --port $PORT"]
