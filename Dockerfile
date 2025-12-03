FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

# Create data folder inside /tmp for Cloud Run
RUN mkdir -p /tmp/uploads

# Expose *example* port (ignored by Cloud Run but kept for local run)
EXPOSE 8000

# IMPORTANT: Use Cloud Run dynamic PORT
CMD ["sh", "-c", "uvicorn smartbvb:app --host 0.0.0.0 --port $PORT"]
