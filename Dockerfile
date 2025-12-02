FROM python:3.11-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the whole project
COPY . /app

# Create data folder for persistent storage
RUN mkdir -p /data/uploads

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "smartbvb:app", "--host", "0.0.0.0", "--port", "8000"]
