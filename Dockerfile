FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
# Cloud Run will inject PORT env var (usually 8080)
ENV PORT=8080

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /data/uploads

# (EXPOSE is optional and ignored by Cloud Run, but we can keep it)
EXPOSE 8080

#  IMPORTANT: use $PORT instead of hardcoding 8000
CMD ["sh", "-c", "uvicorn smartbvb:app --host 0.0.0.0 --port ${PORT}"]
