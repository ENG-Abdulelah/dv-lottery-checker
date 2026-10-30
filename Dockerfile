FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# مكتبات النظام التي يحتاجها OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# تثبيت المتطلبات + gunicorn صراحةً (لضمان وجوده)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==22.0.0

COPY . .

RUN mkdir -p /app/static/uploads /app/static/results

ENV PORT=10000
EXPOSE 10000

# استخدام python -m gunicorn لتجنب مشاكل PATH
#CMD ["sh", "-c", "python -m gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 180 --max-requests 500"]
CMD ["sh", "-c", "python -m gunicorn app:app --bind 0.0.0.0:10000 --workers 1 --timeout 1000 --max-requests 1000"]
