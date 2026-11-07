FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-streaming.txt .
# Install the CPU-only PyTorch wheel first to avoid pulling CUDA libraries
# into an image that only does inference.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-streaming.txt

COPY serving ./serving
COPY streaming ./streaming
COPY models ./models

CMD ["python", "-m", "faust", "-A", "streaming.feature_app", "worker", "-l", "info", "--without-web"]
