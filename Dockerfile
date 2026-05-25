FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY models/ ./models/
COPY config/ ./config/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY src/ ./src/

RUN mkdir -p /app/.cache/huggingface /app/.cache/torch && \
    chmod -R 777 /app/.cache

ENV PYTHONPATH="/app"
ENV HF_HOME="/app/.cache/huggingface"
ENV TORCH_HOME="/app/.cache/torch"

EXPOSE 7860

# TODO: Budai don't forget to CHANGE THIS!!! esp if fullstack team didnt use fastapi for BE
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "600"]
