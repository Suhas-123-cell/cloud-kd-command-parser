# Inference service image (no model weights inside: they are downloaded from SeaweedFS at start-up).
# Build (self-contained context):  docker build -t cc-kd/parser:1.0 service/
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only PyTorch wheel (the default wheel pulls in ~2.5 GB of CUDA libraries).
COPY requirements.txt .
# torch's own dependencies are pinned in requirements.txt, so it is installed with --no-deps.
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r requirements.txt \
 && pip check

COPY schema.json inference.py app.py ./

RUN useradd --uid 1000 --create-home app
USER 1000
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
