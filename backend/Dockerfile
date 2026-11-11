# backend/Dockerfile
# FastAPI backend. Build from the project root (docker compose does this):
#     docker build -f backend/Dockerfile -t eduagent-backend .
# The local models (./models, about 2 GB) are mounted at run time, not copied into the image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# CPU-only torch first: the default wheel on Linux pulls several GB of CUDA libraries
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY backend ./backend
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8010
# Only the frontend's nginx reaches this port, so its X-Forwarded-For (the browser's address,
# used by the login limits) is trusted
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8010", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
