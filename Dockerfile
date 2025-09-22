# Python 3.12系が無難（手元が3.13っぽいけど、サーバは安定版で）
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MODEL_NAME=base \
    COMPUTE_TYPE=int8 \
    PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 初回を速くしたいなら事前DL（任意）
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_NAME}', compute_type='${COMPUTE_TYPE}')"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --log-level info"]
