FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    PORT=8000 \
    MODEL_NAME=base \
    COMPUTE_TYPE=int8 \
    HF_HOME=/root/.cache  

# ffmpeg 入れる
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . .

# ★ここでモデルを事前DLして“焼き込む”（Freeは永続ディスク不可のため）
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_NAME}', compute_type='${COMPUTE_TYPE}', download_root='/root/.cache')"

EXPOSE 8000
# ★並列を抑える（Freeだとこれが効く）
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --limit-concurrency 1 --log-level info"]
