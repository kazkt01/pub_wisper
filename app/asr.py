# app/asr.py
import os, shutil, subprocess
from pathlib import Path
from typing import List, Dict
from faster_whisper import WhisperModel

# --- ENV解決（MODEL_NAME/COMPUTE_TYPE を優先、互換で WHISPER_* も見る） ---
def _resolve_model_name() -> str:
    return (
        os.getenv("MODEL_NAME")
        or os.getenv("WHISPER_MODEL")
        or "base"  # Freeはbase推奨（smallはOOMしやすい）
    )

def _resolve_compute_type(device: str) -> str:
    ct = os.getenv("COMPUTE_TYPE") or os.getenv("WHISPER_COMPUTE_TYPE")
    if ct:
        return ct
    # 既定はGPU=float16, CPU=int8
    return "float16" if device == "cuda" else "int8"

# キャッシュ/焼き込み場所（Dockerfileで HF_HOME=/root/.cache を設定済み）
CACHE_DIR = os.getenv("HF_HOME", "/root/.cache")

_model = None

def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    device = _detect_device()
    name = _resolve_model_name()
    prefer = _resolve_compute_type(device)

    # CPU省メモリ向けの候補を複数試す
    candidates = (
        [prefer, "int8", "int8_float16", "float32"]
        if device == "cpu"
        else [prefer, "float32"]
    )

    # 省メモリ設定（Free対策）
    cpu_threads = int(os.getenv("CPU_THREADS", "1"))
    num_workers = int(os.getenv("NUM_WORKERS", "1"))

    last_err = None
    for ct in candidates:
        try:
            print(f"[whisper] loading model={name} device={device} compute_type={ct}")
            _model = WhisperModel(
                name,
                device=device,
                compute_type=ct,
                download_root=CACHE_DIR,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            print(f"[whisper] ready: {name} ({device}/{ct})")
            return _model
        except Exception as e:
            print(f"[whisper] failed compute_type={ct}: {e}")
            last_err = e
    raise RuntimeError(f"failed to load model; tried={candidates}: {last_err}")

def _to_wav16k(src: Path, dst: Path):
    # 1) ffmpeg
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return
    # 2) mac標準のafconvert
    if shutil.which("afconvert"):
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return
    # 3) ライブラリ最終手段
    try:
        import librosa, soundfile as sf
        y, _ = librosa.load(str(src), sr=16000, mono=True)
        sf.write(str(dst), y, 16000, subtype="PCM_16")
    except Exception as e:
        raise RuntimeError("no converter available (ffmpeg/afconvert/librosa)") from e

def transcribe_path(src_path: Path, language: str = "ja") -> Dict:
    tmp_wav = src_path.with_suffix(".16k.wav")
    _to_wav16k(src_path, tmp_wav)

    model = _load_model()
    segments, info = model.transcribe(
        str(tmp_wav),
        language=language or None,
        vad_filter=False,
        beam_size=1,  # 省メモリ
    )

    items = []
    for s in segments:
        items.append({
            "start": float(s.start or 0),
            "end": float(s.end or 0),
            "text": (s.text or "").strip()
        })
    try:
        tmp_wav.unlink()
    except Exception:
        pass

    text = "".join(i["text"] for i in items).strip()
    return {"language": info.language, "text": text, "segments": items}

def _ts(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}"

def to_srt(segments: List[Dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        lines.append(seg['text'])
        lines.append("")
    return "\n".join(lines)
