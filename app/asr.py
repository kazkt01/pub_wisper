import os, shutil, subprocess
from pathlib import Path
from typing import List, Dict
from faster_whisper import WhisperModel

# 初回確認を軽くしたいので既定は small（必要なら環境変数で上書き）
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_model = None

def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _pick_compute_type(device: str) -> str:
    # 明示指定があれば優先
    if ct := os.getenv("WHISPER_COMPUTE_TYPE"):
        return ct
    # GPUはfloat16、CPUはint8が一番無難
    return "float16" if device == "cuda" else "int8"

def _load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    device = _detect_device()
    prefer = _pick_compute_type(device)
    # CPU向けフォールバック候補
    candidates = [prefer, "int8", "int8_float32", "float32"] if device == "cpu" else [prefer, "float32"]
    last_err = None
    for ct in candidates:
        try:
            print(f"[whisper] loading model={MODEL_NAME} device={device} compute_type={ct}")
            _model = WhisperModel(MODEL_NAME, device=device, compute_type=ct)
            print(f"[whisper] ready: {MODEL_NAME} ({device}/{ct})")
            return _model
        except Exception as e:
            print(f"[whisper] failed compute_type={ct}: {e}")
            last_err = e
    raise RuntimeError(f"failed to load model; tried={candidates}: {last_err}")

def _to_wav16k(src: Path, dst: Path):
    # 1) ffmpeg（推奨）
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
    # 3) ライブラリでフォールバック（最終手段）
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
    # 依存を増やさない最小構成：VADなし・beam軽め
    segments, info = model.transcribe(
        str(tmp_wav),
        language=language or None,
        vad_filter=False,
        beam_size=1,
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
