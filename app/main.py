# app/main.py
from pathlib import Path
import asyncio
import contextlib
import os
import tempfile
import wave

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles

from app.asr import transcribe_path, to_srt

# アプリのインスタンスを作成
app = FastAPI(title="Whisper Minimal API")

# --- CORS（必要に応じて許可ドメインを絞る）、CORS（クロスオリジンリソース共有） ---
# 別ドメインのフロントからAPIを呼べるようにCORSを緩く設定。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 静的配信 ---
# /ui で web フォルダの内容を返す。
# web/index.html がフロントエンドのUI。
app.mount("/ui", StaticFiles(directory="web", html=True), name="web")

# ルートは UI を返す（200）
ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT_DIR / "web" / "index.html"

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(INDEX_FILE)

# /ui → /ui/ に正規化（任意）
@app.get("/ui", include_in_schema=False)
def ui_redirect():
    return RedirectResponse(url="/ui/")

# faviconの404ログ抑止（任意）
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# --- 推論API（同時実行は1本に） ---
_transcribe_lock = asyncio.Lock()

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ja"),
    srt: str = Form("0"),  # "1"/"true"/"yes" でSRT返却
):
    async with _transcribe_lock:
        if not file.filename:
            raise HTTPException(400, "no filename")

        # 一時保存
        try:
            suffix = Path(file.filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(await file.read())
        except Exception as e:
            raise HTTPException(500, f"failed to store upload: {e}")

        # 文字起こし
        try:
            result = transcribe_path(tmp_path, language=language)
        except Exception as e:
            raise HTTPException(500, f"transcription failed: {e}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        # SRT指定ならテキストで返す
        if str(srt).lower() in {"1", "true", "yes"}:
            srt_text = to_srt(result["segments"])
            return PlainTextResponse(srt_text, media_type="text/plain; charset=utf-8")

        return JSONResponse(result)

# --- 起動時ウォームアップ（非同期。WARMUP=1 のときだけ実行） ---
app.state.ready = False

@app.on_event("startup")
async def startup():
    if os.getenv("WARMUP", "0") != "1":
        app.state.ready = True
        return
    asyncio.create_task(_warmup())

async def _warmup():
    silent_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            silent_path = Path(tmp.name)
        with contextlib.ExitStack() as stack:
            wf = stack.enter_context(wave.open(str(silent_path), "wb"))
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * int(16000 * 0.2))
        _ = transcribe_path(silent_path, language="ja")
    except Exception:
        pass
    finally:
        if silent_path:
            try:
                silent_path.unlink(missing_ok=True)
            except Exception:
                pass
        app.state.ready = True

# 任意：準備状況が見たいとき
@app.get("/ready", include_in_schema=False)
def ready():
    return {"ready": bool(getattr(app.state, "ready", False))}
