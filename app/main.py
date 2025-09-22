# app/main.py
from pathlib import Path
import tempfile
import os
import wave
import contextlib

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.asr import transcribe_path, to_srt

app = FastAPI(title="Whisper Minimal API")

# --- CORS（別ドメインのフロントから叩く可能性を考慮） ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 本番はドメインを絞るのがおすすめ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 静的配信（/ui/ で web/index.html を返す） ---
app.mount("/ui", StaticFiles(directory="web", html=True), name="web")

# --- Health/Root ---
@app.get("/", include_in_schema=False)
def health():
    # RenderのhealthCheckPathに合わせて200を返す
    return {"ok": True}

# /ui → /ui/ に正規化（307が気になる場合の明示対応）
@app.get("/ui", include_in_schema=False)
def ui_redirect():
    return RedirectResponse(url="/ui/")

# faviconの404ログを抑止（任意）
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# --- API ---
@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ja"),
    srt: str = Form("0"),  # "1" / "true" / "yes" ならSRTで返す
):
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

    # 音声→文字起こし
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
    srt_flag = str(srt).lower() in {"1", "true", "yes"}
    if srt_flag:
        srt_text = to_srt(result["segments"])
        return PlainTextResponse(srt_text, media_type="text/plain; charset=utf-8")

    return JSONResponse(result)

# --- 起動時ウォームアップ（Freeで“初回固まり”軽減） ---
@app.on_event("startup")
def warmup():
    # 環境変数 WARMUP=0 でスキップ可能（モデルをDockerに“焼き込み”済みなら省略してOK）
    if os.getenv("WARMUP", "1") != "1":
        return
    silent_path: Path | None = None
    try:
        # 0.2秒の無音WAV(16kHz/mono)を作って叩く → モデルDL/初期化を先に済ませる
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            silent_path = Path(tmp.name)
        with contextlib.ExitStack() as stack:
            wf = stack.enter_context(wave.open(str(silent_path), "wb"))
            wf.setnchannels(1)
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * int(16000 * 0.2))
        # モデルロード＆推論の初回コストをここで払う
        _ = transcribe_path(silent_path, language="ja")
    except Exception:
        # ウォームアップ失敗は致命ではないので握りつぶす
        pass
    finally:
        if silent_path:
            try:
                silent_path.unlink(missing_ok=True)
            except Exception:
                pass
