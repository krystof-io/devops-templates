#!/usr/bin/env python3
"""
Minimal FastAPI server wrapping F5-TTS for llama-swap integration.

Usage:
    python f5tts_server.py --port 5000 --voices-dir /models/tts-voices
"""

import argparse
import io
import logging
import os
import time

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("f5tts_server")

app = FastAPI(title="F5-TTS Server")
tts_model = None
voices_dir = None


class SpeechRequest(BaseModel):
    model: str = "f5-tts"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    ref_text: str | None = None
    nfe_step: int = Field(default=32, ge=4, le=128)
    seed: int | None = None


@app.get("/health")
def health():
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ref_audio_path = _resolve_voice(req.voice)

    ref_text = req.ref_text
    if not ref_text:
        ref_text = tts_model.transcribe(ref_audio_path)

    try:
        wav, sr, _ = tts_model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=req.input,
            nfe_step=req.nfe_step,
            speed=req.speed,
            seed=req.seed,
            show_info=logger.info,
            progress=None,
        )
    except Exception as e:
        logger.exception("F5-TTS inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def _resolve_voice(voice: str) -> str:
    if os.path.isabs(voice) and os.path.isfile(voice):
        return voice

    if voices_dir is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' is not an absolute path and --voices-dir is not set",
        )

    if not voice.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        voice = voice + ".wav"

    path = os.path.join(voices_dir, voice)
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"Voice file not found: {path}")
    return path


def main():
    global tts_model, voices_dir

    parser = argparse.ArgumentParser(description="F5-TTS FastAPI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--voices-dir", type=str, default="/models/tts-voices")
    args = parser.parse_args()

    voices_dir = args.voices_dir

    logger.info("Loading F5-TTS model...")
    start = time.time()
    from f5_tts.api import F5TTS
    tts_model = F5TTS(model="F5TTS_v1_Base")
    logger.info(f"F5-TTS model loaded in {time.time() - start:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
