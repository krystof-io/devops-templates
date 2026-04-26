#!/usr/bin/env python3
"""
FastAPI server wrapping Chatterbox TTS (Resemble AI) for llama-swap integration.

Features:
  - Voice cloning from reference audio files in --voices-dir
  - Emotion/expressiveness via exaggeration parameter (0.0-2.0)
  - Pace control via cfg_weight (0.0=slow/deliberate, 1.0=fast)

Usage:
    python chatterbox_server.py --port 5000 --voices-dir /models/tts-voices
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

logger = logging.getLogger("chatterbox_server")

app = FastAPI(title="Chatterbox TTS Server")
tts_model = None
voices_dir = None

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)


@app.get("/health")
def health():
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/v1/audio/voices")
def list_voices():
    if voices_dir is None or not os.path.isdir(voices_dir):
        return {"voices": []}
    voices = []
    for f in sorted(os.listdir(voices_dir)):
        if f.lower().endswith(AUDIO_EXTENSIONS):
            voices.append(os.path.splitext(f)[0])
    return {"voices": voices}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ref_audio_path = _resolve_voice(req.voice)

    logger.info(
        f"Generating: voice={req.voice}, exaggeration={req.exaggeration}, "
        f"cfg_weight={req.cfg_weight}, ref={ref_audio_path}"
    )

    try:
        wav = tts_model.generate(
            text=req.input,
            audio_prompt_path=ref_audio_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
        )
    except Exception as e:
        logger.exception("Chatterbox inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    audio_np = wav.squeeze(0).cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, tts_model.sr, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def _resolve_voice(voice: str) -> str | None:
    if voice == "default":
        return None
    if os.path.isabs(voice) and os.path.isfile(voice):
        return voice
    if voices_dir is None:
        return None
    if voice.lower().endswith(AUDIO_EXTENSIONS):
        path = os.path.join(voices_dir, voice)
        if os.path.isfile(path):
            return path
        return None
    for ext in AUDIO_EXTENSIONS:
        path = os.path.join(voices_dir, voice + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    global tts_model, voices_dir

    parser = argparse.ArgumentParser(description="Chatterbox TTS FastAPI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--voices-dir", type=str, default="/models/tts-voices")
    args = parser.parse_args()

    voices_dir = args.voices_dir

    logger.info("Loading Chatterbox model...")
    start = time.time()
    from chatterbox.tts import ChatterboxTTS

    tts_model = ChatterboxTTS.from_pretrained(device="cuda")
    logger.info(f"Chatterbox loaded in {time.time() - start:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
