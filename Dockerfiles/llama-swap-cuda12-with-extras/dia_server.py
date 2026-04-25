#!/usr/bin/env python3
"""
FastAPI server wrapping Dia TTS (Nari Labs) for llama-swap integration.

Features:
  - Voice cloning from reference audio files in --voices-dir
  - Inline nonverbal tags: (laughs), (sighs), (gasps), (coughs), etc.
  - Multi-speaker with [S1]/[S2] tags

Usage:
    python dia_server.py --port 5000 --voices-dir /models/tts-voices
"""

import argparse
import io
import logging
import os
import time

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("dia_server")

app = FastAPI(title="Dia TTS Server")
tts_model = None
voices_dir = None

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
SAMPLE_RATE = 44100


class SpeechRequest(BaseModel):
    model: str = "dia"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    temperature: float = Field(default=1.2, ge=0.1, le=2.0)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)


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

    text = req.input
    if not text.startswith("[S1]") and not text.startswith("[S2]"):
        text = "[S1] " + text

    audio_prompt = None
    ref_audio_path = _resolve_voice(req.voice)
    if ref_audio_path:
        ref_text = _load_ref_text(ref_audio_path)
        if ref_text:
            if not ref_text.startswith("[S1]") and not ref_text.startswith("[S2]"):
                ref_text = "[S1] " + ref_text
            text = ref_text + " " + text
        audio_prompt = ref_audio_path

    try:
        audio = tts_model.generate(
            text=text,
            audio_prompt=audio_prompt,
            cfg_scale=req.cfg_scale,
            temperature=req.temperature,
            top_p=req.top_p,
            verbose=True,
        )
    except Exception as e:
        logger.exception("Dia inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if isinstance(audio, list):
        audio = audio[0]

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def _load_ref_text(audio_path: str) -> str | None:
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    if os.path.isfile(txt_path):
        text = open(txt_path).read().strip()
        if text:
            logger.info(f"Loaded ref_text from {txt_path}")
            return text
    return None


def _resolve_voice(voice: str) -> str | None:
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

    parser = argparse.ArgumentParser(description="Dia TTS FastAPI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--voices-dir", type=str, default="/models/tts-voices")
    args = parser.parse_args()

    voices_dir = args.voices_dir

    logger.info("Loading Dia model...")
    start = time.time()
    from dia.model import Dia

    tts_model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B-0626",
        compute_dtype="float16",
    )
    logger.info(f"Dia loaded in {time.time() - start:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
