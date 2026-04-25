#!/usr/bin/env python3
"""
FastAPI server wrapping Qwen3-TTS for llama-swap integration.

Supports two modes:
  - voice-design: describe a voice in natural language via the 'voice' field
  - clone: clone from reference audio files in --voices-dir

Usage:
    python qwen3tts_server.py --port 5000 --mode voice-design
    python qwen3tts_server.py --port 5000 --mode clone --voices-dir /models/tts-voices
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

logger = logging.getLogger("qwen3tts_server")

app = FastAPI(title="Qwen3-TTS Server")
tts_model = None
server_mode = None
voices_dir = None

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

MODEL_IDS = {
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "A warm, friendly voice with moderate pace."
    language: str = "English"
    response_format: str = "wav"
    ref_text: str | None = None


@app.get("/health")
def health():
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/v1/audio/voices")
def list_voices():
    if server_mode == "voice-design":
        return {"voices": [
            "A warm, friendly male voice with moderate pace.",
            "A young female voice, energetic and cheerful.",
            "A deep, authoritative narrator voice.",
            "Speak with excitement and wonder in your voice.",
            "A calm, soothing voice for bedtime stories.",
            "An elderly gentleman with a slow, deliberate cadence.",
            "Speak in an incredulous tone with a hint of panic.",
        ]}
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

    try:
        if server_mode == "voice-design":
            wavs, sr = tts_model.generate_voice_design(
                text=req.input,
                language=req.language,
                instruct=req.voice,
            )
        else:
            ref_audio_path = _resolve_voice(req.voice)
            ref_text = req.ref_text or _load_ref_text(ref_audio_path)
            clone_kwargs = dict(
                text=req.input,
                language=req.language,
                ref_audio=ref_audio_path,
            )
            if ref_text:
                clone_kwargs["ref_text"] = ref_text
            else:
                clone_kwargs["x_vector_only_mode"] = True
                logger.info("No ref_text available, using x_vector_only_mode (speaker embedding only)")
            wavs, sr = tts_model.generate_voice_clone(**clone_kwargs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Qwen3-TTS inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
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


def _resolve_voice(voice: str) -> str:
    if os.path.isabs(voice) and os.path.isfile(voice):
        return voice

    if voices_dir is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' is not an absolute path and --voices-dir is not set",
        )

    if voice.lower().endswith(AUDIO_EXTENSIONS):
        path = os.path.join(voices_dir, voice)
        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail=f"Voice file not found: {path}")
        return path

    for ext in AUDIO_EXTENSIONS:
        path = os.path.join(voices_dir, voice + ext)
        if os.path.isfile(path):
            return path

    raise HTTPException(
        status_code=400,
        detail=f"Voice '{voice}' not found in {voices_dir} (tried {', '.join(AUDIO_EXTENSIONS)})",
    )


def main():
    global tts_model, server_mode, voices_dir

    import torch

    parser = argparse.ArgumentParser(description="Qwen3-TTS FastAPI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--mode", type=str, choices=["voice-design", "clone"], required=True)
    parser.add_argument("--voices-dir", type=str, default="/models/tts-voices")
    args = parser.parse_args()

    server_mode = args.mode
    voices_dir = args.voices_dir
    model_id = MODEL_IDS[args.mode]

    logger.info(f"Loading Qwen3-TTS model ({args.mode}): {model_id}")
    start = time.time()
    from qwen_tts import Qwen3TTSModel

    tts_model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    logger.info(f"Qwen3-TTS loaded in {time.time() - start:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
