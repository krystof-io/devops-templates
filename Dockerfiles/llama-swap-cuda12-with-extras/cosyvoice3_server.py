#!/usr/bin/env python3
"""
FastAPI server wrapping CosyVoice 3 (Alibaba FunAudioLLM) for llama-swap integration.

Features:
  - Voice cloning from reference audio + transcript
  - Emotion/style control via text instructions ("instruct" field)
  - Combined clone + emotion in one generation (inference_instruct2)
  - Inline tags: [laughter], [breath], [quick_breath], [cough], [sigh],
    <laughter>text</laughter>, <strong>emphasis</strong>
  - ARPABET pronunciation control for English

Usage:
    python cosyvoice3_server.py --port 5000 --voices-dir /models/tts-voices
"""

import argparse
import io
import logging
import os
import sys
import time

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("cosyvoice3_server")

app = FastAPI(title="CosyVoice 3 Server")
cosyvoice_model = None
voices_dir = None
sample_rate = 24000

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


class SpeechRequest(BaseModel):
    model: str = "cosyvoice3"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    instruct: str | None = None
    ref_text: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.get("/health")
def health():
    if cosyvoice_model is None:
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
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ref_audio_path = _resolve_voice(req.voice)
    if ref_audio_path is None:
        raise HTTPException(status_code=400, detail=f"Voice '{req.voice}' not found")

    info = sf.info(ref_audio_path)
    if info.duration < 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"Reference audio '{req.voice}' is too short ({info.duration:.2f}s). CosyVoice 3 needs at least 1 second.",
        )

    instruct = req.instruct

    try:
        import torch

        all_audio = []

        if instruct:
            if not instruct.endswith("<|endofprompt|>"):
                instruct = instruct + "<|endofprompt|>"
            logger.info(f"Generating (instruct): voice={req.voice}, instruct={instruct!r}")
            for result in cosyvoice_model.inference_instruct2(
                tts_text=req.input,
                instruct_text=instruct,
                prompt_wav=ref_audio_path,
                stream=False,
                speed=req.speed,
            ):
                all_audio.append(result["tts_speech"])
        else:
            ref_text = req.ref_text or _load_ref_text(ref_audio_path)
            if ref_text:
                logger.info(f"Generating (zero-shot): voice={req.voice}")
                for result in cosyvoice_model.inference_zero_shot(
                    tts_text=req.input,
                    prompt_text=ref_text,
                    prompt_wav=ref_audio_path,
                    stream=False,
                    speed=req.speed,
                    ):
                    all_audio.append(result["tts_speech"])
            else:
                logger.info(f"Generating (cross-lingual, no ref_text): voice={req.voice}")
                for result in cosyvoice_model.inference_cross_lingual(
                    tts_text=req.input,
                    prompt_wav=ref_audio_path,
                    stream=False,
                    speed=req.speed,
                    ):
                    all_audio.append(result["tts_speech"])

        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio_tensor = torch.cat(all_audio, dim=1)
        audio_np = audio_tensor.numpy().flatten()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CosyVoice 3 inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
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


def _ensure_model_downloaded(model_dir: str):
    if os.path.isfile(os.path.join(model_dir, "cosyvoice3.yaml")):
        return
    logger.info(f"Downloading Fun-CosyVoice3-0.5B-2512 to {model_dir}...")
    from huggingface_hub import snapshot_download

    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=model_dir)
    logger.info("Download complete")


def main():
    global cosyvoice_model, voices_dir, sample_rate

    repo_dir = os.environ.get("COSYVOICE_REPO", "/app/cosyvoice-repo")
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "third_party", "Matcha-TTS"))

    parser = argparse.ArgumentParser(description="CosyVoice 3 FastAPI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--voices-dir", type=str, default="/models/tts-voices")
    parser.add_argument("--model-dir", type=str, default="/models/cosyvoice3")
    args = parser.parse_args()

    voices_dir = args.voices_dir

    _ensure_model_downloaded(args.model_dir)

    logger.info("Loading CosyVoice 3 model...")
    start = time.time()
    from cosyvoice.cli.cosyvoice import AutoModel

    cosyvoice_model = AutoModel(model_dir=args.model_dir, fp16=True)
    sample_rate = cosyvoice_model.sample_rate
    logger.info(f"CosyVoice 3 loaded in {time.time() - start:.1f}s (sample_rate={sample_rate})")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
