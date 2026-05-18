#!/usr/bin/env python3
"""
Orpheus TTS server for llama-swap integration.

Spawns llama-server internally with the Orpheus GGUF model for token generation,
then decodes the SNAC audio tokens using the SNAC neural codec.

Usage:
    python orpheus_server.py --port 5000 --hf-repo unsloth/orpheus-3b-0.1-ft-GGUF --hf-file orpheus-3b-0.1-ft-Q8_0.gguf
    python orpheus_server.py --port 5000 --model /models/orpheus/orpheus-3b-q8_0.gguf
"""

import argparse
import atexit
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
import wave

import numpy as np
import requests as http_requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from snac import SNAC

logger = logging.getLogger("orpheus_server")

app = FastAPI(title="Orpheus TTS Server")

snac_model = None
llama_process = None
llama_url = None

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
SAMPLE_RATE = 24000


class SpeechRequest(BaseModel):
    model: str = "orpheus"
    input: str
    voice: str = "tara"
    response_format: str = "wav"
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 8192


@app.get("/health")
def health():
    if snac_model is None or llama_process is None:
        raise HTTPException(status_code=503, detail="Not ready")
    if llama_process.poll() is not None:
        raise HTTPException(status_code=503, detail="llama-server exited")
    return {"status": "ok"}


@app.get("/v1/audio/voices")
def list_voices():
    return {"voices": VOICES}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if snac_model is None:
        raise HTTPException(status_code=503, detail="SNAC model not loaded")

    voice = req.voice.lower() if req.voice else "tara"
    prompt = f"<|audio|>{voice}: {req.input}<|eot_id|>"

    payload = {
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "repeat_penalty": 1.1,
        "stream": True,
    }

    try:
        resp = http_requests.post(
            f"{llama_url}/v1/completions",
            json=payload,
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llama-server error: {e}")

    token_ids = []
    token_index = 0

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        token_text = data.get("choices", [{}])[0].get("text", "")
        for part in token_text.split(">"):
            part = part.strip()
            if not part.startswith("<custom_token_"):
                continue
            try:
                raw_id = int(part[14:])
                token_id = raw_id - 10 - ((token_index % 7) * 4096)
                if token_id > 0:
                    token_ids.append(token_id)
                    token_index += 1
            except (ValueError, IndexError):
                continue

    if len(token_ids) < 7:
        raise HTTPException(status_code=500, detail="Too few audio tokens generated")

    try:
        pcm_bytes = decode_snac(token_ids)
    except Exception as e:
        logger.exception("SNAC decoding failed")
        raise HTTPException(status_code=500, detail=f"Audio decoding failed: {e}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def decode_snac(token_ids):
    num_frames = len(token_ids) // 7
    tokens = token_ids[: num_frames * 7]
    device = next(snac_model.parameters()).device

    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=device)

    for j in range(num_frames):
        i = j * 7
        codes_0[j] = tokens[i]
        codes_1[j * 2] = tokens[i + 1]
        codes_1[j * 2 + 1] = tokens[i + 4]
        codes_2[j * 4] = tokens[i + 2]
        codes_2[j * 4 + 1] = tokens[i + 3]
        codes_2[j * 4 + 2] = tokens[i + 5]
        codes_2[j * 4 + 3] = tokens[i + 6]

    codes = [c.unsqueeze(0) for c in [codes_0, codes_1, codes_2]]

    for c in codes:
        if torch.any(c < 0) or torch.any(c > 4095):
            raise ValueError("SNAC token ID out of valid range")

    with torch.inference_mode():
        audio = snac_model.decode(codes)
        audio_int16 = (audio.squeeze() * 32767).clamp(-32768, 32767).to(torch.int16)
        return audio_int16.cpu().numpy().tobytes()


def wait_for_server(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = http_requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except http_requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def main():
    global snac_model, llama_process, llama_url

    parser = argparse.ArgumentParser(description="Orpheus TTS server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--llama-server", type=str, default="/app/llama-server")
    parser.add_argument("--model", type=str, default=None, help="Local GGUF path")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-file", type=str, default=None)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=8192)
    args = parser.parse_args()

    if not args.model and not (args.hf_repo and args.hf_file):
        parser.error("Provide --model or both --hf-repo and --hf-file")

    llama_port = args.port + 1
    llama_url = f"http://127.0.0.1:{llama_port}"

    logger.info("Loading SNAC decoder model...")
    start = time.time()
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
    logger.info(f"SNAC loaded on {snac_device} in {time.time() - start:.1f}s")

    llama_cmd = [
        args.llama_server,
        "--ctx-size", str(args.ctx_size),
        "--n-predict", str(args.ctx_size),
        "--gpu-layers", str(args.gpu_layers),
        "--port", str(llama_port),
        "--host", "127.0.0.1",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
    ]
    if args.model:
        llama_cmd += ["--model", args.model]
    else:
        llama_cmd += ["--hf-repo", args.hf_repo, "--hf-file", args.hf_file]

    logger.info(f"Starting llama-server: {' '.join(llama_cmd)}")
    llama_process = subprocess.Popen(llama_cmd, stdout=sys.stdout, stderr=sys.stderr)

    def cleanup(signum=None, frame=None):
        if llama_process and llama_process.poll() is None:
            llama_process.terminate()
            try:
                llama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llama_process.kill()

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    atexit.register(cleanup)

    logger.info("Waiting for llama-server to be ready...")
    if not wait_for_server(llama_url, timeout=120):
        logger.error("llama-server failed to start within 120s")
        cleanup()
        sys.exit(1)
    logger.info("llama-server is ready")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
