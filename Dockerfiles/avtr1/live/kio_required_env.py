# Every runtime setting of the live pod, and the rule that none of them has a
# default inside the image: each must be passed in by Kubernetes, or the pod
# refuses to start and says which ones are missing.
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build (on the import
# path of both pixi environments). Standard library only.
#
# "Set" means present in the environment and non-empty, except for the few in
# ALLOW_EMPTY where empty is a real choice (KIO_VIDEO_CODEC="" follows the browser).
# check() is called once at start-up by the launcher and by the streamer.

from __future__ import annotations

import json
import os

_STR, _INT, _FLOAT, _FLAG, _JSON = "text", "integer", "number", '"0" or "1"', "JSON object"

# name -> (kind, what it is)
REQUIRED: dict[str, tuple[str, str]] = {
    # process wiring
    "STREAMER_HOST": (_STR, "address the page/signalling server binds to"),
    "STREAMER_PORT": (_INT, "port the page/signalling server listens on"),
    "RENDERER_PORT": (_INT, "port of the renderer API inside the pod (bound to 127.0.0.1)"),
    "PIXI_FROZEN": (_STR, '"true" so pixi never re-solves the locked environments'),
    # storage and Hugging Face posture
    "AVTR1_LOCAL_STORAGE": (_STR, "directory holding the model artifacts and avatar_settings.json"),
    "HF_HOME": (_STR, "Hugging Face cache directory"),
    "HF_HUB_OFFLINE": (_FLAG, '"1" so nothing is fetched from the Hub at run time'),
    "HF_HUB_DISABLE_TELEMETRY": (_FLAG, '"1" to disable Hub telemetry'),
    # WebRTC. There are no STUN/TURN settings: the streamer is patched to use none.
    "KIO_WEBRTC_PUBLIC_IP": (_STR, "LoadBalancer address advertised to browsers for media"),
    "KIO_WEBRTC_UDP_PORTS": (_STR, 'UDP port range for media, e.g. "40000-40009"; must match the Service'),
    "KIO_VIDEO_CODEC": (_STR, '"VP8", "H264" or "" to follow the browser\'s order'),
    "KIO_READY_TIMEOUT": (_FLOAT, "seconds a new session may take to connect, and to wait for the browser's first packet"),
    "KIO_IDLE_TIMEOUT": (_INT, "seconds without activity before a session is ended"),
    "KIO_SESSION_TAKEOVER": (_FLAG, '"1" = a new session ends the running one'),
    # text-to-speech
    "KIO_TTS_URL": (_STR, "OpenAI-style /v1/audio/speech endpoint"),
    "KIO_TTS_MODEL": (_STR, "TTS model name"),
    "KIO_TTS_VOICE": (_STR, "voice used when an avatar has none saved"),
    "KIO_TTS_SAMPLE_RATE": (_INT, "sample rate of the raw audio the model streams (VoxCPM2 48000, Higgs/OmniVoice 24000)"),
    "KIO_TTS_EXTRA_JSON": (_JSON, 'extra fields for every TTS request, "{}" for none'),
    "KIO_TTS_STYLE_TEMPLATE": (_STR, "how a style direction is joined to the text, using {style} and {text}"),
    # speech-to-text
    "KIO_STT_URL": (_STR, "OpenAI-style /v1/audio/transcriptions endpoint"),
    "KIO_STT_MODEL": (_STR, "speech-to-text model name"),
    # LLM
    "KIO_LLM_URL": (_STR, "OpenAI-style /v1/chat/completions endpoint"),
    "KIO_LLM_MODEL": (_STR, "LLM model name"),
    "KIO_LLM_API_KEY": (_STR, 'bearer token for the LLM; "" if it needs none'),
    "KIO_LLM_EXTRA_JSON": (_JSON, 'extra fields for every chat request, "{}" for none'),
    "KIO_LLM_MAX_TOKENS": (_INT, "reply length limit"),
    "KIO_SYSTEM_PROMPT": (_STR, "persona prompt used when an avatar has none saved"),
    "KIO_HISTORY_TURNS": (_INT, "number of past messages sent with each request"),
    # end-of-speech detection
    "KIO_VAD_MIN_RMS": (_FLOAT, "loudness below which nothing counts as speech"),
    "KIO_VAD_FLOOR_FACTOR": (_FLOAT, "speech threshold as a multiple of the background level"),
    "KIO_VAD_START_MS": (_INT, "loud for this long = speech started"),
    "KIO_VAD_END_MS": (_INT, "quiet for this long = the turn is over"),
    "KIO_VAD_MIN_SPEECH_MS": (_INT, "shorter bursts are ignored"),
    "KIO_VAD_MAX_SPEECH_S": (_FLOAT, "longest single utterance"),
    "KIO_BARGE_IN": (_FLAG, '"1" = speaking over the avatar interrupts it'),
    "KIO_BARGE_IN_FACTOR": (_FLOAT, "how much louder an interruption must be"),
    "KIO_BARGE_IN_START_MS": (_INT, "how long an interruption must last"),
}

# Settings where an empty value is a deliberate choice. Everything else must be non-empty.
ALLOW_EMPTY = {"KIO_VIDEO_CODEC", "KIO_LLM_API_KEY", "KIO_WEBRTC_PUBLIC_IP"}

# Read if present; each has a behaviour, not a value, when absent.
OPTIONAL: dict[str, str] = {
    "KIO_TTS_VOICES_URL": "voice-list endpoint; when absent it is derived from KIO_TTS_URL and KIO_TTS_MODEL",
    "KIO_AVATAR_SETTINGS_FILE": "settings file path; when absent it is AVTR1_LOCAL_STORAGE/avatar_settings.json",
}


class MissingSettings(SystemExit):
    pass


def get(name: str) -> str | None:
    """The raw value, or None if Kubernetes did not pass it. Never a default."""
    if name not in REQUIRED and name not in OPTIONAL:
        raise KeyError(f"{name} is not a declared setting (add it to kio_required_env.py)")
    return os.environ.get(name)


def text(name: str) -> str:
    return get(name) or ""


def number(name: str, cast=float):
    """Parsed value; 0 when unset or unparsable. check() has already refused to start in that case."""
    try:
        return cast(get(name))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return cast(0)


def flag(name: str) -> bool:
    return get(name) == "1"


def obj(name: str) -> dict:
    try:
        value = json.loads(get(name) or "")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def problems() -> list[str]:
    out: list[str] = []
    for name, (kind, what) in REQUIRED.items():
        raw = os.environ.get(name)
        if raw is None:
            out.append(f"  {name}: not set  ({what})")
            continue
        if raw == "" and name not in ALLOW_EMPTY:
            out.append(f"  {name}: set but empty  ({what})")
            continue
        try:
            if kind == _INT:
                int(raw)
            elif kind == _FLOAT:
                float(raw)
            elif kind == _FLAG and raw not in ("0", "1"):
                raise ValueError
            elif kind == _JSON and not isinstance(json.loads(raw), dict):
                raise ValueError
        except ValueError:
            out.append(f"  {name}: {raw!r} is not a valid {kind}  ({what})")
    return out


def check(who: str) -> None:
    """Refuse to start unless every setting was passed in and parses."""
    bad = problems()
    if bad:
        raise MissingSettings(
            f"{who}: refusing to start. This image has no built-in defaults; every setting must be "
            f"passed in by Kubernetes (manual/avtr1/deployment.yaml).\n"
            f"{len(bad)} problem(s):\n" + "\n".join(bad)
        )
