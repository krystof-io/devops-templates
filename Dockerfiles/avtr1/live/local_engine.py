# Local conversation engine for the AVTR-1 streamer.
#
# Not upstream code. Copied into src/avaturn_live_streamer/conversation_engines/
# at image build.
#
# One turn:
#   microphone audio (UserSpeechReceived, 20 ms pieces)
#     -> loudness-based end-of-speech detection
#     -> speech-to-text   (OpenAI-style /v1/audio/transcriptions, WAV)
#     -> LLM              (OpenAI-style /v1/chat/completions, streamed)
#     -> text-to-speech   (one request per sentence, raw PCM streamed back)
#     -> SegmentGenerationStarted / SegmentChunkGenerated / SegmentGenerationCompleted
#        on the event bus, exactly as the OpenAI Realtime engine publishes audio.
#
# Talking while she is speaking (barge-in) cancels the reply in progress and
# discards her queued speech. Typed input still works through POST /say:
# mode "say" speaks the text verbatim, mode "ask" sends it to the LLM as if spoken.
#
# Everything tunable is an environment variable set in the Deployment, so the
# usual adjustments need a pod restart, not an image rebuild. There are no
# defaults in this file: see kio_required_env.py.

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import uuid
import wave
from asyncio import TaskGroup
from collections import deque
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel

from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    UserSpeechReceived,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

import kio_avatar_settings as _settings  # per-avatar tuning, shared with the renderer
import kio_required_env as _required  # every setting comes from Kubernetes; no defaults in the image

_LOGGER = get_logger()


# No defaults live here: every value below comes from the environment, and the
# streamer refuses to start (kio_required_env.check) if Kubernetes did not pass
# one. The readers return an empty/zero placeholder for a missing value only so
# this module can still be imported by the image build's import test.
_cfg = _required.text
_num = _required.number

# --- text-to-speech -----------------------------------------------------------
TTS_URL = _cfg("KIO_TTS_URL")
TTS_MODEL = _cfg("KIO_TTS_MODEL")
TTS_VOICE = _cfg("KIO_TTS_VOICE")
# Sample rate of the headerless 16-bit mono PCM the model streams back
# (VoxCPM2 48 kHz; Higgs and OmniVoice 24 kHz).
TTS_SAMPLE_RATE = _num("KIO_TTS_SAMPLE_RATE", int)
_MIN_CHUNK_BYTES = TTS_SAMPLE_RATE * 2 // 5  # publish in pieces of about 200 ms
# Extra fields merged into every TTS request. Higgs picks a random seed per
# request unless one is given, and this engine sends one request per sentence,
# so a fixed seed there keeps the voice steady: {"seed": 1234}.
TTS_EXTRA = _required.obj("KIO_TTS_EXTRA_JSON")
# How a per-avatar style direction is combined with the text. VoxCPM2 reads a
# leading parenthesised phrase: "({style}) {text}". Higgs would speak that
# aloud; its style is inline tags at the start, e.g. <|emotion:amusement|>, so
# it wants "{style}{text}" with the tags themselves in the style field.
TTS_STYLE_TEMPLATE = _cfg("KIO_TTS_STYLE_TEMPLATE")

# Where to ask which voices exist. The audio.cpp server answers
# GET /v1/audio/voices, but llama-swap only routes a request to it when the model
# is named, so unless KIO_TTS_VOICES_URL says otherwise the address is built from
# the two values above through llama-swap's /upstream/<model>/ path.
_SPEECH_SUFFIX = "/v1/audio/speech"
TTS_VOICES_URL = _required.get("KIO_TTS_VOICES_URL") or (
    (TTS_URL[: -len(_SPEECH_SUFFIX)] + f"/upstream/{TTS_MODEL}/v1/audio/voices") if TTS_URL.endswith(_SPEECH_SUFFIX) else ""
)
_voices_cache: dict = {"at": 0.0, "result": None}


async def list_voices() -> dict[str, object]:
    """Voices the TTS server offers, for the tuning panel's dropdown. Cached for 30 s."""
    now = time.monotonic()
    if _voices_cache["result"] is not None and now - _voices_cache["at"] < 30:
        return _voices_cache["result"]
    result: dict[str, object] = {"voices": [], "default": TTS_VOICE, "model": TTS_MODEL, "source": TTS_VOICES_URL}
    if not TTS_VOICES_URL:
        result["error"] = "no voices URL configured"
    else:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as http:
                r = await http.get(TTS_VOICES_URL)
                r.raise_for_status()
                names = r.json().get("voices")
            result["voices"] = sorted(str(v) for v in names if isinstance(v, str)) if isinstance(names, list) else []
        except Exception as exc:  # noqa: BLE001 - the panel falls back to a text box
            result["error"] = f"{type(exc).__name__}: {exc}"
    if not result.get("error"):
        _voices_cache.update(at=now, result=result)
    return result


# --- speech-to-text -----------------------------------------------------------
STT_URL = _cfg("KIO_STT_URL")
STT_MODEL = _cfg("KIO_STT_MODEL")
STT_SAMPLE_RATE = 16000  # what the transcription request is resampled to; not a deployment setting

# --- LLM ----------------------------------------------------------------------
LLM_URL = _cfg("KIO_LLM_URL")
LLM_MODEL = _cfg("KIO_LLM_MODEL")
LLM_API_KEY = _cfg("KIO_LLM_API_KEY")
# Merged into every chat request, e.g. {"reasoning_effort": "none"} to turn a
# model's thinking phase off (strata: first word in ~0.1-0.4 s instead of ~1.2 s).
LLM_EXTRA = _required.obj("KIO_LLM_EXTRA_JSON")
LLM_MAX_TOKENS = _num("KIO_LLM_MAX_TOKENS", int)
SYSTEM_PROMPT = _cfg("KIO_SYSTEM_PROMPT")
HISTORY_TURNS = _num("KIO_HISTORY_TURNS", int)  # user+assistant messages kept

# --- end-of-speech detection ----------------------------------------------------
# Loudness is the RMS of each 20 ms microphone piece, on a 0..1 scale.
VAD_MIN_RMS = _num("KIO_VAD_MIN_RMS")  # never trigger below this
VAD_FLOOR_FACTOR = _num("KIO_VAD_FLOOR_FACTOR")  # x the measured background level
VAD_START_MS = _num("KIO_VAD_START_MS", int)  # loud for this long = speech started
VAD_END_MS = _num("KIO_VAD_END_MS", int)  # quiet for this long = you finished
VAD_MIN_SPEECH_MS = _num("KIO_VAD_MIN_SPEECH_MS", int)  # shorter bursts are ignored
VAD_MAX_SPEECH_S = _num("KIO_VAD_MAX_SPEECH_S")
VAD_PREROLL_MS = 300  # audio kept from just before speech was detected; not a deployment setting
# While she is speaking, her voice leaks from your speakers into the microphone.
# The browser's echo cancellation removes most of it; these make an interruption
# need to be clearly louder and longer than the leak.
BARGE_IN = _required.flag("KIO_BARGE_IN")
BARGE_IN_FACTOR = _num("KIO_BARGE_IN_FACTOR")
BARGE_IN_START_MS = _num("KIO_BARGE_IN_START_MS", int)

# The four conversation settings above are also adjustable per avatar from the
# tuning panel. Their defaults there are this pod's environment values.
_settings.set_default("vad_end_ms", VAD_END_MS)
_settings.set_default("barge_in", BARGE_IN)
_settings.set_default("barge_in_factor", BARGE_IN_FACTOR)
_settings.set_default("llm_max_tokens", LLM_MAX_TOKENS)

# Avatar of the current session (one session at a time), set by the /offer route.
_session_avatar = ""


def set_session_avatar(avatar_id: str) -> None:
    global _session_avatar
    _session_avatar = avatar_id


def _tuned(key: str):
    """This avatar's saved value for a conversation setting, read fresh (the file is re-read when it changes)."""
    return _settings.get(_session_avatar, key)


# Whisper's usual outputs for silence or noise; not worth a reply.
_STT_JUNK = re.compile(r"^[\s\[\(]*(blank_audio|silence|music|no speech|inaudible|applause)[\s\]\)]*$", re.I)
_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")


class LocalEngineOptions(BaseModel):
    type: Literal["local"] = "local"
    voice: str = TTS_VOICE


class LocalEngineConfig(BaseModel):
    type: Literal["local"] = "local"
    voice: str = TTS_VOICE


# The streamer runs one session at a time (see _SessionSlot), so one slot is enough.
_active: "LocalEngine | None" = None
# Shown on the page (GET /transcript). Survives between sessions on purpose.
_transcript: deque[dict[str, object]] = deque(maxlen=60)


def active_engine() -> "LocalEngine | None":
    return _active


def transcript_entries() -> list[dict[str, object]]:
    return list(_transcript)


def _note(role: str, text: str, **extra: object) -> None:
    _transcript.append({"t": round(time.time(), 2), "role": role, "text": text, **extra})


def _wav_bytes(pcm16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.astype("<i2").tobytes())
    return buf.getvalue()


def _split_sentences(text: str, *, final: bool) -> tuple[list[str], str]:
    """Return (complete sentences, remainder). Short fragments ride with the next one."""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end(1)
        piece = text[start:end].strip()
        if len(piece) < 12 and not final:
            continue  # "Yes." / "Hi!" — wait and send with what follows
        if piece:
            out.append(piece)
        start = m.end()
    rest = text[start:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


class LocalEngine:
    def __init__(self, config: LocalEngineConfig) -> None:
        self._config = config
        self._typed: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue(maxsize=16)
        self._history: list[dict[str, str]] = []
        self._turn: asyncio.Task[None] | None = None
        self._playing: set[str] = set()  # segments the renderer says are being spoken
        self._generating = False  # a reply is being produced (LLM and/or TTS)
        self._http: httpx.AsyncClient | None = None

    # ---- called from the /say route -------------------------------------------
    def enqueue(self, text: str, *, interrupt: bool = False, mode: str = "say") -> None:
        """Raises asyncio.QueueFull if backed up."""
        self._typed.put_nowait((text, mode, interrupt))

    # ---- speaking ---------------------------------------------------------------
    @property
    def _she_is_speaking(self) -> bool:
        return self._generating or bool(self._playing)

    @staticmethod
    def _buffer(pcm: bytes) -> SpeechBuffer:
        # The speech scheduler asserts on its native rate; convert from the TTS rate.
        return SpeechBuffer.from_bytes(pcm, TTS_SAMPLE_RATE).resample(constant.NATIVE_SPEECH_SAMPLE_RATE)

    async def _speak(self, bus: EventBus, text: str) -> float | None:
        """Stream one piece of text through TTS. Returns seconds to first audio."""
        assert self._http is not None
        segment_id = uuid.uuid4().hex
        style = str(_tuned("tts_style") or "").strip()
        body = {
            "model": TTS_MODEL,
            "voice": str(_tuned("tts_voice") or "") or self._config.voice,
            "input": TTS_STYLE_TEMPLATE.replace("{style}", style).replace("{text}", text) if style else text,
            # No response_format: the audio.cpp server pinned in llama-swap rejects an
            # explicit "pcm". stream_format "audio" alone returns headerless 16-bit
            # mono PCM, chunked as it is generated.
            "stream_format": "audio",
            **TTS_EXTRA,
        }
        started = False
        first_audio: float | None = None
        pending = b""
        t0 = time.monotonic()
        try:
            async with self._http.stream("POST", TTS_URL, json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread())[:300]
                    _LOGGER.error("TTS request failed", status=resp.status_code, detail=detail)
                    return None
                async for data in resp.aiter_bytes():
                    pending += data
                    if len(pending) < _MIN_CHUNK_BYTES:
                        continue
                    usable = len(pending) - (len(pending) % 2)  # whole 16-bit samples only
                    chunk, pending = pending[:usable], pending[usable:]
                    if not started:
                        started = True
                        first_audio = time.monotonic() - t0
                        await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
                    await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=self._buffer(chunk)))
            usable = len(pending) - (len(pending) % 2)
            if usable:
                if not started:
                    started = True
                    first_audio = time.monotonic() - t0
                    await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
                await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=self._buffer(pending[:usable])))
        except httpx.HTTPError as exc:
            _LOGGER.error("TTS stream error", error=str(exc))
        finally:
            if started:
                await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))
        return first_audio

    # ---- one reply ----------------------------------------------------------------
    async def _llm_sentences(self, user_text: str):
        """Yield the reply sentence by sentence as the LLM streams it."""
        assert self._http is not None
        prompt = str(_tuned("system_prompt") or "").strip() or SYSTEM_PROMPT
        messages = [{"role": "system", "content": prompt}, *self._history[-HISTORY_TURNS:],
                    {"role": "user", "content": user_text}]
        body = {"model": LLM_MODEL, "messages": messages, "stream": True, "max_tokens": int(_tuned("llm_max_tokens")), **LLM_EXTRA}
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        pending = ""
        async with self._http.stream("POST", LLM_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread())[:300]
                raise RuntimeError(f"LLM request failed: HTTP {resp.status_code} {detail!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choices = json.loads(data).get("choices") or []
                except json.JSONDecodeError:
                    continue
                for choice in choices:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        pending += piece
                sentences, pending = _split_sentences(pending, final=False)
                for s in sentences:
                    yield s
        sentences, _ = _split_sentences(pending, final=True)
        for s in sentences:
            yield s

    async def _reply(self, bus: EventBus, user_text: str, *, t_heard: float, stt_s: float | None) -> None:
        """LLM -> TTS for one user turn. Cancelled on barge-in."""
        self._generating = True
        spoken: list[str] = []
        t0 = time.monotonic()
        first_sentence_s: float | None = None
        first_audio_s: float | None = None
        try:
            async for sentence in self._llm_sentences(user_text):
                if first_sentence_s is None:
                    first_sentence_s = time.monotonic() - t0
                spoken.append(sentence)
                tts_first = await self._speak(bus, sentence)
                if first_audio_s is None and tts_first is not None:
                    first_audio_s = time.monotonic() - t_heard
        except asyncio.CancelledError:
            _note("system", "(interrupted)")
            raise
        except Exception as exc:  # noqa: BLE001 - keep the session alive, show the cause
            _LOGGER.error("reply failed", error=str(exc))
            _note("system", f"reply failed: {exc}")
        finally:
            self._generating = False
            reply = " ".join(spoken)
            self._history.append({"role": "user", "content": user_text})
            if reply:
                self._history.append({"role": "assistant", "content": reply})
                _note("avatar", reply, name=_session_avatar, stt_s=stt_s, llm_first_sentence_s=_r(first_sentence_s),
                      heard_to_first_audio_s=_r(first_audio_s))
            _LOGGER.info("turn", user=user_text[:120], reply=reply[:120], stt_s=stt_s,
                         llm_first_sentence_s=_r(first_sentence_s), heard_to_first_audio_s=_r(first_audio_s))

    async def _say_verbatim(self, bus: EventBus, text: str) -> None:
        self._generating = True
        try:
            for sentence in _split_sentences(text, final=True)[0]:
                await self._speak(bus, sentence)
            self._history.append({"role": "assistant", "content": text})
            _note("avatar", text, name=_session_avatar, typed=True)
        finally:
            self._generating = False

    async def _interrupt(self, bus: EventBus, reason: str) -> None:
        was_busy = self._she_is_speaking
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            try:
                await self._turn
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if was_busy:
            _LOGGER.info("interrupting avatar", reason=reason)
            await bus.publish(DiscardAvatarSpeechBuffer())
        self._playing.clear()

    def _start_turn(self, coro) -> None:
        self._turn = asyncio.create_task(coro, name="LocalEngine.turn")

    # ---- workers --------------------------------------------------------------------
    async def _transcribe(self, pcm16: np.ndarray) -> tuple[str, float]:
        assert self._http is not None
        t0 = time.monotonic()
        resp = await self._http.post(
            STT_URL,
            data={"model": STT_MODEL, "response_format": "json"},
            files={"file": ("speech.wav", _wav_bytes(pcm16, STT_SAMPLE_RATE), "audio/wav")},
        )
        resp.raise_for_status()
        text = " ".join(str(resp.json().get("text", "")).split())
        return text, round(time.monotonic() - t0, 3)

    async def _mic_worker(self, bus: EventBus) -> None:
        """End-of-speech detection on the microphone, then STT and a reply."""
        frame_ms = 20.0
        floor = VAD_MIN_RMS / VAD_FLOOR_FACTOR  # running estimate of background loudness
        preroll: deque[np.ndarray] = deque(maxlen=int(VAD_PREROLL_MS / frame_ms))
        speech: list[np.ndarray] = []
        in_speech = False
        loud_ms = quiet_ms = speech_ms = 0.0
        last_level_log = time.monotonic()
        peak = 0.0

        async with bus.subscribe(UserSpeechReceived) as sub:
            bus.ready()
            async for event in sub:
                # Keep the microphone's own sample rate here and resample the whole
                # utterance once for STT; resampling 20 ms pieces one by one would
                # put a seam at every boundary.
                mic_rate = event.buffer.sample_rate
                pcm = np.frombuffer(event.buffer.to_bytes(), dtype="<i2")
                if pcm.size == 0:
                    continue
                frame_ms = 1000.0 * pcm.size / mic_rate
                rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
                peak = max(peak, rms)
                threshold = max(VAD_MIN_RMS, floor * VAD_FLOOR_FACTOR)
                speaking_now = self._she_is_speaking
                if speaking_now:
                    threshold *= float(_tuned("barge_in_factor"))
                loud = rms >= threshold

                if time.monotonic() - last_level_log > 10:
                    _LOGGER.info("mic level", peak_rms=round(peak, 4), floor=round(floor, 4),
                                 threshold=round(threshold, 4), she_is_speaking=speaking_now)
                    last_level_log, peak = time.monotonic(), 0.0

                if not in_speech:
                    preroll.append(pcm)
                    if loud:
                        loud_ms += frame_ms
                    else:
                        loud_ms = 0.0
                        if not speaking_now:  # learn the room only while she is silent
                            floor = 0.97 * floor + 0.03 * rms
                    needed = BARGE_IN_START_MS if speaking_now else VAD_START_MS
                    if loud_ms >= needed:
                        if speaking_now:
                            if not _tuned("barge_in"):
                                loud_ms = 0.0
                                continue
                            await self._interrupt(bus, "user started speaking")
                        in_speech, quiet_ms, speech_ms = True, 0.0, loud_ms
                        speech = list(preroll)
                        preroll.clear()
                    continue

                speech.append(pcm)
                speech_ms += frame_ms
                quiet_ms = 0.0 if loud else quiet_ms + frame_ms
                if quiet_ms < float(_tuned("vad_end_ms")) and speech_ms < VAD_MAX_SPEECH_S * 1000:
                    continue

                # End of utterance.
                in_speech, loud_ms = False, 0.0
                audio = np.concatenate(speech)
                speech = []
                voiced_ms = speech_ms - quiet_ms
                if voiced_ms < VAD_MIN_SPEECH_MS:
                    continue
                t_heard = time.monotonic()
                try:
                    audio = np.frombuffer(
                        SpeechBuffer.from_bytes(audio.astype("<i2").tobytes(), mic_rate)
                        .resample(STT_SAMPLE_RATE).to_bytes(), dtype="<i2")
                    text, stt_s = await self._transcribe(audio)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.error("STT failed", error=str(exc))
                    _note("system", f"speech-to-text failed: {exc}")
                    continue
                if not text or _STT_JUNK.match(text):
                    _LOGGER.info("STT returned nothing usable", text=text, voiced_ms=round(voiced_ms))
                    continue
                _note("you", text, stt_s=stt_s, audio_s=round(audio.size / STT_SAMPLE_RATE, 2))
                await self._interrupt(bus, "new user turn")
                self._start_turn(self._reply(bus, text, t_heard=t_heard, stt_s=stt_s))

    async def _typed_worker(self, bus: EventBus) -> None:
        bus.ready()
        while True:
            text, mode, interrupt = await self._typed.get()
            if interrupt or mode == "ask":
                await self._interrupt(bus, "typed input")
            elif self._turn is not None and not self._turn.done():
                try:
                    await self._turn  # queue behind whatever she is saying
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if mode == "ask":
                _note("you", text, typed=True)
                self._start_turn(self._reply(bus, text, t_heard=time.monotonic(), stt_s=None))
            else:
                self._start_turn(self._say_verbatim(bus, text))

    async def _playback_worker(self, bus: EventBus) -> None:
        """Track which segments the renderer is currently speaking."""
        async with bus.subscribe(
            SegmentPlaybackStarted, SegmentPlaybackCompleted, SegmentPlaybackInterrupted, SegmentPlaybackCancelled
        ) as sub:
            bus.ready()
            async for event in sub:
                if isinstance(event, SegmentPlaybackStarted):
                    self._playing.add(str(event.segment_id))
                else:
                    self._playing.discard(str(event.segment_id))

    async def _watch_shutdown(self, bus: EventBus, others: list[asyncio.Task[None]]) -> None:
        async with bus.subscribe(Shutdown) as sub:
            bus.ready()
            await sub.get_next()
        if self._turn is not None:
            self._turn.cancel()
        for t in others:
            t.cancel()

    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        global _active
        _ = clocks
        _active = self
        _LOGGER.info("local engine started", avatar=_session_avatar, voice=self._config.voice, tts=TTS_URL, stt=STT_URL, llm=LLM_URL,
                     llm_model=LLM_MODEL, barge_in=BARGE_IN, vad_end_ms=VAD_END_MS)
        _note("system", "session started")
        timeout = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                self._http = http
                async with TaskGroup() as tg:
                    workers = [
                        tg.create_task(self._mic_worker(bus.clone()), name="LocalEngine._mic_worker"),
                        tg.create_task(self._typed_worker(bus.clone()), name="LocalEngine._typed_worker"),
                        tg.create_task(self._playback_worker(bus.clone()), name="LocalEngine._playback_worker"),
                    ]
                    tg.create_task(self._watch_shutdown(bus.clone(), workers), name="LocalEngine._watch_shutdown")
                    bus.ready()
        finally:
            # A reply in progress is a task of its own; stop it too, or it would
            # keep talking to a session that has gone (ended, or taken over by a
            # new one on another device).
            if self._turn is not None and not self._turn.done():
                self._turn.cancel()
            self._http = None
            if _active is self:
                _active = None
            _note("system", "session ended")


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


async def build_local(*, stream_id: str, options: LocalEngineOptions):
    _ = stream_id
    cfg = LocalEngineConfig(voice=options.voice)
    return cfg, LocalEngine(cfg).run
