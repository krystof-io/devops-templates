# Per-avatar tuning settings: what exists, the allowed range of each, and the
# file they are saved in.
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build, which puts it on
# the import path of BOTH pixi environments (the repo package is an editable
# install rooted at src/). It must therefore stay standard-library only: the
# streamer environment has no torch, the renderer environment has no aiortc.
#
# One JSON file holds everything:  $AVTR1_LOCAL_STORAGE/avatar_settings.json
#
#     {
#       "selene":       {"noise_trunc_z": 2.0, "gain.eyelids": 4.0, "vad_end_ms": 600},
#       "selene_basic": {"gain.eyelids": 2.5}
#     }
#
# Only values that differ from the defaults are stored. The renderer and the
# conversation engine re-read the file when it changes, so an edit takes effect
# within a moment, mid-conversation, with no restart. Anything else that wants
# these values (for example the Hermes HUD) can read the same file, or GET
# /avatar-settings on the streamer, which returns it together with this schema.

from __future__ import annotations

import json
import os
import tempfile
import time

import kio_avatar_mode as _mode

REGIONS = ("head", "brow", "eyes", "mouth")

# (key, label, group, kind, default, min, max, step, help)
_NUM = "number"
_BOOL = "bool"
_TEXT = "text"
_LONG = "longtext"

_FIELDS: list[tuple] = [
    # --- movement character (upstream's own per-request settings) ----------------
    ("noise_trunc_z", "Movement range", "Movement", _NUM, 1.2, 0.2, 4.0, 0.05,
     "How far the motion may wander. Higher = more movement. Upstream default 1.2."),
    ("noise_alpha", "Movement smoothness", "Movement", _NUM, 2.0, 0.0, 8.0, 0.1,
     "How strongly each frame follows the last. Lower = livelier, higher = calmer. Upstream default 2.0."),
    # --- guidance: how strongly each input steers the motion -----------------------
    ("cfg_self_audio", "Her own speech", "Guidance", _NUM, 2.0, 0.0, 6.0, 0.05,
     "How strongly her voice drives the motion. Upstream default 2.0."),
    ("cfg_other_audio", "The listener (you)", "Guidance", _NUM, 2.0, 0.0, 6.0, 0.05,
     "How strongly she reacts to the other person's audio. Upstream default 2.0."),
    ("cfg_kp", "The portrait", "Guidance", _NUM, 3.0, 0.0, 8.0, 0.05,
     "How strongly she is pulled back to the portrait's own pose and expression. Upstream default 3.0."),
]
# Per-region multipliers on each guidance value. The motion model has separate
# internal outputs for head rotation, brow, eyes and mouth; upstream's runtime
# uses one guidance value for all four. 1.0 = same as the overall value.
for _key, _label in (("cfg_self_audio", "speech"), ("cfg_other_audio", "listener"), ("cfg_kp", "portrait")):
    for _region in REGIONS:
        _FIELDS.append((f"{_key}.{_region}", f"{_region.capitalize()} × {_label}", "Guidance by region", _NUM,
                        1.0, 0.0, 3.0, 0.05,
                        f"Multiplies the '{_label}' guidance for the {_region} only. 1.0 = no change."))
_FIELDS += [
    # --- gains: plain multipliers applied to the generated motion ------------------
    ("gain.head", "Head movement", "Gains", _NUM, 1.0, 0.0, 2.5, 0.05,
     "Scales head rotation away from the portrait's own pose. Below 1 calms the head (and hides the hair seam)."),
    ("gain.brow", "Brow", "Gains", _NUM, 1.0, 0.0, 3.0, 0.05,
     "Scales brow movement away from the portrait's own expression."),
    ("gain.eyes", "Eyes (whole region)", "Gains", _NUM, 1.0, 0.0, 3.0, 0.05,
     "Scales all eye-region movement, eyelids included."),
    ("gain.eyelids", "Eyelids (blink strength)", "Gains", _NUM, 1.0, 0.5, 6.0, 0.1,
     "Extra multiplier on the two eyelid values, on top of the eyes gain. Raise it if she never fully blinks."),
    ("gain.mouth", "Mouth", "Gains", _NUM, 1.0, 0.0, 3.0, 0.05,
     "Scales mouth movement away from the portrait's own expression."),
    # --- poses: what she does while the agent thinks, runs a tool, listens or waits ----
    # (the modes and controls are defined once, in kio_avatar_mode.py)
    ("pose.strength", "Pose strength", "Poses", _NUM, 1.0, 0.0, 2.0, 0.05,
     "Scales every pose at once. 0 turns poses off for this avatar."),
    ("pose.blend_ms", "Pose blend time (ms)", "Poses", _NUM, 400, 80, 2000, 20,
     "How long she takes to move into a pose and back out of it."),
    *[
        (f"pose.{_m}.{_c}", f"{_m.capitalize()}: {_label}", "Poses", _NUM,
         _mode.POSE_DEFAULTS.get(_m, {}).get(_c, 0.0), _lo, _hi, _step, f"While the agent is {_m}. {_help}")
        for _m in _mode.POSE_MODES
        for (_c, _label, _lo, _hi, _step, _help) in _mode.POSE_CONTROLS
    ],
    # --- conversation (used by the local conversation engine) ------------------------
    ("vad_end_ms", "Pause before she answers (ms)", "Conversation", _NUM, 700, 200, 2500, 50,
     "How long you must be quiet before your turn is considered finished."),
    ("barge_in", "You can interrupt her", "Conversation", _BOOL, True, None, None, None,
     "Speaking while she talks cancels her reply. Turn off if her own voice from your speakers cuts her off."),
    ("barge_in_factor", "Interruption threshold", "Conversation", _NUM, 2.5, 1.0, 6.0, 0.1,
     "How much louder than normal speech detection an interruption must be."),
    ("llm_max_tokens", "Reply length limit (tokens)", "Conversation", _NUM, 300, 50, 1000, 10,
     "Upper bound on the length of each reply."),
    ("tts_voice", "Voice", "Conversation", _TEXT, "", None, None, None,
     "Name of a voice in /models/tts-voices. Empty = the pod's default voice."),
    ("tts_style", "Voice style direction", "Conversation", _TEXT, "", None, None, None,
     "Style direction added to each sentence. VoxCPM2: free text, e.g. 'warm, amused'. Higgs: inline tags, e.g. <|emotion:amusement|>. Empty = the server's default."),
    ("system_prompt", "Persona prompt", "Conversation", _LONG, "", None, None, None,
     "System prompt for the LLM. Empty = the pod's default prompt."),
]

SCHEMA: list[dict] = [
    {"key": k, "label": label, "group": group, "kind": kind, "default": default,
     "min": lo, "max": hi, "step": step, "help": help_}
    for (k, label, group, kind, default, lo, hi, step, help_) in _FIELDS
]
_BY_KEY = {f["key"]: f for f in SCHEMA}
DEFAULTS: dict[str, object] = {f["key"]: f["default"] for f in SCHEMA}

def set_default(key: str, value: object) -> None:
    """Let a process replace a default with its own (the engine does this from its environment)."""
    field = _BY_KEY[key]
    field["default"] = value
    DEFAULTS[key] = value


_TEXT_LIMITS = {"tts_voice": 60, "tts_style": 200, "system_prompt": 4000}
_LEGACY = {"eye_gain": "gain.eyelids"}  # name used before the tuning panel existed


def settings_path() -> str:
    # KIO_AVATAR_SETTINGS_FILE lets a test (or another deployment) use a different file.
    explicit = os.environ.get("KIO_AVATAR_SETTINGS_FILE")
    if explicit:
        return explicit
    storage = os.environ.get("AVTR1_LOCAL_STORAGE")
    if not storage:
        raise RuntimeError("AVTR1_LOCAL_STORAGE is not set; the image has no default for it (see kio_required_env.py)")
    return os.path.join(storage, "avatar_settings.json")


def _clean(key: str, value: object) -> object | None:
    """Validate one value against the schema. None = not acceptable, drop it."""
    field = _BY_KEY.get(key)
    if field is None:
        return None
    kind = field["kind"]
    if kind == _NUM:
        if isinstance(value, bool):
            return None
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return None
        v = min(max(v, float(field["min"])), float(field["max"]))
        return int(round(v)) if isinstance(field["default"], int) and not isinstance(field["default"], bool) else round(v, 4)
    if kind == _BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return None
    if not isinstance(value, str):
        return None
    v = value.strip()[: _TEXT_LIMITS.get(key, 200)]
    if key == "tts_voice" and v and not all(c.isalnum() or c in "-_" for c in v):
        return None
    return v


def clean_overrides(values: dict) -> dict[str, object]:
    """Keep only known keys with valid values that differ from the default."""
    out: dict[str, object] = {}
    for key, value in (values or {}).items():
        key = _LEGACY.get(str(key), str(key))
        v = _clean(key, value)
        if v is not None and v != DEFAULTS[key]:
            out[key] = v
    return out


_cache: dict = {"mtime": None, "checked": 0.0, "data": {}}


def load_all(max_age: float = 0.25) -> dict[str, dict[str, object]]:
    """All avatars' overrides. Re-reads the file when it changes, at most every `max_age` seconds."""
    now = time.monotonic()
    if now - _cache["checked"] < max_age:
        return _cache["data"]
    _cache["checked"] = now
    try:
        mtime = os.stat(settings_path()).st_mtime_ns
    except OSError:
        _cache["mtime"], _cache["data"] = None, {}
        return _cache["data"]
    if mtime != _cache["mtime"]:
        try:
            with open(settings_path(), encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            raw = {}
        data: dict[str, dict[str, object]] = {}
        if isinstance(raw, dict):
            for avatar, values in raw.items():
                if isinstance(values, dict):
                    data[str(avatar)] = clean_overrides(values)
        _cache["mtime"], _cache["data"] = mtime, data
    return _cache["data"]


def overrides(avatar_id: str) -> dict[str, object]:
    return load_all().get(avatar_id, {})


def effective(avatar_id: str) -> dict[str, object]:
    """Defaults with this avatar's overrides applied."""
    return {**DEFAULTS, **overrides(avatar_id)}


def get(avatar_id: str, key: str) -> object:
    return overrides(avatar_id).get(key, DEFAULTS[key])


def save(avatar_id: str, values: dict) -> dict[str, object]:
    """Replace one avatar's overrides. Other avatars are untouched. Atomic."""
    path = settings_path()
    data = {k: dict(v) for k, v in load_all(max_age=0.0).items()}
    cleaned = clean_overrides(values)
    if cleaned:
        data[avatar_id] = cleaned
    else:
        data.pop(avatar_id, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".avatar_settings.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _cache["checked"] = 0.0  # force a re-read on next use in this process
    return cleaned
