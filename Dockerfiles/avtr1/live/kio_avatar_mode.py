# What the agent behind the avatar is doing right now ("mode"), so the avatar can
# show it: a glance up and aside while it thinks, and so on.
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build, on the import path
# of both pixi environments. Standard library only.
#
# THE CONTRACT, in one place:
#
#   Who sends it   A client that knows the agent's state: the Hermes HUD posts
#                  {"mode": "<name>"} to /mode (as /avatar/mode on its own host)
#                  whenever the state changes while the avatar is live. The test
#                  page's own conversation engine sets it too.
#   The names      MODES below. They are exactly the Hermes HUD's display states
#                  (hermes-hud src/data/types.ts, DISPLAY_STATES), so the HUD sends
#                  its state unchanged. Any other name is refused.
#   What it does   Nothing by itself. Each mode may have a POSE: small offsets to
#                  the gaze and the head, blended in over a moment on top of the
#                  avatar's own motion (kio_motion_tuning.apply_pose). Only the
#                  modes in POSE_MODES have one; the rest are accepted and mean
#                  "no pose". The numbers are per-avatar tuning
#                  (kio_avatar_settings: pose.<mode>.<control>), with the defaults
#                  there.
#   Lifetime       One session at a time, so one mode. It goes back to "idle"
#                  when a session starts and when it ends.
#
# The streamer writes the mode to a small file in shared memory and the renderer
# (a separate process in the same pod) reads it for every 200 ms chunk.

from __future__ import annotations

import os
import tempfile
import time

# The Hermes HUD's display states, in its order.
MODES = ("idle", "listening", "thinking", "tool", "speaking", "waiting", "compacting", "error", "offline", "sleeping")
# The modes that can have a pose. Their controls are POSE_CONTROLS, each a setting pose.<mode>.<control>.
POSE_MODES = ("thinking", "tool", "listening", "waiting")
# (control, label, lowest, highest, step, what it does)
POSE_CONTROLS = (
    ("gaze_x", "gaze left/right", -20.0, 20.0, 0.5, "Eyes to her left (negative) or right (positive)."),
    ("gaze_y", "gaze down/up", -20.0, 20.0, 0.5, "Eyes down (negative) or up (positive). Down reads as sleepy; up works well."),
    ("head_pitch", "head down/up (degrees)", -15.0, 15.0, 0.5, "Chin down (positive) or up (negative)."),
    ("head_yaw", "head turn (degrees)", -15.0, 15.0, 0.5, "Head turned to one side."),
    ("head_roll", "head tilt (degrees)", -15.0, 15.0, 0.5, "Head tilted towards a shoulder."),
)
# Starting poses, used for every avatar until one is tuned. "thinking" is the look
# tried on the GPU on 2026-11-23 (eyes up and aside, head tilted and turned a
# little). "tool" is a sideways glance; looking down read as half-asleep there.
POSE_DEFAULTS = {
    "thinking": {"gaze_x": 12.0, "gaze_y": 12.0, "head_pitch": -4.0, "head_yaw": 5.0, "head_roll": 5.0},
    "tool": {"gaze_x": -12.0, "gaze_y": 3.0, "head_pitch": 0.0, "head_yaw": -5.0, "head_roll": 0.0},
}

_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
_FILE = os.environ.get("KIO_AVATAR_MODE_FILE") or os.path.join(_DIR, "kio_avatar_mode")
_cache = {"mtime": None, "checked": 0.0, "mode": "idle"}


def set_mode(mode: str) -> str:
    """Record the mode. Raises ValueError for a name that is not in MODES."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of: {', '.join(MODES)}")
    tmp = f"{_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(mode)
    os.replace(tmp, _FILE)
    _cache["checked"] = 0.0
    return mode


def get_mode(max_age: float = 0.1) -> str:
    """The current mode; "idle" if none was ever set. Re-reads the file when it changes."""
    now = time.monotonic()
    if now - _cache["checked"] < max_age:
        return _cache["mode"]
    _cache["checked"] = now
    try:
        mtime = os.stat(_FILE).st_mtime_ns
    except OSError:
        _cache["mtime"], _cache["mode"] = None, "idle"
        return "idle"
    if mtime != _cache["mtime"]:
        try:
            with open(_FILE, encoding="utf-8") as f:
                mode = f.read().strip()
        except OSError:
            mode = "idle"
        _cache["mtime"], _cache["mode"] = mtime, mode if mode in MODES else "idle"
    return _cache["mode"]
