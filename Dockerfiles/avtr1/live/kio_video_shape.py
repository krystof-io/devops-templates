# Per-session video shape and size, applied in the streamer just before a frame
# is handed to the WebRTC encoder.
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build; only the streamer
# imports it (it needs numpy and PyAV, both already in that environment).
#
# The renderer always produces the full 1280x720 picture. A client that shows the
# avatar in a square or a narrow column would throw most of that away, so it can
# ask in its /offer for:
#
#   shape   "wide" (16:9, the whole picture), "square" (1:1) or "portrait" (3:4)
#   zoom    1.0 = the crop is as tall as the picture; 1.5 = two-thirds as tall,
#           closer on the face; "face" = just tall enough for the head and neck,
#           whatever size the head is in this portrait (so every avatar fills the
#           frame alike)
#   height  height in pixels of what is sent (the crop is scaled down to it;
#           never scaled up)
#
# The crop is centred on the face, using where the renderer found it when the
# portrait was registered (GET /avatar-geometry on the renderer). This saves
# network, decoding on the client and encoding work here. It does not save GPU
# time: the model renders the full picture either way.
#
# One session at a time (see _SessionSlot), so one module-level setting is enough.
# Absent fields mean "send the picture as upstream does": wide, zoom 1, full height.

from __future__ import annotations

import av
import numpy as np

SHAPES = {"wide": None, "square": 1.0, "portrait": 0.75}  # width / height; None = the picture's own

FACE_MARGIN = 1.15

_session: dict | None = None  # {"x": int, "y": int, "w": int, "h": int, "out_w": int, "out_h": int}
_requested: dict = {}


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def plan(frame_w: int, frame_h: int, *, shape: str, zoom: float | str, height: int, face: dict | None) -> dict:
    """Work out the crop rectangle and output size. Pure, so it can be tested without video."""
    if zoom == "face":
        # The animated square plus a margin, so hair and shoulders are not cut tight.
        zoom = 1.0 / min(1.0, (face["y1"] - face["y0"]) * FACE_MARGIN) if face else 1.0
    zoom = min(max(float(zoom), 1.0), 4.0)
    aspect = SHAPES[shape] or (frame_w / frame_h)
    h = frame_h / zoom
    w = min(h * aspect, frame_w)
    h = min(h, w / aspect) if SHAPES[shape] else h
    # Face centre, as fractions of the picture; the middle if the renderer could not say.
    cx = (face["x0"] + face["x1"]) / 2 if face else 0.5
    cy = (face["y0"] + face["y1"]) / 2 if face else 0.5
    w, h = _even(w), _even(h)
    x = min(max(cx * frame_w - w / 2, 0), frame_w - w)
    y = min(max(cy * frame_h - h / 2, 0), frame_h - h)
    x, y = int(x) // 2 * 2, int(y) // 2 * 2
    out_h = _even(min(max(int(height), 120), h)) if height else h
    out_w = _even(w * out_h / h)
    return {"x": x, "y": y, "w": w, "h": h, "out_w": out_w, "out_h": out_h}


def set_session(*, shape: str | None, zoom: float | str | None, height: int | None, face: dict | None, frame_w: int, frame_h: int) -> dict | None:
    """Called by /offer. Returns the plan, or None when the picture is sent unchanged."""
    global _session, _requested
    _requested = {"shape": shape, "zoom": zoom, "height": height}
    if not shape and not zoom and not height:
        _session = None
        return None
    p = plan(frame_w, frame_h, shape=shape or "wide", zoom=zoom or 1.0, height=height or 0, face=face)
    unchanged = p["w"] == frame_w and p["h"] == frame_h and p["out_h"] == frame_h
    _session = None if unchanged else p
    return _session


def build(arr: np.ndarray) -> av.VideoFrame:
    """I420 picture (rows = height * 3 / 2, as the renderer sends it) -> the frame to encode."""
    p = _session
    if p is None:
        return av.VideoFrame.from_ndarray(arr, format="yuv420p")
    full_w = arr.shape[1]
    full_h = arr.shape[0] * 2 // 3
    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
    luma = arr[:full_h]
    # The two colour planes follow the luma in the buffer, each at half the width and half the height.
    chroma = arr[full_h:].reshape(2, full_h // 2, full_w // 2)
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h] = luma[y:y + h, x:x + w]
    out[h:] = chroma[:, y // 2:(y + h) // 2, x // 2:(x + w) // 2].reshape(h // 2, w)
    frame = av.VideoFrame.from_ndarray(out, format="yuv420p")
    if (p["out_w"], p["out_h"]) != (w, h):
        frame = frame.reformat(width=p["out_w"], height=p["out_h"], format="yuv420p")
    return frame
