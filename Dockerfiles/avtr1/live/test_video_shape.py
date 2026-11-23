#!/usr/bin/env python3
"""Tests for kio_video_shape. Needs numpy and PyAV (the streamer environment):
    .pixi/envs/streamer/bin/python src/test_video_shape.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kio_video_shape as vs  # noqa: E402

W, H = 1280, 720
ARNOLD = {"x0": 0.285, "y0": 0.119, "x1": 0.681, "y1": 0.823}  # measured on the GPU, 2026-11-22
SELENE = {"x0": 0.267, "y0": 0.0, "x1": 0.749, "y1": 0.851}


def picture() -> np.ndarray:
    """An I420 picture whose brightness encodes the position, so a crop can be checked."""
    arr = np.zeros((H * 3 // 2, W), dtype=np.uint8)
    arr[:H] = (np.arange(W) // 8).astype(np.uint8)[None, :]  # luma: column / 8
    arr[:H] += (np.arange(H) // 180 * 40).astype(np.uint8)[:, None] % 97  # plus a step every 180 rows
    chroma = arr[H:].reshape(2, H // 2, W // 2)
    chroma[0] = (np.arange(W // 2) // 4).astype(np.uint8)[None, :]
    chroma[1] = (np.arange(H // 2) // 2).astype(np.uint8)[:, None]
    return arr


def main() -> None:
    # Nothing asked for: the picture goes out untouched.
    assert vs.set_session(shape=None, zoom=None, height=None, face=ARNOLD, frame_w=W, frame_h=H) is None
    f = vs.build(picture())
    assert (f.width, f.height) == (W, H)
    assert vs.set_session(shape="wide", zoom=1.0, height=720, face=None, frame_w=W, frame_h=H) is None

    # Square, full height, centred on the face and kept inside the picture.
    p = vs.plan(W, H, shape="square", zoom=1.0, height=0, face=ARNOLD)
    assert (p["w"], p["h"], p["y"]) == (720, 720, 0), p
    assert abs((p["x"] + 360) - 0.483 * W) <= 2, p
    p = vs.plan(W, H, shape="square", zoom=1.0, height=0, face={"x0": 0.0, "y0": 0.0, "x1": 0.1, "y1": 0.2})
    assert p["x"] == 0 and p["y"] == 0, p  # a face at the edge does not push the crop outside

    # "face": the smaller head gets the tighter crop, so both fill the frame alike.
    a = vs.plan(W, H, shape="square", zoom="face", height=540, face=ARNOLD)
    s = vs.plan(W, H, shape="square", zoom="face", height=540, face=SELENE)
    assert a["h"] < s["h"] <= H and a["w"] == a["h"], (a, s)
    assert (a["out_w"], a["out_h"]) == (540, 540) and (s["out_w"], s["out_h"]) == (540, 540), (a, s)
    assert vs.plan(W, H, shape="square", zoom="face", height=0, face=None)["h"] == H  # unknown face: no zoom

    # Portrait is 3:4; height never scales up; every number is even.
    p = vs.plan(W, H, shape="portrait", zoom=1.0, height=2000, face=SELENE)
    assert (p["w"], p["h"], p["out_w"], p["out_h"]) == (540, 720, 540, 720), p
    for shape in vs.SHAPES:
        for zoom in (1.0, 1.3, 2.7, "face"):
            for height in (0, 121, 360, 541):
                p = vs.plan(W, H, shape=shape, zoom=zoom, height=height, face=ARNOLD)
                assert all(v % 2 == 0 for v in p.values()), p
                assert 0 <= p["x"] and p["x"] + p["w"] <= W and 0 <= p["y"] and p["y"] + p["h"] <= H, p

    # The crop really is that rectangle of the picture, in all three planes.
    src = picture()
    plan = vs.set_session(shape="square", zoom=1.5, height=None, face=ARNOLD, frame_w=W, frame_h=H)
    out = vs.build(src).to_ndarray(format="yuv420p")
    x, y, w, h = plan["x"], plan["y"], plan["w"], plan["h"]
    assert out.shape == (h * 3 // 2, w), out.shape
    assert np.array_equal(out[:h], src[:H][y:y + h, x:x + w])
    want = src[H:].reshape(2, H // 2, W // 2)[:, y // 2:(y + h) // 2, x // 2:(x + w) // 2]
    assert np.array_equal(out[h:].reshape(2, h // 2, w // 2), want)

    # Scaling down gives the size asked for.
    vs.set_session(shape="square", zoom="face", height=360, face=ARNOLD, frame_w=W, frame_h=H)
    f = vs.build(src)
    assert (f.width, f.height) == (360, 360), (f.width, f.height)
    vs.set_session(shape=None, zoom=None, height=None, face=None, frame_w=W, frame_h=H)
    print("kio_video_shape: all tests passed")


if __name__ == "__main__":
    main()
