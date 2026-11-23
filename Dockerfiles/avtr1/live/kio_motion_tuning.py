# Renderer side of the per-avatar tuning settings (see kio_avatar_settings.py).
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build and called from
# two one-line patches: one in avtr1_renderer/pipeline.py, one in
# avtr1_renderer/avtr1_motion_generator.py. Needs torch and roma, so only the
# renderer environment may import it.
#
# Three kinds of adjustment, all per avatar, all read fresh for every 200 ms chunk:
#
#   1. Upstream's own motion settings (noise range/smoothness, three guidance
#      strengths), which the streamer otherwise always sends at their defaults.
#   2. Per-region guidance. The motion model's 512-wide latent is split into four
#      slices that feed head rotation, brow, eyes and mouth separately (read from
#      model.last_layer.split_sizes = [36, 109, 219, 148]). Upstream fills each
#      guidance vector with one number; here each slice gets its own multiplier.
#   3. Gains on the generated motion: head rotation, and the brow / eyes / mouth
#      expression groups, each scaled about the portrait's own value.
#   4. Poses: a glance and a head movement for what the agent is doing (thinking,
#      running a tool, listening, waiting), blended in on top. See kio_avatar_mode.py.

from __future__ import annotations

import dataclasses

import roma
import torch

import kio_avatar_mode as mode
import kio_avatar_settings as settings
from avtr1_renderer.constants import LIPSYNC_COORDS

# Latent slices of the four output heads, in model order.
_SPLIT_SIZES = (36, 109, 219, 148)
_REGION_SLICES: dict[str, slice] = {}
_start = 0
for _region, _size in zip(settings.REGIONS, _SPLIT_SIZES):
    _REGION_SLICES[_region] = slice(_start, _start + _size)
    _start += _size
_LATENT_DIM = _start  # 512

# Columns of MotionFrame.exp (which follows LIPSYNC_COORDS order) for each group.
_COORDS = list(LIPSYNC_COORDS)
_EXP_COLS = {
    "brow": [i for i, c in enumerate(_COORDS) if c < 33],          # keypoints 1, 2, 6
    "eyes": [i for i, c in enumerate(_COORDS) if 33 <= c <= 50],   # keypoints 11-16
    "mouth": [i for i, c in enumerate(_COORDS) if c >= 51],        # keypoints 17-20
}
_EYELID_COLS = [_COORDS.index(c) for c in (34, 46)]  # keypoints 11 and 15, y (found by measurement)

_OPTION_KEYS = ("noise_trunc_z", "noise_alpha", "cfg_self_audio", "cfg_other_audio", "cfg_kp")


def override_options(options, avatar):
    """RenderOptions with this avatar's saved motion settings applied."""
    saved = settings.overrides(avatar.id)
    changes = {k: float(saved[k]) for k in _OPTION_KEYS if k in saved}
    return dataclasses.replace(options, **changes) if changes else options


def guidance_vector(key: str, value: float, avatar, latent_dim: int, device) -> torch.Tensor:
    """Guidance weights for one condition: `value` everywhere, times each region's multiplier."""
    w = torch.full((latent_dim,), float(value), device=device, dtype=torch.float32)
    if latent_dim != _LATENT_DIM:
        return w  # a different model than the one these slices were read from
    saved = settings.overrides(avatar.id)
    for region, sl in _REGION_SLICES.items():
        m = saved.get(f"{key}.{region}")
        if m is not None and m != 1.0:
            w[sl] *= float(m)
    return w


def apply_gains(motions, avatar) -> None:
    """Called once per chunk from pipeline.py: the per-avatar gains, then the pose for the current mode."""
    _apply_gains(motions, avatar)
    apply_pose(motions, avatar)


def _apply_gains(motions, avatar) -> None:
    """Scale the generated motion in place, about the portrait's own pose and expression."""
    saved = settings.overrides(avatar.id)
    if not any(k.startswith("gain.") for k in saved):
        return
    kp = avatar.kp_info

    head = float(saved.get("gain.head", 1.0))
    if head != 1.0:
        # Rotation relative to the portrait's pose, scaled in axis-angle form.
        r_src = kp.R.to(motions.R.device, motions.R.dtype)  # (1, 3, 3)
        rel = r_src.transpose(-1, -2) @ motions.R  # (N, 3, 3)
        motions.R[:] = r_src @ roma.rotvec_to_rotmat(roma.rotmat_to_rotvec(rel) * head)

    # The model's expression values are in the head's own frame (exp @ R^T), so
    # that is the frame the portrait's resting values must be taken in.
    src_all = (kp.exp @ kp.R.transpose(-1, -2)).reshape(-1)[_COORDS].to(motions.exp.device, motions.exp.dtype)
    gain = torch.ones(len(_COORDS), device=motions.exp.device, dtype=motions.exp.dtype)
    for region, cols in _EXP_COLS.items():
        g = float(saved.get(f"gain.{region}", 1.0))
        if g != 1.0:
            gain[cols] *= g
    lids = float(saved.get("gain.eyelids", 1.0))
    if lids != 1.0:
        gain[_EYELID_COLS] *= lids
    if bool((gain != 1.0).any()):
        motions.exp[:] = src_all + gain * (motions.exp - src_all)


# ---- poses: what she does while the agent thinks, runs a tool, listens or waits -----
#
# kio_avatar_mode holds the contract (the mode names and the pose controls). Here a
# pose becomes offsets on the generated motion: the gaze through four of the eye
# keypoints, the head through a small extra rotation. The gaze numbers are the
# LivePortrait project's own "eyeball direction" retargeting (this renderer is of
# that family); tried on this model on 2026-11-23, where gaze up, gaze aside and the
# head offsets read clearly, gaze down read as half-closed, and its brow control did
# next to nothing (so there is none here).
#
# The renderer is stateless per chunk, but easing into a pose needs memory, so the
# pose in effect is kept here between chunks. One session at a time, so one pose.

_POSE_KEYS = tuple(c[0] for c in mode.POSE_CONTROLS)  # gaze_x, gaze_y, head_pitch, head_yaw, head_roll
_FRAME_MS = 40.0  # 25 frames a second
_pose_now = [0.0] * len(_POSE_KEYS)
_pose_avatar = ""


def _col(keypoint: int, axis: int) -> int:
    return _COORDS.index(keypoint * 3 + axis)


def pose_target(avatar_id: str, current_mode: str) -> list[float]:
    """The pose this avatar should move towards in this mode: its tuned values times its strength."""
    if current_mode not in mode.POSE_MODES:
        return [0.0] * len(_POSE_KEYS)
    strength = float(settings.get(avatar_id, "pose.strength"))
    return [strength * float(settings.get(avatar_id, f"pose.{current_mode}.{k}")) for k in _POSE_KEYS]


def apply_pose(motions, avatar) -> None:
    """Blend the current mode's pose into the generated motion, in place, frame by frame."""
    global _pose_avatar
    if avatar.id != _pose_avatar:  # another avatar: start from no pose rather than the last one's
        _pose_avatar = avatar.id
        _pose_now[:] = [0.0] * len(_POSE_KEYS)
    target = pose_target(avatar.id, mode.get_mode())
    if not any(target) and not any(abs(v) > 1e-3 for v in _pose_now):
        return
    blend_ms = max(float(settings.get(avatar.id, "pose.blend_ms")), _FRAME_MS)
    step = min(1.0, _FRAME_MS / blend_ms * 3.0)  # an exponential approach, about done in blend_ms
    n = motions.exp.shape[0]
    frames = []
    for _ in range(n):
        for i, t in enumerate(target):
            _pose_now[i] += (t - _pose_now[i]) * step
        frames.append(list(_pose_now))
    p = torch.tensor(frames, device=motions.exp.device, dtype=motions.exp.dtype)  # (n, 5)
    gx, gy = p[:, 0], p[:, 1]
    # Gaze: LivePortrait's eyeball-direction mapping onto keypoints 11, 13, 15 and 16.
    right = torch.where(gx > 0, 0.0007, 0.001)
    left = torch.where(gx > 0, 0.001, 0.0007)
    lid = -gy / 2.0
    motions.exp[:, _col(11, 0)] += gx * right
    motions.exp[:, _col(15, 0)] += gx * left
    motions.exp[:, _col(11, 1)] += gy * -0.001 + lid * -0.001
    motions.exp[:, _col(15, 1)] += gy * -0.001 + lid * -0.001
    motions.exp[:, _col(13, 1)] += lid * 0.0003
    motions.exp[:, _col(16, 1)] += lid * 0.0003
    # Head: a small extra rotation in front of the generated one.
    if bool((p[:, 2:].abs() > 1e-4).any()):
        rad = torch.deg2rad(p[:, 2:])
        pitch, yaw, roll = rad[:, 0], rad[:, 1], rad[:, 2]
        zero, one = torch.zeros_like(pitch), torch.ones_like(pitch)
        rx = torch.stack([one, zero, zero, zero, pitch.cos(), -pitch.sin(), zero, pitch.sin(), pitch.cos()], dim=1).reshape(n, 3, 3)
        ry = torch.stack([yaw.cos(), zero, yaw.sin(), zero, one, zero, -yaw.sin(), zero, yaw.cos()], dim=1).reshape(n, 3, 3)
        rz = torch.stack([roll.cos(), -roll.sin(), zero, roll.sin(), roll.cos(), zero, zero, zero, one], dim=1).reshape(n, 3, 3)
        motions.R[:] = (rz @ ry @ rx) @ motions.R


_face_boxes: dict[str, dict | None] = {}


def face_box(avatar) -> dict | None:
    """Where the animated square (face, hair, neck) sits in the picture, as fractions of its width and height.

    Read from the transform the renderer keeps for pasting the animated crop back
    into the full picture (Avatar.M_grid maps picture coordinates to crop
    coordinates, both on a -1..1 scale). Used by the streamer to centre a
    per-session crop on the face; see kio_video_shape.py.
    """
    if avatar.id not in _face_boxes:
        try:
            m = avatar.M_grid.detach().float().cpu().reshape(2, 3)
            inv = torch.linalg.inv(m[:, :2])
            centre = -(inv @ m[:, 2])
            cx, cy = (float(centre[0]) + 1) / 2, (float(centre[1]) + 1) / 2
            hx, hy = float(inv[:, 0].norm()) / 2, float(inv[:, 1].norm()) / 2
            clamp = lambda v: round(min(max(v, 0.0), 1.0), 4)  # noqa: E731
            _face_boxes[avatar.id] = {"x0": clamp(cx - hx), "y0": clamp(cy - hy), "x1": clamp(cx + hx), "y1": clamp(cy + hy)}
        except Exception:  # noqa: BLE001 - a crop is a nicety; never fail a request over it
            _face_boxes[avatar.id] = None
    return _face_boxes[avatar.id]
