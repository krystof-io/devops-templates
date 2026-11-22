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

from __future__ import annotations

import dataclasses

import roma
import torch

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
