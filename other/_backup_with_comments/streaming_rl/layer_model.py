"""
layer_model.py - base/enhancement-layer data-volume computation (A^t).

Implements the professor-confirmed model (config.py, "Base /
enhancement-layer (QP) model" section): every tile is sent at the base
QP level unconditionally; a tile the agent selects for enhancement is
sent at a higher QP level instead (replacing the base stream for that
tile, not stacking a second stream on top); the data-volume accounting
is additive:

    A^t = A_base^t + sum_i x_i^t * delta_A_i^t
    delta_A_i^t = A_i[enhanced QP] - A_i[base QP]

Units: video_bitrate_data is in bps (per the dataset README), and its
value is constant across all frames within a GOP (verified empirically
- see explore_data.ipynb Section C / research_summary.md Section 5.3).
Since T0 = 1 second = 1 GOP (config.T0_SEC), a bps value over one T0
slot is numerically equal to the bit count for that slot - so A^t as
computed here is directly in BITS, consistent with eq. 12's usage
inside the buffer/rate equations (which are also in bits/bits-per-second).

For the first implementation, only config.INITIAL_ENHANCEMENT_LEVELS
(k=0 and k=1) are used - i.e. the action is effectively a binary
per-tile choice (stay at base, or move to k=1) rather than the full
k=0..6 graduated range the mapping function already supports.
"""

import numpy as np

import config


def frame_index_for_gop(gop_index: int) -> int:
    """First frame of a given GOP. Bitrate is constant within a GOP
    (verified empirically), so any frame in the GOP would give the same
    answer - the first frame is just a simple, arbitrary pick.
    """
    return gop_index * config.GOP_SIZE_FRAMES


def tile_bitrate_at_level(rd_data: dict, video_array_index: int, gop_index: int, k: int) -> np.ndarray:
    """Bitrate (bps) for every tile at enhancement level k, for one GOP.
    Returns an array of shape (config.N_TILES,).
    """
    qp_idx = config.qp_array_index_for_enhancement_level(k)
    frame_idx = frame_index_for_gop(gop_index)
    return rd_data["video_bitrate_data"][video_array_index][qp_idx, :, frame_idx]


def compute_A_t(rd_data: dict, video_array_index: int, gop_index: int,
                 enhanced_tile_mask: np.ndarray, enhanced_level: int = 1) -> dict:
    """Compute A^t and its base/enhancement split for one time slot.

    enhanced_tile_mask: boolean (or 0/1) array of shape (N_TILES,) - the
        action x^t, True/1 where tile i is enhanced this slot.
    enhanced_level: which k level (config.INITIAL_ENHANCEMENT_LEVELS)
        enhanced tiles are sent at. Defaults to 1 (the only non-base
        level in the first-implementation scope).

    Returns a dict: {"A_t", "A_base", "A_enh", "base_bitrate",
    "enhanced_bitrate", "delta"} - the totals plus the raw per-tile
    arrays, in case a caller wants to inspect them (e.g. for logging).
    """
    mask = np.asarray(enhanced_tile_mask, dtype=bool)
    assert mask.shape == (config.N_TILES,), f"expected shape ({config.N_TILES},), got {mask.shape}"

    base_bitrate = tile_bitrate_at_level(rd_data, video_array_index, gop_index, k=0)
    enhanced_bitrate = tile_bitrate_at_level(rd_data, video_array_index, gop_index, k=enhanced_level)
    delta = enhanced_bitrate - base_bitrate

    A_base = float(base_bitrate.sum())
    A_enh = float((delta * mask).sum())

    return {
        "A_t": A_base + A_enh,
        "A_base": A_base,
        "A_enh": A_enh,
        "base_bitrate": base_bitrate,
        "enhanced_bitrate": enhanced_bitrate,
        "delta": delta,
    }
