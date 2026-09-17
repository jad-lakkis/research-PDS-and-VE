"""
layer_model.py - base/enhancement-layer data-volume (A^t) and viewport
quality (PSNR) computation.

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

compute_viewport_psnr_db (added later, same file since it needs the
same rd_data/QP-index plumbing as compute_A_t) turns rd.mat's real,
per-(QP, tile, frame) Y-MSE data into a single viewport-PSNR number per
step, verified directly against the loaded data before being wired in
(zero NaNs for Runner, MSE values in a plausible sub-1 to low-teens
range, see other/decision_log.md).
"""

import numpy as np

import config
from streaming_rl import viewport


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


def compute_viewport_psnr_db(rd_data: dict, video_array_index: int, gop_index: int,
                              enhanced_tile_mask: np.ndarray, enhanced_level: int,
                              theta_deg: float, phi_deg: float) -> float:
    """Viewport PSNR (dB) for one GOP.

    Looks up each tile's real Y-MSE (rd_data["video_ymse_data"], shape
    (N_QP_LEVELS, N_TILES, n_frames), verified against the actual loaded
    data - zero NaNs, plausible MSE range) at whichever QP level that
    tile was actually sent this step: base QP if not enhanced, the
    enhanced_level's QP if enhanced - same base/enhanced split compute_A_t
    already uses. Pools those per-tile MSE values into one area-weighted
    average MSE across the actual viewport, using the SAME per-tile
    overlap fractions viewport.coverage() weights C^t_cov by
    (viewport.tile_overlap_fractions) - not a separate/different
    weighting scheme. Applies one PSNR conversion at the end, not an
    average of per-tile PSNR values: PSNR is already a log10 transform,
    so a mean of dB numbers isn't a physically meaningful average error;
    pooling in MSE space first is the standard convention (how frame-
    level PSNR is computed from block MSEs in video quality assessment).

    MAX_I=255: Runner is 8-bit (other/Readme - Dataset info.txt, video
    catalog table), confirmed, not assumed.
    """
    mask = np.asarray(enhanced_tile_mask, dtype=bool)
    assert mask.shape == (config.N_TILES,), f"expected shape ({config.N_TILES},), got {mask.shape}"

    frame_idx = frame_index_for_gop(gop_index)
    base_qp_idx = config.BASE_QP_ARRAY_INDEX
    enhanced_qp_idx = config.qp_array_index_for_enhancement_level(enhanced_level)
    ymse = rd_data["video_ymse_data"][video_array_index]  # (N_QP_LEVELS, N_TILES, n_frames)
    per_tile_mse = np.where(mask, ymse[enhanced_qp_idx, :, frame_idx], ymse[base_qp_idx, :, frame_idx])

    fractions = viewport.tile_overlap_fractions(theta_deg, phi_deg)
    total_frac = float(fractions.sum())
    if total_frac <= 0.0:
        # Degenerate case, not expected in practice (the viewport rectangle
        # always has positive area and therefore always overlaps some
        # tile), guarded rather than silently trusted.
        return float("nan")

    pooled_mse = float(np.sum(fractions * per_tile_mse) / total_frac)
    max_i = 255.0
    return float(10.0 * np.log10((max_i ** 2) / pooled_mse))
