"""
viewport.py - yaw/pitch -> viewport rectangle -> per-tile coverage.

This is the most assumption-heavy module in the package. Every
assumption it depends on is recorded in config.py and repeated here at
the point it's used, so nothing is silently baked in:

  1. yaw/pitch -> (theta, phi): theta is defined to equal yaw directly,
     phi to equal pitch directly (config.YAW_RANGE_DEG=[-180,180],
     config.ANGLE_UNITS="degrees", both empirically confirmed; yaw's
     positive direction is config.YAW_POSITIVE_DIRECTION - ASSUMED, NOT
     VERIFIED). roll is dropped entirely, matching MMSP'25.
  2. Actual-viewport rectangle rule (MMSP'25 Sec. II.C): a rectangle in
     (theta, phi) of half-width config.V_THETA_DEG, half-height
     config.V_PHI_DEG, centered on (theta, phi).
  3. Tile-index -> (row, col) layout: config.TILE_INDEX_ORDER
     ("raster_row_major") - ASSUMED, NOT VERIFIED (no such mapping is
     documented anywhere in the dataset or either paper).
  4. Partial-tile visibility: config.PARTIAL_TILE_VISIBILITY_MODE
     ("area_weighted") - CONFIRMED this session: a tile's contribution
     is weighted by the fraction of its area the viewport covers, not
     a binary in/out classification.

theta is treated as periodic (wraps at +/-180); phi is not (a viewport
cannot wrap over a pole in this rectangle-based model - consistent with
V_PHI_DEG=30 being nowhere near the 90-degree pole boundary in practice).
"""

import numpy as np

import config


def yaw_pitch_to_theta_phi(yaw_deg: float, pitch_deg: float) -> tuple:
    """Convert raw yaw/pitch (degrees, as stored in hn.mat) to (theta, phi).

    Identity mapping by design (see module docstring, assumption #1) -
    not logically forced, but the simplest, most defensible default
    absent any evidence pointing to a different transform. Flagged here,
    not just in config.py, since this is the exact point the assumption
    gets used.
    """
    theta = yaw_deg
    phi = pitch_deg
    return theta, phi


def actual_viewport_bounds(theta_deg: float, phi_deg: float) -> tuple:
    """(theta_min, theta_max, phi_min, phi_max) for the actual-viewport
    rectangle centered at (theta_deg, phi_deg). theta bounds are NOT
    wrapped into [-180,180] here - the wraparound is handled later, in
    _interval_overlap_wrapped, so a center near +/-180 still produces a
    geometrically correct (if numerically out-of-range) rectangle.
    """
    return (
        theta_deg - config.V_THETA_DEG,
        theta_deg + config.V_THETA_DEG,
        phi_deg - config.V_PHI_DEG,
        phi_deg + config.V_PHI_DEG,
    )


def tile_bounds(tile_index: int) -> tuple:
    """(theta_min, theta_max, phi_min, phi_max) for a flat tile index
    (0..N_TILES-1), under the assumed raster row-major layout
    (config.TILE_INDEX_ORDER): row = index // cols, col = index % cols,
    theta in [-180, 180), phi in [-90, 90).
    """
    row = tile_index // config.TILE_GRID_COLS
    col = tile_index % config.TILE_GRID_COLS
    theta_min = -180.0 + col * config.TILE_WIDTH_DEG
    theta_max = theta_min + config.TILE_WIDTH_DEG
    phi_min = -90.0 + row * config.TILE_HEIGHT_DEG
    phi_max = phi_min + config.TILE_HEIGHT_DEG
    return theta_min, theta_max, phi_min, phi_max


def _interval_overlap_wrapped(a_min, a_max, b_min, b_max, period=360.0) -> float:
    """Overlap length between interval [a_min,a_max] and [b_min,b_max] on
    a circle of the given period - checks b shifted by -period/0/+period
    against a, so an interval that wraps past the +/-180 boundary still
    overlaps correctly with tiles on the other side of that boundary.
    Assumes (a_max - a_min) < period, which holds here (2*V_THETA=120 < 360).
    """
    total = 0.0
    for shift in (-period, 0.0, period):
        lo = max(a_min, b_min + shift)
        hi = min(a_max, b_max + shift)
        if hi > lo:
            total += hi - lo
    return total


def _interval_overlap(a_min, a_max, b_min, b_max) -> float:
    """Plain (non-wrapped) interval overlap length, for phi."""
    lo = max(a_min, b_min)
    hi = min(a_max, b_max)
    return max(0.0, hi - lo)


def tile_overlap_fractions(theta_deg: float, phi_deg: float) -> np.ndarray:
    """For every tile (0..N_TILES-1), the fraction of that tile's area
    covered by the actual-viewport rectangle centered at (theta_deg,
    phi_deg). Returns an array of shape (N_TILES,), values in [0,1].
    This directly implements config.PARTIAL_TILE_VISIBILITY_MODE ==
    "area_weighted" - there is currently no other mode implemented,
    since that's the only one confirmed so far.
    """
    assert config.PARTIAL_TILE_VISIBILITY_MODE == "area_weighted", (
        f"viewport.py only implements 'area_weighted'; "
        f"config says '{config.PARTIAL_TILE_VISIBILITY_MODE}'"
    )

    v_theta_min, v_theta_max, v_phi_min, v_phi_max = actual_viewport_bounds(theta_deg, phi_deg)
    tile_area = config.TILE_WIDTH_DEG * config.TILE_HEIGHT_DEG

    fractions = np.zeros(config.N_TILES)
    for i in range(config.N_TILES):
        t_theta_min, t_theta_max, t_phi_min, t_phi_max = tile_bounds(i)
        theta_overlap = _interval_overlap_wrapped(v_theta_min, v_theta_max, t_theta_min, t_theta_max)
        phi_overlap = _interval_overlap(v_phi_min, v_phi_max, t_phi_min, t_phi_max)
        fractions[i] = (theta_overlap * phi_overlap) / tile_area

    return fractions


def coverage(enhanced_tile_mask: np.ndarray, theta_deg: float, phi_deg: float) -> float:
    """C^t_cov (eq. 11), computed under the area-weighted partial-tile
    convention: the enhanced region's area-weighted overlap with the
    actual viewport, divided by the viewport's own total area (which
    is always (2*V_THETA)*(2*V_PHI), a constant).

    enhanced_tile_mask: boolean/0-1 array of shape (N_TILES,) - E^t(x^t).
    """
    mask = np.asarray(enhanced_tile_mask, dtype=float)
    fractions = tile_overlap_fractions(theta_deg, phi_deg)
    tile_area = config.TILE_WIDTH_DEG * config.TILE_HEIGHT_DEG

    covered_area = float(np.sum(mask * fractions) * tile_area)
    viewport_area = (2 * config.V_THETA_DEG) * (2 * config.V_PHI_DEG)
    return covered_area / viewport_area


def mandatory_tile_mask(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Boolean mask (shape (N_TILES,)) of every tile with nonzero overlap
    with the viewport rectangle centered at (theta_deg, phi_deg).

    Used to force-enhance predicted-viewport tiles per eq. 2 ("The
    predicted viewport tiles are sent as enhanced tiles. In addition,
    the agent selects any other tile..."): the caller passes the
    PREDICTED (theta, phi), not the actual one - see
    environment.py::step(). A tile is the smallest transmittable unit,
    so ANY nonzero overlap forces the whole tile to mandatory-enhanced
    status - a separate, binary decision from tile_overlap_fractions
    above, which weights a tile's *coverage/reward* contribution
    continuously by its visible area once it's already been sent.
    """
    return tile_overlap_fractions(theta_deg, phi_deg) > 0.0
