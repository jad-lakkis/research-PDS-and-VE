"""
tile_frame.py - viewport-relative coordinates for the agent's tile bits
(TileStreamingEnv(tile_frame="relative")).

Physical tiles: index i = row * 8 + col (config.TILE_INDEX_ORDER), row = elevation
band (phi from -90 upward, 22.5 deg each), col = azimuth band (theta from -180,
45 deg each).

Relative frame: rows stay physical (elevation does not wrap, and poles and equator
differ); columns are counted from the PREDICTED viewport's column c_pred, which
becomes relative column CENTER_COL. Relative bit (row r, column j) is physical tile
(r, (c_pred + j - CENTER_COL) mod 8), i.e. relative columns 0..7 are azimuth offsets
-4..+3 tile columns from the predicted direction. The map is a cyclic column shift,
one-to-one, fixed by the prediction known at decision time - the transmitted tiles,
buffer, reward and constraints are computed on physical tiles exactly as before.
"""
import numpy as np

import config

CENTER_COL = 4


def wrap_deg(d):
    """angle difference wrapped to [-180, 180)"""
    return (d + 180.0) % 360.0 - 180.0


def predicted_column(theta_deg: float) -> int:
    return int(np.floor((theta_deg + 180.0) / config.TILE_WIDTH_DEG)) % config.TILE_GRID_COLS


def column_shift(theta_deg: float) -> int:
    """shift s such that physical column = (relative column + s) mod 8"""
    return predicted_column(theta_deg) - CENTER_COL


def column_offset_deg(theta_deg: float) -> float:
    """where the predicted direction sits inside its tile column, in [-22.5, 22.5)"""
    c = predicted_column(theta_deg)
    centre = -180.0 + (c + 0.5) * config.TILE_WIDTH_DEG
    return float(wrap_deg(theta_deg - centre))


def rel_to_abs(mask, shift: int) -> np.ndarray:
    grid = np.asarray(mask).reshape(config.TILE_GRID_ROWS, config.TILE_GRID_COLS)
    return np.roll(grid, shift, axis=1).reshape(-1)


def abs_to_rel(mask, shift: int) -> np.ndarray:
    grid = np.asarray(mask).reshape(config.TILE_GRID_ROWS, config.TILE_GRID_COLS)
    return np.roll(grid, -shift, axis=1).reshape(-1)
