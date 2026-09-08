"""
data_loader.py - loading and validating the raw dataset.

Everything in this file is a direct port of logic already built and
verified in explore_data.ipynb (Sections A and B) - the video-catalog
regex parse and the HMD-cell validity classifier. No new analysis
happens here; this just makes that already-checked logic importable
and reusable from the actual environment code, instead of living only
inside notebook cells.
"""

import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

import config


def load_video_catalog(readme_path: str = None) -> pd.DataFrame:
    """Parse the 15-video metadata table out of the dataset's README.

    Ported verbatim from explore_data.ipynb Section A. Returns a
    DataFrame with columns: video_index (1-based, matches the README's
    "no" column), array_index (0-based, matches .mat array indexing),
    name, frames, fps, resolution, bit_depth, projection.
    """
    path = readme_path or config.README_PATH
    with open(path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    rows = []
    for line in readme_text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(.+)$", line)
        if not m:
            continue
        idx_str, rest = m.groups()
        fields = [x.strip() for x in re.split(r"\s*-\s*", rest) if x.strip()]
        if len(fields) != 6:
            continue  # defensively skip anything that doesn't parse cleanly
        name, frames, fps, resolution, depth, projection = fields
        rows.append({
            "video_index": int(idx_str),
            "array_index": int(idx_str) - 1,
            "name": name,
            "frames": int(frames),
            "fps": int(fps),
            "resolution": resolution,
            "bit_depth": depth,
            "projection": projection,
        })

    catalog = pd.DataFrame(rows)
    assert len(catalog) == config.N_VIDEOS, (
        f"expected {config.N_VIDEOS} videos, parsed {len(catalog)} - "
        f"check the regex against a possibly-changed readme"
    )
    return catalog


def load_hn_data(path: str = None) -> np.ndarray:
    """Load hn.mat and return the raw HMD_data cell array (shape (15, 34))."""
    path = path or config.HN_MAT_PATH
    hn = loadmat(path, simplify_cells=True)
    return hn["HMD_data"]


def load_rd_data(path: str = None) -> dict:
    """Load rd.mat and return the full dict of its variables (video_bitrate_data,
    video_ymse_data, RtoD_exp/pow, QPtoR_exp/pow, plus the __-prefixed scipy keys)."""
    path = path or config.RD_MAT_PATH
    return loadmat(path, simplify_cells=True)


def classify_hmd_cell(cell) -> tuple:
    """Classify a single HMD_data cell as empty / unexpected_shape /
    placeholder / valid. Ported verbatim from explore_data.ipynb Section B.

    - size == 0                     -> "empty" (nothing stored)
    - not a 2D, 4-column array       -> "unexpected_shape" (defensive; not
                                        expected to trigger given the README)
    - all-zero                       -> "placeholder" (a real recording can
                                        never have an all-zero timestamp
                                        column, so exact-zero is a safe test)
    - otherwise                      -> "valid"

    Returns (status, arr) where arr is the cell's contents as an ndarray.
    """
    arr = np.asarray(cell)
    if arr.size == 0:
        return "empty", arr
    if arr.ndim != 2 or arr.shape[1] != 4:
        return "unexpected_shape", arr
    if np.all(arr == 0):
        return "placeholder", arr
    return "valid", arr


def get_valid_traces(hmd_data: np.ndarray, video_array_index: int) -> list:
    """Return every valid (user_slot, Nx4 array) trace for one video.

    Each returned array has columns [yaw, pitch, roll, timestamp], degrees
    and seconds (config.ANGLE_UNITS / config.YAW_RANGE_DEG). Skips empty,
    unexpected-shape, and placeholder cells.
    """
    traces = []
    n_user_slots = hmd_data.shape[1]
    for user_slot in range(n_user_slots):
        status, arr = classify_hmd_cell(hmd_data[video_array_index, user_slot])
        if status == "valid":
            traces.append((user_slot, arr))
    return traces


def load_training_video_bundle() -> dict:
    """Convenience loader for config.TRAINING_VIDEO_ARRAY_INDEX (Runner).

    Returns a dict with:
        "catalog_row"   - the video's row from the catalog (pandas Series)
        "rd_data"       - the full rd.mat dict (all 15 videos - callers index
                           into it with the video's array_index as needed)
        "valid_traces"  - list of (user_slot, Nx4 array) for this video only
    """
    catalog = load_video_catalog()
    row = catalog.loc[catalog.array_index == config.TRAINING_VIDEO_ARRAY_INDEX].iloc[0]
    assert row["name"] == config.TRAINING_VIDEO_NAME, (
        f"config.TRAINING_VIDEO_ARRAY_INDEX={config.TRAINING_VIDEO_ARRAY_INDEX} "
        f"points at '{row['name']}', not '{config.TRAINING_VIDEO_NAME}' - "
        f"the catalog and config.py have drifted out of sync"
    )

    rd_data = load_rd_data()
    hn_data = load_hn_data()
    valid_traces = get_valid_traces(hn_data, config.TRAINING_VIDEO_ARRAY_INDEX)

    return {
        "catalog_row": row,
        "rd_data": rd_data,
        "valid_traces": valid_traces,
    }
