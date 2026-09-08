"""
test_viewport.py - empirical verification of viewport.mandatory_tile_mask
(eq. 2's forced predicted-viewport enhancement). Plain assertions, no
pytest dependency, matching tests/test_channel_model.py's convention.

Run: .venv\\Scripts\\python.exe tests\\test_viewport.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl import viewport

N_SAMPLES = 2000
# Keep phi away from the +/-90 pole so the viewport rectangle never
# clips against it - the geometric [9,16] tile bound (config.py) assumes
# a full, unclipped 120x60 deg rectangle.
PHI_MARGIN_DEG = config.V_PHI_DEG + 5.0


def test_mandatory_mask_tile_count_within_geometric_bound():
    rng = np.random.default_rng(0)
    thetas = rng.uniform(-180.0, 180.0, size=N_SAMPLES)
    phis = rng.uniform(-90.0 + PHI_MARGIN_DEG, 90.0 - PHI_MARGIN_DEG, size=N_SAMPLES)

    counts = []
    for theta, phi in zip(thetas, phis):
        mask = viewport.mandatory_tile_mask(theta, phi)
        assert mask.shape == (config.N_TILES,), f"expected shape ({config.N_TILES},), got {mask.shape}"
        assert mask.dtype == np.bool_, f"expected a boolean mask, got dtype {mask.dtype}"
        n = int(mask.sum())
        assert config.B_BAR_MIN_TILES <= n <= config.B_BAR_MAX_TILES, (
            f"theta={theta:.2f}, phi={phi:.2f}: mandatory_tile_mask picked {n} tiles, "
            f"outside the geometric bound [{config.B_BAR_MIN_TILES}, {config.B_BAR_MAX_TILES}]"
        )
        counts.append(n)

    print(f"[PASS] mandatory_tile_mask tile count in [{config.B_BAR_MIN_TILES}, "
          f"{config.B_BAR_MAX_TILES}] for all {N_SAMPLES:,} sampled positions "
          f"(observed range [{min(counts)}, {max(counts)}])")


def test_mandatory_mask_consistent_with_overlap_fractions():
    rng = np.random.default_rng(1)
    thetas = rng.uniform(-180.0, 180.0, size=200)
    phis = rng.uniform(-90.0 + PHI_MARGIN_DEG, 90.0 - PHI_MARGIN_DEG, size=200)

    for theta, phi in zip(thetas, phis):
        fractions = viewport.tile_overlap_fractions(theta, phi)
        mask = viewport.mandatory_tile_mask(theta, phi)
        assert np.all(fractions[mask] > 0.0), "mandatory_tile_mask includes a tile with zero overlap"
        assert np.all(fractions[~mask] == 0.0), "mandatory_tile_mask excludes a tile with nonzero overlap"

    print("[PASS] mandatory_tile_mask agrees with tile_overlap_fractions "
          "(mask True iff overlap fraction > 0) across 200 sampled positions")


if __name__ == "__main__":
    test_mandatory_mask_tile_count_within_geometric_bound()
    test_mandatory_mask_consistent_with_overlap_fractions()
    print()
    print("All viewport tests passed.")
