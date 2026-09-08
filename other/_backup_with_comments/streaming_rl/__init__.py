"""
streaming_rl - implementation package for the tile-selective aerial 360
video streaming RL project.

Modules:
    data_loader     - loads hn.mat / rd.mat / the video catalog, and
                       classifies HMD_data cells as valid/empty/placeholder.
                       Ports the already-verified logic from
                       explore_data.ipynb - nothing here is new analysis,
                       just the same checks made reusable.
    channel_model    - Rician-fading + path-loss channel gain sampling.
    layer_model      - base/enhancement-layer data-volume (A^t) computation.
    viewport         - yaw/pitch -> viewport rectangle -> per-tile
                        coverage fraction.
    environment      - the gymnasium Env tying the above together.

All modules read their parameters from config.py at the project root -
nothing in here hardcodes a value that config.py already owns.
"""
