"""
lumos5g.py - Lumos5G throughput traces as the link rate R_t.

R_t = scale * 1e6 * Throughput_t   (bits/s; Throughput is in Mbit/s, 1-s samples)

Zeros and LTE-fallback periods (nrStatus NOT_RESTRICTED, ~110 Mbit/s) are kept
as recorded. Runs are split into train / validation / test BY RUN (never by
window), stratified by mobility mode x run-mean-throughput tercile (within mode),
so no test second comes from a run seen in training and the splits see similar
link conditions (an unstratified-by-throughput split put the slowest runs in test). Within a run, samples stay in chronological
order; an episode is a contiguous window of `length` seconds.
"""
import numpy as np
import pandas as pd

import config

SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train, validation, test (fraction of runs)
OBS_BUFFER_MODES = ("linear", "log")


def buffer_obs(Z_bits, mode="linear"):
    """The buffer as the observation carries it: Mbit ("linear", the original)
    or log(1 + Z / config.LUMOS5G_OBS_BUFFER_REF_BITS) ("log", --obs-buffer log)."""
    if mode == "linear":
        return Z_bits * config.LUMOS5G_OBS_SCALE
    return np.log1p(Z_bits / config.LUMOS5G_OBS_BUFFER_REF_BITS)


def ratio_obs(ratio, mode="linear"):
    """(Z + T0 R) / forced bits as the observation carries it: as is, or log(1 + ratio)."""
    return ratio if mode == "linear" else np.log1p(ratio)


def load(path=None):
    """{run_num: throughput array (Mbit/s, float)} plus a per-run info frame."""
    d = pd.read_csv(path or config.LUMOS5G_CSV_PATH)
    d = d.sort_values(["run_num", "seq_num"])
    runs = {int(r): x["Throughput"].to_numpy(float) for r, x in d.groupby("run_num")}
    info = d.groupby("run_num").agg(mode=("mobility_mode", "first"), direction=("trajectory_direction", "first"),
                                    seconds=("Throughput", "size"), mean_mbps=("Throughput", "mean"))
    return runs, info


def load_bundle(path=None):
    """Runs, per-run info and the train/val/test split, loaded once and shared by
    every environment instance (like data_loader's video bundle)."""
    runs, info = load(path)
    return {"runs": runs, "info": info, "splits": split_runs(info)}


def split_runs(info, seed=SPLIT_SEED, fractions=SPLIT_FRACTIONS):
    """{'train': [...], 'val': [...], 'test': [...]} run numbers, stratified by mode x throughput tercile."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    tier = info.groupby("mode")["mean_mbps"].transform(lambda x: pd.qcut(x, 3, labels=False))
    for _, g in info.groupby([info["mode"], tier]):
        ids = rng.permutation(g.index.to_numpy())
        n_val, n_test = int(round(fractions[1] * len(ids))), int(round(fractions[2] * len(ids)))
        out["test"] += ids[:n_test].tolist()
        out["val"] += ids[n_test:n_test + n_val].tolist()
        out["train"] += ids[n_test + n_val:].tolist()
    return {k: sorted(v) for k, v in out.items()}


def windows(runs, run_ids, length, stride=None):
    """Contiguous windows [(run, start)] of `length` seconds; non-overlapping by default."""
    stride = stride or length
    return [(r, s) for r in run_ids for s in range(0, len(runs[r]) - length + 1, stride)]


def all_starts(runs, run_ids, length):
    """Every valid (run, start) pair as two int arrays - training episodes are drawn
    uniformly from these (each second of a run is equally likely to start an episode)."""
    pairs = [(r, s) for r in run_ids for s in range(len(runs[r]) - length + 1)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def eval_windows(runs, run_ids, length, n):
    """n fixed windows, evenly spaced over the split's non-overlapping windows."""
    ws = windows(runs, run_ids, length)
    idx = np.round(np.linspace(0, len(ws) - 1, n)).astype(int)
    return [ws[i] for i in idx]


def rate_window(runs, run, start, length, scale=1.0, history=0):
    """R (bits/s) for seconds start..start+length-1, preceded by `history` earlier
    samples (clipped at the run start by repeating the first sample)."""
    y = runs[run]
    idx = np.clip(np.arange(start - history, start + length), 0, len(y) - 1)
    return scale * 1e6 * y[idx]
