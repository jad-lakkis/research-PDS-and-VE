"""
link_condition_tiles.py - WHEN does each method send its extra tiles? Every second of
every test episode is put in one of three groups by how much the link + buffer can carry
(Z_t + T0*R_t) compared with the forced stream (base layer + predicted-viewport tiles):
  outage       can't even carry the forced stream (stall for everyone)
  tight        1-2x the forced stream
  comfortable  more than 2x the forced stream
Stall allowance 0.1, tile budget 12.

Usage: .venv/Scripts/python.exe link_condition_tiles.py
"""
import numpy as np
import pandas as pd

import budget_sweep_lumos as b
import fastsim as fs

b.setup()
G = b.G
v, T, eps = G["v"], G["v"].traj, G["eps"]["test"]
orc = b.oracle(("test", 0.1, 12.0), return_masks=True)
methods = {"literature, tuned (threshold, margin 0.35x)": b.policy(("mask", "avg 0.35x", True)),
           "RL estimate": b.policy(("rl", "EM", 2, 0.5, 2.0, "now")),
           "oracle": lambda s: orc[(s["pos"], s["key"])][s["t"]]}
rows = []
for name, pol in methods.items():
    for p, w in eps:
        R, Z = G["rates"][w][b.N_HIST:], 0.0
        Zf = 0.0   # buffer of the viewport-only policy, to define the group the same way for every method
        for t in range(fs.H):
            s = dict(pos=p, key=w, t=t, Z=Z, R=R[t], Rh=G["rates"][w][t:t + b.N_HIST][::-1], tr=T[p], video=v)
            ex = pol(s) & ~T[p]["mand"][t]
            Af = G["A_m"][p][t]
            A = Af + v.delta[t][ex].sum()
            D = 0.0 if Z + R[t] >= A else 1.0 - (Z + R[t]) / A
            Df = 0.0 if Zf + R[t] >= Af else 1.0 - (Zf + R[t]) / Af
            room = (Zf + R[t]) / Af
            grp = "outage" if room < 1 else ("tight" if room < 2 else "comfortable")
            rows.append(dict(method=name, group=grp, extra=ex.sum(), extra_stall=fs.DISC[t] * (D - Df)))
            Z = max(Z + R[t] - A, 0.0); Zf = max(Zf + R[t] - Af, 0.0)
df = pd.DataFrame(rows)
n_eps = len(eps)
g = df.groupby(["method", "group"]).agg(seconds_share=("extra", "size"), extra_tiles=("extra", "mean"), stall=("extra_stall", "sum"))
g["seconds_share"] = g.seconds_share / (n_eps * fs.H)
g["stall"] = g.stall / n_eps
pd.set_option("display.width", 200)
print(g.round(3).to_string())
print(df.groupby("method").extra.mean().round(2))
