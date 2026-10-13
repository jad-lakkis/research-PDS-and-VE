"""
sensitivity_A.py - how robust is candidate A? (replay only, no training)

Candidate A = today's synthetic channel (same path loss, Rician K 12 dB fading, Shannon rate, 50 mW) with
bandwidth 20 MHz, slow shadowing (sigma 12 dB, memory rho 0.95) and blockages (30 dB, 15% of the time, 3 s on
average). Each of these settings is moved up and down one at a time; the budgets (stall allowance, tile budget) are
varied at A's channel; and A itself is repeated with other random draws to see the noise in the numbers.
Room for RL = link oracle (knows the future link, not the actual viewport) - best heuristic within budgets.
Same replay as replay_channels.py (held-out viewers x 300 windows, no buffer cap).

Outputs (CHANNEL_VARIANTS/): sensitivity_A.csv, sensitivity_A_budgets.csv, fig6_sensitivity_A.png
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/sensitivity_A.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel_variants as cv  # noqa: E402
import replay_channels as rc  # noqa: E402

OUT = "CHANNEL_VARIANTS"
A = dict(W_MHz=20.0, K_dB=12.0, sh_sigma=12.0, sh_rho=0.95, blk_att=30.0, blk_frac=0.15, blk_mean=3.0)
SWEEPS = [("W_MHz", "bandwidth (MHz)", [15.0, 17.5, 20.0, 22.5, 25.0]),
          ("sh_sigma", "shadowing sigma (dB)", [8.0, 10.0, 12.0, 14.0]),
          ("sh_rho", "shadowing memory rho (per s)", [0.8, 0.9, 0.95, 0.98]),
          ("blk_att", "blockage depth (dB)", [20.0, 25.0, 30.0, 35.0]),
          ("blk_frac", "share of time blocked", [0.05, 0.10, 0.15, 0.20, 0.25]),
          ("blk_mean", "mean blockage length (s)", [1.0, 2.0, 3.0, 5.0]),
          ("K_dB", "Rician K (dB)", [12.0, 6.0, 3.0, 0.0])]
ALLOWS, B_BARS = (0.05, 0.1, 0.2, 0.3), (8.0, 11.0, 14.0)
SEEDS = (11, 21, 31)
HELD = list(rc.G["held"])
SHORT = {"H1 viewport only": "H1", "H2 QER 120x60": "QER 120", "H2 QER 160x80": "QER 160", "H3 average-based": "H3",
         "H4 ASL360 threshold": "H4"}
BEST_COL = {"H1": "#7f7f7f", "QER 120": "#e377c2", "QER 160": "#8c564b", "H3": "#d62728", "H4": "#9467bd"}


def episodes(kw, seed):
    return rc.channel_eps(kw, seed, HELD, rc.N_WIN)


def score(eps, R, hs, r_all, allow=0.1, b_bar=11.0, with_oracle=True):
    floor = hs["H1 viewport only"]["J_D"]
    avoid = {m: hs[m]["J_D"] - floor for m in hs}
    within = [m for m in hs if avoid[m] <= allow + 1e-9 and hs[m]["J_B"] <= b_bar + 1e-9]
    best = max(within, key=lambda m: hs[m]["psnr"])
    lo = rc.oracle(eps, R, b_bar, rc.G["EM"], None, rc.TOP_LINK, allow=allow)
    out = dict(allow=allow, b_bar=b_bar, floor=floor, best=SHORT[best], best_psnr=hs[best]["psnr"], link_oracle=lo["psnr"],
               room_link=lo["psnr"] - hs[best]["psnr"], **{f"avoid {SHORT[m]}": avoid[m] for m in hs})
    if with_oracle:
        o = rc.oracle(eps, R, b_bar, rc.G["EO"], None, allow=allow)
        out.update(oracle=o["psnr"], room_oracle=o["psnr"] - hs[best]["psnr"])
    s = cv.stats(r_all / 1e6)
    out.update(median=s["median"], below_need_pct=s["pct_below_forced_need"], over_2x_pct=s["pct_over_2x_need"],
               lag1=s["lag1_autocorr"])
    return out


def main():
    t0 = time.time()
    rows = []
    for key, label, values in SWEEPS:
        for val in values:
            kw = {**A, key: val}
            eps, R, r_all = episodes(kw, SEEDS[0])
            hs = rc.heuristics([p for p, _ in eps], np.stack(R), None)
            rows.append(dict(param=key, label=label, value=val, is_A=(val == A[key]), **score(eps, R, hs, r_all)))
        print(f"{label}: done ({time.time() - t0:.0f} s)", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(f"{OUT}/sensitivity_A.csv", index=False)

    brows = []
    for seed in SEEDS:
        eps, R, r_all = episodes(A, seed)
        hs = rc.heuristics([p for p, _ in eps], np.stack(R), None)
        grid = [(a, b) for a in ALLOWS for b in B_BARS] if seed == SEEDS[0] else [(0.1, 11.0)]
        for a, b in grid:
            brows.append(dict(seed=seed, **score(eps, R, hs, r_all, a, b, with_oracle=False)))
        print(f"budgets / seed {seed}: done ({time.time() - t0:.0f} s)", flush=True)
    bud = pd.DataFrame(brows)
    bud.to_csv(f"{OUT}/sensitivity_A_budgets.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["label", "value", "below_need_pct", "over_2x_pct", "lag1", "floor", "best", "best_psnr", "link_oracle", "oracle",
            "room_link", "room_oracle", "avoid QER 120", "avoid H4"]
    print(sweep[cols].round(2).to_string(index=False))
    seeds = bud[(bud.allow == 0.1) & (bud.b_bar == 11.0)]
    print("A at allowance 0.1 / tile budget 11, three random draws: room", seeds.room_link.round(3).tolist())
    print(bud[bud.seed == SEEDS[0]].pivot_table(index="allow", columns="b_bar", values="room_link").round(2).to_string())
    print(bud[bud.seed == SEEDS[0]].pivot_table(index="allow", columns="b_bar", values="best", aggfunc="first").to_string())

    # figure: one panel per setting + the budget grid
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (key, label, values) in zip(axes.flat, SWEEPS):
        d = sweep[sweep.param == key]
        ax.plot(d.value, d.room_oracle, color="#000000", ls="--", lw=1.2, label="oracle - best heuristic")
        ax.plot(d.value, d.room_link, color="#1f77b4", lw=2, label="link oracle - best heuristic")
        for _, r in d.iterrows():
            ax.plot(r.value, r.room_link, "o", ms=9, color=BEST_COL[r.best], mec="black" if r.is_A else BEST_COL[r.best], mew=1.6)
            ax.annotate(f"{r.below_need_pct:.0f}%", (r.value, r.room_link), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7.5, color="#555555")
        ax.set_xlabel(label)
        ax.set_ylim(0, 4.6)
        ax.set_ylabel("room for RL (dB)")
        ax.grid(alpha=0.3)
        ax.set_title(f"A uses {A[key]:g}", fontsize=9.5)
    ax = axes.flat[-1]
    g = bud[bud.seed == SEEDS[0]].pivot_table(index="allow", columns="b_bar", values="room_link")
    bst = bud[bud.seed == SEEDS[0]].pivot_table(index="allow", columns="b_bar", values="best", aggfunc="first")
    im = ax.imshow(g.values, cmap="Blues", vmin=0, vmax=4, origin="lower", aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.1f}\n({bst.values[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if g.values[i, j] > 2.5 else "black")
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels([f"{b:g}" for b in g.columns])
    ax.set_yticks(range(g.shape[0]))
    ax.set_yticklabels([f"{a:g}" for a in g.index])
    ax.set_xlabel("tile budget")
    ax.set_ylabel("stall allowance")
    ax.set_title("A's channel, other budgets: room (best heuristic)", fontsize=9.5)
    handles = [Line2D([], [], color="#1f77b4", lw=2, label="link oracle - best heuristic within budgets"),
               Line2D([], [], color="#000000", ls="--", lw=1.2, label="oracle - best heuristic within budgets")]
    handles += [Line2D([], [], marker="o", ls="none", ms=9, color=c, label=f"best heuristic: {h}") for h, c in BEST_COL.items()]
    handles += [Line2D([], [], marker="o", ls="none", ms=9, color="white", mec="black", mew=1.6, label="A's own value")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Around candidate A: room for RL when one setting changes (allowance 0.1, tile budget 11). "
                 "Grey % = seconds below the forced-tiles need", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(f"{OUT}/fig6_sensitivity_A.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
