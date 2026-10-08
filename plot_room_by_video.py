"""
plot_room_by_video.py - figures for room_by_video.py:
  room_by_video.png      coverage vs tile budget per video (literature rules,
                         random, RL estimate, perfect-prediction ceiling)
  waste_B8.png           extra tiles per segment that land in / miss the actual viewport (tile budget 8)
  example_segment.png    one real held-out segment: where each method's extra tiles go

Usage: .venv/Scripts/python.exe plot_room_by_video.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import config
import plot_meeting_followup_exp3 as pmf  # house style
import room_to_learn as rl

D = "VIDEO_SURVEY/room_by_video"
FAM = {"literature": ("Best literature rule (budget-tuned)", "#e07b39", "-"),
       "random": ("Random extra tiles", "#9a988f", ":"),
       "rl_estimate": ("RL estimate (adaptive rule, not a baseline)", "#7b3fa0", "-"),
       "ceiling": ("Ceiling: perfect prediction", pmf.INK, "--")}


def main():
    df = pd.read_csv(f"{D}/room_by_video.csv")
    videos = list(dict.fromkeys(df.video))

    # ---------- room_by_video.png ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, v in zip(axes.ravel(), videos):
        d = df[df.video == v]
        pred = d[(d.family == "literature") & (d.choice == "predicted viewport only")]["cov"]
        base = d[d.family == "literature"]["cov"].min() if pred.empty else pred.min()
        for fam, (lab, col, ls) in FAM.items():
            s = d[d.family == fam].sort_values("B_bar")
            if s.empty:
                continue
            ax.plot(s.B_bar, s["cov"], ls, color=col, lw=2.0, label=lab)
            ax.scatter(s.B_bar, s["cov"], color=[col if f else "white" for f in s.held_feasible], edgecolor=col, s=40, zorder=3)
        lit = d[d.family == "literature"].set_index("B_bar")["cov"]
        rle = d[d.family == "rl_estimate"].set_index("B_bar")["cov"]
        common = sorted(set(lit.index) & set(rle.index))
        if common:
            ax.fill_between(common, lit[common], rle[common], color="#7b3fa0", alpha=0.12, label="expected room for RL")
        ax.axhline(base, color="#9a988f", lw=1.0, ls="--", label=f"predicted viewport only ({base:.3f})")
        ax.set_title(v); ax.set_xlabel("tile budget"); ax.set_ylabel("coverage (held-out viewers)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("How much better than the literature rules can RL get? (stall budget 0.3, 50 mW; hollow = over budget on held-out)")
    fig.tight_layout(); fig.savefig(f"{D}/room_by_video.png", dpi=130); plt.close(fig)

    # ---------- waste_B8.png ----------
    d8 = df[df.B_bar == 8.0]
    fig, axes = plt.subplots(1, len(videos), figsize=(4.4 * len(videos), 5), sharey=True)
    for ax, v in zip(axes, videos):
        d = d8[d8.video == v].set_index("family")
        fams = [f for f in ["literature", "random", "rl_estimate", "ceiling"] if f in d.index]
        useful = [d.loc[f, "useful"] for f in fams]
        wasted = [d.loc[f, "extra"] - d.loc[f, "useful"] for f in fams]
        x = np.arange(len(fams))
        ax.bar(x, useful, 0.6, color="#2f9e44", label="lands in the actual viewport")
        ax.bar(x, wasted, 0.6, bottom=useful, color="#e34948", alpha=0.75, label="misses the actual viewport")
        for xi, u, w_ in zip(x, useful, wasted):
            if u + w_ > 0:
                ax.annotate(f"{u / (u + w_):.0%} useful", (xi, u + w_), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=pmf.INK)
        ax.set_xticks(x)
        ax.set_xticklabels([{"literature": f"literature\n({d.loc['literature', 'choice']})", "random": "random",
                             "rl_estimate": "RL estimate", "ceiling": "ceiling"}[f] for f in fams], fontsize=7.5)
        ax.set_title(v)
    axes[0].set_ylabel("extra tiles per segment"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Where the extra tiles go (tile budget 8, held-out viewers)")
    fig.tight_layout(); fig.savefig(f"{D}/waste_B8.png", dpi=130); plt.close(fig)

    # ---------- example_segment.png ----------
    steps = pd.read_pickle(f"{D}/example_steps_B8.pkl")
    best = None
    for (v, fam, name), recs in steps.items():
        if fam != "literature":
            continue
        rle_key = next((k for k in steps if k[0] == v and k[1] == "rl_estimate"), None)
        if rle_key is None:
            continue
        for a, b in zip(recs, steps[rle_key]):
            move = abs(rl.wsigned(a["actual"][0] - a["pred"][0]))
            lit_useful = (a["extra"] & a["hit"]).sum(); lit_n = a["extra"].sum()
            rle_useful = (b["extra"] & b["hit"]).sum()
            if 30 <= move <= 80 and lit_n >= 2:
                score = rle_useful - lit_useful
                if best is None or score > best[0]:
                    best = (score, v, name, a, b)
    if best:
        _, v, lit_name, a, b = best
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
        for ax, rec, title, col in [(axes[0], a, f"Literature: {lit_name}", "#e07b39"), (axes[1], b, "RL estimate", "#7b3fa0")]:
            for i in range(config.N_TILES):
                t0, t1, f0, f1 = rl.viewport.tile_bounds(i)
                if rec["mand"][i]:
                    fc = "#d9d8d2"
                elif rec["extra"][i]:
                    fc = col
                else:
                    fc = "white"
                ax.add_patch(Rectangle((t0, f0), t1 - t0, f1 - f0, facecolor=fc, edgecolor="#c3c2b7", lw=0.6))
            th, ph = rec["actual"]
            for shift in (-360, 0, 360):
                ax.add_patch(Rectangle((th - config.V_THETA_DEG + shift, ph - config.V_PHI_DEG), 2 * config.V_THETA_DEG,
                                       2 * config.V_PHI_DEG, fill=False, edgecolor="#e34948", lw=2.2))
            pt, pp = rec["pred"]
            ax.plot([pt], [pp], "x", color=pmf.INK, ms=10, mew=2)
            ax.plot([th], [ph], "o", color="#e34948", ms=7)
            useful = int((rec["extra"] & rec["hit"]).sum()); n = int(rec["extra"].sum())
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
            ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)")
            ax.set_title(f"{title}: {useful} of {n} extra tiles land in the viewport")
        fig.suptitle(f"{v}, one held-out segment: grey = forced (predicted viewport), colour = extra tiles, "
                     "red box = where the viewer actually looked (x = predicted direction)")
        fig.tight_layout(); fig.savefig(f"{D}/example_segment.png", dpi=130); plt.close(fig)
    print("saved figures to", D)


if __name__ == "__main__":
    main()
