"""
search_channels.py - which synthetic channel leaves RL the most room? (replay only, no training)

Same model as today's synthetic channel (MMSP'25: path loss, Rician fading, Shannon rate, 50 mW), with bandwidth
changed and two standard additions (slow shadowing with memory, blockages). 150 random settings drawn from:
  bandwidth 10-40 MHz | Rician K 0-12 dB (Rayleigh in 10% of draws) | shadowing sigma 0-14 dB, memory rho 0.5-0.99
  per second | blockages 0-40 dB deep, 0-30% of the time, 1-15 s long on average
Each setting: held-out viewers x 300 windows; heuristics H1 viewport only, H2 QER 120x60 / 160x80, H3 average-based,
H4 ASL360 threshold (faithful: waits for 10 s of playable video); link oracle (knows the future link, not the actual
viewport) and oracle (also the actual viewport). Budgets: stall = unavoidable + 0.1, tile budget 11; buffer as now.
Room for RL = link oracle - best heuristic within budgets.
Also computed: the ceiling (link oracle and heuristics with an unlimited link) and three reference channels.

Outputs (CHANNEL_VARIANTS/): search_channels.csv, fig8_search.png, fig9_search_top_channels.png
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/search_channels.py
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
N_SETTINGS = 150
HELD = list(rc.G["held"])
SHORT = {"H1 viewport only": "H1", "H2 QER 120x60": "QER 120", "H2 QER 160x80": "QER 160", "H3 average-based": "H3",
         "H4 ASL360 threshold": "H4"}
BEST_COL = {"H1": "#7f7f7f", "QER 120": "#e377c2", "QER 160": "#8c564b", "H3": "#d62728", "H4": "#9467bd"}
REFS = [("today's model", {}), ("candidate A", dict(W_MHz=20.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15))]


def draw(rng):
    return dict(W_MHz=rng.uniform(10, 40), K_dB=None if rng.random() < 0.1 else rng.uniform(0, 12),
                sh_sigma=rng.uniform(0, 14), sh_rho=rng.uniform(0.5, 0.99), blk_att=rng.uniform(0, 40),
                blk_frac=rng.uniform(0, 0.3), blk_mean=rng.uniform(1, 15))


def evaluate(eps, R, r_all=None):
    hs = rc.heuristics([p for p, _ in eps], np.stack(R), None)
    floor = hs["H1 viewport only"]["J_D"]
    avoid = {m: hs[m]["J_D"] - floor for m in hs}
    within = [m for m in hs if avoid[m] <= 0.1 + 1e-9 and hs[m]["J_B"] <= 11.0 + 1e-9]
    best = max(within, key=lambda m: hs[m]["psnr"])
    lo = rc.oracle(eps, R, 11.0, rc.G["EM"], None, rc.TOP_LINK)
    o = rc.oracle(eps, R, 11.0, rc.G["EO"], None)
    out = dict(floor=floor, best=SHORT[best], best_psnr=hs[best]["psnr"], link_oracle=lo["psnr"], oracle=o["psnr"],
               room=lo["psnr"] - hs[best]["psnr"], room_oracle=o["psnr"] - hs[best]["psnr"],
               **{f"psnr {SHORT[m]}": hs[m]["psnr"] for m in hs}, **{f"avoid {SHORT[m]}": avoid[m] for m in hs})
    if r_all is not None:
        s = cv.stats(r_all / 1e6)
        out.update(median=s["median"], below_need_pct=s["pct_below_forced_need"], over_2x_pct=s["pct_over_2x_need"],
                   all64_pct=s["pct_over_all64"], outage_len=s["mean_outage_length_s"], lag1=s["lag1_autocorr"])
    return out


def main():
    t0 = time.time()
    rows = []
    eps, _, _ = rc.channel_eps({}, 11, HELD, rc.N_WIN)
    ceil = evaluate(eps, [np.full(rc.H, 1e12) for _ in eps])
    print(f"ceiling (unlimited link): link oracle {ceil['link_oracle']:.2f}, oracle {ceil['oracle']:.2f}, "
          f"best heuristic {ceil['best']} {ceil['best_psnr']:.2f}", flush=True)
    for name, kw in REFS:
        eps, R, r_all = rc.channel_eps(kw, 11, HELD, rc.N_WIN)
        rows.append(dict(name=name, **{**cv.BASE, **kw}, **evaluate(eps, R, r_all)))
    eps_l = rc.G["eps"]["test"]
    R_l = [rc.G["rates"][w][rc.NH:] for _, w in eps_l]
    rows.append(dict(name="Lumos5G (measured)", **evaluate(eps_l, R_l, np.concatenate(R_l))))
    print(f"references done ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(2026)
    for k in range(N_SETTINGS):
        kw = draw(rng)
        eps, R, r_all = rc.channel_eps(kw, 11, HELD, rc.N_WIN)
        rows.append(dict(name=f"setting {k + 1}", **kw, **evaluate(eps, R, r_all)))
        if (k + 1) % 10 == 0:
            print(f"{k + 1} settings done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/search_channels.csv", index=False)
    plot(df, ceil)


def plot(df, ceil):
    pd.set_option("display.width", 250)
    srch = df[df.name.str.startswith("setting")].sort_values("room", ascending=False)
    cols = ["name", "W_MHz", "K_dB", "sh_sigma", "sh_rho", "blk_att", "blk_frac", "blk_mean", "below_need_pct", "over_2x_pct",
            "floor", "best", "best_psnr", "link_oracle", "room", "avoid H4", "psnr H4"]
    print(df[~df.name.str.startswith("setting")][cols].round(2).to_string(index=False))
    print(srch[cols].head(12).round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for _, r in srch.iterrows():
        axes[0].plot(r.below_need_pct, r.room, "o", ms=7, color=BEST_COL[r.best], alpha=0.85)
        axes[1].scatter(r.below_need_pct, r.over_2x_pct, c=[r.room], cmap="Blues", vmin=0, vmax=max(2.5, srch.room.max()),
                        s=45, edgecolors="#555555", linewidths=0.4)
    for _, r in df[~df.name.str.startswith("setting")].iterrows():
        for ax, yv in ((axes[0], r.room), (axes[1], r.over_2x_pct)):
            ax.plot(r.below_need_pct, yv, marker="*", ms=16, color="#ffbf00", mec="black", ls="none")
            ax.annotate(r["name"], (r.below_need_pct, yv), textcoords="offset points", xytext=(8, 4), fontsize=8)
    top = ceil["link_oracle"] - df.loc[df.name == "today's model", "psnr H1"].iloc[0]   # H1's quality is the same on every channel
    axes[0].axhline(top, color="black", ls="--", lw=1)
    axes[0].text(1, top + 0.05, "most room any channel can give (link oracle at its unlimited-link best, only viewport-only fits)",
                 fontsize=8)
    axes[0].set_xlabel("seconds where the link is below what the forced tiles need (%)")
    axes[0].set_ylabel("room for RL: link oracle - best heuristic within budgets (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="none", color=c, label=f"best heuristic within budgets: {h}")
                            for h, c in BEST_COL.items()] + [Line2D([], [], marker="*", ls="none", ms=14, color="#ffbf00",
                                                                     mec="black", label="reference channels")],
                   fontsize=8, loc="lower right")
    axes[1].set_xlabel("seconds where the link is below what the forced tiles need (%)")
    axes[1].set_ylabel("seconds with at least 2x what the forced tiles need (%)")
    axes[1].set_title("colour = room for RL (darker = more)", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Search over {len(srch)} synthetic channels (same model; bandwidth, shadowing, blockages varied). "
                 "Allowance 0.1, tile budget 11", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_search.png", dpi=150)
    plt.close(fig)

    best4 = srch.head(4)
    fig, axes = plt.subplots(len(best4), 1, figsize=(12, 2.4 * len(best4)), sharex=True)
    for ax, (_, r) in zip(np.atleast_1d(axes), best4.iterrows()):
        kw = {k: r[k] for k in ("W_MHz", "sh_sigma", "sh_rho", "blk_att", "blk_frac", "blk_mean")}
        kw["K_dB"] = None if pd.isna(r.K_dB) else r.K_dB
        x = cv.rates(**{**cv.BASE, **kw}, n=400, seed=5)[100:300]
        ax.plot(np.arange(200), x, color="#1f77b4", lw=1.3)
        ax.fill_between(np.arange(200), 0, x, where=x < cv.NEED_FORCED, color="#d62728", alpha=0.25, step="mid")
        ax.axhline(cv.NEED_FORCED, color="#d62728", ls="--", lw=1.1)
        ax.axhline(cv.NEED_ALL, color="#7f7f7f", ls=":", lw=1.1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Mbit/s")
        ax.grid(alpha=0.3)
        ax.set_title(f"{r['name']}: {r.W_MHz:.0f} MHz, shadowing {r.sh_sigma:.0f} dB (memory {r.sh_rho:.2f}), blockages "
                     f"{r.blk_att:.0f} dB {100 * r.blk_frac:.0f}% of the time, {r.blk_mean:.0f} s long | room {r.room:.2f} dB, "
                     f"best heuristic {r.best}", fontsize=9, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.suptitle("The four channels with the most room: link rate over 200 s (red dashed = forced tiles need, "
                 "grey dotted = all 64 tiles, red = link below the need)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_search_top_channels.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
