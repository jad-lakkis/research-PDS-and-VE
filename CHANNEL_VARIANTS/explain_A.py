"""
explain_A.py - what happens on candidate channel A, second by second (replay only, no training).

Channel A = today's synthetic model with bandwidth 20 MHz, slow shadowing (12 dB, memory 0.95) and blockages (30 dB,
15% of the time, 3 s long on average). One held-out test episode is drawn as five panels (link, bits sent, buffer,
stall, quality) for: H1 viewport only, H3 average-based, H4 ASL360 threshold (waits for 10 s of playable video),
the link oracle (knows the future link, not the actual viewport) and the oracle (also the actual viewport).
Over all 900 test episodes: what each method does in seconds where the link is short or plentiful.

Outputs (CHANNEL_VARIANTS/): fig7_channel_A_one_episode.png, explain_A_by_link.csv
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/explain_A.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel_variants as cv  # noqa: E402
import replay_channels as rc  # noqa: E402

OUT = "CHANNEL_VARIANTS"
A_KW = dict(W_MHz=20.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15)
G, v, T, H, T0, DISC = rc.G, rc.v, rc.T, rc.H, rc.T0, rc.DISC
COL = {"H1 viewport only": "#7f7f7f", "H3 average-based": "#d62728", "H4 ASL360 threshold": "#9467bd",
       "link oracle": "#1f77b4", "oracle": "#000000"}


def follow(P, R, choose):
    """second by second: choose(t, Z) -> sent tiles (n, 64); returns bits, stall, buffer before, quality, tiles"""
    n = len(P)
    Z = np.zeros(n)
    out = {k: np.zeros((n, H)) for k in ("bits", "stall", "buffer", "psnr", "tiles", "extra")}
    for t in range(H):
        X = choose(t, Z)
        A = v.A_base[t] + (v.delta[t][None, :] * X).sum(axis=1)
        out["buffer"][:, t] = Z
        s = Z + T0 * R[:, t]
        short = s < A
        out["stall"][:, t] = np.where(short, T0 * (1.0 - s / A), 0.0)
        Z = np.where(short, 0.0, s - A)
        out["bits"][:, t] = A
        w = np.stack([T[p]["w"][t] for p in P])
        mse = np.where(X, v.mse_e[t][None, :], v.mse_b[t][None, :])
        out["psnr"][:, t] = 10.0 * np.log10(255.0 ** 2 / ((w * mse).sum(axis=1) / w.sum(axis=1)))
        mand = np.stack([T[p]["mand"][t] for p in P])
        out["tiles"][:, t] = X.sum(axis=1)
        out["extra"][:, t] = (X & ~mand).sum(axis=1)
    return out


def main(kw=A_KW, tag="channel_A", title="Channel A"):
    eps, R_list, r_all = rc.channel_eps(kw, 11, list(G["held"]), rc.N_WIN)
    P, R = [p for p, _ in eps], np.stack(R_list)
    forced = np.stack([G["A_m"][p] for p in P])                       # bits of base + forced tiles, (n, H)
    M = {m: np.stack([rc.MASK[m][p] for p in P]) for m in ("H1 viewport only", "H3 average-based")}
    Xo = rc.oracle(eps, R_list, 11.0, G["EO"], None, return_masks=True)
    Xl = rc.oracle(eps, R_list, 11.0, G["EM"], None, rc.TOP_LINK, return_masks=True)
    res = {"H1 viewport only": follow(P, R, lambda t, Z: M["H1 viewport only"][:, t]),
           "H3 average-based": follow(P, R, lambda t, Z: M["H3 average-based"][:, t]),
           "H4 ASL360 threshold": follow(P, R, lambda t, Z: np.where((Z >= 10.0 * forced[:, t])[:, None],
                                                                     M["H3 average-based"][:, t], M["H1 viewport only"][:, t])),
           "link oracle": follow(P, R, lambda t, Z: Xl[:, t]),
           "oracle": follow(P, R, lambda t, Z: Xo[:, t])}
    d1 = res["H1 viewport only"]["stall"]
    avoid = {m: ((r["stall"] - d1) * DISC).sum(axis=1) for m, r in res.items()}
    print("whole test set (900 episodes): PSNR / avoidable stall / tiles per second")
    for m, r in res.items():
        print(f"  {m:22s} {r['psnr'].mean():6.2f}  {avoid[m].mean():6.3f}  {r['tiles'].mean():5.1f}")

    # every second classed by how much room the link and buffer leave for the forced tiles (on H1's buffer)
    room = (res["H1 viewport only"]["buffer"] + T0 * R) / forced
    cls = np.where(room < 1, "link short (cannot even carry the forced tiles)",
                   np.where(room < 2, "link tight (1-2x the forced tiles)", "link plentiful (> 2x)"))
    rows = []
    for c in ("link short (cannot even carry the forced tiles)", "link tight (1-2x the forced tiles)", "link plentiful (> 2x)"):
        m_ = cls == c
        for m, r in res.items():
            rows.append(dict(seconds=c, share=m_.mean(), method=m, extra_tiles=r["extra"][m_].mean(),
                             extra_stall_per_episode=((r["stall"] - d1) * DISC)[m_].sum() / len(P),
                             quality_gain_dB=(r["psnr"] - res["H1 viewport only"]["psnr"])[m_].mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/explain_{tag}_by_link.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))

    # the example: a link drop after second 8 that makes H3 stall while the oracle does not
    dip = (R < forced) & (np.arange(H)[None, :] >= 8)
    ok = dip.any(axis=1) & (avoid["oracle"] <= 0.02) & (avoid["H3 average-based"] >= 0.3)
    e = int(np.argmax(np.where(ok, avoid["H3 average-based"], -1)))
    print(f"example episode {e}: viewer {P[e]}, avoidable stall H3 {avoid['H3 average-based'][e]:.2f}, "
          f"H4 {avoid['H4 ASL360 threshold'][e]:.2f}, link oracle {avoid['link oracle'][e]:.2f}, oracle {avoid['oracle'][e]:.2f}")
    t = np.arange(H)
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1.2, 1, 0.8, 1.1]))
    low = R[e] < forced[e]
    for ax in axes:
        for k in np.where(low)[0]:
            ax.axvspan(k - 0.5, k + 0.5, color="#d62728", alpha=0.08, lw=0)
    ax = axes[0]
    ax.plot(t, R[e] / 1e6, color="#1f77b4", lw=2, marker="o", ms=3, label="link: bits per second the channel delivers")
    ax.plot(t, forced[e] / 1e6, color="#d62728", ls="--", lw=1.4, label="needed for base layer + forced viewport tiles")
    ax.axhline(cv.NEED_ALL, color="#7f7f7f", ls=":", lw=1.2, label="needed for all 64 tiles (~430)")
    ax.set_ylabel("Mbit/s")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("1. The link (red bands = seconds where it cannot even carry the forced tiles)", fontsize=10, loc="left")
    for m, r in res.items():
        kw = dict(color=COL[m], lw=2.2 if m == "oracle" else 1.5, ls="--" if m == "link oracle" else "-", marker="o", ms=2.5)
        axes[1].plot(t, r["bits"][e] / 1e6, label=m, **kw)
        axes[2].plot(t, r["buffer"][e] / forced[e], **kw)
        axes[3].plot(t, r["stall"][e], **kw)
        axes[4].plot(t, r["psnr"][e], **kw)
    axes[1].plot(t, R[e] / 1e6, color="#1f77b4", lw=1, alpha=0.35)
    axes[1].set_ylabel("Mbit/s")
    axes[1].set_title("2. Bits each method asks for in that second (faint blue = the link)", fontsize=10, loc="left")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    axes[2].axhline(10, color="#9467bd", ls=":", lw=1)
    axes[2].set_ylabel("seconds of video")
    axes[2].set_title("3. Buffer before that second (seconds of playable video stored; dotted = H4's 10-s threshold)", fontsize=10, loc="left")
    axes[3].set_ylabel("seconds")
    axes[3].set_title("4. Stall in that second", fontsize=10, loc="left")
    axes[4].set_ylabel("dB")
    axes[4].set_title("5. Viewport quality (PSNR) in that second", fontsize=10, loc="left")
    axes[4].set_xlabel("second of the episode")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}, one test episode (viewer {P[e]}): stall allowance 0.1, tile budget 11", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{OUT}/fig7_{tag}_one_episode.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:   # channel_variants.rates keywords as JSON, file tag, title
        import json
        main(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3])
    else:
        main()
