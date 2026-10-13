"""
replay_channels.py - how the heuristics and the oracle do on many channels (replay only, no training).

For each channel (synthetic variants from channel_variants.py, plus Lumos5G's test runs for reference):
  episodes  held-out viewers (2, 6, 9) x 300 36-s windows of the channel (Lumos5G: its 2,574 test episodes);
            the stall floor for training is measured on training viewers x 300 other windows
  budgets   the Lumos5G rule: stall = unavoidable stall + allowance 0.1 (judged as avoidable stall on the same
            episodes); tile budget 8 and 11
  buffer    as now (no cap), or capped at 5 s of the forced stream (5 x 129 Mbit; the surplus above it is lost)
  methods   H1 predicted viewport only; H2 QER 120x60 and 160x80; H3 average-based; H4 ASL360 threshold;
            oracle (actual viewport + future link: adds the actual-viewport tiles the prediction missed, largest
            coverage first, pooled over episodes, while the budgets last - as budget_sweep_lumos.oracle);
            link oracle (knows the future link but NOT the actual viewport: same greedy, tiles ranked by how likely
            they are to be seen given the prediction - what perfect link timing alone can add)
Physics as fastsim (verified against the environment). The Lumos5G no-cap case must reproduce
WEEKLY_PROGRESS_2026-10-09_meeting/data/summary_test.csv (checked at start).

Outputs (CHANNEL_VARIANTS/): replay_channels.csv, fig4_heuristics_vs_oracle_by_channel.png, fig5_gap_by_channel.png
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/replay_channels.py
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
import budget_compare_lumos as bc  # noqa: E402
import budget_sweep_lumos as bs  # noqa: E402
import channel_variants as cv  # noqa: E402
import fastsim as fs  # noqa: E402

OUT = "CHANNEL_VARIANTS"
NH, H, NT, T0, DISC = 5, fs.H, fs.NT, fs.T0, fs.DISC
ALLOW, B_BARS = 0.1, (8.0, 11.0)
CAPS = {"no cap": None, "cap 5 s": 5 * 129e6}
N_WIN, TOP_LINK = 300, 15
CHANNELS = [
    ("today's model (10 MHz)", {}),
    ("Rayleigh fading", dict(K_dB=None)),
    ("shadowing 8 dB", dict(sh_sigma=8.0)),
    ("shadowing 12 dB", dict(sh_sigma=12.0)),
    ("blockage 30 dB (20%)", dict(blk_att=30.0)),
    ("20 MHz", dict(W_MHz=20.0)),
    ("20 MHz + shadowing 12 dB", dict(W_MHz=20.0, sh_sigma=12.0)),
    ("20 MHz + Rayleigh + shadowing 8 dB", dict(W_MHz=20.0, K_dB=None, sh_sigma=8.0)),
    ("A: 20 MHz + shadowing 12 + blockage 30", dict(W_MHz=20.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15)),
    ("B: 25 MHz + shadowing 10 + blockage 30", dict(W_MHz=25.0, sh_sigma=10.0, blk_att=30.0, blk_frac=0.15)),
    ("C: 25 MHz + shadowing 12 + blockage 30 + K 3", dict(W_MHz=25.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15, K_dB=3.0)),
]
LUMOS = "Lumos5G (measured, test runs)"
HEUR = ["H1 viewport only", "H2 QER 120x60", "H2 QER 160x80", "H3 average-based", "H4 ASL360 threshold"]

bc.setup()
G = bs.G
v, T = G["v"], G["v"].traj
MASK = {"H1 viewport only": {p: T[p]["mand"] for p in T},
        "H2 QER 120x60": {p: G["masks"]["QER 120x60"][p] | T[p]["mand"] for p in T},
        "H2 QER 160x80": {p: G["masks"]["QER 160x80"][p] | T[p]["mand"] for p in T},
        "H3 average-based": {p: G["masks"]["avg 1.0x"][p] | T[p]["mand"] for p in T}}


def psnr_rows(p, x):
    """viewport PSNR of each second (H,) for the sent tiles x (H, 64)"""
    w = T[p]["w"]
    mse = np.where(x, v.mse_e, v.mse_b)
    return 10.0 * np.log10(255.0 ** 2 / ((w * mse).sum(axis=1) / w.sum(axis=1)))


BITS = {m: {p: v.A_base + (v.delta * MASK[m][p]).sum(axis=1) for p in T} for m in MASK}
QUAL = {m: {p: psnr_rows(p, MASK[m][p]) for p in T} for m in MASK}
NTIL = {m: {p: MASK[m][p].sum(axis=1) for p in T} for m in MASK}


def stall(A, R, z_max):
    """per-second stall (n, H) for bits A and rates R (n, H)"""
    Z, D = np.zeros(len(A)), np.zeros(A.shape)
    for t in range(H):
        s = Z + T0 * R[:, t]
        short = s < A[:, t]
        D[:, t] = np.where(short, T0 * (1.0 - s / A[:, t]), 0.0)
        Z = np.where(short, 0.0, s - A[:, t])
        if z_max is not None:
            Z = np.minimum(Z, z_max)
    return D


def metrics(Q, D, N):
    return dict(psnr=Q.mean(), J_D=(D * DISC).sum(axis=1).mean() / T0, J_B=(N / NT * DISC).sum(axis=1).mean(),
                tiles=N.mean(), stall_pct=100 * (D > 0).mean())


def heuristics(P, R, z_max, h4_unit="playable"):
    """h4_unit: how ASL360's 10-s buffer threshold is counted. "playable" (default since 2026-10-11): 10 s of what a
    second of our video needs before it can play (base layer + forced tiles) - the faithful adaptation, since in our
    model a segment plays only once all its bits arrive. "base": 10 s of base layer only (45 Mbit = 0.35 s of
    playback) - the adaptation used up to the 9 Oct meeting, kept to reproduce those numbers."""
    out = {}
    for m in MASK:
        A = np.stack([BITS[m][p] for p in P])
        out[m] = metrics(np.stack([QUAL[m][p] for p in P]), stall(A, R, z_max), np.stack([NTIL[m][p] for p in P]))
    # H4 ASL360 threshold: H3's region when the buffer holds >= 10 s, else the forced tiles only
    A1, A3 = (np.stack([BITS[m][p] for p in P]) for m in ("H1 viewport only", "H3 average-based"))
    Q1, Q3 = (np.stack([QUAL[m][p] for p in P]) for m in ("H1 viewport only", "H3 average-based"))
    N1, N3 = (np.stack([NTIL[m][p] for p in P]) for m in ("H1 viewport only", "H3 average-based"))
    Z, D, Q, N = np.zeros(len(P)), np.zeros(A1.shape), np.zeros(A1.shape), np.zeros(A1.shape)
    for t in range(H):
        up = Z >= 10.0 * (A1[:, t] if h4_unit == "playable" else v.mean_base_bits)
        A = np.where(up, A3[:, t], A1[:, t])
        Q[:, t], N[:, t] = np.where(up, Q3[:, t], Q1[:, t]), np.where(up, N3[:, t], N1[:, t])
        s = Z + T0 * R[:, t]
        short = s < A
        D[:, t] = np.where(short, T0 * (1.0 - s / A), 0.0)
        Z = np.where(short, 0.0, s - A)
        if z_max is not None:
            Z = np.minimum(Z, z_max)
    out["H4 ASL360 threshold"] = metrics(Q, D, N)
    return out


def ep_stall(A, R, z_max):
    Z, J = 0.0, 0.0
    for t in range(H):
        s = Z + T0 * R[t]
        if s < A[t]:
            J += DISC[t] * (1.0 - s / A[t])
            Z = 0.0
        else:
            Z = s - A[t]
            if z_max is not None and Z > z_max:
                Z = z_max
    return J


def oracle(eps, R, b_bar, W, z_max, top=None, allow=ALLOW, return_masks=False):
    """budget_sweep_lumos.oracle with tile weights W (actual coverage = oracle, expected coverage = link oracle),
    an optional buffer cap and an optional limit on the candidates per second; returns its metrics
    (or, with return_masks, the tiles it sends: (n, H, 64) including the forced ones)"""
    n = len(eps)
    A = [G["A_m"][p].astype(float).copy() for p, _ in eps]
    S = [ep_stall(A[e], R[e], z_max) for e in range(n)]
    b_left = n * b_bar - sum(np.sum(DISC * T[p]["mand"].sum(axis=1)) / NT for p, _ in eps)
    d_left = n * allow
    items = []
    for e, (p, _) in enumerate(eps):
        Wp = W[p]
        for t in range(H):
            cand = np.where(Wp[t] > 0)[0]
            if top is not None and len(cand) > top:
                cand = cand[np.argsort(-Wp[t][cand], kind="stable")[:top]]
            items.extend((Wp[t, i], e, t, i) for i in cand)
    items.sort(key=lambda x: -x[0])
    chosen = np.zeros((n, H, NT), bool)
    for pass_ in (0, 1):
        for _, e, t, i in items:
            cost = DISC[t] / NT
            if chosen[e, t, i] or cost > b_left:
                continue
            A[e][t] += v.delta[t][i]
            s_new = ep_stall(A[e], R[e], z_max)
            ds_ = s_new - S[e]
            if ds_ <= 1e-12 or (pass_ == 1 and ds_ <= d_left):
                chosen[e, t, i] = True
                b_left -= cost
                d_left -= max(ds_, 0.0)
                S[e] = s_new
            else:
                A[e][t] -= v.delta[t][i]
    X = np.stack([chosen[e] | T[p]["mand"] for e, (p, _) in enumerate(eps)])
    if return_masks:
        return X
    Ab = v.A_base[None, :] + (v.delta[None] * X).sum(axis=2)
    Q = np.stack([psnr_rows(p, X[e]) for e, (p, _) in enumerate(eps)])
    return metrics(Q, stall(Ab, np.stack(R), z_max), X.sum(axis=2))


def channel_eps(kw, seed, viewers, n_win, round_robin=False):
    """episodes (viewer, key) and their rates (n, H) in bits/s from one long synthetic series"""
    r = cv.rates(**{**cv.BASE, **kw}, n=n_win * (H + NH), seed=seed) * 1e6
    wins = [r[k * (H + NH):(k + 1) * (H + NH)] for k in range(n_win)]
    if round_robin:
        eps = [(viewers[k % len(viewers)], k) for k in range(n_win)]
    else:
        eps = [(p, k) for k in range(n_win) for p in viewers]
    return eps, [wins[k][NH:] for _, k in eps], r


def evaluate(name, eps_test, R_test, eps_train, R_train, link_stats):
    rows = []
    for cap, z_max in CAPS.items():
        P_test, P_train = [p for p, _ in eps_test], [p for p, _ in eps_train]
        hs = heuristics(P_test, np.stack(R_test), z_max)
        floor_test = hs["H1 viewport only"]["J_D"]
        floor_train = heuristics(P_train, np.stack(R_train), z_max)["H1 viewport only"]["J_D"]
        base = dict(channel=name, buffer=cap, floor_train=floor_train, floor_test=floor_test, **link_stats)
        for b_bar in B_BARS:
            for m, r in hs.items():
                rows.append(dict(base, b_bar=b_bar, method=m, **r))
            rows.append(dict(base, b_bar=b_bar, method="oracle", **oracle(eps_test, R_test, b_bar, G["EO"], z_max)))
            rows.append(dict(base, b_bar=b_bar, method="link oracle", **oracle(eps_test, R_test, b_bar, G["EM"], z_max, TOP_LINK)))
    df = pd.DataFrame(rows)
    df["avoidable_stall"] = df.J_D - df.floor_test
    df["within"] = (df.avoidable_stall <= ALLOW + 1e-9) & (df.J_B <= df.b_bar + 1e-9)
    return df


def main():
    t0 = time.time()
    # check: Lumos5G test runs, no cap, must match the meeting replay
    eps_l = G["eps"]["test"]
    R_l = [G["rates"][w][NH:] for _, w in eps_l]
    hs = heuristics([p for p, _ in eps_l], np.stack(R_l), None, h4_unit="base")
    orc = oracle(eps_l, R_l, 11.0, G["EO"], None)
    ref = pd.read_csv("WEEKLY_PROGRESS_2026-10-09_meeting/data/summary_test.csv").set_index("method")
    pairs = [("H1 viewport only", "predicted viewport only"), ("H2 QER 120x60", "QER 120x60"), ("H2 QER 160x80", "QER 160x80"),
             ("H3 average-based", "average-based (our error data)"), ("H4 ASL360 threshold", "ASL360 threshold (adapted)")]
    diff = max(max(abs(hs[a]["psnr"] - ref.loc[b, "psnr"]), abs(hs[a]["J_D"] - ref.loc[b, "J_D"]), abs(hs[a]["J_B"] - ref.loc[b, "J_B"]))
               for a, b in pairs)
    o = ref.loc["oracle (actual viewport + future link)"]
    diff_o = max(abs(orc["psnr"] - o.psnr), abs(orc["J_D"] - o.J_D), abs(orc["J_B"] - o.J_B))
    print(f"check vs the meeting replay (Lumos5G test, no cap): heuristics max diff {diff:.1e}, oracle max diff {diff_o:.1e}")
    assert diff < 1e-9 and diff_o < 1e-9, "fast replay does not reproduce the meeting numbers"

    viewers_test, viewers_train = list(G["held"]), list(G["train"])
    frames = []
    for name, kw in CHANNELS:
        eps_t, R_t, r_all = channel_eps(kw, 11, viewers_test, N_WIN)
        eps_tr, R_tr, _ = channel_eps(kw, 12, viewers_train, N_WIN, round_robin=True)
        frames.append(evaluate(name, eps_t, R_t, eps_tr, R_tr, cv.stats(r_all / 1e6)))
        print(f"{name}: done ({time.time() - t0:.0f} s)", flush=True)
    lum = np.concatenate([G["rates"][w][NH:] for _, w in eps_l]) / 1e6
    eps_trl = G["eps"]["train"][::40]
    frames.append(evaluate(LUMOS, eps_l, R_l, eps_trl, [G["rates"][w][NH:] for _, w in eps_trl], cv.stats(lum)))
    print(f"{LUMOS}: done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{OUT}/replay_channels.csv", index=False)
    summarize_and_plot(df)


def summarize_and_plot(df):
    names = [c for c, _ in CHANNELS] + [LUMOS]
    rows = []
    for (ch, cap, b), d in df.groupby(["channel", "buffer", "b_bar"], sort=False):
        d = d.set_index("method")
        h = d.loc[HEUR]
        ok = h[h.within]
        best = ok.psnr.idxmax() if len(ok) else None
        rows.append(dict(channel=ch, buffer=cap, b_bar=b, below_need_pct=d.pct_below_forced_need.iloc[0],
                         floor_test=d.floor_test.iloc[0], best_heuristic=best, best_psnr=ok.psnr.max() if len(ok) else np.nan,
                         heuristics_within=len(ok), oracle=d.loc["oracle", "psnr"], link_oracle=d.loc["link oracle", "psnr"],
                         gap_oracle=d.loc["oracle", "psnr"] - (ok.psnr.max() if len(ok) else np.nan),
                         gap_link_oracle=d.loc["link oracle", "psnr"] - (ok.psnr.max() if len(ok) else np.nan)))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/replay_channels_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 46)
    print(s[s.b_bar == 11.0].round(2).to_string(index=False))
    print(s[s.b_bar == 8.0].round(2).to_string(index=False))

    # fig 4: every method's quality on every channel (tile budget 11), no cap vs cap
    col = {"H1 viewport only": "#7f7f7f", "H2 QER 120x60": "#e377c2", "H2 QER 160x80": "#8c564b", "H3 average-based": "#d62728",
           "H4 ASL360 threshold": "#9467bd", "link oracle": "#525252", "oracle": "#000000"}
    mk = {m: "o" for m in HEUR} | {"link oracle": "s", "oracle": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, cap in zip(axes, CAPS):
        for yi, ch in zip(y, names):
            d = df[(df.channel == ch) & (df.buffer == cap) & (df.b_bar == 11.0)].set_index("method")
            for m in HEUR + ["link oracle", "oracle"]:
                ok = bool(d.loc[m, "within"])
                ax.plot(d.loc[m, "psnr"], yi, marker=mk[m], ms=8, color=col[m], mfc=col[m] if ok else "white", mew=1.8, ls="none")
            sr = s[(s.channel == ch) & (s.buffer == cap) & (s.b_bar == 11.0)].iloc[0]
            txt = f"gap {sr.gap_oracle:.1f} / {sr.gap_link_oracle:.1f} dB" if sr.best_heuristic else "no heuristic within"
            ax.text(56.2, yi, txt, va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(51.3, 57.8)
        ax.set_xlabel("viewport PSNR (dB)")
        ax.set_title(f"buffer {cap}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    handles = [Line2D([], [], marker=mk[m], color=col[m], ls="none", ms=8, label=m) for m in HEUR + ["link oracle", "oracle"]]
    handles += [Line2D([], [], marker="o", color="#555555", mfc="white", ls="none", ms=8, label="hollow = breaks a budget")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Heuristics vs oracle on each channel (stall allowance 0.1, tile budget 11). "
                 "Text: oracle / link-oracle gain over the best heuristic within budgets", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(f"{OUT}/fig4_heuristics_vs_oracle_by_channel.png", dpi=150)
    plt.close(fig)

    # fig 5: the gaps only
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, b in zip(axes, B_BARS):
        for k, (cap, c) in enumerate(zip(CAPS, ("#1f77b4", "#ff7f0e"))):
            d = s[(s.buffer == cap) & (s.b_bar == b)].set_index("channel").loc[names]
            ax.barh(y + (0.2 if k == 0 else -0.2), d.gap_link_oracle, 0.38, color=c, label=f"link oracle, {cap}")
            ax.barh(y + (0.2 if k == 0 else -0.2), d.gap_oracle, 0.38, color="none", edgecolor=c, lw=1.2, label=f"oracle, {cap}")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("dB above the best heuristic within budgets")
        ax.set_title(f"tile budget {b:.0f}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Room above the best heuristic within budgets: filled = link oracle (future link, predicted viewport), "
                 "outline = oracle (also the actual viewport)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(f"{OUT}/fig5_gap_by_channel.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
