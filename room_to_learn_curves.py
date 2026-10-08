"""
room_to_learn_curves.py - coverage bought per unit of tile budget, held-out
traces (Experiment X setting: power free, 50 mW for every rule/oracle).
Each rule family is swept over its own knob, so it traces a curve; the
learned Experiment X policies are points. Hollow markers = the point
breaks the stall budget (J_D > 0.3).

Usage: .venv/Scripts/python.exe room_to_learn_curves.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import plot_exp3_full_panels as p3
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv

OUT = rl.OUT_DIR
H, NT = rl.H, rl.NT


def rule_masks(bundle, kind, param, held, mean_base):
    dec = rb.make_policy(kind, param, "p50", mean_base)
    out = []
    for p, s, _ in held:
        env = TileStreamingEnv(trace_indices=[p], bundle=bundle); env.reset(seed=s); ms = []
        for _ in range(H):
            prop, _ = dec(env); ms.append(prop)
            env.step(np.concatenate([prop.astype(np.int64), [4]]))
        out.append(np.array(ms))
    return out


def point(bundle, held, masks):
    sims = [rl.simulate(bundle, p, s, m) for (p, s, _), m in zip(held, masks)]
    return {k: float(np.mean([x[k] for x in sims])) for k in ["J_Q", "J_D", "J_B", "cov", "psnr"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle = data_loader.load_training_video_bundle()
    mean_base = np.mean([layer_model.compute_A_t(bundle["rd_data"], config.TRAINING_VIDEO_ARRAY_INDEX, t, np.zeros(NT, bool),
                                                 enhanced_level=config.INITIAL_ENHANCEMENT_LEVELS[1])["A_base"] for t in range(H)])
    held = [(p, s, rl.trajectory(bundle, p, s)) for p, s in zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS)]
    train = [rl.trajectory(bundle, p, 2000) for p in rb.TRAIN_POS]
    eth = np.concatenate([rl.wsigned(t["at"] - t["pt"])[1:] for t in train])
    eph = np.concatenate([(t["ap"] - t["pp"])[1:] for t in train])
    mot = np.concatenate([rl.motion(t)[1:] for t in train])
    edges = np.quantile(mot, [0.25, 0.5, 0.75])

    def expected(tr, conditional):
        mand = rl.weights(tr["pt"], tr["pp"]) > 0
        mb = np.digitize(rl.motion(tr), edges)
        E = np.zeros((H, NT))
        for t in range(H):
            sel = (np.digitize(mot, edges) == mb[t]) if conditional else np.ones(len(eth), bool)
            E[t] = rl.weights(tr["pt"][t] + eth[sel], tr["pp"][t] + eph[sel]).mean(axis=0)
        return np.where(mand, -1.0, E)

    rows = []
    s_fit = (30.73, 3.34)
    for lam in np.linspace(0, 1.0, 11):
        r = point(bundle, held, rule_masks(bundle, "average", (lam * s_fit[0], lam * s_fit[1]), held, mean_base))
        rows.append(dict(family="Average-based (fixed margin)", knob=lam, **r))
    for w, h in rb.QER_SIZES:
        r = point(bundle, held, rule_masks(bundle, "qer", (w, h), held, mean_base))
        rows.append(dict(family="QER", knob=w, **r))
    for conditional, fam in [(False, "Expected-coverage rule (causal)"), (True, "Expected-coverage rule + motion (causal)")]:
        Es = [expected(tr, conditional) for _, _, tr in held]
        for tau in np.geomspace(0.003, 0.2, 16):
            r = point(bundle, held, [E >= tau for E in Es])
            rows.append(dict(family=fam, knob=tau, **r))
    # oracle sweep over pooled extra-tile budget (knows the actual viewport)
    items, base_cost = [], 0.0
    for e, (_, _, tr) in enumerate(held):
        wa = rl.weights(tr["at"], tr["ap"]); mand = rl.weights(tr["pt"], tr["pp"]) > 0
        base_cost += np.sum(rl.DISC * mand.sum(axis=1)) / NT
        items += [(wa[t, i], e, t, i) for t in range(H) for i in np.where(~mand[t] & (wa[t] > 0))[0]]
    items.sort(key=lambda x: -x[0])
    for target_jb in np.linspace(6.8, 9.0, 12):
        budget = len(held) * target_jb - base_cost
        masks = [np.zeros((H, NT), bool) for _ in held]
        for w_, e, t, i in items:
            c = rl.DISC[t] / NT
            if budget < c:
                break
            masks[e][t, i] = True; budget -= c
        rows.append(dict(family="Oracle (knows actual viewport)", knob=target_jb, **point(bundle, held, masks)))
    df = pd.DataFrame(rows)
    df["stall_ok"] = df["J_D"] <= rl.D_BAR
    df.to_csv(f"{OUT}/coverage_vs_tile_budget_curves.csv", index=False)

    learned = pd.read_csv("EXPERIMENT_X_drop_seed6/nonrl_baselines/comparison_heldout_with_learned.csv", index_col=0)
    learned = learned[learned.index.str.startswith("LEARNED")]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    style = {"Oracle (knows actual viewport)": ("#222222", "--"),
             "Expected-coverage rule + motion (causal)": ("#7b3fa0", "-"),
             "Expected-coverage rule (causal)": ("#b48fd0", "-"),
             "Average-based (fixed margin)": ("#8c6d46", "-"),
             "QER": ("#e07b39", "")}
    for fam, (col, ls) in style.items():
        d = df[df.family == fam].sort_values("J_B")
        if ls:
            ax.plot(d["J_B"], d["cov"], ls, color=col, lw=1.8, label=fam, zorder=2)
        else:
            ax.plot([], [], "s", color=col, label=fam)
        for _, r in d.iterrows():
            ax.scatter(r["J_B"], r["cov"], s=34 if fam != "QER" else 60, marker="s" if fam == "QER" else "o",
                       facecolor=col if r["stall_ok"] else "white", edgecolor=col, linewidth=1.4, zorder=3)
    for name, col in [("PPO", "#3b6fa0"), ("New PDS", "#2f9e44"), ("VE v2", "#c99a1e")]:
        r = learned[learned.index.str.contains(name)].iloc[0]
        ax.scatter(r["J_B"], r["mean_coverage"], s=140, marker="*", color=col, edgecolor=p3.INK, linewidth=0.6, zorder=5,
                   label=f"{name} (learned, no viewing direction)")
    ax.axvline(rl.B_BAR, color="#e34948", ls=":", lw=1.2, label="tile budget 8")
    ax.set_xlim(6.4, 9.2); ax.set_ylim(0.82, 1.005)
    ax.set_xlabel("tile cost J_B (tiles spent)"); ax.set_ylabel("viewport coverage (held-out)")
    ax.set_title("Experiment X setting, held-out traces: coverage bought per tile spent\n"
                 "filled = within stall budget, hollow = breaks stall budget (all rules/oracle at 50 mW)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/coverage_vs_tile_budget.png", dpi=150); plt.close(fig)

    # coverage each curve reaches at the learned methods' tile spend, within the stall budget
    for name in ["PPO", "New PDS"]:
        r = learned[learned.index.str.contains(name)].iloc[0]
        print(f"\nAt {name}'s tile spend J_B={r['J_B']:.2f} ({name} coverage {r['mean_coverage']:.4f}):")
        for fam in style:
            d = df[(df.family == fam) & (df.J_B <= r["J_B"] + 1e-9) & df.stall_ok]
            if len(d):
                b = d.loc[d["cov"].idxmax()]
                print(f"  {fam:42s} best feasible coverage {b['cov']:.4f} (J_B {b['J_B']:.2f}, J_D {b['J_D']:.3f}, PSNR {b['psnr']:.2f})")
    print(f"\nsaved {OUT}/coverage_vs_tile_budget.png and .csv")


if __name__ == "__main__":
    main()
