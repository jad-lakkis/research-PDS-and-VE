"""
build_meeting_x.py - meeting pack: PPO vs New PDS vs VE v2 (seeds 1,2,3,5,7,
all methods on the same matched rollouts). Every figure is drawn here with
short, presentation wording (no experiment/seed labels in titles; legends
show n), and README.md carries every number recomputed from the current
progress logs.

Usage: .venv/Scripts/python.exe build_meeting_x.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_exp3_full_panels as p3
import plot_meeting_followup_exp3 as pmf

SEEDS = [1, 2, 3, 5, 7]
METHODS = ["PPO", "New PDS", "VE v2"]
COLORS = {"PPO": "#3b6fa0", "New PDS": "#2f9e44", "VE v2": "#c99a1e"}
OUT = "MEETING_EXPERIMENT_X"
TAIL = 1000
STEPS_PER_ROLLOUT = 1728
D_BAR, B_BAR = p3.D_BAR, p3.B_BAR
RED = "#e34948"
K = 50  # path smoothing for figures 10 and 18


NAME = {"PPO": "PPO", "New PDS": "PDS", "VE v2": "PDS+VE"}  # presentation names


def display(text):
    return text.replace("New PDS", NAME["New PDS"]).replace("VE v2", NAME["VE v2"])


def lab(m):
    return f"{NAME[m]} (n={len(SEEDS)})"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)


def bar_labels(ax, xs, vals, fmt):
    for x, v in zip(xs, vals):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=pmf.INK)


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(f"{OUT}/*.png"):
        os.remove(f)
    full = {m: [p3.load_progress(p3.PATHS[m].format(s=s)) for s in SEEDS] for m in METHODS}
    n = min(len(d) for r in full.values() for d in r)
    data = {m: [d.iloc[:n] for d in full[m]] for m in METHODS}
    late = {m: [d.iloc[-TAIL:] for d in data[m]] for m in METHODS}
    x_roll = np.arange(1, n + 1)
    sec = []

    # ---------- 01 violation ----------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in METHODS:
        pmf.mean_std_band(ax, x_roll, np.stack([d["eval/violation"].values for d in data[m]]), COLORS[m], lab(m))
    ax.text(0.5, 0.93, r"violation $= \max(0,\ J_D-\bar D) + \max(0,\ J_B-\bar B)$,   $\bar D = 0.3$,  $\bar B = 8$",
            transform=ax.transAxes, ha="center", fontsize=11, color=pmf.INK)
    ax.set_title("Constraint violation (lower = better)")
    ax.set_xlabel("rollout"); ax.set_ylabel("violation"); ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "01_violation.png")

    # ---------- 01b signed distance from the budgets (no max), 01c per constraint ----------
    sgn = {m: [(d["eval/J_D"] - D_BAR + d["eval/J_B"] - B_BAR).values for d in data[m]] for m in METHODS}
    LATE_FROM = 2000
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, late_zoom in [(axes[0], False), (axes[1], True)]:
        sl = slice(LATE_FROM - 1, n) if late_zoom else slice(0, n)
        for m in METHODS:
            pmf.mean_std_band(ax, x_roll[sl], np.stack(sgn[m])[:, sl], COLORS[m], lab(m))
        ax.axhline(0, color=RED, ls="--", lw=1.0, label="at the budgets")
        ax.set_xlabel("rollout"); ax.set_ylabel("signed distance")
        if late_zoom:
            ax.set_xlim(LATE_FROM, n); ax.set_ylim(-1.15, 0.45)
            ax.set_title(f"Zoom: rollout {LATE_FROM} onwards")
        else:
            ax.set_ylim(-1.2, 0.8); ax.set_title("Zoom around 0")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(r"Signed distance from the budgets: $(J_D-\bar D) + (J_B-\bar B)$  (below 0 = inside, more negative = further inside)")
    save(fig, "01b_signed_distance.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, bar, nm, lim in [(axes[0], "eval/J_D", D_BAR, r"Stall: $J_D-\bar D$", (-0.35, 0.3)),
                                  (axes[1], "eval/J_B", B_BAR, r"Tiles: $J_B-\bar B$", (-1.2, 0.8))]:
        for m in METHODS:
            pmf.mean_std_band(ax, x_roll, np.stack([(d[col] - bar).values for d in data[m]]), COLORS[m], lab(m))
        ax.axhline(0, color=RED, ls="--", lw=1.0, label="at the budget")
        ax.set_ylim(*lim); ax.set_xlabel("rollout"); ax.set_ylabel("signed distance"); ax.set_title(nm)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Signed distance from each budget, zoomed around 0 (below 0 = inside the budget)")
    save(fig, "01c_signed_distance_by_constraint.png")
    rows = []
    for m in METHODS:
        vD = [t["eval/J_D"].mean() - D_BAR for t in late[m]]
        vB = [t["eval/J_B"].mean() - B_BAR for t in late[m]]
        rows.append({"method": m, "stall J_D - 0.3": np.mean(vD), "tiles J_B - 8": np.mean(vB),
                     "sum": np.mean(vD) + np.mean(vB), "sum seed std": np.std(np.array(vD) + np.array(vB))})
    sec.append(("Figures 01b-01c: signed distance from the budgets (no max)",
                f"Last {TAIL} rollouts; negative = inside the budget (more negative = more room left).",
                md(pd.DataFrame(rows))))

    # ---------- 02 / 03 speed to thresholds ----------
    thr = [("viol < 5", 5.0), ("viol < 1", 1.0), ("viol < 0.2", 0.2), ("viol = 0", 1e-6)]
    first = {m: {t: [p3.first_rollout_below(d, "eval/violation", v, sustained=1) for d in full[m]] for t, v in thr} for m in METHODS}
    xb = np.arange(len(thr)); width = 0.8 / len(METHODS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, m in enumerate(METHODS):
        vals = [np.mean(first[m][t]) * STEPS_PER_ROLLOUT / 1e6 for t, _ in thr]
        xs = xb + (j - 1) * width
        ax.bar(xs, vals, width, color=COLORS[m], alpha=0.85, label=lab(m))
        bar_labels(ax, xs, vals, "{:.2f}")
    ax.set_xticks(xb); ax.set_xticklabels([t for t, _ in thr])
    ax.set_ylabel("environment steps (millions)"); ax.set_title("lower = better")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "02_speed_to_threshold.png")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, m in enumerate(METHODS):
        vals = [np.mean(first["PPO"][t]) / np.mean(first[m][t]) for t, _ in thr]
        xs = xb + (j - 1) * width
        ax.bar(xs, vals, width, color=COLORS[m], alpha=0.85, label=lab(m))
        bar_labels(ax, xs, vals, "{:.2f}x")
    ax.axhline(1.0, color=RED, ls="--", lw=1.0)
    ax.set_xticks(xb); ax.set_xticklabels([t for t, _ in thr])
    ax.set_ylabel("speed-up vs PPO"); ax.set_title("higher = better")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "03_speedup_vs_ppo.png")

    rows = []
    for t, _ in thr:
        r = {"threshold": t}
        for m in METHODS:
            v = first[m][t]
            r[f"{m} rollouts (mean, range)"] = f"{np.mean(v):.0f} ({min(v)}-{max(v)})"
            r[f"{m} env steps (M)"] = f"{np.mean(v) * STEPS_PER_ROLLOUT / 1e6:.2f}"
        for m in METHODS[1:]:
            r[f"{m} speed-up"] = f"{np.mean(first['PPO'][t]) / np.mean(first[m][t]):.2f}x"
        rows.append(r)
    no_ov = {m: max(first[m]["viol < 1"]) < min(first["PPO"]["viol < 1"]) for m in METHODS[1:]}
    sec.append(("Figures 02-03: speed to violation thresholds",
                f"First single held-out evaluation below each threshold, mean over seeds (range in brackets). 1 rollout = {STEPS_PER_ROLLOUT} steps. "
                + "; ".join(f"every {m} seed reaches viol < 1 before the fastest PPO seed: {'yes' if v else 'no'}" for m, v in no_ov.items()),
                md(pd.DataFrame(rows))))

    # ---------- 04 cumulative violation ----------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in METHODS:
        pmf.mean_std_band(ax, x_roll, np.stack([d["eval/violation"].cumsum().values for d in data[m]]), COLORS[m], lab(m))
    ax.set_title("Total violation accumulated during training (lower = better)")
    ax.set_xlabel("rollout"); ax.set_ylabel("cumulative violation"); ax.legend(frameon=False, fontsize=8)
    save(fig, "04_cumulative_violation.png")
    cum = {m: [d["eval/violation"].sum() for d in data[m]] for m in METHODS}
    sec.append(("Figure 04: total violation accumulated during training", "",
                md(pd.DataFrame([{"method": m, "total violation (mean)": np.mean(cum[m]), "seed std": np.std(cum[m]),
                                  "PPO / method": np.mean(cum["PPO"]) / np.mean(cum[m])} for m in METHODS]), "{:.2f}")))

    # ---------- 05 early checkpoints (linear, no seed dots) ----------
    cps = [250, 500, 1000, 1500]
    early = {m: [np.mean([d["eval/violation"].values[c - 50:c].mean() for d in data[m]]) for c in cps] for m in METHODS}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xc = np.arange(len(cps))
    for j, m in enumerate(METHODS):
        xs = xc + (j - 1) * width
        ax.bar(xs, early[m], width, color=COLORS[m], alpha=0.85, label=lab(m))
        bar_labels(ax, xs, early[m], "{:.2f}")
    ax.set_xticks(xc); ax.set_xticklabels([f"rollout {c}" for c in cps])
    ax.set_ylabel("violation"); ax.set_title("Violation early in training (lower = better)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "05_violation_early_checkpoints.png")
    t3 = pd.DataFrame([{"rollout": c, **{m: early[m][i] for m in METHODS},
                        **{f"PPO / {m}": early["PPO"][i] / early[m][i] for m in METHODS[1:]}} for i, c in enumerate(cps)])
    sec.append(("Figure 05: violation early in training", "Mean over the 50 rollouts ending at each checkpoint.", md(t3)))

    # ---------- 06 every seed within 20% of each budget (no seed dots) ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    rows = []
    for ax, cname, col, bar in [(axes[0], "Stall", "eval/J_D", D_BAR), (axes[1], "Tiles", "eval/J_B", B_BAR)]:
        th = bar * 1.2
        vals = {}
        for m in METHODS:
            ok = np.stack([d[col].values <= th for d in data[m]])
            vals[m] = [max(int(np.argmax(r)) + 1 for r in ok), int(np.argmax(ok.all(axis=0))) + 1]
            rows.append({"constraint": f"{cname.lower()} (<= {th:.2f})", "method": m,
                         "every seed reached once": vals[m][0], "all seeds at once": vals[m][1]})
        xg = np.arange(2)
        for j, m in enumerate(METHODS):
            xs = xg + (j - 1) * width
            ax.bar(xs, vals[m], width, color=COLORS[m], alpha=0.85, label=lab(m))
            for x_, v_, pv in zip(xs, vals[m], vals["PPO"]):
                txt = f"{v_}" if m == "PPO" else f"{v_}\n{pv / v_:.2f}x"
                ax.annotate(txt, (x_, v_), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=pmf.INK)
        ax.set_xticks(xg); ax.set_xticklabels(["every seed reached once", "all seeds at once"])
        ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.25)
        ax.set_ylabel("rollout"); ax.set_title(f"{cname}: within 20% of budget")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("When every seed gets within 20% of each budget (lower = better)")
    save(fig, "06_all_seeds_reach_budget.png")
    sec.append(("Figure 06: when every seed gets within 20% of each budget", "First arrival, each constraint on its own.", md(pd.DataFrame(rows))))

    # ---------- 08 multipliers ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    rows = []
    for ax, col, nm in [(axes[0], "lagrangian/mu_D", "Stall multiplier"), (axes[1], "lagrangian/mu_B", "Tile multiplier")]:
        for m in METHODS:
            lm = [t[col].mean() for t in late[m]]
            pmf.mean_std_band(ax, x_roll, np.stack([pmf.smooth(d[col]).values for d in data[m]]), COLORS[m],
                              f"{lab(m)}, seed std {np.std(lm):.2f}")
            rows.append({"multiplier": nm, "method": m, "peak (mean over seeds)": np.mean([d[col].max() for d in data[m]]),
                         "late mean": np.mean(lm), "across-seed std (late)": np.std(lm),
                         "within-run std (late)": np.mean([t[col].std() for t in late[m]])})
        ax.set_title(nm); ax.set_xlabel("rollout"); ax.set_ylabel("multiplier"); ax.legend(frameon=False, fontsize=8)
    save(fig, "08_multiplier_stability.png")
    sec.append(("Figure 08: Lagrange multipliers", f"Late = last {TAIL} rollouts.", md(pd.DataFrame(rows))))

    # ---------- 09a costs vs budgets over training, 09b budget used ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    for ax, col, bar, nm in [(axes[0], "eval/J_D", D_BAR, "Stall cost J_D"), (axes[1], "eval/J_B", B_BAR, "Tile cost J_B")]:
        for m in METHODS:
            pmf.mean_std_band(ax, x_roll, np.stack([pmf.smooth(d[col]).values for d in data[m]]), COLORS[m], lab(m))
        ax.axhline(bar, color=RED, ls="--", lw=1.0, label=f"budget {bar}")
        ax.set_title(nm); ax.set_xlabel("rollout"); ax.set_ylabel(nm.split()[-1]); ax.legend(frameon=False, fontsize=8)
    save(fig, "09a_costs_vs_budgets.png")
    used = {m: {c: [t[col].mean() / bar * 100 for t in late[m]] for c, col, bar in [("stall", "eval/J_D", D_BAR), ("tiles", "eval/J_B", B_BAR)]}
            for m in METHODS}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, c, nm in [(axes[0], "stall", "Stall budget used"), (axes[1], "tiles", "Tile budget used")]:
        vals = [np.mean(used[m][c]) for m in METHODS]
        xs = np.arange(len(METHODS))
        ax.bar(xs, vals, 0.6, color=[COLORS[m] for m in METHODS], alpha=0.85)
        for x_, v_ in zip(xs, vals):
            ax.annotate(f"{v_:.1f}%", (x_, v_), xytext=(0, -16), textcoords="offset points", ha="center",
                        fontsize=10, color="white", fontweight="bold")
        ax.axhline(100, color=RED, ls="--", lw=1.0)
        ax.set_xticks(xs); ax.set_xticklabels([lab(m) for m in METHODS])
        ax.set_ylim(0, 115); ax.set_ylabel("% of budget"); ax.set_title(nm)
    fig.suptitle(f"Share of each budget used, last {TAIL} rollouts")
    save(fig, "09b_budget_used.png")
    sec.append(("Figures 09a-09b: costs and budget used", f"Last {TAIL} rollouts.",
                md(pd.DataFrame([{"method": m, "stall budget used %": np.mean(used[m]["stall"]),
                                  "per seed": str([int(round(v)) for v in used[m]["stall"]]),
                                  "tile budget used %": np.mean(used[m]["tiles"]),
                                  "tile per seed": str([int(round(v)) for v in used[m]["tiles"]])} for m in METHODS]), "{:.1f}")))

    # ---------- 10 quality vs violation path ----------
    def path(m, col):
        v = np.stack([d[col].values for d in data[m]]).mean(axis=0)
        return np.convolve(v, np.ones(K) / K, mode="valid")

    MARKS = [(250, "o", "rollout 250"), (1000, "s", "rollout 1000"), (n, "*", "end")]

    def marks(ax, xs, ys, m):
        for r, mk, _ in MARKS:
            i = min(max(r - K, 0), len(xs) - 1)
            ax.scatter(xs[i], ys[i], color=COLORS[m], marker=mk, s=90 if mk == "*" else 40,
                       edgecolor=pmf.INK, linewidth=0.7, zorder=4)

    def mark_legend(ax):
        for _, mk, t in MARKS:
            ax.scatter([], [], color="white", marker=mk, s=90 if mk == "*" else 40, edgecolor=pmf.INK, linewidth=0.7, label=t)

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        xv, yq = path(m, "eval/violation"), path(m, "eval/J_Q")
        ax.plot(xv, yq, color=COLORS[m], lw=2.0, label=lab(m)); marks(ax, xv, yq, m)
    ax.invert_xaxis()
    mark_legend(ax)
    ax.set_xlabel("violation"); ax.set_ylabel("quality J_Q"); ax.set_title("Quality vs violation during training")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "10_quality_vs_violation_path.png")
    rows = []
    for m in METHODS:
        row = {"method": m}
        for col, nm in [("eval/mean_coverage", "coverage"), ("eval/mean_psnr_db", "PSNR dB"), ("eval/J_Q", "J_Q")]:
            v = [t[col].mean() for t in late[m]]
            row[f"{nm} mean"] = np.mean(v); row[f"{nm} seed std"] = np.std(v)
        rows.append(row)
    sec.append(("Figure 10: quality", f"Last {TAIL} rollouts; mean over seeds and across-seed std.", md(pd.DataFrame(rows), "{:.4f}")))

    # ---------- 11 steadier by metric (linear) ----------
    specs = [("stall time", "eval/mean_stall_sec"), ("stall cost J_D", "eval/J_D"), ("tile cost J_B", "eval/J_B"),
             ("tiles", "physical/mean_n_enhanced_tiles"), ("quality J_Q", "eval/J_Q"), ("coverage", "eval/mean_coverage"),
             ("PSNR", "eval/mean_psnr_db"), ("tile multiplier", "lagrangian/mu_B"), ("stall multiplier", "lagrangian/mu_D")]
    sd = {nm: {m: np.std([t[col].mean() for t in late[m]]) for m in METHODS} for nm, col in specs}
    ratio = {nm: {m: sd[nm][m] / sd[nm]["PPO"] for m in METHODS[1:]} for nm, _ in specs}
    order = sorted([nm for nm, _ in specs], key=lambda nm: np.mean(list(ratio[nm].values())))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(order)); h = 0.38
    for j, m in enumerate(METHODS[1:]):
        ax.barh(y + (j - 0.5) * h, [ratio[nm][m] for nm in order], h, color=COLORS[m], label=lab(m))
    ax.axvline(1.0, color=RED, ls="--", lw=1.0, label="PPO")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("seed spread relative to PPO"); ax.set_title(f"Seed spread relative to PPO, last {TAIL} rollouts (left of the line = steadier)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "11_steadier_by_metric.png")
    t8 = pd.DataFrame([{"metric": nm, **{f"{m} seed std": sd[nm][m] for m in METHODS},
                        **{f"{m} / PPO": ratio[nm][m] for m in METHODS[1:]}} for nm, _ in specs])
    cnt = {m: int((t8[f"{m} / PPO"] < 1).sum()) for m in METHODS[1:]}
    sec.append(("Figure 11: seed spread by metric", "; ".join(f"{m} steadier than PPO on {c}/{len(specs)} metrics" for m, c in cnt.items()),
                md(t8, "{:.4f}")))

    # ---------- 12-14 convergence in probability ----------
    eps_list = [("ε = 0.2", 0.2), ("ε = 0.1", 0.1)]
    curves_by_m = {m: pmf.sliding_prob_exceed(data[m], "eval/violation", eps_list) for m in METHODS}
    fig, axes = plt.subplots(1, len(eps_list), figsize=(16, 5.2))
    rows = []
    for ax, (lb, e) in zip(axes, sorted(eps_list, key=lambda t: t[1])):
        for m in METHODS:
            xs, curves = curves_by_m[m]
            p = curves[lb]
            ax.plot(xs, p * 100, color=COLORS[m], lw=2.0, label=lab(m))
            below = p < 0.05
            fb = int(xs[np.argmax(below)]) if below.any() else None
            rows.append({"method": m, "eps": lb, "first n with P < 5%": fb if fb else "never",
                         "later windows still < 5%": f"{below[np.argmax(below):].mean() * 100:.0f}%" if fb else "-", "P at the end": p[-1]})
        ax.axhline(5, color=RED, ls="--", lw=1.2, label="5% target")
        ax.set_ylim(-2, 102); ax.set_xlabel("rollout")
        ax.set_ylabel(f"% of evaluations with violation > {e}")
        ax.set_title(f"Violation above {e} (ε = {e})")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Convergence in probability: how often is the violation above ε?  (last 200 rollouts, all seeds; below the red line = converged)")
    save(fig, "12_convergence_in_probability.png")
    sec.append(("Figures 12-14: convergence in probability", "P(violation > eps) over a sliding window of 200 rollouts x all seeds.",
                md(pd.DataFrame(rows))))
    for name, col, bar, eps_iso, sym in [
            ("13_convergence_stall.png", "eval/J_D", D_BAR, [0.0, 0.05, 0.1, 0.2], "J_D"),
            ("14_convergence_tiles.png", "eval/J_B", B_BAR, [0.0, 0.2, 0.5, 1.0], "J_B")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, m in zip(axes, METHODS):
            tl = [(f"ε = {e}", bar + e) for e in eps_iso]
            xs, curves = pmf.sliding_prob_exceed(data[m], col, tl)
            for lb, _ in tl:
                main = lb == "ε = 0.0"
                ax.plot(xs, curves[lb], color=COLORS[m], alpha=1.0 if main else 0.4, lw=2.2 if main else 1.2,
                        label="over budget" if main else lb)
            ax.axhline(0.05, color=RED, ls="--", lw=1.0, label="5%")
            ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout"); ax.set_title(lab(m)); ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel(f"P({sym} > {bar} + ε)")
        fig.suptitle(f"Convergence in probability, {'stall' if sym == 'J_D' else 'tile'} budget only (window of 200 rollouts)")
        save(fig, name)

    # ---------- 15a violation CDF, 15b unused tile budget (mean lines) ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in METHODS:
        v = pmf.pooled_tail(data[m], "eval/violation", TAIL)
        xs_ = np.sort(v); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
        ax.step(xs_, ys_, color=COLORS[m], lw=2.0, where="post", label=f"{lab(m)}, mean {v.mean():.3f}")
        ax.axvline(v.mean(), color=COLORS[m], ls="--", lw=1.4)
    ax.set_xlabel("violation"); ax.set_ylabel("cumulative probability")
    ax.set_title(f"Distribution of violation, last {TAIL} rollouts (dashed = mean)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "15a_violation_cdf.png")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in METHODS:
        dist = B_BAR - pmf.pooled_tail(data[m], "eval/J_B", TAIL)
        ax.hist(dist, bins=60, density=True, histtype="step", linewidth=2.0, color=COLORS[m], label=f"{lab(m)}, mean {dist.mean():.3f}")
        ax.axvline(dist.mean(), color=COLORS[m], ls="--", lw=1.4)
    ax.axvline(0, color=RED, lw=1.0, label="at the budget")
    ax.set_xlabel("unused tile budget (below 0 = over budget)"); ax.set_ylabel("density")
    ax.set_title(f"Distance from the tile budget, last {TAIL} rollouts (dashed = mean)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "15b_tile_budget_distance.png")

    # ---------- 16 / 17 distance from each budget ----------
    for name, col, bar, cname in [("16_distance_from_stall_budget.png", "eval/J_D", D_BAR, "stall"),
                                  ("17_distance_from_tile_budget.png", "eval/J_B", B_BAR, "tile")]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        flat = axes.ravel()
        for m in METHODS:
            over = np.maximum(0, pmf.pooled_tail(data[m], col, TAIL) - bar)
            xs_ = np.sort(over); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
            wide = m == "PPO"  # drawn wider underneath so it stays visible where curves coincide
            flat[0].step(xs_, ys_, color=COLORS[m], lw=5.0 if wide else 2.0, alpha=0.55 if wide else 1.0,
                         where="post", label=lab(m), zorder=2 if wide else 3)
        flat[0].set_xlabel(f"amount over the {cname} budget"); flat[0].set_ylabel("cumulative probability")
        flat[0].set_title("Over-budget amount"); flat[0].legend(frameon=False, fontsize=8)
        for ax, m in zip(flat[1:], METHODS):
            dist = bar - pmf.pooled_tail(data[m], col, TAIL)
            ax.hist(dist, bins=50, density=True, color=COLORS[m], alpha=0.75)
            ax.axvline(0, color=RED, ls="--", lw=1.0)
            ax.axvline(dist.mean(), color=pmf.INK, lw=1.5)
            ax.set_xlabel(f"unused {cname} budget"); ax.set_ylabel("density")
            ax.set_title(f"{lab(m)}, mean {dist.mean():.3f}")
        fig.suptitle(f"Distance from the {cname} budget, last {TAIL} rollouts (black = mean, red = budget)")
        save(fig, name)

    # ---------- 18 tiles vs coverage path ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        xt, yc = path(m, "eval/mean_n_tiles"), path(m, "eval/mean_coverage")
        ax.plot(xt, yc, color=COLORS[m], lw=2.0, label=lab(m)); marks(ax, xt, yc, m)
    mark_legend(ax)
    ax.set_xlabel("tiles per segment"); ax.set_ylabel("coverage"); ax.set_title("Tiles sent vs coverage during training")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "18_tiles_vs_coverage.png")
    sec.append(("Figure 18: tiles and power (last 1000 rollouts)", "",
                md(pd.DataFrame([{"method": m, **{nm: np.mean([t[c].mean() for t in late[m]]) for nm, c in
                                                  [("tiles per segment", "eval/mean_n_tiles"), ("coverage", "eval/mean_coverage"),
                                                   ("power mW", "eval/mean_power_mw"), ("stall s per segment", "eval/mean_stall_sec")]}}
                                 for m in METHODS]), "{:.4f}")))

    text = ("# Meeting pack: PPO vs New PDS vs VE v2\n\n"
            f"PDS = PPO with the post-decision-state critic (two-pass); PDS+VE = PDS with virtual experience (v2). "
            f"Seeds 1, 2, 3, 5, 7 (seed 6 excluded). All methods over the same first {n} rollouts unless stated; "
            f"held-out evaluation on 3 traces after every rollout. Budgets: stall {D_BAR}, tiles {B_BAR}; power unconstrained. "
            "violation = max(0, J_D - 0.3) + max(0, J_B - 8).\n\n")
    for title, note, table in sec:
        text += f"## {title}\n\n" + (note + "\n\n" if note else "") + table + "\n\n"
    text += "## Figures\n\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in sorted(glob.glob(f"{OUT}/*.png"))) + "\n"
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write(display(text))
    print(f"wrote {OUT}/ ({len(glob.glob(f'{OUT}/*.png'))} figures, README.md), matched n={n}")


if __name__ == "__main__":
    main()
