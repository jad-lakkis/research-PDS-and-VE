"""
plot_pds_graphs.py - regenerate the per-arm training graphs (13 panels) for
one PPO or PPO+PDS arm directly from its progress.csv (the full SB3-logged
file, not the older, narrower key_metrics.csv extract).

FIXED (2026-09-24): every column lookup here used to be a BARE name
("violation", "J_D", "mu_D", "coverage", "enhanced_tiles", "reward",
"mean_psnr_db", "pds_value_loss") left over from when this script only
read the old key_metrics.csv extract. progress.csv's real columns are all
namespaced ("eval/violation", "eval/J_D", "lagrangian/mu_D", ...), so
every one of those lookups silently failed `needed_cols.issubset(df.columns)`
and every panel printed "SKIP ... missing columns" - even though the data
was right there under a different name. This was invisible in normal runs
because the script never raises on an all-skipped file, it just quietly
writes nothing into an already-existing (possibly stale) out_dir. Fixed by
migrating every lookup to progress.csv's actual column names, and by
implementing the 4 panels that were genuinely never built at all (policy
entropy, training diagnostics, J_Q vs. its theoretical max, stall time) -
these needed real progress.csv columns that are now being read correctly.
Also: plot_physical_metrics was permanently labeled "(partial)" citing
missing stall/power columns that do exist in progress.csv - expanded to
the full 4-panel version. plot_violation_and_feasibility never actually
plotted feasibility despite its name - added.

Usage: python plot_pds_graphs.py <progress.csv> <d_bar> <p_bar> <b_bar> <out_dir> [dropped_constraints]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def load_progress(csv_path: str) -> pd.DataFrame:
    """progress.csv has no literal 'rollout' column - it's one row per
    rollout, implicit in row order - so it's added here, same convention
    every other multi-panel script in this project already uses
    (plot_official_exp1.py's load_progress). Also guards against SB3's
    CSVOutputFormat rewriting the whole file (fresh header row included)
    whenever a new logger key appears mid-training - read as str first,
    drop any embedded repeat of the header, then coerce numeric."""
    df = pd.read_csv(csv_path, dtype=str)
    df = df[df["eval/violation"] != "eval/violation"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/violation"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def plot_violation_and_feasibility(df: pd.DataFrame, out_dir: str,
                                   dropped_constraints: frozenset = frozenset()) -> None:
    terms = {"D": r"max(0, $J_D-\bar D$)", "P": r"max(0, $J_P-\bar P$)", "B": r"max(0, $J_B-\bar B$)"}
    formula = "violation = " + " + ".join(v for k, v in terms.items() if k not in dropped_constraints)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df["rollout"], df["eval/violation"], color="firebrick")
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set_title(formula)
    axes[0].set_ylabel("violation")
    axes[0].grid(True, alpha=0.3)

    feasible_pct = df["eval/feasible"].rolling(50, min_periods=10).mean() * 100
    axes[1].plot(df["rollout"], feasible_pct, color="seagreen")
    axes[1].set_title("Feasible % (rolling-50) - decision-log #10's primary metric")
    axes[1].set_xlabel("rollout")
    axes[1].set_ylabel("feasible %")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_violation_and_feasibility.png"))
    plt.close(fig)


def plot_constraint_returns_vs_budgets(df: pd.DataFrame, d_bar: float, p_bar: float, b_bar: float, out_dir: str,
                                       dropped_constraints: frozenset = frozenset()) -> None:
    """A dropped constraint gets NO subplot here at all - not just a
    hidden budget line. Drawing "budget = 6.5" against a J_P that was
    never actually constrained (mu_P pinned at 0 the whole run, per
    --drop-constraints) makes an unconstrained quantity look like a
    severe, permanent violation. This was shipped as a real bug in
    FINAL_PART1_EXP's drop-P runs before this fix.
    """
    specs = [("eval/J_D", d_bar, "tab:blue", r"$J_D$ (stall)"),
             ("eval/J_P", p_bar, "tab:orange", r"$J_P$ (power)"),
             ("eval/J_B", b_bar, "tab:green", r"$J_B$ (tiles)")]
    active = [(col, bud, color, ylabel) for col, bud, color, ylabel in specs
              if col[-1] not in dropped_constraints]

    fig, axes = plt.subplots(len(active), 1, figsize=(11, 3 * len(active)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.suptitle("Discounted constraint returns vs. their budgets")

    for ax, (col, bud, color, ylabel) in zip(axes, active):
        ax.plot(df["rollout"], df[col], color=color)
        ax.axhline(bud, color="red", ls="--", label=f"budget = {bud}")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("rollout")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_constraint_returns_vs_budgets.png"))
    plt.close(fig)


def plot_lagrange_multipliers(df: pd.DataFrame, d_bar: float, p_bar: float, b_bar: float, out_dir: str,
                              dropped_constraints: frozenset = frozenset()) -> None:
    """A dropped constraint's multiplier is excluded here too, not just
    shown as a flat 0 line - a viewer shouldn't have to infer "this is
    pinned, not just naturally non-binding" from a suspiciously flat
    curve. Same "no trace at all" treatment as the constraint-returns
    panel above.
    """
    specs = [("lagrangian/mu_D", "D", r"$\mu_D$ (stall)", "tab:blue"),
             ("lagrangian/mu_P", "P", r"$\mu_P$ (power)", "tab:orange"),
             ("lagrangian/mu_B", "B", r"$\mu_B$ (tiles)", "tab:green")]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, code, label, color in specs:
        if code in dropped_constraints:
            continue
        ax.plot(df["rollout"], df[col], label=label, color=color)
    bars = {"D": d_bar, "P": p_bar, "B": b_bar}
    active_bars = ", ".join(f"{k}-bar={v}" for k, v in bars.items() if k not in dropped_constraints)
    ax.set_title(f"Lagrange multiplier trajectories - {active_bars}")
    ax.set_xlabel("rollout")
    ax.set_ylabel("multiplier value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_lagrange_multipliers.png"))
    plt.close(fig)


def plot_physical_metrics(df: pd.DataFrame, out_dir: str) -> None:
    """Full 4-panel version - coverage, enhanced tiles, stall time, and
    transmit power are all real progress.csv columns; nothing here is
    partial anymore (see module docstring)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Physical metrics over training ({len(df)} rollouts)")

    axes[0, 0].plot(df["rollout"], df["eval/mean_coverage"], color="tab:purple")
    axes[0, 0].set_title("Coverage (fraction)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df["rollout"], df["physical/mean_n_enhanced_tiles"], color="olive")
    axes[0, 1].set_title("Enhanced tiles (of 64)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["rollout"], df["eval/mean_stall_sec"], color="crimson")
    axes[1, 0].set_title("Stall time (sec)")
    axes[1, 0].set_xlabel("rollout")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["rollout"], df["physical/mean_power_mW"], color="darkcyan")
    axes[1, 1].set_title("Transmit power (mW)")
    axes[1, 1].set_xlabel("rollout")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_physical_metrics.png"))
    plt.close(fig)


def plot_reward(df: pd.DataFrame, out_dir: str, dropped_constraints: frozenset = frozenset()) -> None:
    # The formula printed in the title should match what the run actually
    # computed - a dropped term is never in the reward, not just always
    # zero, so it's dropped from the displayed formula too.
    terms = {"D": r"- \mu_D cost_D^t", "P": r"- \mu_P cost_P^t", "B": r"- \mu_B cost_B^t"}
    formula = r"$r^t = \beta_Q C_{cov}^t " + " ".join(v for k, v in terms.items() if k not in dropped_constraints) + "$"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["rollout/ep_rew_mean"], color="teal")
    ax.set_title(formula)
    ax.set_xlabel("rollout")
    ax.set_ylabel("mean episode reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_reward.png"))
    plt.close(fig)


def plot_coverage_vs_enhanced_tiles(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    half = len(df) // 2
    ax.scatter(df["physical/mean_n_enhanced_tiles"].iloc[:half], df["eval/mean_coverage"].iloc[:half],
               s=10, color="firebrick", alpha=0.6, label=f"rollout 1-{half}")
    ax.scatter(df["physical/mean_n_enhanced_tiles"].iloc[half:], df["eval/mean_coverage"].iloc[half:],
               s=10, color="tab:blue", alpha=0.6, label=f"rollout {half}-{len(df)}")
    ax.set_title("More enhanced tiles came with higher coverage")
    ax.set_xlabel("enhanced tiles (of 64)")
    ax.set_ylabel("viewport coverage (fraction)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Each dot = one rollout's checkpoint during training (not separate experiments)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_coverage_vs_enhanced_tiles.png"))
    plt.close(fig)


def plot_extra_tiles_approx(df: pd.DataFrame, out_dir: str, mandatory_avg: float = 12.52) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    extra = df["physical/mean_n_enhanced_tiles"] - mandatory_avg
    ax.plot(df["rollout"], extra, color="darkviolet")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_title(f"Discretionary extra tiles only ~= total enhanced - {mandatory_avg} (mandatory average)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("extra tiles (approx.)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_extra_tiles_approx.png"))
    plt.close(fig)


def plot_psnr(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["eval/mean_psnr_db"], color="darkorange")
    ax.set_title("Mean viewport PSNR (held-out eval traces)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_psnr.png"))
    plt.close(fig)


def plot_pds_value_loss(df: pd.DataFrame, out_dir: str) -> None:
    """PDS arms only - normal PPO's progress.csv has no train/pds_value_loss
    column at all, so this panel is naturally skipped for those (see PANELS'
    needed_cols check), not specially cased here."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["train/pds_value_loss"], color="crimson")
    ax.set_title(r"PDS critic loss: mean$(\tilde V_\psi(\tilde\omega^t) - \tilde y^t)^2$")
    ax.set_xlabel("rollout")
    ax.set_ylabel("PDS value loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_pds_value_loss.png"))
    plt.close(fig)


def plot_policy_entropy(df: pd.DataFrame, out_dir: str) -> None:
    """New (2026-09-24) - previously listed as unavailable, but
    train/entropy_loss was in progress.csv the whole time, just under a
    namespaced name. SB3 logs entropy_loss = -mean(entropy), so this negates
    it back to plain policy entropy (a positive quantity that decays as the
    policy sharpens, the usual reading)."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], -df["train/entropy_loss"], color="slateblue")
    ax.set_title("Policy entropy (= -train/entropy_loss)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_policy_entropy.png"))
    plt.close(fig)


def plot_training_diagnostics(df: pd.DataFrame, out_dir: str) -> None:
    """New (2026-09-24) - explained_variance and approx_kl, PPO's own
    standard training-health diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["rollout"], df["train/explained_variance"], color="tab:brown")
    axes[0].axhline(1.0, color="black", ls="--", lw=0.8, label="perfect (1.0)")
    axes[0].set_title("Explained variance (critic fit quality)")
    axes[0].set_xlabel("rollout")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["rollout"], df["train/approx_kl"], color="tab:pink")
    axes[1].set_title("Approx. KL divergence per update")
    axes[1].set_xlabel("rollout")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_training_diagnostics.png"))
    plt.close(fig)


def plot_j_q_vs_max(df: pd.DataFrame, out_dir: str) -> None:
    """New (2026-09-24) - J_Q against its theoretical ceiling. Quality
    cost_Q^t = C_cov^t in [0,1] every step, so the max achievable discounted
    sum over one 36-step episode (config's own episode length,
    environment.py:146) is the same closed form config.py:399 already uses
    for J_B's own max: sum_{t=0}^{35} gamma^t."""
    episode_len = 36
    j_q_max = sum(config.PPO_GAMMA ** t for t in range(episode_len))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["eval/J_Q"], color="mediumvioletred", label=r"$J_Q$")
    ax.axhline(j_q_max, color="black", ls="--", lw=0.8, label=f"theoretical max = {j_q_max:.2f}")
    ax.set_title(r"Viewport quality return $J_Q$ vs. its theoretical max")
    ax.set_xlabel("rollout")
    ax.set_ylabel(r"$J_Q$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_j_q_vs_max.png"))
    plt.close(fig)


def plot_stall_time(df: pd.DataFrame, out_dir: str) -> None:
    """New (2026-09-24) - satisfies decision-log #30(a), "show stall
    behavior explicitly": eval/mean_stall_sec was always in progress.csv."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["eval/mean_stall_sec"], color="tab:red")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_title("Mean stall time per step (held-out eval traces)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("stall time (sec)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "13_stall_time.png"))
    plt.close(fig)


PANELS = [
    ("violation + feasibility", {"eval/violation", "eval/feasible"}, plot_violation_and_feasibility),
    ("constraint returns vs budgets", {"eval/J_D", "eval/J_P", "eval/J_B"}, plot_constraint_returns_vs_budgets),
    ("Lagrange multipliers", {"lagrangian/mu_D", "lagrangian/mu_P", "lagrangian/mu_B"}, plot_lagrange_multipliers),
    ("physical metrics", {"eval/mean_coverage", "physical/mean_n_enhanced_tiles",
                           "eval/mean_stall_sec", "physical/mean_power_mW"}, plot_physical_metrics),
    ("reward", {"rollout/ep_rew_mean"}, plot_reward),
    ("coverage vs enhanced tiles", {"eval/mean_coverage", "physical/mean_n_enhanced_tiles"}, plot_coverage_vs_enhanced_tiles),
    ("extra tiles approx", {"physical/mean_n_enhanced_tiles"}, plot_extra_tiles_approx),
    ("PSNR", {"eval/mean_psnr_db"}, plot_psnr),
    ("PDS value loss (PDS arms only)", {"train/pds_value_loss"}, plot_pds_value_loss),
    ("policy entropy", {"train/entropy_loss"}, plot_policy_entropy),
    ("training diagnostics", {"train/explained_variance", "train/approx_kl"}, plot_training_diagnostics),
    ("J_Q vs max", {"eval/J_Q"}, plot_j_q_vs_max),
    ("stall time", {"eval/mean_stall_sec"}, plot_stall_time),
]


def generate(csv_path: str, d_bar: float, p_bar: float, b_bar: float, out_dir: str,
            dropped_constraints: frozenset = frozenset()) -> None:
    df = load_progress(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    for name, needed_cols, fn in PANELS:
        if not needed_cols.issubset(df.columns):
            missing = needed_cols - set(df.columns)
            print(f"  SKIP '{name}': missing columns {missing}")
            continue
        if fn is plot_violation_and_feasibility:
            fn(df, out_dir, dropped_constraints)
        elif fn is plot_constraint_returns_vs_budgets:
            fn(df, d_bar, p_bar, b_bar, out_dir, dropped_constraints)
        elif fn is plot_lagrange_multipliers:
            fn(df, d_bar, p_bar, b_bar, out_dir, dropped_constraints)
        elif fn is plot_reward:
            fn(df, out_dir, dropped_constraints)
        else:
            fn(df, out_dir)
        print(f"  wrote {name}")


if __name__ == "__main__":
    csv_path, d_bar, p_bar, b_bar, out_dir = sys.argv[1:6]
    # optional 6th arg: comma-separated dropped constraints, e.g. "P"
    dropped = frozenset(sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[6] else frozenset()
    generate(csv_path, float(d_bar), float(p_bar), float(b_bar), out_dir, dropped)
