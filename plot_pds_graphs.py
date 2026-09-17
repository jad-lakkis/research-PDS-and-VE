"""
plot_pds_graphs.py - regenerate the "old style" per-arm training graphs
(same panels/titles/colors as runpod_results_summary/02_most_important_runs'
normal_ppo graphs) from a PPO+PDS arm's key_metrics.csv, plus two new
PDS-specific panels (PSNR, PDS critic value loss).

Column-aware: only draws a panel if every column it needs is present in
the given CSV. The key_metrics.csv files currently synced down from
RunPod (rollout,feasible,violation,J_D,J_P,J_B,mu_D,mu_P,mu_B,coverage,
enhanced_tiles,reward,mean_psnr_db,pds_value_loss) are a NARROWER extract
than the full progress.csv SB3 writes during training (which also has
eval/J_Q, eval/mean_stall_sec, eval/mean_power_mw, train/entropy_loss,
train/explained_variance, train/approx_kl) - four of the old normal_ppo
panels (policy entropy, training diagnostics, J_Q-vs-max, the
stall-time half of the tradeoffs scatter) need columns not in that
narrower extract, and are skipped with a printed note rather than
silently faked. Pull the raw progress.csv (not just key_metrics.csv)
from each arm's --log-dir on the pod to unlock those once needed.

Usage: python plot_pds_graphs.py <key_metrics.csv> <d_bar> <p_bar> <b_bar> <out_dir>
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_violation_and_feasibility(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["violation"], color="firebrick")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_title(r"violation = max(0, $J_D-\bar D$) + max(0, $J_P-\bar P$) + max(0, $J_B-\bar B$)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("violation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_violation_and_feasibility.png"))
    plt.close(fig)


def plot_constraint_returns_vs_budgets(df: pd.DataFrame, d_bar: float, p_bar: float, b_bar: float, out_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Discounted constraint returns vs. their budgets")

    axes[0].plot(df["rollout"], df["J_D"], color="tab:blue")
    axes[0].axhline(d_bar, color="red", ls="--", label=f"budget = {d_bar}")
    axes[0].set_ylabel(r"$J_D$ (stall)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["rollout"], df["J_P"], color="tab:orange")
    axes[1].axhline(p_bar, color="red", ls="--", label=f"budget = {p_bar}")
    axes[1].set_ylabel(r"$J_P$ (power)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["rollout"], df["J_B"], color="tab:green")
    axes[2].axhline(b_bar, color="red", ls="--", label=f"budget = {b_bar}")
    axes[2].set_ylabel(r"$J_B$ (tiles)")
    axes[2].set_xlabel("rollout")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_constraint_returns_vs_budgets.png"))
    plt.close(fig)


def plot_lagrange_multipliers(df: pd.DataFrame, d_bar: float, p_bar: float, b_bar: float, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["mu_D"], label=r"$\mu_D$ (stall)", color="tab:blue")
    ax.plot(df["rollout"], df["mu_P"], label=r"$\mu_P$ (power)", color="tab:orange")
    ax.plot(df["rollout"], df["mu_B"], label=r"$\mu_B$ (tiles)", color="tab:green")
    ax.set_title(f"Lagrange multiplier trajectories - D-bar={d_bar}, P-bar={p_bar}, B-bar={b_bar}")
    ax.set_xlabel("rollout")
    ax.set_ylabel("multiplier value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_lagrange_multipliers.png"))
    plt.close(fig)


def plot_physical_metrics(df: pd.DataFrame, out_dir: str) -> None:
    """Partial version of the old 4-panel figure: only coverage and
    enhanced_tiles are in the current key_metrics.csv extract - stall_time
    and transmit_power need the raw progress.csv (see module docstring)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Physical metrics over training ({len(df)} rollouts) - partial, see script docstring")

    axes[0].plot(df["rollout"], df["coverage"], color="tab:purple")
    axes[0].set_title("Coverage (fraction)")
    axes[0].set_xlabel("rollout")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["rollout"], df["enhanced_tiles"], color="olive")
    axes[1].set_title("Enhanced tiles (of 64)")
    axes[1].set_xlabel("rollout")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_physical_metrics.png"))
    plt.close(fig)


def plot_reward(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["reward"], color="teal")
    ax.set_title(r"$r^t = \beta_Q C_{cov}^t - \mu_D cost_D^t - \mu_P cost_P^t - \mu_B cost_B^t$")
    ax.set_xlabel("rollout")
    ax.set_ylabel("mean episode reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_reward.png"))
    plt.close(fig)


def plot_coverage_vs_enhanced_tiles(df: pd.DataFrame, out_dir: str) -> None:
    """Reproducible half of the old '09_tradeoffs' scatter - the
    stall-time half needs a column not in the current extract."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    half = len(df) // 2
    ax.scatter(df["enhanced_tiles"].iloc[:half], df["coverage"].iloc[:half],
               s=10, color="firebrick", alpha=0.6, label=f"rollout 1-{half}")
    ax.scatter(df["enhanced_tiles"].iloc[half:], df["coverage"].iloc[half:],
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
    extra = df["enhanced_tiles"] - mandatory_avg
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
    """New, PDS-run-specific: this metric didn't exist when the
    normal_ppo baselines in 02_most_important_runs were trained."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["mean_psnr_db"], color="darkorange")
    ax.set_title("Mean viewport PSNR (held-out eval traces)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_psnr.png"))
    plt.close(fig)


def plot_pds_value_loss(df: pd.DataFrame, out_dir: str) -> None:
    """New, PDS-specific: the PDS critic's own MSE loss - has no
    normal_ppo counterpart, since normal_ppo has no PDS critic."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["rollout"], df["pds_value_loss"], color="crimson")
    ax.set_title(r"PDS critic loss: mean$(\tilde V_\psi(\tilde\omega^t) - \tilde y^t)^2$")
    ax.set_xlabel("rollout")
    ax.set_ylabel("PDS value loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_pds_value_loss.png"))
    plt.close(fig)


PANELS = [
    ("violation", {"violation"}, plot_violation_and_feasibility),
    ("constraint returns vs budgets", {"J_D", "J_P", "J_B"}, plot_constraint_returns_vs_budgets),
    ("Lagrange multipliers", {"mu_D", "mu_P", "mu_B"}, plot_lagrange_multipliers),
    ("physical metrics (partial)", {"coverage", "enhanced_tiles"}, plot_physical_metrics),
    ("reward", {"reward"}, plot_reward),
    ("coverage vs enhanced tiles (partial tradeoffs)", {"coverage", "enhanced_tiles"}, plot_coverage_vs_enhanced_tiles),
    ("extra tiles approx", {"enhanced_tiles"}, plot_extra_tiles_approx),
    ("PSNR (new)", {"mean_psnr_db"}, plot_psnr),
    ("PDS value loss (new)", {"pds_value_loss"}, plot_pds_value_loss),
]

SKIPPED_NO_DATA = [
    ("policy entropy", {"policy_entropy"}),
    ("training diagnostics (explained variance / approx KL)", {"explained_variance", "approx_kl"}),
    ("viewport quality J_Q vs max", {"J_Q"}),
    ("stall-time half of the tradeoffs scatter", {"stall_time_sec"}),
]


def generate(csv_path: str, d_bar: float, p_bar: float, b_bar: float, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    for name, needed_cols, fn in PANELS:
        if not needed_cols.issubset(df.columns):
            missing = needed_cols - set(df.columns)
            print(f"  SKIP '{name}': missing columns {missing}")
            continue
        if fn in (plot_constraint_returns_vs_budgets, plot_lagrange_multipliers):
            fn(df, d_bar, p_bar, b_bar, out_dir)
        else:
            fn(df, out_dir)
        print(f"  wrote {name}")

    for name, needed_cols in SKIPPED_NO_DATA:
        print(f"  NOT AVAILABLE '{name}': needs {needed_cols} (raw progress.csv, not in key_metrics.csv)")


if __name__ == "__main__":
    csv_path, d_bar, p_bar, b_bar, out_dir = sys.argv[1:6]
    generate(csv_path, float(d_bar), float(p_bar), float(b_bar), out_dir)
