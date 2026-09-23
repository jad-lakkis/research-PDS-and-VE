"""
plot_pds_vs_baseline.py - overlay a PPO+PDS arm's key_metrics.csv against
its matched non-PDS baseline arm on the same panels, truncated to the
shorter run's rollout count so every comparison is at the same point in
training on both sides.

The two key_metrics.csv schemas differ (PDS's is the narrower extract:
rollout,feasible,violation,J_D,J_P,J_B,mu_D,mu_P,mu_B,coverage,
enhanced_tiles,reward,mean_psnr_db,pds_value_loss; baseline's older
schema uses viewport_quality_JQ / viewport_coverage_frac / stall_time_sec
/ transmit_power_mW / enhanced_tiles_mean / J_D_stall_return / ... /
violation_raw_sum / mu_D_stall_weight / ... / reward_ep_mean) - both are
renamed here to one common set of column names before plotting. PSNR has
no baseline counterpart (the baseline runs predate that metric), so
that panel is PDS-only, labeled as such rather than silently compared
against nothing.

Usage: python plot_pds_vs_baseline.py <pds_key_metrics.csv> <baseline_key_metrics.csv> <d_bar> <p_bar> <b_bar> <out_dir>
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASELINE_RENAME = {
    "violation_raw_sum": "violation",
    "J_D_stall_return": "J_D",
    "J_P_power_return": "J_P",
    "J_B_tiles_return": "J_B",
    "mu_D_stall_weight": "mu_D",
    "mu_P_power_weight": "mu_P",
    "mu_B_tiles_weight": "mu_B",
    "viewport_coverage_frac": "coverage",
    "enhanced_tiles_mean": "enhanced_tiles",
    "reward_ep_mean": "reward",
}


def load_and_align(pds_path: str, baseline_path: str):
    pds = pd.read_csv(pds_path)
    base = pd.read_csv(baseline_path).rename(columns=BASELINE_RENAME)

    n = min(len(pds), len(base))
    print(f"PDS run: {len(pds)} rollouts. PPO run: {len(base)} rollouts. "
          f"Truncating both to {n} rollouts for a matched comparison.")
    return pds.iloc[:n].reset_index(drop=True), base.iloc[:n].reset_index(drop=True)


def plot_violation(pds: pd.DataFrame, base: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(base["rollout"], base["violation"], color="gray", label="PPO")
    ax.plot(pds["rollout"], pds["violation"], color="firebrick", label="PPO+PDS")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_title(r"violation = max(0, $J_D-\bar D$) + max(0, $J_P-\bar P$) + max(0, $J_B-\bar B$)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("violation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_violation.png"))
    plt.close(fig)


def plot_constraint_returns(pds: pd.DataFrame, base: pd.DataFrame, d_bar: float, p_bar: float, b_bar: float, out_dir: str) -> None:
    specs = [("J_D", d_bar, "tab:blue", r"$J_D$ (stall)", "02a_J_D_vs_Dbar.png"),
             ("J_P", p_bar, "tab:orange", r"$J_P$ (power)", "02b_J_P_vs_Pbar.png"),
             ("J_B", b_bar, "tab:green", r"$J_B$ (tiles)", "02c_J_B_vs_Bbar.png")]
    for col, budget, color, label, fname in specs:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(base["rollout"], base[col], color="gray", label="PPO")
        ax.plot(pds["rollout"], pds[col], color=color, label="PPO+PDS")
        ax.axhline(budget, color="red", ls="--", label=f"budget = {budget}")
        ax.set_title(f"Discounted constraint return {label} vs. its budget")
        ax.set_xlabel("rollout")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)


def plot_multipliers(pds: pd.DataFrame, base: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Lagrange multiplier trajectories")

    specs = [("mu_D", "tab:blue", r"$\mu_D$ (stall)"),
             ("mu_P", "tab:orange", r"$\mu_P$ (power)"),
             ("mu_B", "tab:green", r"$\mu_B$ (tiles)")]
    for ax, (col, color, label) in zip(axes, specs):
        ax.plot(base["rollout"], base[col], color="gray", label="PPO")
        ax.plot(pds["rollout"], pds[col], color=color, label="PPO+PDS")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("rollout")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_lagrange_multipliers.png"))
    plt.close(fig)


def plot_physical(pds: pd.DataFrame, base: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Physical metrics")

    axes[0].plot(base["rollout"], base["coverage"], color="gray", label="PPO")
    axes[0].plot(pds["rollout"], pds["coverage"], color="tab:purple", label="PPO+PDS")
    axes[0].set_title("Coverage (fraction)")
    axes[0].set_xlabel("rollout")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(base["rollout"], base["enhanced_tiles"], color="gray", label="PPO")
    axes[1].plot(pds["rollout"], pds["enhanced_tiles"], color="olive", label="PPO+PDS")
    axes[1].set_title("Enhanced tiles (of 64)")
    axes[1].set_xlabel("rollout")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_physical_metrics.png"))
    plt.close(fig)


def plot_reward(pds: pd.DataFrame, base: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(base["rollout"], base["reward"], color="gray", label="PPO")
    ax.plot(pds["rollout"], pds["reward"], color="teal", label="PPO+PDS")
    ax.set_title(r"$r^t = \beta_Q C_{cov}^t - \mu_D cost_D^t - \mu_P cost_P^t - \mu_B cost_B^t$")
    ax.set_xlabel("rollout")
    ax.set_ylabel("mean episode reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_reward.png"))
    plt.close(fig)


def plot_psnr(pds: pd.DataFrame, out_dir: str) -> None:
    """PDS-only - the baseline runs predate the PSNR metric, no counterpart exists."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pds["rollout"], pds["mean_psnr_db"], color="darkorange")
    ax.set_title("Mean viewport PSNR (held-out eval traces)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_psnr_pds_only.png"))
    plt.close(fig)


def plot_coverage_percent(pds: pd.DataFrame, base: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(base["rollout"], base["coverage"] * 100, color="gray", label="PPO")
    ax.plot(pds["rollout"], pds["coverage"] * 100, color="tab:purple", label="PPO+PDS")
    ax.set_title("Viewport coverage")
    ax.set_xlabel("rollout")
    ax.set_ylabel("coverage (% of viewport)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_coverage_percent.png"))
    plt.close(fig)


def generate(pds_path: str, baseline_path: str, d_bar: float, p_bar: float, b_bar: float, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pds, base = load_and_align(pds_path, baseline_path)

    plot_violation(pds, base, out_dir)
    plot_constraint_returns(pds, base, d_bar, p_bar, b_bar, out_dir)
    plot_multipliers(pds, base, out_dir)
    plot_physical(pds, base, out_dir)
    plot_reward(pds, base, out_dir)
    plot_psnr(pds, out_dir)
    plot_coverage_percent(pds, base, out_dir)
    print(f"Wrote 7 comparison panels to {out_dir}")


if __name__ == "__main__":
    pds_path, baseline_path, d_bar, p_bar, b_bar, out_dir = sys.argv[1:7]
    generate(pds_path, baseline_path, float(d_bar), float(p_bar), float(b_bar), out_dir)
