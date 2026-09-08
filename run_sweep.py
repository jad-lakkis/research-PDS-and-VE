"""
One-command orchestrator for the Stage 9 D-bar sweep (see the plan and
other/decision_log.md #21+). Launches every configured D-bar value as a
parallel training arm, auto-chains resumed blocks per arm until that
arm's own convergence check passes (or a hard rollout cap is hit), then
extracts final_results (CSV + a summary graph) per arm and one top-level
sweep comparison once every arm has stopped.

Usage: python run_sweep.py
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ---------------------------------------------------------------------
# Sweep configuration - edit this list to change what's swept (e.g. for
# the future P-bar round: hold d_bar fixed at the chosen best value,
# vary p_bar instead, and set SWEPT_MULTIPLIER = "lagrangian/mu_P").
# Every arm's fixed parameters are baked into its folder name so nothing
# needs opening a file to know which experiment produced which output.
# ---------------------------------------------------------------------
SWEEP_PARENT = ROOT / "runs" / "stage9_dbar_sweep"
ARMS = [
    {"d_bar": 0.02, "p_bar": 10.0, "b_bar": 10.2, "seed": 0},
    {"d_bar": 0.10, "p_bar": 10.0, "b_bar": 10.2, "seed": 0},
    {"d_bar": 0.15, "p_bar": 10.0, "b_bar": 10.2, "seed": 0},
]
SWEPT_MULTIPLIER = "lagrangian/mu_D"  # which multiplier's flatness gates stopping

FIRST_BLOCK_ROLLOUTS = 700   # matches this project's established fresh-start block size
BLOCK_ROLLOUTS = 500         # matches established resumed-block size
STEPS_PER_ROLLOUT = 1728     # config.py: n_steps=1728=48*36, unchanged across this project
CONVERGENCE_WINDOW = 300     # rollouts to check feasibility/flatness over
FEASIBLE_FRACTION_REQUIRED = 0.99
FLATNESS_RELATIVE_DRIFT = 0.05   # total drift over the window vs. the window's own mean
MAX_ROLLOUTS = 4000              # hard safety cap per arm - bounds RunPod cost


def arm_name(arm: dict) -> str:
    return f"dbar_{arm['d_bar']}_pbar_{arm['p_bar']}_bbar_{arm['b_bar']}_seed{arm['seed']}"


def load_full_trajectory(arm_dir: Path) -> pd.DataFrame:
    blocks = sorted(arm_dir.glob("block_v*"), key=lambda p: int(p.name.split("_v")[1]))
    dfs = [pd.read_csv(b / "progress.csv") for b in blocks if (b / "progress.csv").exists()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def check_converged(df: pd.DataFrame) -> bool:
    """Formalizes the manual judgment used throughout this project: feasible
    and holding, and the swept multiplier has stopped moving (not just
    slowed down) over a real recent window - not a single-rollout snapshot,
    which is noisy (see decision_log #21's own false-plateau-read lesson)."""
    if len(df) < CONVERGENCE_WINDOW:
        return False
    tail = df.tail(CONVERGENCE_WINDOW)
    feasible_ok = tail["eval/feasible"].mean() >= FEASIBLE_FRACTION_REQUIRED
    mu = tail[SWEPT_MULTIPLIER].to_numpy()
    slope = np.polyfit(np.arange(len(mu)), mu, 1)[0]
    scale = max(abs(mu.mean()), 1e-6)
    flat_ok = abs(slope) * len(mu) / scale < FLATNESS_RELATIVE_DRIFT
    return bool(feasible_ok and flat_ok)


def run_arm(arm: dict):
    name = arm_name(arm)
    arm_dir = SWEEP_PARENT / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / "sweep_log.txt"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{name}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    block_n = 1
    prev_block_dir = None
    while True:
        block_dir = arm_dir / f"block_v{block_n}"
        rollouts_this_block = FIRST_BLOCK_ROLLOUTS if block_n == 1 else BLOCK_ROLLOUTS
        steps_this_block = rollouts_this_block * STEPS_PER_ROLLOUT

        cmd = [
            PYTHON, str(ROOT / "train_ppo.py"),
            "--d-bar", str(arm["d_bar"]),
            "--p-bar", str(arm["p_bar"]),
            "--b-bar", str(arm["b_bar"]),
            "--seed", str(arm["seed"]),
            "--n-steps", str(STEPS_PER_ROLLOUT),
            "--n-envs", "1",
            "--total-timesteps", str(steps_this_block),
            "--log-dir", str(block_dir),
        ]
        if prev_block_dir is not None:
            cmd += ["--resume-from", str(prev_block_dir)]

        log(f"launching block_v{block_n} ({rollouts_this_block} rollouts)")
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            log(f"block_v{block_n} FAILED (exit {result.returncode}) - stopping this arm")
            (arm_dir / "STOPPED_FAILED").write_text(f"block_v{block_n} exit {result.returncode}\n")
            return

        df = load_full_trajectory(arm_dir)
        total_rollouts = len(df)
        log(f"block_v{block_n} finished - {total_rollouts} rollouts total")

        if check_converged(df):
            log(f"CONVERGED at {total_rollouts} rollouts (feasible + {SWEPT_MULTIPLIER} flat "
                f"over last {CONVERGENCE_WINDOW})")
            (arm_dir / "STOPPED_CONVERGED").write_text(f"{total_rollouts} rollouts\n")
            break
        if total_rollouts >= MAX_ROLLOUTS:
            log(f"HIT SAFETY CAP at {total_rollouts} rollouts without converging")
            (arm_dir / "STOPPED_CAPPED").write_text(f"{total_rollouts} rollouts\n")
            break

        prev_block_dir = block_dir
        block_n += 1

    extract_final_results(arm, arm_dir)


def extract_final_results(arm: dict, arm_dir: Path):
    """Same key-metrics-CSV + graph pattern used for run1's final_results
    this project, trimmed to what the sweep comparison actually needs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_full_trajectory(arm_dir)
    df["rollout"] = range(1, len(df) + 1)
    out_dir = arm_dir / "final_results"
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (out_dir / "key_metrics").mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({
        "rollout": df["rollout"],
        "feasible": df["eval/feasible"].astype(int),
        "violation": df["eval/violation"].round(4),
        "J_D": df["eval/J_D"].round(4),
        "J_P": df["eval/J_P"].round(4),
        "J_B": df["eval/J_B"].round(4),
        "mu_D": df["lagrangian/mu_D"].round(4),
        "mu_P": df["lagrangian/mu_P"].round(4),
        "mu_B": df["lagrangian/mu_B"].round(4),
        "coverage": df["eval/mean_coverage"].round(4),
        "enhanced_tiles": df["physical/mean_n_enhanced_tiles"].round(2),
        "reward": df["rollout/ep_rew_mean"].round(3),
    })
    out.to_csv(out_dir / "key_metrics" / f"{arm_name(arm)}_key_metrics.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(out["rollout"], out["violation"], color="firebrick")
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("violation")
    axes[1].plot(out["rollout"], out["mu_D"], label="mu_D", color="tab:blue")
    axes[1].plot(out["rollout"], out["mu_P"], label="mu_P", color="tab:orange")
    axes[1].plot(out["rollout"], out["mu_B"], label="mu_B", color="tab:green")
    axes[1].set_ylabel("multiplier")
    axes[1].legend()
    axes[2].plot(out["rollout"], out["J_D"], color="tab:blue")
    axes[2].axhline(arm["d_bar"], color="red", ls="--", lw=1, label="D-bar")
    axes[2].set_ylabel("J_D vs D-bar")
    axes[2].set_xlabel("rollout")
    axes[2].legend()
    fig.suptitle(arm_name(arm))
    plt.tight_layout()
    plt.savefig(out_dir / "graphs" / "summary.png")
    plt.close()


def build_sweep_comparison():
    rows = []
    for arm in ARMS:
        arm_dir = SWEEP_PARENT / arm_name(arm)
        df = load_full_trajectory(arm_dir)
        if df.empty:
            continue
        tail = df.tail(min(CONVERGENCE_WINDOW, len(df)))
        rows.append({
            "arm": arm_name(arm),
            "d_bar": arm["d_bar"],
            "total_rollouts": len(df),
            "converged": (arm_dir / "STOPPED_CONVERGED").exists(),
            "hit_cap": (arm_dir / "STOPPED_CAPPED").exists(),
            "final_J_D": tail["eval/J_D"].mean(),
            "final_mu_D": tail["lagrangian/mu_D"].mean(),
            "final_mu_P": tail["lagrangian/mu_P"].mean(),
            "final_mu_B": tail["lagrangian/mu_B"].mean(),
            "final_coverage": tail["eval/mean_coverage"].mean(),
            "final_tiles": tail["physical/mean_n_enhanced_tiles"].mean(),
            "feasible_fraction": tail["eval/feasible"].mean(),
        })
    if not rows:
        print("No arm data found - nothing to compare.")
        return
    comp = pd.DataFrame(rows).sort_values("d_bar")
    out_dir = SWEEP_PARENT / "sweep_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "sweep_comparison.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(comp["d_bar"], comp["final_J_D"], "o-")
    axes[0].plot(comp["d_bar"], comp["d_bar"], "--", color="gray", label="budget")
    axes[0].set_xlabel("D-bar"); axes[0].set_ylabel("achieved J_D"); axes[0].legend()
    axes[1].plot(comp["d_bar"], comp["final_mu_D"], "o-", color="tab:blue")
    axes[1].axhline(0, color="black", ls="--", lw=0.8)
    axes[1].set_xlabel("D-bar"); axes[1].set_ylabel("final mu_D")
    axes[2].plot(comp["d_bar"], comp["total_rollouts"], "o-", color="tab:green")
    axes[2].set_xlabel("D-bar"); axes[2].set_ylabel("rollouts to stop")
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_comparison.png")
    plt.close()
    print(f"Sweep comparison written to {out_dir}")


def main():
    threads = [threading.Thread(target=run_arm, args=(arm,), name=arm_name(arm)) for arm in ARMS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    build_sweep_comparison()
    print("SWEEP DONE")


if __name__ == "__main__":
    main()
