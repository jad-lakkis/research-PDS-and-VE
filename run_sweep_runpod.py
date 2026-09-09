"""
One-command orchestrator for the Stage 9 RunPod joint D-bar/P-bar/B-bar
sweep (see other/decision_log.md #21-22 and the D-bar-only sweep this
supersedes). Launches every configured arm as a PARALLEL training run
(true OS subprocesses, not threads doing the compute - threading here
just avoids blocking between the concurrent subprocess.run() calls),
each auto-chaining resumed blocks until that arm's own convergence
check passes (or a hard rollout cap is hit), then extracts
final_results (CSV + a summary graph) per arm and one top-level sweep
comparison once every arm has stopped.

Sized for a 4-vCPU RunPod CPU pod: 3 arms running concurrently (1 core
each) + 1 core of headroom for the OS/this orchestrator process.

Why this file is separate from run_sweep.py (not a shared/edited copy):
run_sweep.py is the LOCAL laptop's own joint D-bar/P-bar test (2 arms,
sequential, different ARMS) - kept local, uncommitted, deliberately not
pushed. Sharing one filename between two machines running different
configs off the same git remote is a real hazard (a `git pull` on
either machine would silently overwrite the other's config). This file
is RunPod-only and can be committed/pushed safely alongside the
laptop's local edits to run_sweep.py without colliding.

Usage (on the RunPod pod): python run_sweep_runpod.py
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
# Sweep configuration - edit this list to change what's run.
# Every arm's fixed parameters are baked into its folder name so nothing
# needs opening a file to know which experiment produced which output.
# ---------------------------------------------------------------------
SWEEP_PARENT = ROOT / "runs" / "stage9_joint_sweep_runpod"
# Supersedes the earlier D-bar-only RunPod plan (0.02/0.10/0.15, P-bar/
# B-bar left at the loose config.py defaults 10.0/10.2) - that design
# was dropped per explicit user feedback: leaving P-bar/B-bar loose lets
# mu_P/mu_B decay to exactly 0 (same pattern already seen in run1 and
# dbar_0.30 - decision_log #20), so a D-bar-only sweep never exercises
# two-thirds of the Lagrangian mechanism. These arms instead tighten all
# three budgets together, every value grounded in real percentiles of
# run1's own converged-tail J_D/J_P/J_B (rollout 2001-2638, ALL-loose
# baseline: D-bar=3.2/P-bar=10.0/B-bar=10.2) - not round-number guesses.
# Both P-bar and B-bar stay above their hard physical floors (~3.8 for
# P-bar at minimum power every step; ~6.57 for B-bar, the mandatory-
# predicted-viewport-only floor) - below those the constraint would be
# unsatisfiable regardless of policy, not just difficult.
# NOTE (flagged, not yet resolved): these percentiles were measured
# under a LOOSE D-bar=3.2. Tightening D-bar is already known to push
# J_P/J_B up on its own (dbar_0.30 and dbar_0.05 both showed this
# without P-bar/B-bar ever being touched) - so once D-bar is ALSO
# tightened in the same arm, real achieved J_P/J_B will likely land
# higher than these percentiles suggest, i.e. the effective constraint
# is tighter than "p70"/"p30" implies. Some arms may hit MAX_ROLLOUTS
# without converging (as dbar_0.05 did) - that is an informative result
# here, not a failure.
ARMS = [
    # p70-ish joint tightening at a new D-bar point (fills the untested
    # 0.05-0.30 gap)
    {"d_bar": 0.15, "p_bar": 7.0, "b_bar": 7.7, "seed": 0},
    # milder (p80-ish) joint tightening at a different new D-bar point
    {"d_bar": 0.20, "p_bar": 7.5, "b_bar": 7.9, "seed": 0},
    # same D-bar as arm 1, P-bar/B-bar pulled much tighter (p30-ish) -
    # the deliberately aggressive stress case
    {"d_bar": 0.15, "p_bar": 6.3, "b_bar": 7.4, "seed": 0},
]

FIRST_BLOCK_ROLLOUTS = 700   # matches this project's established fresh-start block size
BLOCK_ROLLOUTS = 500         # matches established resumed-block size
STEPS_PER_ROLLOUT = 1728     # config.py: n_steps=1728=48*36, unchanged across this project
CONVERGENCE_WINDOW = 300     # rollouts to check feasibility/flatness over
FEASIBLE_FRACTION_REQUIRED = 0.99
FLATNESS_RELATIVE_DRIFT = 0.05   # total drift over the window vs. the window's own mean
MAX_ROLLOUTS = 3200              # hard safety cap per arm - bounds compute, per user request


def arm_name(arm: dict) -> str:
    return f"dbar_{arm['d_bar']}_pbar_{arm['p_bar']}_bbar_{arm['b_bar']}_seed{arm['seed']}"


def load_full_trajectory(arm_dir: Path) -> pd.DataFrame:
    blocks = sorted(arm_dir.glob("block_v*"), key=lambda p: int(p.name.split("_v")[1]))
    dfs = [pd.read_csv(b / "progress.csv") for b in blocks if (b / "progress.csv").exists()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def check_converged(df: pd.DataFrame) -> bool:
    """Formalizes the manual judgment used throughout this project: feasible
    and holding, and every multiplier has stopped moving (not just slowed
    down) over a real recent window - not a single-rollout snapshot, which
    is noisy (see decision_log #21's own false-plateau-read lesson). Checks
    all three multipliers (not just one), since every arm here jointly
    tightens D-bar, P-bar, and B-bar together."""
    if len(df) < CONVERGENCE_WINDOW:
        return False
    tail = df.tail(CONVERGENCE_WINDOW)
    feasible_ok = tail["eval/feasible"].mean() >= FEASIBLE_FRACTION_REQUIRED
    for mu_col in ["lagrangian/mu_D", "lagrangian/mu_P", "lagrangian/mu_B"]:
        mu = tail[mu_col].to_numpy()
        slope = np.polyfit(np.arange(len(mu)), mu, 1)[0]
        scale = max(abs(mu.mean()), 1e-6)
        if abs(slope) * len(mu) / scale >= FLATNESS_RELATIVE_DRIFT:
            return False
    return bool(feasible_ok)


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
            log(f"CONVERGED at {total_rollouts} rollouts (feasible + mu_D/mu_P/mu_B all flat "
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

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
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
    axes[2].legend()
    axes[3].plot(out["rollout"], out["J_P"], color="tab:orange")
    axes[3].axhline(arm["p_bar"], color="red", ls="--", lw=1, label="P-bar")
    axes[3].set_ylabel("J_P vs P-bar")
    axes[3].set_xlabel("rollout")
    axes[3].legend()
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
            "p_bar": arm["p_bar"],
            "b_bar": arm["b_bar"],
            "total_rollouts": len(df),
            "converged": (arm_dir / "STOPPED_CONVERGED").exists(),
            "hit_cap": (arm_dir / "STOPPED_CAPPED").exists(),
            "final_J_D": tail["eval/J_D"].mean(),
            "final_J_P": tail["eval/J_P"].mean(),
            "final_J_B": tail["eval/J_B"].mean(),
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
    print(f"Sweep comparison written to {out_dir}")


def main():
    # Parallel, not sequential: this pod is sized (4 vCPUs) specifically
    # for these 3 arms to run concurrently, one real core each, +1 core
    # of headroom - see module docstring. Each thread just blocks on its
    # own subprocess.run() call; the actual training work happens in
    # separate OS processes, not in these threads.
    threads = [threading.Thread(target=run_arm, args=(arm,), name=arm_name(arm)) for arm in ARMS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    build_sweep_comparison()
    print("SWEEP DONE")


if __name__ == "__main__":
    main()
