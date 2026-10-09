"""
train_pensieve.py - Pensieve as a baseline in our system (the learned baseline ASL360 compares against).

What Pensieve is
  Mao, Netravali, Alizadeh, "Neural Adaptive Video Streaming with Pensieve", ACM SIGCOMM 2017: an adaptive-bitrate
  (ABR) controller for ordinary (2D) DASH video, learned with reinforcement learning. Before each video chunk it
  picks one bitrate from a fixed ladder. It looks at:
    the last chosen bitrate, the buffer (s), the last 8 throughputs and download times, the size of the next
    chunk at every bitrate, and the number of chunks left.
  Reward = linear QoE: q(bitrate) - 4.3 * rebuffering (s) - |q(bitrate) - q(previous bitrate)|, q = bitrate in
  Mbit/s, 4.3 = the top bitrate (one second of stall costs as much as one second of top quality).
  Network: 1-D convolutions (128 filters, size 4) over the histories and the chunk sizes, 128-unit dense layers for
  the scalars, merged into one 128-unit layer; softmax over the ladder. The critic has its own copy of the body.
  Training: A3C, 16 agents, RMSProp, actor lr 1e-4, critic lr 1e-3, discount 0.99, entropy weight 1 -> 0.1.

How ASL360 used it, and how it is used here
  ASL360 (arXiv 2509.10544): "We implemented Pensieve, without modification, allowing it to choose between low
  quality segments (equivalent to BL segments) and high quality segments (the entire frames are encoded at higher
  quality) based solely on its learned RL policy."
  --ladder asl360 (default): each second, low = base layer + the forced predicted-viewport tiles (the least our
      system sends) or high = all 64 tiles enhanced (the entire frame at the enhancement level).
  --ladder nested: six levels like Pensieve's six bitrates - the forced viewport enlarged by 0, 0.5, 1, 2, 3 x our
      mean prediction error (the average-based rule's box at 1x), and all 64 tiles.
  Unchanged: the state, the QoE reward (q = the level's mean bitrate, scaled so the top level is 4.3), the network
  and the training (A3C -> Stable-Baselines3's synchronous A2C: 16 environments, one 36-s episode each per update,
  gamma 0.99, full returns, RMSProp with the two learning rates, entropy weight decayed 1 -> 0.1).
  Pensieve does not know our budgets; like ASL360, we report it against them.
  Physics are the replay's (fastsim, verified against the environment): buffer, stall, PSNR, tiles. Our
  formulation gives every method the current second's rate R^t before the decision, so Pensieve's throughput
  history ends with R^t. Buffer in seconds = Z / bits of one second of the forced stream.

Usage (from the project folder):
  train:     .venv/Scripts/python.exe train_pensieve.py --link lumos5g --ladder asl360 --seed 1 --log-dir runs/pensieve/lumos_asl360_seed1
  evaluate:  .venv/Scripts/python.exe train_pensieve.py --link lumos5g --ladder asl360 --evaluate runs/pensieve/lumos_asl360_seed1/model.zip
Outputs in --log-dir: progress.csv (training + validation), model.zip, eval_<split>.csv (per episode) and
eval_summary.csv (Lumos5G: test runs and the 36 validation episodes; old channel: the 3 held-out episodes).
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl
from streaming_rl import channel_model, lumos5g

S_LEN = 8                       # Pensieve: 8 past throughputs / download times
Q_TOP = 4.3                     # Pensieve: top bitrate (Mbit/s); also the rebuffering penalty
BUFFER_NORM, DELAY_CAP = 10.0, 10.0
NESTED = (0.0, 0.5, 1.0, 2.0, 3.0)   # enlargements (x mean prediction error) below the all-tiles level
N_HIST = S_LEN - 1              # rate samples before the window (the window's own R^t is the 8th)


# --- data ----------------------------------------------------------------------------------------------------------
def prepare(link):
    """Runner trajectories (fastsim), training / held-out viewers, the ladder's tile masks, and the rate sources."""
    if link == "lumos5g":
        ds.TRAIN_SEEDS, ds.HELD_SEEDS = (), ()        # no synthetic channels needed
    v, train, held = ds.prepare("Runner")             # rician: caches the held-out channels (seeds 1000-1002)
    T = v.traj
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    data = lumos5g.load_bundle() if link == "lumos5g" else None
    return dict(v=v, train=train, held=held, s_fit=s_fit, data=data, link=link)


def ladder_masks(D, ladder):
    """{viewer: (n_levels, H, 64) bool} - tiles sent at each level (forced tiles included)."""
    v, s_fit = D["v"], D["s_fit"]
    out = {}
    for p, tr in v.traj.items():
        if ladder == "asl360":
            lv = [tr["mand"], np.ones((fs.H, fs.NT), bool)]
        else:
            lv = [fs.region_masks(tr["pt"], tr["pp"], config.V_THETA_DEG + k * s_fit[0], config.V_PHI_DEG + k * s_fit[1]) | tr["mand"]
                  for k in NESTED] + [np.ones((fs.H, fs.NT), bool)]
        out[p] = np.stack(lv)
    return out


def level_bits(v, masks):
    """(n_levels, H, viewers...) -> bits of each level per (viewer, t)."""
    return {p: np.stack([v.A_base + (v.delta * m[k]).sum(axis=1) for k in range(m.shape[0])]) for p, m in masks.items()}


def eval_episodes(D, split):
    """[(viewer, rates)] with rates = N_HIST samples before the window + H. Lumos5G: held-out viewers x the test
    runs' windows every 12 s ('test', as the heuristics' replay) or x the 12 validation windows ('val', the
    training runs' evaluation episodes). Old channel: the 3 held-out episodes (viewers 2/6/9, seeds 1000-1002)."""
    v = D["v"]
    if D["link"] == "rician":
        eps = []
        for p, seed in zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS):
            R = fs.rate(config.FIXED_POWER_WATTS, v.channel(p, seed))
            eps.append((p, np.concatenate([np.full(N_HIST, R[0]), R]), (p, seed)))
        return eps
    runs, sp = D["data"]["runs"], D["data"]["splits"]
    wins = lumos5g.windows(runs, sp["test"], fs.H, 12) if split == "test" else lumos5g.eval_windows(runs, sp["val"], fs.H, 12)
    return [(p, lumos5g.rate_window(runs, r, s, fs.H, 1.0, N_HIST), (p, r, s)) for (r, s) in wins for p in D["held"]]


# --- environment ---------------------------------------------------------------------------------------------------
class PensieveEnv(gym.Env):
    """One 36-s episode of Runner; action = ladder level; Pensieve's state and QoE reward; fastsim physics."""

    def __init__(self, D, masks, bits, q, viewers, episodes=None):
        super().__init__()
        self.D, self.v, self.masks, self.bits, self.q = D, D["v"], masks, bits, np.asarray(q, float)
        self.viewers, self.episodes = list(viewers), episodes
        self.n_levels = len(q)
        self.action_space = spaces.Discrete(self.n_levels)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, S_LEN), dtype=np.float32)
        if D["link"] == "lumos5g" and episodes is None:
            self.starts = lumos5g.all_starts(D["data"]["runs"], D["data"]["splits"]["train"], fs.H)
        self._ep_i = 0

    def _draw(self):
        if self.episodes is not None:                     # fixed evaluation list, in order
            p, R, _ = self.episodes[self._ep_i % len(self.episodes)]
            self._ep_i += 1
            return p, R
        p = self.viewers[self.np_random.integers(len(self.viewers))]
        if self.D["link"] == "lumos5g":
            i = self.np_random.integers(len(self.starts[0]))
            R = lumos5g.rate_window(self.D["data"]["runs"], int(self.starts[0][i]), int(self.starts[1][i]), fs.H, 1.0, N_HIST)
        else:   # i.i.d. Rician gain each second at the fixed 50 mW, as the environment draws it
            h = np.array([channel_model.sample_channel_gain(self.np_random) for _ in range(fs.H + N_HIST)])
            R = fs.rate(config.FIXED_POWER_WATTS, h)
        return p, R

    def _obs(self):
        t, s = self.t, np.zeros((6, S_LEN), np.float32)
        s[0, -1] = self.q[self.last] / Q_TOP
        s[1, -1] = self.Z / self.A_forced(t) / BUFFER_NORM
        s[2, :] = self.R[t:t + S_LEN] / 1e9                  # R^{t-7} .. R^t, Gbit/s
        s[3, :] = self.delays / DELAY_CAP
        s[4, :self.n_levels] = self.bits[self.p][:, t] / 1e9   # this second's size at every level, Gbit
        s[5, -1] = (fs.H - t) / fs.H
        return s

    def A_forced(self, t):
        return self.v.A_base[t] + self.v.delta[t][self.tr["mand"][t]].sum()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p, self.R = self._draw()
        self.tr = self.v.trajectory(self.p)
        self.t, self.Z, self.last = 0, 0.0, 0
        self.delays = np.zeros(S_LEN)
        return self._obs(), {}

    def step(self, action):
        a, t, v, tr = int(action), self.t, self.v, self.tr
        x = self.masks[self.p][a, t]
        A = v.A_base[t] + v.delta[t][x].sum()
        Rt = self.R[N_HIST + t]
        D = 0.0 if self.Z + fs.T0 * Rt >= A else fs.T0 * (1.0 - (self.Z + fs.T0 * Rt) / A)
        self.Z = max(self.Z + fs.T0 * Rt - A, 0.0)
        w = tr["w"][t]
        mse = np.where(x, v.mse_e[t], v.mse_b[t])
        psnr = 10.0 * np.log10(255.0 ** 2 / (np.sum(w * mse) / np.sum(w)))
        reward = self.q[a] - Q_TOP * D - abs(self.q[a] - self.q[self.last])
        self.delays = np.roll(self.delays, -1)
        self.delays[-1] = min(A / Rt, DELAY_CAP) if Rt > 0 else DELAY_CAP
        info = dict(psnr=psnr, D=D, tiles=int(x.sum()), A=A, level=a, cov=float(w[x].sum()), qoe=reward)
        self.last, self.t = a, t + 1
        done = self.t >= fs.H
        return (np.zeros((6, S_LEN), np.float32) if done else self._obs()), float(reward), done, False, info


# --- network: Pensieve's actor / critic body -------------------------------------------------------------------------
class PensieveExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_levels, filters=128, hidden=128):
        super().__init__(observation_space, features_dim=hidden)
        self.n_levels = n_levels
        self.last, self.buf, self.left = nn_dense(filters), nn_dense(filters), nn_dense(filters)
        self.tput = th.nn.Conv1d(1, filters, 4)
        self.delay = th.nn.Conv1d(1, filters, 4)
        self.sizes = th.nn.Conv1d(1, filters, 4) if n_levels >= 4 else th.nn.Linear(n_levels, filters)
        n_sizes = filters * (n_levels - 3) if n_levels >= 4 else filters
        self.merge = th.nn.Linear(3 * filters + 2 * filters * (S_LEN - 3) + n_sizes, hidden)

    def forward(self, obs):
        r = th.relu
        parts = [r(self.last(obs[:, 0:1, -1])), r(self.buf(obs[:, 1:2, -1])), r(self.left(obs[:, 5:6, -1])),
                 r(self.tput(obs[:, 2:3, :])).flatten(1), r(self.delay(obs[:, 3:4, :])).flatten(1)]
        sz = obs[:, 4, :self.n_levels]
        parts.append(r(self.sizes(sz.unsqueeze(1))).flatten(1) if self.n_levels >= 4 else r(self.sizes(sz)))
        return r(self.merge(th.cat(parts, dim=1)))


def nn_dense(n):
    return th.nn.Linear(1, n)


class PensieveA2C(A2C):
    """A2C with Pensieve's optimiser: RMSProp, actor lr = the schedule, critic lr = 10x (1e-4 / 1e-3)."""
    critic_lr_ratio = 10.0

    def _setup_model(self):
        super()._setup_model()
        pol = self.policy
        critic = (list(pol.vf_features_extractor.parameters()) + list(pol.mlp_extractor.value_net.parameters())
                  + list(pol.value_net.parameters()))
        ids = {id(x) for x in critic}
        actor = [x for x in pol.parameters() if id(x) not in ids]
        lr = self.lr_schedule(1.0)
        pol.optimizer = th.optim.RMSprop([{"params": actor, "lr": lr}, {"params": critic, "lr": lr * self.critic_lr_ratio}],
                                         lr=lr, alpha=0.99, eps=1e-5)

    def _update_learning_rate(self, optimizers):
        lr = self.lr_schedule(self._current_progress_remaining)
        self.logger.record("train/learning_rate", lr)
        for opt in (optimizers if isinstance(optimizers, list) else [optimizers]):
            opt.param_groups[0]["lr"] = lr
            opt.param_groups[1]["lr"] = lr * self.critic_lr_ratio


# --- evaluation ------------------------------------------------------------------------------------------------------
def run_policy(D, masks, bits, q, episodes, act):
    """act(env, obs) -> level. Per-episode metrics with our formulation's quantities."""
    env = PensieveEnv(D, masks, bits, q, D["held"], episodes=episodes)
    rows = []
    for (p, _, key) in episodes:
        obs, _ = env.reset()
        acc = dict(psnr=0.0, J_D=0.0, J_B=0.0, tiles=0.0, bits_Mbit=0.0, qoe=0.0, stall_pct=0.0, qvar_dB=0.0)
        lv, prev = np.zeros(len(q)), None
        for t in range(fs.H):
            obs, r, done, _, info = env.step(act(env, obs))
            g = fs.DISC[t]
            acc["psnr"] += info["psnr"] / fs.H
            acc["J_D"] += g * info["D"] / fs.T0
            acc["J_B"] += g * info["tiles"] / fs.NT
            acc["tiles"] += info["tiles"] / fs.H
            acc["bits_Mbit"] += info["A"] / 1e6 / fs.H
            acc["qoe"] += r
            acc["stall_pct"] += 100.0 * (info["D"] > 0) / fs.H
            if prev is not None:
                acc["qvar_dB"] += abs(info["psnr"] - prev) / (fs.H - 1)
            prev = info["psnr"]
            lv[info["level"]] += 1.0 / fs.H
        rows.append(dict(viewer=p, episode=str(key), **acc, **{f"share_level{k}": lv[k] for k in range(len(q))}))
    return pd.DataFrame(rows)


def summarize(D, ep, floor):
    s = ep.drop(columns=["viewer", "episode"]).mean()
    s["avoidable_stall"] = s["J_D"] - floor["J_D"].mean()
    if D["link"] == "lumos5g":
        s["within_budgets"] = bool(s["avoidable_stall"] <= config.LUMOS5G_STALL_ALLOWANCE + 1e-9 and s["J_B"] <= config.LUMOS5G_B_BAR + 1e-9)
    else:
        s["within_budgets"] = bool(s["J_D"] <= 0.3 + 1e-9 and s["J_B"] <= 8.0 + 1e-9)
    return s


class EntropyDecayAndEval(BaseCallback):
    """Entropy weight 1 -> 0.1 linearly over decay_updates (Pensieve); validation every eval_every updates."""

    def __init__(self, decay_updates, eval_every, eval_fn):
        super().__init__()
        self.decay_updates, self.eval_every, self.eval_fn, self.updates = decay_updates, eval_every, eval_fn, 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.updates += 1
        self.model.ent_coef = 1.0 - 0.9 * min(1.0, self.updates / self.decay_updates)
        self.logger.record("train/entropy_weight", self.model.ent_coef)
        if self.eval_every and self.updates % self.eval_every == 0:
            for k, x in self.eval_fn(self.model).items():
                self.logger.record(f"val/{k}", float(x))


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--link", choices=["lumos5g", "rician"], default="lumos5g")
    pa.add_argument("--ladder", choices=["asl360", "nested"], default="asl360")
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--updates", type=int, default=20000, help="A2C updates (16 episodes each)")
    pa.add_argument("--n-envs", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-4, help="actor lr; the critic gets 10x (Pensieve)")
    pa.add_argument("--eval-every", type=int, default=250, help="updates between validation runs (0 = off)")
    pa.add_argument("--log-dir", type=str, default="runs/pensieve/run")
    pa.add_argument("--evaluate", type=str, default=None, help="only evaluate this saved model")
    args = pa.parse_args()

    D = prepare(args.link)
    masks = ladder_masks(D, args.ladder)
    bits = level_bits(D["v"], masks)
    mean_bits = np.mean([bits[p].mean(axis=1) for p in D["train"]], axis=0)
    q = Q_TOP * mean_bits / mean_bits[-1]
    print(f"{args.link}, ladder {args.ladder}: mean Mbit/s per level {np.round(mean_bits / 1e6, 1)}, q {np.round(q, 2)}, "
          f"mean prediction error {D['s_fit'][0]:.1f} / {D['s_fit'][1]:.1f} deg")
    splits = ["test", "val"] if args.link == "lumos5g" else ["held"]
    eps = {sp: eval_episodes(D, sp) for sp in splits}
    floors = {sp: run_policy(D, masks, bits, q, eps[sp], lambda env, o: 0) for sp in splits}   # forced tiles only

    def det(model):
        return lambda env, o: int(model.predict(o, deterministic=True)[0])

    if args.evaluate:
        model = PensieveA2C.load(args.evaluate)
        out_dir = os.path.dirname(args.evaluate)
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        venv = DummyVecEnv([lambda i=i: PensieveEnv(D, masks, bits, q, D["train"]) for i in range(args.n_envs)])
        venv.seed(args.seed)
        model = PensieveA2C(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=fs.H, gamma=0.99, gae_lambda=1.0, ent_coef=1.0, vf_coef=0.5,
            max_grad_norm=1e9, normalize_advantage=False, seed=args.seed, verbose=0,
            policy_kwargs=dict(features_extractor_class=PensieveExtractor, features_extractor_kwargs=dict(n_levels=len(q)),
                               share_features_extractor=False, net_arch=dict(pi=[], vf=[])))
        model.set_logger(configure(args.log_dir, ["csv"]))
        val_sp = "val" if args.link == "lumos5g" else "held"

        def val(m):
            s = summarize(D, run_policy(D, masks, bits, q, eps[val_sp], det(m)), floors[val_sp])
            return {k: s[k] for k in ("qoe", "psnr", "J_D", "avoidable_stall", "J_B", "share_level0")}

        t0 = time.time()
        model.learn(total_timesteps=args.updates * args.n_envs * fs.H,
                    callback=EntropyDecayAndEval(int(0.8 * args.updates), args.eval_every, val))
        print(f"trained {args.updates} updates in {time.time() - t0:.0f} s")
        model.save(os.path.join(args.log_dir, "model.zip"))
        out_dir = args.log_dir
    rows = []
    for sp in splits:
        ep = run_policy(D, masks, bits, q, eps[sp], det(model))
        ep.to_csv(os.path.join(out_dir, f"eval_{sp}.csv"), index=False)
        rows.append(dict(split=sp, link=args.link, ladder=args.ladder, **summarize(D, ep, floors[sp]).to_dict()))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
