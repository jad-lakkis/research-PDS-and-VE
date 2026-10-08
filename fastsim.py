"""
fastsim.py - fast re-implementation of TileStreamingEnv's physics for rule /
oracle studies (no learning). Same equations as streaming_rl/environment.py:

  A_t = A_base + sum_i delta_i x_i          (bits, enhancement level 5)
  R_t = W log2(1 + P_t h_t / (W N0))
  D_t = 0 if Z_t + T0 R_t >= A_t else T0 (1 - (Z_t + T0 R_t) / A_t)
  Z_{t+1} = max(Z_t + T0 R_t - A_t, 0)
  coverage = sum_i x_i w_i(actual viewport);  PSNR from area-pooled tile MSE
  J_* = sum_t gamma^t (.)   with J_D on D/T0, J_B on n/64, J_P on P/P_max

Viewing trajectories come from the head traces (previous viewport as the
prediction, mandatory = tiles touching the predicted viewport) and the
channel sequence h_t of each (viewer, seed) is recorded once from the real
environment (it does not depend on the actions). validate() checks this
simulator against the real environment.
"""
import numpy as np

import config
import room_to_learn as rl
from streaming_rl import channel_model, data_loader, viewport
from streaming_rl.environment import TileStreamingEnv

H, NT = 36, config.N_TILES
W, T0 = config.W_HZ, config.T0_SEC
WN0 = W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
PMAX = channel_model.dbm_to_watts(config.P_MAX_DBM)
GAM = config.DISCOUNT_FACTOR_LAMBDA
DISC = GAM ** np.arange(H)
LEVEL = config.INITIAL_ENHANCEMENT_LEVELS[1]
TB = np.array([viewport.tile_bounds(i) for i in range(NT)])


def region_masks(theta, phi, half_w, half_h):
    """(N,64) tiles with nonzero overlap with rectangles centred at (theta, phi) -
    the same rule as viewport.mandatory_tile_mask / run_nonrl_baselines.region_mask."""
    theta, phi = np.atleast_1d(theta)[:, None], np.atleast_1d(phi)[:, None]
    ov_t = np.zeros((theta.shape[0], NT))
    for s in (-360.0, 0.0, 360.0):
        ov_t += np.clip(np.minimum(theta + half_w, TB[:, 1] + s) - np.maximum(theta - half_w, TB[:, 0] + s), 0, None)
    ov_p = np.clip(np.minimum(phi + half_h, TB[:, 3]) - np.maximum(phi - half_h, TB[:, 2]), 0, None)
    return ov_t * ov_p > 0


class Video:
    def __init__(self, name, cat, rd, hn):
        row = cat[cat.name == name].iloc[0]
        self.name, self.vi = name, int(row.array_index)
        self.traces = data_loader.get_valid_traces(hn, self.vi)
        self.bundle = {"rd_data": rd, "valid_traces": self.traces}
        br, ym = rd["video_bitrate_data"][self.vi], rd["video_ymse_data"][self.vi]
        q0, qe = config.BASE_QP_ARRAY_INDEX, config.qp_array_index_for_enhancement_level(LEVEL)
        f = np.arange(H) * config.GOP_SIZE_FRAMES
        self.A_base = br[q0][:, f].sum(axis=0)                 # (H,)
        self.delta = (br[qe][:, f] - br[q0][:, f]).T           # (H,64)
        self.mse_b = ym[q0][:, f].T                            # (H,64)
        self.mse_e = ym[qe][:, f].T
        self.gain = self.mse_b - self.mse_e                    # MSE reduction from enhancing a tile
        self.mean_base_bits = float(self.A_base.mean())
        self.traj, self.h = {}, {}

    def trajectory(self, pos):
        if pos not in self.traj:
            arr = self.traces[pos][1]
            at = arr[np.arange(H) * config.GOP_SIZE_FRAMES, 0]
            ap = arr[np.arange(H) * config.GOP_SIZE_FRAMES, 1]
            pt, pp = np.concatenate([[at[0]], at[:-1]]), np.concatenate([[ap[0]], ap[:-1]])
            wa = rl.weights(at, ap)                            # coverage weights of the actual viewport
            self.traj[pos] = dict(at=at, ap=ap, pt=pt, pp=pp, w=wa, mand=rl.weights(pt, pp) > 0,
                                  err=np.concatenate([[0.0], np.hypot(rl.wsigned(at - pt), ap - pp)[:-1]]))
        return self.traj[pos]

    def channel(self, pos, seed):
        key = (pos, seed)
        if key not in self.h:
            old = config.TRAINING_VIDEO_ARRAY_INDEX
            config.TRAINING_VIDEO_ARRAY_INDEX = self.vi
            env = TileStreamingEnv(trace_indices=[pos], bundle=self.bundle)
            env.reset(seed=seed)
            hs = []
            for _ in range(H):
                _, _, te, tr, info = env.step(np.concatenate([np.zeros(NT, np.int64), [config.N_POWER_LEVELS - 1]]))
                hs.append(info["h_t"])
                if te or tr:
                    break
            config.TRAINING_VIDEO_ARRAY_INDEX = old
            self.h[key] = np.array(hs)
        return self.h[key]


def levels(top_frac):
    return np.array([k / (config.N_POWER_LEVELS - 1) * top_frac * PMAX for k in range(config.N_POWER_LEVELS)])


def rate(p, h):
    return W * np.log2(1.0 + p * h / WN0)


def pmin_level(A, Z, h, P):
    need = (WN0 / h) * (2.0 ** (max(A - Z, 0.0) / (T0 * W)) - 1.0)
    ok = np.where(P >= need - 1e-15)[0]
    return int(ok[0]) if len(ok) else len(P) - 1


def simulate(video, eps, policy, P):
    """policy(s) -> (extra_mask, level). s has pos, t, Z, h, tr (trajectory), video.
    Returns metrics averaged over episodes."""
    out = []
    for pos, seed in eps:
        tr, hs = video.trajectory(pos), video.channel(pos, seed)
        Z = 0.0
        acc = dict(J_cov=0.0, J_psnr=0.0, J_D=0.0, J_B=0.0, J_P=0.0, cov=0.0, psnr=0.0, extra=0.0, useful=0.0, bits=0.0)
        for t in range(H):
            s = dict(pos=pos, seed=seed, t=t, Z=Z, h=hs[t], tr=tr, video=video)
            extra, k = policy(s)
            x = extra | tr["mand"][t]
            A = video.A_base[t] + video.delta[t][x].sum()
            Rt = rate(P[k], hs[t])
            D = 0.0 if Z + T0 * Rt >= A else T0 * (1.0 - (Z + T0 * Rt) / A)
            Z = max(Z + T0 * Rt - A, 0.0)
            w = tr["w"][t]
            cov = float(w[x].sum())
            mse = np.where(x, video.mse_e[t], video.mse_b[t])
            psnr = 10.0 * np.log10(255.0 ** 2 / (np.sum(w * mse) / np.sum(w)))
            g = DISC[t]
            acc["J_cov"] += g * cov; acc["J_psnr"] += g * psnr; acc["J_D"] += g * D / T0
            acc["J_B"] += g * x.sum() / NT; acc["J_P"] += g * P[k] / PMAX
            acc["cov"] += cov / H; acc["psnr"] += psnr / H; acc["bits"] += A / H
            ex = extra & ~tr["mand"][t]
            acc["extra"] += ex.sum() / H; acc["useful"] += (ex & (w > 0)).sum() / H
        out.append(acc)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def simulate_trace(video, eps, policy, rates, n_hist=5, per_episode=False):
    """Same physics as simulate(), but the link rate comes from a recorded trace
    (power fixed): eps = [(viewer_pos, window_key)], rates[window_key] = R (bits/s)
    for n_hist seconds before the window plus its H seconds. R_t is known at
    decision time (as h_t is now); s["Rh"] holds R_{t-1}..R_{t-n_hist}.
    policy(s) -> extra_mask. Extra metrics: stall segments, quality variation,
    buffer (in segments of base + predicted-viewport bits)."""
    out = []
    for pos, key in eps:
        tr, R = video.trajectory(pos), rates[key]
        Z = 0.0
        acc = dict(J_cov=0.0, J_psnr=0.0, J_D=0.0, J_B=0.0, cov=0.0, psnr=0.0, extra=0.0, useful=0.0, bits=0.0,
                   stall_pct=0.0, qvar_dB=0.0, buffer_seg=0.0)
        prev = None
        for t in range(H):
            Rt = R[n_hist + t]
            s = dict(pos=pos, key=key, t=t, Z=Z, R=Rt, Rh=R[t:n_hist + t][::-1], tr=tr, video=video)
            extra = policy(s)
            x = extra | tr["mand"][t]
            A = video.A_base[t] + video.delta[t][x].sum()
            D = 0.0 if Z + T0 * Rt >= A else T0 * (1.0 - (Z + T0 * Rt) / A)
            acc["buffer_seg"] += Z / (video.A_base[t] + video.delta[t][tr["mand"][t]].sum()) / H
            Z = max(Z + T0 * Rt - A, 0.0)
            w = tr["w"][t]
            cov = float(w[x].sum())
            mse = np.where(x, video.mse_e[t], video.mse_b[t])
            psnr = 10.0 * np.log10(255.0 ** 2 / (np.sum(w * mse) / np.sum(w)))
            g = DISC[t]
            acc["J_cov"] += g * cov; acc["J_psnr"] += g * psnr; acc["J_D"] += g * D / T0
            acc["J_B"] += g * x.sum() / NT
            acc["cov"] += cov / H; acc["psnr"] += psnr / H; acc["bits"] += A / H
            acc["stall_pct"] += 100.0 * (D > 0) / H
            if prev is not None:
                acc["qvar_dB"] += abs(psnr - prev) / (H - 1)
            prev = psnr
            ex = extra & ~tr["mand"][t]
            acc["extra"] += ex.sum() / H; acc["useful"] += (ex & (w > 0)).sum() / H
        out.append(acc)
    if per_episode:
        return out
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def validate():
    """Compare against the real environment (Runner, held-out traces, a few rules)."""
    import run_nonrl_baselines as rb
    cat = data_loader.load_video_catalog()
    rd, hn = data_loader.load_rd_data(), data_loader.load_hn_data()
    v = Video("Runner", cat, rd, hn)
    P = levels(0.5)
    eps = [(p, s) for p in config.EVAL_TRACE_INDICES for s in (1000, 1001)]
    s_fit = (7.68, 0.83)
    cases = {
        "pred top": (lambda s: (np.zeros(NT, bool), 4), "mandatory", None, "p50"),
        "avg 0.25x pmin": (lambda s: (region_masks(s["tr"]["pt"][s["t"]], s["tr"]["pp"][s["t"]],
                                                     config.V_THETA_DEG + s_fit[0], config.V_PHI_DEG + s_fit[1])[0],
                                      pmin_level(v.A_base[s["t"]] + v.delta[s["t"]][region_masks(
                                          s["tr"]["pt"][s["t"]], s["tr"]["pp"][s["t"]], config.V_THETA_DEG + s_fit[0],
                                          config.V_PHI_DEG + s_fit[1])[0] | s["tr"]["mand"][s["t"]]].sum(), s["Z"], s["h"], P)),
                           "average", s_fit, "pwmin"),
    }
    bundle = v.bundle
    mb = v.mean_base_bits
    for name, (pol, kind, par, prule) in cases.items():
        fast = simulate(v, eps, pol, P)
        dec = rb.make_policy(kind, par, prule, mb)
        J = []
        for pos, seed in eps:
            env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
            env.reset(seed=seed)
            cov, psnr = [], []
            for _ in range(H):
                prop, k = dec(env)
                _, _, te, trn, info = env.step(np.concatenate([prop.astype(np.int64), [k]]))
                cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"])
                if te or trn:
                    break
            J.append((info["J_D"], info["J_B"], info["J_P"], np.mean(cov), np.mean(psnr)))
        env_m = np.mean(J, axis=0)
        fast_m = np.array([fast["J_D"], fast["J_B"], fast["J_P"], fast["cov"], fast["psnr"]])
        print(f"{name:16s} env  J_D {env_m[0]:.6f} J_B {env_m[1]:.6f} J_P {env_m[2]:.6f} cov {env_m[3]:.6f} psnr {env_m[4]:.6f}")
        print(f"{'':16s} fast J_D {fast_m[0]:.6f} J_B {fast_m[1]:.6f} J_P {fast_m[2]:.6f} cov {fast_m[3]:.6f} psnr {fast_m[4]:.6f}"
              f"   max abs diff {np.max(np.abs(env_m - fast_m)):.2e}")


if __name__ == "__main__":
    validate()
