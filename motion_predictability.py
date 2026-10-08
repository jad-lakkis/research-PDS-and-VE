"""
motion_predictability.py - can the next viewport be predicted better than
"previous viewport" from the information an agent would have, on each video?

At segment t the agent knows the viewer's past one-second viewing directions
(a_{t-1}, a_{t-2}, ...), the time in the video, and (from training) where other
viewers looked at time t. Predictors of a_t, trained on training viewers and
scored on held-out viewers (every 4th viewer; Runner keeps its usual split):
  previous viewport   a_{t-1}  (what the literature rules use)
  extrapolate         a_{t-1} + move_{t-1}, and a damped version (+0.5 move)
  popularity          circular mean of the training viewers' directions at t
  learned linear      ridge regression on [last 3 moves, position, time, popularity offset]
  learned MLP         small neural net on the same features
Score: coverage of the ACTUAL viewport by the tiles that touch a 120x60 viewport
centred on the prediction (the same rule as the forced tiles), and the mean error.

Usage: .venv/Scripts/python.exe motion_predictability.py
"""
import numpy as np
import pandas as pd
import torch

import config
import room_to_learn as rl
from streaming_rl import data_loader

OUT = "VIDEO_SURVEY/motion_predictability.csv"
torch.manual_seed(0)


def wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def circ_mean(theta):
    r = np.deg2rad(theta)
    return np.rad2deg(np.arctan2(np.sin(r).mean(axis=0), np.cos(r).mean(axis=0)))


def coverage_of(pred_t, pred_p, act_t, act_p):
    mand = rl.weights(pred_t, pred_p) > 0
    w = rl.weights(act_t, act_p)
    return (mand * w).sum(axis=1), mand.sum(axis=1)


def features(th, ph, t, pop_t, pop_p, H):
    """rows for segments t >= 1 of one viewer; features use only data known at decision time."""
    rows = []
    for k in range(1, H):
        mv = [wrap(th[max(k - j, 0)] - th[max(k - j - 1, 0)]) for j in range(1, 4)]
        mp = [ph[max(k - j, 0)] - ph[max(k - j - 1, 0)] for j in range(1, 4)]
        rows.append(mv + mp + [ph[k - 1], np.sin(np.deg2rad(th[k - 1])), np.cos(np.deg2rad(th[k - 1])), k / H,
                               wrap(pop_t[k] - th[k - 1]), pop_p[k] - ph[k - 1]])
    return np.array(rows)


def main():
    cat = data_loader.load_video_catalog()
    hn = data_loader.load_hn_data()
    out = []
    for _, v in cat.iterrows():
        traces = data_loader.get_valid_traces(hn, v.array_index)
        H = v.frames // config.GOP_SIZE_FRAMES
        n = len(traces)
        held = list(config.EVAL_TRACE_INDICES) if v["name"] == "Runner" else [i for i in range(n) if i % 4 == 2]
        train = [i for i in range(n) if i not in held]
        TH = np.array([tr[1][np.arange(H) * config.GOP_SIZE_FRAMES, 0] for tr in traces])
        PH = np.array([tr[1][np.arange(H) * config.GOP_SIZE_FRAMES, 1] for tr in traces])

        def pop_for(i):  # leave-one-out for training viewers
            others = [j for j in train if j != i]
            return circ_mean(TH[others]), PH[others].mean(axis=0)

        X, Y, meta = {}, {}, {}
        for i in range(n):
            pt, pp = pop_for(i)
            X[i] = features(TH[i], PH[i], None, pt, pp, H)
            Y[i] = np.stack([wrap(TH[i][1:] - TH[i][:-1]), PH[i][1:] - PH[i][:-1]], axis=1)
            meta[i] = (pt, pp)
        Xtr = np.concatenate([X[i] for i in train]); Ytr = np.concatenate([Y[i] for i in train])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        # ridge regression
        A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        Wr = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ Ytr)
        # small MLP
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
        xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(Ytr / 30.0, dtype=torch.float32)
        for _ in range(1500):
            opt.zero_grad(); loss = torch.nn.functional.mse_loss(net(xt), yt); loss.backward(); opt.step()

        res = {}
        for i in held:
            th, ph = TH[i], PH[i]
            at, ap = th[1:], ph[1:]
            pt, pp = meta[i]
            xs = (X[i] - mu) / sd
            preds = {
                "previous viewport": (th[:-1], ph[:-1]),
                "extrapolate": (wrap(th[:-1] + np.r_[0, wrap(np.diff(th))[:-1]]), ph[:-1] + np.r_[0, np.diff(ph)[:-1]]),
                "extrapolate (damped)": (wrap(th[:-1] + 0.5 * np.r_[0, wrap(np.diff(th))[:-1]]), ph[:-1] + 0.5 * np.r_[0, np.diff(ph)[:-1]]),
                "popularity": (pt[1:], pp[1:]),
            }
            r = np.c_[xs, np.ones(len(xs))] @ Wr
            preds["learned linear"] = (wrap(th[:-1] + r[:, 0]), np.clip(ph[:-1] + r[:, 1], -90, 90))
            with torch.no_grad():
                m = net(torch.tensor(xs, dtype=torch.float32)).numpy() * 30.0
            preds["learned MLP"] = (wrap(th[:-1] + m[:, 0]), np.clip(ph[:-1] + m[:, 1], -90, 90))
            for name, (qt, qp) in preds.items():
                cov, ntiles = coverage_of(qt, qp, at, ap)
                err = np.hypot(wrap(at - qt), ap - qp)
                res.setdefault(name, []).append((cov.mean(), err.mean(), ntiles.mean()))
        mean_move = np.mean(np.abs(wrap(np.diff(TH, axis=1))))
        for name, vals in res.items():
            vals = np.array(vals)
            out.append(dict(video=v["name"], seconds=H, viewers=n, held_out=len(held), mean_move_deg=mean_move, predictor=name,
                            coverage=vals[:, 0].mean(), mean_error_deg=vals[:, 1].mean(), tiles=vals[:, 2].mean()))
        print(v["name"], "done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    p = df.pivot_table(index=["video", "seconds", "viewers", "mean_move_deg"], columns="predictor", values="coverage")
    p["best learned - previous"] = p[["learned linear", "learned MLP", "extrapolate (damped)", "popularity"]].max(axis=1) - p["previous viewport"]
    e = df.pivot_table(index=["video"], columns="predictor", values="mean_error_deg")
    pd.set_option("display.width", 250)
    print("===== coverage of the actual viewport by the tiles around each prediction (held-out viewers) =====")
    print(p.sort_values("mean_move_deg", ascending=False).round(3).to_string())
    print("===== mean prediction error (deg) =====")
    print(e.round(1).to_string())


if __name__ == "__main__":
    main()
