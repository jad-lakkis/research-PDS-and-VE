"""
budget_compare_lumos.py - stall allowance 0.1 with tile budget 10 / 11 / 12: every method,
extra tiles, whether each constraint binds, and when the extra tiles are sent.

Same setup and selection rule as budget_sweep_lumos.py (reuses its per-policy results in
VIDEO_SURVEY/lumos5g/budget_sweep_candidates.csv), plus:
  RL estimate, direction-blind   the RL-estimate rule without knowing where the predicted
                                 viewport is or where the head is moving (as with today's
                                 observation: buffer, link, |head move|); ranks tiles by how
                                 often training viewers looked at them, sends them while the
                                 link + buffer can carry them
Binding test: a constraint binds for a method if loosening it improves that method
(tile budget +1, or stall allowance 0.1 -> 0.2; each method re-tuned on training runs).
Link conditions: each second is "outage" (link + buffer can't carry the forced tiles),
"tight" (1-2x the forced tiles) or "comfortable" (> 2x), judged on the viewport-only buffer.

Usage: .venv/Scripts/python.exe budget_compare_lumos.py
"""
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

import budget_sweep_lumos as b
import fastsim as fs

OUT_DIR = b.OUT_DIR
ALLOW = 0.1
B_MAIN = (10.0, 11.0, 12.0)
B_SENS = (9.0, 10.0, 11.0, 12.0, 13.0)
A_SENS = (0.1, 0.2)
TILES = 64 / float(np.sum(fs.DISC))   # J_B -> tiles per segment (time-weighted average)
FORCED_TILES = 6.653 * TILES          # predicted-viewport tiles alone (J_B of viewport only)
FINE_Q = np.round(np.linspace(0.20, 0.80, 13), 2)   # finer tile-threshold steps for the two RL-estimate rules
FAMILIES = {"lit": "literature, tuned to our budgets", "rl": "RL estimate",
            "blind": "RL estimate, direction-blind (today's observation)", "ceil": "RL estimate, perfect viewport"}


def setup():
    b.setup()
    G, T = b.G, b.G["v"].traj
    pop = np.mean([T[p]["w"] for p in G["train"]], axis=(0, 1))
    G["POP"] = {p: np.where(T[p]["mand"], 0.0, pop[None, :]) for p in T}
    G["tau_sets"]["POP"] = np.quantile(np.concatenate([G["POP"][p][G["POP"][p] > 0] for p in G["train"]]),
                                       [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98])
    G["EMfine"], G["POPfine"] = G["EM"], G["POP"]
    for k in ("EM", "POP"):
        G["tau_sets"][k + "fine"] = np.quantile(np.concatenate([G[k][p][G[k][p] > 0] for p in G["train"]]), FINE_Q)


def fine_candidates():
    return [(fam, f"{tag}fine q{q} a 0.5 reserve {r} link now", ("rl", which, i, 0.5, r, "now"))
            for fam, tag, which in (("rl", "", "EMfine"), ("blind", "blind ", "POPfine"))
            for i, q in enumerate(FINE_Q) for r in (0.5, 1.0, 2.0)]


def blind_candidates():
    return [("blind", f"blind tau q{k} a {a} reserve {r} link {link}", ("rl", "POP", k, a, r, link))
            for k in range(7) for a in (0.5, 0.7, 1.0) for r in (0.0, 0.25, 0.5, 1.0, 2.0) for link in ("now",)]


def breakdown(args):
    """extra tiles and extra stall per link condition, test runs"""
    label, spec, b_bar = args
    G = b.G
    v, T, eps = G["v"], G["v"].traj, G["eps"]["test"]
    if spec == "oracle":
        orc = b.oracle(("test", ALLOW, b_bar), return_masks=True)
        pol = lambda s: orc[(s["pos"], s["key"])][s["t"]]
    else:
        pol = b.policy(spec)
    acc = {g: [0, 0.0, 0.0] for g in ("outage", "tight", "comfortable")}
    for p, w in eps:
        R, Z, Zf = G["rates"][w][b.N_HIST:], 0.0, 0.0
        for t in range(fs.H):
            s = dict(pos=p, key=w, t=t, Z=Z, R=R[t], Rh=G["rates"][w][t:t + b.N_HIST][::-1], tr=T[p], video=v)
            ex = pol(s) & ~T[p]["mand"][t]
            Af = G["A_m"][p][t]
            A = Af + v.delta[t][ex].sum()
            D = 0.0 if Z + R[t] >= A else 1.0 - (Z + R[t]) / A
            Df = 0.0 if Zf + R[t] >= Af else 1.0 - (Zf + R[t]) / Af
            room = (Zf + R[t]) / Af
            g = "outage" if room < 1 else ("tight" if room < 2 else "comfortable")
            acc[g][0] += 1; acc[g][1] += ex.sum(); acc[g][2] += fs.DISC[t] * (D - Df)
            Z, Zf = max(Z + R[t] - A, 0.0), max(Zf + R[t] - Af, 0.0)
    n = len(eps) * fs.H
    return [dict(b_bar=b_bar, method=label, condition=g, share_of_seconds=c / n, extra_tiles=x / max(c, 1),
                 extra_stall=st / len(eps)) for g, (c, x, st) in acc.items()]


def main():
    bl = blind_candidates()
    cache = f"{OUT_DIR}/budget_compare_candidates.csv"
    with Pool(8, initializer=setup) as pool:
        if os.path.exists(cache):   # policy evaluations don't depend on the budgets - reuse them
            ev = pd.read_csv(cache)
            ev.loc[(ev.family == "blind") & ~ev.name.str.startswith("blind "), "name"] = "blind " + ev.name
        else:
            old = pd.read_csv(f"{OUT_DIR}/budget_sweep_candidates.csv")
            res = pool.map(b.evaluate, [c[2] for c in bl], chunksize=4)
            new = pd.DataFrame([dict(family=f, name=n, split=sp, **m) for (f, n, _), r in zip(bl, res) for sp, m in r.items()])
            ev = pd.concat([old, new], ignore_index=True)
        if not ev.name.str.contains("fine q").any():
            fc = fine_candidates()
            res = pool.map(b.evaluate, [c[2] for c in fc], chunksize=2)
            ev = pd.concat([ev, pd.DataFrame([dict(family=f, name=n, split=sp, **m)
                                              for (f, n, _), r in zip(fc, res) for sp, m in r.items()])], ignore_index=True)
            bl = bl + fc
        else:
            bl = bl + fine_candidates()
        ev.to_csv(cache, index=False)
        assert not ev.duplicated(["name", "split"]).any(), "candidate names must be unique"
        floor = ev[ev.name == "predicted viewport only"].set_index("split").J_D
        ev["avoidable_stall"] = ev.J_D - ev.split.map(floor)
        spec = {n: s for _, n, s in b.candidates() + bl}

        tr = ev[ev.split == "train"]
        picks = {}
        for allow, bb, fam in product(A_SENS, B_SENS, FAMILIES):
            f = tr[(tr.family == fam) & (tr.avoidable_stall <= b.MARGIN * allow) & (tr.J_B <= b.MARGIN * bb)]
            picks[(allow, bb, fam)] = f.loc[f.J_cov.idxmax(), "name"] if len(f) else None
        uni = dict(zip(B_MAIN, pool.map(b.uniform_for, [spec[picks[(ALLOW, bb, "rl")]] for bb in B_MAIN])))
        orc = pool.map(b.oracle, [(sp, a, bb) for sp in ("val", "test") for a in A_SENS for bb in B_MAIN])
        jobs = [(FAMILIES[f], spec[picks[(ALLOW, bb, f)]], bb) for bb in B_MAIN for f in ("lit", "rl", "blind")]
        jobs += [("oracle", "oracle", bb) for bb in B_MAIN]
        brk = pd.DataFrame([r for rows in pool.map(breakdown, jobs) for r in rows])
    orc = pd.DataFrame(orc)

    # every method at allowance 0.1, tile budget 10 / 11 / 12
    rows = []
    for sp, bb in product(("val", "test"), B_MAIN):
        e = ev[ev.split == sp].set_index("name")
        fl = e.loc["predicted viewport only"]
        tag = dict(split=sp, allowance=ALLOW, b_bar=bb, allowed_tiles=bb * TILES)

        def add(method, m, choice="", tile_ok=True):
            rows.append(dict(**tag, method=method, choice=choice, psnr=m["psnr"], cov=m.get("cov", 1.0),
                             avoidable_stall=m["J_D"] - fl["J_D"], J_D=m["J_D"], J_B=m["J_B"], tiles=m["J_B"] * TILES,
                             extra_tiles=m["J_B"] * TILES - FORCED_TILES, stall_pct=m["stall_pct"], qvar_dB=m["qvar_dB"],
                             bits_Mbit=m["bits"] / 1e6,
                             within_budgets=bool(m["J_D"] - fl["J_D"] <= ALLOW + 1e-9 and (m["J_B"] <= bb + 1e-9 or not tile_ok))))
        add("oracle (actual viewport + future link)", orc[(orc.split == sp) & (orc.allowance == ALLOW) & (orc.b_bar == bb)].iloc[0])
        for fam in ("ceil", "rl", "blind"):
            add(FAMILIES[fam], e.loc[picks[(ALLOW, bb, fam)]], picks[(ALLOW, bb, fam)])
        add("uniform, same bits as RL estimate (all 64 tiles)", uni[bb][sp], "all 64 tiles", tile_ok=False)
        add(FAMILIES["lit"], e.loc[picks[(ALLOW, bb, "lit")]], picks[(ALLOW, bb, "lit")])
        for n in e[e.family == "published"].index:
            add(n, e.loc[n])
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/budget_compare.csv", index=False)

    # binding test: gain from loosening each budget (method re-tuned on training runs)
    bind = []
    for sp, bb, fam in product(("val", "test"), B_MAIN, FAMILIES):
        e = ev[ev.split == sp].set_index("name")
        base = e.loc[picks[(ALLOW, bb, fam)], "psnr"]
        bind.append(dict(split=sp, b_bar=bb, method=FAMILIES[fam],
                         gain_tile_budget_plus1=e.loc[picks[(ALLOW, bb + 1, fam)], "psnr"] - base,
                         gain_stall_allowance_0p2=e.loc[picks[(0.2, bb, fam)], "psnr"] - base,
                         train_stall_used=tr.set_index("name").loc[picks[(ALLOW, bb, fam)], "avoidable_stall"] / ALLOW,
                         train_tiles_used=tr.set_index("name").loc[picks[(ALLOW, bb, fam)], "J_B"] / bb))
    for sp in ("val", "test"):
        o = orc[orc.split == sp]
        for bb in B_MAIN:
            p1 = o[(o.allowance == ALLOW) & (o.b_bar == bb)].psnr.iloc[0]
            p2 = o[(o.allowance == 0.2) & (o.b_bar == bb)].psnr.iloc[0]
            bind.append(dict(split=sp, b_bar=bb, method="oracle", gain_stall_allowance_0p2=p2 - p1))
    bind = pd.DataFrame(bind)
    bind.to_csv(f"{OUT_DIR}/budget_compare_binding.csv", index=False)
    brk.to_csv(f"{OUT_DIR}/budget_compare_link_conditions.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
    cols = ["method", "within_budgets", "psnr", "cov", "avoidable_stall", "J_D", "tiles", "extra_tiles", "stall_pct", "qvar_dB", "choice"]
    for bb in B_MAIN:
        print(f"===== test runs, stall allowance {ALLOW}, tile budget {bb:g} (allowed {bb * TILES:.1f} tiles, {bb * TILES - FORCED_TILES:.1f} extra) =====")
        print(df[(df.split == "test") & (df.b_bar == bb)][cols].round(3).to_string(index=False))
    print("===== validation runs: PSNR =====")
    print(df[df.split == "val"].pivot_table(index="method", columns="b_bar", values="psnr").round(2).to_string())
    print("===== binding test (PSNR gain when loosening; train usage of each budget) =====")
    print(bind.round(3).to_string(index=False))
    print("===== when are extra tiles sent (test) =====")
    print(brk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
