# Recap and Next Steps — Since the Professor's 4-Point Reply

This picks up right where `research_summary.md` and `environment_design.md` left off: the professor answered 4 open questions, and everything below documents what happened next — turning those answers into `config.py`, building the actual `streaming_rl/` implementation package, a real investigation into one modeling choice that turned out to matter a lot (the enhancement-level `k`), and a fix for a formulation-fidelity gap. It ends with a full roadmap for what's left to finish the project.

---

## 1. The Professor's 4 Points → `config.py`

**(1) Channel model: "we can use the existing channel model."**
Reused MMSP'25's Rician-fading + path-loss model as-is: `P_MAX_DBM=20.0`, `W_HZ=10e6`, `T0_SEC=1.0`, `N0_DBM_PER_HZ=-174.0`, plus the full channel-gain constant set (`CHANNEL_CONST`, `ANTENNA_BEAMWIDTH_ARG`, `RICIAN_K_FACTOR_DB=12.0`, `RICIAN_E_PSI_SQUARED=1.0`).

**(2) BL/EL mechanism: "the signal/tile representation for the highest QP value would be its base layer... adding k number of enhancement layers would represent selecting QP index k (from highest to smallest)..."**
This was genuinely ambiguous on direction and needed checking, not assuming. Confirmed: the dataset's own QP array index 0 = highest bitrate/best quality, index 6 = lowest bitrate/worst quality (empirically, from `explore_data.ipynb`). So "highest QP value" = array index 6 = base layer — `BASE_QP_ARRAY_INDEX=6`. This was cross-checked hard: Runner's real array-index-6 whole-panorama bitrate is 4.41 Mbps, essentially identical to the professor's/MMSP'25's stated `A_base=4.5Mb` — strong evidence the direction is right. The k→array-index mapping was confirmed explicitly: `array_index = 6 - k` (`qp_array_index_for_enhancement_level()` in config.py).

**(3) FOV/mobility/training video: "you can also use the same parameters for FOV, mobility/distance, and the rest. Train and evaluate on one video for the time being."**
`V_THETA_DEG=60.0`, `V_PHI_DEG=30.0`, static-distance `UAV_DISTANCE_M=50.0`. Training video confirmed as `Runner` (`TRAINING_VIDEO_ARRAY_INDEX=4`) — matches MMSP'25's own experiments and the A_base cross-check above.

**(4) Constraint budgets: "power budget should match UAV battery... tiles should correspond to at least the viewport size... use different values for stall time and assess impact."**
- `D_BAR_CANDIDATES = None` — the method (sweep experimentally) is confirmed; specific numbers are not, and weren't invented.
- `P_BAR = None` — no real UAV/battery spec exists yet; left `None` deliberately rather than guessing.
- `B_BAR ≈ 0.20` — geometrically derived: the 120°×60° viewport over the 8×8/45°×22.5° tile grid touches 9–16 tiles depending on alignment, so `B_BAR_MIN_TILES=9`, `B_BAR_MAX_TILES=16`, with `0.20` (~13/64) as a starting point.

**Two more items resolved this session, not part of the original 4:**
- Partial-tile visibility: confirmed **area-weighted** (a tile's contribution is its exact overlap fraction, not a binary in/out classification) — `PARTIAL_TILE_VISIBILITY_MODE = "area_weighted"`.
- Critic count: confirmed as a **two-stage plan**, not a single fixed choice — "we will first implement a standard PPO baseline with one critic, `V(s_t)`. After validating the baseline, we will introduce post-decision states and add a second critic, `V(s̃_t)`, following the previous work." → `N_CRITICS_BASELINE=1`, `N_CRITICS_WITH_PDS=2`.

**Also empirically resolved this session** (yaw/pitch/roll convention, checked against all 194 valid `hn.mat` traces): yaw range is `[-180°,180°]` (44.3% of real values are negative — decisive), pitch/roll are in degrees (bounded to ~±90°/±134°, far beyond what radians would allow). Yaw's *positive direction* (left vs. right) could not be verified — not from the data, not from OpenTrack's own documentation (checked directly) — so it's recorded as an explicit, flagged **assumption** (`YAW_POSITIVE_DIRECTION = "right"`), not a verified fact.

**Still genuinely open, not yet asked of the professor**: D̄'s specific sweep values, and P̄'s real UAV/battery number.

---

## 2. Code Breakdown

- **`config.py`** — single source of truth for every parameter and convention above. Every value is either sourced (with a citation in its comment) or explicitly `None`/TBD with a note on what's missing.
- **`streaming_rl/data_loader.py`** — loads `hn.mat`/`rd.mat`/the video catalog. Directly ports the validity-classifier logic (empty/placeholder/valid) already built and verified in `explore_data.ipynb` — no new analysis, just made reusable.
- **`streaming_rl/channel_model.py`** — Rician-fading + path-loss channel gain. `sample_psi` implements the paper's literal stated distribution — both real and imaginary parts drawn from `Normal(component_mean, component_std)`, matching `ψ ~ CN(√(ν/(2+2ν)), 1/(2+2ν))` directly (this replaced an earlier, statistically-equivalent-but-differently-parametrized version once the literal paper formula was pinned down). Verified in `tests/test_channel_model.py`: `E[|ψ|²]` converges to 1.0, the empirical LOS/scatter power ratio matches `ν=10^1.2` to within 0.1%, all sampled gains are finite and non-negative, and seeded runs are reproducible.
- **`streaming_rl/layer_model.py`** — `A^t = A_base + Σ x_i·ΔA_i`. Verified directly: the "no tiles enhanced" case reproduces Runner's real `4,411,432` bit figure exactly, matching the A_base cross-check independently.
- **`streaming_rl/viewport.py`** — yaw/pitch → (θ,φ) → per-tile area-weighted coverage, with wraparound-aware geometry at the ±180° boundary. Verified via a strong invariant: total covered area across all 64 tiles equals the viewport's own area *exactly*, in every test case including boundary-straddling ones — proof the overlap geometry has no double-counting or gaps.
- **`streaming_rl/environment.py`** — `TileStreamingEnv`, the gymnasium `Env`. Passes gymnasium's official `check_env`. Implements the full PDS timing: known-after-action quantities (`A^t, R^t, D^t, Z^{t+1}`) computed before the random outcome (actual viewport, next channel gain) is revealed, matching eqs. 16–20 of the formulation.
- **`sanity_check.py`** — runs the environment with a random policy across several episodes and checks the numbers are sane (right episode length, no negative buffers/NaNs) before trusting it.
- **`tests/test_channel_model.py`** — the channel-model verification described above.

---

## 3. The k-Selection Investigation

This was the single biggest piece of real investigative work this session, because the first implementation choice turned out to produce a broken (non-)problem.

**k=1 (the original "simplest first step"):** buffer grew into the *billions* of bits over a single episode; stalling never happened under any random-policy episode tested. The enhancement cost was so small relative to channel capacity (~50–70× smaller) that the power/stall tradeoff never bound at all — nothing for an RL agent to learn.

**k=6 (first fix attempt — "just use best quality"):** properly binding, almost too well — measured directly, even a smart, viewport-targeted policy at full power stalled **91% of the time**. Good and bad policies both failed almost always, which is just as unlearnable as never failing.

**The systematic sweep** (`k_selection_experiment.py` — see note in Section 5, this file no longer exists in the project but its findings are real, verified, and reproducible by recreating it): tested k=1 through 6, crossed with three tile policies (enhance nothing / enhance exactly the predicted viewport / enhance a random ~50% of tiles) and all 5 power levels, under both a zero and a realistic ~1.5s startup buffer. Results:

| k | viewport-targeted stall rate | random stall rate |
|---|---|---|
| 1–4 | 0% | 0% (channel too generous even for random) |
| **5** | **0%** | **79.4%** |
| 6 | 91.1% | 97.8% |

k=5 was the unique value where a good policy reliably avoids stalling while a bad one clearly doesn't — `INITIAL_ENHANCEMENT_LEVELS=(0,5)` was set on this basis. Coverage gap (0.87 smart vs. 0.50 random) held at every k under this version of the environment, since coverage only depends on which tiles are chosen, not k.

**The forced-viewport-enhancement fix:** a review question surfaced that the environment wasn't enforcing eq. 2 of the formulation ("predicted viewport tiles are assigned `x_i=1`") — the agent had complete freedom to ignore the viewport entirely. Implemented `_apply_mandatory_viewport_enhancement()`, OR-ing the predicted-viewport tile mask into the submitted action before `A^t`/coverage are computed, then **re-ran the full sweep** rather than assuming k=5 still held. Findings:
- `base_only` and `viewport_targeted` became numerically identical (expected — a correctness check that passed).
- `random`'s raw coverage went *up* slightly (0.933 vs. 0.872) — coverage can only increase when you enhance more area, and random now enhances the mandatory viewport plus ~32 extra tiles, giving it a bigger chance of accidentally catching prediction-error drift. Raw coverage alone stopped discriminating smart from wasteful targeting once the mandatory floor existed.
- But it wasn't actually a wash: total episode reward still clearly favored the efficient policy once stall and tile-budget penalties are counted — **−12.38 (viewport-targeted, mandatory tiles only) vs. −36.58 (random, mandatory + 32 wasted tiles)**, nearly 3× worse.
- k=5's stall-rate gap actually **strengthened** under the corrected environment: 0.961 vs. 0.794 before.

---

## 4. Current Implementation Status

Everything in Sections 1–3 above is a real, verified conclusion. What's actually sitting in the files on disk right now is a step behind that:

| Item | Decided/verified | Currently on disk |
|---|---|---|
| `INITIAL_ENHANCEMENT_LEVELS` | `(0, 5)` | `(0, 1)` |
| Forced predicted-viewport enhancement | Implemented and re-verified | Not present — `environment.py` has no `_apply_mandatory_viewport_enhancement`, no `enhanced_level` override, no `initial_buffer_bits` reset parameter; `step()` hardcodes `enhanced_level=1` |
| `k_selection_experiment.py` | Written, run, findings above | File does not exist in the project |
| `channel_model.py` paper-formula fix | Implemented | **Present** — this one matches |

Nothing else in this document depends on that gap — the findings above are real, reproducible results, just not currently reflected in the running code. Closing that gap is next-step #1 below.

---

## 5. Next Steps to Finish the Project

1. **Re-apply the k=5 / forced-viewport-enhancement work to the actual files**: set `INITIAL_ENHANCEMENT_LEVELS=(0,5)` in `config.py`; re-add `_apply_mandatory_viewport_enhancement`, the `enhanced_level` constructor override, and the `initial_buffer_bits` reset parameter to `environment.py`; recreate `k_selection_experiment.py` if it's worth keeping in the repo as a reusable tool rather than one-off analysis.
2. **Resolve the remaining open items**: get D̄'s specific candidate values and P̄'s real UAV/battery number from the professor; decide whether yaw's positive direction needs real verification (e.g. a visual cross-check) before results are trusted, or whether the current flagged assumption is acceptable to proceed with.
3. **Install `torch` and `stable-baselines3`** into `.venv` (neither is installed yet — everything so far only needed numpy/pandas/scipy/gymnasium).
4. **Build `train_ppo.py`** and train a first baseline against `TileStreamingEnv`.
5. **Validate the baseline**: compare against a trivial/naive reference policy (e.g. always-base, or fixed-tile-selection), confirm learning curves actually improve over training rather than just running without crashing.
6. **Add the second training stage**: post-decision-state value estimation and the virtual-experience batching (Algorithm 1 from the problem formulation), plus the second critic `V(s̃_t)` — per the professor's explicit two-stage plan.
7. **Longer-term extensions**, explicitly deferred, not forgotten: the full graduated `k=0..6` enhancement range (not just base + one level); training across multiple videos for generalization instead of just Runner; the full Gauss-Markov UAV mobility model instead of the static-distance case.

---

## 6. What's Left — Quick Checklist

- [ ] Re-sync `config.py`/`environment.py` with the k=5 + forced-enhancement decisions
- [ ] D̄ candidate values (ask professor)
- [ ] P̄ real number (ask professor, needs UAV spec)
- [ ] Yaw-direction verification (currently an assumption)
- [ ] Install `torch`, `stable-baselines3`
- [ ] `train_ppo.py` + first baseline
- [ ] Baseline validation vs. reference policy
- [ ] Post-decision states + virtual experience + 2nd critic
- [ ] (Later) full k range, multi-video generalization, full mobility model
