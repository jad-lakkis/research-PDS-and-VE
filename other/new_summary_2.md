# Complete Summary — Since the Professor's Reply

Everything decided, built, and tested since the professor answered the 4 open questions, plus what still needs resolving before training can start. This document reflects only what is actually present and working in the code today.

---

## 1. The Professor's 4 Points → What They Became in `config.py`

**(1) Channel model — "we can use the existing channel model."**
Adopted MMSP'25's Rician-fading + path-loss model in full: `P_MAX_DBM=20.0`, `W_HZ=10e6`, `T0_SEC=1.0`, `N0_DBM_PER_HZ=-174.0`, plus the channel-gain constants (`CHANNEL_CONST`, `ANTENNA_BEAMWIDTH_ARG`, `RICIAN_K_FACTOR_DB=12.0`, `RICIAN_E_PSI_SQUARED=1.0`).

**(2) Base/enhancement-layer mechanism — "the signal/tile representation for the highest QP value would be its base layer... adding k number of enhancement layers would represent selecting QP index k."**
Confirmed which end of the dataset's QP scale is "base": the dataset's own array index 0 is highest bitrate/best quality, index 6 is lowest bitrate/worst quality (empirically, from the original dataset exploration). So "highest QP value" = array index 6 = base layer (`BASE_QP_ARRAY_INDEX=6`). Cross-checked against real data: `Runner`'s array-index-6 whole-panorama bitrate is 4.41 Mbps, matching the professor's/MMSP'25's stated `A_base=4.5Mb` almost exactly — strong independent confirmation. The enhancement-level-to-array-index mapping is `array_index = 6 - k` (`config.qp_array_index_for_enhancement_level(k)`).

**(3) FOV / mobility / training video — "use the same parameters for FOV, mobility/distance, and the rest. Train and evaluate on one video for the time being."**
`V_THETA_DEG=60.0`, `V_PHI_DEG=30.0` (full FOV 120°×60°), static distance `UAV_DISTANCE_M=50.0`. Training video: `Runner` (`TRAINING_VIDEO_ARRAY_INDEX=4`) — matches MMSP'25's own experiments and the A_base cross-check above.

**(4) Constraint budgets — "power budget should match UAV battery... tiles should correspond to at least the viewport size... use different values for stall time and assess impact."**
- `B_BAR ≈ 0.20` — derived geometrically: the 120°×60° viewport, over the 8×8 grid (45°×22.5° tiles), touches between 9 and 16 tiles depending on alignment; 0.20 (~13/64) is a starting candidate within that range.
- `P_BAR = None` — no real UAV/battery spec exists yet; left unset rather than guessed.
- `D_BAR_CANDIDATES = None` — the method (sweep several values, assess impact) is confirmed; the specific values are not chosen yet.

**Also resolved this session, beyond the original 4:**
- Partial-tile visibility: a tile's contribution to the coverage/reward calculation is weighted by the exact fraction of its area inside the viewport (not a binary in/out classification) — `PARTIAL_TILE_VISIBILITY_MODE = "area_weighted"`.
- Critic architecture staging: start with one critic (standard PPO baseline, estimating the value of the regular state), and add a second critic later once post-decision states are introduced, matching prior related work.
- Yaw/pitch/roll units and range: checked directly against all 194 valid head-navigation traces. Yaw is confirmed to range `[-180°, 180°]` (44.3% of real values are negative, which a `[0°,360°)` convention would make impossible). Pitch and roll are confirmed to be in degrees, not radians (values reach ~±90°/±134°, far beyond what radians would allow). Yaw's positive *direction* (does it mean turning left or right?) could not be verified from the data or from the likely capture tool's documentation — it's recorded as an explicit assumption, not a confirmed fact (`YAW_POSITIVE_DIRECTION = "right"`).

---

## 2. Code Breakdown — Every File, What It Does

**`config.py`** — the single source of truth for every parameter and convention above. Every value is either sourced (with a citation in its own comment: which dataset finding, which paper section, or which of the professor's exact words it came from) or explicitly left as `None`/TBD with a note on exactly what's missing. Nothing is a silent guess.

**`streaming_rl/data_loader.py`** — loads the dataset. `load_video_catalog()` parses the 15-video metadata table directly from the README at runtime. `load_hn_data()`/`load_rd_data()` wrap `scipy.io.loadmat` for the two `.mat` files. `classify_hmd_cell()` sorts each head-navigation recording into empty / placeholder / valid, exactly matching the classifier already built and verified during dataset exploration. `load_training_video_bundle()` is the single convenience entry point: returns the rate-distortion data and every valid trace for the configured training video (`Runner`) in one call.

**`streaming_rl/channel_model.py`** — the UAV-to-base-station channel gain. `sample_psi()` draws the Rician-fading complex sample directly from the paper's stated distribution: both the real and imaginary components are independently drawn from `Normal(component_mean, component_std)`, with the mean/std derived from the 12 dB K-factor so that `E[|ψ|²]=1` exactly, matching the paper's own formula literally rather than an equivalent-but-differently-parametrized version. `sample_channel_gain()` applies the path-loss formula on top. `transmission_rate()` computes the Shannon capacity. `dbm_to_watts()` handles the dB conversions.

**`streaming_rl/layer_model.py`** — computes `A^t`, the total data volume for one time slot: `A^t = A_base + Σ x_i · ΔA_i`, where the base cost is summed across all 64 tiles unconditionally, and each enhanced tile adds the bitrate difference between its enhanced and base QP levels. `frame_index_for_gop()` maps a GOP index to the representative frame used to look up bitrate (bitrate is constant within a GOP, confirmed during dataset exploration).

**`streaming_rl/viewport.py`** — converts a raw yaw/pitch reading into the actual-viewport rectangle, then computes each of the 64 tiles' coverage fraction. `yaw_pitch_to_theta_phi()` applies the (yaw, pitch) → (θ, φ) convention above. `tile_bounds()` gives each tile's angular boundaries under the assumed 8×8 raster grid layout. `tile_overlap_fractions()` computes the exact overlap area between the viewport rectangle and every tile, correctly handling wraparound at the ±180° boundary. `coverage()` turns that into the normalized quality metric.

**`streaming_rl/environment.py`** — `TileStreamingEnv`, the gymnasium `Env`. `reset()` picks a random valid trace, sets the buffer to 0, samples the initial channel gain, and bootstraps the first viewport prediction to be exactly correct (so the very first prediction-error term is exactly zero by construction). `step()`, for the current time index: unpacks the action into a 64-tile enhancement mask and a discrete power level; computes `A^t`, the transmission rate, the stall time, and the next buffer level (everything knowable immediately after the action, before any randomness resolves); then reveals the real recorded viewport position for that same time index, computes the coverage reward and the prediction error against it, samples the next channel gain, and returns the next observation. Observation space: buffer level, two prediction-error components, channel gain. Action space: a flat `MultiDiscrete` array — 64 binary tile flags plus one discrete power-level index (chosen as a flat array specifically because `stable-baselines3`'s PPO, the intended training library, doesn't support `Dict`/`Tuple` action spaces natively).

**`sanity_check.py`** — a standalone script that runs the environment with a random action policy across several full episodes and reports whether the results are numerically sane, before any training is attempted against it.

**`tests/test_channel_model.py`** — automated checks (plain assertions, no external test framework needed) verifying the channel model's statistical properties empirically rather than just algebraically.

---

## 3. What Was Actually Tested — In Detail, By File

**`tests/test_channel_model.py`** (run directly: `.venv\Scripts\python.exe tests\test_channel_model.py`) — four checks, all passing:
- Drew 500,000 samples of `ψ` and measured the mean of `|ψ|²`: **0.99972**, against a target of exactly 1.0.
- From the same samples, measured the empirical ratio of line-of-sight power to scattered power: **15.858**, against the target `ν = 10^1.2 = 15.849` — the two match to within 0.06%.
- Sampled 10,000 channel gains and confirmed every single one is finite and non-negative.
- Confirmed reproducibility: calling the sampler twice with the same seed produces bit-identical output; different seeds produce different output.

**`streaming_rl/layer_model.py`** (verified directly against real data) — with zero tiles enhanced, `compute_A_t()` for `Runner` returns **4,411,432 bits** exactly, independently reproducing the same figure found earlier when cross-checking `A_base` against the dataset. Enhancing a single additional tile was confirmed to add exactly that tile's own bitrate delta to the total, with no arithmetic drift.

**`streaming_rl/viewport.py`** (verified directly) — for a range of viewport center positions, including ones placed right at the ±180° wraparound boundary, the total area covered summed across all 64 tiles was measured to equal the viewport's own geometric area (120°×60° = 7,200 square degrees) **exactly**, in every case tested. This is a strong correctness proof: any bug in the overlap computation (double-counting a tile, or missing area at the wraparound seam) would have shown up as a mismatch here, and none did. Separately confirmed: a mask covering every tile that the viewport touches yields coverage of exactly 1.0; a mask covering nothing yields exactly 0.0.

**`streaming_rl/data_loader.py`** (verified directly) — loading the real dataset returns exactly 15 videos in the catalog, and exactly 12 valid head-navigation traces for `Runner`, matching the counts already established during the original dataset exploration.

**`streaming_rl/environment.py`** — passes gymnasium's own official `check_env` compliance checker cleanly (only a benign warning about the buffer/channel-gain observation bounds being intentionally unbounded above, which is correct per the formulation). `sanity_check.py` was run for multiple full episodes with a random tile/power policy and confirmed: every episode runs for exactly 36 steps (matching `Runner`'s 1,080 frames ÷ 30 frames-per-GOP), no negative buffer values occurred, no `NaN`/infinite values appeared anywhere in the observations or rewards, and coverage values under random tile selection behaved within the expected [0,1] range throughout.

---

## 4. What's Implemented vs. What's Not

**Implemented and verified**: the full single-agent MDP loop — state, action, reward, transition — as a working gymnasium environment, backed by real dataset values (not synthetic placeholders) for the video content, and a paper-faithful channel model.

**Not yet implemented**:
- **Post-decision states are not yet in the environment.** The current `step()` computes the reward in one pass; it does not separately expose the intermediate post-decision state (the point after the action's consequences are known but before the random viewport/channel outcome is revealed) as its own object for a value function to learn from. This is explicitly phase 2 of the professor's own staged plan (baseline first, PDS + second critic after), so this is expected, not a gap — but it means training cannot yet use the post-decision-state acceleration described in the formulation.
- **Virtual experience is not implemented** — same phase-2 status.
- **The predicted-viewport tiles are not forced to be enhanced.** The problem formulation specifies that predicted-viewport tiles are automatically assigned as enhanced, with the agent only choosing *additional* tiles beyond that. The current environment does not enforce this — the agent has complete freedom over all 64 tiles. (See open question below.)
- **The long-run budget constraints (`D̄`, `P̄`, `B̄`) are not enforced anywhere.** Only the penalty-based per-step reward (weighted by `β_q, β_s, β_p, β_b`) is implemented. The constrained-MDP formulation's actual expected-discounted-sum constraints are a separate mechanism that doesn't exist in the code yet. (See open question below.)
- **PPO training itself is not implemented.** `torch` and `stable-baselines3` are being installed now; `train_ppo.py` doesn't exist yet.

---

## 5. Open Questions for the Meeting

**Already identified, still need answers:**
1. What should the enhancement level `k` be?
2. For a tile that's only partially inside the predicted viewport and gets selected for enhancement — is the full tile sent at the enhanced quality level (since a tile is the smallest transmittable unit, there's no such thing as sending "part of" one)? This is separate from how much that tile's *coverage/reward* contribution is weighted by its visible fraction, which is already resolved (area-weighted).
3. Should the playback buffer start empty (`Z⁰=0`), or with a more realistic pre-buffered amount, the way a real client buffers a few seconds before starting playback?
4. `P̄` — what real UAV/battery number should this be?
5. `D̄` — what specific candidate values should be swept?

**Not yet raised, but genuinely still open:**
6. **Predicted-viewport enforcement** — should the environment force predicted-viewport tiles to always be enhanced (matching the formulation literally), or is it acceptable for the agent to have full freedom and simply be expected to learn to cover the viewport via the reward signal?
7. **Constraint enforcement mechanism** — should `D̄`, `P̄`, `B̄` be implemented as actual tracked constraints (e.g. a Lagrangian/primal-dual approach, penalizing violations of the long-run averages), or is the current per-step penalty reward (the `β` weights) considered a sufficient stand-in for now?
8. **Reward weights** — `β_q, β_s, β_p, β_b` are currently all set equal (1.0 each) with no basis beyond "something runnable." Does the professor have a preferred relative priority between quality, stalling, power, and tile-count (e.g. is quality meant to dominate, or should these be found purely by experimentation)?
9. **Tile-to-grid-position mapping** — the dataset never documents how its flat tile index (0–63) maps to a position in the 8×8 grid. The code assumes standard raster order (left-to-right, top-to-bottom) as the most conventional default, but this is unverified. Worth flagging even if there's no way to confirm it directly.
10. **Yaw direction** — flagged above as an explicit assumption (positive yaw = turning right). Worth mentioning even without an obvious way to verify it, in case the professor's group has documentation on the capture setup that would resolve it.
11. **Observation scaling** — the four state values (buffer level, two prediction-error terms, channel gain) currently span wildly different numeric scales (buffer can be in the millions/billions; channel gain is on the order of 1e-8). Should these be normalized before being given to the policy network, or handled some other way?

---

## 6. Next Steps

1. Resolve the open questions above with the professor.
2. Decide on the constraint-enforcement approach (question 7) and the predicted-viewport-enforcement approach (question 6), and implement whichever is chosen.
3. Finish installing `torch` (CPU-only) and `stable-baselines3` into the project environment.
4. Build the PPO training script and train a first baseline against the current environment.
5. Validate the baseline against a simple reference policy before trusting the training results.
6. Implement post-decision states, the second critic, and virtual experience — phase 2, once the baseline is validated.
7. Longer-term, once the above is working: broaden beyond a single training video, and revisit the static-UAV-distance simplification if full mobility modeling becomes relevant.
