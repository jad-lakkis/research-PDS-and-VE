# Environment Design Notes

Pre-implementation design discussion for the tile-selective aerial 360° video streaming RL environment. Written to be read and discussed with Jacob before any gymnasium/PPO code is written. Companion to `explore_data.ipynb` (Sections A–G there cover what's literally in the dataset and what a closely related paper, Chakareski/Wang/Mastronarde MMSP'25, contributes — this document builds on both).

Two things are kept strictly separate throughout: **confirmed facts** (sourced from the problem formulation, the dataset, or the MMSP'25 paper) and **decisions that still have to be made by the researcher** (presented with tradeoffs, not resolved here).

## 1. Confirmed facts

- State `ω^t = (Z^t, Δ^{t-1}_θ, Δ^{t-1}_φ, h^t)` — identical in both the problem formulation and the MMSP'25 paper's Section III.A.
- `P_max = 20 dBm`, `W = 10 MHz`, `T0 = 1 s`, `N0 = -174 dBm/Hz`, `λ = 0.01` — MMSP'25 Sec. IV, directly reusable.
- Channel gain: `h = 3.24e-4·|ψ|² / (d·atan(24√2/5))²`, with Rician fading `ψ ~ CN(√(ν/(2+2ν)), 1/(2+2ν))`, `ν=10^1.2` (12 dB Rician factor), `E[|ψ|²]=1` — MMSP'25 Sec. II.D. A complete, closed-form model with numeric parameters — this is very likely the "prior work" channel model Jacob mentioned reusing in the meeting.
- Actual-viewport rule: `V_actu = {(θ,φ) : θ̃-V_θ≤θ≤θ̃+V_θ, φ̃-V_φ≤φ≤φ̃+V_φ}`, a rectangle in (azimuth, altitude) angle space centered on the actual viewing direction, using only 2 of the 3 head-orientation DOF (roll is dropped) — MMSP'25 Sec. II.C. `V_θ=60°, V_φ=30°` (full FOV 120°×60°) is the value MMSP'25 used experimentally — a strong reference point, not a documented property of the dataset's own capture rig.
- Predicted-viewport baseline: `(θ̂^t,φ̂^t)=(θ̃^{t-1},φ̃^{t-1})` — "last viewport as prediction," MMSP'25 Sec. IV. Matches what Jacob suggested in the meeting.
- Tile grid: 64 tiles arranged 8×8, stated explicitly for `Runner` (`array_index=4` in `video_catalog`) — MMSP'25 Sec. II.B.
- `A^t = A_base + T0·C·η(2V_θ+2s_θ)·η(2V_φ+2s_φ)` — MMSP'25 Sec. II.C. Confirms the base layer is *always* sent (whole panorama) and the enhancement layer is *additive* on top for the selected region — matches the problem formulation's own eq. (12) exactly (`A^t = A_base + Σ_{i∈E^t(x^t)} A^t_i,enh`). **This directly answers "is the enhancement layer sent on top of the base layer": yes, both the formulation and this precursor paper already assume it.**
- Buffer/stall equations (`Z^{t+1}=max(Z^t+T0R^t-A^t,0)`, `D^t` per MMSP'25 eq. 3) are algebraically identical to the formulation's eqs. (14)–(15).
- "Using this bitrate and the R-D models from [15], we can then compute the respective viewport quality (Y-PSNR)" (MMSP'25 Sec. IV; [15] is the MMSys'21 dataset paper) — confirms that *some* bitrate→quality model, of the same kind as `rd.mat`'s `RtoD_exp/pow`, is what this line of work actually uses to turn an allocated bitrate into a quality signal.
- GOP size = 30 frames, empirically confirmed against `rd.mat` in `explore_data.ipynb` Section C.

## 2. Decisions still to be made (not resolved here — present to Jacob)

**R-D quality model.** The MATLAB `cfit` objects in `RtoD_exp`/`RtoD_pow`/`QPtoR_exp`/`QPtoR_pow` are not readable via `scipy.io.loadmat` (confirmed in `explore_data.ipynb` Section E — they're MCOS handles into an undocumented 386.5 MB binary blob). Two ways forward, both grounded in the real data already available:
- Refit `D=a·e^(bR)` / `D=a·R^b` curves in Python from the raw `(QP, bitrate, YMSE)` points in `video_bitrate_data`/`video_ymse_data` using `scipy.optimize.curve_fit`. Matches the dataset's own methodology most faithfully; extra engineering work (fitting per tile/GOP/video as needed).
- Skip the continuous curve entirely and use the raw 7 discrete QP operating points as a lookup table. Simpler, still real-data-grounded, but the agent's choices are then implicitly restricted to those 7 discrete quality levels per tile rather than a continuous rate.
- Either way, the answer to "am I going to use them" is: not the MATLAB objects directly (they're unreadable) — either a Python refit of the same functional form, or a discrete stand-in for them. **They don't need to be opened again; the raw numeric arrays already extracted are enough for both paths.**

**`A^t` model.** MMSP'25's own formula treats `A_base`, `C`, `η` as fixed constants — content-independent, closed-form, simple. An alternative is a QP-table-grounded model: pick specific QP indices from `rd.mat` to represent "base" and "enhanced" quality per tile/GOP, and sum the real measured bitrates for whichever tiles are selected. Concrete tradeoff:
- *Formulaic (MMSP'25-style)*: `A^t` depends only on the **action** (how many pixels/tiles are enhanced) and fixed constants — simple, closed-form, but ignores that real content needs different bit budgets for the same quality (a busy tile costs more than a flat one at the same QP).
- *QP-grounded*: `A^t` depends on the action **and** on which video/GOP is currently playing (since bitrate varies by content) — more faithful to the actual dataset, but means the environment has to track "current video index + current GOP index" internally as part of its dynamics, not just the RL-visible state `ω^t`.

**Which QP index = base, which = enhanced.** Genuinely open — MMSP'25 doesn't use QP indices at all (its bitrate is a smooth formula), and the problem formulation doesn't specify which of the 7 QP levels to treat as "base" vs "enhanced." No source resolves this; it's a pure modeling choice.

**UAV mobility.** Model the full Gauss-Markov distance dynamics (MMSP'25 eqs. 2a–2c) or use the static special case MMSP'25 actually reports results for (`d=50m` fixed, `ζ=1`, `v0=0`, only the Rician fading term varies per step)? The static case is simpler and matches Jacob's incremental "start simple" guidance from the meeting.

**FOV values.** Adopt `V_θ=60°, V_φ=30°` as-is (MMSP'25's experimental choice), or track down whatever FOV the dataset's own capture rig (Oculus Rift + Whirligig, per the MMSys'21 paper) actually used? Not yet confirmed either way.

**Yaw/pitch → (θ,φ) mapping convention, and roll.** Neither paper states whether `yaw` maps directly onto azimuth `θ` and `pitch` onto altitude `φ` with matching sign/range conventions (e.g. `hn.mat`'s yaw values span roughly `-180°..180°`, as seen in `explore_data.ipynb` Section B/D — need to confirm this lines up with how `θ` is defined in the viewport rule above before implementing it). MMSP'25's model drops roll entirely; the same simplification is available here but is itself a choice, not a given.

## 3. What the environment models

Following the same environment/agent split as MMSP'25 Fig. 5, adapted to the problem formulation's richer structure (discrete tile selection instead of continuous enlargement; the post-decision-state split; the virtual-experience independence assumption):

- **Environment holds and evolves**: playback buffer level, viewport-prediction error, and the aerial-to-ground channel state — exactly the three components of `ω^t`.
- **Agent chooses**: which tiles to enhance (`x^t`) and how much transmit power to use (`P^t`).
- **Reward** closes the loop, combining a component known immediately after the action (stall time, power cost, tile-count cost) with a component that depends on the as-yet-unrevealed actual viewport (viewport quality) — this immediate/random split is exactly the formulation's post-decision-state (PDS) mechanism, and is *not* present in MMSP'25's simpler model (which has no PDS or virtual-experience layer at all — that machinery is unique to the problem formulation, one step beyond what MMSP'25 does).

## 4. Computation timing breakdown

**Loaded once, at environment construction:**
- `video_catalog`, the 194 valid `HMD_data` traces, `video_bitrate_data`/`video_ymse_data`, the tile grid constant (8×8)
- If the "refit" R-D path is chosen: the refit `a`/`b` coefficients (computed once, cached — refitting per-episode would be wasteful)

**Computed at `reset()`:**
- Which video (and which of its valid user traces) this episode replays
- Initial `Z^0` (design choice, e.g. 0 or a partial fill)
- Initial `Δ^{-1}_θ, Δ^{-1}_φ` (design choice, e.g. 0)
- Initial `h^0` (first draw from the channel model), and initial UAV distance `d` if mobility is modeled
- A bootstrap convention for the very first prediction (there's no `t-1` viewport yet)

**Computed at each `step()`, *before* the agent acts** — i.e., what composes the state `ω^t` the agent actually observes:
- `Z^t` and `Δ^{t-1}_θ, Δ^{t-1}_φ`, carried over from the previous step
- `h^t`, sampled fresh from the channel model this step

**Computed at each `step()`, *after* the agent acts** (consequence of the chosen `x^t, P^t`):
- `E^t(x^t)` — immediate from the action
- `A^t` — via whichever model was chosen in Section 2
- `R^t` — Shannon formula, using `P^t`, `h^t` (already known), `W`, `N0`
- `D^t` — stall-time formula
- `Z^{t+1}` — buffer update (deterministic given the above)
- `r^t_known` — the immediately-known reward component (stall + power + tile-count penalties)
- *(this whole batch is exactly the formulation's post-decision state `ω̃^t`)*

**Computed after observing the random outcome** (real trace data reveals `V^t_actu`; a fresh channel draw gives `h^{t+1}`):
- `Δ^t_θ, Δ^t_φ` — comparing the prediction made for slot `t` against the now-revealed actual viewport
- `C^t_cov`, `Q̄^t_viewport` — coverage/quality
- `r^t_random`, then `r^t = r^t_known + r^t_random`
- `ω^{t+1}` — handed back to the agent

## 5. Gym interface

- `reset()` → initial observation `ω^0` (+ info dict)
- `step(α^t)` → `(ω^{t+1}, r^t, terminated, truncated, info)`
- `observation_space`: a `Box` over `(Z^t, Δ^{t-1}_θ, Δ^{t-1}_φ, h^t)`
- `action_space`: combines discrete/binary tile selection (`MultiBinary(64)` or similar) with continuous power (`Box`) — a harder, mixed action space than MMSP'25's three continuous scalars, consistent with the meeting's plan to discretize/bin power initially
- The PPO agent interacts purely through `reset()`/`step()`, agnostic to whichever `A^t`/R-D decision gets made internally

## 6. Open questions for Jacob

- Reuse the MMSP'25 channel-gain model/parameters as-is, or does Jacob have different/updated UAV-specific values?
- Model UAV mobility now, or start with the static-distance special case?
- Which QP index(es) should represent "base" vs "enhanced" quality?
- Adopt `V_θ=60°, V_φ=30°`, or track down the dataset capture rig's actual documented FOV?
- Refit R-D curves from raw data, or use discrete QP-level lookups?
- How many/which of the 15 videos to train on initially — just `Runner` (matching MMSP'25's own experiments), or across the full set?

## Remaining gaps even after this pass

- The exact yaw/pitch/roll sign and range convention relative to `(θ,φ)` still needs to be checked directly against `hn.mat`'s value ranges before any geometry code is written.
- No source (dataset, either paper) gives a per-video-confirmed 8×8 tile grid — only `Runner` is confirmed explicitly; the other 14 videos are assumed identical by pipeline consistency, not independently verified.
- MMSP'25's reward is materially simpler than the problem formulation's (binary outage vs. continuous coverage, no tile-budget term) — nothing here validates that the formulation's richer reward will train as cleanly; that's an empirical question for later.
