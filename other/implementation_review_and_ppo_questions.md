# Implementation Review: Issues, Root Causes, and Questions for X

Consolidated record of everything reviewed since the last progress report — the professor's email comments and a follow-up methodological review of the constraint/Lagrangian mechanism. Organized by root cause, not chronologically, so each issue is traceable to the modeling choice that produced it. Full technical detail and status tracking for each numbered item lives in `other/decision_log.md`; this document is the narrative synthesis, written to be self-contained for an outside reader (X) who has no access to the code.

Verification discipline: every specific number or claim below was either read directly from the source code, computed directly from the actual training logs (`runs/train_*/progress.csv`), or is explicitly flagged as unverified/general knowledge where it could not be checked against a primary source in this project.

---

## 1. What's implemented and working

- Gymnasium environment for the Runner video: 64-tile action mask, discrete transmit-power selection (5 levels), mandatory enhancement of tiles intersecting the predicted viewport, integrating the channel model, buffer/stall dynamics, viewport coverage, and rate-distortion layer model.
- The constrained problem (eqs. 5–8: maximize $J_Q$ s.t. $J_D\le\bar D$, $J_P\le\bar P$, $J_B\le\bar B$) solved via Lagrangian relaxation: adaptive multipliers $\mu_D,\mu_P,\mu_B$ folded into PPO's reward, updated once per rollout by projected dual ascent.
- $\bar D=0.10,\bar P=0.60,\bar B=0.35$ derived experimentally (not assumed), by measuring the exact discounted $J_D,J_P,J_B$ under reference policies on the 9 training traces only.
- PPO (Stable-Baselines3) training with online observation normalization (VecNormalize), validated every rollout on 3 fixed held-out traces, feasibility-first checkpoint selection.
- Result: a 700K-step run reached full, sustained constraint feasibility (final 20/20 validation checks feasible; stall fell 0.531s→0.022s, enhanced tiles 39.1→19.0, power 49.6mW→27.6mW, with a real, expected coverage trade-off, 0.927→0.882).

This is a genuinely working baseline. Everything below is about precision, exactness, and things that haven't been validated yet — not about the baseline being broken.

---

## 2. Issues, grouped by root cause

### A. The discount factor λ=0.01 and its downstream consequences

**A1 — The discounted CMDP quantities are dominated by the first 1–2 steps of each 36-step episode.**
$J_c=\sum_{t=0}^{35}\lambda^t c^t$ at $\lambda=0.01$ has weights $1, 0.01, 0.0001,\ldots$ — steps 2–35 combined contribute under 0.01%. So $J_Q,J_D,J_P,J_B$ are effectively statistics about the first GOP of an episode (an atypical condition: buffer starts at 0, the first viewport prediction is bootstrapped to be perfect), not whole-episode statistics. Root cause: $\lambda=0.01$ was taken directly from the reference paper (MMSP'25, a different algorithm, DDPG) without re-deriving whether it's appropriate for *this* project's literal 36-step discounted-sum formulation. Not a bug — it's mathematically permitted and consistent with our own written CMDP formulation — but it means the constraint, as implemented, is essentially about the episode's first step, which may not be the intended physical behavior. **Open question, not yet resolved.**

**A2 — PPO's own discount ($\gamma=0.99$) does not match the constraint discount ($\lambda=0.01$), so the implementation is only approximate, not exact, relative to eqs. 5–8.**
PPO's value function and advantage estimates (GAE) are computed with $\gamma=0.99$, so PPO's policy gradient is an ascent direction for $\sum_t 0.99^t r_L^t$ — not for the $\lambda=0.01$-discounted Lagrangian ($J_Q-\mu_D J_D-\mu_P J_P-\mu_B J_B$) that the multiplier mechanism is actually built from. These are mathematically different objectives. Root cause: $\gamma=0.99$ was deliberately chosen different from $\lambda=0.01$, because matching them would make PPO's own value/advantage estimation almost fully myopic (effective horizon $\approx1/(1-\lambda)\approx1$ step), likely degrading PPO's learning quality independent of the constrained-RL question. RCPO (Tessler, Mankowitz & Mannor, ICLR 2019) permits a mismatched "guiding" discount under an explicit condition (their Theorem 2: policies reachable under the guiding discount must be a subset of policies satisfying the true constraint) — we have not verified this condition holds, and it would be difficult to demonstrate in this environment. **Deferred; tightly coupled to A1** — reconciling this likely means moving $\lambda$ and $\gamma$ closer together and recalibrating the budgets, not changing either value alone. (Decision log #16.)

**A3 — "Best" checkpoint selection is close to arbitrary among feasible checkpoints, because $J_Q$'s dynamic range is tiny.**
Direct consequence of A1: since $J_Q$ barely varies (1.0063–1.0087 across all 342 rollouts of the 700K run), the rule "among feasible checkpoints, pick the highest $J_Q$" has very little signal to work with. Concretely: the actual selected "best" checkpoint (rollout 192/342) is not the final checkpoint (342), and the final checkpoint is actually slightly better on stall (0.001s vs. 0.067s) and leaner on tiles (17.0 vs. 20.6), at a small coverage cost (84.0% vs. 84.7%) — both are genuinely feasible, but which one "wins" under the selection rule is close to noise. Not a bug in the selection logic — a direct symptom of A1.

### B. PPO mechanics never independently validated (X's area)

**B1 — Rollout size (2,048 steps) is not a multiple of the fixed 36-step episode length, causing rollout/episode boundary misalignment.**
Verified directly against the real 700K log: since $\gcd(2048,36)=4$, this produces a deterministic 9-rollout cycle of episode counts (56,57,57,57,57,57,57,57,57, repeating) — checked against all 342 real rollouts with zero mismatches. Consequence: in $\approx$89% of rollouts, one episode began under the *previous* rollout's policy and completed under the *current* one (the training environment is never reset between rollouts), so its discounted-return contribution to that rollout's multiplier update is a policy-mixed sample, not a clean single-policy Monte Carlo draw. Effect size is real but small ($\approx$1 of 56–57 episodes per update, $\approx$1.7%). Root cause: `n_steps=2048` (a standard PPO default) was never chosen with episode-length divisibility in mind. **Deferred**, not yet fixed. (Decision log #17.)

**B2 — Training uses a single, non-parallelized environment.**
The training stack is `VecNormalize(DummyVecEnv([one env]))` — exactly one environment instance, not multiple parallel workers (e.g. `SubprocVecEnv`). PPO commonly benefits from many parallel environments, both for wall-clock speed and for reducing correlation within a single rollout's data (more diverse trace/channel-seed combinations per rollout). This has not been tried. Confirmed directly from `train_ppo.py`.

**B3 — Most PPO hyperparameters are untouched library defaults, never validated for this environment.**
Only `n_steps` and `gamma` are explicitly set in the code. `n_epochs` (10), `batch_size` (64), `gae_lambda` (0.95), `ent_coef` (0.0), `vf_coef` (0.5), `max_grad_norm` (0.5) are all Stable-Baselines3 library defaults — confirmed by reading the `PPO(...)` call, which passes no override for any of them. Two of the *other* untouched defaults (`learning_rate=3e-4`, `clip_range=0.2`) were directly cross-checked against the real training logs and matched exactly, which is indirect confirmation nothing is silently overriding defaults elsewhere either — but `n_epochs`/`batch_size`/`gae_lambda` specifically are not logged as scalars over training, so they're confirmed by code-reading only, not by a log cross-check. None of these have been tuned or validated for a short-episode (36-step), MultiDiscrete (65-dimensional), single-env setup like this one.

**B4 — The Lagrange-multiplier step size ($\eta=0.01$) is constant, not a formally-required decreasing schedule.**
RCPO's convergence theory relies on a multi-timescale argument (critic fastest, policy intermediate, multiplier slowest), typically requiring a *decreasing* multiplier learning rate satisfying standard stochastic-approximation conditions. Our multiplier updates once per rollout while PPO performs $\approx$320 gradient updates per rollout (10 epochs × 32 minibatches) — qualitatively "slow multiplier, fast policy," matching the spirit of the theory — but $\eta=0.01$ is constant, so the formal convergence guarantee does not strictly transfer. This is true of nearly all applied deep-RL implementations of this theory, not unique to this project, but it means the results demonstrate empirical feasibility, not a formal guarantee.

**B5 — Training throughput (fps) declines substantially and continuously over long runs, not fully explained.**
Measured directly: 67→16 fps over the 700K run (300K run: 71→28; 50K run: 65→54). A verified, real contributor: SB3's CSV logger rewrites the entire output file from scratch whenever a new logging key first appears (confirmed by reading `CSVOutputFormat.write()`), but this should stabilize once all keys have appeared, and does not fully explain a *continuous* decline across the whole run. Not resolved.

### C. Observation normalization

**C1 — Only VecNormalize has been tried; the reference paper's fixed-constant approach has never been implemented or compared.**
The professor explicitly suggested trying both and comparing pros/cons. No fixed-constant run has been done — there is currently no empirical comparison to report, only a justification for why VecNormalize was chosen first (avoids needing a pre-existing dataset to derive constants from; standard SB3/PPO practice). The professor also noted the reference paper's constants come from a DDPG setup, not PPO, so they may not transfer even if tried.

**C2 — A specific, already-diagnosed numerical precision issue in VecNormalize: the channel-gain component is under-scaled.**
Verified against the actual trained model's learned statistics: channel gain $h$'s true variance ($\approx2.5\times10^{-10}$) is dominated by VecNormalize's default $\epsilon=10^{-8}$ (97.5% of `var+epsilon`), causing $h$ specifically to be under-scaled by a measured $\approx$6.3$\times$ relative to correct normalization. The other three observation components do not have this problem (their variances vastly exceed $\epsilon$). Fix identified (pre-scale $h$ by a fixed physical constant, e.g. $\times10^7$, before normalization) but not yet applied — requires retraining once applied. (Decision log #15.)

### D. Reporting/terminology precision (not code bugs)

**D1 — Reported "QP=6" / "QP=1" as if they were absolute encoder QP values; they are dataset array indices.**
The dataset's rd.mat labels its quality axis "QP" with 7 positions (0–6) but never discloses the actual absolute encoder QP numbers behind them — confirmed by reading the full dataset README. What was actually used is the *ordinal position* in that list (index 6 = worst/lowest-bitrate, index 1 = second-best), consistent with this project's own original exploration notebook, which independently used "QP_idx_0..6" naming from the start. The underlying work was always correct; the report's wording implied absolute values, which it should not have. The real absolute QP numbers, if they exist, would be in the original dataset paper (Chakareski et al., ACM MMSys 2021), which has not been read in this project.

**D2 — Checkpoint-selection "violation" is a raw sum of excesses, not a normalized quantity, despite sometimes being described loosely as "normalized."**
The actual computation is $\sum_j\max(0,J_j-\bar b_j)$ — summed directly, not divided by each budget's scale. Functionally fine for ranking checkpoints (still monotonically meaningful), but the word "normalized" has been used imprecisely in places and should be corrected to "raw combined violation."

### E. Evaluation protocol scope

**E1 — Only 3 held-out traces exist, reused for every checkpoint evaluation across every run — this is a validation set, not an independent test set.**
No separate final-test protocol exists yet. With only 12 valid traces total (9 train / 3 held out), eval-metric variance is dominated by *which* traces got held out, not something more validation episodes can fix.

**E2 — Only one training seed has been used; no replication/variance estimate exists.**
All reported numbers (50K/300K/700K) come from single training runs. Seed-to-seed variance is currently unknown.

### F. Open design questions (not bugs — unresolved choices)

**F1 — The power constraint is slack; unclear whether to tighten the budget or change the action space.**
$\mu_P$ decayed from 1.00 to $\approx$0.009 over the 700K run, and the trained policy converged to a single, constant power level (25 mW = level 1 of 5) regardless of state — verified by direct inference on the trained policy, zero variation across 216 evaluated steps. Open question: tighten $\bar P$, or narrow the discrete power range (e.g. 0–50% instead of 0–100%, same 5 slots) so there's actual resolution where the policy operates?

---

## 3. Questions prepared for X

Scoped specifically to PPO mechanics/parameter-setting, since that's the area the professor said X could help with. The dataset/reporting/evaluation-scope issues above (sections D and E) aren't included here — they're not PPO questions.

> Hi [X's name],
>
> [Professor] suggested I reach out to you — I'm running constrained PPO (Stable-Baselines3) on a custom Gymnasium environment for 360° video tile/power selection, and a recent methodological review surfaced a few PPO-specific issues I'd value your take on. Quick context: episodes are always exactly 36 steps (fixed-length, always terminate naturally), the action space is a 65-dimensional MultiDiscrete (64 binary tile flags + 1 power-level choice among 5), and I'm using PPO's reward to carry a Lagrangian-relaxed constrained objective (three adaptive penalty weights, updated once per rollout via dual ascent, on top of an otherwise-standard PPO reward).
>
> 1. **Discount factor mismatch.** My constrained-optimization formulation defines the objective/constraints with a discount factor $\lambda=0.01$ (taken from a reference paper), but PPO's own `gamma` is set to 0.99 — deliberately different, because matching them would make PPO's value function extremely myopic (effective horizon $\approx$1 step). I understand this means PPO isn't exactly optimizing the same discounted quantity my constraints are defined over. In your experience, is it more standard to (a) live with this mismatch and treat $\gamma=0.99$ purely as a "guiding" discount for learning stability, or (b) actually try to bring $\lambda$ and $\gamma$ closer together (recalibrating everything else that depends on $\lambda$) for a cleaner theoretical match? Is there a practical rule of thumb for how much mismatch is tolerable?
>
> 2. **Rollout length vs. episode length.** My PPO rollout is 2,048 steps; episodes are always exactly 36 steps, so 2,048 isn't a multiple of 36 and episodes straddle rollout boundaries (confirmed: about 1 in 56–57 episodes per rollout is "split" across a policy update). Is choosing `n_steps` as a multiple of the episode length something you'd normally do as standard practice, or is this a non-issue in your experience?
>
> 3. **Single environment, no parallelization.** I'm training with exactly one environment instance (`DummyVecEnv` with one env), not multiple parallel workers. How much does PPO typically benefit from parallel environments in your experience — is this something you'd consider close to mandatory for good PPO performance, or a nice-to-have?
>
> 4. **Untouched hyperparameters.** Aside from `n_steps` and `gamma`, everything else is at Stable-Baselines3's library defaults: `n_epochs=10`, `batch_size=64`, `learning_rate=3e-4`, `clip_range=0.2`, `gae_lambda=0.95`, `ent_coef=0.0`, `vf_coef=0.5`. Given short (36-step) episodes and a 65-dimensional MultiDiscrete action space, do any of these strike you as obviously worth revisiting first?
>
> 5. **Constant vs. decaying learning rate for an adaptive constraint-penalty weight.** I update three Lagrange multipliers once per rollout using a constant step size (0.01), rather than a formally-required decreasing schedule. Have you seen constant step sizes work fine in practice for this kind of setup, or is this a common source of instability?
>
> 6. **Declining training speed.** Frames-per-second drops substantially over a long run (e.g. 67→16 fps over 700K steps) — I've partially traced part of this to the logger, but not the full decline. Is this a pattern you've run into with long PPO runs, and if so, what's usually the actual cause?
>
> Thanks a lot for your time — happy to share code/logs if useful.

---

## 4. Summary table

| # | Issue | Cluster | Status | Log ref |
|---|---|---|---|---|
| A1 | $J_Q/J_D/J_P/J_B$ dominated by first 1–2 steps | Discount factor | Open question | Stage 7 #4 |
| A2 | $\gamma_{\text{PPO}}\ne\lambda$ — approximate, not exact, solver | Discount factor | Deferred | #16 |
| A3 | Best-checkpoint selection near-arbitrary among feasible ones | Discount factor (consequence of A1) | Noted, not actioned | — |
| B1 | Rollout/episode boundary misalignment (confirmed, quantified) | PPO mechanics | Deferred | #17 |
| B2 | Single, non-parallelized training environment | PPO mechanics | Not tried | — |
| B3 | Most PPO hyperparameters at untouched defaults | PPO mechanics | Not validated | — |
| B4 | Constant (not decreasing) multiplier learning rate | PPO mechanics | Known theory gap | — |
| B5 | FPS decline over long runs, partially unexplained | PPO mechanics | Open | — |
| C1 | VecNormalize vs. paper's fixed constants never compared | Normalization | Not done | — |
| C2 | $h$ under-scaled by VecNormalize's epsilon (~6.3×) | Normalization | Deferred, fix identified | #15 |
| D1 | QP reported as absolute value; actually array index | Reporting | Wording fix needed | — |
| D2 | "Violation" is raw, not normalized, despite loose wording | Reporting | Wording fix needed | — |
| E1 | 3 validation traces reused; no independent test set | Evaluation scope | Open | — |
| E2 | Single training seed; no replication | Evaluation scope | Open | — |
| F1 | Power constraint slack; tighten $\bar P$ or narrow action range? | Open design question | Undecided | — |
