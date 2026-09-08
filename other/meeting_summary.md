# Meeting Summary — Progress Since Last Report

Covers: replies to the professor's 5 email comments (high-level + detail), how training actually works and why (with sources), the new literature check on the once-per-rollout multiplier update, every issue/limitation found, and the real trade-offs behind the modeling choices. Written to be presentable as-is; a slide deck can be built from this afterward.

---

## 0. Executive summary

- The constrained-PPO baseline works: the 700K-step run reached full, sustained constraint feasibility (last 20/20 validation checks feasible), with a real, expected coverage-vs-resource trade-off.
- All 5 of the professor's comments have concrete answers now (below) — one was a genuine wording error (QP values) now identified and fixable; the rest were requests for comparisons/clarifications now scoped into concrete next steps.
- A follow-up self-review (not prompted by the email, done afterward) found 6 additional precision issues in the constrained-RL mechanism itself — most notably, PPO's own discount (0.99) doesn't match the constraint discount (0.01), so the implementation is currently an *approximation*, not an exact solver, of the written equations. All are logged with root causes, not just symptoms.
- New: checked whether "update the Lagrange multiplier once per rollout" can be defended against a second paper (Stooke, Achiam & Abbeel, ICML 2020, PID Lagrangian methods). Verdict below — partially confirms the intuition, but the precise supporting citation is different from what was guessed, and I corrected it.

---

## 1. Replies to the professor's 5 comments

### Comment 1 — QP values
**High-level:** He's right. Fixed.
**Detail:** We reported "QP=6" / "QP=1" as if these were absolute encoder QP values. They're actually *array indices* — positions 6 and 1 in the dataset's own list of 7 quality representations (index 0 = best/highest-bitrate, index 6 = worst/lowest-bitrate). The dataset's README never discloses the actual absolute QP numbers behind those 7 positions — confirmed by reading it directly. Our own original exploration notebook independently used "QP_idx_0..6" naming from the start, consistent with this. The implementation was never wrong; the report's wording was. If absolute values are needed, they'd have to come from the original dataset paper (Chakareski et al., ACM MMSys 2021), which we have not read.

### Comment 2 — Normalization comparison + contact X
**High-level:** Not done yet. Only VecNormalize has been run. Reaching out to X.
**Detail:** No fixed-constant run exists to compare against — there's no empirical result to report on this yet, only the reasoning for why VecNormalize was tried first (avoids needing pre-existing data to derive constants from; standard practice for PPO). The professor's own caveat — that the paper's constants come from DDPG, not PPO — is a fair, independent reason they might not transfer even if tried. A separate document (`other/implementation_review_and_ppo_questions.md`) has the full question list prepared for X.

### Comment 3 — What is a rollout?
**High-level:** A rollout is the data-collection phase (fixed step count, current policy frozen), not a count of gradient updates.
**Detail:** 2,048 environment steps collected under the *current*, unchanged policy — no learning happens during collection. Afterward, PPO does multiple epochs (10) of minibatch (batch size 64) gradient updates on that same fixed batch — 320 updates total — before collecting a new rollout with the now-updated policy. His guess ("epochs × iterations per epoch") describes that second, later phase, not the rollout itself.

### Comment 4 — Is this how PPO handles constraints?
**High-level:** No — PPO has zero built-in constraint handling. Lagrangian relaxation is a separate, standard technique layered on top of it.
**Detail:** PPO only ever sees one plain scalar reward, computed as $r_L^t=\beta_q\bar Q^t-\mu_D\text{cost}_D^t-\mu_P\text{cost}_P^t-\mu_B\text{cost}_B^t$; it has no idea a constraint exists. The constraint-handling logic (updating $\mu_D,\mu_P,\mu_B$ based on measured constraint satisfaction) is code we wrote ourselves, entirely outside PPO, and could equally wrap around any other policy-gradient algorithm. This is standard practice, not invented here — foundational theory: Altman, *Constrained Markov Decision Processes* (1999); modern deep-RL version of the same alternating pattern: Tessler, Mankowitz & Mannor, "Reward Constrained Policy Optimization" (RCPO), arXiv 2018 / ICLR 2019. His agreement on $\lambda=0.01$ (only the first few steps matter) is correct and connects directly to a deeper issue found afterward — see Section 3 below.

### Comment 5 — Convergence graphs
**High-level:** Will include next time: reward over iterations, multiplier values over iterations, constraint satisfaction over iterations.
**Detail:** All the underlying data already exists in the training logs (`progress.csv` for each run) — this is a presentation-format gap, not a missing-data gap.

---

## 2. How training works, and why it's built this way (with sources)

**The loop, in order, once per rollout:**
1. Collect 2,048 environment steps under the current policy ($\approx$57 episodes of 36 steps each).
2. Update $\mu_D,\mu_P,\mu_B$ once, using that rollout's completed episodes.
3. PPO does its own policy/value update (10 epochs × 32 minibatches = 320 gradient steps) on that same rollout's data.
4. Evaluate on 3 fixed held-out traces; save the checkpoint if it's the new best.
5. Repeat.

**Why update the multiplier once per rollout, and not more or less often — this is a deliberate choice, not an arbitrary one, and it has two independent literature sources:**

1. **The multiplier must change more slowly than the policy.** This is a stated requirement in the constrained-RL convergence literature, not just our own intuition: *"Convergence proofs have relied upon updating the multiplier more slowly than the policy parameters"* (Stooke, Achiam & Abbeel, 2020, Related Work section, citing Tessler et al. 2018 and Paternain et al. 2019). Updating $\mu$ once per rollout while PPO does 320 gradient steps on the policy in between is a large, deliberate separation in update speed, matching this principle. Honest limitation: the formal convergence proofs behind this principle assume a *decreasing* multiplier learning rate; ours ($\eta=0.01$) is constant, so this is satisfied qualitatively, not with a formal guarantee — true of nearly all applied deep-RL implementations of this theory, not unique to us.

2. **Using a finite, sampled estimate of the constraint cost (instead of the true infinite-horizon expectation) is itself standard practice, confirmed by a second, independent paper.** The CMDP objective is technically defined as an infinite-horizon expectation, $J_C(\pi)=E_{\tau\sim\pi}[\sum_{t=0}^\infty\gamma^t C(s_t,a_t,s_{t+1})]$ (Stooke et al., eq. 5) — nobody computes this exactly. Their own Algorithm 1 explicitly uses a "sample estimate $\hat J_C$" (not the true $J_C$) to drive the multiplier update — the same role our $\overline{J_j}^{\text{rollout}}$ plays. More directly relevant: in their actual PPO-based experiments (CPPO, Section 6.2), they state *"the environments' finite horizons allowed use of non-discounted episodic costs as the constraint"* — i.e. they also depart from the theoretical infinite-discounted-sum definition and use a **finite, per-episode sum** as their real, practical constraint statistic. That is structurally the same simplification we made (exact per-episode Monte Carlo returns, no cost critic, because episodes are short and always complete) — independently confirmed as reasonable practice by a second paper, not just RCPO.

**A correction to a specific idea I want to flag before you present it:** the guess that their Algorithm 1's "sample a minibatch" step is literally equivalent to our "rollout," and that this by itself justifies once-per-rollout multiplier updates, is only partly right. Algorithm 1's generic template actually shows $\lambda$ and the policy updating together, every minibatch — a *tighter* coupling than what we do, not a looser one. The precise, directly-quotable support for "update $\mu$ less often than the policy" is the separate Related Work sentence above (point 1), not the Algorithm 1 minibatch structure. Use that sentence, not the minibatch-equivalence argument, if this needs to be defended precisely.

**One more thing worth a mention, not yet acted on:** this paper's actual headline contribution is that plain ("integral-only") Lagrange multiplier updates — exactly the kind we use — are prone to oscillation and overshoot during training, and they propose adding proportional/derivative terms (PID control) to fix it. We have not checked whether our own $\mu_D,\mu_P,\mu_B$ trajectories show this kind of oscillation (only start/end values and a few sampled points have been examined so far, not the full shape). Worth checking before claiming this isn't a concern for us — can do this next if useful.

---

## 3. Issues and limitations found (self-review, after the email)

**High-level:** everything below is a precision/exactness issue in an already-working baseline, not a broken result. Full detail and root causes for each are in `other/decision_log.md` (#15–#17) and `other/implementation_review_and_ppo_questions.md`.

| Issue | One line | Status |
|---|---|---|
| $J_Q,J_D,J_P,J_B$ dominated by first 1–2 steps | $\lambda=0.01$ makes steps 2–35 contribute <0.01% | Open question |
| PPO's $\gamma$ (0.99) $\ne$ constraint $\lambda$ (0.01) | Implementation is approximate, not exact, relative to the written equations | Deferred, tied to the above |
| Best-checkpoint selection near-arbitrary | Direct consequence of $J_Q$'s tiny range | Noted |
| Rollout/episode boundary misalignment | Confirmed empirically: exact 9-rollout cycle, $\approx$1.7% of episodes per update are policy-mixed | Deferred |
| Single, non-parallelized training environment | No parallel envs tried yet | Not tried |
| Most PPO hyperparameters at untouched library defaults | `n_epochs`, `batch_size`, `gae_lambda`, etc. never validated for this environment | Not validated — this is what X can help with |
| Constant (not decreasing) multiplier learning rate | Formal convergence theory wants a decreasing schedule | Known theory gap |
| Training fps declines substantially over long runs | 67→16 fps over 700K steps, only partially explained | Open |
| VecNormalize vs. paper's fixed constants never compared | Only VecNormalize tried | Not done (Comment 2) |
| Channel gain ($h$) under-scaled by VecNormalize's epsilon | $\approx$6.3× under-scaling, fix identified, not applied | Deferred |
| QP reported as absolute value | Should be "array index" | Wording fix (Comment 1) |
| "Violation" metric is raw, not normalized, despite loose wording | Cosmetic/terminology, not a functional bug | Wording fix |
| Only 3 validation traces, reused everywhere | Not an independent test set | Open |
| Single training seed | No variance/replication estimate | Open |
| Power constraint slack ($\mu_P\to0.009$) | Policy converged to one constant power level | Open design question |

---

## 4. Real trade-offs behind the design choices

- **Coverage vs. stall/power/tiles** (the actual result): coverage fell 0.927→0.882 (−4.9%) while stall (−96%), power (−44%), and enhanced tiles (−51%) all dropped sharply. A genuine quality-for-resource trade-off, not noise.
- **$\gamma_{\text{PPO}}=0.99$ vs. $\lambda=0.01$:** matching them would make PPO's own value/advantage estimation nearly fully myopic (effective horizon $\approx$1 step), likely hurting learning quality generally — but keeping them different means the implementation only approximately solves the stated equations. Neither choice is free.
- **Monte Carlo returns vs. a learned cost-critic:** Monte Carlo is simpler (no extra networks) and exact per episode, but only works because episodes are short and always complete — and it's what makes the $\lambda=0.01$ domination issue visible in the first place. A critic would avoid that domination issue but adds real complexity and is the standard choice specifically *because* most benchmark environments don't have our short, always-completing episodes.
- **VecNormalize (adaptive) vs. fixed constants (paper-style):** adaptive avoids needing pre-existing data to derive constants from, but introduces its own numerical issue (the $h$/epsilon under-scaling) that fixed, hand-picked constants wouldn't necessarily have.
- **Lagrangian relaxation vs. CPO/Lyapunov-based methods:** Lagrangian tolerates constraint violations during training (fine here — this is simulation, not a physical robot); CPO/Lyapunov methods hold constraints during training itself but are substantially more complex to implement and weren't judged necessary for this setting.
- **Discrete power levels, 5 slots across 0–100%:** simple action space, but the trained policy only ever uses one of the 5 slots — an open question of whether coarse-but-wide or fine-but-narrow discretization would surface a more meaningful policy.
