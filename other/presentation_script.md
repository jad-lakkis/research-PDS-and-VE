# Presentation Script — `presentation_review_and_defense.tex`

One entry per slide, in order. "Say" is what to talk through. "Show" is exactly what to pull up on screen when moving to a source paper — paper name, section/page, and the literal text to point at.

---

**Slide 1 — Formulation recap**
*Say:* This is the constrained problem from last time, unchanged — maximize discounted viewport quality subject to three budget constraints on stall, power, and enhanced-tile fraction. All three budgets were calibrated experimentally, not assumed. This is solved with Lagrangian relaxation on top of PPO — PPO itself has no constraint-handling ability. Move fast here, this is a recap.

---

**Slide 2 — Lagrangian reward, multiplier update, why once per rollout**
*Say:* The reward PPO actually sees folds the three constraints in as adaptive penalty weights. The multiplier update happens once per rollout — 2,048 steps — while PPO itself does 320 gradient updates on that same data in between (10 epochs, 32 minibatches each). This isn't arbitrary: the constrained-RL literature requires the multiplier to move more slowly than the policy for the mechanism to converge correctly.
*Show:* Stooke, Achiam & Abbeel (2020), page 2, Section 2 "Related Work," paragraph "Constrained Deep RL." Point to the sentence: *"Convergence proofs have relied upon updating the multiplier more slowly than the policy parameters (Tessler et al., 2018; Paternain et al., 2019), implying many constraint-violating policy iterations may occur before the penalty comes into full effect."* This is the citation that defends the once-per-rollout choice.

---

**Slide 3 — Stooke et al.: what it actually says, and a correction**
*Say:* Two things from this paper map directly onto what we did. First, their true cost objective is an infinite-horizon expectation, but their actual algorithm never computes that — it uses a sample estimate instead, exactly like our rollout-averaged $\overline{J_j}$. Second, when they actually implement this on top of PPO for their real experiments, they explicitly switch to non-discounted, finite-episode costs — the same simplification we made, and for the same reason: their episodes have finite horizons, so a full Monte Carlo sum is available and there's no need for a cost critic.
Then the correction, said plainly: I initially thought their "sample a minibatch" step in the general algorithm was equivalent to our rollout, and that this by itself justified updating the multiplier once per rollout. That's not quite right — in their Algorithm 1, the multiplier and the policy update happen together, every minibatch — that's a *tighter* coupling than ours, not looser. The correct citation for "update the multiplier slower than the policy" is the Related Work sentence on the previous slide, not this minibatch structure.
*Show:*
1. Page 3, Section 3 "Preliminaries," Eq. 5: $J_C(\pi)=\mathbb{E}_{\tau\sim\pi}[\sum_{t=0}^\infty\gamma^t C(s_t,a_t,s_{t+1})]$ — the infinite-horizon definition.
2. Page 5, Section 5.2, Algorithm 1, line "Store sample estimate $\hat J_C$ into $\mathcal{J}_C$" — point at the hat notation, that's the estimator, not the true value.
3. Page 7, Section 6.2 "Algorithm: Constraint-Controlled PPO," the sentence: *"The environments' finite horizons allowed use of non-discounted episodic costs as the constraint and input to the controller."*

---

**Slide 4 — Consequences of λ=0.01**
*Say:* Two separate consequences of the same discount choice, and they need to be kept distinct. First: because the weights collapse so fast — 1, then 0.01, then 0.0001 — steps 2 through 35 of every episode contribute almost nothing, so our discounted quantities are really first-step statistics, not whole-episode ones. Second, and separately: PPO's own internal discount is 0.99, not 0.01, so PPO's gradient is actually targeting a different discounted objective than the one the multipliers are built from. RCPO allows using a mismatched "guiding" discount, but only under a specific condition — that every policy reachable under the mismatched discount also satisfies the true constraint. We haven't verified that holds here, and it would be hard to demonstrate in this environment. So right now, this is an approximate, not exact, implementation of the stated equations.
*Show:* If asked for the RCPO condition specifically: Tessler, Mankowitz & Mannor (2018/2019), Theorem 2 (page 5 of that paper, not attached here — mention it by name if pressed, but the primary source open on screen for this slide should stay Stooke et al. since that's the attached PDF).

---

**Slide 5 — Rollout/episode boundary misalignment**
*Say:* This one is entirely our own finding, verified directly against the real training log, not from either paper. Since episodes are always exactly 36 steps and a rollout is 2,048 steps, the two don't divide evenly — the GCD is 4, which produces an exact, repeating 9-rollout cycle of episode counts. I checked this predicted pattern against all 342 real rollouts of the 700K run and got zero mismatches. Practically, this means about 89% of rollouts contain one episode that started under the previous rollout's policy and finished under the current one — so roughly one in every 56 or 57 episodes feeding a multiplier update is a mixed-policy sample, not a clean one. Small, but real and precisely quantified.
*Show:* Nothing external — this is our own log analysis, no paper reference needed.

---

**Slide 6 — Training setup recap**
*Say:* Quick context before results: Runner video, 12 traces, 9 for training and 3 held out and reused for every validation check. Observations are normalized online with VecNormalize, frozen once we move to validation. Checkpoint selection is feasibility-first. Move through this fast.

---

**Slide 7 — Results: progression**
*Say:* The headline result. 50K and 300K never reach a fully feasible checkpoint. 700K does — and stays feasible for the last 20 checkpoints in a row, not just once.

---

**Slide 8 — Results: trade-offs**
*Say:* Coverage drops about 5% while stall, power, and tile usage all drop sharply — that's a real trade-off, not noise. $\mu_D$ and $\mu_B$ keep rising, meaning those two constraints are still actively exerting pressure. $\mu_P$ collapses to near zero, meaning the power constraint became slack — the policy found a power level that satisfies it comfortably and stopped needing to be pushed.

---

**Slide 9 — Issues summary**
*Say:* Don't read the whole table. Name two or three — the discount mismatch, the rollout/episode misalignment, and untouched PPO hyperparameters — then point to the full write-up for the rest.
*Show:* `other/implementation_review_and_ppo_questions.md` if asked for detail on any specific row.

---

**Slide 10 — Trade-offs behind the modeling choices**
*Say:* Each of these four is a genuine either/or, not a mistake to fix. Keep this fast, one sentence per bullet.
*Show:* If CPO or Lyapunov methods come up in discussion, these are Achiam et al. 2017 and Chow et al. 2018 respectively — cited on the references slide, not otherwise attached here.

---

**Slide 11 — PPO vs. PPO+PDS+VE comparison plan**
*Say:* This is the plan for the next stage, not a result — no claim yet that PDS+VE is better. The important part to say out loud is the fairness constraint: virtual experience can add a lot of gradient updates without adding real environment interactions, so wall-clock time and update count have to be reported separately from the primary, real-interaction-based comparison, or the comparison isn't fair.

---

**Slide 12 — References**
*Say:* Move through quickly — this is here so every citation used earlier in the talk is traceable.

---

**Slide 13 — Questions**
*Say:* These are the concrete decision points to leave the meeting with. Several will already have come up naturally during the earlier slides — that's fine, this slide is here to restate them explicitly so nothing gets lost. If time is short, prioritize the λ/γ reconciliation question and the power-constraint question — those are the two with the most direct bearing on what to do before starting the PDS+VE comparison.
