# Which design change leaves room for RL? (rules and oracles only, no training)

All numbers are viewport PSNR on held-out viewers, 7 videos (Runner, GateNight, Bridge, SiyuanGate,
Academic, SouthGate, StudyRoom), simulated with `fastsim.py` (matches the training environment to 1e-14).

- **Literature** = best of the paper's baselines (QER, MMSP average-based, ASL360 threshold), tuned on
  training viewers to meet the budgets (5% safety margin).
- **RL estimate** = a simple adaptive rule (NOT a paper baseline) that ranks extra tiles by expected value
  from the predicted direction and adds them while the link can deliver - used only to estimate what a
  learned policy could reach.
- **Room** = RL estimate - literature (dB). Ceiling = perfect viewport prediction (unreachable).

Full table: `summary_room_for_RL.csv`. Main points:

| Option | Mean room (dB) | Range (dB) |
|---|---|---|
| Current setup | -0.07 | -0.99 to +0.37 |
| Current + agent that knows the recent head-motion direction | **+0.24** | -0.65 to +0.84 |
| Tighter tile budget (7.5) | +0.10 | -0.08 to +0.28 |
| Fixed 25 mW (weaker link) | -0.15 | -0.99 to +0.24 |
| Power budget (3rd constraint)* | +0.12 | -0.16 to +0.51 |
| PSNR objective* | -0.02 | -0.95 to +0.41 |
| PSNR + power budget* | +0.16 | -0.06 to +0.54 |
| Link-limited + choose tiles by value per bit | -0.44 to +0.06 | -1.59 to +0.62 |
| Channel with memory (slow shadowing) | +0.15 to +0.18 | -0.16 to +0.37 |

\* literature tuned on a coarser margin grid; with the finer grid (D0-D2) the literature gained up to
0.7 dB, so these rooms are optimistic.

Conclusions:
1. No budget / objective / power / constraint change creates a large, consistent RL advantage over
   well-tuned literature rules: the mean room stays within about +-0.3 dB.
2. The gap to the ceiling (1-4 dB) is head motion that previous-viewport prediction cannot anticipate;
   no policy with the same information recovers it.
3. The only change with a consistent positive effect is information: telling the agent which way the
   head just moved (direction persists about 70% of the time) - +0.24 dB on average, up to +0.84 dB.
4. A coarse literature margin grid inflated earlier room estimates (e.g. tile budget 7.5 looked like
   +0.66 dB, +0.12 dB once the literature rules got finer margins).

Scripts: `design_options_scan.py`, `motion_room_test.py`, `literature_fine_check.py`,
`channel_memory_test.py`, `bits_aware_test.py`, `fastsim.py`.

## Follow-ups (`popularity_test.py`, `blockage_test.py`, `blockage_robust.py`)

- **Cross-viewer popularity** (where the training viewers looked at the same moment, leave-one-out
  for tuning) + head-motion direction: mean room +0.10 dB (tile budget 8), -0.10 dB (tile budget 10),
  range -0.8 to +0.8 dB. Not a lever.
- **Link blockages** (two-state Markov outages, ~20% of time blocked, ~3 s each, 10 realisations per
  viewer; medium-bitrate videos):

| Video, blockage depth | Literature rule (tuned on training) | Buffer-planning RL estimate | Predicted viewport only |
|---|---|---|---|
| Runner, 10 dB | 52.56 dB, **breaks stall budget** (0.36) | 52.85 dB, within (0.05) | 51.74 dB |
| Runner, 12.5 dB | 52.56 dB, **breaks stall budget** (0.49) | 52.72 dB, within (0.04) | 51.74 dB |
| Runner, 15 dB | 52.45 dB, within (0.21) | 52.81 dB, within (0.12) | 51.74 dB |
| GateNight, 10 / 12.5 dB | 51.95 / 51.68 dB, within | 52.09 / 51.90 dB, within | 51.40 dB |
| GateNight, 15 dB | none within budget on training | 51.68 dB, within | 51.40 dB |
| Bridge, 10 dB | 52.56 dB, within | 52.93 dB, within | 52.54 dB |

  With outages, rules tuned on training viewers can break the stall budget on new viewers, while a
  policy that keeps a buffer reserve before outages stays within budget and gains about +0.2 to +0.4 dB
  over the literature rule (about +1 dB over the predicted viewport). Light videos are unaffected
  (their link has large spare capacity).

## All allowed changes together (`combined_test.py`, `stall_budget_test.py`)

Formulation fixed; agent gets relative tiles + recent head-motion direction; power fixed at 50 mW;
outages 0 / 10 / 12.5 dB; tile budget 8 or 10; stall budget 0.1 / 0.15 / 0.3; 10 channel
realisations per viewer. Mean RL-estimate gain over the best tuned literature rule: +0.2 dB (tile
budget 8), -0.2 to -0.04 dB (tile budget 10). Per video, Runner is the best case: +0.6 to +0.85 dB in
every setting. Rate-matched uniform baseline (same bits, all 64 tiles at one level): RL estimate +3.1 to
+3.6 dB on Runner, +1.1 to +2.6 dB on GateNight/Bridge, but uniform wins on light videos (SiyuanGate,
StudyRoom, Academic). Tighter stall budgets do not add room.

## Is head motion learnable on high-motion videos? (`motion_predictability.py`, `../motion_predictability.csv`)

All 15 videos. Predictors of the next viewport from what an agent knows (past one-second viewing
directions, time in the video, where training viewers looked), trained on training viewers, scored on
held-out viewers by the coverage the forced-tile rule achieves when centred on the prediction.
Best learned/extrapolated predictor minus previous viewport: -0.03 to +0.04 coverage on 14 videos,
+0.08 on Chairlift (10-s clip). Higher motion lowers every method; it does not become predictable.
