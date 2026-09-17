"""
Project configuration - single source of truth for every parameter,
convention, and modeling decision made so far.

Every value below is either:
  (a) SOURCED - cited in its comment (empirical finding in
      explore_data.ipynb, a section of the MMSP'25 paper, or one of
      the professor's direct answers), or
  (b) Explicitly None / TBD, with a comment explaining exactly what is
      still missing and why it wasn't guessed.

Nothing here is invented without a note saying where it came from or
why it's still open. Background/derivations: other/research_summary.md
and other/environment_design.md.
"""

# =====================================================================
# STILL OPEN - not yet answered by the professor at all (raise in a
# follow-up)
# =====================================================================
# 1. D_BAR's specific candidate values to sweep (see D_BAR_CANDIDATES
#    below - the METHOD is confirmed, the actual numbers are not).
#
# (Yaw/pitch/roll convention was resolved this session - mostly
# empirically confirmed, with one explicit flagged ASSUMPTION for yaw
# direction. See the "Field of view / viewport" section below.)


# =====================================================================
# Dataset
# =====================================================================


DATA_DIR = "data"
HN_MAT_PATH = f"{DATA_DIR}/hn.mat"
RD_MAT_PATH = f"{DATA_DIR}/rd.mat"
README_PATH = "other/Readme - Dataset info.txt"   # moved into other/ by the user

# All paths above are relative to the project root - every script in
# this project (config.py, streaming_rl/*, sanity_check.py, etc.) is
# expected to be run FROM c:\Users\NANO\Desktop\prjct_1, matching how
# explore_data.ipynb already does it.

N_VIDEOS = 15                    # confirmed: Readme - Dataset info.txt / video_catalog
N_TILES = 64                     # confirmed: r.dmat array shapes; MMSP'25 Sec. II.B
TILE_GRID_COLS = 8               # confirmed: MMSP'25 Sec. II.B, stated explicitly for Runner
TILE_GRID_ROWS = 8               # (not independently verified for the other 14 videos)
TILE_WIDTH_DEG = 360.0 / TILE_GRID_COLS    # 45.0
TILE_HEIGHT_DEG = 180.0 / TILE_GRID_ROWS   # 22.5

# How the dataset's flat tile index (0..63) maps to a (row, col)
# position in the 8x8 grid. NOT documented anywhere in the dataset,
# the README, or either paper - no tile-index-to-position mapping is
# given at all. ASSUMED, NOT VERIFIED: standard raster (row-major)
# order, row = index // 8, col = index % 8 - the conventional default
# for such grids (also how HEVC tiles are typically ordered). Flagged
# so it can be corrected if a real mapping ever turns up.
TILE_INDEX_ORDER = "raster_row_major"

GOP_SIZE_FRAMES = 30             # empirically confirmed for all 15 videos
                                  # (explore_data.ipynb, Section C: occupied RtoD_exp
                                  # GOP-slots == ceil(frames/30) for every video)
N_QP_LEVELS = 7                  # confirmed: array shape (7 x 64 x frames) in rd.mat

# Training video for now. The professor: "Train and evaluate on one
# video for the time being." Confirmed this session: Runner - it's the
# video MMSP'25 itself uses, and its real QP_idx_6 whole-panorama
# bitrate (4.41 Mbps, computed directly from rd.mat) matched MMSP'25's
# stated A_base=4.5Mb almost exactly (see BASE_QP_ARRAY_INDEX below).
TRAINING_VIDEO_NAME = "Runner"
TRAINING_VIDEO_ARRAY_INDEX = 4   # 0-based index into rd.mat / hn.mat arrays


# =====================================================================
# Base / enhancement-layer (QP) model
#
# The professor's answer (verbatim): "the signal/tile representation
# for the highest QP value would be its base layer (lowest rate,
# highest distortion), then adding k number of enhancement layers
# would represent selecting QP index k (from highest to smallest),
# where data rate for the enhancement layer would be the rate of the
# tile for QP index k minus the data rate for the base layer."
# =====================================================================
# Empirically confirmed direction of the dataset's own QP axis
# (explore_data.ipynb, Section D): array index 0 = highest bitrate /
# best quality; array index 6 = lowest bitrate / worst quality. This
# means the professor's "highest QP value" (= base layer) is array
# index 6. Double-checked this session against a specific contradiction
# risk ("lowest QP as base"): confirmed the base is still the highest-
# QP-value / array-index-6 end - consistent with the A_base=4.5Mb match
# (Runner's real array-index-6 total is ~4.4 Mbps; array-index-0 is
# ~1.08 Gbps, which would not match A_base at all).
BASE_QP_ARRAY_INDEX = 6

# k = number of enhancement layers added on top of base, k in [0, 6].
#   k=0 -> base itself      -> array index 6 (highest QP value)
#   k=6 -> best quality     -> array index 0 (lowest QP value)
# Confirmed mapping: array_index = BASE_QP_ARRAY_INDEX - k.
def qp_array_index_for_enhancement_level(k: int) -> int:
    """Map an enhancement level k (0..6) to the dataset's QP array index."""
    assert 0 <= k <= 6, f"k must be in [0, 6], got {k}"
    return BASE_QP_ARRAY_INDEX - k


# CONFIRMED THIS SESSION: only 2 levels are used - base (k=0) and one
# enhancement level - not the full k=0..6 graduated range yet. Matches
# the start-simple approach used elsewhere (discrete actions first,
# one critic first, etc.). The mapping function above already supports
# the full range for when this gets extended later.
#
# k=5 (array index 1, "one step below best quality") chosen empirically
# by experiments/el_selection_test.py over k=6 (array index 0, "best
# quality"), run under eq. 2's mandatory predicted-viewport enhancement
# (streaming_rl/environment.py): at k=6, even the most conservative
# policy (mandatory tiles only, nothing extra) stalled 97.5-100% of the
# time at every power level tested - too data-heavy to ever succeed,
# nothing for an agent to learn regardless of skill. At k=5, the
# efficient (mandatory-only) policy stalled 0% at mid/max power while a
# wasteful policy (all 64 tiles enhanced) still stalled 100% even at max
# power - a real, learnable gap between good and bad policies.
INITIAL_ENHANCEMENT_LEVELS = (0, 5)   # k values used in training

# A^t = A_base^t + sum_i x_i^t * delta_A_i^t                  (eq. 12 form)
# delta_A_i^t = A_i[selected QP] - A_i[base QP]   (professor-confirmed: "well done")


# =====================================================================
# Field of view / viewport
# Reused as-is from MMSP'25 (the professor: "You can also use the same
# parameters for FOV, mobility/distance, and the rest.")
# =====================================================================
V_THETA_DEG = 60.0    # half-width azimuth   -> full FOV width  = 120 deg
V_PHI_DEG = 30.0      # half-height altitude -> full FOV height = 60 deg

# Actual viewport rule (MMSP'25 Sec. II.C):
#   V_actu = {(theta,phi): theta_tilde - V_THETA <= theta <= theta_tilde + V_THETA,
#                            phi_tilde  - V_PHI   <= phi   <= phi_tilde  + V_PHI}
#
# Yaw/pitch/roll -> (theta, phi) convention, checked this session
# directly against all 194 valid hn.mat traces (147,900 samples):
#   - Yaw range: EMPIRICALLY CONFIRMED [-180, 180] degrees (44.3% of
#     real values are negative; a [0,360) convention would make that
#     impossible). min=-180.01, max=180.00 observed.
#   - Pitch/roll units: EMPIRICALLY CONFIRMED degrees, not radians
#     (pitch bounded to ~[-90, 88], roll to ~[-112, 134] - both far
#     beyond pi or pi/2, which radians would be capped near).
#   - Yaw DIRECTION (does positive yaw mean turning right or left?):
#     NOT verifiable from the data alone (range/sign stats don't reveal
#     direction), and NOT confirmed by the likely capture tool's
#     (OpenTrack) own official documentation (checked directly - it
#     does not state this). ASSUMED, NOT VERIFIED: positive yaw =
#     rightward turn (the common compass-bearing-style convention).
#     Flagged explicitly so this can be corrected later if it proves
#     wrong (e.g. once there's a way to visually cross-check a known
#     head movement against a specific trace). This matters for
#     correctness, not just cosmetics: getting it backward would
#     silently mirror the computed "actual viewport" tiles relative to
#     where the user really looked, corrupting the reward signal
#     without causing any error.
#   - Roll: dropped from the viewport model entirely, following
#     MMSP'25's own approach (only theta/phi, 2 DOF, are used).
YAW_RANGE_DEG = (-180.0, 180.0)          # empirically confirmed
ANGLE_UNITS = "degrees"                  # empirically confirmed (yaw, pitch, roll)
YAW_POSITIVE_DIRECTION = "right"         # ASSUMED, NOT VERIFIED - see note above

# Partial tile visibility: CONFIRMED this session - a partially-covered
# tile is treated CONTINUOUSLY, weighted by the fraction of its area
# that overlaps the viewport - not classified as a binary fully-
# visible/invisible tile (rules out any-overlap and fixed-threshold
# approaches). The overlap-fraction computation itself lives in
# viewport.py; this only records the choice made.
#
# Separately, predicted-viewport tiles are FORCED to mandatory-enhanced
# status regardless of this mode (eq. 2 - see
# viewport.mandatory_tile_mask / environment.py::step): a tile is the
# smallest transmittable unit, so "any nonzero overlap forces the whole
# tile" is a different, binary decision from how much that tile's
# *coverage/reward* contribution is weighted here.
PARTIAL_TILE_VISIBILITY_MODE = "area_weighted"


# =====================================================================
# Communication / channel parameters
# Reused as-is from MMSP'25 (the professor, answer 1: "We can use the
# existing channel model.") "and the rest" (answer 3) is read as
# covering these too, since they were already grouped together in the
# question asked - flagged here in case that reading turns out to be
# wrong.
# =====================================================================
P_MAX_DBM = 20.0
W_HZ = 10e6              # 10 MHz
T0_SEC = 1.0             # time slot length; also the GOP duration
N0_DBM_PER_HZ = -174.0

# UAV mobility: static case (the professor, answer 3), not the full
# Gauss-Markov mobility model (MMSP'25 eqs. 2a-2c).
UAV_DISTANCE_M = 50.0

# Rician-fading + path-loss channel gain model (MMSP'25 Sec. II.D):
#   h = CHANNEL_CONST * |psi|^2 / (UAV_DISTANCE_M * atan(ANTENNA_BEAMWIDTH_ARG))^2
CHANNEL_CONST = 3.24e-4                        # includes -38.47dB path loss + 2.2846dB antenna gain
ANTENNA_BEAMWIDTH_ARG = 24 * (2 ** 0.5) / 5     # atan(...) of this value = antenna beamwidth
RICIAN_K_FACTOR_DB = 12.0
RICIAN_E_PSI_SQUARED = 1.0                      # E[|psi|^2] = 1 (normalized)

# Fixed pre-scale applied to h ONLY inside the observation returned by
# TileStreamingEnv._observation() (streaming_rl/environment.py) - NOT to
# the physical h used everywhere else (channel_model.transmission_rate,
# etc., which stays in raw units). Verified against the actual 700K
# run's learned obs_rms (decision_log.md #15): h's true variance
# (~2.54e-10) was dominated (97.5% of var+epsilon) by VecNormalize's
# default epsilon=1e-8, causing ~6.3x under-scaling of h specifically -
# Z/Delta_theta/Delta_phi don't have this problem (their variances vastly
# exceed epsilon). Multiplying by 1e7 before VecNormalize sees it brings
# h's variance back above epsilon, same idea as the standard "fixed
# pre-scale + adaptive normalization" pattern (e.g. pixel/255 then
# further normalization in vision models). Purely a numerical-precision
# fix - does not change what h means physically, only its scale inside
# the one place (the observation vector) that feeds VecNormalize.
H_OBSERVATION_PRESCALE = 1e7


# =====================================================================
# Reinforcement-learning setup
#
# The professor's answer (verbatim): "We will first implement a
# standard PPO baseline with one critic, V(s_t). After validating the
# baseline, we will introduce post-decision states and add a second
# critic, V(s~_t), following the previous work."
# =====================================================================
DISCOUNT_FACTOR_LAMBDA = 0.99   # Stage 8: changed from the paper's literal 0.01 (MMSP'25 Sec. IV) to
# match PPO_GAMMA below. Two independently-logged issues fixed by one change (decision_log.md #16,
# and the J_Q-near-1 visibility problem): at 0.01, Sum_{t=0}^{35} lambda^t ~= 1.01, so J_Q/J_D/J_P/J_B
# were >99% determined by steps 0-1 alone (steps 2-35 combined contributed <0.01%) - both nearly
# uninformative as a training signal AND numerically dominated by an atypical first-GOP condition
# (Z^0=0, perfect bootstrap prediction). At 0.99, Sum_{t=0}^{35} 0.99^t ~= 30.4, and step 35 alone
# still carries weight 0.99^35~=0.70 relative to step 0 - all 36 steps become meaningfully weighted.
# Separately, this also closes the PPO-gamma-vs-constraint-lambda exactness gap (decision_log.md #16):
# PPO's own gradient targets Sum_t PPO_GAMMA^t * r_L^t, which is now the EXACT same discounted
# quantity the Lagrangian/multiplier updates are built from, since PPO_GAMMA == DISCOUNT_FACTOR_LAMBDA
# - RCPO's Theta_gamma subset-of-Theta condition (needed to justify a mismatched "guiding" discount)
# becomes trivially true rather than an unverified assumption. D_BAR/P_BAR/B_BAR below were
# recalibrated under this new value via experiments/constraint_calibration.py (not carried over from
# the old 0.01 calibration, which measured a numerically different quantity).
ALGORITHM = "PPO"               # per the original meeting plan

# CONFIRMED: two-stage critic plan, not a single fixed choice.
#   Stage 1 (baseline): one critic estimating V(s_t)  - the regular state.
#   Stage 2 (after the baseline is validated): add a second critic
#     estimating V(s~_t) - the post-decision state - once PDS is
#     introduced, following the prior related work.
N_CRITICS_BASELINE = 1
N_CRITICS_WITH_PDS = 2

# Stage 2 critic's own hidden-layer sizes (streaming_rl/pds_policy.py) -
# fully separate small MLP, no shared trunk with the actor/ordinary critic
# (PDS design plan, Section 6). NOT copied from the EHS paper's own
# [64,64] PDS-critic number: their pre-decision/PDS state dimensionality
# ratio isn't known to match ours, and [64,64] on our own 68-dim PDS input
# (64 tile-mask flags + 4 continuous) would make the first hidden layer a
# contraction (68->64) instead of an expansion - unlike the ordinary
# critic's very generous 4->64 first layer. Widened to 96 so the first
# layer expands the input instead, without over-provisioning: most of the
# 68 dims are a sparse/structured binary tile mask (low actual entropy
# relative to its raw width), so a large jump to 128 (EHS's OTHER,
# pre-decision-critic number) isn't clearly justified either - confirmed
# with the user.
PDS_CRITIC_ARCHITECTURE = [96, 64]

# Entropy coefficient for PPOWithPDS's actor loss (streaming_rl/pds_ppo.py
# train(), the existing ent_coef*entropy_loss term stock PPO already has -
# just never given a nonzero value before now). Diagnostic finding (RunPod
# sweep, 4 arms): under the one-step PDS advantage, policy entropy collapsed
# 5-15x faster than the matched non-PDS baseline (which runs at PPO's
# default ent_coef=0.0 the whole time, unchanged) - e.g. PDS entropy at
# rollout ~100-300 already at the level baseline reaches around rollout
# ~1500-2000. Root cause (not yet certain, being tested): the one-step PDS
# advantage has lower variance than baseline's gae_lambda=0.95 once both
# critics are reasonably well-fit (confirmed via explained_variance~0.8),
# producing a more consistent/confident gradient every update and thus
# faster entropy collapse - plausibly leaving too little residual
# exploration for the policy to keep responding to the still-rising
# constraint multipliers (mu_D specifically stayed persistently violated
# in multiple arms despite mu_D correctly climbing).
#
# NOT copied from EHS's Table I entropy coefficient (alpha=0.01, their SAC
# temperature): that number doesn't transfer 1:1 - SAC's alpha enters both
# the actor loss AND the soft critic's bootstrap target, while PPO's
# ent_coef only touches the actor loss, and EHS's action space is ~9
# discrete choices vs. our 2^64*5 (plausibly needing more, not the same,
# exploration pressure). 0.05 is a first working value picked given the
# unusually severe (5-15x) collapse observed - confirmed with the user,
# who is running it alongside 0.03/0.08 as a small sweep, not a single
# committed choice. Overridable per run via train_ppo_pds.py's --ent-coef.
PDS_ENTROPY_COEF = 0.05

ACTION_MODE = "discrete"        # start discrete/binned (per the original meeting)
                                 # before moving to continuous

# How many discrete power levels P^t is binned into. NOT sourced from
# anywhere - a reasonable, arbitrary starting choice for "binned" power,
# same status as the reward weights above.
N_POWER_LEVELS = 5

# Stage 8 (decision_log.md, power-constraint action-space narrowing):
# fraction of P_max the top power level maps to. Originally 1.0 (levels
# spanning 0/25/50/75/100% of P_max) - the trained 700K policy converged
# to a CONSTANT choice at level 1 (25% of P_max) on 100% of evaluated
# steps, verified by direct inference, and never used anything above it.
# Narrowed to 0.5 so the same 5 discrete levels (0/12.5/25/37.5/50% of
# P_max) give real resolution in the range the policy actually operates,
# instead of 3 of the 5 levels (75%, 100%, and the coarse jump to 50%)
# sitting in a region the policy never used at all.
POWER_LEVEL_MAX_FRACTION = 0.5

# PPO's own discount factor. Deliberately NOT reusing
# DISCOUNT_FACTOR_LAMBDA=0.01 (the paper's CMDP/Bellman lambda, MMSP'25
# Sec. IV) here: taken literally as a per-step decay, lambda^t collapses
# to ~0 after 1-2 steps, which would make PPO almost fully myopic if used
# as gamma, and would make a literal sum_t lambda^t*cost^t constraint
# estimate (eqs. 6-8) dominated by the first step or two - flagged in
# conversation, not a new finding. The Lagrangian constraint-cost
# averaging (streaming_rl/lagrangian.py) therefore uses a plain running
# mean instead of a literal discounted sum, and PPO gets a conventional
# discount factor here instead. DISCOUNT_FACTOR_LAMBDA itself stays
# untouched and uninvolved in that code path.
PPO_GAMMA = 0.99


# =====================================================================
# Constraint budgets (constrained MDP: eqs. 6-8)
#
# The professor's answer (verbatim): "I imagine power budget should be
# selected to match the battery power of the UAV (not to exceed it).
# Number of tiles should correspond to at least the size of the
# viewport (but not much bigger). We can use different values for the
# stall time and assess their impact."
# =====================================================================

# RE-DERIVED this session (other/decision_log.md #5-6) after the
# Lagrangian mechanism moved from a windowed running MEAN of per-step
# costs to the literal eqs. 6-8 discounted SUM per episode (J_D/J_P/J_B,
# streaming_rl/environment.py) - a different quantity, dominated by each
# episode's first ~2 steps at DISCOUNT_FACTOR_LAMBDA=0.01 (see that
# constant's own comment). Calibrated by
# experiments/constraint_calibration.py: 3 reference tile-policies
# (mandatory_only/mandatory_random50/mandatory_all) x 3 power levels, on
# the 9 training traces only (never EVAL_TRACE_INDICES), fixed seeds.
# Confirmed with the user afterward, not picked silently.

# D_BAR (stall-time budget, same units as J_D = sum_t lambda^t*cost_D^t):
# per the professor, NOT a single fixed value - meant to be swept
# experimentally. RE-CALIBRATED (Stage 8) under DISCOUNT_FACTOR_LAMBDA=0.99
# (previously 0.01 - see that constant's own comment): efficient policy
# (mandatory-only, max power) J_D~=0.0016, wasteful (mandatory-all, max
# power) J_D~=18.90. D_BAR=3.2 sits at the same ~17% position within the
# efficient-to-wasteful range that the old D_BAR=0.1 sat at under the old
# (0.01) calibration - a mechanical rescale, not a new judgment call.
# Confirmed with the user. Sweep list not populated yet.
D_BAR_CANDIDATES = None   # TBD - single-value D_BAR used for now, see above
D_BAR = 3.2

# P_BAR (power budget, same units as J_P): per the professor, should
# match the UAV's battery capacity - no spec exists, so this is "a
# value that makes sense" empirically. IMPORTANT calibration finding
# (confirmed again under the Stage 8 recalibration): J_P is determined
# ENTIRELY by which power level is chosen and is completely unaffected by
# tile policy - comparing tile-policy efficiency (the method used for
# D_BAR/B_BAR) gives a degenerate, uninformative range for P_BAR
# specifically. RE-CALIBRATED (Stage 8) under DISCOUNT_FACTOR_LAMBDA=0.99
# AND the narrowed power range (POWER_LEVEL_MAX_FRACTION=0.5, see that
# constant's own comment): J_P = 0 / 7.59 / 15.18 at 0% / 25% / 50% of
# P_max. The 700K run's trained policy had converged to a constant 25% of
# P_max (J_P~=7.59 under this new scale) - since Part E's narrowed action
# space made that point the MIDDLE of the range rather than near the
# bottom, P_BAR=10 sits between that old comfortable point and the new
# top of the range (15.18), so using up toward the new range's top now
# actually costs something instead of being free. This one required an
# actual judgment call, not just a mechanical rescale - confirmed with
# the user.
P_BAR_CANDIDATES = None   # TBD - single-value P_BAR used for now
P_BAR = 10.0

# B_BAR (tile-count budget, normalized by N_TILES, same units as J_B):
# per the professor, "at least the size of the viewport (but not much
# bigger)." Geometric derivation: the 120x60 deg viewport, over the 8x8
# grid (45x22.5 deg tiles), spans between ceil(120/45)=3 and 4 tile-
# columns, and between ceil(60/22.5)=3 and 4 tile-rows, depending on
# alignment with the grid - i.e. between 3*3=9 and 4*4=16 tiles out of
# 64.
B_BAR_MIN_TILES = 9
B_BAR_MAX_TILES = 16
B_BAR_CANDIDATES = None   # TBD - single-value B_BAR used for now

# RE-CALIBRATED (Stage 8) under DISCOUNT_FACTOR_LAMBDA=0.99 (previously
# 0.01): efficient (mandatory-only) policy floors at J_B~=6.57 (still
# matches the geometric 9-16 tile bound almost exactly - the same strong
# cross-check as before, just at the new scale), wasteful (mandatory-all)
# maxes at J_B~=30.36. B_BAR=10.2 sits at the same ~15% position within
# the efficient-to-wasteful range that the old B_BAR=0.35 sat at under
# the old (0.01) calibration - a mechanical rescale. Confirmed with the
# user.
B_BAR = 10.2


# =====================================================================
# Train/eval trace split (streaming_rl.environment.TileStreamingEnv's
# trace_indices param). Indices are POSITIONS into valid_traces (0..11
# for the 12 valid Runner traces), NOT the underlying hn.mat user_slot
# values (which are [0,2,4,5,6,7,8,9,10,11,13,25] - confirmed different
# this session). Arbitrary, fixed pick, same status as N_POWER_LEVELS -
# not sourced from the paper or the professor. 3 of 12 (25%) held out
# for evaluation only; train = the other 9.
#
# CAVEAT: with only 3 distinct held-out viewing sessions, eval-metric
# variance is dominated by WHICH traces got held out, not fixed by
# raising n_eval_episodes (that only averages channel noise, not
# viewer-behavior diversity). Revisit (e.g. widen to 4-5 held out) if
# eval curves look trace-dominated once real training data exists.
# =====================================================================
EVAL_TRACE_INDICES = (2, 6, 9)

# Fixed, reused channel seed per held-out trace (same index order as
# EVAL_TRACE_INDICES) - deliberately NOT fresh-random per evaluation
# call, so the channel-gain sequence is bit-identical across every
# checkpoint's evaluation, isolating policy differences from
# environment randomness (streaming_rl/eval.py). Arbitrary but fixed,
# same status as EVAL_TRACE_INDICES.
EVAL_SEEDS = (1000, 1001, 1002)


# =====================================================================
# Lagrangian relaxation of the constrained MDP (eqs. 6-8): adaptive
# mu_D, mu_P, mu_B replace the fixed BETA_S/BETA_P/BETA_B weights during
# training (see streaming_rl/lagrangian.py, train_ppo.py). BETA_Q is
# left untouched - quality is the objective (eq. 5), not a constraint.
# =====================================================================
MU_D_INIT = 1.0    # warm-start at today's fixed BETA_S value, so the Lagrangian
MU_P_INIT = 1.0    # reward starts out equivalent to the current fixed-weight
MU_B_INIT = 1.0    # penalty reward (eq. 9) and adapts from there.

# Dual step size (eta): free engineering/tuning knob, same status as
# N_POWER_LEVELS above - NOT sourced from the paper or the professor.
# RESCALED TWICE (Stage 8 then 8b), both times verified against real
# training data, not guessed. First rescale (full ~30x, matching the J-scale
# growth from the lambda=0.01->0.99 change) went to eta=0.0003 - this
# correctly stopped the runaway mu growth (measured 35x-too-fast at the old
# eta=0.01), but overcorrected: checking mu/budget ratios, the original
# successful run needed mu_D/D_BAR~=17x to reach feasibility, and this run's
# mu_D sat at only ~14x when eta=0.0003 was applied - at that eta, closing
# the remaining gap would take thousands of rollouts, not a few hundred.
# Retuned (Stage 8b) to a MODERATE reduction instead of the full one -
# eta=0.0025 (~4x the 0.0003 value, ~1/4 of the original 0.01) - damps the
# oscillation/overshoot that caused the eval/mean_power spike to the new
# power range's ceiling, while letting mu actually reach the magnitude it
# likely needs within a few hundred rollouts. Confirmed with the user.
MU_LEARNING_RATE = 0.0025

# OBSOLETE, no longer read anywhere: multiplier updates now fire once
# per PPO rollout (train_ppo.py's LagrangianMultiplierCallback, hooked
# to _on_rollout_end) using literal per-episode discounted returns
# (J_D/J_P/J_B), not a windowed running-mean cadence. Left here,
# deliberately unused, as a marker of the old mechanism rather than
# silently deleted - see other/decision_log.md Stage 6 row.
# MU_UPDATE_FREQ_STEPS = 512


# =====================================================================
# Reward weights (eq. 9): r^t = BETA_Q*Qbar_viewport - BETA_S*D^t/T0
#                                - BETA_P*P^t/P_max - BETA_B*sum(x_i)/N
#
# NOT sourced from anywhere - these are free "design parameter"
# weights the researcher chooses (unlike the budgets above, no
# professor guidance exists for these specific numbers). Set to equal
# weight (1.0 each) as a neutral, arbitrary starting point purely so
# the environment has *something* runnable for the sanity check - not
# a considered choice. MMSP'25's own analogous weights (their
# gamma_1/2/3) were explicitly swept across experiments, not fixed;
# expect the same treatment here once training actually starts. These
# stay in place as the "fixed-beta" baseline reward path
# (train_ppo.py --no-lagrangian); the Lagrangian path above replaces
# BETA_S/BETA_P/BETA_B with mu_D/mu_P/mu_B during training instead.
# =====================================================================
BETA_Q = 1.0   # viewport-quality reward weight
BETA_S = 1.0   # stall-time penalty weight
BETA_P = 1.0   # power penalty weight
BETA_B = 1.0   # enhanced-tile-count penalty weight
