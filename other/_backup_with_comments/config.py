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


# CONFIRMED THIS SESSION: for the FIRST implementation, only 2 levels
# are used - base (k=0) and one adjacent enhancement step (k=1, i.e.
# array index 5) - not the full k=0..6 graduated range yet. Matches
# the start-simple approach used elsewhere (discrete actions first,
# one critic first, etc.). The mapping function above already supports
# the full range for when this gets extended later.
INITIAL_ENHANCEMENT_LEVELS = (0, 1)   # k values used in the first pass

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
# approaches). The overlap-fraction computation itself lives in the
# future viewport.py module; this only records the choice made.
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


# =====================================================================
# Reinforcement-learning setup
#
# The professor's answer (verbatim): "We will first implement a
# standard PPO baseline with one critic, V(s_t). After validating the
# baseline, we will introduce post-decision states and add a second
# critic, V(s~_t), following the previous work."
# =====================================================================
DISCOUNT_FACTOR_LAMBDA = 0.01   # MMSP'25 Sec. IV; reusable regardless of algorithm
ALGORITHM = "PPO"               # per the original meeting plan

# CONFIRMED: two-stage critic plan, not a single fixed choice.
#   Stage 1 (baseline): one critic estimating V(s_t)  - the regular state.
#   Stage 2 (after the baseline is validated): add a second critic
#     estimating V(s~_t) - the post-decision state - once PDS is
#     introduced, following the prior related work.
N_CRITICS_BASELINE = 1
N_CRITICS_WITH_PDS = 2

ACTION_MODE = "discrete"        # start discrete/binned (per the original meeting)
                                 # before moving to continuous

# How many discrete power levels P^t is binned into (0 to P_max, linear
# steps in watts). NOT sourced from anywhere - a reasonable, arbitrary
# starting choice for "binned" power, same status as the reward
# weights above. 5 gives {0, 25%, 50%, 75%, 100%} of P_max.
N_POWER_LEVELS = 5


# =====================================================================
# Constraint budgets (constrained MDP: eqs. 6-8)
#
# The professor's answer (verbatim): "I imagine power budget should be
# selected to match the battery power of the UAV (not to exceed it).
# Number of tiles should correspond to at least the size of the
# viewport (but not much bigger). We can use different values for the
# stall time and assess their impact."
# =====================================================================

# D_BAR (stall-time budget): per the professor, NOT a single fixed
# value - meant to be swept experimentally and its impact assessed.
# The METHOD is confirmed; the actual candidate values are STILL OPEN
# (see top of file).
D_BAR_CANDIDATES = None   # TBD - e.g. a list like [0.0, 0.05, 0.1, ...]

# P_BAR (power budget): per the professor, should match the UAV's
# battery capacity (not exceed it). No specific UAV/battery spec has
# been provided yet. Left as None deliberately - not inventing a number.
P_BAR = None   # TBD - needs a real UAV battery/power spec

# B_BAR (tile-count budget, normalized by N_TILES): per the professor,
# "at least the size of the viewport (but not much bigger)."
# Geometric derivation: the 120x60 deg viewport, over the 8x8 grid
# (45x22.5 deg tiles), spans between ceil(120/45)=3 and 4 tile-columns,
# and between ceil(60/22.5)=3 and 4 tile-rows, depending on alignment
# with the grid - i.e. between 3*3=9 and 4*4=16 tiles out of 64.
# Confirmed this session: use this estimate as a starting candidate.
# B_BAR below is a round point value within that range, not a value
# the professor gave directly - revisit once real tile-viewport-overlap
# code exists and the alignment is no longer a rough estimate.
B_BAR_MIN_TILES = 9
B_BAR_MAX_TILES = 16
B_BAR = 0.20   # ~13 tiles / 64; starting candidate, see derivation above


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
# expect the same treatment here once training actually starts.
# =====================================================================
BETA_Q = 1.0   # viewport-quality reward weight
BETA_S = 1.0   # stall-time penalty weight
BETA_P = 1.0   # power penalty weight
BETA_B = 1.0   # enhanced-tile-count penalty weight
