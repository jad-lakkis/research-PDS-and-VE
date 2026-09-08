# Research Summary — UAV 360° Video RL Streaming Project

Presentation-prep material for reporting progress to Prof. Jacob. This deliberately does **not** restate the problem formulation or recap the last meeting — Jacob was there and co-owns the formulation, so re-explaining it would waste the meeting. This covers what's **new** since then: what was done with the dataset (technical approach + real findings), what a closely related paper contributed, how base/enhancement-layer streaming actually works given what the data supports, the environment design, and — most importantly — every open modeling decision and question to walk through together.

---

## 1. How to use this document

Sections 3–6 are the new factual grounding (dataset, technical approach, findings, sibling paper). Section 7 answers a specific mechanics question that came up (base/enhancement layers) in detail. Section 8 is the environment design. Section 9 pulls every "we could do this or that" decision into one place — this is the core of what to discuss. Sections 10–12 are the three lists to present directly: open questions, what's resolved, and inconsistencies found. Section 13 is what happens after the meeting.

---

## 2. Executive summary

- Goal: an RL (PPO-based) agent that selects which tiles of a UAV-captured 360° video to enhance and how much transmit power to use, maximizing viewport quality subject to stall-time, power, and tile-budget constraints (full math in the original formulation document — not repeated here).
- Dataset in hand: Chakareski et al.'s MMSys'21 360° video dataset (`hn.mat` head-navigation data + `rd.mat` rate-distortion data, 15 videos). Fully explored, with real output — see Section 5.
- A second, closely related paper (MMSP'25, same lead author as the dataset, same dataset used in its experiments) was found and mined for concrete parameter values, a full channel-gain model, and an actual-viewport-from-viewing-direction rule (Section 6).
- **Confirmed missing from the dataset**: wireless channel gain (zero network/channel data of any kind) and literal base-layer/enhancement-layer fields (only independent, single-layer 7-QP-level encoding exists — Section 7 explains exactly what this means for implementation).
- **Now resolved, thanks to the sibling paper**: a complete channel-gain formula with numeric parameters; a field-of-view value and an actual-viewport rectangle rule; confirmation the tile grid is 8×8; confirmation the "last viewport as prediction" baseline is paper-precedented.
- **Still open** (Section 10): several modeling conventions (which QP level = base vs. enhanced, formulaic vs. data-grounded bitrate model, mobility on/off, exact FOV source, yaw/pitch→azimuth/altitude convention) that need either a research decision or Jacob's input.
- The MATLAB-fitted R-D curve objects (`cfit`) in `rd.mat` are not readable in Python — but this doesn't block anything: the raw numeric bitrate/distortion data they were fit from is fully readable and sufficient on its own (Section 5.5).
- Next step after the meeting: lock in the open decisions, then implement the gymnasium environment and a PPO baseline.

---

## 3. The dataset — source and what was extracted

**Citation**: J. Chakareski, R. Aksu, V. Swaminathan, and M. Zink, "Full UHD 360° video dataset and modeling of rate-distortion characteristics and head movement navigation," in *Proc. ACM Multimedia Systems Conference*, Istanbul, Turkey, Sept. 2021. Zenodo record 5156999.

**What was actually extracted into the project** (`data/`): exactly two files, `hn.mat` (~3.86 MB, head-navigation data) and `rd.mat` (~74.5 MB, rate-distortion data) — both MATLAB v5 format. No video files, no CSVs, no network/channel data of any kind exist anywhere in this release.

**Verbatim contents of `Readme - Dataset info.txt`** (project root):

> Following dataset includes rate-distortion characteristics data and Head navigation data of 15 8K 360-degree videos.
>
> Videos used in order are: no / Video name / frame numbers / fps / Resolution / Depth / Projection — 01 Academic 1080 30 8192x4096 8bit ERP; 02 Basketball 1080 30 8192x4096 8bit ERP; 03 Bridge 1080 30 8192x4096 8bit ERP; 04 GateNight 1080 30 8192x4096 8bit ERP; 05 Runner 1080 30 8192x4096 8bit ERP; 06 SiyuanGate 1080 30 8192x4096 8bit ERP; 07 SouthGate 1080 30 8192x4096 8bit ERP; 08 StudyRoom 1080 30 8192x4096 8bit ERP; 09 Sward 1080 30 8192x4096 8bit ERP; 10 Chairlift 300 30 8192x4096 10bit ERP; 11 Skateboard 300 30 8192x4096 10bit ERP; 12 Gaslamp 300 30 8192x4096 8bit ERP; 13 Harbor 300 30 8192x4096 8bit ERP; 14 KiteFlite 300 30 8192x4096 8bit ERP; 15 Trolley 300 30 8192x4096 8bit ERP.
>
> **`rd.mat` variables**: `video_bitrate_data` — raw video bitrate (bps), 1×15 cell (video index), each cell `7×64×N` (QP, tile index, N=frame index). `video_ymse_data` — raw video YMSE, same shape. `RtoD_exp` — exponential rate-distortion model, `15×64×N` cell (video, tile, N=frame index), each cell a `cfit` function `D=a·e^(R·b)`. `RtoD_pow` — power-law rate-distortion model, `D=a·R^b`. `QPtoR_exp` — exponential QP-rate model, `R=a·e^(QP·b)`. `QPtoR_pow` — power-law QP-rate model (readme literally writes `D=a·QP^b` here, inconsistent with the variable's own name — flagged in Section 11).
>
> **`hn.mat` variable**: `HMD_data` — README states `12×12` cell data (video index, user index), each cell `N×4` (yaw, pitch, roll, timestamp [sec]).

---

## 4. Technical approach — how the data was extracted and explored

Answering directly: here is what was actually done, coding-wise, to go from the raw MATLAB files to the tables in this document.

- The dataset arrives as two MATLAB `.mat` files — MATLAB's native binary save format. These aren't natively readable by anything outside MATLAB, and no MATLAB license is available (or needed).
- Both files were loaded directly in Python using `scipy.io.loadmat` — a standard function in SciPy (a core Python scientific-computing library) built specifically to read `.mat` files without requiring MATLAB itself. Loaded with the `simplify_cells=True` option, which flattens MATLAB's "cell arrays" (its version of nested/mixed-content arrays) into ordinary nested Python/NumPy arrays wherever possible.
- All exploration happened in a Jupyter notebook (`explore_data.ipynb`) — an interactive Python environment where code runs in cells and the output (tables, printed numbers, plots) stays attached directly below the code that produced it. This is why every table in Section 5 is real, computed output — not a typed-up description of what the data probably contains.
- `pandas` (Python's standard data-table library) was used to turn the raw NumPy arrays into labeled, readable tables — e.g., converting a raw `1080×4` block of numbers into an actual table with column headers `yaw, pitch, roll, timestamp`.
- Where the raw structure couldn't be trusted at face value, small validity-checking functions were written rather than assuming the file's documentation was correct — e.g., a classifier that separates real head-navigation recordings from empty or all-zero placeholder cells, since the README's own occupancy numbers turned out to be wrong (Section 11).
- Where a claim in the README or the dataset paper could be checked against the actual data instead of trusted, it was checked directly — e.g., empirically confirming the GOP size is 30 frames by counting occupied slots in the rate-distortion arrays, rather than taking the paper's word for it.
- **One real limitation hit**: four of `rd.mat`'s variables (the fitted rate-distortion curve models) are stored as MATLAB *class objects* (`cfit`, from MATLAB's Curve Fitting Toolbox), not plain numbers. `scipy.io.loadmat` can see that these objects exist but can't decode their internal contents — that would require MATLAB itself, or reverse-engineering an undocumented internal binary format, which wasn't worth attempting (Section 5.5 has the detail). This is a genuine gap in what's extractable from these specific files, not a mistake in the loading approach — and it turned out not to matter, because the raw numbers those curves were fit from are fully readable on their own.
- The whole notebook was executed end-to-end (via `jupyter nbconvert`, which runs every cell fresh in a real Python kernel and reports any errors) before any of its output was treated as trustworthy — so everything quoted as a "finding" in this document reflects real, verified execution, not code that merely looks correct.
- Result: three files now exist in the project — `explore_data.ipynb` (the exploration work and all real output), `environment_design.md` (the design discussion), and this document.

---

## 5. Dataset exploration findings

All tables below are real, executed output from `explore_data.ipynb`.

### 5.1 Video catalog

| video_index | array_index | name | frames | fps | resolution | bit_depth | projection |
|---|---|---|---|---|---|---|---|
| 1 | 0 | Academic | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 2 | 1 | Basketball | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 3 | 2 | Bridge | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 4 | 3 | GateNight | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 5 | 4 | Runner | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 6 | 5 | SiyuanGate | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 7 | 6 | SouthGate | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 8 | 7 | StudyRoom | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 9 | 8 | Sward | 1080 | 30 | 8192x4096 | 8bit | ERP |
| 10 | 9 | Chairlift | 300 | 30 | 8192x4096 | 10bit | ERP |
| 11 | 10 | Skateboard | 300 | 30 | 8192x4096 | 10bit | ERP |
| 12 | 11 | Gaslamp | 300 | 30 | 8192x4096 | 8bit | ERP |
| 13 | 12 | Harbor | 300 | 30 | 8192x4096 | 8bit | ERP |
| 14 | 13 | KiteFlite | 300 | 30 | 8192x4096 | 8bit | ERP |
| 15 | 14 | Trolley | 300 | 30 | 8192x4096 | 8bit | ERP |

`video_index` is 1-based (matches the README's "no" column); `array_index` is 0-based (matches `.mat` array indexing).

### 5.2 Head-navigation data (`hn.mat`)

**Validity classification.** `HMD_data` has 510 possible `(video, user)` cells (15×34 — not 12×12 as the README claims). A cell is classified: `size==0` → empty; wrong shape → unexpected_shape (didn't occur); `np.all(arr==0)` → placeholder; otherwise → valid. Exact-zero was chosen over checking `dtype==uint8` (coincidental on the one observed placeholder, not a real rule) or `np.allclose` (real recordings can have near-zero yaw/pitch/roll, but never an all-zero **timestamp** column across up to 1080 rows).

**Result**: 315 empty, **194 valid**, 1 placeholder (index `(4,1)`), out of 510 total cells.

Per-video valid-trace counts:

| array_index | video_index | name | valid_trace_count | paper_claimed_min | paper_claimed_max | exceeds_paper_max |
|---|---|---|---|---|---|---|
| 0 | 1 | Academic | 12 | 5 | 12 | False |
| 1 | 2 | Basketball | 14 | 5 | 12 | True |
| 2 | 3 | Bridge | 11 | 5 | 12 | False |
| 3 | 4 | GateNight | 17 | 5 | 12 | True |
| 4 | 5 | Runner | 12 | 5 | 12 | False |
| 5 | 6 | SiyuanGate | 11 | 5 | 12 | False |
| 6 | 7 | SouthGate | 14 | 5 | 12 | True |
| 7 | 8 | StudyRoom | 12 | 5 | 12 | False |
| 8 | 9 | Sward | 12 | 5 | 12 | False |
| 9 | 10 | Chairlift | 22 | 5 | 12 | True |
| 10 | 11 | Skateboard | 30 | 5 | 12 | True |
| 11 | 12 | Gaslamp | 6 | 5 | 12 | False |
| 12 | 13 | Harbor | 10 | 5 | 12 | False |
| 13 | 14 | KiteFlite | 6 | 5 | 12 | False |
| 14 | 15 | Trolley | 5 | 5 | 12 | False |

Total valid traces: **194** vs. the dataset paper's claimed total of 121. 5 of 15 videos exceed the paper's claimed max of 12 traces/video.

**Real sample data** — first and last 5 rows of the first valid recording (`Academic`, recording index `(0,0)`, 1080 rows total, columns `yaw, pitch, roll, timestamp` in degrees/seconds):

| # | yaw | pitch | roll | timestamp |
|---|---|---|---|---|
| 0 | -48.747200 | -7.432520 | 3.842840 | 5.700000e-07 |
| 1 | -0.000190 | -3.716223 | 1.921415 | 3.500786e-02 |
| 2 | -0.001341 | -3.715703 | 1.921299 | 6.728675e-02 |
| 3 | -0.004152 | -3.714361 | 1.920745 | 1.025241e-01 |
| 4 | -0.008139 | -3.712591 | 1.919882 | 1.371408e-01 |
| … | … | … | … | … |
| 1075 | -0.423076 | 41.028340 | -23.829580 | 3.583335e+01 |
| 1076 | 1.285980 | 40.994440 | -22.961280 | 3.586785e+01 |
| 1077 | 2.577210 | 40.924640 | -22.251880 | 3.590037e+01 |
| 1078 | 3.680450 | 40.846040 | -21.606080 | 3.593524e+01 |
| 1079 | 4.571370 | 40.781240 | -21.094280 | 3.596768e+01 |

Sampled at ~30 Hz (1080 rows over ~35.97 s, matching the video's frame rate/count). Yaw values reach exactly 180° in magnitude elsewhere in the data — consistent with degrees, not radians (never explicitly labeled as a unit in the files).

### 5.3 GOP-axis check

The README labels the 3rd axis of `RtoD_exp`/`RtoD_pow`/`QPtoR_exp`/`QPtoR_pow` as "N=frame index," sized 36 — but videos have 1080 or 300 frames, not 36. This was checked empirically against the paper's claimed GOP size of 30 frames:

| array_index | video_index | name | frames | expected_gops (frames/30) | observed_occupied_gop_slots | uniform_across_all_64_tiles | matches_frames_div_30 |
|---|---|---|---|---|---|---|---|
| 0 | 1 | Academic | 1080 | 36 | 36 | True | True |
| 1 | 2 | Basketball | 1080 | 36 | 36 | True | True |
| 2 | 3 | Bridge | 1080 | 36 | 36 | True | True |
| 3 | 4 | GateNight | 1080 | 36 | 36 | True | True |
| 4 | 5 | Runner | 1080 | 36 | 36 | True | True |
| 5 | 6 | SiyuanGate | 1080 | 36 | 36 | True | True |
| 6 | 7 | SouthGate | 1080 | 36 | 36 | True | True |
| 7 | 8 | StudyRoom | 1080 | 36 | 36 | True | True |
| 8 | 9 | Sward | 1080 | 36 | 36 | True | True |
| 9 | 10 | Chairlift | 300 | 10 | 10 | True | True |
| 10 | 11 | Skateboard | 300 | 10 | 10 | True | True |
| 11 | 12 | Gaslamp | 300 | 10 | 10 | True | True |
| 12 | 13 | Harbor | 300 | 10 | 10 | True | True |
| 13 | 14 | KiteFlite | 300 | 10 | 10 | True | True |
| 14 | 15 | Trolley | 300 | 10 | 10 | True | True |

**GOP size = 30 frames, confirmed for all 15 videos.** Separately confirmed: `video_bitrate_data`/`video_ymse_data`'s 3rd axis is genuinely per-*frame* (e.g. shape `(7,64,1080)` for `Academic`), while `RtoD_exp` etc.'s 3rd axis is per-*GOP* — the README's "N=frame index" label means two different things for the two variable families.

### 5.4 Real rate-distortion sample values

One real slice: video `Academic` (array_index 0), frame 0, tiles 0–7, all **7** QP operating points (correcting the "eight" mentioned in conversation — the dataset has 7 QP levels per README and confirmed array shape `7×64×frames`).

**`video_bitrate_data`** (bps):

| QP_idx | tile_0 | tile_1 | tile_2 | tile_3 | tile_4 | tile_5 | tile_6 | tile_7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 4,352,960 | 5,947,048 | 5,408,304 | 1,418,216 | 3,939,208 | 10,181,536 | 7,214,032 | 4,712,112 |
| 1 | 1,231,456 | 1,685,672 | 1,540,952 | 528,008 | 1,275,144 | 3,220,080 | 2,231,536 | 1,322,432 |
| 2 | 414,728 | 513,872 | 451,792 | 232,656 | 470,872 | 994,760 | 687,760 | 432,352 |
| 3 | 166,208 | 176,352 | 149,184 | 107,880 | 177,816 | 287,920 | 220,472 | 165,672 |
| 4 | 89,656 | 87,272 | 76,648 | 67,488 | 86,544 | 107,376 | 102,040 | 85,232 |
| 5 | 51,240 | 50,904 | 49,624 | 46,256 | 49,208 | 52,448 | 54,912 | 49,440 |
| 6 | 35,824 | 36,168 | 34,688 | 35,728 | 34,528 | 34,464 | 36,528 | 34,992 |

**`video_ymse_data`** (Y-channel MSE, unit not stated in README):

| QP_idx | tile_0 | tile_1 | tile_2 | tile_3 | tile_4 | tile_5 | tile_6 | tile_7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.008780 | 0.011951 | 0.009974 | 0.002695 | 0.008915 | 0.019857 | 0.014011 | 0.010338 |
| 1 | 0.023418 | 0.031136 | 0.026312 | 0.005423 | 0.023384 | 0.059978 | 0.041185 | 0.028591 |
| 2 | 0.037226 | 0.045403 | 0.037271 | 0.008593 | 0.038292 | 0.100639 | 0.065685 | 0.044903 |
| 3 | 0.058832 | 0.058105 | 0.045832 | 0.012993 | 0.056292 | 0.140055 | 0.089102 | 0.059086 |
| 4 | 0.090660 | 0.081131 | 0.057625 | 0.018934 | 0.078150 | 0.186401 | 0.112957 | 0.083115 |
| 5 | 0.153574 | 0.115416 | 0.077684 | 0.031139 | 0.108919 | 0.270601 | 0.154951 | 0.134130 |
| 6 | 0.186302 | 0.132282 | 0.101715 | 0.051706 | 0.138247 | 0.351599 | 0.190901 | 0.109589 |

`QP_idx_0` = highest quality/highest bitrate, `QP_idx_6` = lowest quality/lowest bitrate. **Important, easy to miss in the table**: the bitrate-vs-distortion tradeoff is genuinely *content-dependent* — tile_5's bitrate at QP_idx_0 is ~10.2M bps vs. tile_3's ~1.4M bps for the same quality setting, a >7x difference between tiles of the *same video, same instant*. Any base/enhancement modeling that treats all tiles as costing the same is discarding real information this table shows is there. (Walked through in full in Section 7.)

### 5.5 The MATLAB `cfit`/MCOS problem

`RtoD_exp`, `RtoD_pow`, `QPtoR_exp`, `QPtoR_pow` are meant to hold *fitted continuous curves* (e.g. `D=a·e^(b·R)`) — a `cfit` object is MATLAB's Curve Fitting Toolbox representation of such a fitted curve, i.e. the `a`/`b` coefficients plus the functional form. Attempting to read these coefficients in Python revealed they're stored as opaque MCOS (MATLAB class) objects, not plain numbers:

| index | class | arr_values |
|---|---|---|
| (0,0,0) | cfit | [3707764736, 2, 1, 1, 49153, 1] |
| (0,0,1) | cfit | [3707764736, 2, 1, 1, 50113, 1] |
| (0,1,0) | cfit | [3707764736, 2, 1, 1, 49168, 1] |
| (1,0,0) | cfit | [3707764736, 2, 1, 1, 49154, 1] |
| (14,63,9) | cfit | [3707764736, 2, 1, 1, 58752, 1] |

Every value is constant across cells except one varying integer (an object handle) — that handle points into `rd.mat`'s `__function_workspace__` blob (386.5 MB), MATLAB's undocumented internal object-serialization format. **Conclusion: the real `a`/`b` coefficients are not retrievable via `scipy.io.loadmat` in reasonable effort.**

**Practical resolution — you don't need to open them.** `video_bitrate_data` and `video_ymse_data` (Section 5.4) are plain numeric arrays holding the exact raw `(QP, bitrate, distortion)` points these curves were fit *from* — fully readable today. So:
- Picking one of the **7 real, discrete QP-level operating points** directly (a lookup-table approach) is already possible with what's extracted — no MATLAB needed.
- The *only* reason to touch the `cfit`-style curves at all would be wanting a **continuous** rate↔distortion function instead of 7 discrete points — and if so, the fix is to refit the same functional forms (`D=a·e^(bR)`, `D=a·R^b`) yourself in Python (`scipy.optimize.curve_fit`) using the raw points above. That's a modeling choice (Section 9), not a data-access blocker.

### 5.6 Problem-formulation ↔ dataset mapping table

| Symbol | Description | Status | Notes |
|---|---|---|---|
| N, \|N\|=64 | Tile set / tile count | Available | Confirmed via the tile axis length (64) in RtoD_exp/QPtoR_exp/video_bitrate_data/video_ymse_data. |
| Tile grid layout | Spatial arrangement of the 64 tiles (e.g. 8x8 raster) | Missing – modeling decision needed | Not present in hn.mat/rd.mat/README. (Now confirmed for `Runner` via the MMSP'25 sibling paper — see Section 6.) |
| x^t, P^t | Action: per-tile enhance decision + UAV transmit power | Design parameter | RL action variables the agent chooses; P_max is a design constant. Tile cardinality (64) is dataset-confirmed. |
| yaw, pitch, roll, timestamp | Raw head orientation samples | Available | HMD_data, 194 valid traces after placeholder/empty filtering. Degrees, ~30Hz, per-frame. |
| V_actu^t (as tile coverage) | Actual viewport, expressed as which tiles it overlaps | Missing – external source needed | Turning yaw/pitch/roll into tile overlap needs a field-of-view (FOV) value. (Now partially resolved — see Section 6's actual-viewport rule.) |
| Predicted viewport | Viewport prediction used to pick enhanced tiles | Missing – modeling decision needed | No predictor exists in the dataset. (Now resolved as a baseline — see Section 6's "last viewport as prediction.") |
| C_cov^t, Q̄_viewport^t | Coverage ratio / normalized viewport quality (QoE) | Missing – modeling decision needed | Depends on the viewport-to-tile-coverage mapping above. |
| Z^t | Playback-buffer level | Computable | Not stored anywhere; simulated at runtime via the buffer recursion, using T0, R^t and A^t. |
| h^t | UAV-to-base-station channel gain | Missing – external source needed | CONFIRMED: zero wireless/channel/network fields exist anywhere in this dataset. (Now resolved — see Section 6's channel-gain model.) |
| R^t | UAV-to-BS transmission rate (Shannon formula) | Computable | Computable once h^t is sourced, plus design constants W, N0. Distinct from the *encoding* bitrate, which is already available. |
| D^t | Stall time | Computable | Simulated once Z^t, R^t and A^t are available. |
| A_base, A_i,enh^t | Base-layer data size / per-tile enhancement data size | Missing – modeling decision needed | CONFIRMED: no scalable/layered field exists — only single-layer 7-QP data. See Section 7 for exactly what this means and how it can still be implemented. |
| video_bitrate_data, video_ymse_data | Raw per-QP, per-tile, per-frame bitrate (bps) and Y-MSE | Available | rd.mat, shape (7 QP × 64 tiles × frames) per video. |
| RtoD_exp/pow, QPtoR_exp/pow | Fitted rate-distortion / QP-rate curve models | Missing – modeling decision needed | Structurally present but coefficients are opaque cfit/MCOS objects, not extractable. Fallback: refit from raw data. |
| GOP size | Frames per independently-coded GOP-tile | Computable | Empirically confirmed = 30 frames/GOP for all 15 videos. |
| β_q, β_s, β_p, β_b | Reward weights | Design parameter | Chosen by the researcher, not derived from data. |
| D̄, P̄, B̄ | Stall / power / tile-count budgets | Design parameter | Chosen by the researcher; not in this dataset. |
| Video/content metadata | Video name, frame count, fps, resolution, bit depth, projection | Available | Parsed from the README at runtime; matches array shapes. |

**Most important rows to actually walk through with Jacob**: `h^t` (channel gain — completely absent from the dataset, needs an external model — Section 6); `A_base`/`A_i,enh` (no literal base/enhancement field — Section 7 explains exactly what this means); and the FOV-dependent viewport-to-tile mapping (needs a value that isn't in the dataset itself — Section 6).

---

## 6. Sibling paper findings (MMSP'25)

**Chakareski, Wang & Mastronarde, "Reinforcement Learning-Based Dynamic Resource Allocation for Aerial 360° Video VR Streaming," IEEE MMSP 2025.** A direct precursor by the same lead author as the dataset, using the *same dataset* — its experiments use `Runner` (`array_index=4`/`video_index=5` above) — with near-identical state notation to the problem formulation. Its action space differs: a *continuous predicted-viewport enlargement* `(s^t_θ,s^t_φ)` plus power, not discrete per-tile selection.

| Name | Value / Formula | Source | Notes |
|---|---|---|---|
| P_max, W, T0, N0 | 20 dBm, 10 MHz, 1 s, -174 dBm/Hz | Sec. IV | Directly reusable starting values. |
| A_base | 4.5 Mb (fixed constant) | Sec. IV | Treated as fixed (whole panorama at low quality); the dataset instead allows computing A_base per-video/per-GOP from real bitrate at a chosen low-QP index. Decision point. |
| V_θ, V_φ | 60°, 30° (half-extents; full FOV = 120°×60°) | Sec. IV; Sec. II.C | Answers the FOV gap. CAVEAT: this is the paper authors' experimental choice, not a documented dataset/HMD spec. |
| Θ_max, Φ_max | 60°, 30° | Sec. IV | Bounds on the enlargement action (s_θ,s_φ) — does not map onto the tile-selection action space at all. |
| η, C | 23 pixel/deg, 28.8 bps/pixel (C swept in Fig. 7) | Sec. IV | Only relevant under this paper's continuous pixel-area A^t formula, not under a QP-table-grounded approach. |
| ζ, v0, d | 1 (static UAV), 0, 50 m | Sec. IV | Their *reported* experiments use a static UAV (fixed distance) — the full Gauss-Markov mobility model is defined but unused in their results. |
| Channel gain h | h = 3.24e-4·\|ψ\|² / (d·atan(24√2/5))²; ψ ~ Rician, CN(√(ν/(2+2ν)), 1/(2+2ν)), ν=10^1.2 (12 dB Rician factor), E[\|ψ\|²]=1 | Sec. II.D | RESOLVES the channel-gain gap with a complete closed-form model + numeric parameters — very likely the "prior work" params Jacob mentioned reusing. |
| λ (discount factor) | 0.01 | Sec. IV | Reusable regardless of PPO vs. DDPG. |
| I, τ_φ, τ_ϑ, network architecture | I=1024, τ_φ=1e-5, τ_ϑ=1e-3, MLP layer sizes/activations | Sec. IV | DDPG-specific training hyperparameters — PPO is the starting algorithm, so these are reference-only. |
| Tile grid | 64 tiles, arranged in an 8×8 grid | Sec. II.B (stated explicitly for Runner) | CONFIRMS the tile grid layout for Runner specifically; reasonable to extend to the other 14 videos, not independently verified per-video. |
| Actual viewport rule | V_actu = {(θ,φ) : θ̃-V_θ≤θ≤θ̃+V_θ, φ̃-V_φ≤φ≤φ̃+V_φ} | Sec. II.C | A rectangle in (azimuth, altitude) angle space centered on the actual viewing direction. Only 2 DOF used — roll dropped entirely. |
| Predicted-viewport baseline | (θ̂^t,φ̂^t) = (θ̃^{t-1},φ̃^{t-1}) — last viewport as prediction | Sec. IV | CONFIRMS the "last viewport as prediction" baseline is paper-precedented. |
| R-D model usage | "Using this bitrate and the R-D models from [15], we can then compute the respective viewport quality (Y-PSNR)" | Sec. IV; [15] = the MMSys'21 dataset paper | Confirms some bitrate→quality model (of the same kind as RtoD_exp/pow) is what this line of work actually uses. |
| A^t (data volume) formula | A^t = A_base + T0·C·η(2V_θ+2s_θ)·η(2V_φ+2s_φ) | Sec. II.C | Base layer always sent, enhancement additive on top — in the accounting sense (see Section 7). |
| Buffer/stall equations | Z^{t+1}=max(Z^t+T0·R^t-A^t,0); D^t per eq. 3 | Sec. II.E | Algebraically identical to the formulation's own buffer/stall equations. |
| Reward | r^t = -γ1·1{outage} - γ2·P^t/P_max - γ3·D^t/T0 | Sec. III.A | MATERIALLY SIMPLER than the formulation's own reward — binary outage vs. continuous Q_viewport, no tile-budget term at all. Should inform, not replace. |
| Runner fps discrepancy | Paper describes Runner as "8K-60fps"; the dataset README lists it at 30fps | Sec. IV vs. README | Surfaced, not silently resolved. |

**Most important rows to actually walk through with Jacob**: the full channel-gain formula (this is very likely literally his own "prior work" parameters); the confirmed 8×8 tile grid (fills a real gap in the raw dataset files); and the fact that MMSP'25's reward is materially simpler than ours (no tile-budget term, binary outage) — worth being upfront that we're deliberately going further, not that we missed something.

---

## 7. Base layer vs. enhancement layer — how it actually works

This came up directly and is worth presenting explicitly rather than leaving inside a table cell.

**Why it matters for the agent's decisions.** Whatever gets enhanced determines `A^t` (total data to send this slot). `A^t` determines the transmission rate `R^t` needed to avoid the playback buffer draining (stall). `R^t` requires a specific `P^t` for whatever channel gain is currently available. That `P^t` is exactly what the reward's power-penalty term charges the agent for. So yes — base/enhancement choices are directly, causally linked to the power decision; they aren't a separate, independent part of the problem.

**What "base layer / enhancement layer" normally means.** In true scalable video coding (the language both the formulation and MMSP'25's Fig. 4 use), there are two separate bitstreams: a base layer that decodes into a low-quality picture on its own, and an enhancement layer that is *not* independently decodable — it only produces something meaningful when combined with the decoded base layer. Genuinely additive at the bitstream level.

**What's actually in this dataset.** Not that. `rd.mat` contains 7 independent, self-contained HEVC encodings per tile at 7 different quality (QP) levels — confirmed, no scalable/layered field exists anywhere in the files. Going from low to high quality for a tile means swapping in a completely different, complete stream, not adding a refinement on top of the low-quality one.

**How to reconcile this.** For the data-*volume* accounting (`A^t = A_base + Σ enhancement`), additive math works fine and matches eq. (12) exactly — that's just bookkeeping over sizes. Physically, though, for a tile chosen for enhancement, you'd transmit the complete high-quality stream *instead of* the low-quality one, not both stacked together. Worth noting: MMSP'25 doesn't resolve this tension either — its own `A^t` formula uses fixed constants (`A_base=4.5Mb`, `C=28.8 bps/pixel`), not real per-tile data pulled from this dataset. **Neither paper has a working example of pulling true additive-scalable bits out of this specific dataset, because the dataset doesn't contain them.**

**Concretely, with real numbers** (from Section 5.4's `Academic`/frame-0 table): tile_5 costs 10,181,536 bps at the highest quality vs. 34,464 bps at the lowest — enhancing that one tile adds ~10.15 Mbps to `A^t`. Enhancing tile_3 instead (1,418,216 vs. 35,728 bps) only adds ~1.38 Mbps. *Which* tiles get enhanced — not just how many — has a large, content-dependent effect on the data volume, and therefore on the power the agent needs to spend.

**What this means practically, if implementing this today:**
1. Every tile is sent, every GOP, at a chosen "base" QP level — unconditionally. That sets `A_base`.
2. For tiles the agent selects (`x^t_i=1`), that tile is instead sent at a chosen "enhanced" QP level.
3. `A^t` = (sum of base-level bitrate over all 64 tiles) + (sum, over selected tiles only, of enhanced-level bitrate minus base-level bitrate for that tile).
4. This is real, additive, correct accounting — it just isn't literal scalable-bitstream combination, because the source data doesn't support that.

**What's still open** (unchanged, but now precise): *which* of the 7 QP indices plays "base" and which plays "enhanced" — that choice sets how big the jump in step 3 is, tile by tile, and is not specified anywhere in either paper.

**One limitation worth flagging to Jacob**: because this is independent-QP encoding rather than true scalable coding, only whichever 2 of the 7 levels get chosen are reachable — there's no fine-grained/intermediate quality blending the way real SVC would allow.

---

## 8. Environment design discussion

### 8.1 What the environment models

Following the same environment/agent split as MMSP'25 Fig. 5, adapted to the formulation's richer structure (discrete tile selection instead of continuous enlargement; the post-decision-state split; the virtual-experience independence assumption):
- **Environment holds and evolves**: playback buffer level, viewport-prediction error, and the aerial-to-ground channel state — the three components of `ω^t`.
- **Agent chooses**: which tiles to enhance (`x^t`) and how much transmit power to use (`P^t`).
- **Reward** combines a component known immediately after the action (stall time, power cost, tile-count cost) with a component depending on the as-yet-unrevealed actual viewport — exactly the formulation's post-decision-state mechanism, which is *not* present in MMSP'25's simpler model (no PDS or virtual-experience layer there at all).

### 8.2 Computation timing breakdown

**Loaded once, at environment construction**: `video_catalog`, the 194 valid `HMD_data` traces, `video_bitrate_data`/`video_ymse_data`, the tile grid constant (8×8), and (if the "refit" R-D path is chosen) the refit `a`/`b` coefficients, computed once and cached.

**Computed at `reset()`**: which video/user-trace the episode replays; initial `Z^0` (design choice); initial `Δ^{-1}_θ,Δ^{-1}_φ` (design choice); initial `h^0` (first channel draw), and initial UAV distance `d` if mobility is modeled; a bootstrap convention for the first prediction (no `t-1` viewport yet).

**Computed at each `step()`, before the agent acts** (composes the observed state `ω^t`): `Z^t` and `Δ^{t-1}_θ,Δ^{t-1}_φ` carried from the previous step; `h^t` sampled fresh from the channel model.

**Computed at each `step()`, after the agent acts**: `E^t(x^t)` from the action; `A^t` (via whichever model is chosen — Section 7, Section 9); `R^t` (Shannon formula); `D^t`; `Z^{t+1}` (deterministic — this whole batch is exactly the post-decision state `ω̃^t`); `r^t_known`.

**Computed after observing the random outcome**: `V^t_actu` (revealed from the real trace); `h^{t+1}` (fresh draw); `Δ^t_θ,Δ^t_φ`; `C^t_cov`/`Q̄^t_viewport`; `r^t_random`, then `r^t = r^t_known + r^t_random`; `ω^{t+1}` handed back to the agent.

### 8.3 Where does each quantity actually live? (dataset vs. cache vs. simulated state)

A question worth being precise about: nothing computed at runtime ever gets written into `hn.mat`/`rd.mat`, or treated as a new piece of "dataset." Three distinct buckets:

- **Real dataset — read-only, never modified.** `hn.mat`/`rd.mat` stay exactly as extracted. The environment only ever *reads* from them (via `scipy.io.loadmat`, same as `explore_data.ipynb`).
- **Precomputable derived cache — compute once, reuse across every episode.** Things that depend only on the dataset itself, not on any action or randomness: tile-boundary angles for whichever FOV is chosen, refit R-D curve coefficients (if that path is chosen), the parsed video catalog, the list of valid traces. These are worth computing once at environment construction and keeping in memory (optionally saved to a small separate cache file) — but this is a *new derived artifact*, never an edit to the original `.mat` files.
- **Per-step simulated state — cannot exist before the environment runs.** Channel gain, `A^t`, `R^t`, `D^t`, `Z^t`, reward, coverage. These either depend on fresh randomness (channel) or directly on the action the agent just chose (everything downstream of `A^t`) — by definition they don't exist until a `step()` call produces them. They live only in the environment's code/memory during a training run.

**Channel-gain timing, specifically.** From the MDP's perspective this is dynamic: the agent only ever observes `h^t` at time `t`, and `h^{t+1}` is drawn only *after* the action for step `t` — matching the post-decision-state formalism exactly (`h^{t+1}` is explicitly one of the quantities "still unknown at the PDS"). Since channel noise doesn't depend on the action at all (this is exactly the independence the virtual-experience mechanism relies on), there's a free implementation choice with **no behavioral difference**: draw `h^{t+1}` fresh inside every `step()` call, or pre-generate a whole episode's worth of channel draws in one shot at `reset()` and have `step()` just index into that array (marginally more efficient, useful since PPO training typically runs many environments in parallel). Both produce statistically identical training data — this is a performance choice, not a modeling one.

**Contrast with the actual viewport.** Unlike channel gain, `V^t_actu` isn't generated at all — it's read directly from the real, already-loaded `HMD_data` trace for whichever user/video the episode is replaying. No randomness, no formula: the environment just advances an index into an array that already exists in memory.

### 8.4 Gym interface — what it actually gives you

- `reset()` → returns the first observation `ω^0` (a numeric vector: buffer level, previous prediction errors, channel gain) plus an `info` dict. This is what feeds into the PPO policy network to produce the very first action.
- `step(α^t)` → returns five things, every gymnasium environment must return exactly these:
  1. **observation** (`ω^{t+1}`) — the next state vector, fed into the policy for the next decision.
  2. **reward** (`r^t`) — a single number. This *is* the training signal — PPO's entire job is adjusting the policy so the expected sum of these numbers over an episode goes up. Everything else the environment computes (buffer, channel, coverage) only matters insofar as it feeds into this one number.
  3. **terminated** — whether the episode ended naturally (e.g. the replayed trace ran out of frames).
  4. **truncated** — whether the episode was cut off artificially (e.g. a fixed max-steps limit for training efficiency), distinct from a real ending.
  5. **info** — a free-form dict for anything useful to log/plot without influencing training — e.g. stuffing `C^t_cov`, `D^t`, `A^t` in here lets progress be monitored (is quality actually improving? is stall time actually low?) without those values leaking into the reward twice.
- `observation_space`: a `Box` over `(Z^t, Δ^{t-1}_θ, Δ^{t-1}_φ, h^t)`.
- `action_space`: combines discrete/binary tile selection (`MultiBinary(64)` or similar) with continuous power (`Box`) — a harder, mixed action space than MMSP'25's three continuous scalars, consistent with the plan to bin power initially.
- **Why this matters**: PPO never sees any of the internal formulas (channel model, buffer recursion, QP tables) directly — it only ever sees the `(observation, reward, done)` sequence `reset()`/`step()` produce. The environment is a black box from PPO's point of view. Which means: if the reward or state math is wrong, PPO will still run without crashing, and will still "learn" something — just the wrong thing. This is exactly why getting the modeling decisions right (Sections 7–9) matters as much as writing the code itself.

---

## 9. Modeling choices — every fork, consolidated

Every "we could do this or that" raised so far, in one place. None of these are resolved here — they're presented with tradeoffs for discussion.

1. **One critic vs. two critics.** Two critic networks were used in a related IoT Congress paper; starting with one for simplicity, experimenting with two later, is the alternative.
2. **Discrete vs. continuous states/actions.** Starting discrete (including binning continuous power into discrete levels) before moving to continuous is the incremental option.
3. **R-D quality model: refit vs. lookup.** Refit `D=a·e^(bR)`/`D=a·R^b` curves from the real bitrate/YMSE points via `scipy.optimize.curve_fit` (matches the dataset's own methodology most faithfully, extra engineering work) **vs.** use the raw 7 discrete QP operating points directly as a lookup table (simpler, still real-data-grounded, no continuous curve).
4. **`A^t` model: formulaic vs. QP-grounded.** MMSP'25's continuous formula (`A_base`, `C`, `η` as fixed constants, content-independent, simple/closed-form) **vs.** a QP-table-grounded model (pick specific QP indices from `rd.mat`, sum real measured bitrates per selected tile — more faithful to the actual dataset, but makes `A^t` — and therefore buffer/stall/reward — depend on which video/GOP is currently playing). Mechanics of the QP-grounded option spelled out in full in Section 7.
5. **Which QP index = base, which = enhanced.** Unaddressed by either paper — MMSP'25 doesn't use QP indices at all; the formulation doesn't specify which of the 7 levels. Pure open decision (Section 7).
6. **UAV mobility: full dynamics vs. static.** Model the full Gauss-Markov distance dynamics (MMSP'25 eqs. 2a–2c) **vs.** the static special case MMSP'25 actually reports results for (`d=50m` fixed, only Rician fading varies per step) — simpler to start.
7. **FOV source.** Adopt `V_θ=60°,V_φ=30°` as-is (MMSP'25's experimental choice) **vs.** track down the dataset's own capture-rig FOV (Oculus Rift + Whirligig, per the MMSys'21 paper) before committing.
8. **Channel model parameters.** Reuse MMSP'25's Rician/path-loss parameters as-is **vs.** different/updated UAV-specific values Jacob may have.
9. **Training scope: per-video vs. generalizable policy.** Two genuinely different options, not just "how many videos": (a) train and evaluate a separate policy per video — each specialized/overfit to that video's specific content and R-D characteristics, simpler to train, and matches MMSP'25's own precedent exactly (their reported results only ever use `Runner`, a single video — they don't attempt cross-video generalization either); (b) train one policy across multiple/all videos, aiming for a policy that generalizes to content it hasn't specifically seen — more realistic for eventual real-world deployment, but harder to train, and worth flagging: the state `ω^t` as currently defined (`Z^t, Δ^{t-1}_θ, Δ^{t-1}_φ, h^t`) contains nothing that identifies which video/content is currently playing, so a single policy has no direct way to condition its behavior on content difficulty — it would have to infer that indirectly, which may or may not work well; might need the state augmented with a content-descriptor if true generalization is the actual goal. Either way, the rigorous way to *measure* whether generalization happened, rather than assume it, is a held-out split: train on some videos, evaluate on others never seen during training.
10. **Yaw/pitch/roll → (θ,φ) mapping convention.** Neither paper confirms whether `yaw` maps directly onto azimuth `θ` and `pitch` onto altitude `φ` with matching sign/range conventions — needs to be checked against `hn.mat`'s actual value ranges (Section 5.2) before implementing. Roll is dropped entirely under MMSP'25's model — that simplification is available here too, but is itself a choice.
11. **Simulcast/swap accounting vs. pursuing real scalable-coded data.** Given the dataset only supports independent per-QP encoding (Section 7), the practical option is to implement enhancement as a QP swap with additive accounting **vs.** seeking or producing genuinely SVC-encoded data to get true additive layers (no such data currently available; likely not worth the effort, but worth stating explicitly rather than silently assuming).
12. **Partial tile visibility rule.** The actual viewport is a continuous rectangle in (θ,φ) space; tiles are discrete 8×8 grid cells — a tile will often be only *partially* inside the viewport, not cleanly in or out. Three ways to handle it: (a) *any-overlap* — a tile counts as "in" if it overlaps the viewport at all (simple, tends to over-include tiles near the viewport's edges); (b) *threshold* — a tile counts as "in" only if some fraction (e.g. ≥50%) of its area overlaps (a common heuristic in 360-video literature, requires picking a threshold value); (c) *fractional/area-weighted* — don't force a binary decision at all; compute the exact overlap fraction per tile and use it as a continuous weight in the coverage metric. Worth noting as a textual observation, not a recommendation: eq. (11)'s own definition of `C^t_cov` is literally an area intersection over the actual viewport's area, which (c) matches most directly. The same question applies wherever a *predicted* viewport rectangle gets converted into the "always-enhance" tile set, too.

---

## 10. Open questions for Prof. Jacob

1. Reuse the MMSP'25 channel-gain model/parameters as-is, or does Jacob have different/updated UAV-specific values?
2. Model UAV mobility from the start, or begin with the static-distance special case?
3. Which QP index(es) should represent "base" vs. "enhanced" quality?
4. Confirmed understanding to validate: because the dataset only has independent per-QP encodings (not true scalable/SVC streams), "enhancing" a tile in practice means **replacing** its transmitted stream with a higher-quality one for that tile only — not adding a second stream on top of the base layer. The base layer is still sent for every tile, every GOP, unconditionally; enhancement swaps in a better version for selected tiles. Data-volume accounting treats this as additive (`A^t = A_base + deltas`) purely for the bitrate math (Section 7). Does this match what Jacob has in mind, or is genuinely additive/scalable encoding expected?
5. Adopt `V_θ=60°, V_φ=30°`, or track down the dataset capture rig's actual documented FOV?
6. Refit R-D curves from raw data, or use discrete QP-level lookups?
7. Train a single policy across multiple videos aiming for generalization, or train/evaluate per-video independently (matching MMSP'25's own single-video precedent) — and if generalization is the goal, is the current state definition (Section 9, item 9) rich enough, or does it need a content-descriptor added?
8. One critic or two, given the richer post-decision-state/virtual-experience structure this project adds beyond MMSP'25's simpler model?
9. Is there a standard/preferred rule (any-overlap, threshold, or area-weighted — Section 9) for handling tiles that are only partially inside the viewport, or a convention from Jacob's prior work?

---

## 11. What's already resolved / confirmed

- Channel-gain formula found, with full numeric parameters (Section 6).
- Base/enhancement mechanics now understood precisely: the dataset only supports independent per-QP encoding (not true scalable coding), so enhancement is implementable as a QP-level swap with additive data-volume accounting (Section 7).
- Tile grid confirmed 8×8 (for `Runner`; assumed but not independently verified for the other 14 videos).
- GOP size confirmed = 30 frames, empirically, for all 15 videos (Section 5.3).
- "Last viewport as prediction" baseline confirmed paper-precedented (Section 6), not an invented shortcut.
- Buffer/stall equations confirmed identical in structure across the formulation and MMSP'25 (Section 6).
- Actual-viewport-from-viewing-direction rule now in hand: a rectangle in (azimuth, altitude) space (Section 6).
- The MATLAB `cfit` R-D curves are confirmed unreadable via Python, but confirmed *not blocking* — the raw numeric data they were fit from is sufficient on its own (Section 5.5).
- 194 valid head-navigation traces identified and separated from placeholder/empty cells (Section 5.2).
- Channel-gain generation timing clarified: conceptually dynamic (the agent only ever sees `h^t` at time `t`, never ahead), but since channel noise doesn't depend on the action, it can equally be implemented as a pre-generated-at-reset array purely for efficiency, with no behavioral difference (Section 8.3).
- Architecture principle clarified: the dataset is never modified or written to — everything computed at runtime (channel gain, buffer, rate, stall, reward) lives in the environment's own code/memory, separate from the source data (Section 8.3).

---

## 12. Documentation inconsistencies found in the dataset

- README states `HMD_data` is `12×12`; actual shape is `(15,34)`.
- Initial guess of "≥14 placeholder cells" (from dtype heuristics) vs. the measured, verified figure of exactly **1** placeholder cell.
- README's "N=frame index" label means genuinely per-frame for `video_bitrate_data`/`video_ymse_data`, but per-*GOP* for `RtoD_exp`/`RtoD_pow`/`QPtoR_exp`/`QPtoR_pow` — same phrase, two different meanings.
- Dataset paper claims 121 total head-navigation traces (5–12 per video); actual measured count is 194 valid traces, with 5 of 15 videos exceeding the claimed max of 12.
- README's own text for `QPtoR_pow` states `D=a·QP^b`, inconsistent with the variable's name (`QPtoR` implies QP→Rate) and inconsistent with the pattern used one line above for `QPtoR_exp`.
- MMSP'25 describes `Runner` as "8K-60fps"; the dataset's own README lists it at 30fps.

---

## 13. Next steps

1. Meet with Jacob, work through Sections 9–10 above.
2. Lock in the modeling decisions (R-D model, `A^t` model, QP base/enhanced split, partial-tile-visibility rule, mobility on/off, FOV source, channel params, training video set, critic count).
3. **Build the environment** — a `TileStreamingEnv(gymnasium.Env)` class:
   - `__init__`: load `hn.mat`/`rd.mat` once via `scipy.io.loadmat` (same approach as `explore_data.ipynb`); build/cache whichever derived quantities were decided in step 2 (tile-boundary angles for the chosen FOV, refit R-D curves if that path was chosen, the list of valid traces); declare `observation_space`/`action_space` (Section 8.4).
   - `reset()`: pick an episode's video + user trace; set initial buffer/prediction-error/channel values; return the first observation.
   - `step()`: run the full per-step chain from Section 8.2 — action → `E^t(x^t)` → `A^t` → `R^t` → `D^t` → `Z^{t+1}`, then reveal the next real frame's actual viewport from the trace → coverage/quality (using whichever partial-visibility rule was chosen) → reward → draw the next channel gain → assemble the next observation.
4. **Sanity-test the environment before training anything** — run it standalone with a random policy for a few episodes and check the numbers make sense (buffer doesn't behave impossibly, stall time isn't constantly saturated, reward isn't always zero/constant). Same "verify it actually runs and produces sane output" discipline already used for `explore_data.ipynb`.
5. **Wire up PPO** (likely via `stable-baselines3`, not yet installed in the project) against the environment, starting with the discrete/binned action space per the incremental-implementation plan.
6. **Train a first baseline**, and compare it against a trivial reference policy (e.g. always send only the base layer, or a fixed/random tile selection) to confirm the agent is actually learning something better than doing nothing clever.
7. Once that loop works end-to-end, add the post-decision-state value function and virtual-experience batching from the problem formulation's Algorithm 1.
