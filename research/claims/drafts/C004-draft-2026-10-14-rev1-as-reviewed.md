# C004 (DRAFT, not registered): The read coordinate of `L03.H04` at the cue position causally carries the one/two plural shift in new coordinated frames of pinned Pythia-70M

> **Draft for review only. Not in the claim registry.** Drafted on 2026-10-14 from Experiment 101's closed record. The
> user decides whether it is registered, at what maturity, and with which changes. It does not change C002 or C003.
> Every review cited is an AI review, not a human validation.

## Metadata

- Proposed maturity: `CAUSAL_EVIDENCE` (Level 3), narrowly scoped. Not `FALSIFICATION_TESTED`: the specificity,
  held-out-axis and baseline challenges are untested (see Level 4).
- Proposed disposition: `ACTIVE`
- Created: 2026-10-14 (draft)
- Origin: **confirmatory** — the single preregistered confirmation run of Experiment 101 (lock `8da97e35…`, committed by
  the user in `961e814`; the program's prediction table committed by the user in `59d8ec4` before any target prompt).
  The variable itself was defined earlier from exploratory and confirmatory work: the P1 read form of `L03.H04` (009,
  010) and the axis `d̂_T` estimated on exposed `one`/`two` and `each`/`several` pairs (011's lock).
- Related experiments: 101 (primary); 005 (localization, exploratory); 009–011 (the read form; C003's evidence).
- Related claims: C002 (localization of `L00.MLP → L03.H04 → L04/L05 MLPs`) and C003 (prospective prediction by decoded
  programs). This claim sits beside both and changes neither.

## Hypothesis

In pinned `EleutherAI/pythia-70m-deduped`, the scalar read coordinate

    r(x) = ⟨x, w_c⟩,   w_c = γ₃ ⊙ m − mean(γ₃ ⊙ m)·𝟙,   m = W_V W_O d̂_T   (head 3.4)

of the residual `x` entering layer 3 (`RESID_PRE.L3`) at the cue position of a coordinated frame carries the
singular/plural information that the model uses for the noun-number choice at the prediction position, for the cue pair
`one`/`two`:
- setting `r` in the plural (`two`) prompt to the reference (`one`) prompt's value removes the plural shift of the
  noun-number contrast;
- setting `r` in the reference prompt to the plural prompt's value produces that shift;
- equal-length edits along the cue's own centred off-read change do neither;
- doubling the read change makes the plural prompt more plural, and reversing it makes the reference prompt more
  singular.

## Scope

- **Model:** `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, float32 on CPU;
  geometry and edits computed in float64 and cast to float32.
- **Site:** `RESID_PRE.L3` (`blocks.3.hook_in`) at the cue position `p_c`; the residual at the prediction position
  `p_t = p_c + 1` is not edited.
- **Frames:** the 36 frozen new frames of Experiment 101 (candidates 1–36 of its design §11), all of one coordinated
  template `NAME and NAME VERBED {cue} ADJ`; `p_c` 4–6.
- **Cues:** `one` (581) as reference and `two` (767) as plural; `three` and `many` only as a descriptive stratum.
- **Outcome:** `Δc`, the change in `c = log P(singular) − log P(plural)` at `p_t`, averaged over the 79 scorable
  **exposed** nouns that every experiment since 020 scores (the nouns are not fresh; freshness is in the frames).
- **Interventions:** exact REPLACE of `x(p_c)` at `RESID_PRE.L3` (landing and zero-outside-change checked), one forward
  pass per arm.

## Operational definitions

- `Δx = x_two − x_one` at `RESID_PRE.L3`, `p_c`; `Δr = ⟨Δx, w_c⟩`; read part `(Δr/‖w_c‖²)·w_c`; `off_c` the centred
  remainder; `û_off = off_c/‖off_c‖`; `s = |Δr|/‖w_c‖`.
- Arms: NEC-READ `x_two − read`; SUF-READ `x_one + read`; NEC-OFF `x_two − s·û_off`; SUF-OFF `x_one + s·û_off`; NEC-AMP
  `x_two + read`; SUF-NEG `x_one − read`.
- Fractions of `Δc_full = Δc(PL)`: `F_rem = Δc(NEC-READ)/Δc_full`, `F_rem_off`, `F_ins = Δc(SUF-READ)/Δc_full`,
  `F_ins_off`; `N_f = F_rem_off − F_rem`; `S_f = F_ins − F_ins_off`; `A_amp = Δc(NEC-AMP) − Δc_full`;
  `A_neg = Δc(SUF-NEG)`.
- A frame is weak when `Δc_full > −1` nat (none were).

## Predictions (preregistered before any target prompt; Experiment 101's lock and design §7)

- R: `N_f > 0` in at least 26 or 25 of 36 frames by Holm at 0.025, median `N_f ≥ 0.05`.
- I: `S_f > 0` likewise.
- D (only if R and I pass): `A_amp < 0` and `A_neg > 0` each in at least 25 of 36 frames.
- P (only if D passes): the frozen program's `F̂_rem`, `F̂_ins` within `max(τ, 0.02)` of the measured ones in at least 27
  of 36 frames (τ_rem 0.0252, τ_ins 0.0351 from the exposed calibration).
- Precondition: at least 30 of 36 frames with `Δc_full ≤ −1`.

## Evidence

Paths are relative to `experiments/101-read-component-intervention/` unless stated.

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E101-1 | Experiment 101 confirm | supporting (confirmatory) | `evidence/final-report-2026-10-14.md`; `evidence/confirmation-record-2026-10-14.json` | `READ_COMPONENT_EFFECTS_AS_PREDICTED`: precondition 36/36; R 36/36, median N 1.006 (bootstrap 95 % 0.988–1.051); I 36/36, median S 1.035 (1.017–1.059); D 28/36 and 36/36; P 33/36 | report `54f883ad…`; final state `c803fe97…`; run at `59d8ec4` |
| E101-2 | Experiment 101 confirm | measurement validity | `evidence/confirm-2026-10-14/`; the record's `gates` | G1–G7 held: G2 edit geometry within tolerance by ≥ 181×, G3 pass-through by 137×; NOOP bit for bit 36/36; `x3(p_t)` unchanged 1,692/1,692; G4 Level-1 6.14e-3 against 2e-2; G6 bit for bit; 1,980 keys once each | launcher `56a687f3…` |
| E101-3 | independent confirmation review | verification | `evidence/confirmation-review-2026-10-14/REVIEW.md` | AI review, not human: PASS WITH NOTES, no blocker; every primary number recomputed bit for bit | review digest in its `SHA256SUMS` |
| E101-4 | Experiment 101 confirm | descriptive | `evidence/closure-2026-10-14.md` § What the result supports | medians: `F_rem` −0.009, `F_rem_off` 1.000, `F_ins` 1.004, `F_ins_off` −0.033; the read part is ~14 % (8.6–18.9 %) of the cue change's centred squared length | — |
| E101-5 | Experiment 101 secondary | descriptive, no label force | closure § Secondary results | head-only arms: removal share 0.882, installation share 0.894 (value-only 0.877, 0.892); isotropic controls beaten 36/36 both ways | — |
| E101-6 | Experiment 101 secondary | descriptive, partly contradicting | closure § Secondary results | stratum: `three` R 36/36, I 36/36, D 28/36 & 36/36; `many` R 36/36, I 36/36, `A_amp` < 0 only 23/36 (would fail D) | — |
| E101-7 | Experiment 101 calibration | exploratory (exposed frames) | `evidence/calibrate-2026-10-14/`, `calibration-v1.json` | the same pattern on 36 exposed frames: `F_rem` −0.035, `F_ins` 0.980; N, S > 0 36/36; `A_amp` < 0 30/36 | calibration `1eb1fe68…` |
| E101-8 | Experiment 101 protocol | integrity | `evidence/stage1-review-2026-10-14/`, `evidence/confirmation-review-2026-10-14/` § 4 | the prediction table was committed (`59d8ec4`) and pushed before the S2 ledger (17:38:22Z); no tolerance or threshold differs from the lock | lock `8da97e35…`; table `91918004…` |
| E005-A3 | Experiment 005 discovery | localization (exploratory) | `experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md:29–36` | whole-residual patch at `p_c` (coordinated): recovery 0.956 at the layer-3 input, 0.125 at the layer-4 input | — |

## Evidence gates

### Level 0 — Measurement validity

- [x] Quantities and success criteria operationally defined and frozen before the target data (101's design §4–§8,
  lock). Evidence: E101-1, E101-8.
- [x] Inputs, tokens, positions and targets recorded (101's `confirmation-v1.json`, stage-1 files). Evidence: E101-8.
- [x] Reconstructions agree with ground truth within documented tolerances (G2–G6). Evidence: E101-2.
- [x] Model revision, dependencies, runtime and artifacts recorded; reproducible from version control. Evidence:
  E101-2, E101-8.

### Level 1 — Observation

- [x] Measured directly in 36 frames, every frame shifting (`Δc_full` −6.58 to −3.95). Evidence: E101-1.
- [x] Variation reported (per-frame values; bootstrap intervals). Evidence: E101-1, E101-3.
- [x] Labeled confirmatory (single preregistered run). Evidence: E101-8.

### Level 2 — Localization

- [x] Narrowed to a defined component (one coordinate) at a defined site (`RESID_PRE.L3`, `p_c`). Evidence: E101-1.
- [~] Plausible alternative locations compared — **exploratory only**: 005's residual profile places the cue's
  consumption at layer 3 (E005-A3); 101 compares the read part with the whole position (PC-SWAP) and with the off-read
  remainder, but tests no other layer or position prospectively. Evidence: E005-A3, E101-4.
- [x] Obvious artifacts tested (NOOP identity, landing, pass-through, LayerNorm scale records). Evidence: E101-2.
- [x] Replicates within scope: exposed calibration and new confirmation frames agree. Evidence: E101-7, E101-1.

### Level 3 — Causal evidence

- [x] Intervention and predicted outcome documented before the result (the lock and the prediction table committed by
  the user before any target prompt). Evidence: E101-8.
- [x] Necessity-oriented intervention supports the prediction (R: removal, 36/36). Evidence: E101-1.
- [x] Sufficiency-oriented intervention supports the prediction (I: installation, 36/36). Evidence: E101-1.
- [x] Matched and negative controls: equal-length edits along the cue's own centred off-read change (the primary
  matched control) and 8 isotropic directions (predicted null, observed null); NOOP; D's doubling and reversal
  distinguish a read effect from disruption. **Not included: a site or layer negative control** (recorded under
  Limitations and Level 4). Evidence: E101-1, E101-5.
- [x] Effect sizes and uncertainty reported (medians with frame-bootstrap 95 % intervals; exact binomial tails).
  Evidence: E101-1, E101-3.
- [x] Alternative causal explanations considered (closure § Simpler or competing explanations; review § 6.3).
  Evidence: E101-3.

### Level 4 — Falsification tested

- [ ] Counterexamples actively sought — only the descriptive stratum (`many` would fail D). Not done: other templates,
  cue-final frames, held-out cue pairs.
- [ ] Competing explanations tested with discriminating predictions — site and layer specificity, an axis derived
  without `one`/`two`, and simpler baselines (difference-in-means, probe, gradient directions at equal norm) are
  untested.

### Level 5 — Generalized

- [ ] Not claimed. New frames are the only changed axis; the cue pair, template, nouns, site and checkpoint are fixed.

### Level 6 — Algorithmic explanation

- [ ] Not claimed. The program's predictions held in mean fractions (P), but per-noun ceiling errors reach about 1 nat
  and no compact mechanism beyond this stage is claimed.

## Contradicting evidence and failed tests

- **The stratum `many`** would read `…DIRECTION_NOT_ESTABLISHED` under the primary rules (`A_amp` < 0 in 23/36; `F_rem`
  0.21; 15 % of its shift does not pass through `p_c`). Descriptive, no label force. Evidence: E101-6.
- **D is the narrowest family** (28/36 against 25), limited by saturation: the head's transported signal nearly doubles
  under NEC-AMP while `Δc` moves −0.136 nats. Evidence: E101-1, E101-3.
- **P disagrees in 3 of 36 frames** (03, 08, 09), and the frozen readout's per-noun errors reach about 1.06 nats
  (SUF-READ@two) and 1.20 nats (NEC-READ@many) even with exact inputs; P is a check in mean fractions only.
  Evidence: closure § per-arm ceilings.
- **Earlier routes** to causal status (005, 006, 007, 025) did not establish it, on different variables and criteria.
  They do not contradict this claim, but they are retained as the record's context (C003 boundary).

## Competing explanations

| alternative | status | discriminating test |
|---|---|---|
| Generic disruption of the cue's information | excluded (D's signs; inert equal-length off-read and isotropic edits) | — |
| The read edit is larger after LayerNorm | excluded (equal centred lengths; post-LN sizes comparable) | — |
| Subspace-patching / dormant-direction illusion (B8) | largely excluded by the complementary arms (the read part alone reproduces the whole-position swap; the natural remainder alone does nothing) | rowspace/nullspace checks; on-manifold tests |
| **Definitional readout:** `w_c` is the head's read, so editing it moves the head's output by construction | partly true for the head route; the empirical content is that the remainder is inert through **all** readers and that the head's output is the bottleneck for ~88–89 % | compare with directions not derived from the head (below) |
| **Any data-derived direction would do** (difference-in-means, linear probe, gradient, DAS) | untested | equal-norm baselines at the same site (proposed Experiment 102) |
| **Site or layer non-specificity** | untested in 101 (005's exploratory profile suggests layer-3 consumption) | the same edit at `RESID_PRE.L4`, at other positions |
| **In-sample axis** (`d̂_T` estimated with `one`/`two`) | not excluded for this pair; `three` transfers descriptively | an axis derived from other cue pairs only |
| Off-distribution composite states | partly addressed (swaps use natural values), not excluded | on-manifold or natural-variation tests |
| Other readers carry the effect | within scope by design; ~11–12 % outside `L03.H04`'s output | decompose by component |

## Relationship to prior work

From the project's literature review (`research/paper/literature-review/review-2026-09-28.md`):
- **Agreement mechanisms:** Finlayson et al. (2021, A5) — causal mediation of number agreement in GPT-2-class models;
  Lakretz et al. (2019, A14) — number units; Arora, Jurafsky and Potts (2024, A7, CausalGym) — one-dimensional
  intervention directions for agreement tasks in Pythia models including 70M (DAS, probes, difference-in-means), with
  the caution that efficacy alone does not show the model uses the variable.
- **Causal evaluation:** causal abstraction and DAS (B1, B2), the subspace-patching illusion and its reply (B8).
- **Nearest difference:** here the direction is derived from a circuit component's weights (not trained on the target
  behavior), tested with complementary arms (read part vs the cue's own remainder) and predicted by a frozen program
  committed before the target data.

## Potential novelty if confirmed

Partly new and narrow (subject to systematic comparison): a weight-derived, one-dimensional read of a localized head
that carries the whole tested effect at its site while the rest of the cue's change is inert, shown with preregistered
removal, installation, direction and program-prediction tests. Not new: that number agreement has low-dimensional causal
directions in small LMs (A7).

## Supported conclusion

> Within pinned Pythia-70M-deduped, the 36 frozen coordinated frames of Experiment 101, the cue pair `one`/`two`, the 79
> exposed scorable nouns and exact replacement edits of the residual entering layer 3 at the cue position, the evidence
> supports: the read coordinate `⟨x, w_c⟩` carries essentially the whole `one`→`two` shift of the mean
> singular–plural contrast (removal leaves a median −0.009 of it; installation produces 1.004), while an equal-length
> edit along the cue's own centred off-read change carries essentially none (1.000 left; −0.033 produced); doubling and
> reversing the read change move the contrast in the predicted directions (28/36, 36/36 frames); and a frozen program,
> committed before any target prompt, predicted the removal and installation fractions within calibrated tolerances in
> 33/36 frames.

## Not supported

- That `L03.H04` alone mediates the effect (its output carries ~88–89 % in the secondary head-only arms).
- Any generalization beyond the pair, the site, the template, the nouns, the checkpoint and these frames; in particular
  to `many` (its direction family would fail) or to cue-final frames.
- That `w_c` is the unique, optimal or "true" variable, or that a simpler data-derived direction would not do as well.
- That the decoded program is the model's algorithm (Level 6), or that per-noun predictions hold.
- Any population-level rate: the binomial tails treat these 36 frames of one template as exchangeable units under a
  sign-symmetric null.
- That this demonstrates a general method for finding causal variables in LLMs (untested; the proposed 102 tests it).

## Limitations

- The axis `d̂_T` was estimated with the same `one`/`two` pair; freshness is in the frames only; nouns are exposed.
- One template, one site, one checkpoint; no site or layer negative control; a one-dimensional manipulation check.
- Edited states are novel composites (off-distribution risk), though the swaps use the other prompt's natural `r`.
- One confirmation run, no replication; every review is an AI review; no external human expert has reviewed it.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
| 2026-10-14 | — | (proposed) `CAUSAL_EVIDENCE` | — | (proposed) `ACTIVE` | drafted from Experiment 101's closed record; **not registered** | E101-1 … E101-8, E005-A3 |
