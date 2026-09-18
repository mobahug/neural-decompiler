# Experiment 008: Where Is the Apparent Number Signal of `this`-like Cues Suppressed? — A Causal Localization on the Exposed Pool

**Date:** 2026-09-18

**Status:** Revision 3 (pre-measurement clarification of M3's baseline, made during implementation planning; no
threshold, label, or other measurement changed). Revision 2 was approved for implementation planning. No Experiment
008 model run exists. Experiments 005–007 are closed and are not amended by this document.

**Kind:** Discovery-only, discriminating. No confirmation set is frozen, no lock is written, no claim is promoted.
The deliverable is a mechanically derived localization statement that becomes the hypothesis of a later prospective
experiment (009), which will freeze genuinely new cue tokens and frames before testing it.

## Purpose and question

Experiment 007 found a response-supervised rank-1 direction `u₁` of the token-local `L00.MLP` encoding that predicts
the E-patch response of unseen cue tokens about as well as a full-dimensional ridge map, and it found a concentrated
counterexample: the singular-selecting determiners `this` (E-patch shift +0.01 nats, program −2.80) and `another`
(−0.32, program −2.76), joined in the exposed pool by `a` (+0.30, program −3.14) and `every` (+0.41, program −1.33),
produce almost no downstream number effect although their `E(w)` projects strongly onto the plural side of `u₁`.
There is also structure among the determiners that is not "singular versus plural": `this` ≈ 0, `that` ≈ −2.7,
`these` ≈ −5.2, `those` ≈ −5.1.

> Why do cues such as `this` and `another` produce near-zero E-patch effects although their token-local `L00.MLP`
> representation contains a strong plural-side projection? Where along the causal path is that apparent number signal
> suppressed, transformed, or cancelled?

The experiment answers *where* by a staged causal trace of the E-patch intervention, and *how* at the encoding stage by
component patching, on the whole exposed pool. It does not fit a more flexible predictor.

## Competing hypotheses (kept genuinely competing)

- **H0 — axis artifact.** The apparent signal is a property of the 007 direction `u₁`, not of the mechanism: along the
  mechanism's own encoding number axis `d̂_E` (estimated from the original cue pairs), `ΔE_T(this)` carries little
  number signal. Nothing is suppressed; the 007 program mispredicts because `u₁` is not the axis the network reads for
  these tokens.
- **H1 — nonlinear transformation after E.** `ΔE_T(w)` does contain the axis signal, and that component alone would
  drive a plural shift, but the network's response to the whole `ΔE_T(w)` is not the sum of its parts: an
  orthogonal component of `ΔE_T(w)` gates or transforms the signal at the cue position (layers 1–2) or later.
- **H2 — cue-position context interaction.** The effect of an encoding depends jointly on `E(w)` and on the other
  residual components at the cue position (embedding, layer-0 attention); an ordinary plural encoding placed in a
  `this`-cue context is suppressed, or `E(this)` behaves differently in an ordinary context.
- **H3 — transport difference.** The signal survives at the cue position up to layer 3, but `L03.H04` does not carry
  the relevant part of the `this`/`another` residual to the noun position the way it carries numeral and quantifier
  cues (attention to the cue position, or the value mapping).
- **H4 — late cancellation.** Transport occurs — the head's output at the noun position carries the plural signal —
  but later components (the layer-4/5 MLPs, other heads, or the readout) contribute opposing direct effects that
  cancel it.

A fifth possibility, **linear cancellation inside E**, is a sub-case distinguished at the encoding stage: the axis
component of `ΔE_T(w)` drives a plural shift on its own, the orthogonal component drives an opposite shift on its own,
and the two add linearly. It differs from H1 (non-additive) and is recorded as its own class.

## Inherited fixed elements

- Model, runtime, instrumented forward path (`use_attn_result=True`, compatibility mode off), contrast
  `c(x) = log P(sg) − log P(pl)`, exact final LayerNorm (population variance, `ε = 1e-5`), reference cues `ref_T` (`one`
  for cardinal and coordinated-adjective, `each` for quantifier), reference prompts, the weight-only encoder
  `E(w) = MLP₀(ln2₀(W_E[w]))`, the E-patch intervention (replace the `L00.MLP` output at `p_c` in frame `f`'s reference
  prompt by `E(w)`), the E-patch shift `Δc(w, f) = mean over nouns of c_patched − c_ref`, the fixed circuit
  E = `L00.MLP` at `p_c`, T = `L03.H04`, R = {`L04.MLP`, `L05.MLP`} at `p_t`, the exact per-run direct-effect
  decomposition of `c` through the final LayerNorm (Experiment 005's `direct_effects`, identity within 1e-4 nats),
  and the site-axis estimator (unit mean difference of paired vectors with its scale) — all exactly as in
  Experiments 005–007.
- Seeds: runtime `20260916`; control `20260921` (recorded only).

## Exposed pool (everything already executed by Experiments 005–007; nothing fresh)

- **Frames (18):** the six manifest frames, the six Experiment 005 extension frames, and the six Experiment 006/007
  confirmation frames (`The crate holds`, `The gallery shows`, `The index lists`, `The archive keeps`,
  `Ravi and Elena sorted … plain`, `Nora and Felix stacked … heavy`). Cardinal and quantifier frames are cue-final
  (`p_c = p_t`); the coordinated-adjective frames separate the cue and the prediction position.
- **Cue tokens (40):** the sixteen exposed tokens (`one`, `two`, `each`, `several`, `a`, `the`, `three`, `four`,
  `five`, `ten`, `many`, `few`, `some`, `all`, `both`, `every`) and the twenty-four Experiment 007 confirmation tokens
  (`any`, `no`, `another`, `single`, `multiple`, `numerous`, `twelve`, `hundred`, `six`, `seven`, `eight`, `nine`,
  `dozen`, `countless`, `various`, `fewer`, `this`, `that`, `these`, `those`, `big`, `red`, `old`, `fresh`).
- **Nouns (80):** the sixty Experiment 005 nouns and the twenty Experiment 006/007 confirmation nouns; the 79 that are
  single tokens in both forms enter every statistic (`peach` is skipped, as before).
- New *combinations* of exposed tokens, frames, and nouns (for example `this` in a manifest frame, or `two` in a
  confirmation frame) are executed for the first time here. That is allowed: no token, frame, or noun is new, and the
  experiment is discovery-only. The results-state ledger records every executed prompt and noun.
- The 007 `selected` program (rank 1; parameter index digest `3d1bf4a476c821ae6ce817e91512f51bb5909608bca09eb2a63717ed30684701`,
  recorded in the committed 007 lock) is used *descriptively*, to compute the anomaly score below. It is loaded from
  its exported tensors and its digest is verified against the 007 lock; it is not refit.

## Anomaly score and descriptive strata

For every token `w` and frame `f`, with `p = Δĉ_prog(w, f)` the 007 program's predicted E-patch shift and `m = Δc(w, f)`
the measured one, the anomaly score is **sign-normalized** so that "suppressed" and "amplified" mean the same thing
whatever the orientation of the predicted effect:

```text
a(w, f) = sign(p) × (m − p)            sign(0) := +1
   predicted −3, measured  0  →  −3   suppression        predicted +3, measured  0  →  −3   suppression
   predicted −3, measured −5  →  +2   amplification      predicted +3, measured +5  →  +2   amplification
```

The token score `a(w)` is the mean over the eighteen frames; it is recorded with the token's in-sample/out-of-sample
status (the sixteen exposed tokens were used to fit the program; the twenty-four were not) and with the mean
`|p|`, so that scores built on near-zero predictions can be recognized.

Descriptive strata, chosen with knowledge of the Experiment 007 residuals and used only to organize reporting (they
gate nothing): **suppressed** `a(w) ≤ −1.5`, **amplified** `a(w) ≥ +1.5`, **ordinary** otherwise. On the frames
already measured, `this`, `another`, `a`, `every` fall in the suppressed stratum and `all`, `both`, `those`, `these`
near or in the amplified one; the strata are recomputed from the eighteen-frame means and reported as they come out.
Every rule below is applied to **all forty tokens**; nothing is selected by stratum.

## Measurements

All interventions are E-patch-type replacements of the `L00.MLP` output at the cue position of an exposed prompt,
run once per (token, frame) with the sites below captured in the same forward. `Δ` always denotes "patched run minus
the clean run of the same prompt".

### M1 — Staged causal trace of the E-patch (all 40 tokens × 18 frames)

Patch `E(w)` into frame `f`'s reference prompt. Capture:

```text
R0  RESID_POST.L0 at p_c        residual after layer 0 at the cue position (Δ = ΔE_T(w) exactly; identity check ≤ 1e-4)
R1  RESID_PRE.L3  at p_c        the residual L03.H04 reads at the cue position (after layers 1–2)
A   ATTN_PATTERN.L3, head 4, query p_t, key p_c     the head's attention weight to the cue position
T   L03.H04 output at p_t
R2  RESID_POST.L3 at p_t        residual after layer 3 at the prediction position
M4, M5   L04.MLP, L05.MLP outputs at p_t
R3  RESID_POST.L5 at p_t        final residual
c   the contrast, per noun, and its exact direct-effect decomposition
```

### M2 — Component patching at the encoding stage (all 40 tokens × 18 frames)

Split `ΔE_T(w) = ΔE_∥ + ΔE_⊥` with `ΔE_∥ = (ΔE_T(w) · d̂_E) d̂_E` along the encoding number axis (defined below).
Run two more E-patches in the reference prompt: `E(ref_T) + ΔE_∥` (axis component only) and `E(ref_T) + ΔE_⊥`
(orthogonal component only). Record `Δc_∥(w, f)`, `Δc_⊥(w, f)`, and the full `Δc(w, f)` from M1.

### M3 — Cross-context patching (all 40 tokens × 18 frames)

The encoding swap is measured inside a fixed context, so that context effects and encoding effects do not mix. With
`prompt_w` the prompt that has `w` as the actual cue token (its embedding and layer-0 attention context) and
`prompt_pl` the template's plural-cue prompt, four patched runs per (token, frame):

```text
x_in  numerator:  c[prompt_w  with E(pl_T)] − c[prompt_w  with E(ref_T)]    the plural encoding's effect inside w's context
x_out numerator:  c[prompt_pl with E(w)]    − c[prompt_pl with E(ref_T)]    w's encoding's effect inside the plural cue's context
context-only:     c[prompt_w  with E(ref_T)] − c_ref                         w's embedding/attention context with the reference encoding
```

(the `prompt_pl with E(ref_T)` run is shared by every token of a frame; for `w = ref_T` the `prompt_w with E(ref_T)`
run is the clean reference run by the zero-by-definition convention). Prompts with `w` as the cue are built from the
frame and the token id exactly as the 006/007 token prompts were; the clean behavioral shift
`Δc_beh(w, f) = c[prompt_w] − c_ref` of every such prompt is recorded as well.

### M4 — Direct-effect delta decomposition (all 40 tokens × 18 frames, from M1's runs)

For the patched and the clean reference run, the exact per-run decomposition gives, per noun, the direct effect
`DE_k` of every term `k` (the embedding, every head result and MLP output at `p_t`, the attention output biases, and
the β term). `ΔDE_k(w, f) = mean over nouns of [DE_k(patched) − DE_k(ref)]`; `Σ_k ΔDE_k = Δc(w, f)` exactly (the scale
terms of the two LayerNorms differ, so each run is decomposed with its own scale and the identity holds per run).

## Definitions

- **Stage axes.** For each captured site `s ∈ {R0, R1, T, R2, M4, M5, R3}`, the number axis `(d̂_s, σ_s)` is the
  site-axis estimate over the eighteen frames' clean original cue pairs (`pl_T` minus `sg_T`; the sign convention makes
  the plural cue positive): `d̂_s` the unit mean difference, `σ_s` its norm. `d̂_E := d̂_{R0}` is the encoding number
  axis (weight-only: `E(pl_T) − E(sg_T)`). The cosine of `d̂_E` with the 007 direction `u₁`, and with Experiment 005's
  frozen E axis, is reported.
- **Signal fraction.** `ŝ_s(w, f) = ⟨Δ_s(w, f), d̂_s⟩ / ⟨Δ_s(pl_T, f), d̂_s⟩`: the token's axis signal at stage `s` as a
  fraction of the template's plural cue's own E-patch signal at that stage in the same frame; `ŝ_c(w, f) = Δc(w, f) /
  Δc(pl_T, f)`. A stage is uninformative in a frame if `|⟨Δ_s(pl_T, f), d̂_s⟩| < 0.25 σ_s` (recorded as `None`). At
  `R0`, both `ŝ_{R0}` (along `d̂_E`) and the 007 fraction `ŝ_{u₁}(w, f) = ⟨ΔE_T(w), u₁⟩ / ⟨ΔE_T(pl_T), u₁⟩` are recorded.
- **Oriented trace.** Every trace is oriented to its own encoding signal: `q_s(w, f) = sign(ŝ_{R0}(w, f)) × ŝ_s(w, f)`
  for `s ∈ {R0, R1, R2, R3, c}`, so `q_{R0} = |ŝ_{R0}| ≥ 0` and a later `q_s < 0` is a sign flip relative to the
  encoding signal. Collapse is defined on `q`, never on the raw signed fractions (a signal going from −2 to −3 has
  strengthened, not collapsed).
- **Attention fraction.** `â(w, f) = A(patched) / A(ref)` for the head's weight on `p_c` (self-attention in cue-final
  frames), and `Δâ = A(patched) − A(ref)`. The attention weight is **descriptive supporting evidence only**: it says
  where the head looked, not what it transmitted. Transport conclusions rest on the head's output fraction `ŝ_T`
  and on the downstream causal deltas (`R2`, `R3`, the direct effects).
- **Component fractions.** `r_∥ = Δc_∥ / Δc(pl_T, f)`, `r_⊥ = Δc_⊥ / Δc(pl_T, f)`, `r_full = ŝ_c`; additivity gap
  `g = r_full − (r_∥ + r_⊥)`.
- **Context fractions.** `x_in(w, f)` = the `x_in` numerator above `/ Δc(pl_T, f)` (does an ordinary plural encoding
  produce its effect inside `w`'s context?), `x_out(w, f)` = the `x_out` numerator `/ Δc(pl_T, f)` (does `w`'s
  encoding produce its (lack of) effect inside an ordinary plural context?); the context-only shift is reported as a
  fraction of `Δc(pl_T, f)` as well.
- **Cancellation index.** `C(w, f) = Σ_k |ΔDE_k| / |Σ_k ΔDE_k|` over the terms `k` at `p_t`, with the denominator
  floored at `0.25 |Δc(pl_T, f)|`; the two largest opposing terms are named.
- **Aggregation.** Token-level values are means over the eighteen frames (and over each template's frames
  separately); stage-collapse locations are aggregated by the mode over frames with the fraction of agreeing frames.

## Frozen classification rules (fixed here, before any Experiment 008 measurement)

Constants: `s_min = 0.3`, `κ = 0.5`, `g_max = 0.25`, `x_min = 0.5`, `C_min = 2.0`, probe-validity floor `0.5`,
consensus fraction `0.75`.

1. **Probe validity (checked first, on the template's plural cue in every frame).** The axis-only patch of the
   plural cue must reproduce its own effect: `r_∥(pl_T, f) ≥ 0.5` in at least 75% of frames, else the component-patching probe
   (M2) is `PROBE_INVALID` and rules 3–4 are withheld (the trace and the other measurements are still reported). The
   identity `Δ_{R0} = ΔE_T(w)` must hold to 1e-4 in every run (else incident), and the direct-effect identity and
   model gap must hold within Experiment 005's tolerances in every run (else incident).
2. **Encoding signal.** A token *carries the mechanism-axis signal* if `|ŝ_{R0}(w)| ≥ s_min`. A token in the suppressed
   stratum that does not carry it is classified `AXIS_ARTIFACT` (supports H0 for that token).
3. **Encoding-stage class (for tokens that carry the signal).**
   - `LINEAR_CANCELLATION` if `r_⊥(w) ≤ −0.5 · r_∥(w)` and `|g(w)| ≤ g_max`;
   - `NONLINEAR_GATING` if `|g(w)| > g_max` (the whole is not the sum of the parts; supports H1);
   - `AXIS_NOT_SUFFICIENT` if `r_∥(w) < s_min` while `|ŝ_{R0}(w)| ≥ s_min` (the axis component alone does not drive
     the response for this token although it does for the plural cue);
   - `ADDITIVE_ORDINARY` otherwise.
4. **Collapse stage (from M1, full `ΔE_T(w)`, on the oriented trace).** Along the running-residual stages
   `R0 → R1 → R2 → R3 → c`, the collapse stage `σ*(w, f)` is the first stage `s` with `q_s ≤ κ · q_{s−1}`, provided
   `q_{s−1} ≥ s_min`; because `q_{s−1} > 0`, this single condition covers both a drop to at most half of the previous
   oriented signal and a sign flip relative to the encoding signal (`q_s ≤ 0`). `NO_COLLAPSE` if none. Interpretation
   of the location: `R1` — transformation at the cue position in layers 1–2 (H1 at the cue position); `R2` —
   transport into the prediction position (H3): the oriented head-output fraction `q_T` decides whether `L03.H04`
   itself dropped the signal (`q_T ≤ κ · q_{R1}`) or whether other layer-3 components at `p_t` cancelled what the head
   delivered (`q_T > κ · q_{R1}` while `q_{R2}` collapsed); the attention fraction `â` is reported beside `q_T` as
   context and never decides; `R3` — late components (H4; the cancellation index and the named opposing terms say
   which); `c` — the readout.
5. **Context interaction (M3).** `CONTEXT_GATED` if `x_in(w) < x_min` (an ordinary plural encoding is suppressed in
   `w`'s context; supports H2); `CONTEXT_NEUTRAL` otherwise. `x_out` is reported.
6. **Late cancellation (M4).** `LATE_CANCELLATION` if `C(w) ≥ C_min · C(pl_T)` and `C(w) ≥ C_min`; the named opposing
   terms are reported.

### Experiment-level summary (mechanical)

Applied to the suppressed stratum as it comes out of the eighteen-frame means:

- `AXIS_ARTIFACT` — every suppressed token is `AXIS_ARTIFACT` under rule 2.
- `LOCALIZED_<stage>_<class>` — at least 75% of the suppressed tokens that carry the signal share one modal collapse
  stage and one encoding-stage class.
- `CONTEXT_LOCALIZED` — at least 75% of the suppressed tokens are `CONTEXT_GATED` while the E-patch traces show no
  common collapse stage.
- `MIXED` — otherwise; the per-token table is the result.
- `PROBE_INVALID` — rule 1 failed for M2; the trace summary (`LOCALIZED_<stage>` without a class, or `MIXED`) is
  still given.
- Incidents (replication mismatch, identity failures, software defects) stop the phase and are never a summary label.

The amplified and ordinary strata receive the same per-token classification and are summarized descriptively; the
determiner set (`this`, `that`, `these`, `those`, `a`, `the`, `another`, `every`) is reported as a group.

## Replication checks (incidents if violated)

- The 192 E-patch mean shifts of the sixteen exposed tokens on the twelve old frames must equal Experiment 006's
  recorded values within 1e-6 (the committed extract used by Experiment 007).
- The 144 E-patch mean shifts of the twenty-four confirmation tokens on the six confirmation frames must equal
  Experiment 007's recorded values within 1e-6 (a derived extract of the 007 results state is committed with its source
  digests before `explore`).
- The 007 program loaded for the anomaly score must match the 007 lock's parameter digest.

## Interpretation limits

- Discovery-only: every classification is exploratory, the thresholds are fixed in advance but were chosen without a
  power analysis, and the suppressed stratum is small (four tokens on the frames measured so far). The result is a
  localization *hypothesis* with named components, not a confirmed mechanism; Experiment 009 must freeze new cue
  tokens (in particular new singular-selecting determiners and near-synonyms of the suppressed tokens) and new frames
  before any prospective test.
- Signal fractions are projections onto one axis per stage; a signal carried in a different direction at a later stage
  appears as a collapse. The direct-effect decomposition (exact) partly guards against this and is reported next to
  the fractions. The attention weight of `L03.H04` is descriptive only; it does not establish or refute transport.
- Component patching along `d̂_E` tests one axis; if the probe is invalid on the plural cue itself, no conclusion about
  H1 versus linear cancellation is drawn.
- Nothing generalizes beyond the pinned checkpoint, the three templates, single-token regular nouns, and the tokens
  tested.

## Minimal implementation boundary

A module `cue_suppression.py` reusing `plural_mechanism.py` (prompts, clean-run cache, exact replacement with
captures, contrasts, site axes, direct effects) and `cue_decompilation.py` / `supervised_subspace.py` (exposed pool,
E-patch responses, program loading); a derived extract of Experiment 007's confirmation-token E-patch means; a runner
with phases `validate`, `explore` (once; results state with ledger), `report`. No lock, no confirm phase, no new
prompts beyond exposed tokens in exposed frames, no component search, no other model. Tests: axis and fraction
arithmetic on planted vectors, the component split (`ΔE_∥ + ΔE_⊥ = ΔE`), every classification-rule branch on synthetic
tables, the summary rule, replication incidents, and the runner's phase isolation on the fake model.

## Approval and stopping condition

Design first; no implementation until approved. `explore` runs once after the implementation review; the report and
the per-token table are the deliverable; the localization statement is then handed to the Experiment 009 design.

## Revision history

- **Revision 1** (commit `5b13138`): initial draft. Reviewed: direction approved; two corrections requested because
  they affect interpretation — the collapse criterion `ŝ_s ≤ 0.5 · ŝ_{s−1}` misbehaves for negative signals (−2 → −3
  is a strengthening, yet −3 ≤ −1), and the anomaly score `prediction − measured` would call a suppressed positive
  prediction "amplified". Also: treat the `L03.H04` attention weight as descriptive support, not as evidence of
  transport by itself.
- **Revision 2**: (1) traces are oriented to their encoding signal, `q_s = sign(ŝ_{R0}) × ŝ_s`, and collapse is
  `q_s ≤ κ · q_{s−1}` with `q_{s−1} ≥ s_min` (covers both a drop and a sign flip relative to `R0`); the `R2`
  interpretation now decides transport by the oriented head-output fraction `q_T` and the downstream deltas.
  (2) The anomaly score is sign-normalized, `a = sign(p) × (m − p)`, so suppression is negative and amplification
  positive for either orientation; the mean `|p|` is recorded beside it. (3) The attention fraction is declared
  descriptive supporting evidence throughout. No threshold, measurement, or label changed.
- **Revision 3** (pre-measurement, during implementation planning): M3's baseline is the same prompt patched with the
  reference encoding rather than its clean run, so that `x_in` and `x_out` measure the encoding swap inside a fixed
  context (with the clean-run baseline, `x_in` of the plural cue itself would be identically zero and `x_in` of other
  tokens would mix their context effect with the encoding effect); the context-only shift is recorded separately.
  Probe validity is stated on the plural cue (the singular cue's axis-only patch is the identity). Stage fractions
  and `ŝ_c` use the same uninformative-denominator convention (`|Δc(pl_T, f)| < 0.25` nats → `None`). No threshold,
  label, or other measurement changed.
