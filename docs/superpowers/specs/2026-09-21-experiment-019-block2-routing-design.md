# Experiment 019: Frame-Conditioned Routing of Channel D — Is the Frame-Specific Membership of Block 2's Recovery Subset Predicted from the Frame's Reference State Before Any Fresh Cue Is Run, and by the Frame's Operating Point Alone? — A Prospective Routing Test with a Nested Ladder of Selectors

**Date:** 2026-09-21

**Status:** Revision 1 — for review. No Experiment 019 directory, confirmation set, lock, or model run exists.
Experiments 005–018 are closed and are not amended by this document; Experiment 018's closure stands exactly as
recorded (`CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED |
PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES`, never rerun). The one Experiment 018 frame where the globally locked
subset fell below the template base (`quantifier-018-4`, `The atlas maps {cue}`) *motivates* this experiment and is
*not* its object: every statistic below is defined over a predeclared frame population, and that frame is one of ninety
exposed frames with no special role.

**Kind:** Prospective, zero-parameter, mechanism (routing). Experiment 018 showed that channel D — block 2's MLP at the
frame's own operating point, the least compact object of the decoded cue-to-transport program — is carried for unseen
cues by a distributed subset of 256 block-2 neurons chosen once on the exposed pool (`κ_{c_L}` 0.86), that a single
neuron is rejected (0.22), and, descriptively, that *which* members of the population carry the computation depends
on the frame: each frame's own top-256 shared only 118–163 neurons with the locked `S_256`, the frame's own subset
recovered more than the locked one on both fresh sets (0.93 / 0.94 against 0.86 / 0.84), neuron 1102 overtook 1987 in
the four new quantifier frames, and in one new frame the locked subset fell below the template base while the frame's
own top-256 did not. This experiment asks whether that redistribution is *predictable*: whether a frame-specific
subset of matched size, selected from the frame's reference state **before any fresh cue is run in it and without any
observed block-2 neuron effect of that frame**, recovers substantially more of channel D than the best global subset of
the same size; whether the selection can be written as a compact operating-point rule; whether a template-level
selection would do as well; and how far the prospective selection sits from a same-rule ceiling that is allowed to see
the fresh cues' measured responses (the oracle, descriptive only). Nothing is fitted: every selector is a frozen rule
applied to locked reference states, the weights and the exposed cues' weight-only encoding changes; every subset is
an explicit index list committed before the fresh prompts it is scored on.

## Purpose and question

Four accounts of Experiment 018's descriptive observation are to be discriminated, and each is given a falsifier
below:

1. **Fixed population with frame-dependent amplitudes.** The same neurons carry channel D everywhere; the frame only
   rescales their contributions. Then no frame-specific choice of `k` neurons — not even one that knows the fresh
   cues' measured responses — recovers materially more than the best global `k` (no *routing headroom*), and every
   prospective selector's gain over the global subset is nil.
2. **Predictable frame-conditioned routing.** Which neurons carry the computation varies by frame, and the variation
   is determined by the frame's reference state: a selector computed from that state alone — never from a fresh cue's
   forward pass in the frame — picks a `k`-subset that recovers substantially more than the global one, for unseen
   cues and unseen frames. Two levels of this account are separated: the *full evaluation* (Level 0-E: block 2's
   predicted response to the exposed calibration cues at the frame's state, the object Experiment 018 recorded
   descriptively as each frame's "own ranking") and the *operating-point rule* (Level 0-G: the frame's reference
   pre-activations alone, with a locked per-template drive profile — a rule that evaluates no cue in the target
   frame and asks only where the frame has moved each neuron on its GELU relative to the template base).
3. **Template-family subcircuits.** The redistribution is between templates, not between frames: a per-template
   ranking recovers as much as any frame-specific one.
4. **Redistribution not predicted by the tested rules.** Headroom exists (the oracle beats the global subset) but the
   prospective selectors do not close it: the routing depends on the fresh cues' own interaction with the frame, or on
   something the reference state does not carry through the tested rules.

**Design checks on exposed data (outside any results state; no forward pass on any prompt).** Inputs: the Experiment
018 lock (the 78 locked reference states, its scores and subsets, its per-frame top-256 lists), its closed results
state (the twelve 018-fresh frames' stage-1 reference states and per-frame rankings; the measured `c_L` of its 9636
exposed and 2160 confirmed pairs and every rung's prediction), the Experiment 011/012/017 locks and the weights.
Experiment 018's masked chain was used verbatim through its `upstream_parts` (each pair's predicted arriving change
and per-neuron own/template hidden deltas at the frame's state), and the decoded read of any subset was evaluated by
the exact identity `ĉ_L(S) = ĉ_L(S_0) + Σ_{j ∈ S} u_j`, `u_j = e_j · r(W_out^{(2)}[j, :]) / D_T` (the read is linear in
the masked hidden deltas; the identity reproduced the recorded `ĉ_L(S_256)` of all 11 796 pairs to `2e-8`). Every
selector below was evaluated on the three sets Experiment 018 left — its exposed pool (9636 pairs, cue-in-sample),
its Y1 (24 fresh cues × 78 exposed frames, 1872 pairs) and its Y2 (24 fresh cues × 12 new frames, 288 pairs) — at
`k ∈ {16, 64, 256}`, on the decoded read `c_L` (the direct block-2 transport-read quantity, Experiment 018's decision
variable). The oracle of the checks is the *predicted* per-neuron effect of the scored cues themselves (the
measured one, defined below, needs the captured block-2 input of the fresh runs, which the 018 record does not hold;
the reference rung's `c_L R²` of 0.999 makes the two nearly identical in aggregate but not necessarily neuron by
neuron, and the membership floor below allows for that). In the checks the full evaluation of an exposed frame used
the frame's licensed records (Experiment 018's per-frame ranking) and that of a new frame the template's 230 exposed
tokens (its stage-1 ranking); the frozen rule below uses the template's 254 exposed tokens at every frame.

- **The routing headroom exists at every size and is largest in new frames.** The same-rule oracle (`O_k`: the
  frame's top-`k` by the fresh cues' own effects) exceeds the matched global subset (`S'_k`, below) by `κ_{c_L}`
  `+0.08 / +0.10 / +0.08` (Y1, `k` = 16 / 64 / 256) and `+0.17 / +0.14 / +0.10` (Y2). Account 1 leaves no such room.
- **The full evaluation closes essentially all of it.** `E_k(f)` — each frame ranked on the exposed cues' predicted
  effects at its own reference state, Experiment 018's per-frame ranking taken per changed position — reaches
  `κ_{c_L}` 0.729 / **0.816** / 0.938 on Y1 against `S'_k` 0.660 / 0.724 / 0.864 (gains `+0.07 / +0.09 / +0.07`; the
  oracle 0.740 / 0.828 / 0.947), and 0.767 / **0.859** / 0.944 on Y2 against 0.593 / 0.706 / 0.863 (gains `+0.17 /
  +0.15 / +0.08`; the oracle 0.767 / 0.848 / 0.959 — at `k = 64` the prospective selector matches the fresh-cue-informed
  ceiling). At `k = 64` it improves on the global subset in 52 of 78 exposed frames and 8 of 12 new frames, its pooled
  gain survives the removal of any one frame (jackknife minimum 0.085 on Y1, 0.128 on Y2), and it is positive in
  both template families and in every template (Y1: cardinal `+0.04`, quantifier `+0.08`, coordinated `+0.17`; Y2:
  `+0.09 / +0.13 / +0.27`). Its top-64 shares on average 0.87 (Y1) and 0.81 (Y2) of the oracle's top-64, against 0.42 /
  0.40 for the global subset; its scores rank the 2048 neurons against the oracle's with Spearman 0.99 / 0.97.
- **The operating-point rule closes most of what the full evaluation closes — with the right drive profile.** Five
  compact rules were tried, all with the frame's reference pre-activations as their only frame-specific input. The
  first-order rule (the change of GELU slope between the frame's state and the base, times a mean drive) is *worse*
  than the global subset (`k = 64`: 0.670 on Y1, 0.680 on Y2 against 0.724 / 0.706): the neurons that matter cross the
  GELU knee, where a slope is meaningless. A single signed mean drive fails likewise (0.513 / 0.688). Evaluating the
  exact GELU at the frame's state over a locked *distribution* of drives works: three quantiles of the drive over the
  calibration pool 0.775 / 0.850, the full empirical profile 0.764 / 0.870, the profile rescaled by the frame's
  LayerNorm scale 0.770 / 0.858, and — best and equally compact — three quantiles taken *per template* **0.795 /
  0.867** (`G_k(f)` below; full per-template profile 0.791 / 0.861). Its share of the full evaluation's gain over the
  global subset is 0.75 (Y1), 1.01 (Y2), 0.81 on the two sets pooled at `k = 64`; 0.89 / 1.00 / 0.91 at `k = 256`; only
  0.20 / 0.85 / 0.41 at `k = 16`, where the smallest subsets are the most sensitive to the exact drives. Its one weak
  template is cardinal on the exposed frames (`−0.04` against the global subset at `k = 64` on Y1; `+0.05` on the new
  cardinal frames). Its top-64 shares 0.72 / 0.71 of the oracle's; Spearman 0.92 / 0.93.
- **The template average is not enough.** The per-template ranking `T_k` gains `+0.03 / +0.04` over the global subset at
  `k = 64` (Y1 / Y2) and the full evaluation exceeds it by `+0.064 / +0.112` (pooled `+0.071`); at `k = 16` the template
  ranking is *below* the global one (`−0.03`). Account 3 is expected to fail, but the margin is set so that it can win.
- **The matched global comparator.** Experiment 018's locked `S_k` was ranked over both changed positions; the decoded
  read is taken at `p_c`, and a global ranking by the same rule over the `p_c` records of the enlarged pool shares 59–61
  of 64 and 242–251 of 256 neurons with it but recovers more on the new frames (`k = 64`: 0.706 against 0.678). The
  decision comparator is therefore the **matched-rule global subset `S'_k`** (same rule, same pool, same position as
  every frame-specific selector), so that a frame-specific gain cannot be an artifact of a stale or position-mixed
  global list; the inherited lock's `S_k` is reported beside it as a named control.
- **Per-frame behaviour is a gamble, not a guarantee.** The frame-specific selection loses to the global subset by
  more than 0.05 `R²` in 7 of 78 exposed frames at `k = 64` (0 of 12 new frames); per-frame closure fractions range
  from below zero to 0.98 for every selector. No per-frame no-harm guard is used (Experiment 018's lesson); the claim
  is pooled, with a frame-count guard against a gain carried by a minority of frames and a split guard against a gain
  carried by one template family.
- **The frame's top neuron.** By the full evaluation, 1987 is first in 48 of the 78 exposed frames, 1726 in 11, 1102 in
  13 (10 of them quantifier frames), and in the twelve new frames 1987 in 7, 1726 in 3, 1102 and 1924 in one each —
  recorded here as the descriptive fact it is.

> For cue words and frames never used to choose it, does a frame-specific subset of 64 block-2 neurons — selected from
> the frame's reference state by a frozen rule before any fresh cue prompt is run in that frame — recover
> substantially more of channel D's contribution to the transport read than the best global subset of 64 chosen by the
> same rule (Y1 on unseen cues in the exposed frames; Y2 on unseen cues in eighteen unseen frames)? Is the compact
> operating-point rule sufficient for that gain relative to the full evaluation (Y3)? Is a template-level selection
> insufficient (Y4)? Does the predicted subset name the neurons that the fresh cues' measured responses rank highest
> (Y5)? And, descriptively, how much room did a fresh-cue-informed oracle of the same rule have?

## The quantities under test (frozen definitions)

Notation as in Experiments 017 and 018: layers `ℓ ∈ {1, 2, 3}`; `x_ℓ(k)` the frame's reference residual before block
`ℓ` at position `k ≤ p_t`; `p_c`, `p_t` the cue and transport positions (`p_t = p_c` in the cardinal and quantifier
templates, `p_t = p_c + 1` in the coordinated-adjective template); `P = {p_c, p_t}`; `x̄₂^T(p_c)` the Experiment 012
template-mean base and `x̄₂^T(p_t)` the coordinated template's block-2 base locked by Experiment 018 (used verbatim);
`ΔE` weight-only; `r(·)`, `D_T`, `c_L = r(Δx₃(p_c) − ΔE) / D_T`, the head objects `F`, `Π`, `ΔT`, the row, exactly as
frozen in Experiments 017–018. Block 2's MLP: `ln2₂`, `W_in^{(2)}` (512 × 2048), `b_in^{(2)}`, `W_out^{(2)}`, exact GELU.

**Channel D, its masked form and the per-neuron quantities (Experiment 018, verbatim, now per position).** For a
pair `(w, f)` and a changed position `p ∈ P(f)`, with the frame's reference residual `x₂(p)` and the *predicted*
arriving change `Δ̂x₂(p)` (Experiment 016's Level 0-F at `p_c`; Experiment 017's propagation step at `p_t`):

```text
pre_j(f, p)    = ⟨ ln2₂(x₂(p)),    W_in^(2)[:, j] ⟩ + b_in^(2)[j]           the frame's reference pre-activation (the operating point; reference run only)
pre_j^T(p)     = ⟨ ln2₂(x̄₂^T(p)),  W_in^(2)[:, j] ⟩ + b_in^(2)[j]           the template base's pre-activation (locked)
Δpre_j(w,f,p)  = ⟨ ln2₂(x₂(p) + Δ̂x₂(p)) − ln2₂(x₂(p)),      W_in^(2)[:, j] ⟩   the predicted drive at the frame's state
Δpre_j^T(w,f,p)= ⟨ ln2₂(x̄₂^T(p) + Δ̂x₂(p)) − ln2₂(x̄₂^T(p)),  W_in^(2)[:, j] ⟩   the predicted drive at the template base
own_j  = GELU(pre_j + Δpre_j) − GELU(pre_j);   tmpl_j = GELU(pre_j^T + Δpre_j^T) − GELU(pre_j^T)
e_j(w, f, p)   = own_j − tmpl_j                                              the operating-point effect (Experiment 018)
u_j(w, f, p)   = e_j(w, f, p) · r(W_out^(2)[j, :]) / D_T(f)                  its read-unit contribution (signed)
D_S at p:      Δ̂₂^S(p) = Σ_{j ∈ S(p)} own_j · W_out^(2)[j, :] + Σ_{j ∉ S(p)} tmpl_j · W_out^(2)[j, :]
Identity:      ĉ_L(S) = ĉ_L(S_0) + Σ_{j ∈ S(p_c)} u_j(w, f, p_c)             (the read is linear in the masked deltas; checked in process to 1e-9)
```

The novelty of this experiment is that the subset `S` may now depend on the frame *and the position*: the masked
chain takes one mask per changed position, `S(p_c)` and `S(p_t)`; for cue-final frames the two coincide. Everything
else in the chain — Experiment 016's Level 0-F, the propagation step, the layer-3 bases, the head's reduced
query/key/value changes, `F̂`, `Π̂`, `ΔT̂`, the row — is Experiment 017's and 018's, verbatim.

**Scores and selectors (frozen rules; every subset is the top-`k` of a score vector over the 2048 neurons, ties
broken by the lower index; `k ∈ {16, 64, 256}`; the decision size is `k = 64`).** All scores are in read units,
`|e_j| · |r(W_out^{(2)}[j, :])| / |D_T|`, as Experiment 018's ranking was. A *record* is a licensed pair at one of its
changed positions; a *position type* is `p_c` or `p_t` (only coordinated frames have a `p_t ≠ p_c`, so every
`p_t`-type object is a coordinated-template object by construction). The **calibration pool** is the set of licensed
exposed pairs (below): `S'`, `T` and the drive-quantile tables of `G` are means and quantiles over its records, while
`E` evaluates every exposed cue of the template at the frame's state (a full product, since it is predicted, not
measured); every score is computed from the locked reference states, the weights and the exposed cues' `ΔE` alone —
no measured quantity of any pair, and no fresh cue.

```text
S'_k(p)     matched global:    top-k of  mean over all calibration records at position type p of |e_j| |r_j| / |D_T|        (one list per position type; at p_t it equals T_k for the coordinated template)
T_k(τ, p)   template family:   top-k of  the same mean over the calibration records of template τ at position type p        (one list per template and position type)
E_k(f, p)   full evaluation:   top-k of  mean over every exposed cue licensed in template τ(f) of |e_j(w, f, p)| |r_j| / |D_T|  (Experiment 018's frame ranking, per position: the exposed cues' predicted effects at the frame's own state, for every
            frame the same 254 cues — the template's reference cue excluded — whether or not the pair is a licensed record; predicted quantities only)
G_k(f, p)   operating-point:   top-k of  (1/3) Σ_{q ∈ {0.1, 0.5, 0.9}} | Δ_GELU(pre_j(f, p); d_j^q(τ, p)) − Δ_GELU(pre_j^T(p); d_j^q(τ, p)) | · |r_j| / |D_T(τ)|
            with Δ_GELU(a; d) = GELU(a + d) − GELU(a) and d_j^q(τ, p) the q-quantile, over the calibration records of template τ at position type p, of the base drive Δpre_j^T
            (the frame enters only through its 2048 reference pre-activations pre_j(f, p); the drive profile is a locked table of 3 × 2048 numbers per template and position type)
O_k(f, p)   oracle (ceiling):  top-k of  mean over the SCORED FRESH CUES in f of |e_j^meas(w, f, p)| |r_j| / |D_T|        (measured effects, below; computed at stage 2 only; descriptive, enters no prediction)
S_k(018)    inherited:         the Experiment 018 lock's S_k lists, digest-checked                                          (named control; both positions)
R1–R3(k)    random:            three k-subsets from random.Random(20260924): for k in (16, 64, 256), sample(range(2048), k) three times, one generator, in that order; overlaps recorded, never redrawn
S_0, S_2048  the template-base rung and the reference rung (Experiment 017's Level 0)                                      (κ's endpoints)
S'_1, E_1(f) the global top neuron and the frame's top neuron                                                              (descriptive, "one neuron" accounts)
```

`E` is the strongest prospective selector the reference state affords: it evaluates block 2's decoded response to
every calibration cue in the target frame, but to no fresh cue. `G` is the compact mechanism claim: the frame routes by
its operating point — which neurons it has moved across the GELU relative to the template base — with the cue
population's drive on each neuron represented by three locked numbers per template. `T` and `S'` are the two
frame-blind comparators; `E ⊃ G ⊃ T ⊃ S'` in the frame-specific information they use, and the ladder is scored with
one rule so that the comparisons isolate the information, not the rule.

**The measured per-neuron effect and the oracle.** In Pythia's parallel block, block 2's MLP reads `ln2₂` of the
residual before block 2, which the patched run captures at `p_c` (`RESID_PRE.L2`, since Experiment 012) and — added
here to the patched sites — at `p_t`. With the *measured* arriving change `Δx₂^meas(p) = x₂^patched(p) − x₂(p)`:

```text
own_j^meas  = GELU(pre_j(f, p) + ⟨ln2₂(x₂(p) + Δx₂^meas(p)) − ln2₂(x₂(p)), W_in[:, j]⟩) − GELU(pre_j(f, p))
tmpl_j^meas = GELU(pre_j^T(p) + ⟨ln2₂(x̄₂^T(p) + Δx₂^meas(p)) − ln2₂(x̄₂^T(p)), W_in[:, j]⟩) − GELU(pre_j^T(p))
e_j^meas    = own_j^meas − tmpl_j^meas
```

— an exact function of the captured residual and the weights, i.e. the neurons' observed responses in that fresh run.
The oracle ranks each frame by the mean `|e_j^meas| |r_j| / |D_T|` over the frame's scored fresh cues; it is
in-sample on the pairs it is scored on and is therefore a *ceiling of the same ranking rule*, reported and never
labelled. It is not the `R²`-optimal `k`-subset (contributions of opposite sign can cancel), and in the design checks
the full evaluation occasionally exceeds it; that is why the routing floors are stated against the global subset, not
as a fraction of the oracle's gain. Descriptively, the neuron-level fidelity of the chain is recorded per pair as the
`R²` of `e_j` against `e_j^meas` over the 2048 neurons (an accounting, no floor).

**The closure fraction and the routing gain (the statistics of the claims).** For an object `X`, a set of scored
pairs and a rung `S` (a fixed list, or a frame-and-position-specific selector applied pair by pair):

```text
κ_X(S)  = ( R²_X(S) − R²_X(S_0) ) / ( R²_X(S_2048) − R²_X(S_0) )      pooled R² over the scored pairs; unclipped; undefined below a 0.05 gap (Experiment 018, verbatim)
Δκ_k(X) = κ_{c_L}(X_k) − κ_{c_L}(S'_k)                                the routing gain of selector X at size k over the matched global subset
ρ_k     = Δκ_k(G) / Δκ_k(E)                                            the share of the full evaluation's gain that the operating-point rule carries
```

`κ` is Experiment 018's statistic with its evaluability rule; `Δκ` is a difference of two closure fractions on the
same pairs and has the same unit (a share of channel D's contribution to the read). The decision object is `c_L`; `F`,
`Π`, `ΔT` and the row entries are reported for every selector with their `κ` descriptively, with Experiment 018's
predeclared orderings carried (`κ_{c_L} ≥ κ_Π`; the row diffuse).

**Level 1 — the exact chain (identity, incident-guarded, counts toward no floor).** Experiment 017's I1–I7 and
Experiment 018's I8 verbatim on every exposed and fresh pair; plus **I9**: the `E` rule evaluated with Experiment
018's own inputs — its 231 tokens, licensed as it licensed them (the frame's licensed records for its 78 exposed
frames; the template's 230 tokens for its twelve stage-1 frames) — reproduces that experiment's recorded per-frame
top-256 at `p_c` on every cue-final frame (its `frame_subsets` and the stage-1 lists carried in the inherited extract;
on cue-final frames its pooling over positions is trivial) — a replication of the recorded selector with its own
inputs, not evidence for the 019 selector, whose cue set is the enlarged one; and **I10**: the read identity `ĉ_L(S) =
ĉ_L(S_0) + Σ u_j` holds for every rung within `1e-9`.

**Level 0-S (the masked chain with per-position masks; the hypotheses and controls).** Experiment 018's Level 0-S with
the mask table extended to frame-and-position-specific lists. Every rung receives `ΔE`, the weights, the locked axes,
read and bases, the locked score tables and lists, and the frame's reference run only. The oracle's masks are built at
stage 2 from the measured effects and are applied to the *same* predicted upstream parts (so that the oracle differs
from the prospective selectors only in which neurons it names, never in what it predicts for them).

**Locked boundary (formal).** Experiment 018's boundary verbatim, and in addition: `S'_k`, `T_k`, the drive-quantile
tables of `G`, and `E_k(f, p)` and `G_k(f, p)` for every exposed frame are computed at `explore` from the weights, the
locked reference states of the ninety exposed frames and the exposed cues' `ΔE` only; a test poisons every measured
quantity of the exposed pairs and checks that no score, table or list changes; a second test replaces the
confirmation set's tokens and frames by poisoned ones and checks the same. For a fresh frame, `E_k(f, p)` and
`G_k(f, p)` are computed at stage 1 from its reference run (the exposed cues' predicted effects at its state; its
pre-activations) and digested with the prediction table before any fresh cue prompt runs in it. The oracle is computed
only at stage 2, only from stage-2 captures, and enters no prediction row: a test checks that every prediction column
is unchanged when the stage-2 captures are poisoned. Order:

```text
exposed pool (weights, locked states, exposed ΔE)  →  records' e_j, Δpre_j^T  →  S'_k, T_k, quantile tables, E_k(f, p), G_k(f, p) for the 90 exposed frames  →  lock
fresh frame  →  reference run (stage 1)  →  E_k(f, p), G_k(f, p)  →  prediction table for every rung  →  digest  →  fresh cue prompts (stage 2)  →  measured e_j  →  O_k(f, p)  →  scoring
fresh cue word  →  ΔE (weight-only)  →  Experiment 017's chain with the per-position masks  →  ĉ_L, F̂, Π̂, ΔT̂, row for every rung  →  table  →  digest  →  fresh cue prompt
```

**Accounting (descriptive, per pair and per frame):** I1–I10; every rung's `ĉ_L`, `F̂`, `Π̂`, `ΔT̂`, row; the
neuron-level fidelity; per frame and position the four prospective lists, the oracle list, their pairwise overlaps
(`|X_k ∩ O_k| / k` and the Spearman of the score vectors), the frame's top neuron under `E` and `G`, and the `R²` of
`c_L` under every rung with the frame's gap; the per-template and per-family splits; the `k = 16` and `k = 256` ladders
with the same statistics; the inherited `S_k` and the random controls; `κ_Π`, `κ_F`, `κ_ΔT`, `κ_row` per selector; the
reference rung's Experiment 017 statistics.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked bases, the Experiment 015 program, Experiment
016's Level 0-F, Experiment 017's chain, the Experiment 017 lock's layer-3 bases, the Experiment 018 lock's `p_t` base,
its 78 reference states, its scores and subsets and its per-frame lists, the digests of the twelve 018-fresh frames'
stage-1 states (carried in the inherited extract), the results-state, ledger, lock, prediction-artifact, two-stage
confirmation and incident conventions — all exactly as in Experiments 012–018. Seeds: runtime `20260916`, control
`20260924`. The patched capture adds `RESID_PRE.L2` at `p_t` (coordinated frames) so that the measured per-neuron
effect is defined at both changed positions.

## Exposed pool (calibration record only; nothing is fitted)

The 255 exposed cue tokens (Experiment 018's 231 and its 24 confirmed) and the 90 exposed frames (its 78 and its 12),
with the 80 nouns. The licensed pairs are Experiment 018's 9636 and its 2160 confirmed pairs (its 24 tokens in the 78
exposed and the 12 fresh frames), 11 796 in all, with the triangular licensing inherited unchanged; the calibration
pool is exactly this set of pairs at their changed positions, admitted or removed by no measured value. `explore`:

- captures the 90 reference states (residuals before blocks 1–3 at every position `≤ p_t`, the layer-1–2 rows at
  `p_c`, `L03.H04`'s row at `p_t`), checks the 78 against the Experiment 018 lock within `1e-9` and the 12 against its
  stage-1 digests, locks the states;
- computes every record's `e_j` and `Δpre_j^T` from the locked states and the exposed cues' `ΔE` (no measured
  quantity), then `S'_k`, `T_k`, the drive-quantile tables and the random controls; evaluates the template's 254
  exposed cues at every exposed frame's state for `E_k(f, p)` (about 23 000 predicted pairs, no forward pass) and the
  frame's pre-activations for `G_k(f, p)`; checks I9 against the Experiment 018 lock and the inherited extract; locks
  them all as explicit index lists and score tables;
- re-measures the E-patch of every licensed pair with the layer-3 residuals, the head's row and the block-2 input at
  both changed positions captured; replicates Experiment 018's per-pair `F`, `Π`, `ΔT`, `c_L`, row and its `S_0`,
  `S_256`, `S_2048` predictions within `1e-6` from a committed extract (`inherited/experiment-018-pair-extract.json`,
  11 796 pairs, carrying also the twelve stage-1 state digests and the twelve stage-1 per-frame lists); checks
  I1–I10 on every pair; computes every rung's predictions and the exposed oracle (cue-in-sample) from the locked states;
- records the exposed statistics: every rung's `R²` and `κ` for `c_L`, `F`, `Π`, `ΔT` (pairs and token means) and the
  row entries, pooled, per template, per family, per frame; `Δκ_k` and `ρ_k` at every `k`; the overlaps and the
  frame-top neurons; the neuron-level fidelity.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 019 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 255 exposed tokens and
  every exposed noun form; the first eligible entries of the frozen candidate lists below with the same quotas
  (5 / 4 / 5 / 4 / 6); classes for coverage and reporting only, **no class carries an expectation**. The lists are
  Experiment 017's lists (whose first eligible entries Experiments 017 and 018 consumed) *extended* by the entries
  after the `+`, in this order; tokenizer-checked on 2026-09-21 against the 255 exposed tokens, the noun forms and the
  exposed frames' words:
  - `determiner-like` (quota 5): `separate`, `remaining` + `similar`, `different`, `prior`, `recent`, `current`,
    `original`, `typical`, `ordinary`, `identical`, `alternate`, `random` — expected `separate`, `remaining`, `similar`,
    `different`, `prior`.
  - `ordinal-or-numeral` (quota 4): `double`, `triple`, `couple`, `pair` (the last eligible entries of the inherited
    list; every remaining single-token numeral, ordinal and multiplier word is already exposed — the class is exhausted
    after this experiment and will need a new list or a reduced quota next time) — expected all four.
  - `quantity` (quota 5): `generous`, `overall` + `negligible`, `maximum`, `minimum`, `average`, `excessive`,
    `exhaustive`, `comprehensive`, `thorough`, `sweeping`, `bulk`, `spare`, `dense`, `lengthy`, `lots`, `loads`, `tons`,
    `scores` — expected `generous`, `overall`, `negligible`, `maximum`, `minimum`.
  - `possessive-or-pronoun` (quota 4): `she`, `he`, `they`, `we`, `anything`, `something`, `everything`, `nothing` +
    `who`, `thee`, `thou`, `yourself`, `themselves`, `ones`, `others` — expected `she`, `he`, `they`, `we`.
  - `adjective` (quota 6): `rotten`, `faded`, `stolen`, `gentle`, `giant`, `silver`, `purple`, `yellow` + `hidden`,
    `hard`, `dirty`, `rare`, `square`, `wild`, `brave`, `calm`, `eager`, `fierce`, `humble`, `proud`, `shy`, `lazy`, `busy`,
    `sturdy`, `fragile`, `polished`, `muddy`, `straight`, `hollow`, `solid`, `tender`, `tough`, `stale`, `sticky`, `fuzzy` —
    expected `rotten`, `faded`, `stolen`, `gentle`, `giant`, `silver`.
- **Fresh frames (18, six per template), literal, tokenizer-checked on 2026-09-21 to build and to differ from every
  exposed frame text:** cardinal `The garage stores {cue}`, `The nursery raises {cue}`, `The chef prepares {cue}`,
  `The vault protects {cue}`, `The lodge hosts {cue}`, `The mill grinds {cue}`; quantifier `The handbook explains
  {cue}`, `The catalog features {cue}`, `The bulletin reports {cue}`, `The digest summarizes {cue}`, `The charter
  defines {cue}`, `The gazette prints {cue}`; coordinated-adjective `Bruno and Greta baked {cue} warm`, `Iris and Milo
  scrubbed {cue} clean`, `Otto and Vera arranged {cue} flat`, `Fiona and Jack dried {cue} stiff`, `Hana and Piet sliced
  {cue} thin`, `Kurt and Dana brushed {cue} bright` (`p_c` 3–7; `p_t = p_c` or `p_c + 1`). The original cue tokens are
  used for their own cue prompts. Eighteen rather than twelve because the frame is the unit of the routing claim:
  the frame-count guard below needs a population, and each new frame costs one reference run and 24 cue prompts.
- **Two fresh sets, both executed only by `confirm`:** the fresh tokens in the 90 exposed frames (Y1, at most 2160
  pairs; every prediction of every rung in the lock) and in the 18 fresh frames (Y2, at most 432 pairs; predictions at
  stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, read and bases, the 90 locked reference states, the locked score tables and
  lists — no forward pass on any fresh prompt):** the complete prediction table for every fresh token × exposed frame
  and every rung (`S'_k`, `T_k`, `E_k(f, p)`, `G_k(f, p)`, the inherited `S_k`, the random controls, `S_0`, `S_1`,
  `E_1(f)`, `S_2048`, at every `k`) — `ĉ_L`, `F̂`, `Π̂`, `ΔT̂`, the row — and the token means; the frozen floors; the
  exposed statistics; `predictions.md`. The lock records every list and table explicitly and `validate_lock`
  reproduces them from the locked states by the frozen rules.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiments 013–018's procedure):**
  *stage 1* — the eighteen fresh frames' reference runs (exposed tokens only), I1 and the reference half of I5,
  validity (plural-cue head change `≥ 0.25 σ_T`, cue-pair check), each valid frame's `E_k(f, p)` and `G_k(f, p)` and
  their overlaps with `S'_k`, `T_k` and each other, the frame-conditional prediction table for every rung, serialized
  and digested; *stage 2*, only after the digest is re-read from disk — every fresh cue's E-patch in every frame of both
  sets with the layer-3 residuals, the head's row and the block-2 input at both changed positions captured; I1–I10;
  the measured `c_L`, `ΔA_H`, `F`, `Π`, `ΔT` and `e_j^meas`; the oracle lists; scoring against the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–018; a token is scored in a set iff it has at least three
  valid frames there; Y1 needs sixteen scored tokens; Y2 needs at least nine valid cue-final fresh frames, at least
  four valid coordinated fresh frames, and sixteen scored tokens.
- **Precondition (outcome-independent, per set):** the reference rung `S_2048` must reach `c_L` pairs `R² ≥ 0.98`,
  row entry `R² ≥ 0.95` and `ΔT` pairs `R² ≥ 0.95` on the set, and the `c_L` gap (`R²(S_2048) − R²(S_0)`) must be at
  least `0.05` on the pooled scored pairs of the set and on each of its two family splits (the valid cue-final frames
  pooled; the valid coordinated frames pooled; preview gaps `0.23–0.39` and `0.30–0.35`). Otherwise
  `PRECONDITION_FAILED_TOKENS` / `PRECONDITION_FAILED_FRAMES` (naming the condition and, for a split, the split); no
  routing label is issued for that set.
- **Aggregation, fixed:** `κ` and `Δκ` on `R²` pooled over the scored pairs of the set, each pair predicted with its
  own frame's and position's lists; the decision size is `k = 64`; token means and the other sizes reported
  descriptively.
- **Y1 — routing predicted, strict prospective in the cue dimension (24 fresh cues × 90 exposed frames; numbers
  committed before `confirm`):** `Δκ_64(E) ≥ 0.05` on the pooled scored pairs, **and** the frame-count guard —
  `E_64(f)` reaches a higher `c_L R²` than `S'_64` over the scored pairs of the frame in *more than half* of the valid
  frames — **and** the split guard — `Δκ_64(E) ≥ 0.02` on the cue-final split and on the coordinated split. All three
  → `ROUTING_PREDICTED_TOKENS`; otherwise `ROUTING_NOT_PREDICTED_TOKENS` (naming the floor or guard). Preview on
  Experiment 018's Y1: `+0.092`, 52 of 78 frames, splits `+0.072 / +0.166`; jackknife minimum `0.085`.
- **Y2 — routing predicted, frame-conditional prospective (24 fresh cues × 18 new frames; numbers digested at stage
  1):** the same three conditions on the valid fresh frames. All → `ROUTING_PREDICTED_FRAMES_CONDITIONAL`; otherwise
  `ROUTING_NOT_PREDICTED_FRAMES_CONDITIONAL` (naming the floor or guard). Preview on Experiment 018's twelve new
  frames: `+0.154`, 8 of 12 frames, splits `+0.125 / +0.273`; jackknife minimum `0.128`. The eighteen frames are
  reported individually regardless of the outcome.
- **Y3 — the operating-point rule is sufficient (pooled over the scored pairs of both sets, pair-weighted, per-set
  descriptives as in Experiment 018's Y3):** evaluable iff the pooled `Δκ_64(E) ≥ 0.05`; then `ρ_64 ≥ 0.60` →
  `OPERATING_POINT_RULE_SUFFICIENT`; `ρ_64 < 0.60` → `OPERATING_POINT_RULE_INSUFFICIENT`; not evaluable →
  `OPERATING_POINT_RULE_NOT_EVALUABLE`. Preview: `ρ_64` 0.81 pooled (0.75 on Y1, 1.01 on Y2); at `k = 16` only 0.41,
  which is why the label is fixed at the decision size and the other sizes are descriptive. The reading of
  `INSUFFICIENT` is fixed now: *the routing is predictable from the frame's reference state but not from block 2's
  operating point alone — the frame's reshaping of the arriving change (its layer-1–2 reference state through
  Experiment 016's channels and the LayerNorm at its state) selects neurons that the operating point does not.* The
  reading of `SUFFICIENT`: *the frame routes block 2 through its operating point; which neurons carry channel D is set
  by where the frame's reference state places them on the GELU relative to the template base, with the cue
  population's drive on each neuron a fixed per-template profile.*
- **Y4 — the template-family account (pooled over both sets as Y3):** `κ_{c_L}(E_64) − κ_{c_L}(T_64) ≥ 0.05` →
  `TEMPLATE_FAMILY_INSUFFICIENT` (frame-specific selection recovers at least 0.05 beyond the template's own ranking);
  else if `Δκ_64(T) ≥ 0.05` → `TEMPLATE_FAMILY_SUFFICIENT` (the template ranking carries the routing and the frame adds
  less than 0.05); otherwise `TEMPLATE_FAMILY_NOT_DISTINGUISHED` (neither the template nor the frame recovers 0.05
  over the global subset — the case that accompanies a negative Y1/Y2). Preview: `E − T` `+0.071` pooled (`+0.064 /
  +0.112`); `T − S'` `+0.030`.
- **Y5 — membership predicted (secondary; mean over the valid frames of both sets, frame-weighted, at `p_c`):** the
  mean of `|E_64(f, p_c) ∩ O_64(f, p_c)| / 64` is at least `0.60` **and** exceeds the mean of `|S'_64 ∩ O_64(f, p_c)| / 64`
  by at least `0.20` → `MEMBERSHIP_PREDICTED`; otherwise `MEMBERSHIP_NOT_PREDICTED`. The oracle is the *target* of this
  prediction (the neurons the fresh cues' measured responses rank highest), not a selector. Preview against the
  predicted-effect oracle: 0.87 / 0.81 against 0.42 / 0.40; `G`'s 0.72 / 0.71 and `T`'s 0.49 / 0.46 are reported beside it.
- **The headroom reading (preregistered, descriptive, no label; it qualifies a negative Y1 or Y2):** `H_64 =
  κ_{c_L}(O_64) − κ_{c_L}(S'_64)` per set (preview `+0.104 / +0.142`). If a routing label is negative and `H_64 < 0.05`
  on that set, the fixed reading is *no routing headroom at size 64 on this set: the population is fixed to within
  0.05 and the frame acts on amplitudes* (account 1); if negative and `H_64 ≥ 0.05`, *redistribution present, not
  predicted by the tested rules* (account 4). A positive routing label rejects account 1 at size 64 by itself, since a
  frame-specific choice of 64 neurons, made blind to the fresh cues, recovers at least 0.05 more than the best global 64.
- **Descriptive (no floor), reported against the predeclared expectations:** (i) `Δκ_256(E) ≥ 0.03` and `Δκ_16(E) ≥
  0.03` on each set (preview `+0.074 / +0.081` and `+0.069 / +0.174`); (ii) `Δκ_64(E)` against the inherited `S_64(018)`
  at least as large as against `S'_64` (preview `+0.100 / +0.181`); (iii) the random controls' `κ_{c_L}` below `0.50` at
  every `k` and every prospective selector above the best random control by at least `0.30` at `k = 64`; (iv) `κ_{c_L}
  ≥ κ_Π` for every selector, and `κ_row(·) < 0.50` at `k ≤ 256` (Experiment 018's orderings); (v) `G`'s gain smallest in
  the cardinal template (preview negative on the exposed cardinal frames); (vi) `E_1(f)` (the frame's top neuron)
  against `S'_1 = {1987}`, and which neuron is first per frame; (vii) the per-frame distributions of `R²` and `κ` for
  every selector, the number of frames where a selector loses more than `0.05 R²` to `S'_64`, and the jackknife of the
  pooled gain; (viii) `κ_F`, `κ_ΔT` and the head-row `κ` per selector, and the `p_t` lists' overlaps for the coordinated
  frames; (ix) the neuron-level fidelity `R²(e_j, e_j^meas)` per pair (preview unavailable; expected high where the
  reference rung holds).
- Outcome = `Y1 | Y2 | Y3 | Y4 | Y5`. Incidents (replication, identities I1–I10, software defects) stop the phase, are
  recorded with their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol
  version.

## What each outcome means

| Y1 / Y2 | Y3 | Y4 | reading |
|---|---|---|---|
| both positive | sufficient | insufficient | **Account 2, compact form.** The frame's operating point routes channel D: a frame's participating neurons are named from its reference pre-activations and a locked per-template drive profile, blind to the fresh cues, in exposed and new frames; the template ranking does not suffice. The decoded program gains a routing rule: `reference state → operating point → which of the 2048 neurons carry the correction → the transport read`. |
| both positive | insufficient | insufficient | **Account 2, full form.** Routing is predictable from the reference state but needs the frame-conditioned arriving change of calibration cues (Experiment 016's channels at the frame's state); the operating point alone under-selects. The routing rule is real but lives partly upstream of block 2's input. |
| both positive | either | sufficient | **Account 3 with a residue.** The template ranking carries the routing (≥ 0.05 over global) and the frame adds less than 0.05 — but Y1/Y2 positive means the frame-specific selection still cleared 0.05 over the global subset, so the reading is *template-level routing with a frame-level residue below the resolution of this design*; the per-template subcircuits become the object of the next experiment. |
| one positive, one negative | — | — | The dimension that failed is named. Y1 negative with Y2 positive: routing transfers to new frames but not to the new lexical classes in exposed frames — a cue × frame interaction the calibration cues do not carry (account 4 in the cue dimension). Y2 negative with Y1 positive: the selection generalizes to unseen cues in frames whose states contributed to the calibration but not to unseen frames — the reading is *frame-conditional routing not established*, and the headroom reading says whether new frames had room (account 4) or not (account 1). |
| both negative, `H_64 < 0.05` | not evaluable | not distinguished | **Account 1 at size 64.** A fresh-cue-informed choice of 64 neurons does no better than the global 64: the population is fixed to within 0.05 of `κ` and the frame modulates amplitudes. Experiment 018's descriptive observation reduces to amplitude variation inside a fixed set. |
| both negative, `H_64 ≥ 0.05` | not evaluable | not distinguished | **Account 4.** The oracle shows that a different 64 neurons would recover ≥ 0.05 more, and neither the frame's reference state (through the full evaluation or the operating point) nor the template predicts which: the redistribution depends on the fresh cues' own responses in the frame. The chain's decoded upstream state does not carry the routing through the tested rules. |

Y5 is read alongside: `MEMBERSHIP_PREDICTED` with a positive Y1/Y2 says the predicted subset names the right neurons,
not merely a subset with the right aggregate; `MEMBERSHIP_NOT_PREDICTED` with a positive Y1/Y2 says the recovery is
carried by neurons the oracle ranks lower — a warning that the `κ` gain and the membership are different facts and
that the routing statement is about recovery, not about the identity of the top-64.

## Interpretation limits

- Level 0-E and Level 0-G are frame-conditional by construction, as Experiments 016–018's Level 0 are: their inputs
  are reference-run quantities of the frame. Y1 carries the strict boundary for the token dimension; Y2 is conditional
  on the observed reference state of each new frame and aggregate over frames. Nothing here predicts a frame's state
  from its text, and nothing selects neurons from a fresh cue's forward pass.
- The claim is about the transport read (`c_L`) at `p_c`, at size 64, relative to the matched global rule; the other
  sizes, the head objects and the `p_t` position are described, not claimed. It is not the claim that 64 neurons
  suffice (the absolute `κ` of every selector is reported beside its gain), nor that the oracle is the best possible
  subset, nor anything about behaviour.
- `Δκ` is a difference of closure fractions on pooled `R²`; a frame with a very large gap weighs more in the pooled
  value, which is why the frame-count and split guards exist and why the per-frame distribution is always reported.
- Y3 and Y4 are pair-weighted over the union of the two sets (Y1 contributes up to 2160 pairs, Y2 at most 432); their
  per-set values are reported so that a rule working in one dimension only is visible and cannot be read as the label.
- The oracle is in-sample on the pairs it is scored on and is a same-rule ceiling; the headroom reading is
  descriptive and never a label.
- Eighteen new frames, five lexical classes (one of them exhausted by this experiment), three templates, one head, one
  block, this checkpoint.

## Minimal implementation boundary

A module `block_routing.py` reusing `block_concentration.py` (`MaskedChainModel` extended to one mask per changed
position; `upstream_parts`, `RankingAccumulator`, `frame_ranking`, `read_of_outputs`, the `p_t` base, `κ`, the
preconditions, the confirmation-builder helpers, the results-state and lock machinery) and `head_pattern.py`,
`frame_channels.py`, `layer_correction.py` (the pre-activations and the base drive `Δpre_j^T` exposed beside
`hidden`), `attention_patterns.py`, `attention_paths.py`, `read_assembly.py`, `neuron_feature.py`,
`plural_mechanism.py`; the selectors (`S'_k`, `T_k`, the drive-quantile tables, `E_k(f, p)`, `G_k(f, p)`, the random
controls, `S'_1`, `E_1(f)`) as pure functions of the locked states, the weights and the exposed `ΔE`; the measured
per-neuron effect and the oracle as pure functions of the captured block-2 input; a committed extract of Experiment
018's per-pair `F`, `Π`, `ΔT`, `c_L`, row and `S_0` / `S_256` / `S_2048` predictions (11 796 pairs) with the twelve stage-1
state digests and per-frame lists; the confirmation builder with the extended lists and eighteen frames; a runner
with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm` (two stages), `report`. Tests: the
per-position mask with all-ones equals Experiment 018's reference rung to `1e-12` on the fake and with the zero mask
equals its `S_0`; the read identity I10; `E_256` at a cue-final frame equals Experiment 018's `frame_subset` of the same
scores (I9); every selector invariant to poisoned measured quantities and to a poisoned confirmation set; the oracle
computed only from stage-2 captures and every prediction column invariant to poisoned captures; each random control
holds `k` unique indices reproduced from the seed in the frozen order, overlaps recorded, no redraw; the drive
quantiles reproduced from the locked records; `Δκ`, `ρ`, the frame-count guard (a tie is not a win), the split guard,
Y3's evaluability, Y4's three-way logic, Y5, the headroom reading and the preconditions on synthetic tables (a
non-evaluable gap, a split below the gap, exactly half the frames winning, `ρ` at exactly `0.60`, `κ` outside `[0, 1]`
unclipped); the stage barrier; phase isolation; the pinned-model smoke as in Experiments 015–018.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, base, table, list, or prediction may be changed after the lock; no fresh prompt runs before
`confirm`, and no fresh cue prompt runs before its frame's stage-1 lists and predictions are digested.

## Revision history

- **Revision 1**: initial draft, after Experiment 018's closure and the decision to test the frame-dependence of the
  channel-D population as a prospective routing question over a predeclared frame population rather than as an
  explanation of `quantifier-018-4`. Design checks (scratch `extract_019.py`, `analyze_019.py`, `final_019.py`; no
  forward pass; the Experiment 018 lock and closed results state as inputs; the read identity in place of the head
  chain) established: the routing headroom on Experiment 018's fresh sets (`+0.10 / +0.14` at `k = 64`); the full
  evaluation's gain (`+0.09 / +0.15`, 52 of 78 and 8 of 12 frames, jackknife-robust, positive in every template); the
  failure of a first-order slope rule and of a single-mean-drive rule and the success of the exact GELU over a locked
  three-quantile per-template drive profile (`ρ_64` 0.75 / 1.01); the template ranking's small gain (`+0.03 / +0.04`);
  the matched global comparator's difference from the inherited lock (`0.706` against `0.678` on the new frames at
  `k = 64`); the membership overlaps (0.87 / 0.81 against 0.42 / 0.40); the frame-top neurons. Floors: `Δκ_64(E) ≥ 0.05`
  with the frame-count (more than half) and split (`≥ 0.02`) guards; `ρ_64 ≥ 0.60`; Y4's `0.05` margins; Y5's `0.60`
  and `0.20`; the headroom reading at `0.05`; eighteen new frames; the extended candidate lists with the same quotas.
