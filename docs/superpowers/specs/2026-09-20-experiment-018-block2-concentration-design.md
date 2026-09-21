# Experiment 018: Block 2's Operating-Point Dependence — Is the Frame's Contribution to the Transport Signal Carried by a Fixed Subset of Its 2048 Neurons? — A Prospective Compaction Test, with the Within-Frame Pattern-Term Question Carried as a Preregistered Secondary Outcome

**Date:** 2026-09-20

**Status:** Revision 2 — approved at revision 1 subject to the three changes under "Revision history" (the Y2 split
guards' own evaluability; the pooled Y3 statistic's reading and per-set descriptives; the controls' overlap contract
and the runner phase list), which this revision makes. No Experiment 018 directory, confirmation set, lock, or model
run exists.
Experiments 005–017 are closed and are not amended by this document; Experiment 017's closure stands exactly as
recorded (`HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_NOT_REJECTED`,
never rerun), and its Y3 reading is not revisited here. The secondary outcome below (Y4) is a *new* preregistered
question on *new* data; it does not reopen, re-score or reinterpret Experiment 017.

**Kind:** Prospective, zero-parameter, compaction. Experiments 010–017 wrote the chain from a cue word's weight-only
encoding change `ΔE` to the transport head's output as a composed program whose every head input was predicted, not
measured. The least compact object in that program is **channel D** (Experiment 017): block 2's MLP evaluated at the
frame's own operating point — the frame's 2048 reference pre-activations at each changed position, through the exact
GELU, with the predicted arriving change — where Experiments 012 and 016 had kept block 2 at the template base. Channel
D closed the whole of the decoded-`c_L` gap Experiment 016 left (`R² 1.000` against `0.808` on Experiment 017's fresh
pairs) and costs `0.05` of the head's row entry `R²` and `0.32` of the pattern term's when removed. This experiment asks
whether that frame dependence is *concentrated*: whether a fixed subset `S` of block 2's neurons — chosen once, by a
frozen rule, on the exposed pool, and locked before any fresh prompt — evaluated at the frame's operating point, with
every other neuron left at the template base, carries most of channel D's contribution to the transport read for cue
words and frames never used to choose it. Nothing is fitted; the subset is a locked list of neuron indices, a named
object of the program.

## Purpose and question

Three facts from the closed experiments frame the question. First, Experiment 012's block-2 neuron ledger was *not*
concentrated as a whole (`n_80` ≈ 500 of 2048), but one neuron, 1987, carried three times the mass of the next, and
Experiment 014 decoded its activation change from its own input direction. Second, Experiment 017's design checks
found block 2's *operating-point* dependence — the part of block 2's output change that differs between the frame's
own state and the template base — partly concentrated: neuron 1987 alone closed part of the decoded-`c_L` gap and 256
of the 2048 neurons most of it (exposed `0.74 → 0.86 → 0.97` against `1.00`), a finding deferred to this experiment and
not used there. Third, the compaction, if real, replaces a frame-specific object of 2048 pre-activations per changed
position by 256 (or fewer) named pre-activations plus the LayerNorm statistics, which is what a decompiled program
can name; if the subset does not transfer to new frames, block 2's frame dependence is real but frame-specific, which
is a different and equally reportable fact.

**Design checks on exposed data (outside any results state; no forward pass on any prompt).** Inputs: Experiment 017's
66 locked reference states and the stage-1 states of its 12 fresh frames, the measured rows, `F`, `Π`, `ΔT` and `c_L` of
every Experiment 017 pair (7764 exposed, 1584 Y1, 288 Y2), the weights and the Experiment 011/012/017 locks. The
Experiment 017 chain was re-implemented with channel D replaced by a masked version (`D_S` below); with the mask all
ones it reproduces Experiment 017's Level 0 `ΔT̂` to `3e-15`. To preview prospective behaviour, the subset was ranked
on the 6180 licensed pairs among Experiment 016's 54 exposed frames and 183 exposed cues only (the "preview ranking
pool"; the full product would be 9882 pairs, but the pool's licensing is triangular — see "The ranking rule" below —
and no pair was added or dropped for this check) and evaluated on sets disjoint from it in the cue dimension, the
frame dimension, or both:

- **The ranking (read criterion, frozen below).** The top neurons by mean absolute operating-point effect in read
  units are `1987, 1102, 1726, 129, 1311, 1310, 1068, 1924, 815, 1671` with shares `0.027, 0.010, 0.009, 0.006, …` of
  the summed effect; the cumulative share of the top `1, 16, 64, 256, 1024` is `0.03, 0.09, 0.19, 0.42, 0.85`. Ranking on
  all 66 frames instead of 54 changes 5 of the top 256. Ranking by the 512-norm of the effect instead (the "norm"
  criterion) shares only 92 of the top 256 with the read ranking and is worse on every read-side statistic below
  (e.g. `c_L` pairs `0.941` against `0.970` at 256 on the cue-prospective set), so the read criterion is the one frozen.
- **The ladder on the read side (pairs `R²`; `S_n` = top `n` by the read ranking; `S_0` = every neuron at the template
  base, the Experiment 016 decoded-`c_L` convention with the coordinated `p_t` base defined below; `S_2048` = channel D
  in full = Experiment 017's Level 0).**

  | set (pairs) | object | `S_0` | `S_1` | `S_16` | `S_64` | **`S_256`** | `S_512` | `S_1024` | `S_2048` | gap | **κ(`S_256`)** |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | ranking pool (6180) | `c_L` | 0.852 | 0.920 | 0.960 | 0.968 | **0.983** | 0.993 | 0.998 | 1.000 | 0.148 | **0.89** |
  | 016-confirmed cues × 54 frames (1296) | `c_L` | 0.790 | 0.881 | 0.945 | 0.954 | **0.976** | 0.990 | 0.997 | 0.999 | 0.210 | **0.89** |
  | 017-confirmed cues × 54 frames (1296) | `c_L` | 0.799 | 0.855 | 0.929 | 0.942 | **0.970** | 0.986 | 0.996 | 0.999 | 0.201 | **0.85** |
  | 016's 12 frames × 016 cues (288) | `c_L` | 0.761 | 0.827 | 0.922 | 0.944 | **0.965** | 0.983 | 0.996 | 1.000 | 0.239 | **0.86** |
  | 016's 12 frames × 017 cues (288) | `c_L` | 0.762 | 0.799 | 0.913 | 0.934 | **0.959** | 0.978 | 0.996 | 1.000 | 0.237 | **0.83** |
  | 017's 12 frames × 017 cues (288) | `c_L` | 0.883 | 0.893 | 0.944 | 0.941 | **0.973** | 0.983 | 0.997 | 1.000 | 0.117 | **0.77** |
  | 017-confirmed cues × 54 frames | `F` | 0.920 | 0.939 | 0.968 | 0.974 | **0.987** | 0.994 | 0.998 | 1.000 | 0.080 | **0.84** |
  | 017's 12 frames × 017 cues | `F` | 0.949 | 0.954 | 0.976 | 0.972 | **0.986** | 0.990 | 0.998 | 1.000 | 0.050 | **0.73** |
  | 017-confirmed cues × 54 frames | `Π` | 0.657 | 0.720 | 0.834 | 0.853 | **0.894** | 0.933 | 0.961 | 0.995 | 0.338 | **0.70** |
  | 017's 12 frames × 017 cues | `Π` | 0.731 | 0.751 | 0.847 | 0.868 | **0.928** | 0.959 | 0.978 | 0.997 | 0.266 | **0.74** |
  | 017-confirmed cues × 54 frames | row entries | 0.941 | 0.942 | 0.945 | 0.947 | **0.954** | 0.966 | 0.976 | 0.997 | 0.056 | **0.23** |
  | 017's 12 frames × 017 cues | row entries | 0.962 | 0.961 | 0.962 | 0.964 | **0.974** | 0.975 | 0.984 | 0.998 | 0.037 | **0.35** |

  where `κ(S) = (R²(S) − R²(S_0)) / (R²(S_2048) − R²(S_0))` is the **closure fraction**: the share of channel D's
  contribution to the statistic that the subset carries. Over the six preview sets the closure of `c_L` at `S_256` is
  `0.77–0.89`, of `F` `0.73–0.89`, of `Π` `0.64–0.74`, of `ΔT` `0.62–0.85` (its gap is small, `0.026–0.055`, because `ΔT`
  is dominated by `F`), and of the row entries `0.01–0.35`. Token-mean closures at `S_256`: `c_L` `0.82–0.93`, `F`
  `0.83–0.95`, `Π` `0.72–0.86`. The single neuron `S_1 = {1987}` closes `0.08–0.46` of `c_L` and `0.09–0.43` of `F`.
- **The read path concentrates; the head's row does not.** The row entries lose only `0.02–0.06` of `R²` when channel D
  is removed, and no subset up to 1024 neurons recovers that loss (`κ` at 1024: `0.45–0.64`); the operating-point
  dependence that steers the head's query/key computation is diffuse across the block. The claim below is therefore
  made for the transport read (`c_L`, `F`, `Π`) and the diffuseness of the row is predeclared as an expectation, not
  claimed.
- **Specificity.** On 017's twelve frames × 017 cues, three seeded random subsets of 256 neurons at their own
  operating points reach `c_L` `0.895 / 0.905 / 0.907` (`κ` `0.10–0.21`) and the 256 lowest-ranked `0.883` (`κ` `0.00`),
  against `0.973` for `S_256` (`κ 0.77`): the concentration is carried by *which* neurons, not by how many.
- **Frame stability.** Ranking each frame on its own pairs, the frame's top 256 shares `120–162` (median 137) neurons
  with the pooled `S_256`, and its top 64 shares `17–37` (median 28) with the pooled `S_64`. Per frame, `κ(c_L, S_256)`
  on the 24 frame-prospective frames ranges from `0.00` to `0.96` (cue-final frames `0.49–0.96`; coordinated frames
  `0.00–0.94`, one frame — `Anna and Paul carried … cool` — where even the frame's own top 256 closes nothing). Pooled
  within template on 017's twelve frames: cue-final `κ(c_L) 0.86`, `κ(Π) 0.78`; coordinated `0.62`, `0.59` (on 016's
  twelve: `0.87 / 0.71` and `0.77 / 0.51`). In no frame-prospective frame is `S_256` worse than `S_0` on `c_L` (minimum
  difference `+0.001`). The per-frame closure is too variable for a per-frame floor; the frame-conditional claim is
  pooled, with a per-template split guard and a no-harm guard (below).

> Is the frame's contribution through block 2's MLP to the transport read — channel D of Experiment 017 — carried,
> for cue words and frames never used to choose it, by a fixed subset of 256 of the block's 2048 neurons evaluated at
> the frame's own operating point with every other neuron at the template base, chosen by a frozen read-unit rule on
> the exposed pool and locked before any fresh prompt; is the single top neuron (1987) insufficient; and, on the same
> fresh pairs, is the head's pattern term needed within cardinal and quantifier frames relative to the frozen-pattern
> account (the secondary question)?

## The quantities under test (frozen definitions)

Notation as in Experiment 017: layers `ℓ ∈ {1, 2, 3}`; `x_ℓ(k)` the frame's reference residual before block `ℓ` at
position `k ≤ p_t`; `p_c`, `p_t` the cue and transport positions (`p_t = p_c` in the cardinal and quantifier templates,
`p_t = p_c + 1` in the coordinated-adjective template); `P = {p_c, p_t}` the changed positions; `x̄_ℓ^T` the Experiment
012 template-mean bases at `p_c` (`ℓ ∈ {1, 2}`); `x̄₃^T(p)` the Experiment 017 locked layer-3 bases; `ΔE` weight-only;
`d̂_T`, `σ_T = 1.014` the Experiment 011 locked axis and scale of the head's output site; `r(·)` the Experiment 011 locked
read weight (inner product with the centred vector) and `D_T` the template's weight-only denominator `r(E(pl_T) −
E(ref_T))`, so that `c_L = r(Δx₃(p_c) − ΔE) / D_T` is the decoded read of the net layer-1–2 change as in Experiments 012,
016 and 017; the head objects `ΔA_H(p_t, ·)`, `Π`, `F`, `ΔT = F + Π` exactly as frozen in Experiment 017 (measured from one
patched pass per pair; predicted counterparts from predicted rows and predicted values only). Block 2's MLP: `ln2₂`
(gain `γ`, bias `β`, `ε = 1e-5`), `W_in^(2)` (512 × 2048), `b_in^(2)`, `W_out^(2)` (2048 × 512), exact GELU.

**Channel D (Experiment 017, verbatim) and its masked form `D_S` (this experiment).** For each changed position
`p ∈ P` with reference residual `x₂(p)` and predicted arriving change `Δ̂x₂(p)` (Experiment 016's Level 0-F at `p_c`;
the Experiment 017 propagation step at `p_t`):

```text
Operating points:    pre_j(p)   = ⟨ ln2₂(x₂(p)),      W_in^(2)[:, j] ⟩ + b_in^(2)[j]        the frame's reference pre-activation   (frame-specific, reference run)
                     pre_j^T(p) = ⟨ ln2₂(x̄₂^T(p)),    W_in^(2)[:, j] ⟩ + b_in^(2)[j]        the template base's pre-activation     (locked; at p_c the Experiment 012 base;
                                                                                                                                at p_t the base locked here, below)
Predicted changes:   Δpre_j(p)   = ⟨ ln2₂(x₂(p) + Δ̂x₂(p))     − ln2₂(x₂(p)),    W_in^(2)[:, j] ⟩    exact LayerNorm at the frame's state, before/after the PREDICTED change
                     Δpre_j^T(p) = ⟨ ln2₂(x̄₂^T(p) + Δ̂x₂(p))   − ln2₂(x̄₂^T(p)),  W_in^(2)[:, j] ⟩    the same at the template base
Per-neuron changes:  own_j(p)  = GELU(pre_j(p) + Δpre_j(p))     − GELU(pre_j(p))              (a scalar per neuron; exact GELU, not linearised)
                     tmpl_j(p) = GELU(pre_j^T(p) + Δpre_j^T(p)) − GELU(pre_j^T(p))
Channel D:           Δ̂₂(p)     = Σ_j own_j(p) · W_out^(2)[j, :]                                Experiment 017 (512 numbers; b_out cancels)
Template base (S_0): Δ̂₂^∅(p)   = Σ_j tmpl_j(p) · W_out^(2)[j, :]                               Experiments 012/016's convention (at p_c; and at p_t through the base locked here)
Masked (D_S):        Δ̂₂^S(p)   = Σ_{j ∈ S} own_j(p) · W_out^(2)[j, :]  +  Σ_{j ∉ S} tmpl_j(p) · W_out^(2)[j, :]
Operating-point effect of neuron j (the quantity ranked):   e_j(w, f, p) = own_j(p) − tmpl_j(p)          (a scalar; its output contribution is e_j · W_out^(2)[j, :])
```

`D_S` with `S` = all 2048 neurons is channel D, and Experiment 017's Level 0 exactly; with `S = ∅` it is the
template-base account. **Frame-specific input of `D_S`:** the `|S|` reference pre-activations `pre_j(p)`, `j ∈ S`, and
the two LayerNorm statistics of `x₂(p)` (with the two after the predicted change derived from it) per changed position
— `256 + 4` numbers per position for the hypothesis, against `2048 + 4` for channel D. The neurons outside `S` see the
frame only through the arriving change `Δ̂x₂(p)`, which is itself frame-conditioned upstream (Experiment 016's
channels A–C); that is not changed here and is not claimed to be compact.

**The block-2 base at `p_t` (coordinated template; locked at `explore`).** Experiment 017 had no template base at
`p_t` and kept block 2 at the frame's own operating point there even in its `−D` ablation. This experiment locks
`x̄₂^T(p_t)` for the coordinated-adjective template as the mean of `x₂(p_t)` over the exposed coordinated frames (26 at
`explore`: Experiment 017's 22 and its 4 fresh ones) — one 512-vector, the analogue of Experiment 017's layer-3 base at
`p_t` — so that `D_S` is defined identically at both changed positions. Consequently `S_0` here differs from
Experiment 017's `−D` rung on coordinated pairs (there block 2 at `p_t` stayed at the frame's state); `S_2048` is
identical to Experiment 017's Level 0 on every pair.

**The ranking rule and the locked subset (frozen; computed once at `explore`, from the weights, the locked reference
states and the exposed cues' `ΔE` — no measured quantity of any pair).** The ranking population is the exposed pool's set of
*licensed* pairs — exactly the 9636 pairs Experiment 017 recorded (its 7764 exposed, 1584 Y1 and 288 Y2 pairs), each at
its changed positions `p ∈ P`. It is not the full product of the 231 exposed cues and 78 exposed frames (18 018): the
inherited pool builder licenses each cue token in the frames that were exposed when the token entered the pool and in
its own confirmation's fresh frames, never in frames introduced later, so the structure is triangular — 85
screening-era tokens × the 30 frames of Experiments 005–011, Experiment 012's 24 tokens × its 6 fresh frames (the pairs
its record kept), Experiments 013, 014, 015, 016 and 017's 24 tokens each × 42, 48, 54, 66 and 78 frames, and the two
reference cues `one` and `each` only outside their own templates (10 and 20 frames). The rule was fixed by the pool
construction before this design; no pair is admitted to or removed from the ranking population by any measured value,
and the ranking reads none. (The design-check preview pool above is the same structure restricted to Experiment 016's
54 frames and 183 tokens: `85 × 30 + 24 × 6 + 24 × 42 + 24 × 48 + 24 × 54 + 10 + 20 = 6180`.) For each pair and position
the predicted `Δ̂x₂(p)` is Experiment 016's Level 0-F (and the Experiment 017 propagation step at `p_t`), and

```text
score_j = mean over (w, f, p) in the ranking pool of  | e_j(w, f, p) | · | r( W_out^(2)[j, :] ) | / | D_T(f) |        read units; r weight-only
S_n     = the n neurons with the largest score_j (ties broken by the lower index; none are expected)
```

The **hypothesis subset is `S_256`**; the ladder records `S_0, S_1, S_4, S_16, S_64, S_256, S_1024, S_2048`. `S_1` is the
single top-ranked neuron (1987 on the design-check pool; the rule, not the index, is frozen). The specificity controls
are three subsets of 256 neurons drawn without replacement from `0..2047` by `random.Random(20260924)` (the control
seed) in sequence (`sample(range(2048), 256)` three times from one generator), and the 256 lowest-scored neurons
(`bottom_256`). Overlap between a random control and `S_256`, or among the controls, is expected (about 32 neurons
between two independent 256-subsets) and is recorded; no control is redrawn to force disjointness. Every subset and
control is written into the lock as an explicit index list; the lock records `score_j` for all 2048 neurons.

**Level 1 — the exact chain (identity, incident-guarded, counts toward no floor).** Experiment 017's I1–I7 verbatim on
every exposed and fresh pair; plus **I8**: `S_2048` reproduces Experiment 017's recorded Level 0 predictions (`ΔT̂`,
`Π̂`, `F̂`, the row, decoded `c_L`) on every pair of the Experiment 017 extract within `1e-6` (a replication, not a
floor), and `S_0` at `p_c` equals Experiment 017's `−D` rung on every cue-final pair within `1e-6`.

**Level 0-S — the decoded chain with `D_S` (the hypothesis).** Experiment 017's Level 0 with channel D replaced by
`D_S` at both changed positions; everything else — Experiment 016's Level 0-F at `p_c`, the propagation step, the
layer-3 bases, channels A₃ and B₃, the head's reduced query/key/value changes, `F̂`, `Π̂`, `ΔT̂` — verbatim. The
predicted objects per pair: the decoded read `ĉ_L(S) = r(Δ̂x₃^S(p_c) − ΔE) / D_T`, `F̂(S)`, `Π̂(S)`, `ΔT̂(S)`, the row
`ΔÂ_H(S)`. The **reference rung** `S_2048` is Experiment 017's Level 0; the **template-base rung** `S_0` is the rigid
alternative "block 2 carries no operating-point dependence" (Experiments 012/016's convention).

**The closure fraction (the statistic of the claim).** For an object `X ∈ {c_L, F, Π, ΔT, row entries}`, a set of
scored pairs and a rung `S`:

```text
κ_X(S) = ( R²_X(S) − R²_X(S_0) ) / ( R²_X(S_2048) − R²_X(S_0) )         R² pooled over the scored pairs of the set (entries for the row)
```

`R²_X` is `1 − SS_res / SS_tot` with `SS_tot` about the measured mean, pooled over the scored pairs (over the entries
for the row), as in Experiments 012–017. `κ` is a plain ratio of two `R²` differences and is **never clipped**: a value
below `0` (the subset is worse than the template base) or above `1` (the subset is better than the full channel) is
reported as computed and is meaningful, and every floor and guard below compares the unclipped value. `κ` is
*undefined* — and `X` is *not evaluable* on that set, which is a precondition failure, never a pass or a fail — iff
the denominator, channel D's contribution to `X` there (the "gap"), is below `0.05`; this stops `κ` from becoming a
ratio of noise where channel D happens not to matter. Design-check gaps: `c_L` `0.117–0.239`, `Π` `0.159–0.338`, `F` `0.045–0.086`, `ΔT` `0.026–0.055`, rows `0.024–0.058`;
the floors below are placed on `c_L` and `Π`, whose gaps are large on every preview set, and `F`, `ΔT` and the rows are
reported with their `κ` descriptively. `κ` is invariant to the unit and, unlike an absolute `R²` floor, does not reward
a fresh set on which channel D happens to matter little.

**The alternative committed beside the predictor (Y3) — the single-neuron account:** `S_1`, block 2's frame
dependence carried by the top-ranked neuron alone (neuron 1987 by the design-check ranking, Experiment 014's neuron).
Rigid: the same chain, the same rule, `n = 1`. Its decision variable is `κ_{c_L}` alone: `c_L` is the direct block-2
transport-read quantity — the read of the layer-1–2 net change arriving at the head, the object channel D was
introduced to close in Experiments 016–017 and the quantity the ranking is defined in — whereas `Π` lies downstream of
that read, through the head's query/key computation whose block-2 dependence is diffuse (above), so a single neuron's
share of `Π` would mix the two organizations. `κ_Π(S_1)` is reported beside the decision descriptively. Design-check
`κ_{c_L}(S_1)`: `0.28` and `0.08` on the cue- and frame-prospective 017 sets, `0.43–0.46` on the in-sample and 016-cue
sets (`κ_Π(S_1)`: `0.07–0.37`).

**The ablation ladder and controls (descriptive, predeclared orderings, no gate):** the eight rungs, the three random
subsets and `bottom_256` on every set, for `c_L`, `F`, `Π`, `ΔT` (pairs and token means) and the row entries.
Predeclared, to be reported against: (i) `κ_{row}(S_256) < 0.50` on both fresh sets — the row's operating-point
dependence is diffuse (exposed `0.21–0.35`); (ii) `κ_{c_L}(S_256)` exceeds the largest of the three random subsets'
`κ_{c_L}` by at least `0.30` and exceeds `bottom_256`'s (exposed margins `≥ 0.56`); (iii) `κ_{c_L}(S_256) ≥ κ_Π(S_256)`
(exposed on every set). No monotonicity of the ladder in `n` is predeclared (the exposed ladder is not strictly
monotone at 16 → 64 on one set).

**Locked boundary (formal).** Level 0-S never receives a measured fresh `x₂`, `x₃`, head state, or any quantity of a
fresh cue forward pass, at either position — Experiment 017's boundary, verbatim, with the poisoned-capture test at
both positions covering every column of every rung. In addition: `S_n`, the controls and `x̄₂^T(p_t)` are computed at
`explore` from the weights, the locked reference states of the exposed frames and the exposed cues' `ΔE` only; a test
poisons every measured quantity of the exposed pairs (rows, residuals, `F`, `Π`, `ΔT`, `c_L`) and checks that the ranking
and the subsets are unchanged; a second test replaces the confirmation set's tokens and frames by poisoned ones and
checks the same. The fresh cues' `ΔE` and the fresh frames' states enter no subset. Order:

```text
exposed pool (weights, locked states, exposed ΔE)  →  score_j  →  S_n, controls, x̄₂^T(p_t)  →  lock
fresh cue word  →  ΔE (weight-only)  →  Experiment 017's chain with D_S  →  ĉ_L(S), F̂(S), Π̂(S), ΔT̂(S), row(S) for every rung  →  table  →  digest  →  fresh cue prompt (stage 2)
```

**Accounting (descriptive, per pair):** I1–I8; every rung's `ĉ_L`, `F̂`, `Π̂`, `ΔT̂`, row; the per-neuron `e_j` for `j ∈
S_256` (which of the 256 fire, per template); the per-frame closure of `c_L` for every fresh frame; the per-frame
ranking overlap with `S_256` and `S_64` (each fresh frame ranked on the exposed cues' predicted changes at its own
reference state — reference-run quantities only, at stage 1) and the frame-own `S_256`'s closure against the locked
`S_256`'s (the "frame-specific oracle", descriptive); the twelve fresh frames individually; the per-template split; the
Experiment 017 objects (rows, `Π`, `ΔT`) at the reference rung against Experiment 017's floors, for the record.

**The secondary question (Y4): is the pattern term needed within cardinal and quantifier frames?** Experiment 017
established prospectively that `L03.H04`'s pattern change and its term `Π` are predicted from the decoded upstream
state (`Π` token means `R² 1.000`, per-frame `Π` `R² 0.948–1.000`); its Y3 asked whether the frozen-pattern account
(`ΔT̂_frozen = F̂`, `Π̂ ≡ 0`) is *measurably incomplete* on the pooled cue-final pairs and found `0.958` against a `0.95`
ceiling with a `0.042` gap, and the fixed reading applied. Descriptively, the pooled value was carried by the
between-template spread of `ΔT`, and within each of Experiment 017's twelve new frames the frozen pattern reached `ΔT`
`R²` `0.320 / 0.825 / 0.714 / 0.450` (cardinal) and `0.730 / −0.068 / 0.815 / −0.463` (quantifier) where Level 0 reached
`0.990–1.000` — the pattern term dominates *within* a frame, where the value term's spread across cues is small. That
observation was not preregistered and is not applied to Experiment 017. Here it is asked as a new question on the
new fresh frames, from the **reference rung `S_2048`** (Experiment 017's Level 0, verbatim) — never from `S_256`:

> Within each new cardinal and quantifier frame, does omitting the decoded pattern term produce a substantial
> predictive loss — the frozen pattern's per-frame `ΔT` `R²` below `0.90` — wherever the reference rung itself
> predicts the frame's `ΔT` (`R² ≥ 0.95`)?

This does not ask whether the attention-pattern mechanism exists or has been decoded; Experiment 017 already
established that prospectively. It asks whether omitting that decoded pattern term produces a substantial predictive
loss within cardinal and quantifier templates on this fresh evaluation set. It is conditional on the reference rung
(a frame where the rung itself fails is not interpreted), stratified per frame (so neither the template mixture nor
the cue composition sets the headline), and restricted to the cue-final templates (the coordinated template's pattern
share moved opposite to the cue-final templates' between Experiment 017's exposed and fresh cues — `0.028 → 0.049`
while `0.131 → 0.095` and `0.167 → 0.094` — so no directional claim is made for it; it is reported descriptively).
**Y4 cannot change Y1–Y3**, and its failure means only that the new cue sample did not establish the preregistered
within-frame condition; it does not reopen Experiment 017's mechanism or its Y3.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked bases, the Experiment 015 program, Experiment
016's Level 0-F, Experiment 017's chain (its module used verbatim for everything but channel D), the Experiment 017
lock's layer-3 bases and 66 reference states and its stage-1 digests of the twelve 017-fresh frames, the
results-state, ledger, lock, prediction-artifact, two-stage confirmation and incident conventions — all exactly as in
Experiments 012–017. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration record only; nothing is fitted)

The 231 exposed cue tokens (Experiment 017's 207 and its 24 confirmed) and the 78 exposed frames (its 66 and its 12),
with the 80 nouns. `explore`:

- captures the 78 reference states (the residuals before blocks 1, 2 and 3 at every position `≤ p_t`, the layer-1–2
  rows at `p_c` and `L03.H04`'s row at `p_t`), checks the 66 against the Experiment 017 lock within `1e-9` and the 12
  against Experiment 017's stage-1 digests, locks the states, and computes and locks `x̄₂^T(p_t)`;
- computes the ranking pool's `e_j` from the locked states and the exposed cues' `ΔE` (no measured quantity), the
  scores, `S_n`, the controls; locks them;
- re-measures the E-patch of every pair of Experiment 017's record (its 7764 exposed and its 1872 confirmed pairs) with
  the layer-3 residuals and the head's row captured; replicates Experiment 017's per-pair `F`, `Π`, `ΔT`, `c_L`, row
  and Level 0 predictions within `1e-6` (a committed extract, as Experiment 017 did with Experiment 016's); checks
  I1–I8 on every pair; computes every rung's predictions from the locked states;
- records the exposed statistics: every rung's `R²` and `κ` for `c_L`, `F`, `Π`, `ΔT` (pairs and token means) and the
  row entries, pooled, per template, per frame; the controls; the per-frame ranking overlaps; the reference rung's
  Experiment 017 statistics; the frozen pattern per frame.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 018 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 231 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists below (Experiment 017's lists, whose
  first eligible entries it consumed), with the same quotas; classes for coverage and reporting only, **no class
  carries an expectation**. Tokenizer-checked on 2026-09-20, the first eligible entries are expected to be:
  - `determiner-like` (quota 5): `primary`, `secondary`, `principal`, `opposite`, `distinct`, `separate`, `remaining`.
  - `ordinal-or-numeral` (quota 4): `duo`, `solo`, `triplet`, `tens`, `double`, `triple`, `couple`, `pair`.
  - `quantity` (quota 5): `maximal`, `greater`, `massive`, `immense`, `modest`, `generous`, `overall`.
  - `possessive-or-pronoun` (quota 4): `ye`, `him`, `me`, `it`, `she`, `he`, `they`, `we`, `anything`, `something`,
    `everything`, `nothing`.
  - `adjective` (quota 6): `fancy`, `sweet`, `bitter`, `spicy`, `ripe`, `raw`, `rotten`, `faded`, `stolen`, `gentle`,
    `giant`, `silver`, `purple`, `yellow`.
- **Fresh frames (12, four per template), literal, tokenizer-checked to build and to differ from every exposed frame
  text:** cardinal `The harbor shelters {cue}`, `The studio records {cue}`, `The stable houses {cue}`, `The depot
  receives {cue}`; quantifier `The syllabus outlines {cue}`, `The registry tracks {cue}`, `The notice announces {cue}`,
  `The atlas maps {cue}`; coordinated-adjective `Ivan and Rosa carved {cue} smooth`, `Jonah and Mei rinsed {cue} pale`,
  `Ella and Kofi trimmed {cue} rough`, `Ruth and Dario tasted {cue} sour`. The original cue tokens are used for their own
  cue prompts. Eight cue-final and four coordinated frames, so that Y4 has eight frames and the coordinated split of
  Y2's guard has four.
- **Two fresh sets, both executed only by `confirm`:** the fresh tokens in the 78 exposed frames (Y1, at most 1872
  pairs; every prediction of every rung in the lock) and in the 12 fresh frames (Y2, at most 288 pairs; predictions at
  stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, read and bases, the 78 locked reference states, `x̄₂^T(p_t)`, the locked subsets —
  no forward pass on any fresh prompt):** the complete prediction table for every fresh token × exposed frame and
  every rung and control — `ĉ_L`, `F̂`, `Π̂`, `ΔT̂`, `ΔT̂_frozen` (from the reference rung), the row — and the token
  means; the frozen floors; the exposed statistics; `predictions.md`.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiments 013–017's procedure):**
  *stage 1* — the twelve fresh frames' reference runs (exposed tokens only), I1 and the reference half of I5, validity
  (plural-cue head change `≥ 0.25 σ_T`, cue-pair check), the frame-conditional prediction table for every rung, the
  per-frame ranking overlaps and the frame-specific oracle (reference-run quantities only), serialized and digested;
  *stage 2*, only after the digest is re-read from disk — every fresh cue's E-patch in every frame of both sets with
  the layer-3 residuals and the head's row captured; I1–I7; the measured `c_L`, `ΔA_H`, `F`, `Π`, `ΔT`; scoring against
  the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–017; a token is scored in a set iff it has at least three
  valid frames there; Y1 needs sixteen scored tokens; Y2 needs at least six valid cue-final fresh frames, at least
  three valid coordinated fresh frames, and sixteen scored tokens.
- **Precondition (outcome-independent, per set):** the reference rung `S_2048` must reach `c_L` pairs `R² ≥ 0.98`,
  row entry `R² ≥ 0.95` and `ΔT` pairs `R² ≥ 0.95` on the set (Experiment 017's chain must hold on the new pairs before a
  compaction of it is interpreted; exposed `0.999–1.000 / 0.995–0.998 / 0.9994–0.9996`), and the `c_L` and `Π` gaps
  (`R²(S_2048) − R²(S_0)`) must be at least `0.05` on the pooled scored pairs of the set; **for Y2, in addition, each
  split used as a decision guard — the valid cue-final fresh frames pooled and the valid coordinated fresh frames
  pooled — must itself have a `c_L` gap of at least `0.05`** (preview: cue-final `0.095–0.301`, coordinated `0.274–0.302`).
  Otherwise `PRECONDITION_FAILED_TOKENS` / `PRECONDITION_FAILED_FRAMES` (naming the condition and, for a split, the
  split); no concentration label is issued for that set, and a split whose gap is below `0.05` is never scored as a
  pass or a fail.
- **Aggregation, fixed:** `κ` on `R²` pooled over the scored pairs of the set (one value per object per set); token
  means over identical frame sets reported descriptively.
- **Y1 — `S_256`, strict prospective (fresh cues × exposed frames; numbers committed before `confirm`):** `κ_{c_L}(S_256)
  ≥ 0.70` and `κ_Π(S_256) ≥ 0.50`. Both → `CHANNEL_D_CONCENTRATED_TOKENS`; otherwise `CHANNEL_D_NOT_CONCENTRATED_TOKENS`
  (naming the floor). Preview values on the two cue-prospective sets: `0.89 / 0.71` and `0.85 / 0.70`.
- **Y2 — `S_256`, frame-conditional prospective (fresh cues × new frames; numbers digested at stage 1):** the same two
  floors on the pooled fresh-frame pairs, **and a split guard** — `κ_{c_L}(S_256) ≥ 0.60` over the valid cue-final fresh
  frames pooled and `≥ 0.40` over the valid coordinated fresh frames pooled (neither template split may carry the
  pooled value alone; preview `0.86–0.87` and `0.62–0.77`; each split's own `c_L` gap `≥ 0.05` is part of the
  precondition above) — **and a no-harm guard** — in every valid fresh frame,
  `R²_{c_L}(S_256) ≥ R²_{c_L}(S_0) − 0.05` over its scored pairs (the locked subset never makes a frame worse than the
  template base; preview minimum difference `+0.001` over 24 frames). All → `CHANNEL_D_CONCENTRATED_FRAMES_CONDITIONAL`;
  otherwise `CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL` (naming the floor or guard and the frames). The twelve
  frames are reported individually regardless of the outcome. Preview pooled values on the three frame-prospective
  sets: `κ_{c_L}` `0.77 / 0.83 / 0.86`, `κ_Π` `0.74 / 0.64 / 0.68`.
- **Y3 — the single-neuron account is rejected:** over the scored pairs of both sets pooled, `κ_{c_L}(S_1) < 0.50` *and*
  `κ_{c_L}(S_256) − κ_{c_L}(S_1) ≥ 0.25`. Both → `SINGLE_NEURON_REJECTED`; otherwise `SINGLE_NEURON_NOT_REJECTED`; with
  either set failing its precondition, `SINGLE_NEURON_NOT_EVALUABLE`. `κ_{c_L}` is the sole decision variable (the
  direct block-2 read quantity; see "The alternative"); `κ_Π(S_1)` is reported beside it and enters no label. **The
  pooled statistic is pair-weighted**: Y1 contributes up to 1872 pairs and Y2 at most 288, so `SINGLE_NEURON_REJECTED`
  means rejection on the pair-weighted combined prospective distribution — dominated by the cue-new, exposed-frame
  pairs — and *not* rejection separately in each generalization dimension. `κ_{c_L}(S_1)` and the gap
  `κ_{c_L}(S_256) − κ_{c_L}(S_1)` are reported for Y1 and for Y2 individually, descriptively (preview: `0.28 / 0.57` on the
  cue-prospective 017 set, `0.08 / 0.69` on the frame-prospective one), so that a single neuron failing on one set and
  working unusually well on the other is visible and cannot be read as the pooled label. Preview: `κ_{c_L}(S_1)` `0.08–0.46`, gap to `S_256`
  `≥ 0.43`. Y3 is evaluated independently of Y1 and Y2: the single neuron could carry half of the gap while `S_256`
  carries it all, and the reading of that case is fixed now — *block 2's operating-point dependence of the transport
  read is carried substantially by one neuron and the concentration claim reduces toward Experiment 014's object.*
- **Y4 — secondary, from the reference rung only, cannot change Y1–Y3:** a valid fresh cardinal or quantifier frame
  is *evaluable* iff the reference rung's `ΔT` `R²` over its scored pairs is at least `0.95`; at least six of the eight
  must be evaluable, else `PATTERN_TERM_NOT_EVALUABLE_WITHIN_FRAMES`. If every evaluable frame's frozen-pattern `ΔT`
  `R²` (`ΔT̂_frozen = F̂` of the reference rung) is **below `0.90`** → `PATTERN_TERM_NEEDED_WITHIN_FRAMES`; otherwise
  `PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES` (naming the frames at or above `0.90`). The four coordinated frames'
  values are reported beside it with no expectation and enter no label. The reading of a failure is fixed now: *the
  new cue sample did not establish the preregistered within-frame variance-share condition; the pattern mechanism
  (Experiment 017) is not in question, and Experiment 017's Y3 is not reopened.* Calibration on Experiment 017's twelve
  frames: all eight cue-final frames evaluable (`0.990–1.000`); frozen `0.320, 0.825, 0.714, 0.450, 0.730, −0.068, 0.815,
  −0.463`, the largest `0.075` below the ceiling.
- **Descriptive (no floor):** the ladder and controls against their predeclared orderings (i)–(iii); `κ` for `F`, `ΔT`
  and the rows; token means; the per-frame closure and ranking overlaps; the frame-specific oracle; which of the 256
  neurons fire, per template; the reference rung's Experiment 017 statistics; the per-template frozen-pattern values
  including the coordinated frames.
- Outcome = `Y1 | Y2 | Y3 | Y4`. Incidents (replication, identities I1–I8, software defects) stop the phase, are
  recorded with their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol
  version.

## Interpretation limits

- Level 0-S is frame-conditional by construction, as Experiment 017's Level 0 is: its inputs are reference-run
  quantities of the frame through layer 3. Y1 carries the strict boundary for the token dimension; Y2 is conditional
  on the observed reference state and aggregate over frames. Nothing here predicts a frame's state from its text.
- Passing Y1–Y3 shows that channel D's contribution to the transport read — the frame's operating-point dependence
  of block 2's MLP as it reaches the head's value path and the decoded read — is carried mostly by a fixed, named set
  of 256 neurons (an eighth of the block), the same set for unseen cues and unseen frames, and that the single top
  neuron does not suffice. It is *not* the claim that channel D is compact for the head's attention row (the row's
  small operating-point dependence is diffuse and is predeclared so), nor that the arriving change `Δ̂x₂` is compact
  (Experiment 016's channels stay as they are), nor that 256 is the smallest such set (the ladder is descriptive), nor
  anything about behaviour. `κ` measures the share of channel D's contribution, not the absolute fidelity of the
  reduced chain, which is reported beside it.
- Y3's label is a statement about the pair-weighted union of the two fresh sets, not about each generalization
  dimension; the per-set values beside it are the only basis for a statement about new frames alone.
- Y4 is a variance-share statement about the fresh cue sample within frames, conditional on the reference rung; it
  is not a mechanism claim in either direction.
- Level 1 and Level 0-S are kept apart throughout; I8 (the reference rung reproducing Experiment 017) is an
  instrument check and contributes nothing to the evidence for `S_256`.
- Twelve new frames, five lexical classes, three templates, one head, one block, this checkpoint.

## Minimal implementation boundary

A module `block_concentration.py` reusing `head_pattern.py` (the `HeadChainModel`, its `Variant` extended by a
per-neuron mask for channel D at both positions and the `p_t` base; `head`, `head_terms`, the capture, identities,
extract, confirmation and lock machinery), `frame_channels.py`, `layer_correction.py` (`LayerWeights.hidden` split into
own/template hidden deltas), `attention_patterns.py`, `attention_paths.py`, `read_assembly.py`, `neuron_feature.py`,
`plural_mechanism.py`; the ranking (`score_j`, `S_n`, the seeded controls, `bottom_256`) as pure functions of the
locked states, the weights and the exposed `ΔE`; the `p_t` base computed and locked at `explore`; a committed extract
of Experiment 017's per-pair `F`, `Π`, `ΔT`, `c_L`, row and Level 0 predictions (9636 pairs) for replication; the
confirmation builder with the frozen lists and twelve frames; a runner with phases `validate`, `freeze-confirmation`,
`explore`, `lock`, `confirm` (in two stages), `report`. Tests: the masked channel with the all-ones mask equals
`head_pattern`'s Level 0 to `1e-12` on the fake; with the zero mask at `p_c` equals its `−D` rung on cue-final pairs; the
ranking is invariant to poisoned measured quantities and to a poisoned confirmation set; the subset lists are
explicit and reproduced from the lock; each control holds 256 unique indices, is reproduced exactly from the seed,
and its overlaps with `S_256` and with the other controls are recorded (no redraw); `κ`, the preconditions, the guards, Y3
and Y4 on synthetic tables (including a non-evaluable gap, a frame below the reference-rung condition, a frame at
exactly `0.90`); the stage barrier; phase isolation; the poisoned-capture test at both positions for every rung; the
pinned-model smoke as in Experiments 015–017.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, base, subset, or prediction may be changed after the lock; no fresh prompt runs before
`confirm`, and no fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1**: initial draft, after Experiment 017's closure and the decision to fold the within-frame pattern-term
  question into this experiment as a secondary outcome rather than spend a standalone experiment and a fresh cue set
  on a variance-share question whose mechanism Experiment 017 already settled. Design checks (scratch
  `design_check_018.py`, `design_check_018b.py`; no forward pass; Experiment 017's records and locks as inputs)
  established: the read criterion over the norm criterion; the read path's concentration (`κ_{c_L}(S_256)` `0.77–0.89`
  across six preview sets) against the row's diffuseness (`0.01–0.35`); the specificity against random and bottom
  subsets; the per-frame variability that rules out a per-frame floor and motivates the split and no-harm guards; the
  single neuron's `0.08–0.46`. Floors: `κ_{c_L} ≥ 0.70`, `κ_Π ≥ 0.50`, the Y2 split guard `0.60 / 0.40`, the no-harm guard
  `0.05`, Y3 `< 0.50` with a `0.25` margin, Y4 per-frame frozen `< 0.90` conditional on the reference rung `≥ 0.95`.
  Tightened before review, at the user's request: `κ` written out as an unclipped ratio with its evaluability rule
  and the meaning of values outside `[0, 1]`; the ranking population stated as the pool's licensed pairs with the
  triangular licensing that makes it 9636 (and the preview pool 6180) rather than a full product, fixed before this
  design and read from no measurement; `κ_{c_L}` justified as Y3's sole decision variable with `κ_Π(S_1)` descriptive.
  Reviewed: **approve after minor revision 2** — no design-check rerun, threshold tuning or forward pass; three
  preregistration details: (1) each Y2 split used as a decision guard must itself have a `c_L` gap `≥ 0.05`, else
  `PRECONDITION_FAILED_FRAMES` rather than pass/fail; (2) the pooled Y3 statistic is pair-weighted toward Y1, so
  `SINGLE_NEURON_REJECTED` is to be read on the combined prospective distribution, with `κ_{c_L}(S_1)` and the
  `S_256 − S_1` gap reported per set descriptively; (3) random 256-subsets are not disjoint from `S_256` or each other
  — record overlaps, do not redraw — and the runner phase list has no duplicated `confirm`. The inherited Y4
  calibration values were verified against the Experiment 017 record.
- **Revision 2**: the three changes made, textually only; `S_256`, the κ floors, the ranking rule, the Y1–Y4 structure,
  the fresh-set size and the confirmation protocol are unchanged. The Y2 precondition now names the split gaps
  (preview cue-final `0.095–0.301`, coordinated `0.274–0.302`); Y3 carries its pair-weighted reading and per-set
  descriptives (preview `0.28 / 0.57` and `0.08 / 0.69`); the controls' overlap is recorded, never redrawn; the phases
  are `validate → freeze-confirmation → explore → lock → confirm (two stages) → report`.
