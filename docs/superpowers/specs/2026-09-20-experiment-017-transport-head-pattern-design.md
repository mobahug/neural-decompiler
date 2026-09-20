# Experiment 017: The Transport Head's Own Pattern Change — Predicting `L03.H04`'s Attention Row and Its End-to-End Number Signal `ΔT` from the Decoded Upstream State — A Prospective Test

**Date:** 2026-09-20

**Status:** Revision 1 — draft for review. No Experiment 017 directory, confirmation set, lock, or model run exists.
Experiments 005–016 are closed and are not amended by this document; Experiment 016's closure stands exactly as
recorded, and the block-2 operating-point concentration observed in the design checks is deferred to Experiment 018.

**Kind:** Prospective, zero-parameter, chain-closing. Experiments 010–016 decoded, layer by layer and in the network's
own terms, how a cue word's weight-only encoding change `ΔE` becomes the transport head's *input*: block 1's MLP
(012), the layer-1–2 attention-pattern changes from the decoded Q/K program (015) with the frame's reference operands,
scale and block-1 operating point (016), and the read `c_L` of the corrected residual. The transport head `L03.H04`
itself has been described with a *frozen* attention pattern since Experiments 009, 011 and 013: its output change was
modelled as the reference pattern carrying the changed values (`P2`, Experiment 013's `c_H`), and the head's own
pattern change was left as a measured remainder. This experiment closes the chain at the head. It asks whether the
cue-induced change in `L03.H04`'s own attention row at the transport position, and therefore its final transported
number signal `ΔT`, can be predicted *prospectively* from the already decoded upstream state — never from a measured
fresh layer-3 residual or a fresh head state — and commits the frozen-pattern account as the rigid alternative to be
rejected. Nothing is fitted; no label is attached to any head, position, frame or cue before scoring.

## Purpose and question

Two facts from the closed experiments frame the question. First, the head's frozen-pattern description is good but
not complete: on the exposed pairs the reference pattern with the changed values explains `R² 0.94` of the head's
output change along `d̂_T`, and the part it cannot carry — the *pattern-change term*, the changed attention weights
times the changed values — has standard deviation `0.16` against `ΔT`'s `0.65`, with self-weight changes of the cue
position of up to `0.44`. Second, the exact recomputation of the head's row from the frame's own reference state and
`ΔE` is an identity (the whole cue-position computation through layers 1–2 and the head's own Q/K/V program at layer 3
are reproduced to `1e-5`; see "Level 1" below), so an *own-state* prediction of the head's pattern would carry no
risk. The claim must therefore be made at the reduced, compositional level: the layer-3 input change `Δ̂x₃` is the
decoded chain's prediction — Experiment 016's Level 0-F at the cue position, block 2's MLP at the frame's operating
point, one propagation step to the transport position where the two differ — and the head's own query, key and value
changes are token-local in direction with the frame's reference operands and scale, exactly as Experiment 016 reduced
layers 1 and 2.

**Design checks on exposed data (outside any results state; 18 exposed frames — six per template, including the
frames that were fresh in Experiments 015 and 016 — × 47 exposed cues = 846 pairs; the patched runs re-executed for
the measurement only; the layer-3 template means over the six sampled frames per template; to be recomputed inside
Tier A over all exposed frames).**

- **Where the decoded `c_L` gap sat.** Experiment 016's decoded `c_L` at Level 0-F kept block 2's MLP at the
  template base and reached pairs `R² 0.79` on its fresh frames (`0.85` on Y1; `0.785` over both sets) where Level 1
  reached `1.00`. Evaluating block 2's MLP at the frame's own operating point `x₂(p_c)` with the *predicted* arriving
  change closes the gap to `R² 1.000`; the layer-1–2 attention pieces contribute `0.003`. Block 2's frame dependence
  is partly concentrated (neuron 1987 alone closes part of the gap, 256 of the 2048 neurons most of it: exposed `0.74 →
  0.86 → 0.97` against `1.00`); that concentration is Experiment 018's question and is not used here — Experiment 017
  evaluates the whole block-2 MLP at the frame's operating point, as Experiment 016's channel C did for block 1.
- **Level 1 — the exact chain is an identity.** From the frame's reference residuals before blocks 1, 2 and 3 at every
  position `≤ p_t` and the weight-only `ΔE`, the weight-only program (Experiment 015's `LayerProgram` at layers 1–3,
  the exact block-1 and block-2 MLPs) reproduces the captured patched layer-3 residuals at `p_c` and `p_t` (relative
  error `≤ 5.4e-6`), `L03.H04`'s captured patched row at `p_t` (`≤ 1.2e-5` per entry) and the measured `ΔT`
  (`≤ 3.3e-5`). Level 1 is the incident-guarded check that the instrument is right; it counts toward no floor.
- **Level 0 — the decoded chain with the head reduced as in Experiment 016.** `Δ̂x₃` from Experiment 016's Level 0-F at
  `p_c` plus block 2's MLP at the frame's operating point (mean relative error of `Δ̂x₃` against the exact change
  `0.020`, maximum `0.067`); at layer 3 the head's query, key and value *changes* token-local in direction at a
  layer-3 template-mean base with the frame's scale, the changed logits formed with the frame's own reference
  operands, the frame's reference keys, values and row otherwise. Pooled entry `R²` of `L03.H04`'s row change
  **0.998** (per-frame minimum `0.986`); end-to-end `ΔT`: pairs **1.000**, token means **1.000** (Spearman
  `1.000`); the pattern-change term: pairs **0.997**, token means **0.9995** (Spearman `1.000`; per-frame minimum
  `0.977`); the self-weight change `R² 1.000`. With the exact head program on the same `Δ̂x₃` (the "exact-head" rung,
  isolating the layer-3 reduction): rows `0.9997`, `ΔT` `0.9999`, term `0.9996`.
- **The ladder (pooled over the 846 pairs; rows / `ΔT` / pattern-change term, `R²`).** Level 1 `1.000 / 1.000 /
  1.000`; exact-head rung `1.000 / 1.000 / 1.000`; **Level 0** `0.998 / 1.000 / 0.997`; Level 0 with block 2's MLP at
  the template base (Experiment 016's decoded-`c_L` convention; `Δ̂x₃` relative error `0.19`) `0.947 / 0.965 / 0.855`;
  Level 0 with no frame channel at layer 3 (Experiment 015's template-only reduction at the head) `0.851 / 0.959 /
  0.743`; **frozen pattern** (the reference row carrying the predicted value changes) `— / 0.941 / ≡ 0`. The head's
  own frame channels and block 2's operating point are each needed; neither is fitted.
- **The frozen-pattern alternative by template (`ΔT`, pairs).** Cardinal `0.890`, quantifier `0.797`,
  coordinated-adjective `0.981`; the cue-final templates pooled `0.913` (token means `0.874`), the coordinated template
  `0.981` (token means `0.986`). The pattern-change term's share of the `ΔT` variance: `0.087` on cue-final pairs,
  `0.018` on coordinated pairs. The split is structural and is predeclared below: in the cue-final templates the cue
  position *is* the head's query position, so the head's own query changes with the cue; in the coordinated template
  the query sits one position later and its change arrives through a single propagation step, so the head's pattern
  moves little and the frozen-pattern account is expected to survive there.

> Can the cue-induced change in `L03.H04`'s own attention row at the transport position, and therefore its
> end-to-end transported number signal `ΔT`, be predicted prospectively — for cue words and frames never measured on
> it — from the decoded upstream state alone (the weight-only encoding change, the frame-conditioned token-local
> changes of layers 1–2, block 1's and block 2's MLPs at the frame's operating points, one propagation step where the
> transport position follows the cue) with the head's own query, key and value changes reduced as Experiment 016
> reduced layers 1–2, and is the frozen-pattern account rejected on the same fresh data where the cue is the head's
> query position?

## The quantities under test (frozen definitions)

Notation as in Experiments 015–016: layers `ℓ ∈ {1, 2, 3}`; `x_ℓ(k)` the frame's reference residual before block `ℓ`
at position `k ≤ p_t`; `p_c` the cue position; `p_t` the transport position (`p_t = p_c` in the cardinal and quantifier
templates, `p_t = p_c + 1` in the coordinated-adjective template — both fixed by the templates since Experiment 009);
`x̄_ℓ^T` the Experiment 012 template-mean bases at `p_c` for `ℓ ∈ {1, 2}`; `ΔE = E(w) − E(ref_T)` weight-only; `d̂_T`
the Experiment 011 locked axis of the head's output site, `σ_T = 1.014` its locked scale; the weight-only program
`LN_ℓ`, `q̃_h`, `k̃_h`, `v_h`, `R_p`, `s_h`, `A_h`, `W_O^h` exactly as frozen in Experiment 015 (partial rotary rotation
on the first 16 of 64 dimensions, scale `1/√64`, softmax over `k ≤` the query position, no BOS); `H = L03.H04`;
`μ(x)`, `σ(x) = √(mean((x − μ)²) + ε)`, `x_c = x − μ(x)` as in Experiment 016.

**Measured (exact; one patched forward pass per pair, as in Experiments 010–016).** From the captured patched run:
`x₃'(p_c)`, `x₃'(p_t)` (the layer-3 residuals at the cue and transport positions), `A_H'(p_t, ·)` (the head's row at
`p_t`, `p_t + 1` entries), and the head's output change `ΔT = ⟨T' − T, d̂_T⟩` (Experiment 010's `head_change`). Derived:

```text
ΔA_H(p_t, k) = A_H'(p_t, k) − A_H(p_t, k),  k ≤ p_t                                    the head's row change (the primary object)
o(k)  = v_H(x₃(k)) W_O^H,   o'(k) = v_H(x₃'(k)) W_O^H  for k ∈ {p_c, p_t}, o'(k) = o(k) otherwise   (positions < p_c are causally invariant)
Π     = ⟨ Σ_k ΔA_H(p_t, k) o'(k), d̂_T ⟩                                                  the pattern-change term
F     = ⟨ Σ_k A_H(p_t, k) [o'(k) − o(k)], d̂_T ⟩                                          the frozen-pattern part (the reference row carrying the value change)
ΔT    = F + Π                                                                             exactly (identity I7, checked per pair)
```

Read units: `ΔT`, `F`, `Π` are in residual units along `d̂_T` (`σ_T = 1.014`); token means are over identical frame
sets. Every statistic used for a floor (`R²`, Spearman) is invariant to this choice of unit.

**Level 1 — the exact chain (identity, incident-guarded, counts toward no floor).** From `x_ℓ(k)` for `ℓ ∈ {1, 2, 3}`,
`k ≤ p_t`, and `ΔE`: `Δx₁(p_c) = ΔE`; at each layer `ℓ ∈ {1, 2}` and each position `p ∈ {p_c, p_t}` the weight-only
program recomputes the attention outputs with the changed residuals at the changed positions and the exact MLP at the
frame's own state, `Δx_{ℓ+1}(p) = Δx_ℓ(p) + Δout_ℓ(p) + Δmlp_ℓ(p)`; at layer 3 the head's row and output from the
changed `x₃(p_c)`, `x₃(p_t)`. Identities: **I4** `‖x₃(p) + Δx₃(p) − x₃'(p)‖ / ‖x₃'(p)‖ ≤ 1e-4` for `p ∈ {p_c, p_t}`;
**I5** the recomputed row equals the captured patched row within `1e-4` per entry, and the reference row recomputed
from the reference state equals the captured reference row within `1e-4`; **I6** the recomputed `ΔT` equals the
measured `ΔT` within `1e-3` (design check `3.3e-5`); **I7** `ΔT = F + Π` from the captured quantities within `1e-4`.
Experiments 015–016's I1–I3 and the 016 recovery identities remain incident-guarded on every exposed pair.

**Level 0 — the decoded chain with the head reduced (the hypothesis; weights, the locked axes, read and bases, the
frame's reference run at positions `≤ p_t`; no patched quantity).**

```text
Upstream (positions p_c; Experiment 016's Level 0-F verbatim, then block 2 at the frame's operating point):
    Δ̂x₁(p_c) = ΔE
    Δ̂x₂(p_c) = ΔE + Δ̂₁(p_c) + Δ̂out₁(p_c)                     Experiment 016's Level 0-F: channels A, B, C; the layer-1 rows and values it predicts
    Δ̂out₂(p_c) = Σ_{h ∈ layer 2} [ Σ_k Â_h v̂_h − Σ_k A_h v_h ] W_O^h   from Experiment 016's Level 0-F layer-2 rows and values at p_c
    Δ̂₂(p_c)  = Σ_j [ GELU(pre_j + Δpre_j) − GELU(pre_j) ] W_out[j],  pre_j = ⟨ln2₂(x₂(p_c)), W_in[:, j]⟩ + b_in[j],
                Δpre_j = ⟨ln2₂(x₂(p_c) + Δ̂x₂(p_c)) − ln2₂(x₂(p_c)), W_in[:, j]⟩                              CHANNEL D: block 2's MLP at the frame's operating point
    Δ̂x₃(p_c) = Δ̂x₂(p_c) + Δ̂out₂(p_c) + Δ̂₂(p_c)

One propagation step (coordinated-adjective frames only, p_t = p_c + 1; x_ℓ(p_t) itself is unchanged at layer 1):
    Δ̂x₂(p_t) = Δ̂out₁(p_t):  layer 1's attention at query p_t with the key and value at p_c recomputed by the weight-only program at x₁(p_c) + ΔE
                              (exact at the frame's own state: Δx₁(p_c) = ΔE exactly), everything else the frame's reference; no MLP change at p_t
    Δ̂x₃(p_t) = Δ̂x₂(p_t) + Δ̂out₂(p_t) + Δ̂₂(p_t):  layer 2's attention at query p_t with the key, value and query recomputed at x₂(p_c) + Δ̂x₂(p_c) and
                              x₂(p_t) + Δ̂x₂(p_t); block 2's MLP at the frame's operating point x₂(p_t) with Δ̂x₂(p_t)
    (There is no locked template base at p_t below layer 3; the step therefore uses the frame's own reference residuals at p_t with the
     predicted arriving changes. Nothing at p_t is read from a fresh run.)

The head (layer 3, L03.H04 only), reduced as Experiment 016 reduced layers 1–2, at the changed positions P = {p_c, p_t}:
    x̄₃^T(p)     the layer-3 template-mean base at position p (locked in this experiment at `explore`: the mean of x₃(p) over the exposed frames of the
                template; one base at p_c per template and one at p_t for the coordinated template — four 512-vectors)
    Channel A₃:  q̃_ref = LN₃(x₃(p_t)) W_Q^H + b_Q^H;  k̃_ref(p) = LN₃(x₃(p)) W_K^H + b_K^H, p ∈ P            (64 numbers each; the frame's own operands)
    Channel B₃:  σ_f(p) = σ(x₃(p)),  σ_f'(p) = σ(x₃(p) + Δ̂x₃(p)),  p ∈ P                                        (two scalars per changed position; σ_f' from the PREDICTED change)
    n_Δ(p)       = γ₃ ⊙ [ (x̄₃^T(p) + Δ̂x₃(p))_c / σ_f'(p)  −  (x̄₃^T(p))_c / σ_f(p) ]                            template direction of the change, frame scale
    Δq̃          = n_Δ(p_t) W_Q^H;   Δk̃(p) = n_Δ(p) W_K^H;   Δv̂(p) = n_Δ(p) W_V^H,  p ∈ P;  Δk̃(k) = Δv̂(k) = 0 otherwise
    ŝ(p_t, k)    = ⟨ R_{p_t} (q̃_ref + Δq̃),  R_k (k̃_ref(k) + Δk̃(k)) ⟩ / 8,  k ≤ p_t     (the general rotated form; for k ∉ P the frame's reference logit exactly;
                                                                                        for p_t = p_c the same-position rotation cancels and this is Experiment 016's self-logit form)
    Â_H(p_t, ·)  = softmax_{k ≤ p_t} ŝ(p_t, ·);   ΔÂ_H = Â_H − A_H
    ô(k)         = (v_H(x₃(k)) + Δv̂(k)) W_O^H
    Π̂            = ⟨ Σ_k ΔÂ_H(p_t, k) ô(k), d̂_T ⟩;   F̂ = ⟨ Σ_k A_H(p_t, k) [ô(k) − o(k)], d̂_T ⟩;   ΔT̂ = F̂ + Π̂
```

Dimensionality of what the frame contributes at the head, recorded as in Experiment 016: channel A₃, `64 × (1 + |P|)`
numbers; channel B₃, `2 |P|` scalars (half of them derived from the predicted change); channel D, the 512-vector
`x₂(p)` through 2048 pre-activations per changed position; and, as in Experiments 015–016, the frame's reference
keys, values and row at layer 3 (which are the frame's row, not a reduction). The reductions are: the direction of
every query, key and value change at `p_c` at layers 1–3 is the template's (Experiments 012, 016 and the layer-3 base
here); the frame enters the *changes* only through its scales, its operands and its two MLP operating points.

**The exact-head rung (descriptive):** the same `Δ̂x₃` with the head's *exact* program at the frame's own state
(`LN₃(x₃(p) + Δ̂x₃(p))` in place of the base form). It isolates the cost of the layer-3 reduction from the cost of the
upstream chain and is not a hypothesis.

**The alternative, committed beside the predictor and defined as rigidly — the frozen pattern (Y3):** `ΔT̂_frozen =
F̂`, the head's reference row carrying the same predicted value changes `Δv̂(p)`; equivalently `Π̂_frozen ≡ 0`. This is
the account of Experiments 009, 011 and 013 (`P2`) with the values predicted rather than measured (on the exposed
pairs the frozen pattern with the exact value changes gives `0.940`, with the predicted ones `0.941`: the alternative's
shortfall is a pattern effect, not a value-prediction effect). No scale, intercept or calibration.

**The ablation ladder (descriptive, predeclared):** Level 1 (identity); the exact-head rung; Level 0; Level 0 with
block 2's MLP at the template base `x̄₂^T` (channel D off; Experiment 016's decoded-`c_L` convention); Level 0 with no
frame channel at layer 3 (channels A₃ and B₃ replaced by the base's operands and scale — Experiment 015's reduction
at the head); the frozen pattern. Predeclared ordering to be reported against, no gate: on the fresh pairs the
layer-3 frame channels' ablation costs more entry `R²` of the row than channel D's (exposed `0.15 > 0.05`) and more of
the pattern-change term (exposed `0.25 > 0.14`); the frozen pattern's `ΔT` shortfall is larger on cue-final than on
coordinated pairs (exposed `0.087 > 0.018`).

**Locked boundary (formal): Level 0 never receives a measured fresh `x₃`, a fresh head state, or any quantity of a
fresh cue forward pass.** Every frame-conditioned input originates from the frame's *reference* prompt (the residuals
`x_ℓ(k)`, `ℓ ≤ 3`, `k ≤ p_t`; the head's reference `q̃`, `k̃`, `v`, row); every cue-conditioned quantity (`ΔE`, `Δ̂₁`,
`Δ̂out₁`, `Δ̂x₂`, `Δ̂out₂`, `Δ̂₂`, `Δ̂x₃`, every `σ'`, `Δq̃`, `Δk̃`, `Δv̂`, `ΔÂ_H`, `Π̂`, `F̂`, `ΔT̂`) is derived algebraically
from the weights, the locked axes, read and bases and those reference quantities before the fresh cue prompt runs.
The order at `confirm` is: reference prompt → channels → predicted upstream change → predicted `Δ̂x₃` → predicted
layer-3 scales → Level 0 table → digest → fresh cue prompt. The prediction table is computed with every capture and
intervention entry point disabled; `confirm` reproduces the locked table before any fresh prompt; the differences
between the predicted and the captured `x₃` (the "`x₃` remainder", relative) and between the predicted and the
captured layer-3 scales (the "scale remainder") are recorded descriptively and never fed back. A test replaces the
patched capture by a poisoned tensor and checks that the table is unchanged.

**Accounting (descriptive, per pair):** I4–I7; the exact-head rung; the ladder; the `x₃` remainder; `σ_f'/σ_f` at layer
3; the self-weight change `ΔA_H(p_t, p_c)` predicted and measured; `F`, `Π`, `ΔT` predicted and measured; the
pattern-change term's share of the `ΔT` variance per template; the per-template frozen-pattern gap; the twelve fresh
frames individually; the decoded-`c_L` ladder extended by one rung (Experiment 016's Level 0-F, then with channel D);
and, after scoring only, which positions the head moves weight between for which cues, in the network's own terms.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked bases, the Experiment 015 program, Experiment
016's Level 0-F (its module used verbatim for the upstream change), the results-state, ledger, lock,
prediction-artifact, two-stage confirmation and incident conventions — all exactly as in Experiments 012–016. Seeds:
runtime `20260916`, control `20260924`.

## Exposed pool (calibration record only; nothing is fitted)

The 207 exposed cue tokens (Experiment 016's 183 and its 24 confirmed) and the 66 exposed frames (its 54 and its 12),
with the 80 nouns. `explore`:

- captures the 66 reference states (the residuals before blocks 1, 2 and 3 at every position `≤ p_t`, the layer-1–2
  pattern rows at `p_c` and `L03.H04`'s row at `p_t`; I1 and the reference half of I5), checks the cue-position
  residuals before blocks 1 and 2 against the Experiment 016 lock and stage-1 digests within `1e-9`, locks the states,
  and computes and locks the four layer-3 template-mean bases;
- re-measures the E-patch of every recorded pair of Experiment 016 (its 6180 exposed and its confirmed pairs) with
  the layer-3 residuals at `p_c`, `p_t` and the head's row at `p_t` captured; replicates Experiment 016's per-pair
  `c_ΔA` (measured and Level 0-F), sixteen self-weight changes and decoded `c_L` values within `1e-6` (a committed
  extract); checks I1–I7 and the Experiment 016 recovery identities on every pair; computes the measured `ΔA_H`, `F`,
  `Π`, `ΔT` and the Level 0, exact-head, ablation and frozen-pattern predictions from the locked states;
- records the exposed statistics (pooled row-entry `R²`, `ΔT` and `Π` on pairs and token means, per template, per
  frame, the self-weight change), the ladder, the `x₃` and scale remainders and the decoded-`c_L` ladder.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 017 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 207 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists, with quotas; classes for coverage and
  reporting only, **no class carries an expectation**. Words left unchosen by earlier experiments' lists were never
  executed and are eligible.
  - `determiner-like` (quota 5): `further`, `usual`, `adjacent`, `nearby`, `leading`, `primary`, `secondary`,
    `principal`, `opposite`, `distinct`, `separate`, `remaining`.
  - `ordinal-or-numeral` (quota 4): `twice`, `once`, `score`, `trio`, `duo`, `solo`, `triplet`, `tens`, `double`,
    `triple`, `couple`, `pair`.
  - `quantity` (quota 5): `finite`, `partial`, `whole`, `entire`, `complete`, `maximal`, `greater`, `massive`,
    `immense`, `modest`, `generous`, `overall`.
  - `possessive-or-pronoun` (quota 4): `none`, `them`, `us`, `you`, `ye`, `him`, `me`, `it`, `she`, `he`, `they`,
    `we`, `anything`, `something`, `everything`, `nothing`.
  - `adjective` (quota 6): `violet`, `metal`, `paper`, `leather`, `blunt`, `costly`, `fancy`, `sweet`, `bitter`,
    `spicy`, `ripe`, `raw`, `rotten`, `faded`, `stolen`, `gentle`, `giant`, `silver`, `purple`, `yellow`.
- **Fresh frames (12, four per template), literal:** cardinal `The bakery sells {cue}`, `The museum displays {cue}`,
  `The fleet carries {cue}`, `The orchard bears {cue}`; quantifier `The report lists {cue}`, `The lecture covers
  {cue}`, `The survey counts {cue}`, `The archive holds {cue}`; coordinated-adjective `Clara and Dev packed {cue}
  hot`, `Anna and Paul carried {cue} cool`, `Eva and Tom hauled {cue} damp`, `Alice and Ben served {cue} crisp`.
  Texts must differ from every exposed frame text; the original cue tokens are used for their own cue prompts. Eight
  cue-final and four coordinated frames, so that Y3's cue-final evaluation and the coordinated split both have fresh
  frames.
- **Two fresh sets, both executed only by `confirm`:** the fresh tokens in the 66 exposed frames (Y1, at most 1584
  pairs; every prediction in the lock) and in the 12 fresh frames (Y2, at most 288 pairs; predictions at stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, read and bases, the 66 locked reference states, the four layer-3 bases — no forward
  pass on any fresh prompt):** the complete prediction table for every fresh token × exposed frame — `ΔÂ_H(p_t, ·)`
  (every entry), the self-weight change, `F̂`, `Π̂`, `ΔT̂`, `ΔT̂_frozen`, the exact-head rung's row, `Π̂` and `ΔT̂`, the
  two ablations' row, `Π̂` and `ΔT̂`, `σ_f'/σ_f` at layer 3, the decoded `c_L` with channel D — and the token means; the
  frozen floors; the exposed statistics; `predictions.md`.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiment 013's procedure):**
  *stage 1* — the twelve fresh frames' reference runs (exposed tokens only), I1 and the reference half of I5,
  validity (plural-cue head change `≥ 0.25 σ_T`, cue-pair check), the frame-conditional prediction table,
  serialized and digested; *stage 2*, only after the digest is re-read from disk — every fresh cue's E-patch in every
  frame of both sets with the layer-3 residuals and the head's row captured; I1–I7; the measured `ΔA_H`, `F`, `Π`,
  `ΔT`; scoring against the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–016; a token is scored in a set iff it has at least three
  valid frames there; Y2 needs at least eight valid fresh frames and sixteen scored tokens; Y1 needs sixteen scored
  tokens.
- **Aggregation, fixed:** the row criterion on every entry `ΔA_H(p_t, k)`, `k ≤ p_t`, pooled over the scored pairs of
  the set (one `R²` per set); the read-unit criteria on `ΔT` over the pooled scored pairs and on token means over
  identical frame sets, and on the token means of `Π`. The floors are set at the near-exactness the claim asserts;
  the `ΔT` floor is chosen above the frozen-pattern account's exposed value (`0.94`), so that passing it is not
  something the alternative could do.
- **Y1 — Level 0, strict prospective (fresh cues × exposed frames; numbers committed before `confirm`):** the head's
  row change, pooled entry **`R² ≥ 0.95`**; the pattern-change term's token means **Spearman `≥ 0.90`** and
  **`R² ≥ 0.90`**; `ΔT` on the pooled pairs **`R² ≥ 0.95`** and on token means **`R² ≥ 0.95`**. All five →
  `HEAD_PATTERN_PREDICTED_TOKENS`; otherwise `HEAD_PATTERN_NOT_PREDICTED_TOKENS` (naming the floor). Exposed-pool
  values: `0.998 / 1.000 / 0.9995 / 1.000 / 1.000`.
- **Y2 — Level 0, frame-conditional prospective (fresh cues × new frames; numbers digested at stage 1):** the same five
  floors **and a frame-collapse guard: every valid fresh frame's pooled row-entry `R²` over its scored pairs
  `≥ 0.90`.** All six → `HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL`; otherwise
  `HEAD_PATTERN_NOT_PREDICTED_FRAMES_CONDITIONAL` (naming the floor, and the frames below the guard). The twelve
  frames are reported individually regardless of the outcome. Exposed per-frame minimum of the row-entry `R²`: `0.986`
  over the 18 design-check frames.
- **Y3 — the frozen-pattern account is rejected where the cue is the head's query position:** over the scored
  **cue-final** pairs of both sets pooled (cardinal and quantifier frames, `p_t = p_c`), the frozen pattern's `ΔT`
  `R²` is **below 0.95** *and* at least **0.05 below** Level 0's. Both → `FROZEN_PATTERN_REJECTED`; otherwise
  `FROZEN_PATTERN_NOT_REJECTED`; with fewer than two scored cue-final token means `FROZEN_PATTERN_NOT_EVALUABLE`.
  Exposed cue-final: `0.913` against `1.000`. **Predeclared split:** on the coordinated pairs the frozen pattern is
  *not* expected to be rejected (exposed gap `0.018`, below the margin); the coordinated gap is reported against that
  expectation and enters no label. A `NOT_REJECTED` with a passing Y1 would mean that on these fresh cues the head's
  own pattern change contributes less than `0.05` of the `ΔT` variance even where the query changes — a genuine result
  about the size of the term, recorded as such.
- **Descriptive (no floor):** the ladder against its predeclared ordering; the exact-head rung; the `x₃` and scale
  remainders; the self-weight change; `F` and `Π` per template and the pattern-change term's variance share; the
  per-template frozen-pattern gap; the twelve frames individually and the per-frame minimum over the exposed frames of
  Y1; the decoded-`c_L` ladder with channel D; which positions the head moves weight between for which cues (in the
  network's own terms, after scoring).
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities I1–I7, software defects) stop the phase, are recorded
  with their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Level 0 is frame-conditional by construction: its inputs are reference-run quantities of the frame at positions
  `≤ p_t` through layer 3, and its keys, values and rows are the frame's. Y1 carries the strict boundary for the
  token dimension; Y2 is conditional on the observed reference state and aggregate over frames. Nothing here predicts
  a frame's attention state from its text alone.
- Passing Y1–Y3 shows that the decoded upstream state — the weight-only encoding change, block 1's and block 2's MLPs
  at the frame's operating points, the frame-conditioned token-local attention changes of layers 1–2, one propagation
  step where the transport position follows the cue — together with the head's own reference operands and scale is
  sufficient to predict the transport head's own attention-pattern change and its end-to-end output change nearly
  exactly on unseen cues and frames, and that the frozen-pattern account, the working description of the head since
  Experiments 009, 011 and 013, is measurably incomplete where the cue is the head's query position. It is *not* the
  claim that the frame is a few numbers (the channels' dimensionality is recorded), nor that block 2's frame
  dependence is concentrated (Experiment 018), nor anything about behaviour. Where the head moves its weight, and
  what that does for the number signal, is described after scoring in the network's own terms; no gloss is
  preregistered.
- Twelve new frames, five lexical classes, three templates, one head, this checkpoint.

## Minimal implementation boundary

A module `head_pattern.py` reusing `frame_channels.py` (Level 0-F for the upstream change at `p_c`, `_sigma`, the
channel switches), `attention_patterns.py` (the program, `ReferenceRow`, row statistics, `locked_state`, comparators,
`compact_analysis`, `_max_numeric_difference`), `layer_correction.py` (`LayerWeights.delta_out` for the MLPs),
`attention_paths.py` (`capture_frame_013`, `head_identity`, `table_digest`, `assert_stage_one_digest`),
`read_assembly.py` (`measure_token_010` with the layer-3 extra sites, `head_change`), `neuron_feature.py` (pool
chain), `plural_mechanism.py`; a `HeadChainModel` with switches (channel D on/off; layer-3 frame channels on/off;
exact head) so that Level 0, the two ablations, the exact-head rung and Level 1 are one code path and the identities
are tested as identities; the layer-3 template bases computed and locked at `explore`; a committed extract of
Experiment 016's per-pair `c_ΔA`, self-weight changes and decoded `c_L` for replication; the confirmation builder with
the frozen lists and twelve frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`,
`confirm`, `report`, `confirm` in two stages. Tests: the Level 1 chain against a captured patched run on the fake
(I4–I7 to `1e-9`); Level 0 with the exact head and all frame channels plus the renormalization terms recovers Level 1
on the fake; the coordinated propagation step; the boundary (a poisoned patched capture leaves the table unchanged);
the floors, Y3 with its cue-final restriction, the guard and the ladder on synthetic tables; the stage barrier; phase
isolation; the pinned-model smoke on a neutral prompt as in Experiments 015–016.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, base, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and
no fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1**: initial draft, after the reviewer's choice of the transport head's own pattern change as Experiment
  017 (three levels — Level 1 identity, Level 0 compositional, the frozen pattern as the rigid alternative; the locked
  boundary that Level 0 never receives a measured fresh `x₃` or fresh head state; the cue-final versus coordinated
  split predeclared; the block-2 concentration kept for Experiment 018).
