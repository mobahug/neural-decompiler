# Experiment 010 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run once the approved Experiment 010 design (revision 2, commit `746c87c`): the exact attribution of `L03.H04`'s head-readable number signal — Experiment 009's P1 read functional with the frame's frozen reference scalars — to the token-local encoding (and its axis/orthogonal and per-neuron parts) and to every attention head and MLP of layers 1–2 at the cue position, over the 63 exposed cue tokens × 24 exposed frames, with the frozen relay/borne classification, the data-derived strata, and the descriptive predictor check. Discovery-only: no fitting, no confirmation set, no claim.

**Architecture:** One module `read_assembly.py` reusing `head_transport.py` (`HeadWeights`, `ov_levels` for the P1 cross-check, results-state and incident patterns), `cue_suppression.py` (`Pool008` dataclass, `stage_axes`, `_sites`, `_ratio`, replication), `supervised_subspace.py` / `cue_decompilation.py` (extract loaders, `_token_vectors`), and `plural_mechanism.py`; a committed extract of Experiment 009's confirmation E-patch means; a runner with phases `validate`, `explore`, `report`.

**Spec:** `docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260923`. Pool: the forty tokens and eighteen frames of Experiment 008 plus the twenty-three tokens and six frames of Experiment 009's confirmation set (63 × 24); the eighty nouns. Nothing outside the pool is executed; the ledger records every prompt and noun.
- Exact identities checked in every run (incidents otherwise): `Δ_{R0} = ΔE` (1e-4); `ρ_f(Δr_c) = ρ_f(ΔE) + Σ_k ρ_f(Δout_k)` (1e-4 relative to the plural cue's measured head change); `f_total` equals the P1 fraction computed by `ht.ov_levels` (1e-6); `Σ_j c_j = ρ_f(ΔE)` (1e-6 relative); the head-result reconstruction (1e-3 relative).
- Replication: E-patch mean shifts must match Experiment 006 (192 pairs, 59 exposed nouns), Experiment 007 (144 pairs, 20 confirmation nouns), and Experiment 009 (138 pairs, 79 nouns) within 1e-6.
- Every constant (`s_min 0.35`, `s_high 0.65`, `c_min 0.10`, `g_max 0.25`, consensus `0.75`, `n_80 ≤ 40`, uninformative floor `0.25 σ_T`, top-20) is a named constant tested against the spec.

## File map

- `src/neural_decompiler/read_assembly.py`
- `experiments/010-read-direction-assembly/run.py`, `inherited/experiment-009-confirmation-epatch-means.json` (derived extract; committed before `explore`), `README.md`, `evidence/`
- `tests/test_read_assembly.py`, `tests/test_experiment_010_runner.py`
- `.gitignore`: `outputs/experiment-010/*`

## Shared definitions

- **Pool:** `build_pool_010(manifest, extension, confirmation_006, confirmation_009)` → a `cs.Pool008` with 24 frames (origins `manifest`/`extension`/`confirmation`/`confirmation-009`), 63 tokens (sources `exposed-16`/`confirmation-24`/`confirmation-23`, categories carried over), 80 nouns, the reference ids, and the plural cue per template.
- **Captures per E-patch run:** every head result and MLP output of layers 1 and 2 at `p_c` (the 18 component keys `K` in the frozen order), `RESID_PRE.L1` and `RESID_PRE.L3` at `p_c`, `ATTN_PATTERN.L3` and `L03.H04` at `p_t`, plus the logits; each frame's reference run is captured once with the same sites (and `RESID_PRE.L3` at every key position for the P1 cross-check).
- **Read functional:** `rho(x) = (A_c^ref / σ_c^ref) · ⟨x − mean(x), γ₃ ⊙ m⟩`, `m = head.read_direction(d̂_T)`, `σ_c^ref = head.scale(r_c^ref)`, `A_c^ref` the reference attention row at `p_c`; denominator `⟨ΔT(pl_T, f), d̂_T⟩` from the plural cue's own E-patch in the frame (uninformative below `0.25 σ_T`).
- **Fractions and layer statistics:** `f_E`, `f_∥`, `f_⊥`, `f_k`, `f_total`, `f_L1`, `f_L2`, `f_layers`, `G = Σ|f_k|`, `D = max_j |Σ_{k≤j} f_k|` along the frozen order; token-level means over informative frames and per template.
- **Neurons:** `Δa = GELU(pre(w)) − GELU(pre(ref))` from the layer-0 MLP (weight-only), `c_j = Δa_j · rho(W_out[j])`; `n_80` on `|c_j|` (desc, ties by index asc); positive/negative masses; top-20 on template-mean `c_j`; Jaccard versus the template's plural cue for the low and high strata.
- **Weight-only score:** `g_E(w, T) = ⟨ΔE − mean, γ₃ ⊙ m⟩ / ⟨ΔE_pl − mean, γ₃ ⊙ m⟩` (no reference scalars; `d̂_T` is the frozen axis), reported beside `f_E`.
- **Strata:** by the token's measured `q_T` (mean over informative frames): low `≤ 0.35`, mid, high `≥ 0.65`.
- **Rules:** `RELAY := |f_layers| ≤ 0.10 ∧ G ≤ 0.25 ∧ D ≤ 0.25`; per-token classes `E_BORNE` / `LAYER_BORNE` / `RELAYED` / `AMPLIFIED` / `MIXED` exactly as the spec; consistent opposers/supporters (`|f_k| ≥ 0.10` in ≥ 75% of informative frames of ≥ 75% of the stratum's tokens); summary over the low stratum (`E_BORNE` / `LAYER_BORNE` naming opposers / `MIXED`, `+CONCENTRATED_ENCODING`); descriptive Spearman/MAE of `f_E`, `g_E`, `f_total` against measured `q_T` over the 63 tokens.

---

### Task 1: Plan, extract, pool, functional

- [ ] Commit this plan; `.gitignore`; the 009 extract (`word|frame_id → dc` over the 79 nouns, with source digests; 138 keys); `build_pool_010`; `ReadFunctional` (scalars from a frame's reference run) and `rho`; the neuron split. Tests: pool counts (63 × 24, sources, plural cues); `rho` equals `⟨ΔT₁, d̂_T⟩` from `ht.ov_levels` on the fake; neuron terms sum to `rho(ΔE)`; `n_80` determinism and ties; Jaccard; extract round trip and the committed extract.
- [ ] Commit `feat: add experiment 010 pool, read functional, and inherited 009 extract`.

### Task 2: Measurement, attribution, rules

- [ ] `measure_010` (reference capture per frame, one E-patch per (token, frame) with the component captures; reference-token zeros), `attribute` (fractions, layer statistics, identities), `classify_token`, `consistent_components`, `strata`, `summarize`, `predictor_check`. Tests: identity on the fake; a planted component change lands in its `f_k`; every classification branch on synthetic tables (including cancelling layer updates → not `RELAY`); consistency rule; summary branches; strata boundaries.
- [ ] Commit `feat: add experiment 010 attribution and rules`.

### Task 3: Exploration, report, runner

- [ ] `run_exploration` (axes, measurements, replication on the three extracts, attribution table, strata, classes, consistency, neurons, summary, predictor check), `render_report` (the ledger table per token: `f_E`, `f_∥`, `f_⊥`, `f_L1`, `f_L2`, `G`, `D`, `f_total`, `q_T`, class; the component table for the low and high strata; neurons), `run.py` with three phases and incident bookkeeping. Tests: state machine on the fake (validate → explore once → report; refusal to re-run; incident path).
- [ ] Commit `feat: add experiment 010 runner`.

### Task 4: Verification and the single run

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `report`; evidence; README with the ledger; root README; memory; hand the summary to the Experiment 011 design (no 011 set is selected or executed from these results).
