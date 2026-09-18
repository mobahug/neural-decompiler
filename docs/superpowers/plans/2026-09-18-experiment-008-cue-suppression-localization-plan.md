# Experiment 008 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run once the approved Experiment 008 design (revision 2, commit `af0b7cc`): a discovery-only causal localization, on the fully exposed pool (40 cue tokens × 18 frames × 79 single-token nouns), of where the apparent number signal of `this`-like cues is suppressed — a staged trace of the E-patch intervention with exact direct-effect deltas (M1, M4), component patching along the encoding number axis (M2), cross-context patching (M3), the sign-normalized anomaly score against the frozen 007 program, and the frozen classification rules and summary.

**Architecture:** One module `cue_suppression.py` reusing `plural_mechanism.py` (prompts, clean-run cache with standard captures, `run_patched` with captures, contrasts, `site_axis`, `direct_effects`/`check_direct_effects`, results-state digests), `cue_decompilation.py` (exposed pool, confirmation loading, results state and phase isolation, `e_slice`), and `supervised_subspace.py` (`load_linear_program`, the inherited 006 extract loader). One runner with phases `validate`, `explore` (once), `report`. No lock, no confirm phase.

**Spec:** `docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; runtime seed `20260916`, control seed `20260921` (recorded only).
- Pool: 18 frames = 12 exposed (`cd.exposed_pool`) + 6 confirmation frames (`cd.load_confirmation(...).frames`, origin recorded as `confirmation`); 40 tokens = the 16 exposed `(name, id)` pairs + the 24 confirmation tokens (`word`, `token_id`, category); 80 nouns = 60 exposed + 20 confirmation (79 single-token). Every prompt is an exposed token in an exposed frame; the ledger records every executed prompt and noun; no other token, frame, or noun may appear.
- The 007 `selected` program is loaded from `outputs/experiment-007/parameters/selected` through `ss.load_linear_program` and its `parameters.json` text digest must equal the committed 007 lock's `parameters.selected` (`3d1bf4a4…4701`); otherwise `explore` refuses (a missing directory is a precondition failure, not an incident).
- Replication incidents: the 192 old-pool E-patch means versus the committed 006 extract (1e-6); the 144 confirmation-token means on the six confirmation frames versus a committed extract of the 007 results state (1e-6); the identity `Δ_{R0} = ΔE_T(w)` (1e-4) in every run; the direct-effect identity and model gap within Experiment 005's tolerances in every run.
- Every constant of the spec (`s_min = 0.3`, `κ = 0.5`, `g_max = 0.25`, `x_min = 0.5`, `C_min = 2.0`, probe floor `0.5`, consensus `0.75`, stage-uninformative floor `0.25 σ_s`, cancellation denominator floor `0.25 |Δc(pl_T, f)|`, strata `±1.5`) is a named constant tested against the spec.
- Discovery-only: no lock, no confirmation, no claim edit; the report and the per-token table are the deliverable.

## File map

- `src/neural_decompiler/cue_suppression.py`
- `experiments/008-cue-suppression-localization/run.py`
- `experiments/008-cue-suppression-localization/inherited/experiment-007-confirmation-epatch-means.json` (derived extract; committed before `explore`)
- `experiments/008-cue-suppression-localization/README.md`, `evidence/`
- `tests/test_cue_suppression.py`, `tests/test_experiment_008_runner.py`
- `.gitignore`: `outputs/experiment-008/*`

## Shared definitions

- **Sites captured per patched run** (`run_patched(..., capture_sites=...)`): `("RESID_PRE.L1", p_c)` (= `R0`), `("RESID_PRE.L3", p_c)` (`R1`), `("ATTN_PATTERN.L3", p_t)` (head 4, key `p_c`), `("L03.H04", p_t)` (`T`), `("RESID_PRE.L4", p_t)` (`R2`), `("L04.MLP", p_t)`, `("L05.MLP", p_t)`, `("RESID_POST.L5", p_t)` (`R3`), plus `EMBED` and every universe key at `p_t` for the direct-effect decomposition. Clean runs come from `PromptCache` (standard sites cover the same).
- **Stage axes**: `pm.site_axis(label, sg_vectors, pl_vectors)` over the 18 frames' clean original cue pairs at each site (`a` = singular cue, `b` = plural cue, so the plural cue is positive); `d̂_E` from the weight-only `E(sg_T)`, `E(pl_T)`.
- **Signal fraction** `ŝ_s(w, f) = ⟨Δ_s(w, f), d̂_s⟩ / ⟨Δ_s(pl_T, f), d̂_s⟩`, `None` if `|⟨Δ_s(pl_T, f), d̂_s⟩| < 0.25 σ_s`; `ŝ_c = Δc(w, f)/Δc(pl_T, f)`. **Oriented trace** `q_s = sign(ŝ_{R0}) ŝ_s`.
- **Component split**: `ΔE_∥ = (ΔE · d̂_E) d̂_E`, `ΔE_⊥ = ΔE − ΔE_∥`; patches `E(ref_T) + ΔE_∥` and `E(ref_T) + ΔE_⊥` (float32 slices via `cd.e_slice`-style reshaping); `r_∥, r_⊥, r_full, g`.
- **Cross-context**: prompt `Prompt(frame, token_id, word)` for `w` as cue; patch `E(pl_T)` there (`x_in`), and `E(w)` in the `pl_T` prompt (`x_out`); each relative to its own clean run; `Δc_beh(w, f)` recorded.
- **Direct-effect deltas**: `pm.direct_effects` on the patched and the clean reference run; `ΔDE_k` = mean over nouns of the difference; cancellation index `C` with the floored denominator; the two largest opposing terms.
- **Anomaly score** `a(w, f) = sign(p) (m − p)`, `sign(0) = +1`; token mean over 18 frames with mean `|p|` and in-sample flag.
- **Rules 1–6 and the summary** exactly as the spec's "Frozen classification rules"; token-level values are 18-frame means (and per-template means); collapse stage aggregated by mode over frames (ties → earliest), with the agreeing fraction.

---

### Task 1: Plan, inherited extract, skeleton

- [ ] Commit this plan. Write the 007 extract from `outputs/experiment-007/results.json` (`confirmation.tokens.per_token[word].frames[frame_id].epatch_measured` → `word|frame_id`), with source digests; tests: 144 keys covering 24 tokens × 6 confirmation frames, digest, model.
- [ ] Module constants; `.gitignore`; `Pool008` (frames with origin, tokens with category, nouns, reference ids, plural cue per template) built from `cd.exposed_pool` and `cd.load_confirmation`.
- [ ] Commit `feat: add experiment 008 skeleton and inherited 007 extract`.

### Task 2: Axes, traces, component and context patches, direct-effect deltas

- [ ] Tests (fake model): stage axes have unit direction and positive scale with the plural cue positive; `Δ_{R0}` equals `ΔE` on the fake; the component split reconstructs `ΔE` and `ΔE_∥ ⊥ ΔE_⊥`; fractions are 1 for the plural cue at every stage by construction; oriented trace has `q_{R0} ≥ 0`; cross-context runs use the right prompts; direct-effect deltas sum to `Δc` within 1e-4.
- [ ] Implement `stage_axes`, `trace_epatch`, `component_patches`, `context_patches`, `direct_effect_deltas`, `signal_fractions`, `oriented_trace`.
- [ ] Commit `feat: add experiment 008 staged trace and patches`.

### Task 3: Anomaly score, classification rules, summary

- [ ] Tests on synthetic tables: sign-normalized score examples from the spec (±3 → 0 gives −3; ±3 → ±5 gives +2); strata; probe validity; rule 2 (`AXIS_ARTIFACT`); rule 3's four classes; rule 4 collapse on oriented traces (a −2 → −3 trace does not collapse; a sign flip collapses; `NO_COLLAPSE`; the `R2` transport sub-decision by `q_T`); rule 5; rule 6; summary labels for every branch; mode aggregation with ties to the earliest stage.
- [ ] Implement `anomaly_scores`, `probe_validity`, `classify_token`, `collapse_stage`, `summarize`.
- [ ] Commit `feat: add experiment 008 classification rules`.

### Task 4: Exploration orchestration, runner, report

- [ ] Tests (fake model, sandbox with the real manifest, extension, confirmation file, both extracts built from the fake): `validate`; `explore` once — ledger contains only pool prompts, both replication checks recorded, every token classified, the summary label is one of the frozen labels; refusal to re-run; `report` renders the per-token table and the summary; parser has exactly three phases and no override flag; a planted replication mismatch is recorded as an incident with its commit and blocks a rerun at the same commit.
- [ ] Implement `run_exploration`, `render_report`, `run.py`.
- [ ] Commit `feat: add experiment 008 runner`.

### Task 5: Verification, then the single run

- [ ] Full offline suite; independent review against the spec (revision 2); fix findings; commit; push.
- [ ] `explore` once (A0 contract test inside), `report`; evidence copy; README with the per-token table and the summary; root README; memory. Hand the localization statement to the Experiment 009 design.
