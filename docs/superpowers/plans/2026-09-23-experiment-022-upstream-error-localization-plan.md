# Experiment 022 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan revision 2** (2026-09-23). This revision records the reviewer's decisions on the plan's questions Q1–Q10 and
tracks design revision 3 (`b0c7382`). The earlier open-questions table is now the frozen decisions table below. Q3, Q5
and Q7 change the implementation, and Q2 is the design's off-by-one correction. One clarification remains (R-1, at the
end of the decisions).

**Goal.** Implement the approved Experiment 022 design (revision 3, commit `b0c7382`): a prospective localization of
the committed program's upstream `Δx3` error by an exact five-factor Shapley attribution, with four claims evaluated
separately on Y1 (24 new cues × 108 exposed frames) and Y2 (24 new cues × 18 new frames). The floors are calibrated
once on exposed data, and the eight condition results are reported individually.

**Spec.** `docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md`, revision 3,
`b0c7382`. **The spec wins over this plan.** Planning changes nothing scientific: not the factors, the compositions,
the value function, the claims, the guards, the order statistics, the gap rule, the populations, the candidate lists,
the tolerances or the reporting semantics. Every place where the design was underspecified was put to the reviewer
(Q1–Q10), and the decisions are frozen in the next section. The plan picks no scientific rule silently.

**Architecture.** There is one new module, `src/neural_decompiler/upstream_localization.py` (imported as `ul`), and one
runner, `experiments/022-upstream-error-localization/run.py`.

The module *calls* the frozen modules and never edits them:
- 020's `readout_decompilation.py` (`rd`): the downstream program, `NounSet`, the reference capture, `predicted_dx3`;
- 017's `head_pattern.py` (`hp`): `exact_chain`, `_row_and_values`, `_attention_output`, `HeadChainModel`;
- 016's `frame_channels.py` (`fch`), 015's `attention_patterns.py` (`atp`), 012's `layer_correction.py` (`lc`);
- 021's `readout_calibration.py` (`rc`): the provenance pools and the 020-closure checks.

It adds only what the design adds:
- the block-0 decomposition, the multi-position exact chain and the reduced chain with its input replaced;
- the canonical 32-coalition composition;
- the single Shapley kernel and its independent direct check;
- the exact order-statistic envelopes, the four-way condition classification and the gap rule;
- the freeze, the calibration, the lock, the confirmation scoring and the report.

## Review decisions on the plan's questions (2026-09-23; frozen for implementation)

- **Q1: freeze before calibrate.** The order follows the spec: `validate → freeze → calibrate → lock → confirm →
  report → replicate-021`.
  - `freeze` is tokenizer- and structure-only and uses no model output.
  - `calibrate` may depend only on the frozen file's counts, classes and templates, never on a fresh outcome.
  - `lock` binds the committed freeze artifact and its digest.
- **Q2: the undefined-draw stop is at ≥ 250 (C1, C2, C4) and ≥ 125 (C3).** This is design revision 3. It is a pre-lock
  calibration stop for review: not scientific evidence, not an incident, and never retried automatically.
- **Q3: I3 is a per-position L2 relative error, matching the spike.** At each applicable changed position `p` (both
  `p_c` and `p_t` in coordinated frames; `p_c` in cue-final frames), the frozen formula is:

  `e_p = ‖Δx̂3(p) − Δx3(p)‖₂ / max(‖Δx3(p)‖₂, 1e-12)`.

  The gate uses `max_p e_p` per pair, and its maximum over all pairs must be `≤ 1e-4`. No other norm may be chosen
  later.
- **Q4: R1 is checked against 021's table at 1e-9 before `calibrate`.** A pinned-model pre-calibration test compares
  re-measured exposed `Δc` with 021's digest-bound exposed table at `1e-9`. On failure it stops for review. The
  tolerance is never loosened.
- **Q5: the exclusion set is extracted deterministically from structured data** (details in *Freeze*):
  - the cue token ids (`tokens[*]["token_id"]`) and frame texts (`frames[*].text_template`) of every committed
    confirmation file 006, 009, 011–020, each loaded by its own frozen loader, exactly as 021's `_base_inputs` chain
    loads them;
  - the 020 provenance pool's tokens, frames, reference ids and plural cues;
  - the screening manifest and extension through `pm.load_inputs`.

  No text search is used. The source list (paths and file sha256) and the extracted sets' sha256 are bound into the
  confirmation file. Over-exclusion is allowed; under-exclusion is not.
- **Q6: draw order is frozen as 021's.** Cues are ordered by token id and frames by `frame_id` within each stratum.
- **Q7: the frozen table is a committed artifact.** The canonical locked Y1 composition table is committed as a
  digest-bound companion to the lock (about 35 MB, binary; format under *Artifact schemas*). `confirm` reconstructs it
  from the weights and the locked inputs before any fresh prompt, and requires **exact (bitwise) equality** with the
  committed artifact (I7).
- **Q8: direction checks use the median of the defined draws only.** Undefined counts are hard-gated separately (Q2).
- **Q9: the report shows a `CDF percentile`.** It is the fraction of *defined* calibration draws `≤` the fresh value,
  with the undefined count shown. It is labeled `CDF percentile`, and for the upper-bound C2 the report states that a
  high percentile lies toward the unfavorable upper tail. It is descriptive only; the frozen envelope decides the
  result.
- **Q10: `replicate-021` checks only what 021 stored.** It checks the gates that can be rebuilt without new prompts:
  - I4, the full composition against 021's stored ceiling `Δĉ`;
  - I5, the empty composition against 021's stored Level-0 `Δĉ`;
  - I6, efficiency.

  I1–I3 are reported as unavailable, with no substitute. The record stays post-report and exploratory and cannot affect
  any 022 result.

**Remaining clarification, R-1 (it blocks nothing before Task 6; it needs the reviewer's confirmation).** Q7 speaks of
the canonical locked "Y1/Y2" table, but under the design the **Y2** compositions cannot exist before `confirm`: they
need each new frame's reference state, which only the stage-1 S1-REF capture provides. The plan therefore:
- commits the **Y1** table with the lock;
- writes the **Y2** table at stage 1 in the same binary format, with its digest in the stage-1 record, re-read and
  re-verified at the barrier before any S2-TARGET prompt;
- commits it (about 6 MB) as a closure evidence artifact after `confirm`, beside the committed digest record.

No Y2 composition is computed or committed before stage 1.

## The five-factor game, made explicit (the reviewer's point)

**Players.** `F = {R, emb, Bv, Bp, T}`, encoded as bits `R = 1`, `emb = 2`, `Bv = 4`, `Bp = 8`, `T = 16`. A
coalition is a mask `m ∈ {0, …, 31}`; `|m|` is its popcount.

**Pair composition.** For a pair `p` (cue `t`, frame `f`, reference `r`, positions `p_c`, `p_t`):
- `u_pc(m) = ΔE + [m∋emb]·Δemb + [m∋Bv]·V(p_c) + [m∋Bp]·P(p_c)`.
- `u_pt(m) = [m∋T]·ΔA0(p_t)` if `p_t ≠ p_c`. In cue-final frames there is **no `p_t` input at all**; see below.
- `Δx̂3(m)` = the exact layers-1–2 program on `{p_c: u_pc(m)} (∪ {p_t: u_pt(m)})` if `m ∋ R`; otherwise the reduced
  program with the same inputs.
- `Δĉ(m)` = the frozen downstream readout of `Δx̂3(m)`, over the 79 scorable exposed nouns.

**Cue-final frames: `T` as a null player, represented without an inconsistent game.**
- In a cue-final frame `p_t = p_c`, so the target position is the cue position. Block 0's attention change there is
  already `ΔA0(p_c) = V(p_c) + P(p_c)`, split into `Bv` and `Bp`. No separate cue→target pathway exists.
- The frozen definition is `u_pt(m) := ∅` (no input at a second position) for every `m`. Therefore
  `Δx̂3(m ∪ T) = Δx̂3(m)` for every `m`.
- The implementation makes this **bitwise**: it canonicalizes `m ↦ m & ~T` before computing a cue-final composition.
  Each of the 16 distinct compositions is computed once, and `sse[p, m | T]` is the *same stored number* as
  `sse[p, m]`.

Consequences, all exact:
1. **The game is complete.** For a group of cue-final pairs, `v(m ∪ T) = v(m)` for all 32 masks. It is a well-defined
   five-player game with every coalition present.
2. **`T` is a null player.** Every marginal contribution of `T` is exactly `0`, so `φ_T = 0.0` exactly, and
   `σ_T = 0`.
3. **The other players equal the four-player game.** For `f ≠ T`, `φ_f` equals the Shapley value of the four-player
   game on `{R, emb, Bv, Bp}`. Proof: pair each `S ∌ T` with `S ∪ {T}`. Their marginal contributions for `f` are equal,
   and `w₅(k) + w₅(k+1) = k!(4−k)!/5! + (k+1)!(3−k)!/5! = k!(3−k)!·5/5! = k!(3−k)!/4! = w₄(k)`.

   So using five players everywhere is consistent: cue-final shares sum to 1 over `{R, emb, Bv, Bp}`, and the claims
   C1 and C2 read only those.
4. **Groups are never mixed in one game.** Cue-final claims (C1, C2) use only cue-final pairs; coordinated claims (C3,
   C4) use only coordinated pairs.
   - A pooled descriptive game over both groups would still be valid, because `v` is additive in SSE across pairs and
     `T` contributes exactly nothing on cue-final pairs. It is reported only descriptively, if at all.
   - A coordinated frame always has `p_t = p_c + 1`, which the freeze and the reference capture assert.

**Value, gap, Shapley, shares** (per group `g` and multiset of pairs with multiplicities `μ`):
- `SSE_g(m) = Σ_p μ_p · sse[p, m]`, with `sse[p, m] = Σ_n (Δc[p,n] − Δĉ[p,m,n])²`.
- `SST_g` is the pooled two-pass sum of squares: `Σ_p μ_p·M2_p + Σ_p μ_p·N·(mean_p − mean_g)²`, with `N = 79`. This is
  021's pooling; it is never the one-pass `Σy² − (Σy)²/n`.
- `v_g(m) = 1 − SSE_g(m)/SST_g`, and `G_g = v_g(31) − v_g(0)`.
- `φ_f = Σ_{m ∌ f} w(|m|)·(v_g(m | f) − v_g(m))`, with `w = (1/5, 1/20, 1/30, 1/20, 1/5)` for `|m| = 0…4`. The
  weights are held as exact integer ratios and applied in a fixed coalition order.
- `σ_f = φ_f / G_g`.
- The claim statistics: `s1 = σ_Bv + σ_Bp` (cue-final), `s2 = σ_R` (cue-final), `s3 = σ_R` (coordinated),
  `s4 = σ_T` (coordinated).
- **Interpretable** iff `SST_g > 0` and `G_g ≥ 0.02`. A non-finite share where interpretable is an incident.

**One canonical implementation.** A single function, `ul.group_statistics(sse_g: Tensor[D, 32], sst_g: Tensor[D])`,
returns `v`, `G`, `φ[D, 5]`, `σ` and the interpretable flag. It is vectorized over `D`.
- The calibration calls it with `D = 10,000` draws.
- `confirm` calls the *same function* with `D = 1` (the fresh population, every pair once).
- `replicate-021` calls it too.

The group sums come from one function, `ul.group_sums(sse_pairs, moments_pairs, index, multiplicity)`, also shared.

**Independent direct check (implementation-only).** `ul.direct_group_statistics` materializes the pairs' `Δc` and
`Δĉ(m)`, computes each `v(m)` by a flattened two-pass `R²` (`rd._r2`), and computes `φ` by the **permutation** formula:
the average of marginal contributions over all 120 orders. It shares no code with the kernel's subset formula. It is
compared with the kernel on the first 16 draws of each population and on the fresh populations, at
`|k − d| / max(1, |d|) ≤ 1e-10`.

## Global constraints (frozen by revision 2)

- **Frozen modules, checked by git blob.** Every phase and a tier-A test compute `sha1(b"blob <len>\0" + bytes)`, with
  no git call:

  | module | blob |
  |---|---|
  | `readout_decompilation.py` | `caa73b40192f4c910dc63371bd19db75a3258339` |
  | `readout_calibration.py` | `9107da975128d9b0383f3b346695675104e4cde9` |
  | `head_pattern.py` | `386682fe0a47d093dfbee2507afc1df61811ab15` |
  | `frame_channels.py` | `c95d6fb4c98b8069c26ec46e9d8f85eb927bb9e8` |
  | `attention_patterns.py` | `3f5acd65397bd11723ba6fc2ec4253f1d5c81943` |
  | `layer_correction.py` | `04df0cc9a20093cc48ee5ef62da7f206bf1a3186` |
  | `plural_mechanism.py` | `d39da8a8d9931005d411258bcddbb7f9beed35e4` |
  | `encoding_read.py` | `cab99c942de970332e726a5626b584242d4cf600` |
  | `head_transport.py` | `936093a4e82c83e299a65a7c85529a7226f80c95` |
  | `models.py` | `b1c6f03379af7e0918d0d1a6460a264651603fb2` |

  The blobs are recorded in the calibration record and the lock.
- **Inherited inputs**, asserted as in 021:
  - the 011/012/017 lock digests;
  - 020's closure, extract, results file and state, and its confirmation set `e098e2b4…`;
  - 021's committed calibration record (file `240e0345…`, content `f939a84d…`) and its exposed-table digest
    (`table_sha256.measured = 799ec908…`), verified before the local table is read;
  - for `replicate-021` only, 021's results state `8caf2a02…` and confirmation extract `678714b0…`.
- **Runtime:** CPU, float32, 4 threads, the pinned checkpoint and the library versions, equal to 020's explore record
  at every phase that loads the model (021's `_check_runtime`).
- **No fresh noun.** `NounSet.build(weights, pool.nouns, [])` everywhere; there are 79 scorable exposed nouns.
- **021's spent set never enters** calibration or confirmation. The calibration reads only 021's calibration record
  and exposed table, and no 021 stage-1 or stage-2 artifact. A test runs `calibrate` with those artifacts absent.

## Constants (copied from revision 2; a tier-A test pins every one)

- `DESIGN = {"path": …, "revision": 2, "commit": "0f1009c"}`; `B = 10_000`; `CROSS_CHECK_DRAWS = 16`.
- Order statistics: `LOWER_RANK = 250` (element `[249]`); `UPPER_RANK = 9751` (element `[9750]`); `C3_LOW_RANK = 125`
  (element `[124]`); `C3_HIGH_RANK = 9876` (element `[9875]`).
- Guards: C1 `≥ 0.50`; C2 `≤ 0.10`; C3 `0.10 ≤ s ≤ 0.90`; C4 `> 0`.
- Gap rule: `GAP_MIN = 0.02`, and `SST > 0`.
- Tolerances:

  | gate | tolerance |
  |---|---|
  | I1 | 1e-4 absolute |
  | I2 | 1e-12 |
  | I3 | `max_p ‖Δx̂3(p) − Δx3(p)‖₂ / max(‖Δx3(p)‖₂, 1e-12) ≤ 1e-4`, over the pair's applicable changed positions and then over all pairs *(Q3)* |
  | I4 | 1e-3 nats |
  | I5 | 0.0 |
  | I6 | `1e-12·max(1, \|G\|)` |
  | R1 | 1e-9 |
  | kernel/direct | 1e-10 on `\|k−d\|/max(1, \|d\|)` |

- Draw index: `int.from_bytes(sha256(f"022|primary|{b}|{s}|{i}".encode("utf-8")).digest()[:8], "big") % n_s`, with
  strata `cue/<class>` (slots 0–5) and `frame/<template>` (slots 0–5). Within a stratum, cues are ordered by token id
  and frames by `frame_id` *(Q6, frozen)*.
- Calibration stop *(Q2, design revision 3)*: at **≥ 250** undefined values of C1, C2 or C4, or **≥ 125** of C3. It is
  a stop for review, not an incident, and never automatically retried.
- Report: `CDF percentile = #{defined draws ≤ fresh value} / #{defined draws}`, with the undefined count *(Q9)*.
- Pools: `rc.production_pools(pool)` — cues 45/45/36/49, frames unscreened 14/14/14. The Y1-like frames are the 108
  exposed frames.
- Candidate lists (cues and frames) and the structural rules, verbatim from the spec. Quotas: 6 per class, 6 per
  template. `p_c` ranges: cue-final 3–5, coordinated 4–7, with `p_t = p_c + 1` for coordinated frames.
- Result names: `PASS`, `ENVELOPE_ONLY_FAILURE`, `GUARD_FAILURE`, `NOT_INTERPRETABLE`, evaluated in precedence
  `NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS`.

## Objects and naming (`ul`)

**Block 0 and the factors:**
- `block0_program(model)` returns `atp.LayerProgram.from_model(model, 0)`.
- `reference_embeddings(weights, frame, reference_id)` returns the float64 rows `W_E[prompt_ids]`.
- `Block0Terms` holds `total`, `value`, `pattern`, the per-head value norms, the reference self-weights, and the
  attention from `p_t` to `p_c`. It is built by `block0_terms(program0, x0_all, p_c, q, d_emb)` with the query
  recomputed only when `q = p_c`, using `hp._row_and_values` and `hp._attention_output`.
- `Factors` holds `delta_e` (`chain.fcm.read.encoding_delta`), `d_emb`, `value_pc`, `pattern_pc`, and `attn_pt`
  (`None` in cue-final frames). It is built by `factors(...)`.

**Layers 1–2:**
- `exact_chain_multi(programs, lw, x1_all, x2_all, x3_all, p_c, p_t, dx1)` is 017's `exact_chain` loop over a dict of
  layer-1 inputs. A test requires it to be bitwise equal to `hp.exact_chain(..., delta_e)["dx3"]` when
  `dx1 = {p_c: delta_e}`.
- `reduced_chain(chain, rows16, x1_all, x2_all, p_c, p_t, template, u_pc, u_pt=None)` is
  `HeadChainModel.upstream(LEVEL0)`, with `fcm.predict_channels`' `LEVEL0F` body inlined, `delta_e := u_pc`, and
  `u_pt` entering layer 1 at `p_t` exactly. A test requires it to be bitwise equal to `rd.predicted_dx3` when
  `u_pc = ΔE`, `u_pt = None` (I5).

**Compositions:**
- `canonical_mask(m, p_t, p_c)` returns `m & ~T` when `p_t == p_c`.
- `compose(ctx, m)` returns `Δx̂3(m)`, and `dc_hat(ctx, m)` returns
  `program.contrast(state, program.blocks_3_to_5(state, Δx̂3)["dh6"], nouns)[scorable]`.
- `PairContext` holds the frame state, `rows16`, `x0_all`, the factors, `nouns`, `program`, `chain`, `lw` and
  `programs`.
- `pair_table(ctx, dc_measured)` returns `sse[32]`, the per-pair moments, `Δĉ(0)`, `Δĉ(31)` and the gate values.

**Shapley and envelopes:**
- The Shapley core: `group_sums`, `group_statistics`, `direct_group_statistics`, `shares`, `claim_statistics`.
- Envelopes:
  - `envelope(values, rank)` builds a lower view with undefined values at `−∞` and an upper view with undefined values
    at `+∞`, and takes ascending element `rank − 1` of the view that rank's direction uses;
  - `condition_envelopes(values)` covers C1 and C4 lower, C2 upper, and C3 low and high;
  - `direction_checks` (against the median of the defined draws, *Q8*), and `undefined_counts` with the stop at ≥ 250
    or ≥ 125 *(Q2)*.
- `i3_error(pred, measured, positions)` is the frozen Q3 formula; `cdf_percentile(values, fresh)` is the Q9 formula.
- `classify(claim, value, interpretable, envelope)` returns one of the four results in precedence order. It is the only
  function that decides a result, both in the calibration's result rates and at `confirm`.

**Exclusion and table files:**
- `exclusion_sources(root)` lists the fixed sources *(Q5)*.
- `extract_exclusion(sources)` returns the sorted cue ids and frame texts, each with its sha256. It uses only the frozen
  loaders' structured fields: `tokens[*]["token_id"]`, `frames[*].text_template`, the pool's tokens and frames,
  reference ids and plural cues, and `pm.load_inputs`.
- `write_table_file`, `read_table_file` and `table_file_digest` handle the committed binary table format *(Q7)*.

**Phase bodies:**
- `freeze_confirmation(tokenizer, pool, sources)`;
- `calibration_rematerialize(...)`, `calibration_draws(...)`, `calibration_record(...)`;
- `build_lock(...)`, `validate_lock(...)`;
- `stage_one_022(...)`, `stage_two_022(...)`, `score_022(...)`;
- `render_report(...)`, `replicate_021(...)`.

## Artifact schemas

All JSON artifacts are canonical JSON (`pm.canonical_json`) plus a newline, with `content_sha256` computed over the
record without that key (021's convention).

1. **`experiments/022-upstream-error-localization/confirmation-v1.json`** — committed after `freeze`.
   - `experiment`, `schema_version`, `kind`, `design`.
   - `candidates` (verbatim lists) and `rules` (quotas, `p_c` ranges, the coordinated suffix rule).
   - `exclusion` *(Q5)*: the sorted cue token ids and frame texts, and each set's sha256. `sources` gives, per source,
     the loader (e.g. `cd.load_confirmation`, `ht.load_confirmation`, …, `rd.load_confirmation`, `rd.build_pool_020`,
     `pm.load_inputs`), the path and the file sha256.
   - `tokenizer` (model id and revision).
   - `cues`: `word`, `token_id`, `class`, `candidate_rank`, for 24 cues.
   - `frames`: `frame_id`, `template_id`, `text_template`, `prefix_ids`, `suffix_ids`, `p_c`, `p_t`, `candidate_rank`,
     for 18 frames.
   - `rejected`: each rejected candidate with its reason.
   - `counts`: `{"classes": {…6…}, "templates": {…6…}}`.
   - `manifest`: `S1-REF` 18, `S1-VALIDITY` 18, and `S2-TARGET` split into `Y1` (2,592) and `Y2` (432), each a sorted
     list of `pm.Prompt(...).key`.
   - `content_sha256`.
2. **`outputs/experiment-022/results.json`** — the state, in 021's schema style.
   - `run_id`; the input digests (every 021 digest key, plus the 022 confirmation, the 021 calibration record and
     table, the design commit and the module blobs); `versions`; `phases`.
   - `executed_prompt_keys` (a sorted set) and `executed_noun_keys`.
   - `calibration`: the gate maxima recorded before enforcement, R1, the table and draw digests, the record digest, and
     incidents.
   - `lock`; `confirmation`: the stage-1 record, the stage-2 digests, the gates and the eight results.
3. **`outputs/experiment-022/calibration-table.pt`** — per exposed pair: `cue`, `frame`, `template`,
   `sse[P, 32]`, the moments `(N, mean, M2)`, `Δc[P, 79]`, `Δĉ(0)`, `Δĉ(31)`, the ceiling `Δĉ`, and the per-pair
   gate values. Digested.
4. **`outputs/experiment-022/draw-values.pt`** — per population (Y1, Y2) and group: `G[B]`, `SST[B]`, `φ[B, 5]`,
   the interpretable flag `[B]`, and `s1`–`s4[B]`. Digested; the index digests go into the record.
5. **`calibration-v1.json`** — installed byte-identically from `outputs/experiment-022/candidate-calibration.json`,
   committed.
   - `design`, `run_id`, the commits, the input digests, the module blobs, `constants`.
   - `pools`; `rematerialization` (pair counts, the I1–I6 and R1 maxima).
   - `draws`: B, strata sizes, index digests.
   - `envelopes[population][claim]` with the raw rank values, and `direction_checks`.
   - `undefined_counts`; `guard_bound`.
   - `result_rates[population][claim]`: the shares of each result over the draws.
   - `joint_rates`: all four of a population `PASS`, and all eight `PASS` — descriptive only.
   - `summaries`: median and tails; `cross_check`; the table and draw digests; `content_sha256`.
6. **`preregistration-lock.json`** — installed byte-identically from `candidate-lock.json`, committed.
   - `design`; `protocol_code_commit`; the input digests; the module blobs.
   - The calibration record's file and content digests; the confirmation file digest.
   - The eight condition definitions: statistic, group, direction, envelope bounds, guard and result precedence.
   - `y1_table`: the file sha256 and index sha256 of the committed companion below, plus digests of the per-pair
     Level-0 `Δx̂3` and the factor vectors.
   - The reporting semantics (the three-way reading, C4's positive-contribution wording, and "no aggregate label").
   - `content_sha256`.
   - A human-readable `preregistration.md` renders the eight conditions with their envelopes and guards.

   **Companion artifact** *(Q7)*, committed with the lock and bound to it by digest:
   - `experiments/022-upstream-error-localization/locked-y1-table.f64`: raw IEEE-754 float64, little-endian, C order,
     about 35 MB. It holds two contiguous blocks: cue-final `[1728 pairs, 16 canonical masks (T cleared), 79 nouns]`,
     then coordinated `[864, 32, 79]`.
   - `locked-y1-table.json`: the index, giving dtype, byte order, block offsets and shapes, the pair order (cue token
     id, then `frame_id`, *Q6*), the mask order (ascending mask integers), the noun order (the 79 scorable exposed
     `lexical_key`s), and the file's sha256.
   - A deterministic byte writer and reader are used. `torch.save` is not used, because its zip container is not
     guaranteed byte-stable.
7. **Stage-1 record**, in the results state and digested: each new frame's reference state (020's `locked_state`
   format), the descriptive validity verdicts, the Y2 table file's sha256 and index, and the stage-1 digest.
   - The **Y2 table** is written at stage 1 to `outputs/experiment-022/y2-table.f64` plus its `.json` index, in the same
     format (cue-final `[288, 16, 79]`, coordinated `[144, 32, 79]`, about 6 MB).
   - It is re-read and re-verified at the barrier, and committed as closure evidence after `confirm` *(R-1)*.
8. **`outputs/experiment-022/stage2-measurements.pt`** — per target pair: `Δc[79]`, `Δx1` and `Δx3` at the changed
   positions, and the ceiling `Δĉ`. Written with digests *before* any gate is enforced or anything is scored.
9. **`outputs/experiment-022/report.md`** and **`replicate-021.json`/`.md`**. The latter is labeled exploratory and
   carries no result names.

## Phase control flow

- **`validate`** (no model). Checks:
  - the module blobs and every inherited digest;
  - 020's closure and ledger isolation;
  - 021's calibration record and exposed-table digest;
  - the pools (`rc.production_pools`);
  - the candidate lists against the constants;
  - that no `outputs/experiment-022/results.json` contradicts the committed artifacts.

  It writes nothing.
- **`freeze`** (tokenizer only; no model forward) *(Q1)*.
  - Build the exclusion sets by the structured extraction `extract_exclusion(exclusion_sources(root))` *(Q5)*. The
    sources and digests are recorded; a source that fails its own frozen digest check is an incident.
  - Walk each ordered cue list: take the first 6 single-token, leading-space, non-excluded entries per class.
  - Walk each ordered frame list: build each frame with the project's own `pm.Frame` construction (the same builder
    020's confirmation used). Take the first 6 per template that pass the checks: the text is new, the cue slot is one
    token, `p_c` is in the template's range, and for coordinated frames `p_t = p_c + 1`.
  - If any class or template falls short, write nothing and stop for review.
  - Otherwise write the confirmation file, and commit it by hand.
- **`calibrate`** (once; exposed only).
  1. Verify the inputs, the runtime, the clean tree and the commit. Run the A0 contract test.
  2. Record the attempt.
  3. Load the model.
  4. For each of the 108 frames, in `frame_id` order: rebuild the state from 020's locked state
     (`rd.state_from_locked`) and compute `rows16` and `x0_all`.
  5. For each of the frame's 175 pool cues: assert the key is in 020's ledger and outside the 022 manifest and 020's
     manifest; run **one forward** (`pm.capture_prompt`: logits plus `RESID_PRE.L1` and `RESID_PRE.L3` at `p_c`, `p_t`);
     record the execution; compute the factors, the canonical compositions (16 or 32), `sse`, the moments, the ceiling
     `Δĉ`, and the gate values I1–I5.
  6. After all pairs, write the table and the gate maxima to disk.
  7. Enforce I1–I5 and **R1** (`Δc` against 021's table, after verifying the table's digest).
  8. Build the draws, then the kernel statistics for Y1-like and Y2-like, then the direct cross-check.
  9. Compute the undefined counts and apply the stop at **≥ 250** (C1, C2, C4) or **≥ 125** (C3) *(Q2)*. Then compute
     the envelopes, then the direction checks against the median of the defined draws *(Q8)*; a violation is an
     incident.
  10. Compute the result rates, the joint rates and the summaries.
  11. Write the candidate record; the after-phase 020/021 re-check runs before the record is written.
  12. Stop for the floor review. The record is installed byte-identically and committed.
- **`lock`** (no forward pass; `pytest_free_guard`, as in 021).
  1. Verify the installed record (digest, constants, rows, inputs); no scientific path may have changed since the
     calibrate commit.
  2. Load the weights only.
  3. For the 24 frozen cues × 108 exposed frames: build the factors and the canonical compositions. Write the Y1 table
     to the byte format (`candidate-locked-y1-table.f64` plus its `.json` index).
  4. Recompute them a second time under the guard: the bytes must be identical (provenance).
  5. Write `candidate-lock.json` and `preregistration.md`. The lock, the markdown and the **table companion** are
     installed byte-identically and committed together *(Q7)*, then reviewed independently and signed off.
- **`confirm`** (once; no resume).
  1. **Pre-checks**, before any fresh prompt:
     - validate the lock and the record, including the committed table companion's file and index digests;
     - the ledger isolation holds, and the runtime is unchanged;
     - reconstruct the Y1 table from the weights and the locked inputs, and require **I7**: the reconstructed bytes
       equal the committed companion exactly *(Q7)*.
  2. **Stage 1:**
     - For each new frame, capture S1-REF with `rd.capture_frame_020`, and run S1-VALIDITY (recorded descriptively; it
       selects nothing). A structural failure of a capture is an incident.
     - Compute the Y2 table from the captured states and write it in the byte format *(R-1)*.
     - Write the stage-1 record and its digest.
     - Re-read both at the **barrier**, and re-verify the table bytes and the stage-1 digest there. The barrier also
       checks that the ledger holds no S2-TARGET key.
  3. **Stage 2:** run all 3,024 S2-TARGET prompts once each (the Y1 block, then the Y2 block), each capturing logits,
     `RESID_PRE.L1` and `RESID_PRE.L3`. Write `stage2-measurements.pt` and its digests.
  4. **Gates:** enforce I1–I4 and I6, with I3 by the frozen per-position formula *(Q3)*. I5 holds by construction, and
     its tables were computed and checked at lock and in stage 1.
  5. **Scoring:** build the group sums, then `group_statistics` (`D = 1`), then the direct check, then `s1`–`s4` for
     Y1 and Y2, then `classify` for each of the eight conditions.
  6. Write the results. An incident is recorded, then raised; the phase stays `running` and can never resume.
- **`report`**. Headline: the 2 × 4 table. For each condition it shows the fresh statistic, the envelope (bound or
  range, with its rank), the guard, `G`, the **CDF percentile** with the undefined count *(Q9)*, and the result. For
  C2, the report states that a high CDF percentile lies toward the unfavorable upper tail; the percentile is
  descriptive and the envelope decides. Beside the table:
  - the Shapley values and shares per population and group, and the efficiency residual (I6);
  - C3's direction;
  - C4's wording, fixed to "contributes positively";
  - the descriptive records: the ladder, the R-only and inputs-only rungs, per-template and per-class shares, the
    `Δx3` errors, block 0's head profile, and the validity verdicts.

  There is no aggregate label.
- **`replicate-021`** (only after `report`). It reads 021's stage-1 states and stage-2 tables (digest-verified) and
  computes the 022 statistics with the same kernel. It checks only the gates that 021's stored data can rebuild —
  I4, I5 and I6 *(Q10)* — and records I1–I3 as unavailable, with no substitute. It writes a record labeled
  exploratory, with numbers only and no result names, that cannot affect any 022 result.
- **Closure** (after `report`, when authorized). Commit the Y2 table companion as evidence *(R-1)*, together with the
  usual README, report and extract.

## Ledger and isolation

- The ledger is 021's: a sorted set, recorded before or at execution.
- **`calibrate`:** the ledger must end equal to the 18,900 calibration keys, all in 020's exposed ledger, none in any
  manifest.
- **`confirm`:** it adds the 3,060 manifest keys, each exactly once (a spy test counts executions). At the barrier the
  ledger holds zero S2-TARGET keys. At the end it holds the 18,900 plus 3,060 keys and no others.
- **Leakage refusals, each with a test:**
  - a manifest key run by `calibrate`;
  - a 020 or 021 confirmation key anywhere before `confirm`;
  - a fresh frame text or cue among the calibration pools;
  - `calibrate` reading a 021 stage-1 or stage-2 artifact;
  - `lock` reaching a forward pass;
  - an S2-TARGET key before the barrier.

## Incidents and failures

These follow 021:
- An incident is written to the state before any slow re-check. It binds its commit, so no rerun happens at that
  commit.
- `calibrate` resumes only after a recorded incident, at a new commit. An unrecorded termination is reconciled as an
  incident at the next start.
- `lock` runs once. `confirm` runs once, with no resume: an interruption consumes the protocol version.

**Incidents:**
- I1–I7, R1 and the ledger or isolation checks;
- a missing or non-finite measurement;
- a structural capture failure;
- a direction-check violation;
- a kernel/direct excess.

**Stops for review (not incidents):**
- a freeze shortfall;
- the undefined-count stop *(Q2)*;
- R1 inexactness found in the tier-C pre-check before `calibrate` *(Q4)*.

## Testing

**Tier A** (pure, no model):
- the constants and blobs;
- the masks and canonicalization;
- **the Shapley kernel:** efficiency; symmetry; null player; the cue-final five-player values equal the four-player
  values (1e-12) with `φ_T == 0.0` exactly; permutation invariance; closed-form games (additive, pure-interaction,
  identical players); kernel against the direct permutation formula on random games;
- `group_sums` with multiplicities and two-pass pooling (a 1e4 offset test);
- **the envelope indices:** C1 and C4 from `[249]`; **C2 from `[9750]` — a test fails if C2 uses `[249]`**; C3 from
  `[124]` and `[9875]`;
- the undefined-value convention; the direction checks against the median of the defined draws *(Q8)*; the stop at
  **exactly 250** undefined values (C1, C2, C4) and **exactly 125** (C3), with 249 and 124 proceeding *(Q2)*;
- **the I3 formula** *(Q3)*: per position, the maximum over positions and pairs, the `1e-12` denominator floor, and a
  cue-final pair checked at `p_c` only;
- **the CDF percentile** *(Q9)*: defined draws only, ties counted as `≤`, the undefined count reported;
- **the table byte format** *(Q7)*: a write/read round trip is bitwise; the digest is stable across two writes; a
  single flipped bit fails validation;
- **`classify`** at every boundary: exactly `0.50`, `0.10`, `0.90`, `0`, and exactly on each envelope bound (the
  inclusive `≥` and `≤`, the strict `> 0`), plus the precedence order;
- the gap rule: exactly `0.02` counts as interpretable, `SST = 0` does not, and a mask of units is never altered;
- golden values of the draw index for fixed `(b, s, i)`, and draw reproducibility;
- the freeze on a stub tokenizer: quotas, a shortfall stop, exclusion, and the structural rules;
- **the structured exclusion** *(Q5)*, on stub loaders:
  - every source's cue ids and frame texts are collected;
  - a planted duplicate is excluded;
  - a missing source is refused;
  - the recorded source list and digests match;
  - no text search is used (the extractor takes loader objects only).

**Tier B** (runner on a fake-closed world, 021's pattern, with small B and patched constants):
- every phase and refusal; the ledger and leakage plants;
- the barrier;
- incidents at each stage, with measurements on disk before gates;
- the four results reachable end to end;
- `NOT_INTERPRETABLE` without unit removal;
- lock tampering (floor, table digest, guard, semantics, design commit);
- a committed table companion whose bytes differ from the reconstruction, refused by I7 before any fresh prompt;
- a tampered Y2 table caught at the barrier;
- `replicate-021` refused before `report`, and reporting I1–I3 as unavailable.

**Tier C** (the pinned model, `pythia_smoke`-style opt-in), on a handful of exposed pairs per template:
- the bitwise equalities (`exact_chain_multi` against `hp.exact_chain`; `reduced_chain` against `rd.predicted_dx3`);
- I1–I4 within tolerance, with I3 by the frozen formula;
- the **R1 pre-calibration test** at `1e-9` against 021's digest-bound exposed table, on a sample of exposed pairs
  across all three templates *(Q4)*. It must pass on the clean gated commit before `calibrate` is authorized. On
  failure, stop for review; the tolerance is never loosened.

`CURRENT_EXPERIMENT` becomes `"022"` when the runner test exists.

## Runtime and storage (estimates on this machine)

| phase | work | time | storage |
|---|---|---|---|
| `freeze` | tokenizer only | seconds | about 40 KB committed |
| `calibrate` | 18,900 forwards (about 20 min); about 403,000 coalition compositions (72 cue-final frames × 175 × 16 + 36 coordinated × 175 × 32, at about 15–20 ms each, about 2 h); draws, kernel and cross-check in minutes | about 2.5 h | table about 55 MB; draws about 3 MB; record about 1–2 MB committed |
| `lock` | 2 × about 55,000 compositions (the second is the provenance recompute) | about 35 min | **Y1 table companion about 35 MB, committed** *(Q7)*; lock under 1 MB committed |
| `confirm` | I7 reconstruction (about 17 min); 36 stage-1 prompts; about 9,200 Y2 compositions (3 min); 3,024 forwards (about 4 min); scoring | about 25–30 min | Y2 table about 6 MB (committed at closure, *R-1*); stage-2 measurements about 55 MB (local); results state about 10 MB |

Peak memory stays under about 2 GB. The machine should be on AC power with sleep blocked, as for 021.

## File map

- **Add** `src/neural_decompiler/upstream_localization.py`.
- **Add** `experiments/022-upstream-error-localization/run.py` and its `README.md`.
- **Add** `tests/test_upstream_localization.py` (tier A, plus marked tier-C cases) and
  `tests/test_experiment_022_runner.py` (tier B).
- **Change** `tests/test_tiers.py` (`CURRENT_EXPERIMENT = "022"`) and `.gitignore` (`outputs/experiment-022/*`).
- **Created later by the phases**, each committed at its step:
  - `confirmation-v1.json`;
  - `calibration-v1.json`;
  - `preregistration-lock.json`, `preregistration.md` and the Y1 table companion `locked-y1-table.f64` with its
    `.json` index *(Q7)*;
  - at closure, the Y2 table companion under `evidence/` *(R-1)*.
- No frozen module is edited.

## Tasks (small reviewable commits; the checkboxes are for execution)

**Task 1: skeleton and constants**
- [ ] `.gitignore`; the module with constants, blob checks and input digests; tier-A tests pinning them.
- [ ] Commit: "feat: experiment 022 module skeleton, constants and frozen-input checks".

**Task 2: pure core**
- [ ] Masks and canonicalization; `group_sums`, `group_statistics` and `direct_group_statistics`.
- [ ] The envelopes, direction checks, undefined convention, `classify` and the gap rule.
- [ ] `i3_error`, `cdf_percentile` and the table byte format with its writer and reader.
- [ ] Every tier-A test above.
- [ ] Commit: "feat: experiment 022 Shapley kernel, envelopes and condition classification (pure core)".

**Task 3: model-side programs**
- [ ] `block0_terms`, `factors`, `exact_chain_multi`, `reduced_chain`, `compose` and `pair_table`.
- [ ] The tier-C bitwise and gate tests on exposed pairs, including the R1 pre-check *(Q3, Q4)*.
- [ ] Commit: "feat: experiment 022 block-0 decomposition and the 32-coalition composition".

**Task 4: freeze**
- [ ] The candidate lists, the structured exclusion extraction through the frozen loaders *(Q5)*, the structural
      checks, the manifest and the file writer.
- [ ] Tests on a stub tokenizer and the real one.
- [ ] Commit: "feat: experiment 022 tokenizer-only freeze of the new cues and frames".

**Task 5: calibrate**
- [ ] The re-materialization loop, R1, gates, table, draws *(Q6)*, kernel statistics, cross-check, undefined stop
      *(Q2)*, envelopes, direction checks *(Q8)* and the record.
- [ ] A synthetic full-scale dry run with no model: B = 10,000 at the real pool sizes, checking time and memory.
- [ ] Commit: "feat: experiment 022 calibration (exposed only)".

**Task 6: lock, confirm, report and the runner**
- [ ] `build_lock` and `validate_lock` *(Q7)*; stage 1, the barrier and stage 2; `score_022`; `render_report`
      *(Q9)*; `replicate_021` *(Q10)*.
- [ ] The runner, fake-world runner tests, `CURRENT_EXPERIMENT` and the README.
- [ ] Commit: "feat: experiment 022 lock, confirmation, report and runner".

**Task 7: review**
- [ ] An independent implementation and data-flow review against the spec; fixes in separate commits.
- [ ] Tier B, then tier C on the clean gated commit. **Stop.**

**Then the scientific sequence, each step only when authorized:**
1. `freeze`, then commit.
2. `calibrate` once, then install and commit the record; **stop for the floor review**.
3. `lock`, then commit; the independent lock review and the sign-off.
4. `confirm` once.
5. `report`, then closure.
6. `replicate-021`.

## Stopping conditions

- Q1–Q10 are decided, and Q2 is in design revision 3. Implementation (Tasks 1–7) may begin once the reviewer accepts
  this plan revision. R-1, the Y2 table committed at closure, is needed only by Task 6.
- During implementation, stop at the end of Task 7.
- Stop before each scientific step.
- Stop on any incident, and on any stop-for-review condition.
