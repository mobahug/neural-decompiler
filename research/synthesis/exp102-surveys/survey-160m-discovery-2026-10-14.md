# Experiment 102 survey: moving the discovery method to Pythia-160M-deduped

This is a read-only survey, made on 2026-10-14 on branch `c004-exp102-design` at `a9b6b33`. No model was loaded and no
forward pass, runner or test was run. I read `config.json`, the tokenizer-file hashes and the JSON headers of the
safetensors files (metadata only) in the HF cache. Paths are relative to the repository root.

## 1. Facts about the 160M model

| | 70M (`e93a9faa`) | 160M (`582159a2`) |
|---|---|---|
| layers × heads | 6 × 8 | **12 × 12** |
| d_model / d_head / d_mlp | 512 / 64 / 2048 | **768** / 64 / **3072** |
| rotary | pct 0.25, so 16 dims (inv_freq has 8 entries); base 10000 | same |
| parallel residual, LN eps, activation | true, 1e-5, gelu | same |
| vocab, embeddings | 50304, untied | same |
| parameters (non-embedding) | 70,426,624 (18.9M) | 162,322,944 (85.1M) |
| head + MLP components | 54 | **156** |
| stored weights | F16 plus U8 mask buffers, 166,029,852 B | F16, 374,998,696 B |

- **Config.** The configs differ only in `hidden_size`, `intermediate_size`, `num_attention_heads` and
  `num_hidden_layers`.
- **Snapshot files.** `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and
  `special_tokens_map.json`. The `.no_exist` markers (vocab/merges/generation_config) are the same as for 70M.
- **No `refs/main`.** The 160M cache has no `refs/main`, so loading must pass the commit SHA. `models.py:53–56` already
  does this.
- **The tokenizer is byte-identical.**
  - The SHA-256 of `tokenizer.json` is `c24618a1…f25624`, and it is the same git blob `f74dfbfa…` in both caches.
  - `tokenizer_config.json` is `70e38394…` and `special_tokens_map.json` is `6f50ab5a…` in both.
  - The manifest freeze already asserted that the two pinned vocabularies and special tokens are equal
    (`candidate_screening.py:775–786`; the manifest's `vocabulary_sha256` is `9c7009d0…`).
  - So every token id carries over unchanged: `one` 581, `two` 767, `three` 1264, `many` 1142 and the 79 nouns. So do
    every p_c/p_t and every tokenizer-only eligibility rule.
- **Earlier use of 160M.** Only Experiments 003 and 004 (capital recall) ran it.
  - 003 loaded it through TransformerBridge with the same versions pinned today: transformer_lens 3.9.0, torch 2.14.0,
    transformers 5.17.0 (`outputs/experiment-003/results.json`).
  - The revision resolved and compatibility mode was off.
  - Peak memory was 1.14 GB with both models loaded (003 README:109).
  - The behaviour screen skipped 160M (`screening/behavior-candidates/README.md:29–30`). **No number-agreement prompt has
    ever run on 160M.**
- **Compute.**
  - FLOPs per token are about 89 M for 70M and 247 M for 160M (×2.8), and hook overhead doubles with the layer count.
  - The estimate is therefore about 0.15–0.25 s per instrumented forward pass for 160M, against 70M's ~0.07 s (D101:360;
    025 ran 90,720 passes in 1 h 49 min, research-state:510–514).
  - 101's heavy-capture runs took ~0.15–0.19 s on 70M (closure:146–147; calibrate ran 2,016 runs in 384 s), which
    projects to ~0.4–0.5 s on 160M.
  - Measure this in a smoke run before relying on it.

## 2. Which code is tied to 70M

**Already model-agnostic** (reusable by import):
- `models.py:53–56, 128–155`.
- `capture.py` and `interventions.py`, which hold no geometry constants.
- `components.py:121–143`: resolution is driven by `cfg`.
- `components.py:401–417`: the parallel GPT-NeoX block identity, which 160M satisfies.
- `candidate_screening.py:1260–1279`: `canonical_component_id` writes `L{:02d}.H{:02d}`, and `component_universe` reads
  `cfg`.
- `candidate_screening.py:524–550`: the gate evaluator.
- The measurement layer of `plural_mechanism.py`:
  - the 2-digit key regexes (`:685–691`);
  - `capture_prompt` and `run_patched` (`:786`, `:819`), `contrasts` (`:844`), `Weights.from_model` (`:939`);
  - `standard_sites` (`:1001`), `counterfactual_rows` (`:1068`), the neutralized and isolation runs (`:1113`,
    `:1146`);
  - `site_axis` (`:1181`) and `direct_effects` (`:1206`);
  - A2, A3 and A4 (`:1384`, `:1416`, `:1432`), which accept any `DiscoveryContext` (its manifest may be `None`).
- `head_transport.py:113` (`HeadWeights.from_model(model, layer, head)`), `:134` (`read_direction`) and `:139–147`
  (`reconstruct_head_result`, `ov_levels`).
- `read_assembly.read_weight` (`:216–217`) and `encoding_read.EncodingRead` (`:93`).
- `attention_patterns.LayerProgram.from_model` (`:168–183`), which takes the rotary dims from `cfg`.
- In `read_intervention.py`:
  - `ReadGeometry` (`:519`), `iso_direction` (`:551`), `pair_quantities`, `edit_vector` and `residual_replacement`
    (`:623–659`), `g2_values` (`:661`);
  - `binomial_tail`, `holm_thresholds` and `nearest_rank` (`:273–296`).

**Hard-coded 70M assumptions:**

| module (experiment) | assumption | lines |
|---|---|---|
| `plural_mechanism` (005) | A1 checks against 70M screen constants and `PYTHIA_70M` results | `:1272–1280`, `:1329` |
| | the model dict must equal 70M in the extension, state and lock | `:359`, `:428`, `:516`, `:3015` |
| | "54 components" docstring | `:725` |
| | A5 needs all three templates, each with ≥ 2 frames | `:1484–1486` |
| | A7 needs the cardinal and quantifier templates | `:1680` |
| | A6 needs A5's `R_out` axis | |
| `cue_suppression` (008) | `HEAD_KEY, HEAD_LAYER, HEAD_INDEX = "L03.H04", 3, 4` | `:60` |
| | stage sites `RESID_PRE.L1`, `L04.MLP`, `L05.MLP` | `:77–80` |
| | exactly 40 tokens and 18 frames; model check | `:129`, `:180` |
| `head_transport` (009) | head constants; `capture_reference` reads `RESID_PRE.L3` and attention row [4] | `:49`, `:489–516` |
| `read_assembly` (010) | `COMPONENT_ORDER` = 8 heads + MLP of layers 1–2, single-digit format | `:59–60` |
| | 63 tokens / 24 frames | `:87` |
| `encoding_read` (011) | model checks, 63×24 fractions, the lock's 512-d `axes_vectors`/`read_weight` | `:237`, `:327`, `:441–449` |
| `layer_correction` (012) | `LAYERS = (1, 2)`, the MLP keys | `:81–84`, `:297` |
| `attention_paths` (013) | `HEAD_LAYERS = (1, 2)`, `N_HEADS = 8` | `:78–80` |
| | `head_key = f"L0{layer}.H0{head}"` | `:118–119` |
| `attention_patterns` (015) | inherits `HEAD_LAYERS` | `:84–86`, `:230–231` |
| `frame_channels` (016) | inherits `HEAD_LAYERS`; `c[1] + c[2]` | `:81–83`, `:245–246` |
| `neuron_feature` (014) | `NEURON = 1987`, `NEURON_LAYER = 2` | `:57–58` |
| `head_pattern` (017) | `PROGRAM_LAYERS = (1, 2, HEAD_LAYER)` | `:87–89` |
| `block_concentration` (018) | `N_NEURONS = 2048`, `BLOCK = 2` | `:60`, `:97–99` |
| `block_routing` (019) | inherits 018's constants; `range(2048)` | `:63–64`, `:540` |
| `readout_decompilation` (020) | `READOUT_LAYERS = (3,4,5)`, `PROGRAM_LAYERS = (1..5)` | `:61–63` |
| | rows of `ATTN_PATTERN.L4`/`L5` | `:343–349` |
| | labels `f"L0{layer}.H0{index}"` | `:289–290`, `:1216–1217` |
| `upstream_localization` (022) | `ModelPrograms` is bound to the 011/012/017 locks | `:625–651` |
| `block0_completion` (023) | `HEAD_ORDER` for 8 heads | `:174` |
| `read_intervention` (101) | cues 581/767/1264/1142, 79 nouns, 36 frames | `:380–381` |
| | `w_c` from 011's lock | `:531–538` |
| | sites from `hp.HEAD_LAYER`/`HEAD_INDEX` | `:1086–1118` |
| | `rows4`/`rows5` | `:1063` |
| tests | the contract test loads 70M and asserts 6/8/512/2048 | `tests/test_pythia_bridge_contract.py:39, 43–44, 89–94` |

**Structural constraints for 102:**
1. **Do not edit pinned modules.**
   - 101 (`read_intervention.py:90–127`), 022 (`upstream_localization.py:64`), 023 (`block0_completion.py:78`), 024
     (`readout_routing.py:83`) and 025 (`cue_rotation.py:77`) pin the modules by git blob and refuse to run when one
     changes.
   - Parameterizing `HEAD_LAYER` in place would break their reproducibility guards.
   - 102 needs a **new module** that imports the agnostic functions and re-implements, with parameters:
     - `stage_axes` for an arbitrary head site;
     - `capture_reference` for an arbitrary (layer, head);
     - 101's head-arm Δ;
     - a `ReadGeometry` built from 102's own lock.
2. **A label-format defect.** `f"L0{layer}.H0{index}"` yields `L010.H011` for 160M, which `pm.component_ref` rejects
   (regex `:686`). Use `canonical_component_id` everywhere.
3. **A 160M contract test is needed:** 12/12/768/3072, the block identity and NOOP.
4. **Ledger keys carry no model.** `Prompt.key = frame_id|cue_label|token_id` (`plural_mechanism.py:200`), so a 160M
   ledger must be model-scoped explicitly.
5. **Do not port the 70M program.** None of the decoded program in 012–023 is needed. 101's P family has no 160M
   counterpart.

## 3. How the 70M answer was found

| step | input | measurement | decision rule | mechanical or human |
|---|---|---|---|---|
| Screen | manifest, 12 prompts | 10 behaviour gates; compactness probe at the last position | gates `candidate_screening.py:524–550`; top-k against random sets | mechanical. The probe already gave the k = 2 set {`L00.MLP`, **`L03.H04`**}, recovery 0.905 (screen README:60–62; 005 design:177–180), so 005 was not blind |
| 005 A2 | 6 development frames (only **2 coordinated**), 12 prompts, 20 nouns | bidirectional singleton patching of all 54 components at p_c and p_t | top-3 heads by coordinated mean d_patch at p_t become the T candidates; L_T = the top head's layer (`plural_mechanism.py:2128–2140`) | mechanical, preregistered (005 design:521–525) |
| 005 A3 | coordinated frames | `RESID_PRE.L` patching at p_c | descriptive; L0 = 1 checked (`:1425`) | L3 0.956 → L4 0.125 (DR005:31–36) |
| 005 A4 | coordinated frames | attention p_t→p_c | descriptive only | **`L03.H04` ranked 4th (0.747)**, behind `L05.H07` 0.999, `L02.H07` 0.893 and `L00.H05` 0.829 (DR005:40–45) |
| 005 A6 and tree | coordinated frames | q1 head alone, q2 blocked fraction, q3 m_R, q4 R alone, q5 top-3 | H1 if q1 ≥ 0.5 and q2 ≥ 0.5 (`:1817–1828`) | mechanical. q1 0.816, q2 0.865, next head 0.122 → H1, `L03.H04` (DR005:20, 49–53) |
| 005 outcome | — | — | M1 `NO_COMPACT_MECHANISM`; the confirm gave `BEHAVIOR_NOT_REPLICATED` (FR005:55–57, 129–132) | **human**: `L03.H04` kept as a constant from 008 on (`cue_suppression.py:60`) and never re-derived |
| 008 | 40 tokens, 18 frames, 80 nouns | E-patch traces through fixed stage sites | frozen rules (`:46–58`) → `MIXED`; the transport drop was descriptive: "the frozen rules did not anticipate this pattern" (008 README:68) | human interpretation |
| 009 | 40 exposed tokens; 24 fresh tokens × 6 frames | exact ΔT split into P1/P2/P3 | the lowest level with R² ≥ 0.90 and MAE ≤ 0.10 is locked (`head_transport.py:51–52`) → P1 (R² 0.909; fresh Spearman 0.957; 009 README:62–64, 102–106) | the P1–P3 forms and `m = W_V W_O d̂_T` were designed by humans (009 design:35–56); the level choice was mechanical |
| 010 | 63 tokens, 24 frames | exact attribution of ρ_f over E and layers 1–2 | frozen classes → `MIXED` | `g_E` was a human proposal |
| 011 | 24 exposed frames' clean pairs (`one`/`two`, `each`/`several`, cue-final frames included) | `d̂_T` = unit mean of (T_pl − T_sg) at p_t (`cue_suppression.py:261–273`; `encoding_read.py:440–449`); `read_weight = γ₃⊙W_V W_O d̂_T` (`read_assembly.py:216–217`) | Y1 Spearman ≥ 0.80 → 0.848 | mechanical once the head and the pairs are given |
| 101 | 011's lock | `w_c = read_weight − mean`, ‖w_c‖ ≈ 0.766 (D101:45–59; `read_intervention.py:531–538`) | frozen design | the site `RESID_PRE.L3@p_c` follows from the head's layer |

**Lessons:**
- The head's identity rests on **one mechanical step on two coordinated frames**, which the screen had anticipated.
- **Attention alone would have picked the wrong head**, and the head's **direct-effect share** would mislead too: its
  effect is mediated by the L04/L05 MLPs (m_R 0.987). Ranking must be causal: singleton patching at p_t.
- Given the head, γ and `d̂_T`, the read is fully determined. The only free choice was the P1 functional form, which was
  validated mechanically.

## 4. A proposed mechanical discovery algorithm (MD-1)

MD-1's code, thresholds and splits are preregistered before any 160M output is seen, and the discovery runs once.

**The data:**
- **D (discovery):** the 36 exposed coordinated frames of 020's pool, which were 101's calibration set and have never
  run on 160M.
  - They are split by frame-id hash into **D1** and **D2**, 18 each.
  - Cues: `one`/`two`.
  - Nouns: the 79 scorable ones.
- **C (confirmation):** see §5.

**Step 0: retrodiction on 70M, using spent data only.**
- Run steps a–e unchanged on 70M with D.
- It passes if all four hold: h* = `L03.H04`, L* = 3, `SINGLE_HEAD`, and cos(w_c^MD-1, w_c^011) ≥ 0.95. The last
  condition is needed because MD-1's `d̂_T` comes from coordinated `one`/`two` pairs only.
- Cost: about 6k 70M passes, under 10 min.
- **If it fails, MD-1 is not the method that found the 70M answer, and it must be revised before 160M.**

**The steps** (pass counts are for 160M; threshold justifications are the 70M values):

| step | data | statistic and rule | ties and failure | passes |
|---|---|---|---|---|
| (a) behaviour | D; the 12 manifest prompts (descriptive) | Δc_full(`one`→`two`) ≤ −1 nat in ≥ 30/36 frames (101's precondition); the 10 screen gates reported | otherwise stop: `BEHAVIOR_ABSENT_160M` | 84 |
| (b) position | D | bidirectional patching of `RESID_PRE.L` at p_c and at p_t, L = 0..11, gives R_c(L) and R_t(L); R_c(0) = 1 within 1e-4; R_c(1) is the share that bypasses transport at p_c | an identity failure is an incident | 1,728 |
| (c) layer | D | Δ_L = R_c(L) − R_c(L+1), with R_c(12) := 0; L* = argmax Δ_L. `CONCENTRATED` if Δ_L* ≥ 0.5 and R_c(L*) ≥ 0.7 (70M: 0.831, 0.956) | otherwise `DISTRIBUTED`: the transport layers are the fewest largest drops summing to ≥ 0.7 (at most 3) | 0 |
| (d) head | rank on D1, verify on D2 | for all 144 heads: s_h = singleton recovery at p_t (A2) and a_h = attention p_t→p_c (A4). `SINGLE_HEAD` if all of: s_h* ≥ 0.5 on D1 and D2, h* is also top on D2, s_(2) ≤ 0.5·s_h*, h* is in a transport layer, a_h* ≥ 0.3, and blocked fraction b ≥ 0.5 (A6 iii; 70M: 0.816 / 0.122 / 0.747 / 0.865). Value-path share and the p_t MLPs are reported | ties go by canonical order. `MULTI_HEAD` if the top-k (k ≤ 3) cumulative recovery is ≥ 0.7 and every member passes the layer and attention gates. `NO_TRANSPORT_HEAD` if the top-3 cumulative recovery is < 0.5: stop with `DISCOVERY_FAILED`. High s with a < 0.3 or outside the layers: proceed flagged `INCONSISTENT` | 10,368 (+864 MLPs, +~500 blocked/value path; the optional p_c map adds 11,232) |
| (e) read | D (`one`/`two`); D1 × 008's 40 exposed tokens for P1 | `d̂_T` = `pm.site_axis` of h*'s output at p_t; m = W_V W_O d̂_T; w = γ_L*⊙m (`HeadWeights.from_model(model, L*, h*)`, `ra.read_weight`); w_c = w − mean(w). P1 check: R² ≥ 0.80 between (A_c/σ_c)⟨Δx, w_c⟩ and ⟨ΔT, d̂_T⟩ gives `P1_SUFFICIENT`. The read share ⟨Δx, ŵ⟩²/‖Δx_c‖² is reported (70M ≈ 0.14) | otherwise `P1_NOT_ESTABLISHED`: the edit stays defined but the scope narrows. Multi-head: one covector per head; edits act on the projection onto their centred orthonormal span | 720 |
| (f) freeze | — | the lock binds the revision, every label of a–e, L*, h* (or the set T), `d̂_T` and `w_c` as float64 with digests, and G1 (recomputing from the weights matches within 1e-9). Then C, the cues, nouns, arms and thresholds | no re-estimation after the lock. Calibration runs 101's arms on D (~1,200 runs) and checks only the §6 revision triggers | — |

**The confirmation (g) on C.**
- It uses 101's arms, the R, I and D families and 101's frozen numbers:
  - Holm 26/25 of 36;
  - medians ≥ 0.05;
  - D at 25/36;
  - the precondition at 30/36.
- P is dropped, because 160M has no decoded program.
- Three secondary families are added:
  - the read covectors of the **other heads of layer L*** as controls, which tests MD-1's selectivity;
  - a **site control**: the same edit at `RESID_PRE.L*+1@p_c` (research-state:405–410);
  - the head-only arms, for attribution.
- Cost: ~2,000 runs.

**Total:** about 17k passes for MD-1 plus calibration and confirm (28k with the p_c map). That is about 1–2 h of CPU, or
2–3.5 h if the heavy-capture runs reach ~0.4–0.5 s.

**The main failure modes:**
- The transport is distributed over 144 heads, or redundant across them.
- Consumption happens over several layers.
- The winner's curse when taking the maximum of 144 noisy scores. The D1/D2 replication rule addresses this; 005 chose
  its head from 2 frames.
- ΔT is not one-dimensional, so the rank-1 read loses signal.
- Self-repair through other readers of x(p_c). In 70M about 11–12 % of the effect bypassed the head (closure:40–44).

## 5. Data

**What is spent on 70M:**
- the 12 manifest prompts and all 60 manifest nouns (development, holdout and reserve);
- 198 frame texts: 162 from 005–023 (101's exclusion record) plus 101's 36;
- 431 cue ids used as cues in 005–025;
- the 108 exposed frames of 020's pool, 36 per template (023 design:100);
- the 80 exposed nouns (79 scorable; `peach` is two tokens, 020 design:143–147);
- 020's 24 fresh nouns, inspected since 020;
- for 70M, fresh plural cues are exhausted (D101:108).

**On 160M nothing has been executed or inspected.** 101's candidates 37–60 are 24 coordinated frames, all marked
eligible (`confirmation-v1.json`, picks 1–36). They have never run on any model.

**Are 70M-executed prompts fresh for 160M?**
- **For:**
  - The 160M outcome has never been observed.
  - The methodology names "another checkpoint or model" as a source of confirmation evidence (RESEARCH_METHODOLOGY:245).
  - 020 defines freshness by inspection (020 design:198), and no 160M statistic exists.
  - The frames were chosen by tokenizer rules, not by outcomes.
- **Against:**
  1. Both models share training data, tokenizer and recipe, so the known per-frame 70M results (weak frames; the frames
     where P disagreed: 03, 08, 09) carry information about 160M and can steer human choices.
  2. The template, the cue pair and the nouns survived *because* they worked on 70M, a selection effect that favours
     transfer.
  3. Model-free ledger keys would flag these prompts as executed.
  4. Confirming on 101's frames confounds "new model" with "same prompts".
- **Resolution:** use spent-on-70M frames for D, where the procedure is mechanical and no choice is left to bias. Use
  never-executed frames for C: candidates 37–60 plus 12 new frames under 101 §11, frozen by the tokenizer alone. 101's 36
  frames can serve as an optional descriptive set paired across the two models. State in the scope that the template,
  cues and nouns were selected on 70M.

**Cues:**
- `one`/`two` is primary, because it defines w_c.
- `three`/`many` form the stratum.
- `each`/`several` live only in cue-final frames, where p_c = p_t and there is no transport, so they are excluded from
  MD-1. They entered 70M's `d̂_T`, which the retrodiction tolerance absorbs.
- The numerals and quantity words of 009 and 011 have never run on 160M and could form a new-cue stratum.
- Template: `NAME and NAME VERBED {cue} ADJ`, with p_c from 4 to 7 and p_t = p_c + 1.

**Nouns:**
- Keep the 79 for comparability. They were chosen by tokenizer rules, not by outcome.
- Add 020's 24 fresh nouns as a stratum at no forward cost; they are output alternatives only (020 design:198–206).

**Per-noun results:**
- Measured per-noun Δc is free: 79 contrasts per pass (`pm.contrasts`).
- Per-noun *predictions* are another matter. 101 judged P in noun means because the frozen readout's per-noun errors
  reached 0.7–1.2 nats (closure:139–141), and 160M has no readout at all.
- A per-noun predictor, such as a first-order JVP at `RESID_PRE.L*@p_c`, needs tooling the package lacks
  (research-state:504).
- Per-noun counts over 36 frames are 79 correlated tests that share frames. Report them descriptively, for example the
  share of nouns with N > 0 in ≥ 26/36 frames, with no force on the label.

## 6. Risks, unknowns and the cheapest decisive checks

**Risks:**
1. Distributed or redundant transport. The single-head rule could fail even though a compact read subspace exists; the
   `MULTI_HEAD` branch exists for this case.
2. Multi-layer consumption across 12 layers.
3. Selection noise over 144 heads.
4. P1's rank-1 adequacy, and whether a `one`/`two`-only `d̂_T` is representative.
5. Stronger self-repair, which would weaken R relative to I.
6. Saturation. A larger |Δc_full| threatens D's `A_amp`, which passed by only 3 frames on 70M (closure:58–59).
7. Overshoot, if the read is a larger share of Δx (`‖off_c‖ < s`).
8. Code hazards: the label format, the pinned blobs, the 70M-only A0 test, model-free ledgers, and A5/A6's need for
   three templates.
9. The scope stays the same as 101's: one template, one checkpoint, frames as units, no stated population.

**The cheapest decisive checks, in order:**
1. With zero or a few passes: tokenizer and position checks on D and C, and a 160M contract test.
2. **The 70M retrodiction of MD-1** (about 6k 70M passes, under 10 min). It decides whether MD-1 is the method.
3. **The 160M behaviour precondition on D** (72 passes, under 1 min). It decides feasibility without revealing the
   mechanism.
4. A smoke timing (~20 passes).
5. The preregistered MD-1 run itself (about 13k passes, ~45–60 min). It is the decisive mechanism check: steps b–d
   answer single head or distributed.

**Caution on order.** Looking at the 160M layer or head results before freezing MD-1's thresholds makes the discovery
human-tuned. If the team explores first, label the 160M discovery exploratory, like 005's Tier A, and rely on C. The
claim then becomes "a read found on 160M is causal", not "the method transfers".
