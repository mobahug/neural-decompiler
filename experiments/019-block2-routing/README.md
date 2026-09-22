# Experiment 019: Frame-Conditioned Routing of Channel D — Is the Frame-Specific Membership of Block 2's Recovery Subset Predicted from the Frame's Reference State?

Implements the approved design
[`docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md`](../../docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md)
(revision 3, commit `b130b50`; wording `efe0623`) through the plan
[`docs/superpowers/plans/2026-09-21-experiment-019-block2-routing-plan.md`](../../docs/superpowers/plans/2026-09-21-experiment-019-block2-routing-plan.md).
Prospective, zero-parameter, mechanism (routing). Experiment 018 showed that channel D — block 2's MLP at the frame's
own operating point — is carried for unseen cues by a locked subset of 256 neurons and, descriptively, that *which*
neurons carry it depends on the frame. This experiment asks whether that redistribution is predictable: Experiment
018's masked chain now takes one mask per changed position, and a nested ladder of selectors scored with one read-unit
rule is evaluated at `k ∈ {16, 64, 256}` (decision size 64) — the matched global `S'_k` (the decision comparator), the
template ranking `T_k(τ, p)`, the full prospective evaluation `E_k(f, p)` (the template's 254 exposed cues' *predicted*
effects at the frame's own reference state), the operating-point-plus-drive rule `G_k(f, p)` (the frame's reference
pre-activations plus a frozen per-template three-quantile drive profile through the exact GELU), with the inherited
Experiment 018 `S_k`, seeded random `k`-subsets, `S'_1` and the frame's top neuron `E_1(f)` as controls. Every
selector is a frozen rule applied to locked reference states, the weights and the exposed cues' `ΔE`; every subset is an
explicit index list committed before the fresh prompts it is scored on. Two post-confirmation objects enter no
selector: the measured-effect ranking `O_k(f, p)` (descriptive; the target of the membership question) and the
leave-one-cue-out greedy `O*_k(f, −w)` — a *cross-validated empirical headroom witness*, never a global oracle, ceiling
or upper bound — whose headroom `H*_64` classifies only a negative routing result (a low `H*` means the witness did
not establish enough selectable headroom for the fixed-population-versus-unpredicted distinction; it is never a proof
that no frame-specific headroom exists).

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 255 exposed cue tokens (Experiment
  018's 231 and its 24 confirmed) and the 90 exposed frames (its 78 and its 12), with the 80 nouns; the licensed pairs
  are Experiment 018's 9636 exposed and 2160 confirmed pairs (11 796).
- [`inherited/experiment-018-pair-extract.json`](inherited/experiment-018-pair-extract.json) — Experiment 018's
  per-pair measured `F`, `Π`, `ΔT`, `c_L`, head row and its `S0` / `S256` / `S2048` predictions over the 11 796 pairs, the
  digests of its twelve fresh frames' stage-1 reference states and its per-frame top-256 lists (78 at explore, 12 at
  stage 1). Built deterministically by the committed
  [`build_inherited_extract.py`](build_inherited_extract.py) from the closed 018 results state
  (`outputs/experiment-018/results.json`, run `86ccb888f52600ef`, state sha256 `2c9195d5…`) and the 018 lock; its
  `source` names the results-state file sha256 and `state_sha256`, the lock file sha256 and content sha256, the run id,
  the explore and confirm commits and the extraction schema version; `build_inherited_extract.py --check` rebuilds it
  and compares (content sha256 `625394936cadb4bd9ac348d07f22af96b60b4ab0c7399d24df7d5e493bf1ae04`, byte-identical on
  2026-09-21). The Experiment 011 lock's axes and read weight, the Experiment 012 lock's bases, the Experiment 018
  lock's layer-3 bases, `p_t` base, 78 reference states and subsets are used verbatim.
- `confirmation-v1.json` — frozen by tokenizer rules before any Experiment 019 model output: 24 fresh cue tokens by
  lexical class (the first eligible entries of Experiment 017's lists extended), **with no expectation attached**;
  eighteen fresh frames, six per template; the complete prompt-key manifest (every fresh cue in the 90 exposed frames
  and the 18 fresh frames, the fresh frames' reference and cue-pair prompts). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/019-block2-routing/run.py validate
uv run python experiments/019-block2-routing/run.py freeze-confirmation
uv run python experiments/019-block2-routing/run.py explore
uv run python experiments/019-block2-routing/run.py lock
uv run python experiments/019-block2-routing/run.py confirm
uv run python experiments/019-block2-routing/run.py report
```

- `explore` (once, exposed pool): the 90 frames' reference states (the 78 checked against the Experiment 018 lock and
  the 12 against its stage-1 digests; locked); every prospective selector — `S'`, `T`, the drive-quantile tables, `E`
  and `G` for every exposed frame and position, `S'_1`, `E_1`, the random controls, the inherited `S_k` — computed and
  digested from the locked states, the weights and the exposed cues' `ΔE` *before any pair is measured*; I9 (the `E`
  rule with Experiment 018's inputs reproduces its recorded per-frame lists); the 11 796 licensed pairs re-measured with
  the layer-3 residuals, the head's row and the block-2 input at both changed positions captured; replication of the
  018 extract; the identities I1–I10; every prospective rung predicted; the exposed oracles (cue-in-sample ranking; the
  leave-one-cue-out witness) and the exposed statistics.
- `lock` (weights, the locked axes, read, bases, states and lists — no forward pass): the prediction table for every
  fresh token × exposed frame and every prospective rung; the frozen floors; `candidate-lock.json` and
  `candidate-predictions.md`. Installing them as `preregistration-lock.json` and `predictions.md` and committing them
  is the preregistration act.
- `confirm` (once): validates the artifacts (digests, floors, the lists as the frozen rule applied to the locked
  scores), recomputes the selectors from the locked states and every locked prediction (refusing before any fresh
  prompt on a mismatch), then **stage 1** (the eighteen fresh frames' reference and cue-pair prompts only; validity;
  each frame's own `E` and `G` lists from its locked state; the frame-conditional prediction table; serialized and
  digested — zero of the 432 fresh-cue × fresh-frame target pairs and none of the 2160 exposed-frame pairs run before
  the barrier) and, only after re-reading that digest from disk, **stage 2** (every fresh cue in both sets; the
  measured objects and per-neuron effects; then, and only then, the ranking oracle per frame and the witness per pair;
  scoring: Y1, Y2 with the frame-count and split guards and the headroom classification, Y3 with the denominator
  condition and per-set floors, Y4 on Y2 and pooled, Y5 per set).
- `report`: renders `outputs/experiment-019/report.md`.

## Status — 2026-09-21: complete; outcome `ROUTING_PREDICTED_TOKENS | ROUTING_PREDICTED_FRAMES_CONDITIONAL | OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT | TEMPLATE_FAMILY_INSUFFICIENT | MEMBERSHIP_PREDICTED`

Tier A (`explore`) ran once at `2dbfdd1` on the 11 796 licensed pairs (255 tokens, 90 frames; 14:50–16:09 UTC, CPU,
4 threads): every exposed frame's state matched the Experiment 018 lock or its stage-1 digest; the Experiment 018
extract replicated (deviation 0.0); the identities held (I4 8.5e-6, I5 1.2e-5, I6 3.3e-5, I7 8.6e-7, I8 1.5e-14,
I10 5.6e-16; I9 — Experiment 018's own per-frame ranking recomputed from the locked states — reproduced all 90
recorded lists exactly); the selectors were computed and digested (`62108946c0061d7e…`) from the locked states, the
weights and the exposed cues' `ΔE` before any pair was measured: `S'_1 = {1987}`, `S'_64` at `p_c` starting
`111, 129, 173, 228, 287, …` (59 of 64 shared with the inherited 018 `S_64`), the frame's top neuron by `E` being
1987 in 58 of the 90 frames, 1726 in 15, 1102 in 10 (mostly quantifier frames), 1311 in 3. Exposed record
(cue-in-sample; the floors are frozen constants), `κ_{c_L}` at `k = 64`: `S'` 0.754, `T` 0.777, `E` 0.838, `G` 0.822,
inherited `L` 0.748, ranking oracle `O` 0.838, witness `O*` 0.946; gains over `S'_64`: `T` +0.022, `E` +0.084, `G`
+0.068, `O*` +0.191; `ρ_64` 0.814; `E_64` above `S'_64` in 66 of 90 frames; membership against the measured-effect
ranking's top-64 at `p_c`: `E` 0.925, `G` 0.738, `T` 0.487, `S'` 0.420.

The candidate lock and predictions were installed byte-identical and committed by hand (`9033041`; lock content
sha256 `6c0cb1849955e60bd84be34f89239221e42ce1ddb8cdb5cb00a14c18e6e48165`, predictions sha256
`1781102d748f5ab44ec0cf180917f2daca68da6589adeb3c9f7d18c2ebe15b9f`; the predictions are copied to
[`evidence/candidate-predictions-2026-09-21.md`](evidence/candidate-predictions-2026-09-21.md)); only `README.md`,
`predictions.md` and `preregistration-lock.json` changed since the explore commit; the targeted gate passed at
`9033041` (`tests/test_block_routing.py`, `tests/test_experiment_019_runner.py`, the pinned-model contract); the
reviewer signed off; and the single `confirm` ran at `9033041` on a clean tree straight through both stages
(19:13–19:31 UTC, CPU, float32, 4 threads — equal to the explore runtime record; results state sha256
`94af0d082ea7cadac0e2fc4d8b93f58c3831f8a1d3ea5057fa233961737467e3`). Before any fresh prompt the lock was validated,
the selectors were recomputed from the locked states with zero difference and the complete 2160-row prediction table
was reproduced with zero difference. **Stage 1** ran only the eighteen fresh frames' reference prompts and cue pairs —
all eighteen valid (head informative, cue effect 79/72 in every frame; `p_c` 3–7, `p_t = p_c` or `p_c + 1`) — computed
each frame's own `E_64` and `G_64` from its locked state (`E_64` shares 17–30 neurons with `S'_64` and 40–57 with
`G_64`), computed the 432 frame-conditional predictions and digested them with the states and the lists
(`19149bd32e71875c1f934a7b85103a36b7897e6fd3f51946d3034ac831c592cd`); the ledger held zero of the 2160 exposed-frame
target pairs and zero of the 432 fresh-frame target pairs at the barrier. **Stage 2** started only after re-reading
that digest from disk and ran the 24 fresh cues in the 90 exposed frames (2160 pairs) and in the eighteen fresh frames
(432 pairs); the ranking oracle and the leave-one-cue-out witness were computed only after every measurement. The
final report is copied verbatim to [`evidence/final-report-2026-09-21.md`](evidence/final-report-2026-09-21.md).
Ledger after confirm: 2916 prompt keys (the 270 of Tier A plus all 2646 keys of the frozen manifest, each once), 80
noun keys. Experiment 019 is closed; nothing is amended or rerun.

- **Identities (checks only):** on the 2592 fresh pairs the exact chain reproduced the captured layer-3 residuals (I4
  5.4e-6), the head's captured rows (I1 1.7e-5, I5 1.2e-5), the measured head change (I6 2.9e-5) and the split (I7
  7.6e-7); the reference rung equalled Experiment 017's Level 0 (I8 6.9e-15); the read identity I10 5.0e-16; every
  re-captured reference state matched its locked copy or stage-1 digest.
- **Y1 — strict prospective (24 fresh cues × 90 exposed frames, 2160 pairs): `ROUTING_PREDICTED_TOKENS`.**
  Precondition met (24 scored tokens; reference rung `c_L` R² 0.999, rows 0.995, `ΔT` 0.999; gaps pooled 0.252,
  cue-final 0.281, coordinated 0.315). `Δκ_64(E)` **0.090** (floor 0.05); `E_64` above `S'_64` in **64 of 90** frames
  (no ties; the guard needs more than half); split gains cue-final **0.070**, coordinated **0.159** (floor 0.02);
  jackknife over frames 0.084–0.104. `κ_{c_L}` at `k = 64`: `S'` 0.722, `T` 0.747, **`E` 0.813**, `G` 0.800,
  inherited `L` 0.709, `O` 0.823, `O*` 0.932, random controls 0.294 / −0.039 / −0.013. Per template `κ` (`S'` / `T` /
  `E` / `G`): cardinal 0.641 / 0.634 / 0.725 / 0.630, quantifier 0.799 / 0.848 / 0.867 / 0.882, coordinated
  0.546 / 0.517 / 0.705 / 0.662. At `k = 16`: `S'` 0.675, `E` 0.731 (+0.056), `G` 0.692; at `k = 256`: `S'` 0.855,
  `E` 0.943 (+0.088), `G` 0.935. `κ_Π(E_64)` 0.659 against `S'` 0.558; `κ_row(E_64)` 0.306 (diffuse).
- **Y2 — frame-conditional prospective (24 fresh cues × 18 new frames, 432 pairs):
  `ROUTING_PREDICTED_FRAMES_CONDITIONAL`.** Precondition met (reference rung `c_L` 1.000, rows 0.995, `ΔT` 0.999;
  gaps pooled 0.198, cue-final 0.232, coordinated 0.255; 12 valid cue-final and 6 valid coordinated frames).
  `Δκ_64(E)` **0.144**; `E_64` above `S'_64` in **14 of 18** frames; split gains cue-final **0.147**, coordinated
  **0.133**; jackknife 0.113–0.169. `κ_{c_L}` at `k = 64`: `S'` 0.649, `T` 0.713, **`E` 0.793**, `G` **0.813**, `L`
  0.650, `O` 0.793, `O*` 0.890, random 0.272 / −0.107 / 0.020. Per template (`S'` / `T` / `E` / `G`): cardinal
  0.653 / 0.731 / 0.818 / 0.878, quantifier 0.677 / 0.784 / 0.815 / 0.800, coordinated 0.569 / 0.493 / 0.701 / 0.770.
  At `k = 16`: `E` +0.062 over `S'`; at `k = 256`: +0.082.
- **Y3 — the operating-point-plus-drive rule: `OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT`.** `ρ_64 = Δκ_64(G) /
  Δκ_64(E)`: Y1 **0.862**, Y2 **1.136**, pooled **0.913** (floors 0.50 per set, 0.60 pooled; denominators 0.090 /
  0.144 / 0.097, all above the 0.05 condition). `ρ` is unclipped by design: on the new frames `G_64`'s gain over the
  same `S'_64` baseline (0.164) exceeded `E_64`'s (0.144), which is what a value above one means — not more than the
  whole of the routing gain explained. Descriptively `G`'s gain per template is −0.012 / 0.083 / 0.115 (cardinal /
  quantifier / coordinated) on Y1 and 0.226 / 0.123 / 0.201 on Y2; "smallest in cardinal" holds on Y1 only. That `G`
  carries the majority of `E`'s gain on both dimensions is the preregistered result; that it exceeds `E` on one set is
  recorded, not generalized.
- **Y4 — the template-family account: `TEMPLATE_FAMILY_INSUFFICIENT`.** `A = κ(E_64) − κ(T_64)`: pooled **0.068**, on
  the new frames **0.080** (both ≥ 0.05); `B = Δκ_64(T)`: pooled 0.029, Y2 0.064, Y1 0.024. The frame-specific
  evaluation exceeds the template ranking by the margin on both sets, so template identity alone does not account for
  the routing.
- **Y5 — membership: `MEMBERSHIP_PREDICTED`.** Mean `|E_64 ∩ O_64| / 64` against the measured-effect ranking's top
  64 at `p_c`: Y1 **0.867** (floor 0.60) against `S'` 0.410 (margin 0.457 ≥ 0.20; `G` 0.723, `T` 0.472); Y2 **0.867**
  against `S'` 0.375 (`G` 0.707, `T` 0.435).
- **Pooled (2592 pairs):** `κ_{c_L}` at `k = 64` `S'` 0.713, `T` 0.742, `E` 0.810, `G` 0.802, `O*` 0.927; `ρ_64`
  0.913; `H*_64` 0.213.
- **Descriptive expectations (no outcome force).** (i) `E`'s gain holds at the other sizes on both sets and pooled
  (`k = 16`: 0.056 / 0.062; `k = 256`: 0.088 / 0.082); (ii) the gain over the inherited 018 `S_64` at least equals the
  gain over `S'_64` on Y1 (0.103 vs 0.090) and pooled (0.108 vs 0.097) but **not on Y2: 0.1431 against 0.1439, short
  by 0.0008** — recorded exactly as a descriptive expectation that did not hold; it changes none of Y1–Y5; (iii) the
  best random 64-subset stays below 0.5 with the margin on every set (0.294 / 0.272); (iv) `κ_{c_L} ≥ κ_Π` and the
  row diffuse on every set; (v) as above; (vi) `S'_1` 0.281 / 0.231 against `E_1(f)` 0.373 / 0.256; (x) the witness's
  own `κ` 0.906 / 0.932 / 0.933 at `k = 16 / 64 / 256` on Y1 and 0.870 / 0.890 / 0.889 on Y2.
- **Retained limits.** *Not a per-frame rule:* `E_64` loses more than 0.05 R² to `S'_64` in **7 of 90** Y1 frames
  (`cardinal-013-2`, `cardinal-014-2`, `cardinal-016-2`, `coordinated-adjective-009-1`, `quantifier-012-2`,
  `quantifier-015-2`, `quantifier-new-2`; `G`: 10) and in **2 of 18** Y2 frames (`coordinated-adjective-019-4`,
  `coordinated-adjective-019-6`; `G`: none); the result is an aggregate routing result under the preregistered
  majority and family guards, not a rule that holds in every frame. *Headroom:* `O*` is a cross-validated empirical
  headroom witness — a leave-one-cue-out greedy fit — never a global oracle, ceiling or upper bound; `H*_64` is
  **0.210** on Y1 and **0.241** on Y2 and `E` captures **0.43** and **0.60** of it (the ranking proxy `H_64` 0.100 /
  0.143): the tested rule explains a substantial part of the transferable routing structure the witness demonstrates,
  not all of it. *Heterogeneity:* the pronouns `he` and `she` have `c_L` near zero or negative on both sets (`he`
  −0.013 / 0.001, `she` −0.041 / −0.019; `ĉ_L(E_64)` −0.003 / 0.016 and −0.035 / −0.006), and the token means'
  `ĉ_L(E_64)` sits closer to the measured `c_L` than `ĉ_L(S'_64)` does for all 24 tokens on Y1 and for 23 of 24 on Y2
  (`different`: `E_64` 0.2701, `S'_64` 0.2810, measured 0.2757); `ĉ_L(E_64)` falls below the
  measured value only for `triple` on Y1 and for `different`, `pair` and `similar` on Y2; the lexical classes carried
  no preregistered directional expectation, and this and the per-frame spread remain descriptive.

What this settles and what it does not. The frozen interpretation table gives **Account 2, compact form**, verbatim:
the frame's operating point, with a frozen per-template drive profile, routes channel D: a frame's participating
neurons are named from its reference pre-activations and the locked drive tables, blind to the fresh cues, in
exposed and new frames; the template ranking does not suffice. The decoded program gains a routing rule:
`reference state → operating point (+ the cue population's typical drives) → which of the 2048 neurons carry the
correction → the transport read`. Stated at its safe strength: a fixed 64-neuron population is not the whole story —
frame-specific routing within block 2 is prospectively predictable from the frame's reference state; a compact
operating-point-plus-typical-drive rule carries most of that routing gain on unseen cues and on unseen frames, while
template-family identity alone is insufficient; and the predicted selector substantially recovers the neurons ranked
highest by the fresh measured effects. The rule is population-level rather than universal per frame, and it does not
exhaust the additional selectable headroom demonstrated by the cross-validated witness. This is not the claim that 64
neurons suffice (every selector's absolute `κ` stands beside its gain), nor that either oracle is the best possible
subset, nor that a frame's state is predicted from its text, nor anything about behaviour. Limits: one checkpoint,
three templates, five lexical classes (the ordinal class exhausted by this experiment), one head, one block, the
transport read at `p_c` at size 64 relative to the matched global rule (the other sizes, the head objects and `p_t`
described, not claimed), eighteen new frames.
