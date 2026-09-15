# Experiment 004: Prompt Elicitation and Held-Out Factual Recall

## Purpose

Experiment 003 found that pinned Pythia-160M improved the intended capital's canonical rank in all 24 cases and improved canonical correct-versus-control selectivity, but produced `" the"` as top-1 in every case. Experiment 004 tests whether a small, predefined prompt family can elicit strict next-token capital recall without selecting prompts on the confirmation data.

## Research question and hypothesis

**Question:** Can one prompt format, selected on a predefined development set, make the intended single-token capital the actual top-1 next-token prediction of pinned Pythia-160M on a separately held-out set?

**Hypothesis:** At least one frozen prompt format will improve development behavior, and the deterministically selected format will produce the intended capital as top-1 for at least 6 of 12 held-out countries.

The 6/12 boundary is a predeclared project decision criterion, not a universal statistical threshold. Rank or selectivity cannot rescue failure of the top-1 criterion.

## Fixed model and measurement coordinates

- Model: `EleutherAI/pythia-160m-deduped`
- Revision: `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`
- Device and dtype: CPU, float32
- Loader: current `TransformerBridge.boot_transformers`
- Compatibility mode: disabled
- Execution: sequential cases under `torch.inference_mode()`
- Model weights: normal Hugging Face / TransformerLens cache only

The logit lens uses normalized accumulated residual states and the bridge's complete unembedding operation, including bias. Final projected logits must match actual final logits with `atol=1e-5`, `rtol=1e-5`; target rank and top-1 ID must agree exactly. Any failure stops the run.

## Frozen split

The split alternates entries from Experiment 002's already committed order. It is independent of Experiment 004 outcomes.

Development: Spain/Madrid, Austria/Vienna, Netherlands/Amsterdam, Norway/Oslo, Poland/Warsaw, Greece/Athens, Canada/Ottawa, China/Beijing, Egypt/Cairo, Chile/Santiago, Iran/Tehran, Switzerland/Bern.

Held-out: Portugal/Lisbon, Belgium/Brussels, Denmark/Copenhagen, Sweden/Stockholm, Hungary/Budapest, Ireland/Dublin, Japan/Tokyo, Russia/Moscow, Thailand/Bangkok, Iraq/Baghdad, Philippines/Manila, Syria/Damascus.

All country and whitespace-prefixed capital targets must remain single tokens. The held-out countries are untouched by Experiment 004 selection but were measured under older formats in Experiment 003. Confirmation is therefore conditional evidence for transfer of the selected new format, not independent country-level replication.

## Frozen candidate formats

1. `F1`: `The capital of {country} is`
2. `F2`: `{country}'s capital is`
3. `F3`: `What is the capital of {country}?`
4. `F4`: `Question: What is the capital of {country}?\nAnswer:`
5. `F5`: fixed two-shot question/answer prompt using France/Paris and Germany/Berlin, followed by the target question and `Answer:`.

France and Germany are outside the 24 evaluated countries. Demonstrations and formatting are identical for all F5 cases. Correctness is strict immediate next-token top-1 equality with the whitespace-prefixed capital token; fuzzy matching and multi-token generation are excluded.

## Controls

Reuse the three balanced incorrect-capital controls from Experiments 002 and 003, defined by cyclic offsets `(1, 7, 13)` in the original 24-pair ordering. Record each pairwise correct-minus-control margin, the mean control logit, and correct-minus-mean-control margin. These controls measure discrimination within the selected capital-token set, not the full vocabulary.

## Development selection

Evaluate all five formats on the 12 development countries: 60 cases. Rank formats lexicographically by:

1. higher intended-target top-1 count;
2. lower median intended-target rank;
3. higher mean correct-minus-three-controls final-logit margin;
4. fixed format order `F1 < F2 < F3 < F4 < F5`.

Use full-precision values. Graph appearance and individual examples cannot affect selection.

## Executable selection lock

`experiments/004-prompt-elicitation/selection-lock.json` is a hard integrity boundary. It records:

- exact development and held-out split;
- exact candidate formats and stable digest;
- pinned model identifier and revision;
- complete selection and tie-break rules;
- development artifact path and SHA-256 digest;
- selected format;
- protocol/code commit SHA used for development;
- creation timestamp.

Held-out execution requires the committed lock. It verifies schema and content, the frozen protocol digest, development artifact digest, deterministic winner reconstruction, model identity, and the current protocol/code commit relationship. It refuses missing, malformed, modified, uncommitted, or inconsistent locks. It has no held-out format override and never regenerates the lock silently.

## Two-phase execution

1. The development command evaluates only development cases and writes `outputs/experiment-004/development-results.json` plus a candidate lock.
2. The reviewed lock is placed at its tracked path and committed with the development implementation and protocol.
3. Only after that commit exists may the held-out command validate the lock and evaluate the 12 held-out cases with the locked format.
4. The final result and report combine exploratory development results with the conditionally confirmatory held-out result.

Any early held-out inspection, execution of multiple held-out formats, or post-development rule change makes the result exploratory.

## Measurements

For each case record exact prompt/tokenization, target and token ID, actual top-1 text/ID, strict success, target logit/rank/probability, three control measurements, raw target-logit and selectivity trajectories, adjacent-stage extrema, final-block changes, final-projection parity, and provenance. Intermediate probabilities are labeled logit-lens projections rather than actual intermediate predictions.

## Held-out decision

- Strong support: at least 6/12 intended capitals top-1.
- Partial, insufficient improvement: 1–5/12.
- Held-out failure: 0/12.
- Pythia-160M becomes a candidate causal substrate only after strong support.

Secondary rank and selectivity measurements remain descriptive and cannot change the primary decision.

## Outputs

```text
experiments/004-prompt-elicitation/
    README.md
    run.py
    selection-lock.json

outputs/experiment-004/
    development-results.json
    results.json
    report.html
    development-prompt-comparison.svg
    heldout-behavior.svg
    heldout-selectivity.svg
```

Generated outputs remain gitignored. The lock is version controlled.

## Interpretation limits

Experiment 004 is observational. It may show that a predefined format elicits better strict next-token behavior in this checkpoint and sample. It cannot establish that the model generally knows capitals, locate knowledge, identify a circuit, or attribute behavior to any layer, head, MLP, or causal mechanism. Few-shot success would also remain partly attributable to task-format induction.

C001 remains observational. Experiment 004 may add format-robustness evidence about its raw-logit trajectory but cannot promote it merely because behavior improves.

No anomaly is created solely because prompt formats differ.

## Resource envelope and outcome routing

The run uses one cached Pythia-160M model, 72 sequential cases, approximately 1.2–1.8 GB peak memory, and an estimated 1–3 minutes on the target M1 8 GB Mac.

- At least 6/12 held-out successes: propose, but do not implement, a causal Experiment 005.
- 1–5/12: consider one tightly justified follow-up or a different task; do not proceed directly to broad causal work.
- 0/12: do not automatically download Pythia-410M; compare changing task, scale, or research substrate.

