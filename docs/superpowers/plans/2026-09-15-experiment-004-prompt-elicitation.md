# Experiment 004 Prompt Elicitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a two-phase, integrity-locked prompt-elicitation experiment for pinned Pythia-160M.

**Architecture:** Pure protocol, ranking, digest, and lock-validation functions live in a small reusable module. The experiment runner reuses Experiment 002's measured-case path, exposes separate `development` and `heldout` commands, and refuses held-out execution unless a committed lock reproduces the deterministic development winner. Reporting is generated only from validated artifacts.

**Tech Stack:** Python 3.12, PyTorch, TransformerLens 3 TransformerBridge, pytest, JSON, Matplotlib SVG, static HTML.

**Spec:** `docs/superpowers/specs/2026-09-15-experiment-004-prompt-elicitation-design.md`

## Global Constraints

- Use only `EleutherAI/pythia-160m-deduped` revision `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`, CPU float32.
- Keep TransformerBridge compatibility mode disabled and require final-logit all-close plus exact rank/top-1 parity.
- Process cases sequentially under `torch.inference_mode()`; use normal model caches and do not train.
- Never expose a held-out format override or silently regenerate a failed lock.
- Do not inspect held-out outcomes before the selection lock is committed.
- Keep all interpretation observational and do not implement Experiment 005.

---

### Task 1: Pure protocol, selection, and digest contracts

**Files:**
- Create: `src/neural_decompiler/prompt_elicitation.py`
- Create: `tests/test_prompt_elicitation.py`

**Interfaces:**
- Produces frozen `CapitalPair` splits and `PromptFormat` definitions.
- Produces `format_digest()`, `select_winner(format_summaries)`, `sha256_file(path)`, `build_selection_lock(...)`, and `validate_selection_lock(...)`.

- [ ] Write tests for exact split membership, exact five formats, stable digest, lexicographic selection at every tie-break, malformed/tampered locks, artifact digest mismatch, winner mismatch, model mismatch, protocol mismatch, and committed-lock enforcement through an injected Git-state reader.
- [ ] Run `uv run pytest tests/test_prompt_elicitation.py -q` and verify the tests fail because the module does not exist.
- [ ] Implement the smallest pure functions and immutable constants that satisfy the tests.
- [ ] Run `uv run pytest tests/test_prompt_elicitation.py -q` and the full suite.
- [ ] Commit the green task.

### Task 2: Two-phase runner and measurement orchestration

**Files:**
- Create: `experiments/004-prompt-elicitation/run.py`
- Create: `tests/test_experiment_004.py`
- Modify: `.gitignore`

**Interfaces:**
- `run.py development` loads the pinned model, evaluates exactly 60 development cases, writes `development-results.json`, prints deterministic ranking, and writes a candidate lock outside the tracked lock path.
- `run.py heldout` accepts no format argument, validates the tracked lock, evaluates exactly 12 locked-format cases, and writes final artifacts.

- [ ] Write failing tests for CLI phase isolation, exact case counts, absence of a format override, sequential analysis, model identity, strict parity propagation, candidate-lock content, and refusal paths.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement runner orchestration by reusing Experiment 002's tokenization, control, projection, and selectivity path; do not duplicate numerical logic.
- [ ] Run focused and full tests.
- [ ] Commit the green task.

### Task 3: Deterministic reports and experiment documentation

**Files:**
- Modify: `experiments/004-prompt-elicitation/run.py`
- Create: `experiments/004-prompt-elicitation/README.md`
- Modify: `tests/test_experiment_004.py`

**Interfaces:**
- Held-out completion writes `results.json`, `report.html`, `development-prompt-comparison.svg`, `heldout-behavior.svg`, and `heldout-selectivity.svg`.

- [ ] Add failing tests for report sections, conditional-confirmation language, decision labels, SVG series/counts, C001 boundaries, and required artifact metadata.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement static report generation and the one-command-per-phase README.
- [ ] Run focused and full tests.
- [ ] Commit the green task.

### Task 4: Real development run and committed integrity lock

**Files:**
- Generate: `outputs/experiment-004/development-results.json`
- Generate: `outputs/experiment-004/selection-lock.candidate.json`
- Create: `experiments/004-prompt-elicitation/selection-lock.json`

**Interfaces:**
- The tracked lock is byte-for-byte reviewed from the candidate and binds the development artifact to the protocol/code commit.

- [ ] Commit all protocol, code, tests, and documentation before running development; record that commit as the development protocol/code SHA.
- [ ] Run the development command only and verify 60 cases, all parity gates, pinned revision, format rankings, and artifact metadata.
- [ ] Copy the candidate lock to the tracked lock using `apply_patch`, verify its digest and deterministic winner, and commit it without inspecting held-out data.
- [ ] Prove the lock is committed and the working tree is clean.

### Task 5: Locked held-out run, visual QA, and evidence review

**Files:**
- Generate: all final `outputs/experiment-004/` artifacts.
- Modify only after evidence review: `experiments/004-prompt-elicitation/README.md` and `research/claims/C001-final-block-target-logit-increase.md` if the measured evidence warrants it.

**Interfaces:**
- Held-out execution obtains the format exclusively from the verified lock and emits the predeclared decision.

- [ ] Run held-out exactly once after the lock commit.
- [ ] Verify 12 and only 12 held-out cases, strict parity, artifact schema, and the 6/12 decision.
- [ ] Inspect HTML and SVG output for readability and consistency with JSON.
- [ ] Review C001 and anomaly status under `docs/RESEARCH_METHODOLOGY.md`; do not force changes.
- [ ] Run the complete test suite and a fresh artifact-integrity check.
- [ ] Commit only warranted evidence documentation changes; generated outputs remain ignored.
- [ ] Stop without implementing Experiment 005.
