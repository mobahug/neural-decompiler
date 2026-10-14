# Experiment 101 — the independent confirmation review (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent reviewed the one confirm,
recomputing the gates, the primary statistics and the label from the saved measurements with its own code first and the
frozen functions afterwards.

**Verdict on execution and computation: PASS WITH NOTES — no blocker.** The recomputed label is
**`READ_COMPONENT_EFFECTS_AS_PREDICTED`**; its code reproduces every number of the state's result bit for bit. It ran no
forward pass and no 101 prompt (a 624-sequence token guard and refused module calls, never fired), and left the
repository unchanged.

**The findings and what was done:** SF1 (design §2's per-frame overshoot report is not recorded at confirm) and SF2
(the per-arm ceilings and E₁ are recorded but not reported) are answered in `../closure-2026-10-14.md` with its tables.
N1–N10 are recorded there and in the experiment README (the nearly vacuous head counts; `many` would fail D; D's
saturation; the "results sha256" in the log is the content digest; `awaiting_commit` stays true; raw dictionaries in the
report; the implicit limitations; the evidence copy; the smoke's re-run of a spent frame; the stale README status).

**Not copied:** `tmp/` (scratch). `SHA256SUMS` is the reviewer's; `SHA256SUMS.repository` lists what is here.
