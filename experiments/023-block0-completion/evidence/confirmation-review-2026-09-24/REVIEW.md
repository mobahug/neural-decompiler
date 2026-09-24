# Independent confirmation review — 2026-09-24

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the single `confirm` run read-only, after the
run and before `report`, with its own code. It independently reproduced the recorded outcome: Y1/cue_final
ENVELOPE_ONLY_FAILURE; Y1/coordinated, Y2/cue_final and Y2/coordinated PASS.

Guards in every process: an audit hook and a `torch.load` refusal made 022's table unavailable (6 routes refused on a
live probe). The model was loaded once for its weights; 0 module calls happened during the load. Afterwards
`nn.Module.__call__` and the capture, run, patch and intervention entry points were refused. The only phase run was
the stock `validate`, inside the guard, which left the state unchanged.

| item | result |
|---|---|
| 1. Prompt accounting (`acct`) | The reviewer rebuilt the manifest from the freeze file and 020's pool: 18 + 18 + 2,592 + 432 = 3,060 unique keys. It equals the 3,060 executed `capture_prompt` calls (numbered 1–3,060 without gaps, each once), the ledger, and the manifest in the freeze file. 0 extra, 0 missing, 0 duplicated. `run_capture` 3,060; `run_patched` 0; `run_interventions` 0. 0 of 36,252 forbidden keys was executed. |
| 2. Ordering and barrier (`acct`) | The last stage-1 prompt ran at 17:40:30.974Z. Both Y2 files were created and last modified at 17:40:52.235Z, written once. The first target ran at 17:40:53.158Z. The stage-1 record reproduces its digest and names the lock and `c7efec7`. |
| 3. Y2 table (`rebuild`) | Rebuilt from the 18 recorded stage-1 states (each matching its digest) and the weights. Two independent serializers give exactly `4623adde…`, and the index is exactly `e87091ef…`. |
| 4. The four results (`score`) | Every statistic agrees within 4.4e-15 in float64 and 1.8e-15 in exact rational arithmetic. Labels are recomputed against the lock's full-precision floors. |
| 5. Identities (`rebuild`, `extra`) | All 3,024 pairs recomputed. I1 1.3246e-5, I3 2.0288e-5 and I4 4.3322e-5 equal the recorded values in value and location. I5 is 0. The saved ceiling reproduces bit for bit. The I7 Y1 rebuild is identical to `37603f05…`. |
| 6. Inputs and isolation (`extra`, `final`) | The state, lock and record bind the committed files. The 022 table was never used. Every model-execution path reachable from confirm goes through the counted entry points. |
| 7. `" shiny"` (`score`) | Row 684 of Y1/coordinated. The primary result includes all 864 pairs. Descriptive only: without it (863 pairs), g = 0.9987161006600696, still PASS. Nothing excludes it. |
| 8. State and one-shot (`final`) | The state digest recomputes. Dropping the two descriptive records added after the results reproduces the runner's completion digest `2cba1ece…`, so the results are exactly as first written. Confirm ran once and a second confirm is refused. |

| condition | g | F (lock, full precision) | g − F (exact) | result |
|---|---|---|---|---|
| Y1/cue_final | 0.9955204329648195 | 0.9968974947302105 | −1.3771e-3 | ENVELOPE_ONLY_FAILURE |
| Y1/coordinated | 0.9987141803462269 | 0.9986343903419131 | +7.97900043138065e-5 | PASS |
| Y2/cue_final | 0.9974185656823554 | 0.9950528086546427 | +2.3658e-3 | PASS |
| Y2/coordinated | 0.9965628370581808 | 0.9951864564481918 | +1.3764e-3 | PASS |

Y1/coordinated's margin is about 7×10¹¹ times the numerical error, and its classification used the full-precision
floor.

**Notes (non-blocking).**
1. A refused second confirm would leave no trace in the state. "No second attempt" rests on the launcher files, the
   file times and the completion-digest reconstruction.
2. The accounting counted forward calls at the capture layer. The only other forward paths in the codebase cannot
   be reached from confirm.
3. The I7 record is timestamped to the second; the code writes it before the first capture.
4. Y1/cue_final is well below its envelope: 1 in 10,000 draws is at or below it. Under the frozen reading this is a
   quantitative shift, not a refutation.
5. The Y2 table is committed as closure evidence (`5344f44`).
