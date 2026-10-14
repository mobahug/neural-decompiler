# Experiment 101 — the lock (2026-10-14; weights only)

The production `lock` ran **exactly once**, at commit `3621eec4b6a901119a4b96d4fdc07fa3b19652ab` on `exp101-phases`,
under a guarded launcher. It wrote two candidates, **which are not installed or committed**: design §9 says "The user
commits the lock", so the user decides whether to install and commit them, after the independent lock review
(`../lock-review-2026-10-14/`, an AI review, not a human validation: **PASS WITH NOTES — no blocker to committing the lock**).

| candidate | digest |
|---|---|
| `outputs/experiment-101/candidate-lock.json` (68,943 bytes) | file sha256 `8dc6e6d38c876a2dd9bf905fbeaf36b774b487967597795beb032bb4a0751087`; content sha256 `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061` |
| `outputs/experiment-101/candidate-preregistration.md` (8,703 bytes) | sha256 `3761b8130203d5d15ac0e30055aa135626a6d97f0ab807df6cf4d31b6af8e980` |
| the results state after the lock (git-ignored) | `state_sha256` `a4657f80eed3e23587ea44597b77ce064c432eea5623799c838acdb668f40e12` |

The lock binds the run `30ca8697c8702496`, the freeze (`ef356506…`), the calibration (`1eb1fe68…`; τ_rem
0.025153865322968608, τ_ins 0.03511072680382421, P evaluable), `w_c` (digest `253a4733…`, ‖w_c‖ 0.7659205190757624)
and the 288 isotropic directions by digest, the module (`read_intervention.py` at blob `d7db6197…`) and the 20 pinned
blobs, the model's parameter and embedding digests (`fd953f1c…`, `9cd6f39b…`), the stage-1 files' paths, layout and
table metadata, the statistics, the labels, the scope and the frozen tolerances (unchanged; rule 1).

**Why the lock ran before the user's decision.** The lock is a mechanical function of committed inputs. The one choice
the calibration left open, whether the Level-1 tail risk at confirm justifies an amendment (the calibration review's
N1), cannot be taken inside this protocol version in any case: a tolerance lives in `ri.TOLERANCES` and is bound by the
installed calibration file (its module blob and tolerances), and calibrate runs once. So an amendment needs a new
protocol version whether or not the lock has run, and running it forecloses no option. The user's decision is the
commit.

## The launcher (`launch_lock.py`, sha256 `90e8159f97fc38234835c58e6443474677dfe0236ca48d09e6e67fe124e5e485`)

025's lock-launcher pattern, with the freeze review's N1 applied to the audit hook. `torch.nn.Module.__call__` refuses
for the whole run; every capture and intervention entry point and its aliases refuse (9 found by identity); every
measurement-side computation refuses (`ri.run_arm`, `ri.capture_reference`, `ri.measure_frame`, `ri.stage_runs`, the
gates, the run records, `ri.frame_values`, the statistics, `ri.head_delta`, `ri.assert_measurements`, `ul.i3_error`,
`rd.NounSet.contrasts`, `rd.ReadoutProgram.level1_detail`); the model load is counted and every module and model-class
forward is sealed; the audit hook allows repository writes only under `outputs/experiment-101/`. The program's own
functions stay available: the lock recomputes the exposed prediction table on 020's locked (spent) states and compares
its digest with the calibration's (plan N2).

- **The dry run** (`LOCK101_DRY_RUN=1`; `dryrun_run.json`, `dryrun.log`; 09:30:29–09:30:55 UTC): the same preflight
  (a clean tree, the calibration committed, no lock installed, the state's lock not started, the installed calibration
  the state's); the weights loaded under every guard, 430 modules sealed, 0 module calls; the exposed table recomputed
  in a third process: `ce055e09…`, equal to the calibration's. It did not call `lock`; the state file's sha256 was the
  same before and after (`state_before_dryrun.txt`).
- **The production run** (`lock_run.json`, `lock.log`, `lock.exit`; 09:31:30–09:32:09 UTC): token
  `LOCK101_CALIBRATION` = the installed calibration's content sha256; an exclusive sentinel (`lock.sentinel`); **exit
  0, lock invocations 1, model loads 1, module calls 0, refusals 0, forbidden reads or writes 0**; repository writes:
  the two candidates and the results state (one temporary file and one rename). The reads under `outputs/` were 020's
  results state, 101's state, and the launcher's own hashing of 101's outputs after the run.

## Files

| file | what |
|---|---|
| `launch_lock.py`, `launcher_sha256_at_production.txt` | the guarded launcher and its sha256 at production |
| `dryrun_run.json`, `dryrun.log`, `state_before_dryrun.txt` | the dry run |
| `lock_run.json`, `lock.log`, `lock.exit`, `lock.sentinel` | the production run |
| `DECISION.md` | the decision this lock puts to the user, with the recommendation and the commands |
| `prepared/launch_stage1.py` (version 2), `prepared/launch_stage1_v1_as_reviewed.py` | the guarded stage-1 launcher, prepared and not run: version 1 as the lock review read it, version 2 with its S1–S3 and N8 applied (diff-checked by the reviewer, not fully re-reviewed) |
| `prepared/hook_self_test.py/.out.json` | an isolated test of version 2's hook decisions (the functions extracted with `ast`; nothing of the launcher runs): 9 allowed and 10 refused, all as expected |
| `SHA256SUMS` | this directory's digests |
