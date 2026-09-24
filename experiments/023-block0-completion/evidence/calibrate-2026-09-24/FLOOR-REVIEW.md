# Floor review — 2026-09-24

**Verdict: PASS.** It was conditional on one point: that the design calibrates the four conditions marginally. The
design text confirms it.

**The calibration run.** `calibrate` ran once at `4e5deeb` (15:53:07–15:53:31Z, exit 0). The launcher blocked 022's
table and the weight blob; the runner's model load was refused and so was every module call. No refusal fired.
- 10,000 draws per condition, 0 undefined.
- Kernel against a plain loop: 8.6e-14 over 192 values. Largest E6 difference: 7.5e-15.
- The record, `94db6df2…`, is committed alone in `51b5c7d`.

| condition | F = v₍₂₅₀₎ | median of draws | guard-bound |
|---|---|---|---|
| Y1/cue_final | 0.996897 | 0.998224 | no |
| Y1/coordinated | 0.998634 | 0.999205 | no |
| Y2/cue_final | 0.995053 | 0.998071 | no |
| Y2/coordinated | 0.995186 | 0.998423 | no |

**The reviewer's point: marginal or joint.** Each condition passes in 97.51 % of draws, while all four pass together
in 91.78 %. The design fixes the marginal reading:
- the conditions table gives each of K1–K4 its own `F = v₍₂₅₀₎` of 10,000 draws;
- "There is no aggregate label. Each condition is read on its own";
- calibration records "descriptively, the share in which all four conditions pass" among the values "Recorded,
  never corrected";
- the lock's frozen semantics state "no all-pass requirement".

So 91.78 % is what four separate 97.5 % floors produce. The floors were not changed. They were also not tuned against
Experiment 022's spent data; any comparison with it is descriptive only.

**The reviewer's second point: Experiment 020's untracked results state.** The file was hash-pinned in three places:
the pinned module's constants, 020's committed closure, and 023's inputs. It was not preserved in the repository, so
an exact, deterministic, recoverable copy was archived before lock (`13b8d39`, `evidence/archive/`).
