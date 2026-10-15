# Review of the C004 draft (2026-10-14)

**AI review (independent, read-only), not human validation.** This review was written by a Claude agent (Opus 5.5)
on 2026-10-14.
- **Repository state.** HEAD `a9b6b33` on `c004-exp102-design`, equal to `origin/exp101-phases`; `main` is still at
  `96f0cc8`.
- **What was reviewed.** The untracked draft `research/claims/drafts/C004-draft-2026-10-14.md`, sha256 `b51a04f9…a4bc9a`.
- **What was done.** Nothing in the repository was created, changed, staged or committed. No model, runner or test ran.
  The only write is this file.

**Read in full:**
- `docs/RESEARCH_METHODOLOGY.md`;
- C002, C003 and their reviews (`research/claims/reviews/`);
- Experiment 101's `README.md`, `preregistration.md`, the closure, the final report and the confirmation-record extract;
- the confirmation review's `REVIEW.md`, the calibration review's fraction tables, and the 101 design (revision 3, with
  its §14 decision);
- the literature review's A5, A7, A14 and B1–B12;
- 005's discovery report.

**Read in part:** 005's design and manifest (for E005-A3's sample), the lock review's N3, the research-state memo §4, and
the *uncommitted* `research/synthesis/exp101-critical-assessment-2026-10-14.md` and its post-hoc check output. Those last
two are context only; no finding depends on them.

## Verdict: **REGISTER WITH CHANGES**

- **Numbers.** Every number in the draft matches the record. I found no numeric error. There are seven descriptive or
  citation errors (§1).
- **Maturity.** `CAUSAL_EVIDENCE`, narrowly scoped, is supportable. The Level-3 gates as the methodology words them are
  met on confirmatory data:
  - a preregistered intervention and outcome (the lock and the prediction table, both committed by the user before any
    target prompt);
  - necessity (R, 36/36) and sufficiency (I, 36/36);
  - a matched control (the equal-length off-read slice) and negative controls (8 isotropic directions, the NOOP), plus D;
  - effect sizes with intervals, and alternatives considered.

  Three things are untested: a site or layer control, an axis estimated without the tested pair, and simpler baseline
  directions. Each limits specificity, generality or uniqueness, which the claim does not assert. They are Level-4
  tests, and the draft correctly leaves Level 4 unchecked. Registering lower would conflict with "Set current maturity to
  the highest level whose required gates remain satisfied" (METH:301), unless a Level-3 gate is judged unmet (§2.4).
- **Two blockers before registration.**
  - **B1.** Level 2's "plausible alternative locations" gate is marked `[~]`. The methodology has no such state, and
    as drafted no maturity above `OBSERVATIONAL` is available, including `LOCALIZED`. Recommended fix: check it, and say
    what each comparison rests on.
  - **B2.** The hypothesis sentence claims "the singular/plural information that the model uses for the noun-number
    choice", more than the four tested outcomes show.
- **Should-fix items.** Title and population; the description of how `w_c` was obtained; the supported conclusion,
  which mixes preregistered and descriptive results and drops label-scope qualifiers; missing non-claims; evidence types
  and missing evidence facts; the definitional component; and the relationship to C002 and C003. Two of C003's Level-3
  sentences no longer describe the record.
- **If the user rejects B1's fix** (the gate requires a prospective site/layer comparison of this coordinate), no
  maturity above `OBSERVATIONAL` is available. The verdict then becomes **DO NOT REGISTER YET**, pending such a
  comparison.

Citation shorthand: `METH:n` = `docs/RESEARCH_METHODOLOGY.md` line n; `D101:n` = the 101 design; `CL:n` = the closure;
`CR:n` = the confirmation review's `REVIEW.md`; `LIT:n` = the literature review; `C002:n`, `C003:n` = the claim files;
`draft:n` = the draft.

---

## 1. Numbers and citations (check 1)

### Verified against the record (all match)

| draft item | draft value | record | source |
|---|---|---|---|
| lock; lock commit | `8da97e35…`; `961e814` | `lock_sha256` `8da97e35…`; commit message "experiment 101 lock" | record extract; `git log` |
| table; table commit | `91918004…`; `59d8ec4` | `table_content_sha256` `9191800440af…`; commit "stage 1 - reference states and prediction table" | record `confirmation.barrier`; `git log` |
| report; final state | `54f883ad…`; `c803fe97…` | report file sha256 `54f883ad…`; `final_state_sha256` `c803fe97…`, the state after the report phase (`1e3eaeb3…` → `c803fe97…`) | `evidence/report-2026-10-14/README.md` |
| calibration; launcher | `1eb1fe68…`; `56a687f3…` | equal | record `calibration.content_sha256`; `confirm-2026-10-14/launcher_sha256_at_production.txt` |
| tokens; positions | `one` 581, `two` 767; `p_c` 4–6 | `cue_ids` sg 581, pl 767; frozen `p_c` is 4 in 10 frames, 5 in 22, 6 in 4 (the eligibility rule allowed 4–7, D101:375) | `confirmation-v1.json` |
| precondition; `Δc_full` | 36/36; −6.58 to −3.95 | 36/36; −6.581 to −3.947 | CR:41 |
| R | 36/36, median N 1.006 [0.988, 1.051] | 36/36, 1.00638 [0.98786, 1.05115] | report; CR:42 |
| I | 36/36, median S 1.035 [1.017, 1.059] | 36/36, 1.03503 [1.01720, 1.05920] | report; CR:43 |
| D | 28/36 and 36/36; median −0.136 | 28/36 (median −0.13596), 36/36 (median +0.92231) | record `results.D` |
| P | 33/36; frames 03, 08, 09; τ 0.0252 / 0.0351 | 33/36; 03, 08, 09; τ 0.025154 / 0.035111 | CR:46, 137–140 |
| fractions | `F_rem` −0.009, `F_rem_off` 1.000, `F_ins` 1.004, `F_ins_off` −0.033 | equal | CR:126–129; CL:20–22 |
| read share | ~14 % (8.6–18.9 %) | equal | CL:37–39; CR:215 |
| head shares | 0.882 / 0.894; value-only 0.877 / 0.892 | 0.8816 / 0.8943; 0.8774 / 0.8924 | CR:183 |
| isotropic | beaten 36/36 both ways | 36/36, margins 1.013 / 0.983 | CR:182 |
| stratum `three` | 36/36, 36/36, 28/36, 36/36 | equal | CR:180 |
| stratum `many` | 36/36, 36/36, `A_amp` < 0 23/36; `F_rem` 0.21; 15 % | equal (0.214; `F_swap_pl` 0.152) | CR:181 |
| calibration | `F_rem` −0.035, `F_ins` 0.980; N, S 36/36; `A_amp` < 0 30/36 | equal | calibration review REVIEW.md:367–373; `calibration-v1.json` shares |
| gates | G2 ≥ 181×, G3 137×, NOOP 36/36, `x3(p_t)` 1,692/1,692, Level-1 6.14e-3, G6 bit for bit, 1,980 keys, G1–G7 | equal; G7 is the accounting row of the confirm README's gate table | CR:88–109; `confirm-2026-10-14/README.md` |
| freeze timing | table pushed before the S2 ledger (17:38:22Z) | equal | CR:150–157 |
| D saturation | ΔT ×1.94 under NEC-AMP while `Δc` moves −0.136 | equal | CR:188, 262 |
| per-noun errors | 1.06 (SUF-READ@two), 1.20 (NEC-READ@many) | 1.059, 1.196 | CL:107, 128 |
| E005-A3 | `discovery-report…:29–36`; 0.956 at the layer-3 input, 0.125 at layer 4 | lines 29–36: L3 0.956, L4 0.125 (L = input to layer L, as D101:152–153 reads it) | 005 discovery report |
| thresholds | Holm 26/25 at 0.025; floors 0.05; D 25; P 27; precondition 30 | equal | `preregistration.md:21–31` |

### Errors and imprecisions found

1. **draft:179–180, "not trained on the target behavior": misleading.** `d̂_T` "is the mean plural-minus-singular
   change of the head's output over the exposed frames' clean pairs, `one`/`two` and `each`/`several`, so `w_c` was
   estimated from the very pair 101 edits" (D101:64–65). The direction is not optimized against intervention outcomes,
   but it is estimated on the target behavior. The draft's own Origin (draft:16) and Limitations (draft:214) say so
   correctly. → S2.
2. **draft:120, "8 isotropic directions (predicted null, observed null)": "predicted" is inexact.** The isotropic arms
   have no prediction in the program's table (D101:191–201). They carry a preregistered secondary statistic (D101:299–300)
   with no label force. "Expected null" is accurate. → S6.
3. **draft:163, "the natural remainder alone does nothing": overstated.** In the plural prompt, the remainder changes
   the fraction by a median −0.037 (range −0.254 to +0.043), toward singular in 29 of 36 frames (CR:219–220). That is
   near-zero, not zero, and it is the opposing component 010 described. → S11.
4. **draft:110 cites only E101-2 for the LayerNorm scale records.** E101-2 holds the gates. The LayerNorm records are in
   the confirmation review §5 (CR:189), which is E101-3, and in CL:72. → S5.
5. **draft:107–109 cites E101-4 for PC-SWAP.** E101-4 has no PC-SWAP value, and neither does any other evidence entry.
   `F_swap_pl` 0.044 and `F_swap_ref` 1.008 (CR:185) are cited in gates and competing explanations without an entry.
   → S5.
6. **draft:174, A5 "in GPT-2-class models": inaccurate.** The literature review says "GPT-2, Transformer-XL and XLNet"
   (LIT:416). A14 omits that it concerns "a two-layer LSTM language model" (LIT:509). A7 omits "for verb and reflexive
   agreement" (LIT:440). → N8.
7. **draft:165 and draft:210, "the proposed 102": no such design exists.** There is none in `docs/superpowers/specs/`
   and no committed file on this branch. → N2.

---

## 2. The proposed maturity `CAUSAL_EVIDENCE` (check 2)

### 2.1 Levels 0 and 1: met

- **Level 0.** The quantities and criteria were frozen in the lock, and every gate held with margins from 3.26× upward
  (CR:88–109). The inputs, tokens and positions are recorded (`confirmation-v1.json`), and so are the model revision
  and the runtime.
- **Level 1.** Direct measurement in 36 frames with per-frame values, labelled confirmatory.
- **Missing items.** The draft does not list three of the ladder's items. They are present in C002 and C003. → S10.
  - "failure behavior … documented" (METH:59), which is the incident procedure: `README.md:81–82`,
    `preregistration.md:61`;
  - "Repeated or related cases are examined" (METH:68), which is the stratum and the calibration;
  - "The conclusion avoids causal interpretation" (METH:70).

### 2.2 Level 2: met on the evidence, but marked `[~]` (B1)

The draft marks "Plausible alternative locations compared" `[~]` (draft:107–109). The methodology requires that "All
required gates through a maturity level must be explicitly satisfied in the claim file before the claim moves to that
level" (METH:50). It offers checked, unchecked, or not applicable with justification. As drafted, Level 2 is not
satisfied, so neither `LOCALIZED` nor `CAUSAL_EVIDENCE` is available.

**Is the gate actually satisfiable? Yes, in my judgment, with each comparison's status stated.** For a coordinate at a
site, the plausible alternative locations and the evidence that compares them are:

| alternative location | evidence | status |
|---|---|---|
| other directions at the same site | the read part against the cue's own off-read slice (R, I); 8 isotropic directions; the whole remainder (NEC-READ −0.009 against PC-SWAP-PL 0.044) | R and I confirmatory and label-bearing; the rest preregistered secondary, on the same run |
| other positions at the layer-3 input | PC-SWAP: the whole `x(p_c)` carries all but a median 0.044 [0.010, 0.061] of the `two` shift (CR:185) | secondary, descriptive, 36 new frames |
| other readers of the site | the head-only arms: about 88–89 % of the read edit's effect passes through `L03.H04`'s output at `p_t` (CR:183) | secondary, descriptive, 36 new frames |
| other layers at `p_c` | 005's A3 whole-residual profile: 0.956 at the layer-3 input, 0.125 at the layer-4 input | **exploratory and thin** (see below) |

**The project's precedent supports checking it.** C002 checks the same gate on 005's exploratory Tier A alone
(C002:92). Level 2 does not require confirmatory evidence (METH:72–78).

**But E005-A3 is thinner than the draft shows.** 005's coordinated development cases are 2 frames, `Lena and Omar
displayed {cue} small` and `Mira and Noah packed {cue} bright`, with `one`/`two` and 20 development nouns (40 cases;
`screening/behavior-candidates/manifest-v1.json`). The patch was the whole residual, not the read coordinate. → S5.

The fix and its exact text are in B1. **The alternative reading:** the user may require a prospective comparison of
*this coordinate* across layers or positions. Then the gate is unmet, no maturity above `OBSERVATIONAL` is available,
and registration should wait for such a comparison.

The other Level-2 items:
- **"Replicates within scope"** holds, with one caveat. "Some exposed frames entered 012's bases and `d̂_T`"
  (D101:228), so the calibration (E101-7) is partly in-sample for the axis and is not independent replication. → S5.
- **"Localization is not presented as proof of causality"** (METH:78) is not listed. → S10.

### 2.3 Level 3: met within the narrow scope

| METH gate | evidence | assessment |
|---|---|---|
| intervention and outcome documented before the result (METH:82) | the lock (`961e814`) and the table (`59d8ec4`), committed by the user and pushed before the S2 ledger; no tolerance or threshold differs from the lock (CR:150–165) | met, strongly |
| necessity (METH:83) | R: `N_f` > 0 in 36/36 against the matched control | met |
| sufficiency (METH:84) | I: `S_f` > 0 in 36/36 | met |
| matched, negative and other appropriate controls (METH:85) | matched: the equal-length slice of the cue's own centred off-read change, the demanding control (D101:427–430); negative: 8 isotropic directions and the NOOP identity; direction: D | met for the site-scoped claim (see 2.4) |
| effect sizes and uncertainty (METH:86) | medians with frame-bootstrap intervals; exact tails | met |
| alternatives considered (METH:87) | the closure's table (CL:67–80), CR §6.3, the draft's table | met; S11 adds two rows |

### 2.4 The three contested points

**(a) No site or layer negative control.**
- **What it would test.** A site control asks whether the same edit has the effect somewhere else: at `RESID_PRE.L4`,
  at an earlier layer, or at another position. That is a *specificity* question.
- **What the claim asserts.** Only what editing this coordinate at this site does. It does not assert that nothing else
  does.
- **Where the gate is already covered.** The matched and isotropic controls test the claim's own contrast at its own
  site. The main worry a site control would address is that `w_c` is a generic number direction that any reader picks
  up. The head arms answer most of that descriptively: 88–89 % of the effect runs through `L03.H04`'s output.
- **My reading.** The absence is compatible with Level 3 provided the claim disclaims site specificity (S4) and the title
  does not imply layer-3 consumption beyond the evidence (S1).
- **The other reading.** If the user reads "other appropriate controls" (METH:85) as requiring a site control for any
  claim that names a site, Level 3 is not met. The claim would then sit at `LOCALIZED`, with B1 still to resolve.

**(b) The in-sample axis.**
- **Why it is compatible.** It is compatible with Level 3 within the defining pair: the frames are held-out data,
  which is what METH:169 asks for, and the label's own scope is "the cue pair one/two from which the read was
  estimated" (`preregistration.md:47`).
- **What it limits.** Generality (Level 5) and informativeness, not the validity of the causal test.
- **What it requires.** The supported conclusion must carry the qualifier (S3). B12's point also applies: success on
  genuinely new cues is more diagnostic than on held-out data from the same pool (LIT:761–763). Only `three` speaks to
  this, descriptively.

**(c) Simpler baseline directions are untested.**
- **Why it is compatible.** The claim disclaims uniqueness and optimality (draft:206). `w_c` was not optimized against
  intervention outcomes, so A7's selectivity worry, that DAS "can produce effects for arbitrary mappings"
  (LIT:437–438), applies less.
- **Where it belongs.** These are Level-4 discriminating tests (METH:97), correctly unchecked.

### 2.5 Recommendation

**Register at `CAUSAL_EVIDENCE`, `ACTIVE`, after B1 and B2 and with the should-fix items.**
- **Why not lower.** METH:301 sets maturity at the highest level whose gates are met. `LOCALIZED` would need a
  Level-3 gate judged unmet, and it would still need B1 resolved.
- **Why not defer.** The methodology does not make Level-4 tests a precondition for Level 3. Claims can be narrowed,
  contested or downgraded later (METH:302–306). The design anticipated exactly this review: "A claim scoped to that
  stage could then be reviewed for higher maturity. That review is a separate decision." (D101:40–41)
- **What lowers informativeness without failing a gate.** These must be stated (S3, S7):
  - the definitional part of R and I for the head route;
  - a pass the exposed calibration had made near-certain (36/36 there);
  - D is an extrapolation test with a 3-frame margin;
  - one template, one run, AI-only reviews.

---

## 3. Scope and wording (check 3)

- **Overclaims.**
  - The hypothesis says "carries the singular/plural information that the model uses for the noun-number choice"
    (draft:27–29). → B2.
  - The title says "causally carries … in new coordinated frames" (draft:1). → S1.
  - The draft cites A7's caution that "an intervention's efficacy does not show that the model uses the variable"
    (LIT:440–441) and then uses "the model uses" in the hypothesis.
  - The edits produce composite states the model never produces (D101:455–457), and D's targets lie beyond the natural
    `r` values.
  - "carries" is acceptable once defined interventionally; "uses … information … choice" is not.
- **Implied generality.**
  - "in new coordinated frames" reads as a population.
  - The lock review asked that "A later claim review should state which population it reads" (lock review
    `REVIEW.md:532–535`). The label's non-claim says "these frames".
  - The draft's supported conclusion gets this right; the title and hypothesis do not. → S1, B2.
- **Exposed nouns.** Stated correctly in Scope and the conclusion. The outcome is a noun mean, and per-noun effects of
  the edits are not reported. → S4.
- **In-sample axis.** Stated in Origin, Competing explanations and Limitations, but missing from the supported
  conclusion and contradicted by "Nearest difference". → S2, S3.
- **One template.** In Scope. The conclusion says "the 36 frozen coordinated frames" without "of one template". → S3.
- **The definitional component.**
  - It is in the competing-explanations table (draft:164), but absent from the supported conclusion, the novelty
    statement and the limitations.
  - `w_c` is the covector through which `L03.H04` writes onto `d̂_T`. Any edit orthogonal to `w_c` (the matched and
    isotropic controls) leaves the head's `d̂_T` output unchanged to first order. The manipulation check confirms it:
    NEC-OFF 1.026, SUF-OFF 0.003, NEC-ISO 0.99–1.00, SUF-ISO ≤ 0.016 of PL's ΔT (CR:188).
  - So for the head route the read/off-read contrast is partly guaranteed. Given C002's head share (0.816) and 009/011's
    linear read-out, the research-state memo judged that "the primary label adds little information at the margin"
    (research-state memo:411–418).
  - The empirical content is that the remainder is near-inert through every reader, that the head is the bottleneck for
    88–89 %, and that the effect is complete (CR:264). → S7.
- **CausalGym.**
  - "Not new: that number agreement has low-dimensional causal directions in small LMs (A7)" (draft:187–188) is right,
    but should say verb and reflexive agreement.
  - The "Nearest difference" sentence currently overstates the distinction (item 1 of §1). → S2, N8.

---

## 4. Missing contradicting evidence and limitations (check 4)

| item | in the draft? | action |
|---|---|---|
| `many` would fail D (23/36), R and I weaker | yes (draft:147–148, 205) | keep. Also record that `many`'s D count fell from 26/36 in the exposed calibration (calibration review REVIEW.md:379–380) to 23/36. → N7 |
| D's 3-frame margin; saturation | yes (draft:149–150) | soften "limited by saturation" to "attributed in the record to saturation". D's edits also leave the natural range of `r` (D101:455–457). → S7, N6 |
| P's 3 disagreements; per-noun errors about 1 nat; P close to a recomputation | errors yes; near-recomputation only implicit | add design §5's sentence ("an agreement is close to a plain recomputation of the layers", D101:215) to the conclusion. → S3 |
| not the only causal route: `F_swap_pl` 0.044 (`two`), 0.152 (`many`) | **no** | the closure's first non-claim (CL:26). → S4 |
| site or layer specificity untested | in Limitations and Level 4, not in *Not supported* | → S4 |
| per-noun effects of the edits not reported | **no** | → S4 |
| the remainder's small opposing effect (−0.037; 29/36 toward singular; range to −0.254) | **no** (draft:163 says "does nothing") | → S11 |
| calibration partly in-sample for `d̂_T` | **no** | → S5 |
| E005-A3's two-frame sample | **no** | → S5 |
| the pass was expected (calibration 36/36) | **no** | → S7 |
| outputs other than `c` (behavioral specificity of the edits) not reported | **no** | → N10 |
| self-repair (D101:458–459; LIT:338) | **no** | not a threat to a complete removal; add a row. → S11 |
| one run; AI-only reviews; no human expert | yes (draft:217) | keep |
| evidence not on `main` | **no** | → N9 |

---

## 5. Consistency with C002 and C003 (check 5)

**No silent upgrade in the draft as written.** Its relationship statements are accurate: C002 localizes; C003 is about
prospective prediction by decoded programs. "This claim sits beside both and changes neither" (draft:19) is too thin
for what registration would leave behind. → S8.

- **C002.**
  - **The overlap.** 101's head-only arms are interventional evidence that `L03.H04`'s output carries 88–89 % of the
    read edit's effect. A reader could take that as causal status for C002's stage T.
  - **Why it does not count for C002.** C002's Level-3 gates concern its own components and families on its templates
    and noun splits, and they are unchecked (C002:101–106). 101 edits a direction in the residual stream, not C002's
    components; the research-state memo says so too (memo:375–376). Its head arms are secondary, in 36 frames of one
    template for one pair.
  - **Not a replication.** The 88–89 % share is a different measure from C002's 0.816 head-alone recovery.
  - **What C004 must say.** That it does not satisfy C002's Level-3 gates.
- **C003.**
  - **Dated statements.** Two Level-3 sentences in C003 describe the record as of 2026-09-28. They are "No preregistered,
    outcome-bearing necessity test in the model targeted a decoded variable" (C003:259–260) and "No decoded variable was
    itself manipulated and shown sufficient in a preregistered, outcome-bearing test" (C003:262–263). After 101 they
    are no longer true of the record. 101's README calls it "the causal test of claim C003".
  - **What stays unchanged.** C004 does not lift C003's Level 3, which concerns the programs' variables as a whole
    (C003:396).
  - **What 101 adds.** P is a prospective prediction of a *new* intervention, which bears on C003's Level-6 criterion
    "only partly" met (C003:295–297). By design it is close to a recomputation (D101:212–216), which also bears on
    C003's competing explanation that "Much of the accuracy comes from recomputing exact layers" (C003:331).
  - **What C004 must say.** That recording 101 in C003 needs C003's own review, and that C003 stays `LOCALIZED` until
    then.
- **The registry.** `research/claims/README.md` gains a C004 line on registration.
  - As with C003, a cross-reference from C002 and C003 to C004 is a separate decision (C003 registration review,
    lines 103–105).
  - The C003 registration procedure is the model to follow: scripted edits, an independent check, every change listed.

---

## 6. The supported conclusion (check 6)

The draft's conclusion (draft:192–199) is accurate number by number, but it has four problems:
- it presents descriptive magnitudes ("essentially the whole") as if they were what was preregistered; the preregistered
  criteria were sign counts against the matched control and median floors of 0.05 (`preregistration.md:21–29`);
- it drops the label's qualifier "from which the read was estimated" and "of one coordinated template";
- it does not say that P is close to a recomputation, or that D is narrow and extrapolative;
- it omits the definitional component and the closure's non-claim that this is not the only route.

**Proposed exact wording:**

> Within pinned Pythia-70M-deduped (revision `e93a9faa`, float32 on CPU), the scope is:
> - the 36 frozen new frames of Experiment 101, all of one coordinated template (`NAME and NAME VERBED {cue} ADJ`);
> - the cue pair `one`/`two`, from which the read direction was estimated;
> - the mean contrast over the 79 exposed scorable nouns;
> - exact replacement edits of the residual entering layer 3 at the cue position.
>
> In that scope, one preregistered confirmation run supports the following.
> - **Removal and installation (R and I, preregistered).** Setting the `two` prompt's read coordinate `⟨x, w_c⟩` to the
>   `one` prompt's value removed more of the plural shift than an equal-length edit along the cue's own centred
>   off-read change, in 36 of 36 frames. Setting the `one` prompt's coordinate to the `two` value installed more of it,
>   in 36 of 36. Descriptively, removal left a median −0.009 of the shift (per frame −0.170 to +0.083), and
>   installation produced 1.004 (0.814 to 1.069). The control left 1.000 and produced −0.033.
> - **Direction (D, preregistered).** Reversing the read change made the `one` prompt more singular in 36 of 36 frames
>   (median +0.922 nats). Doubling it made the `two` prompt more plural in 28 of 36, three frames above the threshold
>   (median −0.136 nats). Both edits move the coordinate beyond the natural values the swaps use.
> - **The frozen program (P, preregistered).** Its table, committed before any target prompt, matched the measured
>   removal and installation fractions within the calibrated tolerances in 33 of 36 frames. With exact upstream inputs
>   this is close to a recomputation of the frozen readout (design §5). It checks the readout's fidelity under these
>   edits in noun-mean fractions, not per noun.
> - **Descriptive, with no label force.**
>   - The read part is about 14 % of the cue change's centred squared length.
>   - The rest of the cue change has a near-zero net effect: a median −0.037 of the shift in the `two` prompt.
>   - About 88–89 % of the read edit's effect passes through `L03.H04`'s output at the prediction position, and the rest
>     through other readers of the residual.
>
> For `L03.H04`'s route, the read/off-read contrast follows in part from how `w_c` is defined. The coordinate is not
> the only route: with the whole cue-position residual swapped back, a median 0.044 of the `two` shift remains. Site,
> layer, held-out-axis and baseline challenges are untested. Nothing here extends beyond these frames, this pair,
> template, site, nouns and checkpoint.

---

## 7. Findings

### Blockers (fix before registration)

**B1. Level 2's "plausible alternative locations" gate is `[~]`, a state the methodology does not allow.**
- **The rule.** Gates must be "explicitly satisfied" (METH:50). C002 and C003 use only `[x]` and `[ ]`. As drafted,
  Level 2 fails, and with it both `LOCALIZED` and `CAUSAL_EVIDENCE`.
- **The fix.** Check the gate, with the status of each comparison stated (§2.2). This follows C002's precedent
  (C002:92). If the user does not accept it, the verdict is DO NOT REGISTER YET.
- Old (draft:107–109):
  > - [~] Plausible alternative locations compared — **exploratory only**: 005's residual profile places the cue's
  >   consumption at layer 3 (E005-A3); 101 compares the read part with the whole position (PC-SWAP) and with the off-read
  >   remainder, but tests no other layer or position prospectively. Evidence: E005-A3, E101-4.
- New:
  > - [x] Plausible alternative locations compared. Evidence: E101-1, E101-4, E101-5, E005-A3.
  >   - Within the site, on the 36 new frames: the read part against the cue's own off-read change, the primary
  >     control (E101-1, E101-4), and against 8 isotropic directions (E101-5).
  >   - Across positions at the layer-3 input: the whole `x(p_c)` carries all but a median 0.044 of the `two` shift
  >     (PC-SWAP, E101-5).
  >   - Across readers of the site: about 88–89 % of the read edit's effect passes through `L03.H04`'s output at `p_t`
  >     (E101-5).
  >   - Across layers at `p_c`, exploratory only: 005's whole-residual profile on two exposed coordinated frames
  >     recovers 0.956 at the layer-3 input and 0.125 at the layer-4 input (E005-A3).
  >   - The same read edit at another layer or position was not tested (Level 4).

**B2. The hypothesis states more than was tested.**
- **What it claims.** "the singular/plural information" (only the `one`→`two` shift was tested); "that the model
  uses" (a claim about the natural computation from edits that produce composite states, against A7's caution the
  draft itself cites); "the noun-number choice" (the outcome is a noun-mean log-probability contrast).
- **Why it blocks.** Registration freezes the hypothesis, as it did for C003.
- Old (draft:27–29):
  > of the residual `x` entering layer 3 (`RESID_PRE.L3`) at the cue position of a coordinated frame carries the
  > singular/plural information that the model uses for the noun-number choice at the prediction position, for the cue pair
  > `one`/`two`:
- New:
  > of the residual `x` entering layer 3 (`RESID_PRE.L3`) at the cue position carries, in the 36 frames of Experiment
  > 101, the `one`→`two` shift of the noun-mean contrast `c` at the prediction position, in this interventional sense:
- Old (draft:30–31): "removes the plural shift of the noun-number contrast;"
- New: "removes the plural shift of the noun-mean contrast `c`;"
- Add after the bullets (draft:35):
  > The preregistered criteria (*Predictions*) are relative to the matched control. That removal and installation are
  > essentially complete is a descriptive observation of the same run.

### Should-fix

**S1. The title: population, and the head's role.**
- **The problems.** "in new coordinated frames" reads as a population (lock review N3(a)). "The read coordinate of
  `L03.H04`" can be read as the head's own channel, but a head-specific statement "remains a separate, secondary
  conclusion" (D101:519–521).
- Old (draft:1, registered form):
  > The read coordinate of `L03.H04` at the cue position causally carries the one/two plural shift in new coordinated frames of pinned Pythia-70M
- New:
  > Setting the `L03.H04` read coordinate of the residual entering layer 3 at the cue position removes and installs the `one`→`two` plural shift in 36 new coordinated frames of pinned Pythia-70M
- Add to *Scope*:
  > - **Population:** these 36 frames, the first 36 eligible candidates of one hand-written list, chosen by the tokenizer
  >   only. No population of frames, templates or cue pairs is inferred; this answers the lock review's N3(a).

**S2. "Nearest difference" and the novelty statement misdescribe how `w_c` was obtained.** (D101:64–65; §1 item 1)
- Old (draft:179–181):
  > - **Nearest difference:** here the direction is derived from a circuit component's weights (not trained on the target
  >   behavior), tested with complementary arms (read part vs the cue's own remainder) and predicted by a frozen program
  >   committed before the target data.
- New:
  > - **Nearest difference:**
  >   - **The direction.** It is the read covector of a localized head: `L03.H04`'s value–output weights and layer-3
  >     LayerNorm gain, applied to a head-output axis `d̂_T`. That axis is a difference of means on exposed `one`/`two`
  >     and `each`/`several` pairs (design §2). So the direction is estimated on the tested pair, but it is not optimized
  >     against intervention outcomes, as DAS is (B2, A7).
  >   - **The test.** Complementary arms set the read part against the cue's own remainder, which checks the components
  >     the variable leaves out (B12).
  >   - **The prediction.** A frozen program committed before the target data predicted the fractions, close to a
  >     recomputation of the frozen readout (design §5).
- Old (draft:185–187): "a weight-derived, one-dimensional read of a localized head that carries the whole tested effect
  at its site while the rest of the cue's change is inert"
- New: "a one-dimensional read of a localized head, derived from its weights and a difference-of-means output axis
  estimated on the same cue pair, which carries essentially the whole tested effect at its site while the rest of the
  cue's change is near-inert"

**S3. Replace the supported conclusion (draft:192–199) with the wording in §6.**
- **Why.** It separates the preregistered criteria from the descriptive magnitudes. It restores the label's qualifiers
  ("from which the read direction was estimated", "of one coordinated template"). It adds design §5's
  near-recomputation, D's margin and extrapolation, the definitional component, and the not-the-only-route statement.
- **Also the Origin line** (draft:13): add "confirmatory for R, I, D and P as preregistered; the magnitudes and head
  shares are descriptive records of the same run, computed after the label".

**S4. *Not supported* omits four items.** Add:
> - That the read coordinate is the only causal route (the closure's non-claim, closure:26). With the whole `x(p_c)`
>   swapped back, a median 0.044 of the `two` shift remains, and 0.152 for `many`.
> - That the effect is specific to this site or layer. The same edit at `RESID_PRE.L4`, at earlier layers or at another
>   position was not tested. Consumption at layer 3 rests on 005's exploratory two-frame profile and the head arms'
>   88–89 % share.
> - Per-noun effects of the edits: the outcome is the mean over 79 nouns, and per-noun removal and installation are not
>   reported.
> - That C002's Level-3 gates, or C003's causal gates for its programs' variables, are met (see *Relationship to C002
>   and C003*).

**S5. Evidence entries: types, missing facts and samples.**
- **Types** must be "supporting, contradicting, null, or methodological" (METH:156):
  - E101-1, E101-4, E101-5 and E101-7 are supporting, with E101-7 exploratory;
  - E101-2, E101-3 and E101-8 are methodological;
  - E005-A3 is supporting (exploratory);
  - E101-6 is either methodological (scope-limiting), following the C002 decision of 2026-09-27 for evidence beyond the
    declared cue scope (C002:69), or contradicting if the user prefers it in that list. It must be one of the four.
- **Add the facts the gates cite** to E101-5 (or a new entry citing CR:180–189):
  - PC-SWAP: `F_swap_pl` 0.044 [0.010, 0.061], `F_swap_ref` 1.008 [0.989, 1.021];
  - additivity 0.989; half doses 0.703 and 0.717;
  - the remainder: −0.037, toward singular in 29/36 frames; +0.008 on the reference side;
  - the manipulation check: NEC-AMP ×1.943, NEC-OFF 1.026, SUF-OFF 0.003;
  - LayerNorm: 7.44 read, 7.95 off-read, 7.32 isotropic.
- **Draft:110.** Cite E101-3 for the LayerNorm records.
- **E005-A3, old result:** "whole-residual patch at `p_c` (coordinated): recovery 0.956 at the layer-3 input, 0.125 at
  the layer-4 input".
- **E005-A3, new result:**
  > whole-residual patch at `p_c` on 005's two coordinated development frames (`Lena and Omar displayed {cue} small`, `Mira and Noah packed {cue} bright`; `one`/`two`; 20 development nouns; exploratory): recovery 0.956 at the layer-3 input, 0.125 at the layer-4 input; the whole residual, not the read coordinate
  - and set its revision column to code `83d1ae4`.
- **E101-7:** append "some exposed frames entered 012's bases and `d̂_T` (design §6), so this is partly in-sample for
  the axis, not independent replication".
- **Revision/digest "—" for E101-4 to E101-6:** give the closure commit `a9b6b33`.
- **E101-3:** give the REVIEW.md sha256, `b13dff8f…`.

**S6. The Level-3 controls gate (draft:119–122).**
- **What it misses.** It states no prediction status. It does not say that both controls are orthogonal to `w_c`, or
  how the gate is read without a site control.
- Old:
  > - [x] Matched and negative controls: equal-length edits along the cue's own centred off-read change (the primary
  >   matched control) and 8 isotropic directions (predicted null, observed null); NOOP; D's doubling and reversal
  >   distinguish a read effect from disruption. **Not included: a site or layer negative control** (recorded under
  >   Limitations and Level 4). Evidence: E101-1, E101-5.
- New:
  > - [x] Matched, negative and other controls. Evidence: E101-1, E101-2, E101-5.
  >   - **Matched** (primary, label-bearing): equal-length edits along the cue's own centred off-read change. They leave
  >     a median 1.000 of the shift and install −0.033.
  >   - **Negative** (secondary, no label force): 8 isotropic directions orthogonal to `w_c`, expected null and observed
  >     null. The read edits beat their median in 36/36 frames both ways. The NOOP identity holds bit for bit (G3).
  >   - **Direction** (label-bearing): D's doubling and reversal separate a signed read effect from disruption.
  >   - **What the controls can show.** Both the matched and the isotropic controls are orthogonal to `w_c` by
  >     construction. For `L03.H04`'s route their null is therefore partly definitional; their empirical content is
  >     the null through every other reader.
  >   - **Not included: a site or layer negative control.** The claim makes no statement that the effect is specific
  >     to this site (*Not supported*), so the gate is read as met for the site-scoped claim.

**S7. Limitations: add the definitional component, the expected pass and D's extrapolation.** Add:
> - **The definitional component.** For `L03.H04`'s route, part of the read/off-read contrast follows from how `w_c` is
>   defined. An edit orthogonal to `w_c` leaves the head's `d̂_T` output unchanged to first order: NEC-OFF 1.026 and
>   SUF-OFF 0.003 of PL's ΔT. The definition does not imply three things: that the remainder is near-inert through
>   every reader, that the head's output carries 88–89 %, and that the effect is complete.
> - **An expected pass.** The exposed calibration already showed N, S > 0 in 36/36 frames (E101-7). The confirmation's
>   value is confirmatory status on new frames, not a risky prediction borne out.
> - **D extrapolates.** NEC-AMP and SUF-NEG move `r` beyond the natural values the swaps use (design §13). D passed by
>   3 frames, and the record attributes the shortfall to saturation.

**S8. Add a *Relationship to C002 and C003* section, modeled on C003's *Relationship to C002*.**
- Old (draft:18–19): "This claim sits beside both and changes neither."
- New:
  > This claim sits beside both and changes neither (see *Relationship to C002 and C003*).
- The new section:
  > - **C002 is unchanged.** 101 edits a direction in the residual stream, not C002's components. Its head-only arms are
  >   secondary and descriptive, in 36 frames of one template for one pair. They do not satisfy C002's Level-3 gates,
  >   which concern C002's own families on its templates and noun splits. The 88–89 % head share is a different measure
  >   from C002's 0.816 head-alone recovery and does not replicate it.
  > - **C003 is unchanged.**
  >   - **What this claim covers.** Causal evidence for one decoded variable, the P1 read coordinate at one site, for one
  >     pair in one template. It does not establish that the programs' variables are the model's causal variables
  >     (C003, *Not supported*).
  >   - **Dated statements.** Two of C003's Level-3 sentences no longer describe the record: "No preregistered,
  >     outcome-bearing necessity test in the model targeted a decoded variable" and "No decoded variable was itself
  >     manipulated and shown sufficient in a preregistered, outcome-bearing test".
  >   - **The follow-up.** Recording 101 in C003 needs C003's own review. It would cover the bearing on one stage's causal
  >     gates, and P's bearing on Level 6's "predicts outcomes of new interventions", qualified by design §5's
  >     near-recomputation. C003 stays `LOCALIZED` until that review.

**S9. The hypothesis bullets:** done with B2's added sentence. No further change.

**S10. Add the ladder items the draft omits** (C002 and C003 list them):
- **Level 0:** "Numerical validation, tolerances and failure behavior documented". Evidence: E101-2, and the incident
  procedure (`README.md:81–82`; `preregistration.md:61`).
- **Level 1:** "Repeated or related cases examined". Evidence: E101-6, E101-7.
- **Level 1:** "The conclusion avoids causal interpretation at this level". Evidence: this record.
- **Level 2:** "Not presented as proof of causality". Evidence: this record.

**S11. Competing explanations (draft:163 and new rows).**
- Old (B8 row): "the natural remainder alone does nothing"
- New: "the natural remainder has a near-zero net effect (median −0.037 of the shift in the plural prompt, toward
  singular in 29/36 frames, range −0.254 to +0.043; +0.008 in the reference prompt)"
- In the same row, add:
  - **What is met.** B8's own recommendation of a circuit-derived direction (LIT:695) is met.
  - **What is not run.** Rowspace and nullspace checks.
  - **What remains open.** The 11–12 % through other readers could include dormant pathways.
- Add a row: "**Self-repair** (design §13) | not evident: removal is complete (`F_rem` −0.009) | —".
- Add a row: "**The `p_t` route** | 0.044 of the `two` shift (0.152 of `many`'s) bypasses `x(p_c)` at layer 3
  (PC-SWAP-PL); not edited by design | an edit at `p_t`".

### Notes

- **N1.** "G1–G7 held" (draft:78) is accurate. The record's gate book names G2–G6; G1 is the geometry checked at the
  lock, stage 1 and the barrier, and G7 is the accounting. Cite the confirm README's gate table.
- **N2.** "proposed Experiment 102" (draft:165, 210): no 102 design exists in the repository. Write "a proposed next
  experiment (not yet designed)", or cite it once it is committed.
- **N3.** The identifiers `E101-1…` and `E005-A3` are local and stable, which is acceptable. C002 and C003 use
  `E001…`; consider matching them.
- **N4.** Some facts appear only in the uncommitted `research/synthesis/exp101-critical-assessment-2026-10-14.md` and
  its post-hoc output: D's targets lie outside the natural `r` range in 36/36 frames, and the cosines of `ŵ` with
  data-derived directions are 0.41–0.43. Commit those files and label them exploratory, post hoc, before citing them.
- **N5.** The *Scope* interventions bullet (draft:48–49) omits the head-only arms, which replace `L03.H04`'s output at
  `p_t` and on which E101-5 rests. Add them as secondary.
- **N6.** "limited by saturation" (draft:149) → "attributed in the record to saturation". Extrapolation beyond the
  natural range of `r` is a second reading.
- **N7.** Next to E101-6, record that `many`'s `A_amp` < 0 count was 26/36 in the exposed calibration and 23/36 in the
  confirmation.
- **N8.** Prior-work descriptors:
  - A5: "GPT-2, Transformer-XL and XLNet" (LIT:416), not "GPT-2-class". A5 also found head results inconsistent under
    number swaps (LIT:417–418), a contrast worth stating.
  - A14: "in a two-layer LSTM language model" (LIT:509).
  - A7: "for verb and reflexive agreement" (LIT:440), with its selectivity result (LIT:437–438).
  - Consider citing B12: faithfulness on unmapped variables (LIT:757–759), which the near-inert remainder addresses, and
    new cues being more diagnostic than held-out data from the same pool (LIT:761–763), which the in-sample axis
    leaves open.
- **N9.** The evidence is on `origin/exp101-phases` (`a9b6b33`), not on `main` (`96f0cc8`). Merge before registering
  in the `main` registry, or say on which branch the evidence lives.
- **N10.** The edits' effects on outputs other than `c` are not reported, for example the rest of the next-token
  distribution. This is a limitation, not a gate issue.
- **N11.** Every review cited is an AI review, and so is this one. Keep the draft's statement (draft:5, 217) in the
  registered file and the status-history row.

---

## Appendix: why not REGISTER AT A LOWER MATURITY or DO NOT REGISTER YET

- **`LOCALIZED` with causal evidence recorded** is what C002 and C003 did when their Level-3 gates lacked confirmatory
  data. Here the necessity, sufficiency and control gates have confirmatory data. Holding the claim at `LOCALIZED`
  would need a reason to call one of them unmet, and the only candidate is the site control (§2.4a). Even
  `LOCALIZED` requires B1's gate.
- **Deferral until a 102-style experiment** would make Level-4 tests (site, layer, held-out axis, baselines) a condition
  of Level 3. The methodology does not, and it expects claims to be narrowed or downgraded when later evidence requires
  it (METH:302–306).
- **The case for caution is real, and it belongs in the record's wording, not its level:**
  - the definitional component;
  - an expected pass;
  - D's narrow, extrapolative margin;
  - one template, one pair, one run and AI-only review.

  S3 and S7 put it there.
