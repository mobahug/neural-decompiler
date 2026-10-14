# Experiment 101 — preregistration

- Lock `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061`; run `30ca8697c8702496` at `3621eec4b6a901119a4b96d4fdc07fa3b19652ab`; design revision 3 (`a87e284`, decided `516df20`), plan revision 3 (`96f0cc8`, G4 decided `1e48c1c`)
- The analysis module `src/neural_decompiler/read_intervention.py` at blob `d7db6197140fe871da591b5c578cc675f05f7d69`; configuration `production`
- The freeze `ef35650678e98a638838af10b75f340cec331f542589d6f089100eee8822624a`; manifest `7c7c94942dc883505b97403042e095d8d616168f3a0a1d8cf495edef25f8b8bd` (36 S1-REF keys, 1980 S2-TARGET keys, 55 runs per frame)
- The calibration `1eb1fe68ac2d49228ff8eff6c72391cc0695eaeeb3ae82ed122f9aa738f42e1e`: τ_rem 0.025153865322968608, τ_ins 0.03511072680382421; P evaluable True

## The variable

- `w_c` = the centred 011 read weight (digest `253a4733d71243bb003c320cfe7fe0b183149294caee7bb2451c1be7c7a08bd8`, ‖w_c‖ 0.7659205190757624); the edit at RESID_PRE.L3, p_c
- The isotropic controls: 8 per frame, tag `101|iso|{frame_id}|{j}`
- û_off: per pair at run time, from the plain runs of the same frame: Δx = x_w − x_ref at RESID_PRE.L3, p_c; Δr = ⟨Δx, w_c⟩; read = (Δr/⟨w_c, w_c⟩)·w_c; off_c = (Δx − read) − mean(Δx − read)·𝟙; û_off = off_c/‖off_c‖; s = |Δr|/‖w_c‖ (design §2, §14 D2)
- The cue order two, three, many; the run order per frame: REF, PL@two, NOOP@two, NEC-READ@two, SUF-READ@two, NEC-OFF@two, SUF-OFF@two, NEC-AMP@two, SUF-NEG@two, NEC-HALF@two, SUF-HALF@two, PC-SWAP-PL@two, PC-SWAP-REF@two, NEC-ISO-1@two, NEC-ISO-2@two, NEC-ISO-3@two, NEC-ISO-4@two, NEC-ISO-5@two, NEC-ISO-6@two, NEC-ISO-7@two, NEC-ISO-8@two, SUF-ISO-1@two, SUF-ISO-2@two, SUF-ISO-3@two, SUF-ISO-4@two, SUF-ISO-5@two, SUF-ISO-6@two, SUF-ISO-7@two, SUF-ISO-8@two, HEAD-NEC@two, HEAD-SUF@two, HEAD-NEC-V@two, HEAD-SUF-V@two, PL@three, NEC-READ@three, SUF-READ@three, NEC-OFF@three, SUF-OFF@three, NEC-AMP@three, SUF-NEG@three, NEC-HALF@three, SUF-HALF@three, PC-SWAP-PL@three, PC-SWAP-REF@three, PL@many, NEC-READ@many, SUF-READ@many, NEC-OFF@many, SUF-OFF@many, NEC-AMP@many, SUF-NEG@many, NEC-HALF@many, SUF-HALF@many, PC-SWAP-PL@many, PC-SWAP-REF@many

## The statistics

- **fractions**: F_rem = Δc(NEC-READ)/Δc_full, F_rem_off = Δc(NEC-OFF)/Δc_full, F_ins = Δc(SUF-READ)/Δc_full, F_ins_off = Δc(SUF-OFF)/Δc_full, A_amp = Δc(NEC-AMP) − Δc_full, A_neg = Δc(SUF-NEG); a weak frame's fractions are not computed at confirm
- **median**: the middle value for odd n; the mean of the n/2-th and (n/2 + 1)-th sorted values for even n (18th and 19th of 36); the mean of an infinite and a finite value is the infinite one
- **worst**: a weak frame enters a median as its worst value: −∞ for N, S, A_neg, F_head_ins, the head shares and the isotropic margins; +∞ for A_amp and F_head_rem
- **direction_free**: the midpoint and PC-SWAP fractions, the additivity and P's errors exclude weak frames, with their count
- **strict**: N_f > 0, S_f > 0, A_amp < 0, A_neg > 0 strictly; ties fail; the precondition counts Δc_full ≤ −1; P agrees when both errors are ≤ max(τ, 0.02); medians pass at ≥ 0.05; counts pass at ≥ the threshold; P is not evaluable when τ_rem or τ_ins > 0.20
- **holm**: exact binomial tails as Fractions; Holm at 1/40 on the two count p-values alone (the smaller against 1/80, the larger against 1/40)
- **bootstrap**: one [10,000 × 36] index matrix from torch.Generator().manual_seed(20261001) and torch.randint, shared by every median; the interval is the 250th and the 9,750th of the sorted resampled medians
- **head_counts**: the head arms' counts (descriptive): F_head_rem < 1 and F_head_rem_v < 1 (the head-only removal leaves less than the full shift), F_head_ins > 0 and F_head_ins_v > 0 (the head-only installation produces some of it); weak frames fail; the shares are reported only under their guards, a guard-excluded frame counted and left out
- **R**: N_f = F_rem_off − F_rem > 0: the read removal takes away more of the shift than an equal-length off-read removal; **I**: S_f = F_ins − F_ins_off > 0: installing the read part produces more of the shift than an equal-length off-read installation
- Holm thresholds 26 and 25 of 36; median floor 0.05
- **D**: A_amp = Δc(NEC-AMP) − Δc_full < 0 and A_neg = Δc(SUF-NEG) > 0: doubling the read change adds plural shift, reversing it adds singular preference, the opposite of disruption; each count at least 25 of 36
- **P**: |F̂_rem − F_rem| ≤ max(τ_rem, 0.02) and |F̂_ins − F_ins| ≤ max(τ_ins, 0.02), the program's table committed before any target prompt; at least 27 of 36
- The order: R and I by Holm at 0.025 on the two count p-values alone (exact binomial tails under a symmetric sign); a family passes when Holm rejects its count and its median is at least 0.05, and a failed floor does not change the other family's threshold; D only if R and I pass; P only if D passes; a family not reached is reported descriptively and cannot change the label
- Weak frames: a frame whose Δc_full exceeds −1 nat counts as a failure in every count, enters every median as the worst value, and is excluded from the direction-free summaries with its count; ties count as failures
- The precondition: at least 30 of 36 frames with Δc_full ≤ −1

## The outcome

| condition | label |
|---|---|
| any gate fails, or an incident is recorded | `NOT_INTERPRETABLE` |
| the precondition fails | `BEHAVIOR_PRECONDITION_NOT_MET` |
| neither R nor I passes | `READ_COMPONENT_EFFECTS_NOT_ESTABLISHED` |
| R passes, I does not | `READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED` |
| I passes, R does not | `READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED` |
| R and I pass, D fails | `READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED` |
| R, I and D pass, P is uninformative | `READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE` |
| R, I and D pass, P fails | `READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED` |
| R, I, D and P pass | `READ_COMPONENT_EFFECTS_AS_PREDICTED` |

The scope of every label: the read component of the residual entering layer 3 at the cue position, as every reader of that residual sees it, in coordinated frames, for the cue pair one/two from which the read was estimated.

Not claimed by any label:
- no label says that the component is unused, or that it has no causal role
- no label assigns the effect to L03.H04 alone: the residual edit reaches every component that reads it
- no label generalizes beyond the one/two pair, this site, this checkpoint and these frames

The claim hierarchy:
- **generalization**: does the effect extend to three and many? Secondary only; the stratum cannot rescue a failed primary result
- **head_not_supported**: if the primary passes but the head-only evidence does not support the head: the decoded residual direction is causal at this site, within the scope, but its effect cannot be assigned uniquely to L03.H04
- **primary**: does manipulating the decoded read direction at the cue-position residual causally alter the one/two singular–plural choice in new coordinated frames? Answered by R, I and D against the matched non-read control, with the labels above
- **secondary**: is L03.H04 specifically the component that mediates the effect? The head-only arms answer it descriptively; they cannot change the primary label

- Secondary: the stratum, the isotropic controls, the head arms, the midpoint, PC-SWAP, the additivity, the manipulation check, the LayerNorm records and every bootstrap interval are computed after the result is written and can never rescue, alter or redefine it
- Gates: G4 is an integrity gate, not an outcome filter: a gate failure after the measurements stops outcome scoring and is reviewed under the incident procedure; no frame is dropped, replaced or selectively ignored; no tolerance is widened by calibration or changed after target results exist

## The gates

- G1_basis: 1e-12
- G1_read_weight: 1e-09
- G2_direction: 0.0001
- G2_length: 0.0001
- G2_read: 0.0001
- G2_u_off32: 1e-06
- G3_pass_through: 0.0001
- G4_I3: 0.0001
- G4_I4: 0.001
- G4_level1: 0.02
- G4_logit: 0.02
- G4_readout: 0.02
- G5_output: 0.001
- G5_row: 0.0001
- environment: 1e-09

## The stage-1 files

- written once at stage 1 directly at these paths, with their digests in the stage-1 record; committed by the user before confirm; the barrier re-reads them, checks their digests, the layout and the metadata, and recomputes the table bit for bit before any S2 key
- `experiments/101-read-component-intervention/stage1/states.f64`
- `experiments/101-read-component-intervention/stage1/states.json`
- `experiments/101-read-component-intervention/stage1/table.json`
