# Experiment 101 — report

- Run `30ca8697c8702496`; phases: calibrate complete, lock complete, stage1 complete, confirm complete, report not_started
- Design revision 3 (`a87e284`, decided `516df20`), plan revision 3 (`96f0cc8`); configuration `production`

## The outcome

**`READ_COMPONENT_EFFECTS_AS_PREDICTED`**

The scope of every label: the read component of the residual entering layer 3 at the cue position, as every reader of that residual sees it, in coordinated frames, for the cue pair one/two from which the read was estimated.

Not claimed by any label:
- no label says that the component is unused, or that it has no causal role
- no label assigns the effect to L03.H04 alone: the residual edit reaches every component that reads it
- no label generalizes beyond the one/two pair, this site, this checkpoint and these frames

## The families

| family | count | threshold | median | result |
|---|---|---|---|---|
| R | 36 of 36 (p 1/68719476736) | Holm (26/25) | +1.006380 | **PASS** |
| I | 36 of 36 (p 1/68719476736) | Holm (26/25) | +1.035027 | **PASS** |
| D | A_amp < 0: 28, A_neg > 0: 36 | 25 each | — | **PASS** |
| P | 33 agree | 27 | τ_rem +0.025154, τ_ins +0.035111 | **PASS** |

- The precondition: 36 of 36 frames with Δc_full ≤ −1 (required 30): met
- The order: R and I by Holm at 0.025 on the two count p-values alone (exact binomial tails under a symmetric sign); a family passes when Holm rejects its count and its median is at least 0.05, and a failed floor does not change the other family's threshold; D only if R and I pass; P only if D passes; a family not reached is reported descriptively and cannot change the label
- Weak frames: a frame whose Δc_full exceeds −1 nat counts as a failure in every count, enters every median as the worst value, and is excluded from the direction-free summaries with its count; ties count as failures

## The claim hierarchy

- **primary**: does manipulating the decoded read direction at the cue-position residual causally alter the one/two singular–plural choice in new coordinated frames? Answered by R, I and D against the matched non-read control, with the labels above
- **secondary**: is L03.H04 specifically the component that mediates the effect? The head-only arms answer it descriptively; they cannot change the primary label
- **generalization**: does the effect extend to three and many? Secondary only; the stratum cannot rescue a failed primary result
- **head_not_supported**: if the primary passes but the head-only evidence does not support the head: the decoded residual direction is causal at this site, within the scope, but its effect cannot be assigned uniquely to L03.H04

## Validity (every gate passed for a result to exist)

- Gate failures: []
- The accounting: {'equal': True, 'executed': 1980, 'ledger': 1980, 'manifest': 1980}
- The barrier: True
- G6 before the result write: True
- G6 at the report: bit for bit True

## Secondary records (no label force)

- the stratum, the isotropic controls, the head arms, the midpoint, PC-SWAP, the additivity, the manipulation check, the LayerNorm records and every bootstrap interval are computed after the result is written and can never rescue, alter or redefine it
- Primary medians and intervals: {'A_amp': {'excluded': 0, 'interval': [-0.2754431615901902, -0.055789223200158666], 'median': -0.135958514636076, 'n': 36}, 'A_neg': {'excluded': 0, 'interval': [0.72661349139636, 1.1251645631428007], 'median': 0.9223115172567244, 'n': 36}, 'N': {'excluded': 0, 'interval': [0.9878578569433053, 1.0511500466488637], 'median': 1.0063799347533808, 'n': 36}, 'S': {'excluded': 0, 'interval': [1.0172004231252862, 1.0591976568846606], 'median': 1.0350265212692265, 'n': 36}}
- The stratum: {'many': {'A_amp_negative': 23, 'A_neg_positive': 36, 'I_count': 36, 'R_count': 36, 'direction_free': {'F_half_ins': {'excluded': 0, 'interval': [0.5406918513249781, 0.6482122721600916], 'median': 0.5862552388194391, 'n': 36}, 'F_half_rem': {'excluded': 0, 'interval': [0.7743901347420121, 0.8469674821737366], 'median': 0.8097797139843015, 'n': 36}, 'F_swap_pl': {'excluded': 0, 'interval': [0.08959785744323571, 0.1888678485733797], 'median': 0.1518481969686253, 'n': 36}, 'F_swap_ref': {'excluded': 0, 'interval': [0.9714630804875264, 1.021054249700414], 'median': 0.9905808492426154, 'n': 36}, 'additivity': {'excluded': 0, 'interval': [1.0972628401752456, 1.2312417601345664], 'median': 1.1343532119051047, 'n': 36}}, 'medians': {'A_amp': {'excluded': 0, 'interval': [-0.11370154272151911, 0.007265501384494222], 'median': -0.060752385779272, 'n': 36}, 'A_neg': {'excluded': 0, 'interval': [0.7303235017800638, 1.0840855851958064], 'median': 0.9305914383900314, 'n': 36}, 'N': {'excluded': 0, 'interval': [0.7018691954916748, 0.7901514283952467], 'median': 0.7658030647906078, 'n': 36}, 'S': {'excluded': 0, 'interval': [0.8963940145794072, 0.9505432598890744], 'median': 0.9277257392723146, 'n': 36}}, 'weak': 0}, 'three': {'A_amp_negative': 28, 'A_neg_positive': 36, 'I_count': 36, 'R_count': 36, 'direction_free': {'F_half_ins': {'excluded': 0, 'interval': [0.6219571767675434, 0.7001065939417274], 'median': 0.663923106561247, 'n': 36}, 'F_half_rem': {'excluded': 0, 'interval': [0.6672461280596008, 0.7477975438236709], 'median': 0.715346377075638, 'n': 36}, 'F_swap_pl': {'excluded': 0, 'interval': [0.023730687327966037, 0.09280298302031745], 'median': 0.07852580206263055, 'n': 36}, 'F_swap_ref': {'excluded': 0, 'interval': [0.9772733706502463, 1.0114704894708986], 'median': 0.999414661724929, 'n': 36}, 'additivity': {'excluded': 0, 'interval': [0.9873214476160928, 1.0589248686850952], 'median': 1.0295082278956014, 'n': 36}}, 'medians': {'A_amp': {'excluded': 0, 'interval': [-0.2619876137262658, -0.05393653579905067], 'median': -0.1345199391811711, 'n': 36}, 'A_neg': {'excluded': 0, 'interval': [0.7187144605419302, 1.0973649809632122], 'median': 0.9282133850870251, 'n': 36}, 'N': {'excluded': 0, 'interval': [0.9200730259424875, 1.0028196411147112], 'median': 0.9547267278115054, 'n': 36}, 'S': {'excluded': 0, 'interval': [0.9922051418731596, 1.0321285654224739], 'median': 1.0154184531529116, 'n': 36}}, 'weak': 0}}
- The isotropic controls: {'iso_ins_margin': {'excluded': 0, 'interval': [0.9647540971846322, 1.015129648164972], 'median': 0.9826499629359879, 'n': 36, 'positive': 36}, 'iso_rem_margin': {'excluded': 0, 'interval': [0.9795875666103023, 1.0426859620897124], 'median': 1.012842392930998, 'n': 36, 'positive': 36}}
- The head arms: {'F_head_ins': {'count': 36, 'criterion': 'F_head_ins > 0.0', 'excluded': 0, 'interval': [0.884475674643479, 0.9161374762706999], 'median': 0.8984492321178645, 'n': 36}, 'F_head_ins_v': {'count': 36, 'criterion': 'F_head_ins_v > 0.0', 'excluded': 0, 'interval': [0.880317301407465, 0.9134795374010654], 'median': 0.8990367079617831, 'n': 36}, 'F_head_rem': {'count': 36, 'criterion': 'F_head_rem < 1.0', 'excluded': 0, 'interval': [0.08677698756417401, 0.1348637246088881], 'median': 0.1112790439653891, 'n': 36}, 'F_head_rem_v': {'count': 36, 'criterion': 'F_head_rem_v < 1.0', 'excluded': 0, 'interval': [0.0828421879799342, 0.14711199861278887], 'median': 0.1088784253304872, 'n': 36}, 'head_install_share': {'excluded': 0, 'guard_excluded': 0, 'interval': [0.8862294647261347, 0.9063086000283102], 'median': 0.8943155243735512, 'n': 36}, 'head_install_share_v': {'excluded': 0, 'guard_excluded': 0, 'interval': [0.8861163155478018, 0.9055222408776834], 'median': 0.8923580041570336, 'n': 36}, 'head_removal_share': {'excluded': 0, 'guard_excluded': 0, 'interval': [0.8740031560568631, 0.8914478268524269], 'median': 0.8816274276075726, 'n': 36}, 'head_removal_share_v': {'excluded': 0, 'guard_excluded': 0, 'interval': [0.8613108098149351, 0.886955119031701], 'median': 0.8774007065456442, 'n': 36}}
- Direction-free: {'F_half_ins': {'excluded': 0, 'interval': [0.6651491976848017, 0.7618547644997292], 'median': 0.716630875462228, 'n': 36}, 'F_half_rem': {'excluded': 0, 'interval': [0.6621546782313825, 0.7406965636465807], 'median': 0.702785372305812, 'n': 36}, 'F_swap_pl': {'excluded': 0, 'interval': [0.009718385077211823, 0.06108726639762713], 'median': 0.0440546637760839, 'n': 36}, 'F_swap_ref': {'excluded': 0, 'interval': [0.9894896592572479, 1.0212356091080412], 'median': 1.0075964800826889, 'n': 36}, 'additivity': {'excluded': 0, 'interval': [0.9525060833503474, 1.0170251979089848], 'median': 0.9885094995451867, 'n': 36}}
- The program: {'N_hat_minus_N': {'excluded': 0, 'interval': [-0.0047806130795853274, 0.005981905665650267], 'median': 0.00035691429232859173, 'n': 36}, 'P_error_ins': {'excluded': 0, 'interval': [0.0089146501600737, 0.014522959394242752], 'median': 0.011295214972680045, 'n': 36}, 'P_error_rem': {'excluded': 0, 'interval': [0.004463328081879023, 0.009380210452569104], 'median': 0.006002374740293533, 'n': 36}, 'S_hat_minus_S': {'excluded': 0, 'interval': [-0.014877592162048092, -0.001165341284149024], 'median': -0.007294032685728258, 'n': 36}}
