# Citation re-check: the applied bibliography findings

Verdict: PASS. All the applied bibliography findings are in the files correctly, and nothing else changed. The checks
ran against these live files, which did not change between 10:07 and 10:09:

| file | SHA-1 |
|---|---|
| references.bib | c37a6092 |
| citekeys.json | 9ba60b23 |
| draft-v3-2026-09-30.md | df467091 |
| build/main.pdf | 39793fd7 |
| build/main.tex | c7b97979 |

The staged main.pdf and main.tex are byte-identical to the live build, and the build used the live bib. The staged
references.reviewed.bib is 0c82373a, the version reviewed in the first round.

**The bib.** Compared with the reviewed version, exactly 15 entries changed, all of them ones listed.
- **Grant (S1):** `grant2025addressing` became `grant2026addressing`, an `@inproceedings` with booktitle ICLR and year
  2026. That matches OpenReview cZrTMqYVL6, which records an ICLR 2026 Oral.
- **The nine ICML entries (S2, S3, S4, S6 and N2):** Anani, Huang, Joshi, Vaccaro, Zhang, Mueller, Rushing, Weiss and
  the unchanged Pythia. A script compared each one's booktitle, series, volume, pages, year and publisher with the
  BibTeX on its PMLR page, and all nine match exactly.
- **Geiger 2025 (S5):** volume 26, number 83 and pages 1–64 now match the JMLR BibTeX.
- **The five ACL entries (N4):** Brinkmann, Hao, McDougall, Syed and van Miltenburg now link the same URLs as their ACL
  Anthology records.
- **Nothing else changed.** No author or title changed in any entry, and the set of keys is the same apart from the
  Grant rename. Bertinetto has no `pages` field, but that was removed in the reviewed version, not in this round.

**citekeys.json and the draft.**
- **citekeys.json** has only one change: "Grant et al., 2026" now maps to `grant2026addressing`.
- **The draft** has only the expected edits:
  - §8 now cites "(Makelov et al., 2024; Grant et al., 2026)";
  - Grant's list entry reads "(2026) … ICLR 2026";
  - the Anani, Huang, Zhang, Mueller, Rushing and Weiss list entries now give their PMLR volumes and pages, and
    Geiger's gives "JMLR 26(83), pp. 1–64";
  - Chan's entry links the Alignment Forum post, and the "as given by the mirror" note is gone.
- **No stale year:** Grant appears only twice in the draft, both times with 2026.
- **main.tex** still has 81 `\citealp` commands, the same keys in the same order as the Markdown. That includes
  `grant2026addressing`, and all 63 keys are cited. `main.log` reports no undefined citations.

**check_bib.py.** Output: `checked 63 references and 63 in-text forms: PASS`, exit status 0.

**The PDF, pp. 11 and 13–18.**
- **Grant:** it sorts between Gong and Gross and reads "In International Conference on Learning Representations, 2026".
  In §8 on p. 11 it renders "(Makelov et al., 2024; Grant et al., 2026)".
- **Geiger 2025** reads "Journal of Machine Learning Research, 26(83): 1–64, 2025."
- **All nine ICML entries** read "In Proceedings of the Nth International Conference on Machine Learning, volume V of
  Proceedings of Machine Learning Research, pp. X–Y. PMLR, year."
- **The five ACL entries** show their aclanthology.org URLs.
- **Unchanged entries:** no other change or rendering problem was found.

**One optional point, not a defect.** N4 concerned the bib's url field. The draft's own reference list still links
arXiv for those five ACL papers. That is harmless, except that McDougall's arXiv page carries the older title,
"…Understanding an Attention Head". N3, N6 and N7 were left unapplied, which is fine; all three were optional.
