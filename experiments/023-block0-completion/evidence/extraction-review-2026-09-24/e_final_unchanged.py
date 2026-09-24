"""Final: the three outputs/experiment-023 files and the 022 table re-hashed; the tree still clean; nothing in the
repository (outside .git) modified after the review's start marker."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")

from revlib import CAND_DATA, CAND_INDEX, EXPECTED, REPO, RESULTS, SCRATCH, TABLE022, Report, git, sha256_file

rep = Report("final: unchanged")
rep.check("candidate f64 unchanged", sha256_file(CAND_DATA) == EXPECTED["f64_sha256"])
rep.check("candidate json unchanged", sha256_file(CAND_INDEX) == EXPECTED["json_sha256"])
results_sha = sha256_file(RESULTS)
rep.check("results.json unchanged (file sha256 recorded at the start of the review)", results_sha == open(SCRATCH / "results_sha_at_start.txt").read().strip(), results_sha)
rep.check("022 table unchanged (size and sha256)", TABLE022.stat().st_size == EXPECTED["table_size"] and sha256_file(TABLE022) == EXPECTED["table_sha256"])
rep.check("HEAD unchanged", git("rev-parse", "HEAD").decode().strip() == EXPECTED["head"])
porcelain = git("status", "--porcelain", "--untracked-files=all").decode()
rep.check("git status --porcelain --untracked-files=all empty", porcelain == "", repr(porcelain))
marker = (SCRATCH / "start.marker").stat().st_mtime
newer = []
for dirpath, dirnames, filenames in os.walk(REPO):
    if ".git" in os.path.relpath(dirpath, REPO).split(os.sep):
        continue
    for name in filenames:
        path = os.path.join(dirpath, name)
        try:
            if os.lstat(path).st_mtime > marker:
                newer.append(path)
        except OSError:
            pass
rep.check("no file in the repository (outside .git; .venv included) modified since the review started", not newer, str(newer[:10]))
rep.summary()
