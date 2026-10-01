"""Build the anonymized supplementary ZIP for the TMLR submission.

The ZIP holds every tracked file at HEAD apart from the binary arrays, which do not fit TMLR's 100 MB limit. Strings
that identify the author, the account or the machine are replaced with placeholders, and the build stops if any is
left. It also writes SUPPLEMENT-README.md, EXCLUDED.tsv and MODIFIED.tsv into the ZIP. The output goes to
build/supplement/, which git ignores. Usage, from the repository root:

    .venv/bin/python research/paper/tmlr-build/supplement.py
"""
import hashlib
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "build" / "supplement"
NAME = "supplement.zip"
LIMIT = 100 * 2**20

BINARY = re.compile(r"\.(pt|f64|gz|npy|npz|bin|safetensors)$")
SELF = "research/paper/tmlr-build/supplement.py"  # carries the identifying patterns themselves

# Longest first, so that a session path is replaced before the user name inside it.
REPLACEMENTS = [
    (re.compile(r"/private/tmp/claude-\d+/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/[0-9a-f-]{36}"),
     "<SCRATCH>"),
    (re.compile(r"-Users-gaborhorvath-ulenius-myprojects-neural-decompiler"), "-<HOME>-<REPO>"),
    (re.compile(r"/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"), "<REPO>"),
    (re.compile(r"/Users/gaborhorvath-ulenius"), "<HOME>"),
    (re.compile(r"https?://github\.com/mobahug/neural-decompiler[^\s)\]>\"']*"), "<repository link withheld>"),
    (re.compile(r"github\.com/mobahug/neural-decompiler"), "<repository withheld>"),
    (re.compile(r"github\.com/mobahug/?"), "<account withheld>"),
    (re.compile(r"mobahug"), "<account>"),
    (re.compile(r"gaborhorvath-ulenius", re.I), "<user>"),
    (re.compile(r"Gabor\s+(Horvath[- ])?Ulenius", re.I), "<author>"),
    (re.compile(r"ulenius", re.I), "<author>"),
]
IDENTIFYING = re.compile(rb"gaborhorvath|ulenius|mobahug|gaborulenius|\bgabor\b", re.I)

README = """# Supplementary material: code and records

This archive holds the project's code and records, the basis of the paper's experiments 005-025, as a snapshot of
{date}. Everything the paper cites is here, apart from the binary arrays listed in `EXCLUDED.tsv`, which do not fit
the 100 MB limit.

## Contents

- `experiments/`: each experiment's README (its protocol, outcome and closure), its runner, its preregistration lock
  and predictions where it had them, and its evidence (run records and review records).
- `src/neural_decompiler/` and `tests/`: the shared package and its tiered test suite.
- `docs/`: the research methodology, the instrumentation contracts, and every design (`docs/superpowers/specs/`) and
  implementation plan (`docs/superpowers/plans/`).
- `research/`: the claim registry, the synthesis and source map, the literature review, the paper's drafts and
  reviews, the figure scripts, the TMLR build, and the audit of the statement on AI tools.
- `screening/`: the behavior screening that proposed the studied behavior.
- `pyproject.toml` and `uv.lock`: the Python environment.

## Running it

Python 3.10-3.12 and the locked dependencies (`uv sync`, or `pip install -e .` with the versions in `uv.lock`). The
experiments use `EleutherAI/pythia-70m-deduped` at a pinned revision, on CPU in float32; set `HF_HUB_OFFLINE=1` once
the weights are cached. A bare `pytest` runs the fast test tier; `tests/conftest.py` describes the others.

## Anonymization

- **Replaced strings.** Strings that identify the authors, their account or their machine were replaced in
  {n_modified} text files with placeholders: `<REPO>` and `<HOME>` for local paths, `<SCRATCH>` for an assistant
  session's scratch directory, `<user>`, `<account>` and `<author>` for names, and `<repository link withheld>` for
  the repository's address. `MODIFIED.tsv` lists each file with its SHA-256 before and after the replacement.
- **Digests.** The records pin many files by digest. The pinned digests of the files in `MODIFIED.tsv` no longer
  match their anonymized copies here; the originals will verify against the repository after review.
- **Commit hashes.** The records cite commits of the project's git repository, which is withheld during review. The
  history is not included.
- **Excluded files.** `EXCLUDED.tsv` lists the {n_excluded} binary arrays left out, with their sizes and SHA-256
  digests.
"""


def main() -> None:
    date = subprocess.run(["git", "--no-optional-locks", "log", "-1", "--format=%cd", "--date=short"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    tar = subprocess.run(["git", "--no-optional-locks", "archive", "--format=tar", "HEAD"], cwd=REPO,
                         capture_output=True, check=True).stdout
    kept, excluded, modified = {}, [], []
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for member in sorted(tf.getmembers(), key=lambda m: m.name):
            if not member.isfile() or member.name == SELF:
                continue
            data = tf.extractfile(member).read()
            if BINARY.search(member.name):
                excluded.append((member.name, len(data), hashlib.sha256(data).hexdigest()))
                continue
            if IDENTIFYING.search(data):
                text = data.decode("utf-8")  # every identifying file is text; a decode error stops the build
                for pattern, replacement in REPLACEMENTS:
                    text = pattern.sub(replacement, text)
                new = text.encode("utf-8")
                modified.append((member.name, hashlib.sha256(data).hexdigest(), hashlib.sha256(new).hexdigest()))
                data = new
            kept[member.name] = data
    kept["SUPPLEMENT-README.md"] = README.format(date=date, n_modified=len(modified),
                                                 n_excluded=len(excluded)).encode()
    kept["EXCLUDED.tsv"] = ("path\tbytes\tsha256\n" + "".join(f"{p}\t{n}\t{h}\n" for p, n, h in excluded)).encode()
    kept["MODIFIED.tsv"] = ("path\tsha256_original\tsha256_anonymized\n"
                            + "".join(f"{p}\t{a}\t{b}\n" for p, a, b in modified)).encode()
    left = [name for name, data in kept.items() if IDENTIFYING.search(name.encode()) or IDENTIFYING.search(data)]
    if left:
        raise SystemExit(f"identifying strings remain in {len(left)} files, e.g. {left[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(kept):
            info = zipfile.ZipInfo("supplement/" + name, date_time=(2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, kept[name])
    size = path.stat().st_size
    if size > LIMIT:
        raise SystemExit(f"{path} is {size / 2**20:.1f} MB, over the 100 MB limit")
    print(f"wrote {path}: {len(kept)} files, {len(excluded)} excluded, {len(modified)} anonymized, "
          f"{size / 2**20:.1f} MB, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
