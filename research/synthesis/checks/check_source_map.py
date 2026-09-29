#!/usr/bin/env python3
"""Mechanical checks of the synthesis source map. Read-only: it writes nothing into the repository.

1. Quotes. Every quote — text in “…” — must be followed directly by a citation group in parentheses, and must appear
   verbatim in the cited lines of one of the cited files (one line of slack either side), as the file stood at the
   pinned commit or at the commit the citation names. The comparison ignores Markdown emphasis (`*`), block-quote
   markers (`>`) and line breaks. A quote from a commit message cites `(message of <hash>)`. A quote with no citation,
   or with a citation that does not parse, fails. So does straight-quoted prose ("…") outside quotes and code. The
   citation group may sit inside an HTML comment that opens after the quote, with only whitespace between them,
   `“…”<!-- (023:142) -->`, as in drafts whose traceability markers are hidden from the rendered text; citations inside
   comments are checked like any other. Outside fenced code, every HTML comment must close before the next one opens,
   and no line may begin with one, because a line-initial `<!--` would start an HTML block instead of a hidden
   comment.
2. Citations. Every parenthesized group that looks like a citation (a three-digit experiment number or a file name,
   then `:` and a digit) must parse completely, and must name a file that exists at its commit and lines that exist
   in it.
3. Commit hashes. Every 7–40-character hexadecimal string in backticks must name a commit of this repository, or be in
   the short allowlist of known non-commit identifiers below.
4. Labels. Every outcome label in backticks (`ALL_CAPS_WITH_UNDERSCORES`) must occur in a committed record. In the
   overview table, every backticked label in an experiment's row must occur in that experiment's own README.

What it does not check: paraphrases, numbers outside quotes, and classifications. The independent review covers those.

Citation forms: `(023:142)`, `(023:12–14)`, `(023:140–145, 152)` for `experiments/023-*/README.md`;
`(path/to/file.md:35–36)` for any other file; several references joined by `,` or `;` in one group; `(006@9566372:78–79)`
or `(path@commit:L)` for a file as it stood at an earlier commit. Everything is read at the pinned commit unless a
citation names another. Usage: check_source_map.py <file.md> [more files]
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
PIN = "d7c5e6279e171ef0d8d8bc8aa4b8b512356d962e"
PATH = r"(?:[0-9]{3}|[A-Za-z0-9_./-]+\.(?:md|json|py))"
LINES = r"[0-9]+(?:–[0-9]+)?(?:,\s*(?![0-9]{3}(?:@[0-9a-f]+)?:)[0-9]+(?:–[0-9]+)?)*"
REF = re.compile(rf"(?P<path>{PATH})(?:@(?P<rev>[0-9a-f]{{7,40}}))?:(?P<lines>{LINES})")
LOOKS_LIKE_CITATION = re.compile(rf"(?:\b[0-9]{{3}}|\.(?:md|json|py))(?:@[0-9a-f]+)?:[0-9]")
FILLER = re.compile(r"^(?:[\s,;]|for example|and)*$")
MESSAGE = re.compile(r"^message of `?(?P<rev>[0-9a-f]{7,40})`?$")
QUOTE = re.compile(r"“([^”]+)”")
GROUP = re.compile(r"\s*(?:<!--\s*)?\(([^()]*)\)")
_files: dict[tuple[str, str], list[str] | None] = {}
NON_COMMITS = {"e93a9faa9c77e5d09219f6c868bfc7a1bd65593c": "the pinned Pythia-70M-deduped model revision (Hugging Face), not a git commit"}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True)


def readme_of(number: str, rev: str) -> str | None:
    names = [name for name in git("ls-tree", "--name-only", rev, "experiments/").stdout.split() if name.split("/")[-1].startswith(number + "-")]
    return f"{names[0]}/README.md" if len(names) == 1 else None


def lines_of(path: str, rev: str) -> list[str] | None:
    if (rev, path) not in _files:
        shown = git("show", f"{rev}:{path}")
        _files[(rev, path)] = shown.stdout.split("\n") if shown.returncode == 0 else None
    return _files[(rev, path)]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "")).strip()


def parse_group(group: str):
    """Return ([(path, rev, [(start, end), ...]), ...], error)."""
    group = group.strip()
    message = MESSAGE.match(group)
    if message:
        return [("MESSAGE", message["rev"], [])], None
    refs, rest, position = [], [], 0
    for match in REF.finditer(group):
        rest.append(group[position:match.start()])
        position = match.end()
        path, rev = match["path"], match["rev"] or PIN
        if re.fullmatch(r"[0-9]{3}", path):
            resolved = readme_of(path, rev)
            if resolved is None:
                return [], f"no unique README for {path} at {rev[:7]}"
            path = resolved
        spans = []
        for part in match["lines"].split(","):
            bounds = [int(x) for x in part.strip().split("–")]
            spans.append((bounds[0], bounds[-1]))
        refs.append((path, rev, spans))
    rest.append(group[position:])
    if not refs or not all(FILLER.match(piece) for piece in rest):
        return [], f"unparsed citation `({group})`"
    return refs, None


def check_refs(refs, where: str, failures: list[str]) -> list[str]:
    window: list[str] = []
    for path, rev, spans in refs:
        if path == "MESSAGE":
            shown = git("log", "-1", "--format=%B", rev)
            if shown.returncode != 0:
                failures.append(f"{where}: no commit {rev}")
            window.append(shown.stdout)
            continue
        content = lines_of(path, rev)
        if content is None:
            failures.append(f"{where}: {path} does not exist at {rev[:7]}")
            continue
        for start, end in spans:
            if start < 1 or end > len(content) or start > end:
                failures.append(f"{where}: lines {start}–{end} outside {path} ({len(content)} lines at {rev[:7]})")
            window.extend(re.sub(r"^\s*>\s?", "", line) for line in content[max(0, start - 2):min(len(content), end + 1)])
    return window


def main(files: list[str]) -> int:
    failures: list[str] = []
    counts = {"quotes": 0, "citations": 0, "hashes": 0, "labels": 0}
    for name in files:
        text = Path(name).read_text(encoding="utf-8")
        # 1. quotes
        for match in QUOTE.finditer(text):
            counts["quotes"] += 1
            quote, where = match.group(1), f"{name}: quote “{match.group(1)[:70]}…”"
            group = GROUP.match(text, match.end())
            if not group:
                failures.append(f"{where}: no citation follows it")
                continue
            refs, error = parse_group(group.group(1))
            if error:
                failures.append(f"{where}: {error}")
                continue
            window = check_refs(refs, where, failures)
            if normalize(quote) not in normalize(" ".join(window)):
                failures.append(f"{where}: not found in the cited lines ({group.group(1)})")
        code_free = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced blocks would shift the backtick pairing
        bare = re.sub(r"`[^`]*`", "", QUOTE.sub("", code_free))
        for match in re.finditer(r"\"[^\"\n]{3,}\"", bare):
            failures.append(f"{name}: straight-quoted text {match.group(0)[:70]}: quote it with “…” and a citation, or paraphrase it")
        # hidden citations: every HTML comment closes before the next one opens, and none begins a line
        for match in re.finditer(r"<!--", code_free):
            end, following = code_free.find("-->", match.end()), code_free.find("<!--", match.end())
            if end == -1 or (following != -1 and following < end):
                failures.append(f"{name}: an HTML comment ({code_free[match.start():match.start() + 60]!r}) does not close")
        fenced = False
        for number, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced and line.lstrip().startswith("<!--"):
                failures.append(f"{name}:{number}: a line begins with an HTML comment, which would open an HTML block")
        # 2. citations
        for match in re.finditer(r"\(([^()]*)\)", text):
            body = match.group(1)
            if not LOOKS_LIKE_CITATION.search(body) and not MESSAGE.match(body.strip()):
                continue
            counts["citations"] += 1
            refs, error = parse_group(body)
            if error:
                failures.append(f"{name}: {error}")
                continue
            check_refs(refs, f"{name}: citation ({body[:60]})", failures)
        # 3. commit hashes
        for value in sorted(set(re.findall(r"`([0-9a-f]{7,40})`", code_free))):
            counts["hashes"] += 1
            if value not in NON_COMMITS and git("cat-file", "-e", f"{value}^{{commit}}").returncode != 0:
                failures.append(f"{name}: `{value}` is not a commit of this repository")
        # 4. labels
        spans = [part.strip() for span in re.findall(r"`([^`]+)`", code_free) for part in span.replace("\\|", "|").split("|")]
        for label in sorted({part for part in spans if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9<>]+)+", part)}):
            counts["labels"] += 1
            if git("grep", "-q", "-F", label, PIN, "--", "experiments", "docs", "research", "screening", "README.md").returncode != 0:
                failures.append(f"{name}: label `{label}` occurs in no committed record at {PIN[:7]}")
        for row in re.finditer(r"^\| (?P<exp>[0-9]{3}) \|(?P<cells>.*)\|\s*$", text, flags=re.MULTILINE):
            cells = row["cells"].split(" | ")
            if len(cells) < 4:
                continue
            readme = readme_of(row["exp"], PIN)
            content = "\n".join(lines_of(readme, PIN) or []) if readme else ""
            for span in re.findall(r"`([^`]+)`", cells[2]):
                for label in (part.strip() for part in span.replace("\\|", "|").split("|")):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]*", label) and label not in content:
                        failures.append(f"{name}: overview row {row['exp']}: `{label}` is not in {readme}")
    print(f"checked {counts['quotes']} quotes, {counts['citations']} citation groups, {counts['hashes']} commit hashes and "
          f"{counts['labels']} labels in {len(files)} file(s), against {PIN[:7]} unless a citation names another commit")
    for failure in failures:
        print("FAIL", failure)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
