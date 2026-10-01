"""Check references.bib and citekeys.json against the draft's reference list, which stays the readable copy.

For every entry of the list there must be one BibTeX entry with the same year, the same title (ignoring case,
braces and punctuation) and the same author surnames in the same order; a list that ends "and colleagues" must
match the first authors only. Every in-text form in citekeys.json must name a key of references.bib.
Usage: check_bib.py draft.md
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
TEX_ACCENT = {"'": "́", "`": "̀", '"': "̈", "^": "̂", "~": "̃", "v": "̌", "c": "̧",
              "u": "̆", "=": "̄"}


def detex(s):
    s = re.sub(r"\{?\\([`'\"^~=]|[vcu](?=[{ ]))\s*\{?([A-Za-z])\}?\}?",
               lambda m: unicodedata.normalize("NFC", m.group(2) + TEX_ACCENT[m.group(1)]), s)
    return s.replace("\\&", "&").replace("{", "").replace("}", "")


def fold(s):
    s = unicodedata.normalize("NFKD", detex(s))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def draft_entries(path):
    text = re.sub(r"<!--.*?-->", "", Path(path).read_text(encoding="utf-8"), flags=re.S)
    block = text.split("## References")[1].split("## Appendix A")[0]
    items, cur = [], None
    for line in block.split("\n"):
        if line.startswith("- "):
            if cur:
                items.append(cur)
            cur = line[2:].strip()
        elif cur is not None and line.startswith("  "):
            cur += " " + line.strip()
    items.append(cur)
    out = []
    for it in items:
        m = re.match(r"(.*?) \((\d{4})\)\. (?:Preface\. )?\*((?:[^*\\]|\\.)+)\*", it)
        names = m.group(1)
        colleagues = names.endswith(" and colleagues")
        names = names[:-len(" and colleagues")] if colleagues else names
        title = "Preface" if " Preface. " in it[:it.index("*")] + " " else m.group(3).replace("\\*", "*")
        out.append({"surnames": re.split(r", | and ", names), "colleagues": colleagues, "year": m.group(2),
                    "title": title.rstrip(".")})
    return out


def bib_entries(path):
    out = {}
    for key, body in re.findall(r"@\w+\{([^,]+),\n(.*?)\n\}", Path(path).read_text(encoding="ascii"), flags=re.S):
        fields = dict(re.findall(r"^  (\w+) = \{(.*)\},$", body, flags=re.M))
        out[key] = {"authors": [a.split(",")[0].strip() for a in fields["author"].split(" and ")],
                    "year": fields["year"], "title": fields["title"]}
    return out


def main():
    draft, bib = draft_entries(sys.argv[1]), bib_entries(HERE / "references.bib")
    keys = json.loads((HERE / "citekeys.json").read_text(encoding="utf-8"))
    failures = []
    if len(draft) != len(bib):
        failures.append(f"{len(draft)} entries in the draft's list, {len(bib)} in references.bib")
    unmatched = dict(bib)
    for entry in draft:
        want = [fold(s) for s in entry["surnames"]]
        match = [k for k, b in unmatched.items()
                 if b["year"] == entry["year"] and fold(b["title"]) == fold(entry["title"])]
        if len(match) != 1:
            failures.append(f"no single BibTeX entry for {entry['surnames'][0]} ({entry['year']}): {entry['title']}")
            continue
        got = [fold(a) for a in unmatched.pop(match[0])["authors"]]
        if got[:len(want)] != want or (not entry["colleagues"] and len(got) != len(want)):
            failures.append(f"authors differ for {match[0]}: list {entry['surnames']}, BibTeX {got}")
    for form, key in keys.items():
        if key not in bib:
            failures.append(f"citekeys.json names an unknown key: {form} -> {key}")
    if set(keys.values()) != set(bib):
        failures.append(f"keys without an in-text form: {sorted(set(bib) - set(keys.values()))}")
    for failure in failures:
        print("FAIL", failure)
    print(f"checked {len(draft)} references and {len(keys)} in-text forms:", "FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
