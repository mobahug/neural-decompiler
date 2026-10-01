#!/usr/bin/env bash
# Page-count build of draft v3 in the mandatory TMLR template. This is not the submission build: see README.md.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
build="$here/build"
# JmlrOrg/tmlr-style-file, the repository TMLR's author guidelines link, at its main branch of 2023-06-30.
template_commit=7bf90efe3a0debbba703c05c43f3ff7e4d4a2992

mkdir -p "$build/figures"
for f in tmlr.sty tmlr.bst fancyhdr.sty math_commands.tex; do
    if [ ! -f "$build/$f" ]; then
        curl -fsSL "https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/$template_commit/$f" -o "$build/$f"
    fi
done
cp "$repo"/research/paper/figures/*.png "$repo"/research/paper/figures/*-print*.pdf "$build/figures/"
cp "$here/references.bib" "$build/"

PYTHONDONTWRITEBYTECODE=1 "$repo/.venv/bin/python" "$here/check_bib.py" "$repo/research/paper/draft-v3-2026-09-30.md"
PYTHONDONTWRITEBYTECODE=1 "$repo/.venv/bin/python" "$here/md2tex.py" \
    "$repo/research/paper/draft-v3-2026-09-30.md" "$build/main.tex"
(cd "$build" && tectonic --keep-intermediates --keep-logs main.tex > build.log 2>&1)

# The page of each body float, of the References heading, and of the last page, from the labels in main.aux.
PYTHONDONTWRITEBYTECODE=1 "$repo/.venv/bin/python" - "$build/main.aux" <<'EOF'
import re, sys
labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{[^}]*\}\{([^}]*)\}", open(sys.argv[1]).read()))
for name in sorted(k for k in labels if k.startswith(("fig:", "tab:"))):
    print(f"{name:8s} page {labels[name]}")
print(f"References begin on page {labels['refsstart']}; the last page is {labels['docend']}.")
EOF
