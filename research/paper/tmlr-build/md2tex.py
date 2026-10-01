"""Convert draft v3 (Markdown) into LaTeX for the TMLR template, for a page-count build.

It builds the anonymous version: the internal header and every hidden trace are dropped, the statement on AI tools
becomes an unmarked first-page footnote, and the repository link is removed. The output is plain ASCII LaTeX.
Usage: md2tex.py draft.md main.tex
"""
import json
import re
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MARKER = re.compile(r"^(\s*)([-*]|\d+\.)\s+")


def special(line):
    return (not line.strip()) or line.startswith(("|", "#", "![", "```"))


def unwrap(lines):
    """Join each paragraph, list item and blockquote onto one line, so that hidden traces can be removed whole."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if special(line):
            out.append(line); i += 1; continue
        if line.startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            out.append("> " + " ".join(x[1:].strip() for x in lines[i:j])); i = j; continue
        m = MARKER.match(line)
        prefix = m.group(0) if m else re.match(r"^\s*", line).group(0)
        j = i + 1
        while j < len(lines) and not special(lines[j]) and not MARKER.match(lines[j]) and not lines[j].startswith(">"):
            j += 1
        body = " ".join(x.strip() for x in lines[i:j])
        body = body[len(prefix.strip()):].strip() if m else body.strip()
        out.append(prefix + body); i = j
    return out


text = "\n".join(unwrap(SRC.read_text(encoding="utf-8").split("\n")))
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

UNI = {
    "–": "--", "—": "---", "§": r"\S{}", "²": r"\textsuperscript{2}", "“": "``", "”": "''", "‘": "`", "’": "'",
    "Δ": r"$\Delta$", "κ": r"$\kappa$", "−": r"$-$", "á": r"\'{a}", "é": r"\'{e}", "à": r"\`{a}", "…": r"\ldots{}",
    "γ": r"$\gamma$", "≤": r"$\leq$", "Π": r"$\Pi$", "∥": r"$\parallel$", "⊙": r"$\odot$", "≥": r"$\geq$",
    "τ": r"$\tau$", "ć": r"\'{c}", "ρ": r"$\rho$",
}
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
           ">": r"\textgreater{}"}


def plain(t):
    """Escape plain text and map its non-ASCII characters."""
    t = re.sub(r"(.)̂", lambda m: "\x00HAT" + m.group(1) + "\x00", t)
    t = re.sub(r"[₀-₉]+", lambda m: "\x00SUB" + m.group(0).translate(SUB) + "\x00", t)
    out = []
    for part in re.split(r"(\x00[^\x00]*\x00)", t):
        if part.startswith("\x00HAT"):
            out.append(r"$\hat{" + part[4:-1] + "}$")
        elif part.startswith("\x00SUB"):
            out.append("$_{" + part[4:-1] + "}$")
        else:
            for ch in part:
                if ch == "\x01":  # an empty group, set by quotes() between a single and a double closing quote
                    out.append("{}")
                elif ch in SPECIAL:
                    out.append(SPECIAL[ch])
                elif ord(ch) > 127:
                    if ch not in UNI:
                        raise SystemExit(f"unmapped character {ch!r} (U+{ord(ch):04X})")
                    out.append(UNI[ch])
                else:
                    out.append(ch)
    return "".join(out)


def quotes(t):
    """Straight double quotes inside a curly quotation become single quotes; other pairs become TeX quotes."""
    t = re.sub("\u201c([^\u201d]*)\u201d",
               lambda m: "\u201c" + re.sub(r'"([^"]*)"', "`\\1'\x01", m.group(1)) + "\u201d", t)
    return re.sub(r'"([^"]*)"', r"``\1''", t)


CITEKEYS = json.loads((Path(__file__).parent / "citekeys.json").read_text(encoding="utf-8"))
CITE_AUTHORS = sorted({k.rsplit(", ", 1)[0] for k in CITEKEYS}, key=len, reverse=True)
CITE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in CITE_AUTHORS)
                  + r"), ((?:19|20)\d\d(?:, (?:19|20)\d\d)*)")
USED = set()


def cite(t):
    """'Geiger et al., 2021, 2024' -> \\citealp{...}; the surrounding parentheses stay in the text."""
    out, last = [], 0
    for m in CITE.finditer(t):
        keys = []
        for year in m.group(2).split(", "):
            k = CITEKEYS.get(f"{m.group(1)}, {year}")
            if k is None:
                raise SystemExit(f"no BibTeX key for {m.group(1)}, {year}")
            keys.append(k)
        USED.update(keys)
        out.append(plain(quotes(t[last:m.start()])))
        out.append(r"\citealp{" + ",".join(keys) + "}")
        last = m.end()
    out.append(plain(quotes(t[last:])))
    return "".join(out)


def inline(t):
    out, i, buf = [], 0, []

    def flush():
        if buf:
            out.append(cite("".join(buf)))
            buf.clear()

    while i < len(t):
        if t[i] == "\\" and i + 1 < len(t) and t[i + 1] in "*_[]()#`":
            buf.append(t[i + 1]); i += 2; continue
        if t[i] == "`":
            j = t.index("`", i + 1)
            code = plain(t[i + 1:j])
            if re.fullmatch(r"[0-9a-f]{20,}", code):  # a long hash may break every eight characters
                code = r"\allowbreak{}".join(code[k:k + 8] for k in range(0, len(code), 8))
            flush(); out.append(r"\texttt{" + code + "}"); i = j + 1; continue
        if t.startswith("**", i):
            j = t.index("**", i + 2)
            flush(); out.append(r"\textbf{" + inline(t[i + 2:j]) + "}"); i = j + 2; continue
        if t[i] == "*" and i + 1 < len(t) and t[i + 1] != " ":
            j = i + 1
            while True:
                j = t.index("*", j)
                if t[j - 1] != "\\":
                    break
                j += 1
            flush(); out.append(r"\emph{" + inline(t[i + 1:j]) + "}"); i = j + 1; continue
        m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", t[i:])
        if m:
            flush()
            label, url = m.group(1), m.group(2)
            if url.startswith("http") and "github.com/mobahug" not in url:
                out.append(r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + inline(label) + "}")
            elif "github.com/mobahug" in url:
                out.append("[repository link removed for anonymity]")
            else:
                out.append(inline(label))
            i += m.end(); continue
        buf.append(t[i]); i += 1
    flush()
    return "".join(out)


lines = text.split("\n")
blocks, cur = [], []
for line in lines:
    if line.strip():
        cur.append(line)
    elif cur:
        blocks.append(cur); cur = []
if cur:
    blocks.append(cur)

body, title, abstract, ai_note = [], "", "", ""
state = {"appendix": False, "refs": False, "pin": False}


LINE_PT = 469.75  # the template's 6.5 in text width
# F1, F1a, F2, F3, F5, F7 and F8 are drawn for the screen; the build uses their print variants (see the figures'
# README).
PRINT_VARIANTS = {"F1-chain.png": ["F1-chain-print.pdf"], "F2-prediction-map.png": ["F2-prediction-map-print.pdf"],
                  "F7-sequence.png": ["F7-sequence-print-1.pdf", "F7-sequence-print-2.pdf"],
                  "F1a-chain-schematic.png": ["F1a-chain-schematic-print.pdf"], "F3-head.png": ["F3-head-print.pdf"],
                  "F5-closure-ladder.png": ["F5-closure-ladder-print.pdf"],
                  "F8-boundary.png": ["F8-boundary-print.pdf"]}
PART_ROWS = {"F7-sequence.png": ["005--015", "016--025"]}
# Appendix D of the anonymous build: the Markdown names the repository, a commit and paths that would identify it.
ANONYMOUS = {
    "**The record.**": (
        "**The record.** Experiments 005–025, the registered claims C002 and C003, the synthesis and the literature "
        "review are committed in a version-controlled project repository, withheld during anonymous review. Each "
        "experiment's design, report and evidence are there, with its lock and confirmation set where it had them "
        "(Table A2), and its README states its protocol and outcome."),
    "**The audit trail.**": (
        "**The audit trail.** The source of this paper keeps, for the statements that rest on the record, a hidden "
        "trace to it: a file of the record, its lines and a pinned commit. A checker verifies the quotes and the "
        "traces mechanically."),
    "**The figures.**": (
        "**The figures.** Two scripts regenerate every figure. One reads every plotted value from a pinned commit and "
        "checks it and its status against the record; the other draws the figures, byte-identical from run to run. "
        "Nothing in them is computed, and the figures had independent reviews of their provenance and drawing."),
    "**Record caveats.**": (
        "**Record caveats.** The record lists its own caveats, from errata and conclusions narrowed or restated at or "
        "after closure to digest mismatches it investigated and explained, inconsistencies in its records and "
        "misleading commit subjects; it states that none of them changes a frozen result unless its entry says so. "
        "The wiring erratum, the frame-conditional tables that were not committed before scoring, the gaps in the "
        "review records and the restatements after closure are stated in §3 and Table A2; the full list is withheld "
        "with the repository during review."),
}


def min_fraction(column, header):
    """The widest unbreakable word of a column, in \\footnotesize, as a fraction of the line (with column padding).
    Code words wider than 0.3 of the line may break after their underscores."""
    best = 0.0
    for index, cell in enumerate([header] + column):
        for code in re.findall(r"`([^`]*)`", cell):
            for word in code.split():
                width = len(word) * 4.2
                if width > 0.3 * LINE_PT:
                    width = max(len(part) + 1 for part in re.split(r"[_|]", word)) * 4.2
                best = max(best, width)
        for word in re.split(r"[\s\-\u2013]+", re.sub(r"`[^`]*`|\*", " ", cell)):  # TeX breaks after hyphens
            best = max(best, len(word) * (4.6 if index == 0 else 4.1))
    return (best + 8) / LINE_PT


def column_fractions(head, data):
    n = len(head)
    lens = [max(6, sum(len(r[k]) for r in [head] + data) / (len(data) + 1)) for k in range(n)]
    weights = [l ** 0.8 for l in lens]
    avail = 1.0 - 0.02 * n
    mins = [min_fraction([r[k] for r in data], head[k]) for k in range(n)]
    fixed = {}
    while True:  # columns below their minimum get it; the rest share what is left in proportion
        free = [k for k in range(n) if k not in fixed]
        rest = avail - sum(fixed.values())
        if not free:
            return [fixed[k] * avail / sum(fixed.values()) for k in range(n)]
        tot = sum(weights[k] for k in free)
        new = {k: mins[k] for k in free if rest * weights[k] / tot < mins[k]}
        if not new:
            return [fixed.get(k, rest * weights[k] / tot) for k in range(n)]
        fixed.update(new)


def table(rows, caption, label):
    cells = [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", r.strip()[1:-1])] for r in rows]
    head, data = cells[0], cells[2:]
    spec = "".join(r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % f for f in column_fractions(head, data))
    long_table = len(data) > 20
    hdr = " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\ \hline"
    rows_tex = "\n".join(" & ".join(inline(c).replace(r"\_", r"\_\allowbreak{}") for c in r) + r" \\" for r in data)
    if long_table:
        return (r"{\footnotesize\setlength{\tabcolsep}{3pt}" "\n" r"\begin{longtable}{" + spec + "}\n"
                + r"\caption{" + caption + "}" + label + r"\\ \hline" + "\n" + hdr + "\n\\endfirsthead\n\\hline\n"
                + hdr + "\n\\endhead\n" + rows_tex + "\n\\hline\n\\end{longtable}}\n")
    place = "tbp" if state["appendix"] else "!tb"  # body floats never take a page of their own
    return (r"\begin{table}[" + place + "]\n" r"\caption{" + caption + "}" + label + "\n"
            r"{\footnotesize\setlength{\tabcolsep}{3pt}" "\n" r"\begin{tabular}{" + spec + "}\n\\hline\n" + hdr
            + "\n" + rows_tex + "\n\\hline\n\\end{tabular}}\n\\end{table}\n")


def split_caption(block_text, kind):
    """'**Table 3. Title.** rest' -> ('\\textbf{Title.} rest', number)."""
    m = re.match(r"\*\*" + kind + r" ([A-Z]?\d+)\. (.*?)\*\*(.*)$", block_text, flags=re.S)
    if not m:
        m = re.match(r"\*\*" + kind + r" ([A-Z]?\d+)\. (.*?),\*\*(.*)$", block_text, flags=re.S)
        if not m:
            raise SystemExit(f"caption not parsed: {block_text[:80]}")
        return r"\textbf{" + inline(m.group(2)) + ",}" + inline(m.group(3)), m.group(1)
    return r"\textbf{" + inline(m.group(2)) + "}" + inline(m.group(3)), m.group(1)


def item_lines(block):
    """A list block, one item per line after unwrapping, possibly nested by indentation."""
    out, stack = [], []
    for line in block:
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        depth = len(m.group(1)) // 3
        env = "enumerate" if m.group(2)[0].isdigit() else "itemize"
        while len(stack) > depth + 1 or (len(stack) == depth + 1 and stack[-1] != env):
            out.append(r"\end{" + stack.pop() + "}")
        if len(stack) < depth + 1:
            out.append(r"\begin{" + env + "}"); stack.append(env)
        text = inline(m.group(3))
        out.append((r"\item{} " if text.startswith("[") else r"\item ") + text)  # "[status]:" is not a label
    while stack:
        out.append(r"\end{" + stack.pop() + "}")
    return "\n".join(out)


i = 0
while i < len(blocks):
    b = blocks[i]
    first = b[0]
    joined = " ".join(x.strip() for x in b)
    if first.startswith("# "):
        title = inline(first[2:].strip()); i += 1; continue
    if first.startswith("**Draft v3"):
        i += 1; continue
    if first.startswith("## "):
        name = first[3:].strip()
        if name == "Abstract":
            abstract = inline(" ".join(x.strip() for x in blocks[i + 1])); i += 2; continue
        if name == "References":
            state["refs"] = True
            body.append("\n".join([r"\FloatBarrier",
                                   r"\renewcommand{\bibsection}{\section*{References}\label{refsstart}}",
                                   r"\bibliography{references}", r"\bibliographystyle{tmlr}"]))
            i += 1; continue
        m = re.match(r"Appendix ([A-Z]): (.*)$", name)
        if m:
            if not state["appendix"]:
                body.append("\n".join([
                    r"\clearpage", r"\appendix",
                    r"\renewcommand{\thefigure}{\thesection\arabic{figure}}",
                    r"\renewcommand{\thetable}{\thesection\arabic{table}}",
                    r"\renewcommand{\theHfigure}{\thesection.\arabic{figure}}",
                    r"\renewcommand{\theHtable}{\thesection.\arabic{table}}"]))
                state["appendix"] = True
            starts_with_figure = i + 1 < len(blocks) and blocks[i + 1][0].startswith("![")
            state["pin"] = starts_with_figure  # its first figure is set here, on the heading's page
            body.append(r"\FloatBarrier" "\n" + (r"\clearpage" "\n" if starts_with_figure else "")
                        + r"\section{" + inline(m.group(2)) + "}" "\n" r"\setcounter{figure}{0}\setcounter{table}{0}")
            state["refs"] = False
            i += 1; continue
        body.append(r"\section{" + inline(re.sub(r"^\d+\.\s*", "", name)) + "}"); i += 1; continue
    if first.startswith("### "):
        body.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", first[4:].strip())) + "}"); i += 1; continue
    if first.startswith("**Use of AI tools.**"):
        note = re.sub(r"^\*\*Use of AI tools\.\*\* \*\(In the TMLR build, a first-page footnote\.\)\* ", "", joined)
        ai_note = r"\textbf{Use of AI tools.} " + inline(note); i += 1; continue
    if state["refs"]:  # the Markdown list is the readable copy; the build uses references.bib
        i += 1; continue
    if first.startswith("!["):
        m = re.match(r"!\[[^\]]*\]\(figures/([^)]+)\)", first)
        cap, num = split_caption(" ".join(x.strip() for x in blocks[i + 1]), "Figure")
        width = {"F4-opposition-ledger.png": 0.75, "F6a-completion.png": 0.8, "F5-closure-ladder.png": 0.85}.get(
            m.group(1), 1.0)
        place = "H" if state["pin"] else "tbp" if state["appendix"] else "!tb"
        state["pin"] = False
        parts = PRINT_VARIANTS.get(m.group(1), [m.group(1)])  # print variants are drawn at the text width
        if m.group(1) in PRINT_VARIANTS:
            width = 1.0
        if len(parts) > 1:
            cap += r" This part shows %s; the second part follows." % PART_ROWS[m.group(1)][0]
        body.append(r"\begin{figure}[" + place + "]\n" r"\centering" "\n"
                    + r"\includegraphics[width=%.2f\linewidth]{figures/%s}" % (width, parts[0]) + "\n"
                    + r"\caption{" + cap + "}\\label{fig:" + num + "}\n\\end{figure}")
        for part, rows_shown in zip(parts[1:], PART_ROWS.get(m.group(1), [])[1:]):
            body.append(r"\begin{figure}[" + place + "]\n" r"\centering" "\n"
                        + r"\includegraphics[width=\linewidth]{figures/%s}" % part + "\n"
                        + r"\par\smallskip{\small Figure %s, continued: %s.}" % (num, rows_shown) + "\n\\end{figure}")
        i += 2; continue
    if first.startswith("**Table ") and i + 1 < len(blocks) and blocks[i + 1][0].startswith("|"):
        cap, num = split_caption(joined, "Table")
        body.append(table(blocks[i + 1], cap, r"\label{tab:" + num + "}"))
        i += 2; continue
    if first.startswith("|"):
        raise SystemExit("table without caption: " + first[:60])
    if first.startswith(">"):
        inner = " ".join(x.lstrip(">").strip() for x in b)
        body.append(r"\begin{center}\fbox{\begin{minipage}{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}\small"
                    "\n" + inline(inner) + "\n" r"\end{minipage}}\end{center}")
        i += 1; continue
    if re.match(r"^\s*([-*]|\d+\.)\s", first):
        # A lead-in paragraph may sit on the list's first line only when the block starts with text.
        body.append(item_lines(b)); i += 1; continue
    # A paragraph, possibly followed by list lines in the same block (e.g. "Contributions." then items).
    para, rest = [], []
    for x in b:
        (rest if (rest or re.match(r"^\s*([-*]|\d+\.)\s", x)) else para).append(x)
    text = " ".join(p.strip() for p in para)
    lead = next((key for key in ANONYMOUS if text.startswith(key)), None)
    body.append(inline(ANONYMOUS[lead] if state["appendix"] and lead else text))
    if rest:
        body.append(item_lines(rest))
    i += 1

doc = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
\input{math_commands.tex}
\usepackage{float}
\usepackage{hyperref}
\usepackage{url}
\usepackage{graphicx}
\usepackage{array}
\usepackage{longtable}
\usepackage{placeins}
\setlength{\LTcapwidth}{\textwidth}

\title{%s}
\author{Anonymous authors}

\begin{document}
\maketitle
\begin{abstract}
%s
\end{abstract}
{\renewcommand{\thefootnote}{}\footnotetext{%s}}

%s

\FloatBarrier
\label{docend}
\end{document}
""" % (title, abstract, ai_note, "\n\n".join(body))
unused = set(CITEKEYS.values()) - USED
if unused:
    raise SystemExit(f"BibTeX keys never cited: {sorted(unused)}")
OUT.write_text(doc, encoding="ascii")
print("wrote", OUT, len(doc), "chars")
