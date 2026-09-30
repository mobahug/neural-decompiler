"""Convert draft v3 (Markdown) into LaTeX for the TMLR template, for a page-count build.

It builds the anonymous version: the internal header and every hidden trace are dropped, the statement on AI tools
becomes an unmarked first-page footnote, and the repository link is removed. The output is plain ASCII LaTeX.
Usage: md2tex.py draft.md main.tex
"""
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
                if ch in SPECIAL:
                    out.append(SPECIAL[ch])
                elif ord(ch) > 127:
                    if ch not in UNI:
                        raise SystemExit(f"unmapped character {ch!r} (U+{ord(ch):04X})")
                    out.append(UNI[ch])
                else:
                    out.append(ch)
    return "".join(out)


def quotes(t):
    """Straight double quotes in pairs become TeX quotes."""
    return re.sub(r'"([^"]*)"', r"``\1''", t)


def inline(t):
    out, i, buf = [], 0, []

    def flush():
        if buf:
            out.append(plain(quotes("".join(buf))))
            buf.clear()

    while i < len(t):
        if t[i] == "\\" and i + 1 < len(t) and t[i + 1] in "*_[]()#`":
            buf.append(t[i + 1]); i += 2; continue
        if t[i] == "`":
            j = t.index("`", i + 1)
            flush(); out.append(r"\texttt{" + plain(t[i + 1:j]) + "}"); i = j + 1; continue
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
state = {"appendix": False, "refs": False}


def table(rows, caption, label):
    cells = [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", r.strip()[1:-1])] for r in rows]
    head, data = cells[0], cells[2:]
    n = len(head)
    lens = [max(6, sum(len(r[k]) for r in [head] + data) / (len(data) + 1)) for k in range(n)]
    weights = [l ** 0.8 for l in lens]
    tot = sum(weights)
    avail = 1.0 - 0.02 * n
    spec = "".join(r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % (avail * w / tot) for w in weights)
    long_table = len(data) > 20
    hdr = " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\ \hline"
    rows_tex = "\n".join(" & ".join(inline(c) for c in r) + r" \\" for r in data)
    if long_table:
        return (r"{\footnotesize\setlength{\tabcolsep}{3pt}" "\n" r"\begin{longtable}{" + spec + "}\n"
                + r"\caption{" + caption + "}" + label + r"\\ \hline" + "\n" + hdr + "\n\\endfirsthead\n\\hline\n"
                + hdr + "\n\\endhead\n" + rows_tex + "\n\\hline\n\\end{longtable}}\n")
    return (r"\begin{table}[tbp]" "\n" r"\caption{" + caption + "}" + label + "\n"
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
        out.append(r"\item " + inline(m.group(3)))
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
            body.append(r"\FloatBarrier" "\n" r"\section*{References}\label{refsstart}")
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
            body.append(r"\section{" + inline(m.group(2)) + "}" "\n" r"\setcounter{figure}{0}\setcounter{table}{0}")
            state["refs"] = False
            i += 1; continue
        body.append(r"\section{" + inline(re.sub(r"^\d+\.\s*", "", name)) + "}"); i += 1; continue
    if first.startswith("### "):
        body.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", first[4:].strip())) + "}"); i += 1; continue
    if first.startswith("**Use of AI tools.**"):
        note = re.sub(r"^\*\*Use of AI tools\.\*\* \*\(In the TMLR build, a first-page footnote\.\)\* ", "", joined)
        ai_note = r"\textbf{Use of AI tools.} " + inline(note); i += 1; continue
    if state["refs"]:
        refs = [x for x in b if x.startswith("- ")]
        if not refs:
            i += 1; continue
        items = [inline(re.sub(r"^-\s+", "", x.strip())) for x in refs]
        body.append(r"\begin{list}{}{\leftmargin=1.5em\itemindent=-1.5em\itemsep=2pt\parsep=0pt}" "\n"
                    + "\n".join(r"\item " + x for x in items) + "\n" r"\end{list}")
        i += 1; continue
    if first.startswith("!["):
        m = re.match(r"!\[[^\]]*\]\(figures/([^)]+)\)", first)
        cap, num = split_caption(" ".join(x.strip() for x in blocks[i + 1]), "Figure")
        width = {"F4-opposition-ledger.png": 0.75, "F6a-completion.png": 0.8, "F5-closure-ladder.png": 0.85}.get(
            m.group(1), 1.0)
        body.append(r"\begin{figure}[tbp]" "\n" r"\centering" "\n"
                    + r"\includegraphics[width=%.2f\linewidth]{figures/%s}" % (width, m.group(1)) + "\n"
                    + r"\caption{" + cap + "}\\label{fig:" + num + "}\n\\end{figure}")
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
    body.append(inline(" ".join(p.strip() for p in para)))
    if rest:
        body.append(item_lines(rest))
    i += 1

doc = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
\input{math_commands.tex}
\usepackage{hyperref}
\usepackage{url}
\usepackage{graphicx}
\usepackage{array}
\usepackage{longtable}
\usepackage{placeins}

\title{%s}
\author{Anonymous authors}

\begin{document}
\maketitle
\begin{abstract}
%s
\end{abstract}
{\renewcommand{\thefootnote}{}\footnotetext{%s}}

%s

\label{docend}
\end{document}
""" % (title, abstract, ai_note, "\n\n".join(body))
OUT.write_text(doc, encoding="ascii")
print("wrote", OUT, len(doc), "chars")
