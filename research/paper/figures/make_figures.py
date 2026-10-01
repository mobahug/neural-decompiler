#!/usr/bin/env python3
"""Draw the paper's figures F1–F8 from the audited record. Writes SVG and PNG files beside this script.

Every value and every status comes from `figure_data.py`, which reads it from a pinned commit and checks it against its
source; see the figure-data audit (figure-data-audit-2026-09-30.md). Nothing is computed here. The only
transformations are presentational: selecting, ordering, labelling, colouring by a recorded status, choosing axes and
drawing identity lines. No curve is fitted and no statistic is recomputed; the recorded results are printed as
recorded. The images carry no figure titles or footnote notes: the captions in the figures' README and in the draft
carry them. Panel titles and in-plot annotations stay.

    .venv/bin/python research/paper/figures/make_figures.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

import figure_data as data  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "neural-decompiler",
                     "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.7})

# One colour per recorded status, used by every figure. Where a status is shown by colour alone (the chips and badges
# of F1, F2 and F7), unestablished is also hatched, exploratory striped and descriptive dotted, so that no status
# depends on colour.
COLOURS = {
    data.PROSPECTIVE: "#009E73",
    data.PROSPECTIVE_CONDITIONAL: "#7FCDB5",
    data.LOCKED: "#0072B2",
    data.EXPLORATORY: "#E69F00",
    data.DESCRIPTIVE: "#A6A6A6",
    data.UNESTABLISHED: "#D55E00",
}
HATCHES = {data.UNESTABLISHED: ("////", "#8F3F00"), data.EXPLORATORY: ("|||", "#FFFFFF"),
           data.DESCRIPTIVE: ("...", "#FFFFFF")}
GATE_COLOURS = {"checked": "#009E73", "Not established": "#D55E00", "Not reached": "#E0E0E0",
                "Not claimed": "#F5F5F5", "no such item": "#FFFFFF"}
LEGEND_ORDER = [data.PROSPECTIVE, data.PROSPECTIVE_CONDITIONAL, data.LOCKED, data.DESCRIPTIVE, data.EXPLORATORY,
                data.UNESTABLISHED]
NOTE = "#333333"


def _luminance(colour: str) -> float:
    """The relative luminance of an sRGB colour (WCAG 2)."""
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    red, green, blue = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def ink(fill: str) -> str:
    """Dark or white text, whichever has the higher contrast on the fill."""
    dark, light = "#111111", "#FFFFFF"
    on_fill = _luminance(fill) + 0.05
    return dark if on_fill / (_luminance(dark) + 0.05) >= (_luminance(light) + 0.05) / on_fill else light


def style(status: str, hatched: bool = True) -> dict:
    """The fill of a status, with its hatch where the status is shown by colour alone."""
    fill = {"facecolor": COLOURS[status]}
    if hatched and status in HATCHES:
        fill.update(hatch=HATCHES[status][0], hatchcolor=HATCHES[status][1])
    return fill


def status_legend(ax, statuses=LEGEND_ORDER, hatched=True, **kwargs):
    handles = [Patch(edgecolor="#555555", label=s, **style(s, hatched)) for s in statuses]
    return ax.legend(handles=handles, frameon=False, **kwargs)


def save(fig, name: str, print_variant: bool = False) -> None:
    """SVG and PNG for the screen; a print variant is written as PDF, for the TMLR build, and PNG, for previews."""
    if print_variant:
        fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None}, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(OUT / f"{name}.png", dpi=300, metadata={"Software": None}, bbox_inches="tight",
                    pad_inches=0.02)
    else:
        fig.savefig(OUT / f"{name}.svg", metadata={"Date": None}, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.png", dpi=200, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)


# Layout in inches. The diagrams (F1, F7) use one axes filling the figure, with data units in inches and y growing
# downward, so that measured text sizes place and size their boxes.
_measure = {}


def extent(text: str, size: float, weight: str = "normal") -> float:
    """The rendered width of a line of text, in inches."""
    if "fig" not in _measure:
        _measure["fig"] = plt.figure()
    fig = _measure["fig"]
    t = fig.text(0, 0, text, fontsize=size, weight=weight)
    width = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return width


def wrap(text: str, size: float, width: float) -> list[str]:
    """Greedy word wrap to a width in inches."""
    out, current = [], ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip()
        if current and extent(trial, size) > width:
            out.append(current)
            current = word
        else:
            current = trial
    out.append(current)
    return out


def canvas(width: float, height: float = 30.0):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def crop(fig, ax, height: float) -> None:
    """Fix the canvas height once the content is placed; data units stay inches."""
    width = fig.get_size_inches()[0]
    fig.set_size_inches(width, height)
    ax.set_ylim(height, 0)


def box(ax, x, y, width, height, fill="#FFFFFF", status=None, edge="#444444", dashed=False, rounding=0.04,
        linewidth=0.6):
    fills = style(status) if status else {"facecolor": fill}
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                edgecolor=edge, linewidth=linewidth, linestyle="--" if dashed else "-", **fills))


def text_block(ax, x, y, text, width, size=7.0, **kwargs):
    """Wrapped text from the top; returns its height in inches."""
    wrapped = wrap(text, size, width)
    ax.text(x, y, "\n".join(wrapped), ha="left", va="top", fontsize=size, linespacing=1.3, **kwargs)
    return len(wrapped) * size * 1.3 / 72


def on_fill(status: str) -> dict | None:
    """A backing box of the fill colour for text on a hatched fill, so the hatch does not cross the letters."""
    return {"facecolor": COLOURS[status], "edgecolor": "none", "pad": 0.5} if status in HATCHES else None


# Badges: small boxes carrying one status each, with an optional caption (F1, F7).
BADGE_SIZE, BADGE_H, BADGE_GAP = 6.2, 0.17, 0.03


def badge_width(caption: str, size: float = BADGE_SIZE) -> float:
    return max(0.15, extent(caption, size) + 0.08) if caption else 0.15


def draw_badge(ax, x, middle, width, caption, status, size=BADGE_SIZE, height=BADGE_H):
    box(ax, x, middle - height / 2, width, height, status=status, edge="#333333", rounding=0.03, linewidth=0.5)
    if caption:
        ax.text(x + width / 2, middle, caption, ha="center", va="center", fontsize=size,
                color=ink(COLOURS[status]), bbox=on_fill(status))


# F1 -----------------------------------------------------------------------------------------------------------------
F1_TEXT, F1_LINE, F1_PAD = 7.0, 0.125, 0.05


def f1_layout(chip, width=None, text_size=F1_TEXT, line=F1_LINE, badge_size=BADGE_SIZE, badge_h=BADGE_H,
              beside_min=0.95, pad_v=0.14):
    """Badges beside the text, or in rows above it when they would leave the text less than beside_min inches."""
    badges = [(caption, status, badge_width(caption, badge_size)) for caption, status in chip["badges"]]
    in_line = sum(b[2] for b in badges) + BADGE_GAP * (len(badges) - 1)
    if width is None:
        width = in_line + extent(chip["text"], text_size) + 2 * F1_PAD + 0.08
    sizes = {"text_size": text_size, "badge_size": badge_size, "badge_h": badge_h}
    beside = width - in_line - 2 * F1_PAD - 0.06
    if beside >= beside_min:
        text_lines = wrap(chip["text"], text_size, beside)
        return {"rows": [badges], "lines": text_lines, "width": width, "stacked": False,
                "height": pad_v + line * len(text_lines), **sizes}
    rows, used = [[]], 0.0
    for b in badges:
        if rows[-1] and used + BADGE_GAP + b[2] > width - 2 * F1_PAD:
            rows.append([])
            used = 0.0
        used += (BADGE_GAP if rows[-1] else 0) + b[2]
        rows[-1].append(b)
    text_lines = wrap(chip["text"], text_size, width - 2 * F1_PAD)
    return {"rows": rows, "lines": text_lines, "width": width, "stacked": True,
            "height": pad_v + len(rows) * (badge_h + 0.04) + line * len(text_lines), **sizes}


def f1_chip(ax, x, y, layout):
    box(ax, x, y, layout["width"], layout["height"], edge="#888888")
    text = "\n".join(layout["lines"])
    size, badge_size, badge_h = layout["text_size"], layout["badge_size"], layout["badge_h"]
    if not layout["stacked"]:
        middle, bx = y + layout["height"] / 2, x + F1_PAD
        for caption, status, width in layout["rows"][0]:
            draw_badge(ax, bx, middle, width, caption, status, badge_size, badge_h)
            bx += width + BADGE_GAP
        ax.text(bx + 0.03, middle, text, ha="left", va="center", fontsize=size, linespacing=1.15)
        return
    top = y + 0.06
    for row in layout["rows"]:
        bx = x + F1_PAD
        for caption, status, width in row:
            draw_badge(ax, bx, top + badge_h / 2, width, caption, status, badge_size, badge_h)
            bx += width + BADGE_GAP
        top += badge_h + 0.04
    ax.text(x + F1_PAD + 0.02, top + 0.01, text, ha="left", va="top", fontsize=size, linespacing=1.15)


def chain_header(ax, stages, xs, column, y, header):
    """The stages, in the model's order. The encoding (the layer-0 MLP) runs in parallel with block-0 attention."""
    for i, (name, subtitle) in enumerate(stages):
        ax.add_patch(Rectangle((xs[i], y), column, header, facecolor="#F2F2F2", edgecolor="#444444", linewidth=0.8))
        ax.text(xs[i] + column / 2, y + (0.18 if subtitle else header / 2), name, ha="center", va="center",
                fontsize=8.5, weight="bold")
        if subtitle:
            ax.text(xs[i] + column / 2, y + 0.36, subtitle, ha="center", va="center", fontsize=7, color=NOTE)
        if i >= 1 and i < len(stages) - 1:
            start = xs[i] + column + (0.06 if i == 1 else 0)
            ax.annotate("", xy=(xs[i + 1], y + header / 2), xytext=(start, y + header / 2),
                        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.8})
    box(ax, xs[0] - 0.06, y - 0.06, xs[1] + column - xs[0] + 0.12, header + 0.12, "none", edge="#666666", dashed=True)
    ax.text((xs[0] + column + xs[1]) / 2, y + header / 2, "∥", ha="center", va="center", fontsize=12)


def f1a(record):
    """The chain's stages alone, as a schematic: F1's stage row."""
    stages = record["stages"]
    column, gap = 1.63, 0.2
    xs = [0.12 + i * (column + gap) for i in range(len(stages))]
    fig, ax = canvas(xs[-1] + column + 0.12)
    chain_header(ax, stages, xs, column, 0.12, 0.5)
    crop(fig, ax, 0.12 + 0.5 + 0.14)
    save(fig, "F1a-chain-schematic")


def f1(record):
    stages, chips = record["stages"], record["chips"]
    left, column, gap = 1.45, 1.63, 0.2
    xs = [left + i * (column + gap) for i in range(len(stages))]
    right = xs[-1] + column
    fig, ax = canvas(right + 0.1)
    y, header = 0.12, 0.5
    chain_header(ax, stages, xs, column, y, header)
    y += header + 0.22
    lanes = [("E-patch", "E-patch lane", "stage by stage"),
             ("real swap", "real-swap lane", "end to end; only Δc was scored")]
    for lane, name, subtitle in lanes:
        top = y
        ax.plot([0.05, right], [top, top], color="#BBBBBB", linewidth=0.6)
        ax.text(0.05, top + 0.12, name, fontsize=8.5, weight="bold", va="top")
        text_block(ax, 0.05, top + 0.32, subtitle, left - 0.2, size=7.2, color=NOTE)
        bottoms = []
        for i, (stage, _) in enumerate(stages):
            cursor = top + 0.12
            here = [c for c in chips if c["lane"] == lane and c["stage"] == stage]
            for c in here:
                layout = f1_layout(c, column)
                f1_chip(ax, xs[i], cursor, layout)
                cursor += layout["height"] + 0.07
            if lane == "real swap" and stage == "transport":
                cursor += text_block(ax, xs[i] + 0.04, cursor + 0.02, "End to end, L03.H04 is computed inside "
                                     "block 3, as part of the readout.", column - 0.08, size=6.8, color=NOTE,
                                     style="italic")
            elif not here:
                ax.text(xs[i] + column / 2, cursor + 0.12, "—", ha="center", va="center", color="#999999")
                cursor += 0.25
            bottoms.append(cursor)
        y = max(bottoms)
        if lane == "real swap":
            # End to end, the transport head sits inside the readout: one frame around both columns.
            box(ax, xs[4] - 0.08, top + 0.06, xs[5] + column - xs[4] + 0.16, y - top - 0.02, "none", edge="#666666",
                dashed=True)
        if lane == "E-patch":
            link = next(c for c in chips if c["lane"] == "link")
            layout = f1_layout(link)
            start, end = xs[0] + column / 2, xs[4] + column / 2
            middle, x = y + 0.04 + layout["height"] / 2, (start + end) / 2 - layout["width"] / 2
            ax.plot([start, x], [middle, middle], color="#444444", linewidth=0.8)
            ax.annotate("", xy=(end, middle), xytext=(x + layout["width"], middle),
                        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.8})
            f1_chip(ax, x, y + 0.04, layout)
            y += layout["height"] + 0.12
        y += 0.1
    top = y
    ax.plot([0.05, right], [top, top], color="#BBBBBB", linewidth=0.6)
    ax.text(0.05, top + 0.12, "causal status", fontsize=8.5, weight="bold", va="top")
    x, tallest = xs[0], 0.0
    for c in (c for c in chips if c["lane"] == "causal"):
        layout = f1_layout(c)
        f1_chip(ax, x, top + 0.12, layout)
        x += layout["width"] + 0.2
        tallest = max(tallest, layout["height"])
    y = top + 0.12 + tallest + 0.15
    used = [s for s in LEGEND_ORDER if any(status == s for c in chips for _, status in c["badges"])]
    status_legend(ax, used, loc="upper left", bbox_to_anchor=(xs[0] - 0.1, y), bbox_transform=ax.transData,
                  ncol=len(used), fontsize=7.5)
    y += 0.35
    crop(fig, ax, y + 0.1)
    save(fig, "F1-chain")


# F2 -----------------------------------------------------------------------------------------------------------------
def f2(record):
    rows, columns, cells = record["rows"], record["columns"], record["cells"]
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(rows) + 1.9))
    ax.set_xlim(0, 3.2 + 2.6 * len(columns))
    ax.set_ylim(len(rows) + 0.2, -0.9)
    ax.axis("off")
    for j, column in enumerate(columns):
        ax.text(3.2 + 2.6 * j + 1.3, -0.45, column, ha="center", va="center", fontsize=8.5, weight="bold")
    for i, row in enumerate(rows):
        ax.text(3.1, i + 0.5, row, ha="right", va="center", fontsize=8)
        for j, column in enumerate(columns):
            x = 3.2 + 2.6 * j
            ax.add_patch(Rectangle((x + 0.03, i + 0.06), 2.54, 0.88, facecolor="#FAFAFA", edgecolor="#E0E0E0",
                                   linewidth=0.5))
            for cell in cells:
                if cell.row == row and cell.column == column:
                    fill = COLOURS[cell.status]
                    box(ax, x + 0.05, i + 0.08, 2.5, 0.84, status=cell.status)
                    ax.text(x + 1.3, i + 0.44, cell.text, ha="center", va="center", fontsize=6.8, color=ink(fill),
                            bbox=on_fill(cell.status))
                    ax.text(x + 2.5, i + 0.88, cell.status, ha="right", va="bottom", fontsize=5.8, style="italic",
                            color=ink(fill), bbox=on_fill(cell.status))
    used = [s for s in LEGEND_ORDER if any(cell.status == s for cell in cells)]
    status_legend(ax, used, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=6, fontsize=7.5)
    save(fig, "F2-prediction-map")


# F3 -----------------------------------------------------------------------------------------------------------------
def _identity(ax, points):
    """Axes from the plotted values, with a margin, and the identity line across them."""
    values = [p.x for p in points] + [p.y for p in points]
    span = max(values) - min(values)
    low = math.floor((min(values) - 0.06 * span) * 20) / 20
    high = math.ceil((max(values) + 0.06 * span) * 20) / 20
    ax.plot([low, high], [low, high], color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")


F3_VARIANTS = {
    "screen": {"figsize": (15, 5.4), "title": 9.5, "note": 7.5, "axis": {}, "sizes": (26, 14, 40, 60),
               "widths": (0.5, 1.0, 1.4, 0.6), "fit": None, "note_at": (0.02, 0.98), "legend": "lower right",
               "legend_kw": {}, "offset": (0.12, -0.12), "name": "F3-head"},
    "print": {"figsize": (6.5, 2.8), "title": 6.3, "note": 5.2, "axis": {"fontsize": 6.0}, "sizes": (9, 4, 14, 22),
              "widths": (0.35, 0.7, 1.0, 0.45), "fit": 1.9, "note_at": (0.0, -0.3), "legend": "upper left",
              "legend_kw": {"bbox_to_anchor": (0.0, -0.43), "borderaxespad": 0.0, "borderpad": 0.0, "handlelength": 1.0,
                            "handletextpad": 0.4},  # beneath the note, clear of the points
              "offset": (0.2, -0.17), "name": "F3-head-print"},
}


def fit(text, size, width, joiner=" "):
    """Rewrap a text's lines to a width in inches; None keeps the text as written."""
    return text if width is None else "\n".join(wrap(joiner.join(text.split("\n")), size, width))


def f3(record, variant="screen"):
    v = F3_VARIANTS[variant]
    small, tiny, ring, big = v["sizes"]
    thin, medium, thick, arrow = v["widths"]
    fig, axes = plt.subplots(1, 3, figsize=v["figsize"])
    a1, a2, b = record["A1"], record["A2"], record["B"]
    ax = axes[0]
    ax.scatter([p.x for p in a1["points"]], [p.y for p in a1["points"]], s=small, color=COLOURS[data.DESCRIPTIVE],
               edgecolor="#333333", linewidth=thin)
    _identity(ax, a1["points"])
    ax.set_xlabel("P1 read-out, token mean", **v["axis"])
    ax.set_ylabel("measured transport q_T, token mean", **v["axis"])
    ax.set_title(fit("A1. 009: 23 token means [descriptive]", v["title"], v["fit"]), loc="left", fontsize=v["title"])
    ax.text(*v["note_at"], fit("009's read-out test was scored over 138 pairs\n(not committed, not shown): Spearman "
                            "0.957, MAE 0.066\n[prospective, given the measured input]", v["note"], v["fit"]),
            transform=ax.transAxes, va="top", fontsize=v["note"])
    ax = axes[1]
    rest = [p for p in a2["points"] if p.label != "an"]
    an = [p for p in a2["points"] if p.label == "an"]
    ax.scatter([p.x for p in rest], [p.y for p in rest], s=tiny, color=COLOURS[data.PROSPECTIVE], alpha=0.8,
               edgecolor="none")
    ax.scatter([p.x for p in an], [p.y for p in an], s=ring, facecolor="none", edgecolor="#000000", linewidth=medium,
               label=f"an, its {a2['an_frames']} licensed cue-final frames")
    _identity(ax, a2["points"])
    ax.set_xlabel("P1 read-out of the measured head input", **v["axis"])
    ax.set_ylabel("measured transport q_T", **v["axis"])
    ax.set_title(fit("A2. 011, read-out replicated: the 142 scored pairs", v["title"], v["fit"]), loc="left",
                 fontsize=v["title"])
    ax.text(*v["note_at"], fit("Spearman 0.974, MAE 0.076 over these pairs\n[prospective, given the measured input]",
                            v["note"], v["fit"]), transform=ax.transAxes, va="top", fontsize=v["note"])
    ax.legend(loc=v["legend"], frameon=False, fontsize=v["note"], **v["legend_kw"])
    ax = axes[2]
    rest = [p for p in b["points"] if p.label != "an"]
    an = [p for p in b["points"] if p.label == "an"]
    ax.scatter([p.x for p in rest], [p.y for p in rest], s=small, color=COLOURS[data.LOCKED], edgecolor="#333333",
               linewidth=thin)
    ax.scatter([p.x for p in an], [p.y for p in an], s=big, color=COLOURS[data.LOCKED], edgecolor="#000000",
               linewidth=thick)
    for p in an:
        ax.annotate(f"an, a mean over its {b['an_frames']} licensed\ncue-final frames: predicted 0.18,\n"
                    "measured 0.30 (the lowest)", (p.x, p.y), xytext=(p.x + v["offset"][0], p.y + v["offset"][1]),
                    fontsize=v["note"],
                    arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": arrow})
    _identity(ax, b["points"])
    ax.set_xlabel("locked prediction ḡ_E (encoding read)", **v["axis"])
    ax.set_ylabel("measured transport q̄_T", **v["axis"])
    ax.set_title(fit("B. 011, encoding read: 24 new cues, locked predictions", v["title"], v["fit"]), loc="left",
                 fontsize=v["title"])
    ax.text(*v["note_at"], fit("Spearman 0.848, MAE 0.077 over the 24 cues [prospective]\n"
                            "the read is weight-only apart from the frozen axis", v["note"], v["fit"], "; "),
            transform=ax.transAxes, va="top", fontsize=v["note"])
    if variant == "print":
        for ax in axes:
            ax.tick_params(labelsize=5.5, length=2.5, width=0.5)
    fig.tight_layout()
    save(fig, v["name"], print_variant=variant == "print")


# F4 -----------------------------------------------------------------------------------------------------------------
def f4(record):
    order = ["this", "a", "another", "every", "either", "these"]
    rows = record["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    y = list(range(len(order)))
    colour = COLOURS[data.EXPLORATORY]
    ax.barh([v - 0.18 for v in y], [rows[t]["f_par"] for t in order], height=0.34, color=colour,
            edgecolor="#333333", linewidth=0.5, label="axis part f_∥ (filled)")
    ax.barh([v + 0.18 for v in y], [rows[t]["f_perp"] for t in order], height=0.34, color="#FFFFFF",
            edgecolor=colour, linewidth=1.2, label="rest f_⊥ (outlined)")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of the plural cue's measured head change (means over informative frames)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "F4-opposition-ledger")


# F5 -----------------------------------------------------------------------------------------------------------------
F5_VARIANTS = {
    "screen": {"figsize": (10, 4.6), "marker": 46, "star": 90, "line": 0.6, "note": 7.5, "controls": 8, "axis": {},
               "legend_markers": (6.5, 6), "legend": 7.5, "legend_columns": 3, "fit": None, "note_x": 0.5,
               "name": "F5-closure-ladder"},
    "print": {"figsize": (6.5, 3.3), "marker": 15, "star": 32, "line": 0.4, "note": 5.3, "controls": 5.8,
              "axis": {"fontsize": 6.0}, "legend_markers": (4.0, 3.7), "legend": 5.4, "legend_columns": 2,
              "fit": 1.05, "note_x": 0.3, "name": "F5-closure-ladder-print"},
}


def f5(record, variant="screen"):
    v = F5_VARIANTS[variant]
    rungs, controls, oracle, status = record["rungs"], record["controls"], record["oracle"], record["status"]
    names = {key: f"$S_{{{key[1:]}}}$" for key in rungs}
    names.update({key: key for key in controls + [oracle]})
    positions = {r: i for i, r in enumerate(rungs)}
    positions.update({c: len(rungs) + 0.8 + i for i, c in enumerate(controls)})
    positions[oracle] = len(rungs) + len(controls) + 1.4
    (y1_decision, _), (y2_decision, _) = record["decision"]["y1_S256"], record["decision"]["y2_S256"]
    fig, ax = plt.subplots(figsize=v["figsize"])
    for key in rungs + controls + [oracle]:
        x = positions[key]
        # The locked rung is plotted at its recorded decision values, the other rungs at the ladder's two decimals.
        y1 = float(y1_decision) if key == "S256" else record["y1"][key]["c_L"]
        y2 = float(y2_decision) if key == "S256" else record["y2"][key]["c_L"]
        status1 = status["locked, exposed frames"] if key == "S256" else status["other"]
        status2 = status["locked, new frames"] if key == "S256" else status["other"]
        ax.scatter([x - 0.12], [y1], s=v["marker"], color=COLOURS[status1], edgecolor="#222222", linewidth=v["line"],
                   zorder=3)
        ax.scatter([x + 0.12], [y2], s=v["marker"], marker="D", color=COLOURS[status2], edgecolor="#222222",
                   linewidth=v["line"],
                   zorder=3)
    single, against, gap = record["single_neuron_pooled"]
    ax.scatter([positions["S1"]], [float(single)], s=v["star"], marker="*", color=COLOURS[status["single neuron"]],
               edgecolor="#222222", linewidth=v["line"], zorder=4)
    ax.annotate(f"single neuron, both sets pooled, pair-weighted: κ {single}\nvs S₂₅₆'s {against} (gap {gap}; "
                "rejected) [prospective]", (positions["S1"], float(single)), xytext=(positions["S1"] + 0.4, -0.13),
                fontsize=v["note"], arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": v["line"]})
    ax.annotate(f"locked rung: {y1_decision} [prospective]", (positions["S256"] - 0.12, float(y1_decision)),
                xytext=(positions["S256"] - 2.7, 1.02), fontsize=v["note"],
                arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": v["line"]})
    ax.annotate(fit(f"new frames: {y2_decision}; label unestablished\n(no-harm guard failed in one frame)", v["note"],
                    v["fit"]),
                (positions["S256"] + 0.12, float(y2_decision)), xytext=(positions["S256"] + v["note_x"], 0.52),
                fontsize=v["note"], arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": v["line"]})
    for divider in (len(rungs) + 0.4, len(rungs) + len(controls) + 0.9):
        ax.axvline(divider, color="#BBBBBB", linewidth=0.8)
    ax.text((positions[controls[0]] + positions[controls[-1]]) / 2, 1.07, "controls", fontsize=v["controls"],
            color="#555555", ha="center", va="center")
    ax.set_xticks([positions[k] for k in rungs + controls + [oracle]], [names[k] for k in rungs + controls + [oracle]])
    ax.set_ylabel("κ, closure of channel D's contribution (c_L)", **v["axis"])
    ax.set_ylim(-0.25, 1.12)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=v["legend_markers"][0], markerfacecolor="#FFFFFF",
                      markeredgecolor="#222222", label="new cues × exposed frames"),
               Line2D([], [], marker="D", linestyle="", markersize=v["legend_markers"][1], markerfacecolor="#FFFFFF",
                      markeredgecolor="#222222", label="new cues × new frames"),
               Patch(facecolor=COLOURS[status["locked, exposed frames"]], edgecolor="#222222",
                     label="locked rung, exposed frames [prospective]"),
               Patch(facecolor=COLOURS[status["locked, new frames"]], edgecolor="#222222",
                     label="locked rung, new frames: label unestablished"),
               Patch(facecolor=COLOURS[status["other"]], edgecolor="#222222",
                     label="other rungs, the controls and the oracle [descriptive]")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=v["legend_columns"], frameon=False,
              fontsize=v["legend"])
    if variant == "print":
        ax.tick_params(labelsize=5.6, length=2.5, width=0.5)
    fig.tight_layout()
    save(fig, v["name"], print_variant=variant == "print")


# F6 -----------------------------------------------------------------------------------------------------------------
def g_panel(ax, record, conditions):
    """023's g per condition against its floor F, with the calibration draws' range and median."""
    for i, name in enumerate(conditions):
        c = record["conditions"][name]
        low, median, high = c["draws"]
        ax.plot([low, high], [i, i], color="#BBBBBB", linewidth=5, solid_capstyle="butt", zorder=1)
        ax.plot([median, median], [i - 0.18, i + 0.18], color="#888888", linewidth=1.2, zorder=2)
        ax.plot([c["F"], c["F"]], [i - 0.3, i + 0.3], color="#000000", linewidth=1.6, zorder=3)
        ax.scatter([c["g"]], [i], s=60, color=COLOURS[record["status"][name]], edgecolor="#222222", zorder=4)
        ax.text(1.0072, i, c["result"], va="center", fontsize=7.5)
    ax.set_yticks(range(len(conditions)), conditions)
    ax.invert_yaxis()
    ax.set_xlim(0.985, 1.0165)
    ax.set_xlabel("g, the share of the gap to the ceiling recovered")


def f6(record):
    conditions = ["Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.5, 1.2, 0.8]})
    ax = axes[0]
    g_panel(ax, record, conditions)
    ax.set_title("A. 023: g (dot) against its floor F (black tick)\ngrey bar: the calibration draws' range; grey "
                 "tick: their median", loc="left", fontsize=9.5)
    ax = axes[1]
    for i, name in enumerate(conditions):
        c = record["conditions"][name]
        for key, marker, offset in (("R2_0", "o", -0.2), ("R2_1", "s", 0.0), ("R2_C", "^", 0.2)):
            ax.scatter([c[key]], [i + offset], marker=marker, s=40, color=COLOURS[data.DESCRIPTIVE],
                       edgecolor="#222222", linewidth=0.6, zorder=3)
    ax.set_yticks(range(len(conditions)), conditions)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("flattened Δc R² over pairs and nouns [descriptive]")
    ax.set_title("B. 023: E-only (●), completed (■), ceiling (▲)", loc="left", fontsize=9.5)
    ax = axes[2]
    values = record["flattened_021"]
    ax.scatter([values["Y1"], values["Y2"]], [0, 1], s=50, color=COLOURS[data.DESCRIPTIVE], edgecolor="#222222")
    for i, key in enumerate(["Y1", "Y2"]):
        ax.text(values[key] + 0.012, i, f"{values[key]}", va="center", fontsize=8)
    ax.set_yticks([0, 1], ["Y1: new cues in\nexposed frames, with\nexposed nouns", "Y2: new cues\nin new frames"])
    ax.invert_yaxis()
    ax.set_ylim(1.5, -0.5)
    ax.set_xlim(0.5, 0.7)
    ax.set_xlabel("flattened Δc R² [descriptive]")
    ax.set_title("C. 021, on 020's wiring", loc="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    status_legend(fig, [data.PROSPECTIVE, data.PROSPECTIVE_CONDITIONAL, data.UNESTABLISHED, data.DESCRIPTIVE],
                  hatched=False, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=4, fontsize=7.5)
    save(fig, "F6-end-to-end")


def f6a(record):
    """F6's panel A alone, for the body of the paper."""
    conditions = ["Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    g_panel(ax, record, conditions)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    status_legend(fig, [data.PROSPECTIVE, data.PROSPECTIVE_CONDITIONAL, data.UNESTABLISHED], hatched=False,
                  loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, fontsize=7.5)
    save(fig, "F6a-completion")


# F7 -----------------------------------------------------------------------------------------------------------------
F7_SIZE = 6.2


def f7_chip_width(label, parts, size=F7_SIZE, badge_size=BADGE_SIZE):
    if len(parts) == 1 and not parts[0][0]:
        return extent(label, size) + 0.12
    return 0.05 + sum(badge_width(c, badge_size) + BADGE_GAP for c, _ in parts) + 0.04 + extent(label, size) + 0.08


def f7_chip(ax, x, y, label, parts, height=0.2, size=F7_SIZE, badge_size=BADGE_SIZE, badge_h=BADGE_H):
    """A chip filled with its label's status, or a white chip with one badge per population."""
    width = f7_chip_width(label, parts, size, badge_size)
    if len(parts) == 1 and not parts[0][0]:
        status = parts[0][1]
        box(ax, x, y, width, height, status=status, rounding=0.03)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size,
                color=ink(COLOURS[status]), bbox=on_fill(status))
        return width
    box(ax, x, y, width, height, rounding=0.03)
    bx = x + 0.05
    for caption, status in parts:
        draw_badge(ax, bx, y + height / 2, badge_width(caption, badge_size), caption, status, badge_size, badge_h)
        bx += badge_width(caption, badge_size) + BADGE_GAP
    ax.text(bx + 0.04, y + height / 2, label, ha="left", va="center", fontsize=size)
    return width


def f7(record):
    rows = record["rows"]
    width, subject_x, chips_x = 14.0, 0.55, 4.25
    fig, ax = canvas(width)
    used = [s for s in LEGEND_ORDER if any(status == s for row in rows for _, parts in row["chips"]
                                           for _, status in parts)]
    status_legend(ax, used, loc="upper left", bbox_to_anchor=(chips_x - 0.1, 0.1), bbox_transform=ax.transData,
                  ncol=6, fontsize=7.5)
    y = 0.5
    ax.text(0.05, y, "exp", fontsize=8.5, weight="bold", va="center")
    ax.text(subject_x, y, "subject (verbatim)", fontsize=8.5, weight="bold", va="center")
    ax.text(chips_x, y, "frozen outcome (verbatim), one chip per label, coloured by its recorded status; the "
            "status, verbatim, beneath", fontsize=8.5, weight="bold", va="center")
    y += 0.2
    for row in rows:
        ax.plot([0.05, width - 0.05], [y, y], color="#E6E6E6", linewidth=0.5)
        top = y + 0.07
        subject_height = text_block(ax, subject_x, top + 0.02, row["subject"].replace("`", ""), chips_x - subject_x
                                    - 0.15, size=7.3)
        x, line_top = chips_x, top
        for label, parts in row["chips"]:
            needed = f7_chip_width(label, parts)
            if x > chips_x and x + needed > width - 0.05:
                x, line_top = chips_x, line_top + 0.26
            x += f7_chip(ax, x, line_top, label, parts) + 0.06
        for text in row["plain"]:
            ax.text(x + 0.02, line_top + 0.1, text, fontsize=7, va="center")
        status_height = text_block(ax, chips_x, line_top + 0.25, "status: " + row["status_text"],
                                   width - 0.05 - chips_x, size=6.6, color=NOTE)
        height = max(subject_height + 0.04, line_top - top + 0.27 + status_height)
        ax.text(0.05, top + 0.1, row["experiment"], fontsize=8.5, weight="bold", va="center")
        y = top + height + 0.07
    y += 0.1
    crop(fig, ax, y + 0.1)
    save(fig, "F7-sequence")


# F8 -----------------------------------------------------------------------------------------------------------------
def gate(ax, x, y, width, height, state, dashed=False, size=6.8, linewidth=0.6):
    fill = GATE_COLOURS[state]
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.04", facecolor=fill,
                                edgecolor="#444444", linewidth=linewidth, linestyle="--" if dashed else "-"))
    ax.text(x + width / 2, y + height / 2, state, ha="center", va="center", fontsize=size, color=ink(fill))


def f8_rows(record):
    """C003's items, each with C002's matching item, or None where C002 has no such item."""
    c002, c003 = record["C002"], record["C003"]
    rows = []
    iterator = iter(c002)
    for item in c003:
        if item["title"] == "Repeated or related cases examined":
            rows.append((item, None))
        else:
            rows.append((item, next(iterator)))
    assert next(iterator, None) is None
    return rows


def f8(record):
    rows = f8_rows(record)
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(rows) + 1.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(len(rows) + 0.2, -1.0)
    ax.axis("off")
    ax.text(7.35, -0.5, "C002", ha="center", fontsize=9, weight="bold")
    ax.text(9.85, -0.5, "C003", ha="center", fontsize=9, weight="bold")
    level = None
    for i, (item3, item2) in enumerate(rows):
        if item3["level"] != level:
            level = item3["level"]
            ax.plot([0, 11.2], [i, i], color="#BBBBBB", linewidth=0.6)
            ax.text(0.0, i + 0.5, f"Level {level[0]}", va="center", fontsize=7.5, weight="bold")
        if item2 and item2["title"] != item3["title"]:
            ax.text(1.0, i + 0.34, item3["title"], va="center", fontsize=7.2)
            ax.text(1.0, i + 0.7, f"C002: {item2['title']}", va="center", fontsize=6.2, color="#555555")
        else:
            ax.text(1.0, i + 0.5, item3["title"], va="center", fontsize=7.2)
        for x, item in ((6.3, item2), (8.8, item3)):
            gate(ax, x, i + 0.08, 2.1, 0.84, item["state"] if item else "no such item", dashed=item is None)
    save(fig, "F8-boundary")


def f8_print(record):
    """F8 at the text width: titles wrapped beside two gate columns."""
    rows = f8_rows(record)
    level_w, gate_w, gap = 0.48, 1.08, 0.07
    title_w = PRINT_WIDTH - level_w - 2 * (gate_w + gap) - 0.04
    gate_x = (level_w + title_w + gap, level_w + title_w + 2 * gap + gate_w)
    title_size, c002_size = 5.9, 5.3
    fig, ax = canvas(PRINT_WIDTH)
    for x, name in zip(gate_x, ("C002", "C003")):
        ax.text(x + gate_w / 2, 0.05, name, ha="center", va="top", fontsize=6.8, weight="bold")
    y, level = 0.25, None
    for item3, item2 in rows:
        title = wrap(item3["title"], title_size, title_w - 0.04)
        c002 = wrap(f"C002: {item2['title']}", c002_size, title_w - 0.04) if item2 and item2["title"] != \
            item3["title"] else []
        height = max(0.2, 0.05 + len(title) * title_size * 1.2 / 72 + len(c002) * c002_size * 1.2 / 72 + 0.04)
        if item3["level"] != level:
            level = item3["level"]
            ax.plot([0.0, PRINT_WIDTH - 0.02], [y, y], color="#BBBBBB", linewidth=0.5)
            ax.text(0.0, y + height / 2, f"Level {level[0]}", va="center", fontsize=6.2, weight="bold")
        ax.text(level_w, y + 0.04, "\n".join(title), va="top", fontsize=title_size, linespacing=1.2)
        if c002:
            ax.text(level_w, y + 0.04 + len(title) * title_size * 1.2 / 72, "\n".join(c002), va="top",
                    fontsize=c002_size, color="#555555", linespacing=1.2)
        for x, item in zip(gate_x, (item2, item3)):
            gate(ax, x, y + 0.025, gate_w, height - 0.05, item["state"] if item else "no such item",
                 dashed=item is None, size=5.6, linewidth=0.45)
        y += height
    crop(fig, ax, y + 0.04)
    save(fig, "F8-boundary-print", print_variant=True)


# Print variants -----------------------------------------------------------------------------------------------------
# F1, F1a, F2, F3, F5 and F7 are drawn for the screen, 10–15 in wide, so their text would print at 3–4 pt at the TMLR
# text width. Their print variants are drawn at that width, 6.5 in, from the same record, with text at 4.9–7.2 pt.
# Only the layout changes: F1 turns its stages into rows, F2 wraps its cells, F7 splits its rows over two pages, and
# F1a, F3 and F5 keep their layout at a smaller size, with their notes rewrapped.
PRINT_WIDTH = 6.5


def f1a_print(record):
    """F1a's stage row at the text width, with the subtitles wrapped."""
    stages = record["stages"]
    n, gap = len(stages), 0.12
    column = (PRINT_WIDTH - 0.08 - gap * (n - 1)) / n
    xs = [0.04 + i * (column + gap) for i in range(n)]
    name_size, sub_size = 6.6, 5.4
    subs = [wrap(sub, sub_size, column - 0.16) if sub else [] for _, sub in stages]
    header = 0.07 + 0.12 + max(len(lines) for lines in subs) * 0.085 + 0.06
    fig, ax = canvas(PRINT_WIDTH)
    y = 0.06
    for i, (name, _) in enumerate(stages):
        ax.add_patch(Rectangle((xs[i], y), column, header, facecolor="#F2F2F2", edgecolor="#444444", linewidth=0.6))
        if subs[i]:
            ax.text(xs[i] + column / 2, y + 0.07, name, ha="center", va="top", fontsize=name_size, weight="bold")
            ax.text(xs[i] + column / 2, y + 0.2, "\n".join(subs[i]), ha="center", va="top", fontsize=sub_size,
                    color=NOTE, linespacing=1.1)
        else:
            ax.text(xs[i] + column / 2, y + header / 2, name, ha="center", va="center", fontsize=name_size,
                    weight="bold")
        if 1 <= i < n - 1:
            ax.annotate("", xy=(xs[i + 1], y + header / 2), xytext=(xs[i] + column, y + header / 2),
                        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.6})
    box(ax, xs[0] - 0.035, y - 0.035, xs[1] + column - xs[0] + 0.07, header + 0.07, "none", edge="#666666", dashed=True)
    ax.text((xs[0] + column + xs[1]) / 2, y + header / 2, "∥", ha="center", va="center", fontsize=8)
    crop(fig, ax, y + header + 0.06)
    save(fig, "F1a-chain-schematic-print", print_variant=True)


def f1_print(record):
    stages, chips = record["stages"], record["chips"]
    x0, head_w, gap = 0.05, 0.9, 0.1  # x0 keeps the dashed frames inside the canvas
    lane_w = (PRINT_WIDTH - 2 * x0 - head_w - 2 * gap) / 2
    lane_x = {"E-patch": x0 + head_w + gap, "real swap": x0 + head_w + 2 * gap + lane_w}
    sizes = {"text_size": 6.0, "line": 0.098, "badge_size": 5.3, "badge_h": 0.13, "beside_min": 0.9, "pad_v": 0.075}
    fig, ax = canvas(PRINT_WIDTH)
    used = [s for s in LEGEND_ORDER if any(status == s for c in chips for _, status in c["badges"])]
    status_legend(ax, used, loc="upper left", bbox_to_anchor=(lane_x["E-patch"] - 0.08, 0.0),
                  bbox_transform=ax.transData, ncol=len(used), fontsize=6.0, handlelength=1.4, columnspacing=0.9)
    y = 0.3
    for lane, name, subtitle in (("E-patch", "E-patch lane", "stage by stage"),
                                 ("real swap", "real-swap lane", "end to end; only Δc was scored")):
        ax.text(lane_x[lane], y, name, fontsize=7.2, weight="bold", va="top")
        ax.text(lane_x[lane], y + 0.14, subtitle, fontsize=6.2, color=NOTE, va="top")
    y += 0.34
    link = next(c for c in chips if c["lane"] == "link")
    spans, link_y = {}, None
    for i, (stage, subtitle) in enumerate(stages):
        top, bottoms = y, []
        for lane in ("E-patch", "real swap"):
            x, cursor = lane_x[lane], top + 0.05
            here = [c for c in chips if c["lane"] == lane and c["stage"] == stage]
            if lane == "E-patch" and stage == "transport":
                here = [link] + here
            for c in here:
                layout = f1_layout(c, lane_w, **sizes)
                f1_chip(ax, x, cursor, layout)
                if c is link:
                    link_y = cursor + layout["height"] / 2
                cursor += layout["height"] + 0.035
            if lane == "real swap" and stage == "transport":
                cursor += text_block(ax, x + 0.04, cursor + 0.02, "End to end, L03.H04 is computed inside block 3, "
                                     "as part of the readout.", lane_w - 0.08, size=6.0, color=NOTE,
                                     style="italic") + 0.04
            elif not here:
                ax.text(x + lane_w / 2, cursor + 0.05, "—", ha="center", va="center", color="#999999", fontsize=7)
                cursor += 0.11
            bottoms.append(cursor)
        name_lines = wrap(stage, 7.0, head_w - 0.1)
        sub_lines = wrap(subtitle, 5.6, head_w - 0.1) if subtitle else []
        block = len(name_lines) * 0.12 + len(sub_lines) * 0.095
        bottom = max(max(bottoms) + 0.01, top + block + 0.16)  # the stage's own label sets a minimum height
        ax.add_patch(Rectangle((x0, top + 0.04), head_w, bottom - top - 0.08, facecolor="#F2F2F2",
                               edgecolor="#444444", linewidth=0.6))
        mid = (top + bottom) / 2 - block / 2
        ax.text(x0 + head_w / 2, mid, "\n".join(name_lines), ha="center", va="top", fontsize=7.0, weight="bold",
                linespacing=1.15)
        if sub_lines:
            ax.text(x0 + head_w / 2, mid + len(name_lines) * 0.12, "\n".join(sub_lines), ha="center", va="top",
                    fontsize=5.6, color=NOTE, linespacing=1.15)
        ax.plot([lane_x["E-patch"], PRINT_WIDTH - x0], [bottom, bottom], color="#DDDDDD", linewidth=0.4)
        spans[stage] = (top, bottom)
        y = bottom + 0.09
    names = [s for s, _ in stages]
    for upper, lower in zip(names[1:-1], names[2:]):  # the order of the stages; the encoding runs beside block 0
        ax.annotate("", xy=(x0 + head_w / 2, spans[lower][0] + 0.04), xytext=(x0 + head_w / 2, spans[upper][1] - 0.04),
                    arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.7})
    first, second = spans[names[0]], spans[names[1]]
    box(ax, x0 - 0.035, first[0], head_w + 0.07, second[1] - first[0], "none", edge="#666666", dashed=True)
    ax.text(x0 + head_w / 2, (first[1] + second[0]) / 2, "∥", ha="center", va="center", fontsize=9)
    readout, transport = spans[names[5]], spans[names[4]]
    box(ax, lane_x["real swap"] - 0.035, transport[0] + 0.01, lane_w + 0.07, readout[1] - transport[0] + 0.02, "none",
        edge="#666666", dashed=True)
    # The encoding read links the encoding stage to the transport: from the encoding's cell, down a gutter between
    # the stage cells' dashed frame and the E-patch lane, into the encoding-read chip.
    gutter, start = x0 + head_w + 0.065, (spans[names[0]][0] + spans[names[0]][1]) / 2
    ax.plot([x0 + head_w, gutter, gutter], [start, start, link_y], color="#444444", linewidth=0.7)
    ax.annotate("", xy=(lane_x["E-patch"], link_y), xytext=(gutter, link_y),
                arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.7})
    top = y
    ax.add_patch(Rectangle((x0, top + 0.04), head_w, 0.26, facecolor="#FFFFFF", edgecolor="#444444", linewidth=0.6))
    ax.text(x0 + head_w / 2, top + 0.17, "causal status", ha="center", va="center", fontsize=7.0, weight="bold")
    tallest = 0.0
    for lane, c in zip(("E-patch", "real swap"), (c for c in chips if c["lane"] == "causal")):
        layout = f1_layout(c, lane_w, **sizes)
        f1_chip(ax, lane_x[lane], top + 0.04, layout)
        tallest = max(tallest, layout["height"])
    crop(fig, ax, top + 0.04 + max(tallest, 0.3) + 0.06)
    save(fig, "F1-chain-print", print_variant=True)


def keep_with_value(text):
    """A metric's symbol stays on the line of its value: wrap() breaks only at ordinary spaces."""
    for symbol in ("R² ", "ρ ", "κ ", "MAE ", "Δκ₆₄ ", "A "):
        text = text.replace(symbol, symbol[:-1] + "\u00a0")
    return text


def f2_print(record):
    rows, columns, cells = record["rows"], record["columns"], record["cells"]
    label_w, size, status_size, head_size = 1.16, 5.7, 4.9, 6.3
    col_w = (PRINT_WIDTH - label_w - 0.04) / len(columns)
    fig, ax = canvas(PRINT_WIDTH)
    used = [s for s in LEGEND_ORDER if any(cell.status == s for cell in cells)]
    status_legend(ax, used, loc="upper left", bbox_to_anchor=(label_w, 0.0), bbox_transform=ax.transData,
                  ncol=len(used), fontsize=6.2, handlelength=1.4, columnspacing=0.9)
    y, head_h = 0.27, 0.0
    for j, column in enumerate(columns):
        lines = wrap(column, head_size * 1.08, col_w - 0.1)  # bold runs about 8 % wider
        ax.text(label_w + 0.04 + col_w * j + col_w / 2, y, "\n".join(lines), ha="center", va="top",
                fontsize=head_size, weight="bold", linespacing=1.15)
        head_h = max(head_h, len(lines) * head_size * 1.15 / 72)
    y += head_h + 0.06
    line_h = size * 1.12 / 72
    for row in rows:
        here = {cell.column: cell for cell in cells if cell.row == row}
        lines = {c: [w for part in cell.text.replace("-\n", "-").split("\n")
                     for w in wrap(keep_with_value(part), size, col_w - 0.14)] for c, cell in here.items()}
        height = 0.04 + max([len(v) for v in lines.values()] + [1]) * line_h + status_size * 1.05 / 72 + 0.04
        label = wrap(row, 5.9, label_w - 0.08)
        ax.text(label_w - 0.05, y + height / 2, "\n".join(label), ha="right", va="center", fontsize=5.9,
                linespacing=1.1)
        for j, column in enumerate(columns):
            x = label_w + 0.04 + col_w * j
            ax.add_patch(Rectangle((x + 0.015, y + 0.015), col_w - 0.03, height - 0.03, facecolor="#FAFAFA",
                                   edgecolor="#E0E0E0", linewidth=0.4))
            cell = here.get(column)
            if cell is None:
                continue
            fill = COLOURS[cell.status]
            box(ax, x + 0.025, y + 0.025, col_w - 0.05, height - 0.05, status=cell.status, rounding=0.02,
                linewidth=0.5)
            ax.text(x + col_w / 2, y + 0.04, "\n".join(lines[column]), ha="center", va="top", fontsize=size,
                    color=ink(fill), linespacing=1.15, bbox=on_fill(cell.status))
            ax.text(x + col_w - 0.05, y + height - 0.04, cell.status, ha="right", va="bottom", fontsize=status_size,
                    style="italic", color=ink(fill), bbox=on_fill(cell.status))
        y += height
    crop(fig, ax, y + 0.05)
    save(fig, "F2-prediction-map-print", print_variant=True)


def f7_print(record):
    rows = record["rows"]
    size, subject_size, status_size, head_size = 5.8, 6.0, 5.5, 6.6
    badge_size, chip_h, badge_h = 5.2, 0.15, 0.12
    exp_w, subject_w = 0.3, 1.4
    chips_x = exp_w + subject_w + 0.08
    used = [s for s in LEGEND_ORDER if any(status == s for row in rows for _, parts in row["chips"]
                                           for _, status in parts)]
    for part, subset in enumerate((rows[:11], rows[11:]), start=1):
        fig, ax = canvas(PRINT_WIDTH)
        status_legend(ax, used, loc="upper left", bbox_to_anchor=(chips_x - 0.06, 0.0), bbox_transform=ax.transData,
                      ncol=len(used), fontsize=6.0, handlelength=1.4, columnspacing=0.9)
        y = 0.3
        ax.text(0.0, y, "exp", fontsize=head_size, weight="bold", va="top")
        ax.text(exp_w, y, "subject (verbatim)", fontsize=head_size, weight="bold", va="top")
        head = wrap("frozen outcome (verbatim), one chip per label, coloured by its recorded status; the status, "
                    "verbatim, beneath", head_size * 1.08, PRINT_WIDTH - chips_x)
        ax.text(chips_x, y, "\n".join(head), fontsize=head_size, weight="bold", va="top", linespacing=1.15)
        y += len(head) * head_size * 1.15 / 72 + 0.06
        for row in subset:
            ax.plot([0.0, PRINT_WIDTH], [y, y], color="#E6E6E6", linewidth=0.4)
            top = y + 0.05
            subject_height = text_block(ax, exp_w, top, row["subject"].replace("`", ""), subject_w - 0.05,
                                        size=subject_size)
            x, line_top = chips_x, top
            for label, parts in row["chips"]:
                if x > chips_x and x + f7_chip_width(label, parts, size, badge_size) > PRINT_WIDTH:
                    x, line_top = chips_x, line_top + chip_h + 0.04
                x += f7_chip(ax, x, line_top, label, parts, chip_h, size, badge_size, badge_h) + 0.04
            for text in row["plain"]:
                ax.text(x + 0.02, line_top + chip_h / 2, text, fontsize=size, va="center")
            status_height = text_block(ax, chips_x, line_top + chip_h + 0.025, "status: " + row["status_text"],
                                       PRINT_WIDTH - chips_x, size=status_size, color=NOTE)
            height = max(subject_height, line_top - top + chip_h + 0.025 + status_height) + 0.04
            ax.text(0.0, top + chip_h / 2, row["experiment"], fontsize=head_size, weight="bold", va="center")
            y = top + height
        crop(fig, ax, y + 0.05)
        save(fig, f"F7-sequence-print-{part}", print_variant=True)


def main():
    record = data.everything()
    f1(record["F1"])
    f1a(record["F1"])
    f2(record["F2"])
    f3(record["F3"])
    f4(record["F4"])
    f5(record["F5"])
    f6(record["F6"])
    f6a(record["F6"])
    f7(record["F7"])
    f8(record["F8"])
    f1a_print(record["F1"])
    f1_print(record["F1"])
    f2_print(record["F2"])
    f3(record["F3"], "print")
    f5(record["F5"], "print")
    f7_print(record["F7"])
    f8_print(record["F8"])
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("F*.svg"))), "and the print variants",
          ", ".join(sorted(p.name for p in OUT.glob("F*-print*.pdf"))))


if __name__ == "__main__":
    main()
