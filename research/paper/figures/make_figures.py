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


def save(fig, name: str) -> None:
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


def badge_width(caption: str) -> float:
    return max(0.15, extent(caption, BADGE_SIZE) + 0.08) if caption else 0.15


def draw_badge(ax, x, middle, width, caption, status):
    box(ax, x, middle - BADGE_H / 2, width, BADGE_H, status=status, edge="#333333", rounding=0.03, linewidth=0.5)
    if caption:
        ax.text(x + width / 2, middle, caption, ha="center", va="center", fontsize=BADGE_SIZE,
                color=ink(COLOURS[status]), bbox=on_fill(status))


# F1 -----------------------------------------------------------------------------------------------------------------
F1_TEXT, F1_LINE, F1_PAD = 7.0, 0.125, 0.05


def f1_layout(chip, width=None):
    """Badges beside the text, or in rows above it when they would leave the text less than 0.95 in."""
    badges = [(caption, status, badge_width(caption)) for caption, status in chip["badges"]]
    in_line = sum(b[2] for b in badges) + BADGE_GAP * (len(badges) - 1)
    if width is None:
        width = in_line + extent(chip["text"], F1_TEXT) + 2 * F1_PAD + 0.08
    beside = width - in_line - 2 * F1_PAD - 0.06
    if beside >= 0.95:
        text_lines = wrap(chip["text"], F1_TEXT, beside)
        return {"rows": [badges], "lines": text_lines, "width": width, "stacked": False,
                "height": 0.14 + F1_LINE * len(text_lines)}
    rows, used = [[]], 0.0
    for b in badges:
        if rows[-1] and used + BADGE_GAP + b[2] > width - 2 * F1_PAD:
            rows.append([])
            used = 0.0
        used += (BADGE_GAP if rows[-1] else 0) + b[2]
        rows[-1].append(b)
    text_lines = wrap(chip["text"], F1_TEXT, width - 2 * F1_PAD)
    return {"rows": rows, "lines": text_lines, "width": width, "stacked": True,
            "height": 0.14 + len(rows) * (BADGE_H + 0.04) + F1_LINE * len(text_lines)}


def f1_chip(ax, x, y, layout):
    box(ax, x, y, layout["width"], layout["height"], edge="#888888")
    text = "\n".join(layout["lines"])
    if not layout["stacked"]:
        middle, bx = y + layout["height"] / 2, x + F1_PAD
        for caption, status, width in layout["rows"][0]:
            draw_badge(ax, bx, middle, width, caption, status)
            bx += width + BADGE_GAP
        ax.text(bx + 0.03, middle, text, ha="left", va="center", fontsize=F1_TEXT, linespacing=1.15)
        return
    top = y + 0.06
    for row in layout["rows"]:
        bx = x + F1_PAD
        for caption, status, width in row:
            draw_badge(ax, bx, top + BADGE_H / 2, width, caption, status)
            bx += width + BADGE_GAP
        top += BADGE_H + 0.04
    ax.text(x + F1_PAD + 0.02, top + 0.01, text, ha="left", va="top", fontsize=F1_TEXT, linespacing=1.15)


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


def f3(record):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    a1, a2, b = record["A1"], record["A2"], record["B"]
    ax = axes[0]
    ax.scatter([p.x for p in a1["points"]], [p.y for p in a1["points"]], s=26, color=COLOURS[data.DESCRIPTIVE],
               edgecolor="#333333", linewidth=0.5)
    _identity(ax, a1["points"])
    ax.set_xlabel("P1 read-out, token mean")
    ax.set_ylabel("measured transport q_T, token mean")
    ax.set_title("A1. 009: 23 token means [descriptive]", loc="left", fontsize=9.5)
    ax.text(0.02, 0.98, "009's read-out test was scored over 138 pairs\n(not committed, not shown): Spearman 0.957, MAE 0.066\n"
            "[prospective, given the measured input]", transform=ax.transAxes, va="top", fontsize=7.5)
    ax = axes[1]
    rest = [p for p in a2["points"] if p.label != "an"]
    an = [p for p in a2["points"] if p.label == "an"]
    ax.scatter([p.x for p in rest], [p.y for p in rest], s=14, color=COLOURS[data.PROSPECTIVE], alpha=0.8,
               edgecolor="none")
    ax.scatter([p.x for p in an], [p.y for p in an], s=40, facecolor="none", edgecolor="#000000", linewidth=1.0,
               label=f"an, its {a2['an_frames']} licensed cue-final frames")
    _identity(ax, a2["points"])
    ax.set_xlabel("P1 read-out of the measured head input")
    ax.set_ylabel("measured transport q_T")
    ax.set_title("A2. 011, read-out replicated: the 142 scored pairs", loc="left", fontsize=9.5)
    ax.text(0.02, 0.98, "Spearman 0.974, MAE 0.076 over these pairs\n[prospective, given the measured input]",
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax = axes[2]
    rest = [p for p in b["points"] if p.label != "an"]
    an = [p for p in b["points"] if p.label == "an"]
    ax.scatter([p.x for p in rest], [p.y for p in rest], s=26, color=COLOURS[data.LOCKED], edgecolor="#333333",
               linewidth=0.5)
    ax.scatter([p.x for p in an], [p.y for p in an], s=60, color=COLOURS[data.LOCKED], edgecolor="#000000",
               linewidth=1.4)
    for p in an:
        ax.annotate(f"an, a mean over its {b['an_frames']} licensed\ncue-final frames: predicted 0.18,\n"
                    "measured 0.30 (the lowest)", (p.x, p.y), xytext=(p.x + 0.12, p.y - 0.12), fontsize=7.5,
                    arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.6})
    _identity(ax, b["points"])
    ax.set_xlabel("locked prediction ḡ_E (encoding read)")
    ax.set_ylabel("measured transport q̄_T")
    ax.set_title("B. 011, encoding read: 24 new cues, locked predictions", loc="left", fontsize=9.5)
    ax.text(0.02, 0.98, "Spearman 0.848, MAE 0.077 over the 24 cues [prospective]\n"
            "the read is weight-only apart from the frozen axis", transform=ax.transAxes, va="top", fontsize=7.5)
    fig.tight_layout()
    save(fig, "F3-head")


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
def f5(record):
    rungs, controls, oracle, status = record["rungs"], record["controls"], record["oracle"], record["status"]
    names = {key: f"$S_{{{key[1:]}}}$" for key in rungs}
    names.update({key: key for key in controls + [oracle]})
    positions = {r: i for i, r in enumerate(rungs)}
    positions.update({c: len(rungs) + 0.8 + i for i, c in enumerate(controls)})
    positions[oracle] = len(rungs) + len(controls) + 1.4
    (y1_decision, _), (y2_decision, _) = record["decision"]["y1_S256"], record["decision"]["y2_S256"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for key in rungs + controls + [oracle]:
        x = positions[key]
        # The locked rung is plotted at its recorded decision values, the other rungs at the ladder's two decimals.
        y1 = float(y1_decision) if key == "S256" else record["y1"][key]["c_L"]
        y2 = float(y2_decision) if key == "S256" else record["y2"][key]["c_L"]
        status1 = status["locked, exposed frames"] if key == "S256" else status["other"]
        status2 = status["locked, new frames"] if key == "S256" else status["other"]
        ax.scatter([x - 0.12], [y1], s=46, color=COLOURS[status1], edgecolor="#222222", linewidth=0.6, zorder=3)
        ax.scatter([x + 0.12], [y2], s=46, marker="D", color=COLOURS[status2], edgecolor="#222222", linewidth=0.6,
                   zorder=3)
    single, against, gap = record["single_neuron_pooled"]
    ax.scatter([positions["S1"]], [float(single)], s=90, marker="*", color=COLOURS[status["single neuron"]],
               edgecolor="#222222", linewidth=0.6, zorder=4)
    ax.annotate(f"single neuron, both sets pooled, pair-weighted: κ {single}\nvs S₂₅₆'s {against} (gap {gap}; "
                "rejected) [prospective]", (positions["S1"], float(single)), xytext=(positions["S1"] + 0.4, -0.13),
                fontsize=7.5, arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.6})
    ax.annotate(f"locked rung: {y1_decision} [prospective]", (positions["S256"] - 0.12, float(y1_decision)),
                xytext=(positions["S256"] - 2.7, 1.02), fontsize=7.5,
                arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.6})
    ax.annotate(f"new frames: {y2_decision}; label unestablished\n(no-harm guard failed in one frame)",
                (positions["S256"] + 0.12, float(y2_decision)), xytext=(positions["S256"] + 0.5, 0.52), fontsize=7.5,
                arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.6})
    for divider in (len(rungs) + 0.4, len(rungs) + len(controls) + 0.9):
        ax.axvline(divider, color="#BBBBBB", linewidth=0.8)
    ax.text((positions[controls[0]] + positions[controls[-1]]) / 2, 1.07, "controls", fontsize=8, color="#555555",
            ha="center", va="center")
    ax.set_xticks([positions[k] for k in rungs + controls + [oracle]], [names[k] for k in rungs + controls + [oracle]])
    ax.set_ylabel("κ, closure of channel D's contribution (c_L)")
    ax.set_ylim(-0.25, 1.12)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6.5, markerfacecolor="#FFFFFF",
                      markeredgecolor="#222222", label="new cues × exposed frames"),
               Line2D([], [], marker="D", linestyle="", markersize=6, markerfacecolor="#FFFFFF",
                      markeredgecolor="#222222", label="new cues × new frames"),
               Patch(facecolor=COLOURS[status["locked, exposed frames"]], edgecolor="#222222",
                     label="locked rung, exposed frames [prospective]"),
               Patch(facecolor=COLOURS[status["locked, new frames"]], edgecolor="#222222",
                     label="locked rung, new frames: label unestablished"),
               Patch(facecolor=COLOURS[status["other"]], edgecolor="#222222",
                     label="other rungs, the controls and the oracle [descriptive]")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "F5-closure-ladder")


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


def f7_chip_width(label, parts):
    if len(parts) == 1 and not parts[0][0]:
        return extent(label, F7_SIZE) + 0.12
    return 0.05 + sum(badge_width(c) + BADGE_GAP for c, _ in parts) + 0.04 + extent(label, F7_SIZE) + 0.08


def f7_chip(ax, x, y, label, parts, height=0.2):
    """A chip filled with its label's status, or a white chip with one badge per population."""
    width = f7_chip_width(label, parts)
    if len(parts) == 1 and not parts[0][0]:
        status = parts[0][1]
        box(ax, x, y, width, height, status=status, rounding=0.03)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=F7_SIZE,
                color=ink(COLOURS[status]), bbox=on_fill(status))
        return width
    box(ax, x, y, width, height, rounding=0.03)
    bx = x + 0.05
    for caption, status in parts:
        draw_badge(ax, bx, y + height / 2, badge_width(caption), caption, status)
        bx += badge_width(caption) + BADGE_GAP
    ax.text(bx + 0.04, y + height / 2, label, ha="left", va="center", fontsize=F7_SIZE)
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
def gate(ax, x, y, width, height, state, dashed=False):
    fill = GATE_COLOURS[state]
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.04", facecolor=fill,
                                edgecolor="#444444", linewidth=0.6, linestyle="--" if dashed else "-"))
    ax.text(x + width / 2, y + height / 2, state, ha="center", va="center", fontsize=6.8, color=ink(fill))


def f8(record):
    c002, c003 = record["C002"], record["C003"]
    rows = []
    iterator = iter(c002)
    for item in c003:
        if item["title"] == "Repeated or related cases examined":
            rows.append((item, None))
        else:
            rows.append((item, next(iterator)))
    assert next(iterator, None) is None
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
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("F*.svg"))))


if __name__ == "__main__":
    main()
