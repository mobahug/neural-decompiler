#!/usr/bin/env python3
"""Every value the paper's figures plot, read from the committed record. Read-only.

The values follow the figure-data audit (research/paper/figures/figure-data-audit-2026-09-30.md). Each is read with
`git show <commit>:<path>` from a pinned commit, never from a working copy and never from `outputs/`. Each is checked
against an anchor: the header of the table it comes from, or a string that must stand on the cited line. Nothing is
computed: values are parsed and passed on as recorded, and the only transformations the figures apply are
presentational (selection, order, labels, colour by recorded status). Run it to check every source:

    .venv/bin/python research/paper/figures/figure_data.py
"""
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PIN = "d7c5e62"  # the record's pin: experiment READMEs, reports and extracts
SOURCE_MAP = ("research/synthesis/source-map-005-025.md", "b722e9c")
C002 = ("research/claims/C002-count-cued-noun-number-circuit.md", "454327d")
C003 = ("research/claims/C003-frame-conditioned-decoded-program.md", "eeff9ff")

# The recorded evidence statuses (source map, lines 33–45) and the gate states the claims use.
PROSPECTIVE = "prospective"
PROSPECTIVE_CONDITIONAL = "prospective, conditional"
LOCKED = "locked prediction"
EXPLORATORY = "exploratory"
DESCRIPTIVE = "descriptive"
UNESTABLISHED = "unestablished"

_cache: dict[tuple[str, str], list[str]] = {}


def lines(path: str, rev: str = PIN) -> list[str]:
    """The file's lines at a commit, 1-based through `line`."""
    if (path, rev) not in _cache:
        shown = subprocess.run(["git", "--no-optional-locks", "show", f"{rev}:{path}"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        _cache[(path, rev)] = shown.stdout.split("\n")
    return _cache[(path, rev)]


def line(path: str, number: int, rev: str = PIN) -> str:
    return lines(path, rev)[number - 1]


def readme(experiment: str) -> str:
    listed = subprocess.run(["git", "--no-optional-locks", "ls-tree", "--name-only", PIN, "experiments/"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    names = [name for name in listed if name.split("/")[-1].startswith(experiment + "-")]
    assert len(names) == 1, experiment
    return f"{names[0]}/README.md"


def _space(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def expect(path: str, first: int, last: int, *anchors: str, rev: str = PIN) -> None:
    """Every anchor must stand in the cited lines, as the audit cites them; runs of whitespace compare as one space."""
    text = _space(" ".join(lines(path, rev)[first - 1:last]))
    for anchor in anchors:
        assert _space(anchor) in text, f"{path}@{rev}:{first}–{last} lacks {anchor!r}"


def table_row(path: str, number: int, rev: str = PIN) -> list[str]:
    text = line(path, number, rev).strip()
    assert text.startswith("|") and text.endswith("|"), f"{path}:{number} is not a table row"
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", text[1:-1])]


def number(cell: str) -> float:
    return float(cell.replace("−", "-"))


# Every colour is a status the record gives. The source map's section "Evidence status, by experiment" lists each
# experiment's findings under the five category headers (research/synthesis/source-map-005-025.md@b722e9c:97–248).
STATUS_HEADERS = {PROSPECTIVE: "Prospective confirmation", PROSPECTIVE_CONDITIONAL: "Prospective confirmation",
                  LOCKED: "Locked prediction", EXPLORATORY: "Exploratory", DESCRIPTIVE: "Descriptive / post hoc",
                  UNESTABLISHED: "Unestablished"}


def _plain(cell: str) -> str:
    return cell.replace("`", "").replace("\\|", "|")


def _flat(text: str) -> str:
    return _space(_plain(text)).strip()


def _status_bullets(experiment: str) -> list[str]:
    """The experiment's block in the source map's status section, one string per top-level bullet."""
    path, rev = SOURCE_MAP
    text = lines(path, rev)
    marker = f"- **{experiment}.**"
    start = next(i for i, t in enumerate(text) if t.startswith(marker))
    assert 97 <= start + 1 <= 248, (experiment, start + 1)
    bullets = [text[start][len(marker):]]
    for t in text[start + 1:]:
        if t.startswith("- **") or t.startswith("## "):
            break
        if t.startswith("  - "):
            bullets.append(t[4:])
        else:
            bullets[-1] += " " + t
    return [_flat(b) for b in bullets]


def status_of(experiment: str, status: str, anchor: str) -> None:
    """The experiment's status block must list `anchor` under the header of `status`."""
    header = STATUS_HEADERS[status]
    assert any(b.startswith(header) and _flat(anchor) in b for b in _status_bullets(experiment)), \
        f"source map, status block {experiment}: no {header!r} bullet with {anchor!r}"
    if status == PROSPECTIVE_CONDITIONAL:
        assert "conditional" in anchor, (experiment, anchor)
    if status == PROSPECTIVE:
        assert "conditional" not in anchor, (experiment, anchor)


def overview(experiment: str) -> tuple[str, str]:
    """The overview's outcome and status cells for an experiment, without Markdown code marks."""
    path, rev = SOURCE_MAP
    for n in list(range(74, 93)) + [94, 95]:
        cells = table_row(path, n, rev)
        if cells[0] == experiment:
            return _plain(cells[3]), _plain(cells[4])
    raise KeyError(experiment)


def cite(path: str, first: int, last: int, *anchors: str, rev: str = PIN) -> tuple:
    return ("lines", path, first, last, anchors, rev)


@dataclass
class Point:
    label: str
    x: float
    y: float
    status: str
    source: str
    note: str = ""


# F3, panel A1: 009's P1 read-out against the measured transport, as token means [descriptive].
def f3_a1() -> dict:
    path = "experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md"
    header = table_row(path, 75)
    assert header[0] == "token" and header[3] == "q_T measured" and header[4] == "q_T predicted" and header[7] == "P1"
    points = []
    for n in range(77, 100):
        row = table_row(path, n)
        points.append(Point(row[0], number(row[7]), number(row[3]), DESCRIPTIVE, f"{path}:{n}"))
    assert len(points) == 23
    expect(readme("009"), 62, 64, "138 fresh", "0.957", "0.066")
    return {"points": points, "result": "Spearman 0.957, MAE 0.066 over 138 pairs, not shown",
            "status": "[prospective, given the measured input] for the pairs; the token means are descriptive",
            "source": f"{path}:75–99; 009:62–64"}


def _locked_means() -> dict:
    """011's locked token means: the bold rows of its predictions file."""
    path = "experiments/011-encoding-read-prospective/predictions.md"
    means = {}
    for n, text in enumerate(lines(path), 1):
        match = re.match(r"\| \*\*(\S+)\*\* \| \| \*\*mean over (\d+) licensed frames\*\* \| \*\*(-?\d+\.\d+)\*\* \|", text)
        if match:
            means[match.group(1)] = {"g_E": match.group(3), "frames": int(match.group(2)), "source": f"{path}:{n}"}
    assert len(means) == 24, len(means)
    return means


# F3, panel A2: 011 Y2's 142 committed pairs, P1 read against the measured transport.
def f3_a2() -> dict:
    path = "experiments/012-layer-correction-token-local/inherited/experiment-011-component-fractions.json"
    record = json.loads("\n".join(lines(path)))
    assert record["source"]["state_sha256"].startswith("6a73a080")
    expect(readme("011"), 65, 65, "6a73a080")
    expect(readme("012"), 137, 137, "011 (142 pairs)")
    points = []
    for key in sorted(record["entries"]):
        entry = record["entries"][key]
        token = key.split("|")[0]
        points.append(Point(token, entry["f_total"], entry["q_T"], PROSPECTIVE, f"{path}#entries[{key}]"))
    assert len(points) == 142
    an_frames = _locked_means()["an"]["frames"]
    assert sum(p.label == "an" for p in points) == an_frames == 4
    expect(readme("011"), 87, 89, "142 fresh", "0.974", "0.076")
    return {"points": points, "an_frames": an_frames, "result": "Spearman 0.974, MAE 0.076 over these 142 pairs",
            "status": "[prospective, given the measured input]", "source": f"{path}; 011:87–89"}


# F3, panel B: 011's encoding read on 24 new cues, locked predictions against measured means.
def f3_b() -> dict:
    path = "experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md"
    header = table_row(path, 39)
    assert header[0] == "token" and header[4] == "predicted ḡ_E" and header[5] == "measured q̄_T"
    locked, points = _locked_means(), []
    for n in range(41, 65):
        row = table_row(path, n)
        assert row[4] == locked[row[0]]["g_E"], (row[0], row[4], locked[row[0]]["g_E"])
        points.append(Point(row[0], number(locked[row[0]]["g_E"]), number(row[5]), LOCKED,
                            f"{locked[row[0]]['source']}; {path}:{n}"))
    assert len(points) == 24
    expect(readme("011"), 75, 79, "0.848", "0.077", "`an` 0.18", "0.30")
    expect(readme("011"), 18, 18, "`an` licensed in the four cue-final")
    expect(readme("010"), 44, 45, "weight-only", "apart from the frozen axis")
    status_of("011", LOCKED, "the per-token predictions")
    status_of("011", PROSPECTIVE, "Y1, the encoding read")
    return {"points": points, "an_frames": locked["an"]["frames"], "result": "Spearman 0.848, MAE 0.077 over the 24 cues",
            "status": "locked predictions; Y1 [prospective]", "source": f"{path}:39–64; 011:75–79"}


# F4: the opposition ledger, token means over 24 frames [exploratory].
def f4() -> dict:
    path = "experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md"
    expect(path, 25, 25, "fractions of the plural cue's measured head change; means over informative frames")
    header = table_row(path, 27)
    assert header[:6] == ["token", "category", "q_T", "f_E", "f_∥", "f_⊥"]
    rows = {}
    for n, token in [(30, "every"), (31, "this"), (32, "another"), (33, "a"), (45, "either"), (89, "these")]:
        row = table_row(path, n)
        assert row[0] == token, (n, row[0])
        rows[token] = {"q_T": number(row[2]), "f_E": number(row[3]), "f_par": number(row[4]),
                       "f_perp": number(row[5]), "f_perp_text": row[5], "category": row[1], "source": f"{path}:{n}"}
    assert rows["either"]["category"] == "singular-selecting" and rows["either"]["f_perp_text"] == "0.156"
    expect(readme("010"), 85, 89, "24 of 24 frames")
    return {"rows": rows, "status": EXPLORATORY, "source": f"{path}:27–33, 45, 89; 010:85–89"}


# F5: channel D's closure ladder, κ of c_L per rung, from 018's final report.
def _ladder(path: str, first: int, last: int) -> dict:
    ladder = {}
    for n in range(first, last + 1):
        match = re.match(r"\s*- `(\w+)`: (.*)$", line(path, n))
        assert match, f"{path}:{n}"
        kappas = re.findall(r"\((-?\d+\.\d+)\)", match.group(2))
        assert len(kappas) == 5, f"{path}:{n}"
        ladder[match.group(1)] = {"c_L": float(kappas[0]), "Pi": float(kappas[2]), "source": f"{path}:{n}"}
    return ladder


def f5() -> dict:
    path = "experiments/018-block2-concentration/evidence/final-report-2026-09-21.md"
    expect(path, 68, 68, "pooled pairs (1872 pairs)", "c_L R² (κ) / F R² (κ) / Π R² (κ)")
    expect(path, 91, 91, "pooled pairs (288 pairs)", "c_L R² (κ) / F R² (κ) / Π R² (κ)")
    y1 = _ladder(path, 69, 81)
    y2 = _ladder(path, 92, 104)
    expect(path, 66, 66, "κ_c_L(S_256) 0.863", "κ_Π(S_256) 0.724")
    expect(path, 89, 89, "κ_c_L(S_256) 0.843", "κ_Π(S_256) 0.798", "no-harm guard FAILED")
    expect(path, 124, 124, "κ_c_L(S_1) 0.219", "S_256 0.860", "gap 0.641")
    expect(readme("018"), 102, 104, "0.863", "0.724")
    rungs = ["S0", "S1", "S4", "S16", "S64", "S256", "S1024", "S2048"]
    controls = ["R1", "R2", "R3", "B256"]
    expect(readme("018"), 15, 16, "three seeded random 256-subsets", "the 256 lowest-ranked (`B256`) are the controls")
    expect(readme("018"), 55, 55, "the frame's own subset (the oracle, descriptive)")
    expect(readme("018"), 118, 119, "`κ_{c_L}(S_1 = {1987})` **0.219**", "against `S_256`'s 0.860", "**0.641**")
    expect("docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md", 217, 217,
           "The ablation ladder and controls (descriptive")
    status_of("018", PROSPECTIVE, "Y1, the concentration on unseen cues")
    status_of("018", UNESTABLISHED, "Y2 (conditional): the no-harm guard failed in one new frame")
    status_of("018", DESCRIPTIVE, "the rung ladder")
    status_of("018", PROSPECTIVE, "Y3, the rejection of the single-neuron account, pooled over both sets and pair-weighted")
    status = {"locked, exposed frames": PROSPECTIVE, "locked, new frames": UNESTABLISHED, "other": DESCRIPTIVE,
              "single neuron": PROSPECTIVE}
    return {"y1": y1, "y2": y2, "rungs": rungs, "controls": controls, "oracle": "oracle", "status": status,
            "decision": {"y1_S256": ("0.863", "0.724"), "y2_S256": ("0.843", "0.798")},
            "single_neuron_pooled": ("0.219", "0.860", "0.641"),
            "source": f"{path}:66–81, 89–104, 124; 018:102–104"}


# F6: 023's four conditions and draws, and 021's descriptive flattened R² on 020's wiring.
def f6() -> dict:
    path = "experiments/023-block0-completion/evidence/final-report-2026-09-24.md"
    header = table_row(path, 28)
    assert header[:9] == ["condition", "g", "envelope F", "meaning guard", "result", "gap (SSE0 − SSEC)/SST", "R²₀",
                          "R²₁", "R²_C"], header
    conditions = {}
    for n in range(30, 34):
        row = table_row(path, n)
        conditions[row[0]] = {"g": number(row[1]), "F": number(row[2]), "result": row[4].strip("*"),
                              "R2_0": number(row[6]), "R2_1": number(row[7]), "R2_C": number(row[8]),
                              "source": f"{path}:{n}"}
    draws_header = table_row(path, 17)
    assert draws_header[:5] == ["condition", "envelope F", "min of defined draws", "median", "max"], draws_header
    for n in range(19, 23):
        row = table_row(path, n)
        conditions[row[0]]["draws"] = (number(row[2]), number(row[3]), number(row[4]))
    status = {"Y1/cue_final": UNESTABLISHED, "Y1/coordinated": PROSPECTIVE, "Y2/cue_final": PROSPECTIVE_CONDITIONAL,
              "Y2/coordinated": PROSPECTIVE_CONDITIONAL}
    for name, condition in conditions.items():
        passed = {"PASS": True, "ENVELOPE_ONLY_FAILURE": False}[condition["result"]]
        assert status[name] == (UNESTABLISHED if not passed else
                                PROSPECTIVE_CONDITIONAL if name.startswith("Y2") else PROSPECTIVE), name
    status_of("023", UNESTABLISHED, "Y1/cue_final at its envelope")
    status_of("023", PROSPECTIVE, "Y1/coordinated")
    status_of("023", PROSPECTIVE_CONDITIONAL, "Y2/cue_final and Y2/coordinated, conditional")
    expect(SOURCE_MAP[0], 92, 92, "three prospective confirmations (Y2 conditional)",
           "one unestablished at its envelope (guard met)", rev=SOURCE_MAP[1])
    readme021 = readme("021")
    expect(readme021, 132, 132, "flattened `R²` 0.6167 (descriptively")
    expect(readme021, 137, 137, "Flattened `R²` 0.5797 (descriptively")
    expect(readme021, 13, 14, "through `p_t` instead of the cue position `p_c`")
    return {"conditions": conditions, "status": status,
            "flattened_021": {"Y1": 0.6167, "Y2": 0.5797}, "flattened_status": DESCRIPTIVE,
            "source": f"{path}:17–33; 021:132, 137; source map:92"}


# F7: the sequence, from the source map's overview. Each chip is a label verbatim from the outcome cell, or for 006 and
# 020 the status cell's words for the confirmatory test that was never run. A chip has one status, or one badge per
# population where its conditions differ in status. Every status is checked against the source map's status block of
# the experiment.
AXIS_ONLY = "Y3, the rejection of the axis-only alternative on both sets pooled"
F7_LABELS = {
    "005": [("v1 NO_COMPACT_MECHANISM", [("", EXPLORATORY, "the v1 program-floor failure")]),
            ("v2 BEHAVIOR_NOT_REPLICATED", [("", UNESTABLISHED, "BEHAVIOR_NOT_REPLICATED")])],
    "006": [("QUALITY_GATE_FAILED", [("", EXPLORATORY, "the pre-lock quality gate")]),
            ("the confirmatory test was never run", [("", UNESTABLISHED,
                                                      "No lock could be written, so none was tested")])],
    "007": [("PROGRAM_NOT_SUPPORTED (Y1 failed)", [("", UNESTABLISHED, "at label level: PROGRAM_NOT_SUPPORTED")])],
    "008": [("MIXED", [("", EXPLORATORY, "everything")])],
    "009": [("HEAD_MECHANISM_CONFIRMED_P1", [("", PROSPECTIVE, "Y1, the head's P1 read-out given its measured input")]),
            ("TRANSPORT_RULE_FAILED", [("", UNESTABLISHED, "Y2, the transport rule from E")]),
            ("NOT_LOCKED", [("", UNESTABLISHED, "Y3, never locked")])],
    "010": [("MIXED", [("", EXPLORATORY, "the MIXED summary")])],
    "011": [("ENCODING_READ_PREDICTS_TRANSPORT", [("", PROSPECTIVE, "Y1, the encoding read")]),
            ("HEAD_P1_REPLICATED", [("", PROSPECTIVE, "Y2, P1 replicated given the measured head input")])],
    "012": [("LAYER_CORRECTION_TOKEN_LOCAL_MLP", [("", PROSPECTIVE, "Y1 (0.864, 0.061, R² 0.795)")]),
            ("COMPOSITE_PREDICTS_P1", [("", PROSPECTIVE, "Y2 (0.922, 0.062)")])],
    "013": [("RESIDUAL_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1;")]),
            ("RESIDUAL_PREDICTED_FRAMES_CONDITIONAL", [("", PROSPECTIVE_CONDITIONAL, "Y2 (conditional)")]),
            ("ATTRIBUTION_PREDICTED", [("", PROSPECTIVE, "Y3, the attribution, pooled over both sets")])],
    "014": [("NEURON_FEATURE_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1;")]),
            ("NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL", [("", PROSPECTIVE_CONDITIONAL, "Y2 (conditional)")]),
            ("AXIS_ONLY_REJECTED", [("", PROSPECTIVE, AXIS_ONLY)])],
    "015": [("PATTERN_CHANGE_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1;")]),
            ("PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL", [("", UNESTABLISHED, "Y2 (conditional)")]),
            ("AXIS_ONLY_REJECTED", [("", PROSPECTIVE, AXIS_ONLY)])],
    "016": [("FRAME_CHANNELS_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1;")]),
            ("FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL", [("", PROSPECTIVE_CONDITIONAL, "Y2 (conditional)")]),
            ("SCALE_ONLY_REJECTED", [("", PROSPECTIVE,
                                      "Y3, the rejection of the scale-only alternative on both sets pooled")])],
    "017": [("HEAD_PATTERN_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1,")]),
            ("HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL", [("", PROSPECTIVE_CONDITIONAL, "Y2 (conditional)")]),
            ("FROZEN_PATTERN_NOT_REJECTED", [("", UNESTABLISHED, "Y3, on the cue-final pairs of both sets")])],
    "018": [("CHANNEL_D_CONCENTRATED_TOKENS", [("", PROSPECTIVE, "Y1, the concentration on unseen cues")]),
            ("CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL",
             [("", UNESTABLISHED, "Y2 (conditional): the no-harm guard failed in one new frame")]),
            ("SINGLE_NEURON_REJECTED", [("", PROSPECTIVE, "Y3, the rejection of the single-neuron account")]),
            ("PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES", [("", UNESTABLISHED, "Y4: the within-frame pattern term")])],
    "019": [("ROUTING_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1 to Y5")]),
            ("ROUTING_PREDICTED_FRAMES_CONDITIONAL",
             [("", PROSPECTIVE_CONDITIONAL, "with Y2 and the new-frame parts of Y3–Y5 conditional")]),
            ("OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT", [("", PROSPECTIVE, "Y1 to Y5")]),
            ("TEMPLATE_FAMILY_INSUFFICIENT", [("", PROSPECTIVE, "Y1 to Y5")]),
            ("MEMBERSHIP_PREDICTED", [("", PROSPECTIVE, "Y1 to Y5")])],
    "020": [("none assigned", [("", EXPLORATORY, "everything")]),
            ("the confirmatory test was never run", [("", UNESTABLISHED, "The confirmation was never run")])],
    "021": [("CONTRAST_PREDICTED_TOKENS", [("", PROSPECTIVE, "Y1 and Y3")]),
            ("CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL", [("", UNESTABLISHED, "Y2 (conditional)")]),
            ("NOUN_READOUT_FIXED", [("", PROSPECTIVE, "Y1 and Y3")])],
    "022": [("all eight PASS", [("Y1", PROSPECTIVE, "all eight conditions, each on its own"),
                                ("Y2", PROSPECTIVE_CONDITIONAL, "with Y2 conditional")])],
    "023": [("three PASS", [("Y1", PROSPECTIVE, "Y1/coordinated"),
                            ("Y2", PROSPECTIVE_CONDITIONAL, "Y2/cue_final and Y2/coordinated, conditional")]),
            ("Y1/cue_final ENVELOPE_ONLY_FAILURE", [("", UNESTABLISHED, "Y1/cue_final at its envelope")])],
    "024": [("NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS",
             [("", PROSPECTIVE, "the primary association")])],
    "025": [("CAUSAL_EFFECT_NOT_ESTABLISHED", [("", UNESTABLISHED, "CAUSAL_EFFECT_NOT_ESTABLISHED")])],
}
# Outcome words that are not labels, printed as plain text after the chips.
F7_PLAIN = {"022": ["no aggregate label"], "023": ["no aggregate label"]}


def f7() -> dict:
    path, rev = SOURCE_MAP
    header = table_row(path, 71, rev)
    assert header[:5] == ["exp", "subject", "kind of test, as recorded", "frozen outcome (verbatim)",
                          "status of the outcome"], header
    rows = []
    for n in list(range(74, 93)) + [94, 95]:
        cells = table_row(path, n, rev)
        experiment, subject, outcome, status = cells[0], cells[1], _plain(cells[3]), _plain(cells[4])
        chips = []
        for label, parts in F7_LABELS[experiment]:
            assert label in outcome or label in status, f"source map:{n} lacks {label!r}"
            for _, category, anchor in parts:
                status_of(experiment, category, anchor)
            chips.append((label, [(caption, category) for caption, category, _ in parts]))
        plain = F7_PLAIN.get(experiment, [])
        assert all(text in outcome for text in plain), f"source map:{n}"
        for token in re.findall(r"[A-Z][A-Z0-9_]{3,}", outcome):
            assert any(token in label for label, _ in chips), f"source map:{n}: {token} has no chip"
        rows.append({"experiment": experiment, "subject": subject, "chips": chips, "plain": plain,
                     "status_text": status, "source": f"{path}@{rev}:{n}"})
    assert len(rows) == 21
    # The footnote's qualifiers: the pooled statistics' new-frame share, and the head read-outs' measured input.
    expect(path, 61, 61, "the new-frame share of statistics pooled over both sets: 013–016 Y3, 018 Y3, and 019 Y3–Y5",
           rev=rev)
    expect(path, 62, 62, "Given the measured input.", "(009 Y1, 011 Y2)", rev=rev)
    return {"rows": rows}


# F8: the gate items of C002 and C003, in the claims' own words.
def _gates(path: str, rev: str, first: int, last: int) -> list[dict]:
    items, level = [], None
    for n in range(first, last + 1):
        text = line(path, n, rev)
        heading = re.match(r"### Level (\d) — (.+)$", text)
        if heading:
            level = (int(heading.group(1)), heading.group(2).strip())
            continue
        item = re.match(r"- \[( |x)\] (.*)$", text)
        if not item:
            continue
        checked, body = item.group(1) == "x", item.group(2)
        title = body.split(". ")[0].rstrip(".")
        if checked:
            state = "checked"
        else:
            words = re.search(r"(Not established|Not reached|Not claimed)", body)
            assert words, f"{path}:{n}"
            state = words.group(1)
            if title.startswith(state):
                title = level[1]
        items.append({"level": level, "title": title, "state": state, "source": f"{path}@{rev}:{n}"})
    return items


def f8() -> dict:
    c002 = _gates(*C002, 71, 121)
    c003 = _gates(*C003, 204, 303)
    assert len(c002) == 24 and len(c003) == 25, (len(c002), len(c003))
    return {"C002": c002, "C003": c003}


# F2: the prediction map. Each cell's numbers and words must stand in its cited lines, and its status in its
# experiment's status block.
@dataclass
class Cell:
    row: str
    column: str
    text: str
    status: str
    experiment: str
    first: int
    last: int
    numbers: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    state: str = ""


F2_COLUMNS = ["new cues, exposed frames", "new frames", "both sets pooled", "exposed or development data"]
F2_CELLS = [
    Cell("head read-out (009)", "new frames", "ρ 0.957, MAE 0.066\n138 pairs; given the measured input", PROSPECTIVE,
         "009", 62, 64, ["0.957", "0.066"], ["138 fresh (token, frame) pairs"],
         "Y1, the head's P1 read-out given its measured input"),
    Cell("encoding read (011)", "new frames", "ρ 0.848, MAE 0.077\n24 cues", PROSPECTIVE, "011", 75, 76,
         ["0.848", "0.077"], ["Over the 24 token means"], "Y1, the encoding read"),
    Cell("head read-out, replicated (011)", "new frames", "ρ 0.974, MAE 0.076\ngiven the measured input",
         PROSPECTIVE, "011", 87, 89, ["0.974", "0.076"], ["Over the 142 fresh (token, frame) pairs"],
         "Y2, P1 replicated given the measured head input"),
    Cell("layers 1–2, net (012)", "new frames", "ρ 0.864, MAE 0.061, R² 0.795\nexposed base states", PROSPECTIVE,
         "012", 81, 83, ["0.864", "0.061", "0.795"], [], "Y1 (0.864, 0.061, R² 0.795)"),
    Cell("encoding read + correction vs P1 (012)", "new frames", "ρ 0.922, MAE 0.062\nexposed base states",
         PROSPECTIVE, "012", 91, 93, ["0.922", "0.062"],
         ["encoding read plus modelled correction", "versus the measured P1 fraction"], "Y2 (0.922, 0.062)"),
    Cell("attention value paths (013)", "new cues, exposed frames", "ρ 0.877, R² 0.668", PROSPECTIVE, "013", 88, 90,
         ["0.877", "0.668"], [], "Y1;"),
    Cell("attention value paths (013)", "new frames", "ρ 0.919, R² 0.747", PROSPECTIVE_CONDITIONAL, "013", 100, 101,
         ["0.919", "0.747"], [], "Y2 (conditional)"),
    Cell("neuron 1987 (014)", "new cues, exposed frames", "ρ 0.950, R² 0.970", PROSPECTIVE, "014", 96, 97,
         ["0.950", "0.970"], [], "Y1;"),
    Cell("neuron 1987 (014)", "new frames", "ρ 0.910, R² 0.957", PROSPECTIVE_CONDITIONAL, "014", 101, 102,
         ["0.910", "0.957"], [], "Y2 (conditional)"),
    Cell("pattern change, token-local (015)", "new cues, exposed frames", "ρ 0.986, R² 0.895", PROSPECTIVE, "015",
         105, 106, ["0.986", "0.895"], [], "Y1;"),
    Cell("pattern change, token-local (015)", "new frames", "R² 0.490; layer-2 entry R² 0.423\nagainst 0.50",
         UNESTABLISHED, "015", 109, 111, ["0.490", "0.423"], ["**R² 0.490** (floor 0.50)", "**0.423** at layer 2"],
         "Y2 (conditional). The token-mean R² 0.490 and the layer-2 entry R² 0.423 missed 0.50"),
    Cell("pattern change, frame-conditioned (016)", "new cues, exposed frames", "ρ 0.998, R² 0.994", PROSPECTIVE,
         "016", 106, 107, ["0.998", "0.994"], [], "Y1;"),
    Cell("pattern change, frame-conditioned (016)", "new frames", "ρ 1.000, R² 0.996", PROSPECTIVE_CONDITIONAL, "016",
         111, 112, [], ["Spearman **1.000**", "**R² 0.996**"], "Y2 (conditional), with the per-frame guard"),
    Cell("the head's own pattern (017)", "new cues, exposed frames", "row R² 0.997\nΔT pairs R² 0.999", PROSPECTIVE,
         "017", 116, 120, [], ["Head-row entry R² **0.997**", "`ΔT` pairs R² **0.999**"], "Y1,"),
    Cell("the head's own pattern (017)", "new frames", "row R² 0.998\nΔT pairs R² 1.000", PROSPECTIVE_CONDITIONAL,
         "017", 121, 123, [], ["Row entry R² **0.998**", "`ΔT` pairs **1.000**"], "Y2 (conditional)"),
    Cell("the head's own pattern (017)", "both sets pooled", "frozen pattern: ΔT R² 0.958\non 1,248 cue-final pairs; "
         "not rejected", UNESTABLISHED, "017", 126, 128, ["0.958"], ["Over the 1248 scored cue-final pairs of both sets"],
         "Y3, on the cue-final pairs of both sets"),
    Cell("channel D subset (018)", "new cues, exposed frames", "κ 0.863 (c_L), 0.724 (Π)", PROSPECTIVE, "018", 102,
         104, ["0.863", "0.724"], [], "Y1, the concentration on unseen cues"),
    Cell("channel D subset (018)", "new frames", "no-harm guard failed\nin one frame", UNESTABLISHED, "018", 106, 111,
         [], ["quantifier-018-4"], "Y2 (conditional): the no-harm guard failed in one new frame"),
    Cell("channel D subset (018)", "both sets pooled", "single neuron: κ 0.219\nvs S₂₅₆'s 0.860; rejected",
         PROSPECTIVE, "018", 118, 121, ["0.219"], ["REJECTED", "against `S_256`'s 0.860"],
         "Y3, the rejection of the single-neuron account"),
    Cell("routing (019)", "new cues, exposed frames", "Δκ₆₄ 0.090\n64 of 90 frames; aggregate", PROSPECTIVE, "019",
         120, 123, ["0.090"], ["64 of 90"], "Y1 to Y5"),
    Cell("routing (019)", "new frames", "Δκ₆₄ 0.144\n14 of 18 frames; aggregate", PROSPECTIVE_CONDITIONAL, "019", 129,
         133, ["0.144"], ["14 of 18"], "with Y2 and the new-frame parts of Y3–Y5 conditional"),
    Cell("routing (019)", "both sets pooled", "template family insufficient:\nA 0.068 against the 0.05 margin",
         PROSPECTIVE, "019", 145, 146, ["0.068"], ["TEMPLATE_FAMILY_INSUFFICIENT", "`A = κ(E_64) − κ(T_64)`",
                                                    "pooled **0.068**", "(both ≥ 0.05)"], "Y1 to Y5"),
    Cell("axis-only alternative (014)", "both sets pooled", "R² −1.41; rejected", PROSPECTIVE, "014", 104, 105,
         ["-1.41"], [], AXIS_ONLY),
    Cell("axis-only alternative (015)", "both sets pooled", "R² −1.16; rejected", PROSPECTIVE, "015", 123, 124,
         ["-1.16"], [], AXIS_ONLY),
    Cell("scale-only alternative (016)", "both sets pooled", "layer-2 entry R² 0.672\nrejected", PROSPECTIVE, "016",
         118, 119, ["0.672"], [], "Y3, the rejection of the scale-only alternative on both sets pooled"),
    Cell("compact program (005 v1)", "exposed or development data",
         "quantifier template's mean under-\npredicted by 1.44 nats, against 1.0", EXPLORATORY, "005", 98, 101,
         ["1.44"], ["1.44 nats (quantifier)", "1.0-nat Tier A program floor"], "the v1 program-floor failure"),
    Cell("low-rank encoding (006)", "exposed or development data", "normalized RMSE 0.623\nagainst 0.50", EXPLORATORY,
         "006", 67, 72, ["0.623"], ["normalized RMSE 0.623 (> 0.50)"], "normalized RMSE 0.623 failed"),
    Cell("rank-1 program (007)", "new frames", "ρ 0.785 against 0.80\nnew cues, frames and nouns", UNESTABLISHED,
         "007", 87, 88, ["0.785"], ["Spearman 0.785 over the 24 fresh tokens (floor 0.80)"],
         "at label level: PROGRAM_NOT_SUPPORTED"),
    Cell("rank-1 transport rule (009)", "new frames", "ρ 0.765 against 0.80", UNESTABLISHED, "009", 69, 70,
         ["0.765"], ["Spearman 0.765 (floor 0.80)"], "Y2, the transport rule from E"),
]
# Words the cells and the note rest on, outside the cells' own lines.
F2_EXTRA_CHECKS = [
    cite("docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md", 235, 236,
         "the 24 fresh tokens in the 6 fresh frames × 20 fresh nouns"),
    cite(SOURCE_MAP[0], 62, 64, "The head read-outs (009 Y1, 011 Y2)", rev=SOURCE_MAP[1]),
    cite(readme("012"), 8, 9, "the base state taken from exposed reference prompts"),
    cite(readme("019"), 164, 165, "the result is an aggregate routing result"),
    cite(readme("019"), 145, 146, "on the new frames **0.080**"),
    cite(SOURCE_MAP[0], 61, 61, "the new-frame share of statistics pooled over both sets", rev=SOURCE_MAP[1]),
    cite(SOURCE_MAP[0], 65, 67, "Label-level versus criterion-level", rev=SOURCE_MAP[1]),
    cite("research/synthesis/synthesis-005-025.md", 272, 272, "009, 011 and 012 tested fresh cues in new frames only",
         rev="3d87d3c"),
    cite(readme("007"), 59, 60, "E-patch interventions on the six fresh frames (Y families)"),
    cite(readme("006"), 71, 71, "no lock may be written"),
    cite(readme("019"), 146, 147, "The frame-specific evaluation exceeds the template ranking by the margin on both sets"),
]


def f2() -> dict:
    cells = [cell for cell in F2_CELLS if cell.experiment]
    for cell in cells:
        text = _space(" ".join(lines(readme(cell.experiment))[cell.first - 1:cell.last]).replace("−", "-"))
        for value in cell.numbers:
            assert value in text, f"{cell.experiment}:{cell.first}–{cell.last} lacks {value!r}"
        expect(readme(cell.experiment), cell.first, cell.last, *cell.words)
        status_of(cell.experiment, cell.status, cell.state)
    for _, path, first, last, anchors, rev in F2_EXTRA_CHECKS:
        expect(path, first, last, *anchors, rev=rev)
    status_of("006", UNESTABLISHED, "every confirmatory family. No lock could be written, so none was tested")
    rows = []
    for cell in cells:
        if cell.row not in rows:
            rows.append(cell.row)
    return {"rows": rows, "columns": F2_COLUMNS, "cells": cells}


# F1: the chain, in two lanes, with the chips of the audit's F1 table. Each chip holds badges; each badge carries one
# recorded status, checked against its experiment's status block in the source map (and, where it names a frozen
# label, against the overview's outcome cell), or against cited lines that state the status in words.
SYNTHESIS = ("research/synthesis/synthesis-005-025.md", "3d87d3c")
F1_STAGES = [("the encoding", "E(w), the layer-0 MLP"), ("block 0", "embedding, attention"), ("layers 1–2", ""),
             ("Δx3", "entering layer 3"), ("transport", "L03.H04"), ("the readout", "blocks 3–5"),
             ("Δc", "the contrast change")]
STATUS_WORDS = {PROSPECTIVE: ["prospective"], PROSPECTIVE_CONDITIONAL: ["conditional"], LOCKED: ["locked"],
                EXPLORATORY: ["exploratory"], DESCRIPTIVE: ["descriptive", "Descriptive"],
                UNESTABLISHED: ["unestablished", "Not established", "never run", "not established"]}


def state(experiment: str, anchor: str, label: str | None = None) -> tuple:
    """The badge's status is listed in the experiment's status block with `anchor`; `label` is in its outcome."""
    return ("status", experiment, anchor, label)


def badge(caption: str, status: str, *sources: tuple) -> dict:
    return {"caption": caption, "status": status, "sources": sources}


Y1_022, Y2_022 = "all eight conditions, each on its own", "with Y2 conditional"
Y2_023 = "Y2/cue_final and Y2/coordinated, conditional"
F1_CHIPS = [
    # (lane, stage, text, badges[, sources for the chip's text]); lane "link" is the E-patch lane's link from the
    # encoding to transport.
    ("E-patch", "the encoding", "010: an opposing off-axis part for this, a, another and every",
     [badge("", EXPLORATORY, state("010", "the encoding-borne opposition", "MIXED"),
            cite(readme("010"), 85, 88, "The opposition is already inside the encoding", "24 of 24 frames"))]),
    ("E-patch", "the encoding", "compact encoding programs",
     [badge("005", EXPLORATORY, state("005", "the v1 program-floor failure", "NO_COMPACT_MECHANISM"),
            cite(readme("005"), 98, 101, "NO_COMPACT_MECHANISM")),
      badge("006", EXPLORATORY, state("006", "the pre-lock quality gate", "QUALITY_GATE_FAILED"),
            cite(readme("006"), 67, 72, "QUALITY_GATE_FAILED")),
      badge("007", UNESTABLISHED, state("007", "at label level: PROGRAM_NOT_SUPPORTED", "PROGRAM_NOT_SUPPORTED"),
            cite(readme("007"), 87, 94, "PROGRAM_NOT_SUPPORTED"))]),
    ("link", "the encoding", "011: the encoding read predicts transport",
     [badge("", PROSPECTIVE, state("011", "Y1, the encoding read", "ENCODING_READ_PREDICTS_TRANSPORT"),
            cite(readme("011"), 75, 76, "zero-parameter encoding read: PASS"),
            cite(SYNTHESIS[0], 258, 258, "encoding → transport", "[prospective]", rev=SYNTHESIS[1]))]),
    ("E-patch", "layers 1–2", "012: the net layer-1–2 correction, and the composite against the P1 read-out, from "
     "exposed base states",
     [badge("", PROSPECTIVE, state("012", "Y1 (0.864, 0.061, R² 0.795)", "LAYER_CORRECTION_TOKEN_LOCAL_MLP"),
            state("012", "Y2 (0.922, 0.062)", "COMPOSITE_PREDICTS_P1"))],
     [cite(readme("012"), 8, 9, "the base state taken from exposed reference prompts"),
      cite(readme("012"), 91, 92, "the composite account reaches the head's linear read-out: PASS")]),
    ("E-patch", "layers 1–2", "013: the heads' value paths",
     [badge("Y1", PROSPECTIVE, state("013", "Y1;", "RESIDUAL_PREDICTED_TOKENS")),
      badge("Y2", PROSPECTIVE_CONDITIONAL, state("013", "Y2 (conditional)", "RESIDUAL_PREDICTED_FRAMES_CONDITIONAL"))]),
    ("E-patch", "layers 1–2", "014: neuron 1987",
     [badge("Y1", PROSPECTIVE, state("014", "Y1;", "NEURON_FEATURE_PREDICTED_TOKENS")),
      badge("Y2", PROSPECTIVE_CONDITIONAL, state("014", "Y2 (conditional)",
                                                 "NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL")),
      badge("rival", PROSPECTIVE, state("014", AXIS_ONLY, "AXIS_ONLY_REJECTED"))]),
    ("E-patch", "layers 1–2", "015: the heads' pattern change",
     [badge("Y1", PROSPECTIVE, state("015", "Y1;", "PATTERN_CHANGE_PREDICTED_TOKENS")),
      badge("Y2", UNESTABLISHED, state("015", "Y2 (conditional)", "PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL")),
      badge("rival", PROSPECTIVE, state("015", AXIS_ONLY, "AXIS_ONLY_REJECTED"))]),
    ("E-patch", "layers 1–2", "016: the pattern change, frame channels",
     [badge("Y1", PROSPECTIVE, state("016", "Y1;", "FRAME_CHANNELS_PREDICTED_TOKENS")),
      badge("Y2", PROSPECTIVE_CONDITIONAL, state("016", "Y2 (conditional)",
                                                 "FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL")),
      badge("rival", PROSPECTIVE, state("016", "Y3, the rejection of the scale-only alternative on both sets pooled",
                                        "SCALE_ONLY_REJECTED"))]),
    ("E-patch", "layers 1–2", "018: channel D, a neuron subset",
     [badge("Y1", PROSPECTIVE, state("018", "Y1, the concentration on unseen cues", "CHANNEL_D_CONCENTRATED_TOKENS")),
      badge("Y2", UNESTABLISHED, state("018", "Y2 (conditional): the no-harm guard failed in one new frame",
                                       "CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL")),
      badge("rival", PROSPECTIVE, state("018", "Y3, the rejection of the single-neuron account",
                                        "SINGLE_NEURON_REJECTED"))]),
    ("E-patch", "layers 1–2", "019: channel D's routing",
     [badge("Y1", PROSPECTIVE, state("019", "Y1 to Y5", "ROUTING_PREDICTED_TOKENS")),
      badge("Y2", PROSPECTIVE_CONDITIONAL, state("019", "with Y2 and the new-frame parts of Y3–Y5 conditional",
                                                 "ROUTING_PREDICTED_FRAMES_CONDITIONAL")),
      badge("rival", PROSPECTIVE, state("019", "Y1 to Y5", "TEMPLATE_FAMILY_INSUFFICIENT"),
            cite(readme("019"), 145, 145, "the template-family account: `TEMPLATE_FAMILY_INSUFFICIENT`"))]),
    ("E-patch", "transport", "P1 read-out, given the measured input",
     [badge("009", PROSPECTIVE, state("009", "Y1, the head's P1 read-out given its measured input",
                                      "HEAD_MECHANISM_CONFIRMED_P1"),
            cite(readme("009"), 62, 64, "fixed-normalization linear OV", "**0.957**")),
      badge("011", PROSPECTIVE, state("011", "Y2, P1 replicated given the measured head input", "HEAD_P1_REPLICATED"),
            cite(readme("011"), 87, 89, "head read-out replication: PASS"))]),
    ("E-patch", "transport", "017: the head's row and ΔT",
     [badge("Y1", PROSPECTIVE, state("017", "Y1,", "HEAD_PATTERN_PREDICTED_TOKENS")),
      badge("Y2", PROSPECTIVE_CONDITIONAL, state("017", "Y2 (conditional)",
                                                 "HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL"))]),
    ("E-patch", "transport", "the pattern term's necessity",
     [badge("017", UNESTABLISHED, state("017", "Y3, on the cue-final pairs of both sets", "FROZEN_PATTERN_NOT_REJECTED"),
            cite(readme("017"), 126, 131, "FROZEN_PATTERN_NOT_REJECTED")),
      badge("018", UNESTABLISHED, state("018", "Y4: the within-frame pattern term",
                                        "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES"),
            cite(readme("018"), 122, 126, "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES"))]),
    ("E-patch", "transport", "009: the rank-1 weight-only rule",
     [badge("", UNESTABLISHED, state("009", "Y2, the transport rule from E", "TRANSPORT_RULE_FAILED"),
            cite(readme("009"), 69, 70, "weight-only transport rule: FAIL", "the rank-1 rule reached Spearman 0.765"))]),
    ("E-patch", "Δc", "009: the contrast rule, never locked",
     [badge("", UNESTABLISHED, state("009", "Y3, never locked", "NOT_LOCKED"),
            cite(readme("009"), 77, 77, "contrast rule: `NOT_LOCKED`"))]),
    ("real swap", "block 0", "022: the error localized primarily to block-0 attention",
     [badge("Y1 C1", PROSPECTIVE, state("022", Y1_022),
            cite(readme("022"), 187, 187, "Y1/C1", "block-0 attention dominates the cue-final gap", "**PASS**")),
      badge("Y1 C4", PROSPECTIVE, state("022", Y1_022),
            cite(readme("022"), 190, 190, "Y1/C4", "block-0 cue→target attention contributes positively", "**PASS**")),
      badge("Y2 C1", PROSPECTIVE_CONDITIONAL, state("022", Y2_022),
            cite(readme("022"), 191, 191, "Y2/C1", "block-0 attention dominates the cue-final gap", "**PASS**")),
      badge("Y2 C4", PROSPECTIVE_CONDITIONAL, state("022", Y2_022),
            cite(readme("022"), 194, 194, "Y2/C4", "block-0 cue→target attention contributes positively", "**PASS**"))],
     [cite(readme("022"), 243, 245, "is localized primarily to block-0 attention")]),
    ("real swap", "block 0", "023: block-0 completion; no aggregate label",
     [badge("Y1 c-f", UNESTABLISHED, state("023", "Y1/cue_final at its envelope"),
            cite(readme("023"), 142, 142, "Y1/cue_final", "ENVELOPE_ONLY_FAILURE")),
      badge("Y1 co", PROSPECTIVE, state("023", "Y1/coordinated"),
            cite(readme("023"), 143, 143, "Y1/coordinated", "**PASS**")),
      badge("Y2 c-f", PROSPECTIVE_CONDITIONAL, state("023", Y2_023),
            cite(readme("023"), 144, 144, "Y2/cue_final", "**PASS**")),
      badge("Y2 co", PROSPECTIVE_CONDITIONAL, state("023", Y2_023),
            cite(readme("023"), 145, 145, "Y2/coordinated", "**PASS**"))],
     [cite(readme("023"), 149, 150, "There is no aggregate PASS/FAIL criterion")]),
    ("real swap", "block 0", "023: cheaper block-0 rules recover far less in cue-final frames",
     [badge("", DESCRIPTIVE, state("023", "the cheaper block-0 rules"),
            cite(readme("023"), 178, 181, "Descriptive only, no outcome force",
                 "Cheaper block-0 rules recover far less in cue-final frames"))]),
    ("real swap", "layers 1–2", "022: the layer-1–2 reductions, at most 0.007 of the gap",
     [badge("Y1 C2", PROSPECTIVE, state("022", Y1_022), cite(readme("022"), 188, 188, "Y1/C2", "−0.0000118", "**PASS**")),
      badge("Y1 C3", PROSPECTIVE, state("022", Y1_022), cite(readme("022"), 189, 189, "Y1/C3", "0.0059509", "**PASS**")),
      badge("Y2 C2", PROSPECTIVE_CONDITIONAL, state("022", Y2_022),
            cite(readme("022"), 192, 192, "Y2/C2", "0.0038511", "**PASS**")),
      badge("Y2 C3", PROSPECTIVE_CONDITIONAL, state("022", Y2_022),
            cite(readme("022"), 193, 193, "Y2/C3", "0.0069294", "**PASS**"))],
     [cite(C003[0], 237, 238, "with the layer-1–2 reductions at most 0.007 of the gap", rev=C003[1]),
      cite(readme("022"), 12, 12, "0–0.7 % of either gap")]),
    ("real swap", "Δx3", "022: the Δx3 reconstruction",
     [badge("", DESCRIPTIVE, cite(readme("022"), 225, 229, "`Δx3` reconstruction", "nearly exact"),
            cite(C003[0], 396, 397, "of `Δx3` is descriptive (022:225–229; 023:130)", rev=C003[1]))]),
    ("real swap", "the readout", "020: its ceiling, 0.976, on exposed data",
     [badge("", EXPLORATORY, state("020", "the ceiling 0.9760"), cite(readme("020"), 132, 132, "**0.9760**"))]),
    ("real swap", "the readout", "020: confirmation never run",
     [badge("", UNESTABLISHED, state("020", "The confirmation was never run"),
            cite(readme("020"), 8, 11, "confirmation intentionally not run"))]),
    ("real swap", "the readout", "021, 023: fresh ceilings and R²_C",
     [badge("", DESCRIPTIVE, state("021", "the ceiling (0.9683 and 0.9652)"),
            cite(readme("023"), 140, 145, "| R²_C |", "0.9593", "0.9579", "0.9613", "0.9531"),
            cite(C003[0], 70, 71, "descriptive on fresh data", rev=C003[1]))]),
    ("real swap", "the readout", "024: nounness predicts its error; not causal",
     [badge("", PROSPECTIVE, state("024", "the primary association",
                                   "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"),
            cite(readme("024"), 373, 373, "predictive, not causal"))]),
    ("real swap", "Δc", "021: the contrast change on 020's wiring",
     [badge("Y1", PROSPECTIVE, state("021", "Y1 and Y3", "CONTRAST_PREDICTED_TOKENS")),
      badge("Y2", UNESTABLISHED, state("021", "Y2 (conditional)", "CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL")),
      badge("Y3", PROSPECTIVE, state("021", "Y1 and Y3", "NOUN_READOUT_FIXED"))]),
    ("causal", "", "C003: necessity, sufficiency and control gates not established",
     [badge("", UNESTABLISHED, cite(C003[0], 257, 267, "Not established", rev=C003[1]),
            cite(C003[0], 273, 274, "No necessity, sufficiency or control gate is checked.", rev=C003[1]))]),
    ("causal", "", "025: CAUSAL_EFFECT_NOT_ESTABLISHED",
     [badge("", UNESTABLISHED, state("025", "CAUSAL_EFFECT_NOT_ESTABLISHED", "CAUSAL_EFFECT_NOT_ESTABLISHED"))]),
]
# Notes the drawing must carry (the audit's F1 section): the encoding runs in parallel with block-0 attention, and end
# to end the head is computed inside block 3, as part of the readout.
F1_NOTES = [cite(readme("012"), 8, 10, "the block-0 MLP is parallel to block-0 attention"),
            cite(C003[0], 62, 63, "the head is computed exactly as part of block 3, in the readout", rev=C003[1])]


def f1() -> dict:
    stages = [name for name, _ in F1_STAGES]
    chips = []
    for lane, stage, text, badges, *chip_sources in F1_CHIPS:
        assert lane in ("E-patch", "link", "real swap", "causal") and (stage in stages or lane == "causal")
        for _, path, first, last, anchors, rev in (chip_sources[0] if chip_sources else []):
            expect(path, first, last, *anchors, rev=rev)
        for b in badges:
            stated, by_block = [], False
            for source in b["sources"]:
                if source[0] == "status":
                    _, experiment, anchor, label = source
                    status_of(experiment, b["status"], anchor)
                    assert label is None or label in overview(experiment)[0], (experiment, label)
                    by_block = True
                else:
                    _, path, first, last, anchors, rev = source
                    expect(path, first, last, *anchors, rev=rev)
                    stated.extend(anchors)
            words = STATUS_WORDS[b["status"]]
            assert by_block or any(word in anchor for anchor in stated for word in words), (text, b["caption"])
        chips.append({"lane": lane, "stage": stage, "text": text,
                      "badges": [(b["caption"], b["status"]) for b in badges]})
    for _, path, first, last, anchors, rev in F1_NOTES:
        expect(path, first, last, *anchors, rev=rev)
    return {"stages": F1_STAGES, "chips": chips}


def everything() -> dict:
    return {"F1": f1(), "F2": f2(), "F3": {"A1": f3_a1(), "A2": f3_a2(), "B": f3_b()}, "F4": f4(), "F5": f5(),
            "F6": f6(), "F7": f7(), "F8": f8()}


if __name__ == "__main__":
    data = everything()
    counts = {
        "F1 chips": len(data["F1"]["chips"]), "F2 cells": len(data["F2"]["cells"]),
        "F3 A1 points": len(data["F3"]["A1"]["points"]), "F3 A2 points": len(data["F3"]["A2"]["points"]),
        "F3 B points": len(data["F3"]["B"]["points"]), "F4 tokens": len(data["F4"]["rows"]),
        "F5 rungs": len(data["F5"]["y1"]), "F6 conditions": len(data["F6"]["conditions"]),
        "F7 rows": len(data["F7"]["rows"]), "F8 items": len(data["F8"]["C002"]) + len(data["F8"]["C003"]),
    }
    print("every source checked:", ", ".join(f"{key} {value}" for key, value in counts.items()))
