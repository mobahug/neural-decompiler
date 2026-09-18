"""Experiment 005: prospective mechanism study of count-cued noun number selection.

The scientific inputs are the frozen screening manifest (``regular-plural``
cases only) and one tokenizer-only extension set. Every regular-plural case in
a split is one of twelve prompts read out through a noun pair's unembedding
rows, so this module works at the prompt level: one forward pass per prompt
scores every noun of a split. Every constant below is copied from the approved
design (revision 4) and must not be changed without a design amendment.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest, Split, load_manifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 4).

CANDIDATE_ID = "regular-plural"
MANIFEST_RELATIVE_PATH = "screening/behavior-candidates/manifest-v1.json"
EXTENSION_RELATIVE_PATH = "experiments/005-regular-plural-mechanism/extension-v1.json"
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260918
EXTENSION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
CUE_WORD_COUNT = 12
NEW_FRAMES_PER_TEMPLATE = 2
TEMPLATE_ORDER = ("cardinal", "quantifier", "coordinated-adjective")
CUE_FINAL_TEMPLATES = ("cardinal", "quantifier")
COORDINATED_TEMPLATE = "coordinated-adjective"
RULE_CLASSES = ("simple-suffix", "sibilant-es", "consonant-y")

# New frames are fixed literals; ``{cue}`` is the cue slot. Order matters.
EXTENSION_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The basket carries {cue}"),
    ("cardinal", "The museum owns {cue}"),
    ("quantifier", "The menu offers {cue}"),
    ("quantifier", "The report cites {cue}"),
    ("coordinated-adjective", "Ana and Luis painted {cue} tiny"),
    ("coordinated-adjective", "Kai and Sara carried {cue} heavy"),
)

# Ordered candidate cue words; the first twelve tokenizer-eligible are frozen.
EXTENSION_CUE_WORDS: tuple[str, ...] = (
    "a", "the", "three", "four", "five", "ten", "many", "few", "some", "all",
    "both", "every", "any", "no", "another", "single", "multiple", "numerous",
    "twelve", "hundred",
)

PHASES = ("discover", "calibrate", "revise", "lock", "confirm", "report")
PHASE_STATUSES = ("not_started", "running", "complete", "invalidated")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class PhaseError(RuntimeError):
    """A phase was requested out of order, twice, or with mismatched provenance."""


class IncidentError(RuntimeError):
    """A validity check failed; the experiment stops as a software incident."""


# ---------------------------------------------------------------------------
# Small helpers shared with the screen's conventions.


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="seconds")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    unknown, missing = set(value) - expected, expected - set(value)
    if unknown or missing:
        raise ValueError(f"{path} has unknown or missing fields: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _ids(value: Any, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a nonempty list of token IDs")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in value):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return tuple(value)


# ---------------------------------------------------------------------------
# Prompt-level data model derived from the frozen manifest.


@dataclass(frozen=True)
class Frame:
    """A sentence frame with one cue slot; the noun is only the scored next token."""

    template_id: str
    frame_id: str
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    cue_ids: Mapping[str, int]  # {"sg": id, "pl": id}
    text_template: str
    origin: str = "manifest"  # or "extension"

    def __post_init__(self) -> None:
        if self.template_id not in TEMPLATE_ORDER:
            raise ValueError(f"unknown template {self.template_id}")
        if set(self.cue_ids) != {"sg", "pl"}:
            raise ValueError("cue_ids must map sg and pl")
        if self.origin not in {"manifest", "extension"}:
            raise ValueError("origin must be manifest or extension")
        object.__setattr__(self, "prefix_ids", tuple(self.prefix_ids))
        object.__setattr__(self, "suffix_ids", tuple(self.suffix_ids))
        object.__setattr__(self, "cue_ids", dict(self.cue_ids))
        if self.template_id == COORDINATED_TEMPLATE and len(self.suffix_ids) != 1:
            raise ValueError("coordinated frames carry exactly one adjective token after the cue")
        if self.template_id in CUE_FINAL_TEMPLATES and self.suffix_ids:
            raise ValueError("cue-final frames have no tokens after the cue")

    @property
    def p_c(self) -> int:
        return len(self.prefix_ids)

    @property
    def p_t(self) -> int:
        return len(self.prefix_ids) + 1 + len(self.suffix_ids) - 1

    def prompt_ids(self, cue_token_id: int) -> tuple[int, ...]:
        return self.prefix_ids + (int(cue_token_id),) + self.suffix_ids

    def to_dict(self) -> dict[str, Any]:
        return {"template_id": self.template_id, "frame_id": self.frame_id, "prefix_ids": list(self.prefix_ids),
                "suffix_ids": list(self.suffix_ids), "cue_ids": dict(self.cue_ids), "text_template": self.text_template,
                "origin": self.origin, "p_c": self.p_c, "p_t": self.p_t}


@dataclass(frozen=True)
class Noun:
    lexical_key: str
    split: Split
    rule_class: str
    sg_ids: tuple[int, ...]
    pl_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.rule_class not in RULE_CLASSES:
            raise ValueError(f"unknown rule class {self.rule_class}")
        object.__setattr__(self, "sg_ids", tuple(self.sg_ids))
        object.__setattr__(self, "pl_ids", tuple(self.pl_ids))
        if len(self.sg_ids) != len(self.pl_ids) or not self.sg_ids:
            raise ValueError("noun forms must have equal, positive token counts")

    @property
    def single_token(self) -> bool:
        return len(self.sg_ids) == 1

    @property
    def key(self) -> str:
        return f"{self.split.value}:{self.lexical_key}"

    def to_dict(self) -> dict[str, Any]:
        return {"lexical_key": self.lexical_key, "split": self.split.value, "rule_class": self.rule_class,
                "sg_ids": list(self.sg_ids), "pl_ids": list(self.pl_ids), "single_token": self.single_token}


@dataclass(frozen=True)
class Prompt:
    """One concrete token sequence: a frame with one cue token in its slot."""

    frame: Frame
    cue_token_id: int
    cue_label: str  # "sg", "pl", or the extension word

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.frame.prompt_ids(self.cue_token_id)

    @property
    def key(self) -> str:
        return f"{self.frame.frame_id}|{self.cue_label}|{self.cue_token_id}"

    @property
    def p_c(self) -> int:
        return self.frame.p_c

    @property
    def p_t(self) -> int:
        return self.frame.p_t


def regular_plural_cases(manifest: ScreeningManifest, split: Split | None = None):
    return tuple(case for case in manifest.cases if case.candidate_id == CANDIDATE_ID and (split is None or case.split is split))


def derive_frames(manifest: ScreeningManifest, tokenizer_strings: Mapping[str, str] | None = None) -> tuple[Frame, ...]:
    """Recover the twelve-prompt structure: six frames, each with a singular and a plural cue."""
    seen: dict[tuple[str, tuple[int, ...], tuple[int, ...]], Frame] = {}
    counters: dict[str, int] = {}
    for case in regular_plural_cases(manifest):
        a_ids, b_ids = case.x_a.prompt_token_ids, case.x_b.prompt_token_ids
        if len(a_ids) != len(b_ids):
            raise ValueError(f"{case.case_id}: matched prompts differ in length")
        differing = [index for index, (left, right) in enumerate(zip(a_ids, b_ids)) if left != right]
        if len(differing) != 1:
            raise ValueError(f"{case.case_id}: matched prompts must differ at exactly the cue position")
        p_c = differing[0]
        key = (case.template_id, a_ids, b_ids)
        if key in seen:
            continue
        counters[case.template_id] = counters.get(case.template_id, 0) + 1
        prefix_text = case.x_a.prompt_text
        frame = Frame(
            template_id=case.template_id,
            frame_id=f"{case.template_id}-{counters[case.template_id]}",
            prefix_ids=a_ids[:p_c],
            suffix_ids=a_ids[p_c + 1:],
            cue_ids={"sg": a_ids[p_c], "pl": b_ids[p_c]},
            text_template=_text_template(case.x_a.prompt_text, case.x_b.prompt_text),
        )
        seen[key] = frame
    frames = tuple(seen.values())
    if len(frames) != 6 or any(sum(frame.template_id == template for frame in frames) != 2 for template in TEMPLATE_ORDER):
        raise ValueError("manifest must yield exactly two frames per template")
    for frame in frames:
        expected = (3, 3) if frame.template_id in CUE_FINAL_TEMPLATES else (5, 6)
        if (frame.p_c, frame.p_t) != expected:
            raise ValueError(f"{frame.frame_id}: positions {(frame.p_c, frame.p_t)} differ from the frozen {expected}")
    return tuple(sorted(frames, key=lambda frame: (TEMPLATE_ORDER.index(frame.template_id), frame.frame_id)))


def _text_template(text_sg: str, text_pl: str) -> str:
    words_sg, words_pl = text_sg.split(" "), text_pl.split(" ")
    if len(words_sg) != len(words_pl):
        raise ValueError("matched prompt texts differ in word count")
    differing = [index for index, (left, right) in enumerate(zip(words_sg, words_pl)) if left != right]
    if len(differing) != 1:
        raise ValueError("matched prompt texts must differ in exactly one word")
    words_sg[differing[0]] = "{cue}"
    return " ".join(words_sg)


def nouns_for(manifest: ScreeningManifest, split: Split) -> tuple[Noun, ...]:
    nouns: dict[str, Noun] = {}
    for case in regular_plural_cases(manifest, split):
        noun = Noun(case.lexical_key, split, case.rule_class, case.x_a.a_token_ids, case.x_a.b_token_ids)
        previous = nouns.setdefault(case.lexical_key, noun)
        if previous != noun:
            raise ValueError(f"{case.lexical_key}: inconsistent noun forms across cases")
    result = tuple(nouns.values())
    if len(result) != 20:
        raise ValueError(f"{split.value}: expected 20 nouns, found {len(result)}")
    return result


def manifest_prompts(frames: Sequence[Frame]) -> tuple[Prompt, ...]:
    prompts = []
    for frame in frames:
        for label in ("sg", "pl"):
            prompts.append(Prompt(frame, frame.cue_ids[label], label))
    return tuple(prompts)


# ---------------------------------------------------------------------------
# Extension set: tokenizer-only construction, digest, loading, validation.


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(ids)))


def eligible_cue_words(tokenizer: Any) -> tuple[tuple[str, int], ...]:
    """The first twelve space-prefixed single-token words of the frozen ordered list."""
    chosen: list[tuple[str, int]] = []
    for word in EXTENSION_CUE_WORDS:
        ids = _encode(tokenizer, " " + word)
        if len(ids) == 1:
            chosen.append((word, ids[0]))
        if len(chosen) == CUE_WORD_COUNT:
            break
    if len(chosen) != CUE_WORD_COUNT:
        raise ValueError("fewer than twelve tokenizer-eligible cue words")
    return tuple(chosen)


def _build_new_frame(tokenizer: Any, template_id: str, text_template: str, cue_ids: Mapping[str, int], frame_id: str) -> Frame:
    prompts = {}
    for label in ("sg", "pl"):
        cue_text = _decode(tokenizer, [cue_ids[label]])
        text = text_template.replace("{cue}", cue_text.strip())
        ids = _encode(tokenizer, text)
        occurrences = [index for index, token in enumerate(ids) if token == cue_ids[label]]
        if len(occurrences) != 1:
            raise ValueError(f"{frame_id}: cue token must occur exactly once in {text!r}")
        prompts[label] = (ids, occurrences[0])
    (sg_ids, sg_pos), (pl_ids, pl_pos) = prompts["sg"], prompts["pl"]
    if sg_pos != pl_pos or sg_ids[:sg_pos] != pl_ids[:pl_pos] or sg_ids[sg_pos + 1:] != pl_ids[pl_pos + 1:]:
        raise ValueError(f"{frame_id}: the two cue prompts must share prefix and suffix tokens")
    return Frame(template_id, frame_id, sg_ids[:sg_pos], sg_ids[sg_pos + 1:], dict(cue_ids), text_template, origin="extension")


def build_extension_payload(tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str) -> dict[str, Any]:
    """Construct the extension set from tokenizer rules only; no model output is involved."""
    frames = derive_frames(manifest)
    by_template = {template: [frame for frame in frames if frame.template_id == template] for template in TEMPLATE_ORDER}
    cue_ids_by_template = {template: dict(by_template[template][0].cue_ids) for template in TEMPLATE_ORDER}
    for template in TEMPLATE_ORDER:
        if any(dict(frame.cue_ids) != cue_ids_by_template[template] for frame in by_template[template]):
            raise ValueError(f"{template}: manifest frames disagree on cue tokens")
    new_frames = []
    counters: dict[str, int] = {}
    for template_id, text_template in EXTENSION_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        frame_id = f"{template_id}-new-{counters[template_id]}"
        frame = _build_new_frame(tokenizer, template_id, text_template, cue_ids_by_template[template_id], frame_id)
        new_frames.append({
            "frame_id": frame.frame_id, "template_id": template_id, "text_template": text_template,
            "prefix_ids": list(frame.prefix_ids), "suffix_ids": list(frame.suffix_ids), "cue_ids": dict(frame.cue_ids),
            "p_c": frame.p_c, "p_t": frame.p_t,
            "prompts": {label: {"text": _decode(tokenizer, frame.prompt_ids(frame.cue_ids[label])),
                                "token_ids": list(frame.prompt_ids(frame.cue_ids[label]))} for label in ("sg", "pl")},
        })
    cue_words = [{"word": word, "token_id": token_id} for word, token_id in eligible_cue_words(tokenizer)]
    cue_word_prompts = []
    for frame in frames:
        for entry in cue_words:
            ids = frame.prompt_ids(entry["token_id"])
            text = _decode(tokenizer, ids)
            if _encode(tokenizer, text) != ids:
                raise ValueError(f"{frame.frame_id}/{entry['word']}: text does not round-trip to the constructed token IDs")
            cue_word_prompts.append({"frame_id": frame.frame_id, "word": entry["word"], "token_id": entry["token_id"],
                                     "text": text, "token_ids": list(ids), "p_c": frame.p_c, "p_t": frame.p_t})
    payload = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
        "original_frames": [frame.to_dict() for frame in frames],
        "reference_cue_ids": {template: cue_ids_by_template[template]["sg"] for template in TEMPLATE_ORDER},
        "new_frames": new_frames,
        "cue_words": cue_words,
        "cue_word_prompts": cue_word_prompts,
        "construction": "tokenizer-only; no model output",
    }
    payload["content_sha256"] = extension_content_digest(payload)
    return payload


def extension_content_digest(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
    return sha256_text(canonical_json(unsigned))


@dataclass(frozen=True)
class Extension:
    manifest_sha256: str
    original_frames: tuple[Frame, ...]
    reference_cue_ids: Mapping[str, int]
    new_frames: tuple[Frame, ...]
    cue_words: tuple[tuple[str, int], ...]
    cue_word_prompts: tuple[Prompt, ...]
    content_sha256: str

    def new_frame_prompts(self) -> tuple[Prompt, ...]:
        return manifest_prompts(self.new_frames)


def freeze_extension(path: Path, tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the extension set is frozen and cannot be rebuilt")
    payload = build_extension_payload(tokenizer, manifest, manifest_sha256)
    validate_json_safe(payload, path="extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload["content_sha256"]


def _parse_frame(value: Mapping[str, Any], origin: str) -> Frame:
    _require_exact_keys(value, {"frame_id", "template_id", "text_template", "prefix_ids", "suffix_ids", "cue_ids", "p_c", "p_t"} | ({"prompts"} if origin == "extension" else {"origin"}), "frame")
    cue_ids = value["cue_ids"]
    if not isinstance(cue_ids, Mapping) or set(cue_ids) != {"sg", "pl"}:
        raise ValueError("frame cue_ids must map sg and pl")
    frame = Frame(value["template_id"], value["frame_id"], _ids(value["prefix_ids"], "prefix_ids"),
                  _ids(value["suffix_ids"], "suffix_ids", allow_empty=True), {k: int(v) for k, v in cue_ids.items()},
                  value["text_template"], origin=origin)
    if (frame.p_c, frame.p_t) != (value["p_c"], value["p_t"]):
        raise ValueError(f"{frame.frame_id}: stored positions disagree with token structure")
    if origin == "extension":
        for label in ("sg", "pl"):
            stored = value["prompts"][label]
            if tuple(stored["token_ids"]) != frame.prompt_ids(frame.cue_ids[label]):
                raise ValueError(f"{frame.frame_id}: stored prompt token IDs disagree with the frame")
    return frame


def validate_extension(payload: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str) -> Extension:
    _require_exact_keys(payload, {"schema_version", "manifest_sha256", "model", "original_frames", "reference_cue_ids", "new_frames",
                                  "cue_words", "cue_word_prompts", "construction", "content_sha256"}, "extension")
    if payload["schema_version"] != EXTENSION_SCHEMA_VERSION:
        raise ValueError("extension schema version is not frozen")
    if payload["content_sha256"] != extension_content_digest(payload):
        raise ValueError("extension content_sha256 does not match canonical payload")
    if payload["manifest_sha256"] != manifest_sha256:
        raise ValueError("extension was built against a different manifest digest")
    if payload["model"] != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("extension model provenance is incompatible")
    frames = derive_frames(manifest)
    stored_frames = tuple(_parse_frame(entry, "manifest") for entry in payload["original_frames"])
    if stored_frames != frames:
        raise ValueError("extension original frames disagree with the manifest-derived frames")
    reference = payload["reference_cue_ids"]
    if set(reference) != set(TEMPLATE_ORDER) or any(reference[frame.template_id] != frame.cue_ids["sg"] for frame in frames):
        raise ValueError("reference cue IDs must be each template's singular cue")
    new_frames = tuple(_parse_frame(entry, "extension") for entry in payload["new_frames"])
    expected_ids = []
    counters: dict[str, int] = {}
    for template_id, text_template in EXTENSION_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected_ids.append((f"{template_id}-new-{counters[template_id]}", template_id, text_template))
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in new_frames] != expected_ids:
        raise ValueError("extension new frames are not the frozen literal frames")
    by_template = {template: [frame for frame in frames if frame.template_id == template] for template in TEMPLATE_ORDER}
    for frame in new_frames:
        if dict(frame.cue_ids) != dict(by_template[frame.template_id][0].cue_ids):
            raise ValueError(f"{frame.frame_id}: new frames must use the template's original cue tokens")
    cue_words = payload["cue_words"]
    if not isinstance(cue_words, list) or len(cue_words) != CUE_WORD_COUNT:
        raise ValueError("extension must contain exactly twelve cue words")
    words: list[tuple[str, int]] = []
    for entry in cue_words:
        _require_exact_keys(entry, {"word", "token_id"}, "cue_word")
        if entry["word"] not in EXTENSION_CUE_WORDS:
            raise ValueError(f"{entry['word']} is not in the frozen cue-word list")
        words.append((entry["word"], int(entry["token_id"])))
    order = [EXTENSION_CUE_WORDS.index(word) for word, _ in words]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError("cue words must follow the frozen order without repeats")
    frames_by_id = {frame.frame_id: frame for frame in frames}
    prompts: list[Prompt] = []
    expected_prompts = [(frame.frame_id, word) for frame in frames for word, _ in words]
    for entry in payload["cue_word_prompts"]:
        _require_exact_keys(entry, {"frame_id", "word", "token_id", "text", "token_ids", "p_c", "p_t"}, "cue_word_prompt")
        frame = frames_by_id.get(entry["frame_id"])
        if frame is None:
            raise ValueError(f"cue-word prompt references unknown frame {entry['frame_id']}")
        if (entry["word"], entry["token_id"]) not in words:
            raise ValueError(f"cue-word prompt uses an unfrozen word {entry['word']}")
        if tuple(entry["token_ids"]) != frame.prompt_ids(entry["token_id"]) or (entry["p_c"], entry["p_t"]) != (frame.p_c, frame.p_t):
            raise ValueError(f"{entry['frame_id']}/{entry['word']}: stored prompt disagrees with the frame")
        prompts.append(Prompt(frame, int(entry["token_id"]), entry["word"]))
    if [(prompt.frame.frame_id, prompt.cue_label) for prompt in prompts] != expected_prompts:
        raise ValueError("cue-word prompts must cover every original frame and word in order")
    return Extension(payload["manifest_sha256"], frames, dict(reference), new_frames, tuple(words), tuple(prompts), payload["content_sha256"])


def load_extension(path: Path, manifest: ScreeningManifest, manifest_sha256: str) -> Extension:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read extension: {path}") from error
    return validate_extension(payload, manifest, manifest_sha256)


# ---------------------------------------------------------------------------
# Results state: one machine-readable artifact, phase isolation, non-execution ledger.


def state_digest(state: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    return sha256_text(canonical_json(unsigned))


_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_path", "manifest_sha256", "extension_path", "extension_sha256",
               "protocol_code_commit", "git_dirty", "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys",
               "discovery", "calibration", "mechanism_versions", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, manifest_sha256: str, extension_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not _COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": sha256_text(manifest_sha256 + extension_sha256 + protocol_code_commit + utc_now())[:16],
        "created_at": utc_now(),
        "manifest_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": manifest_sha256,
        "extension_path": EXTENSION_RELATIVE_PATH,
        "extension_sha256": extension_sha256,
        "protocol_code_commit": protocol_code_commit,
        "git_dirty": False,
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
        "versions": dict(versions),
        "phases": {phase: {"status": "not_started"} for phase in PHASES},
        "executed_prompt_keys": [],
        "executed_noun_keys": [],
        "discovery": {},
        "calibration": {"passes": []},
        "mechanism_versions": [],
        "lock": None,
        "confirmation": None,
        "invalidated_runs": [],
    }


def write_results_state(path: Path, state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    validate_json_safe(payload, path="results_state")
    payload["state_sha256"] = state_digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload["state_sha256"]


def load_results_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseError(f"could not read results state: {path}") from error
    if not isinstance(state, Mapping) or state.get("schema_version") != RESULTS_SCHEMA_VERSION or set(state) != _STATE_KEYS:
        raise PhaseError("results state schema is not recognized")
    if state["state_sha256"] != state_digest(state):
        raise PhaseError("results state digest mismatch: the artifact was modified outside the runner")
    if set(state["phases"]) != set(PHASES) or any(entry.get("status") not in PHASE_STATUSES for entry in state["phases"].values()):
        raise PhaseError("results state phase records are invalid")
    return dict(state)


def assert_provenance_identical(state: Mapping[str, Any], *, manifest_sha256: str, extension_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> None:
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    if not _COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if state["manifest_sha256"] != manifest_sha256 or state["extension_sha256"] != extension_sha256:
        raise PhaseError("manifest or extension digest differs from the recorded run")
    if dict(state["versions"]) != dict(versions):
        raise PhaseError("dependency versions differ from the recorded run")


def record_execution(state: dict[str, Any], prompts: Sequence[Prompt], nouns: Sequence[Noun]) -> None:
    """Every executed prompt and scored noun is recorded; reserve and extension must stay absent before confirm."""
    prompt_keys = set(state["executed_prompt_keys"])
    prompt_keys.update(prompt.key for prompt in prompts)
    state["executed_prompt_keys"] = sorted(prompt_keys)
    noun_keys = set(state["executed_noun_keys"])
    noun_keys.update(noun.key for noun in nouns)
    state["executed_noun_keys"] = sorted(noun_keys)


def assert_not_executed(state: Mapping[str, Any], *, extension: Extension, reserve: Sequence[Noun]) -> None:
    executed_prompts = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in extension.new_frame_prompts()} | {prompt.key for prompt in extension.cue_word_prompts}
    if executed_prompts & forbidden:
        raise PhaseError("extension prompts were executed before the confirmation boundary")
    executed_nouns = set(state["executed_noun_keys"])
    if executed_nouns & {noun.key for noun in reserve}:
        raise PhaseError("reserve nouns were scored before the confirmation boundary")


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    """Enforce the order discover → calibrate (→ revise → calibrate) → lock → confirm → report."""
    if phase not in PHASES:
        raise PhaseError(f"unknown phase {phase}")
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    passes = len(state["calibration"]["passes"])
    if phase == "discover":
        if status["discover"] == "running" and not state["mechanism_versions"]:
            return  # crash recovery: nothing was concluded, the deterministic measurements are recomputed
        if status["discover"] != "not_started":
            raise PhaseError("discover already ran; discovery is never re-run in one protocol version")
    elif phase == "calibrate":
        if status["discover"] != "complete":
            raise PhaseError("calibrate requires the completed discover phase")
        if status["lock"] == "complete":
            raise PhaseError("calibrate cannot run after the lock")
        if passes >= 2:
            raise PhaseError("only two calibration passes are allowed in one protocol version")
        if passes == 1 and status["revise"] != "complete":
            raise PhaseError("a second calibration pass requires a completed revise phase")
        if status["calibrate"] == "complete" and passes == 0:
            raise PhaseError("calibration state is inconsistent")
    elif phase == "revise":
        if passes != 1 or status["revise"] != "not_started":
            raise PhaseError("revise is allowed exactly once, after the first calibration pass")
        last = state["calibration"]["passes"][-1]
        if last.get("floors_passed"):
            raise PhaseError("revise is allowed only when the first calibration pass failed its floors")
    elif phase == "lock":
        if passes == 0 or not state["calibration"]["passes"][-1].get("floors_passed"):
            raise PhaseError("lock requires a completed calibration pass whose floors passed")
        if status["lock"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
    elif phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already ran; a second scientific attempt requires a new protocol version")
    elif phase == "report":
        if status["discover"] != "complete":
            raise PhaseError("report requires at least the completed discover phase")


def manifest_digest_from_path(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload["content_sha256"])


def load_inputs(root: Path) -> tuple[ScreeningManifest, str, Extension]:
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest = load_manifest(manifest_path)
    manifest_sha256 = manifest_digest_from_path(manifest_path)
    extension = load_extension(root / EXTENSION_RELATIVE_PATH, manifest, manifest_sha256)
    return manifest, manifest_sha256, extension


# ---------------------------------------------------------------------------
# Prompt-level execution primitives: sites, captures, exact replacements, readout.

import torch  # noqa: E402  (kept below the model-independent section on purpose)

from .candidate_screening import aligned_patch_shift, canonical_component_id, component_universe  # noqa: E402
from .capture import CapturePlan, CaptureRequest, InstrumentationSettings, run_capture  # noqa: E402
from .components import ComponentKind, ComponentRef, resolve_component  # noqa: E402
from .interventions import (  # noqa: E402
    Intervention,
    InterventionOperation,
    InterventionPlan,
    ReplacementSource,
    run_interventions,
)

ATTN_SETTINGS = InstrumentationSettings(use_attn_result=True)
Site = tuple[str, int]
DENOMINATOR_FLOOR_OVERALL = 0.25
DENOMINATOR_FLOOR_STRATUM = 0.10
RANDOM_SET_COUNT = 100

_KEY_PATTERNS = {
    "head": re.compile(r"^L(\d{2})\.H(\d{2})$"),
    "mlp": re.compile(r"^L(\d{2})\.MLP$"),
    "resid_pre": re.compile(r"^RESID_PRE\.L(\d+)$"),
    "resid_post": re.compile(r"^RESID_POST\.L(\d+)$"),
    "pattern": re.compile(r"^ATTN_PATTERN\.L(\d+)$"),
}


def component_key(ref: ComponentRef) -> str:
    if ref.kind in (ComponentKind.ATTN_HEAD, ComponentKind.MLP_OUT):
        return canonical_component_id(ref)
    if ref.kind is ComponentKind.TOKEN_EMBED:
        return "EMBED"
    if ref.kind is ComponentKind.RESID_PRE:
        return f"RESID_PRE.L{ref.layer}"
    if ref.kind is ComponentKind.RESID_POST:
        return f"RESID_POST.L{ref.layer}"
    if ref.kind is ComponentKind.ATTN_PATTERN:
        return f"ATTN_PATTERN.L{ref.layer}"
    raise ValueError(f"no key for component kind {ref.kind.value}")


def component_ref(key: str) -> ComponentRef:
    if key == "EMBED":
        return ComponentRef(ComponentKind.TOKEN_EMBED)
    if match := _KEY_PATTERNS["head"].match(key):
        return ComponentRef(ComponentKind.ATTN_HEAD, layer=int(match.group(1)), head=int(match.group(2)))
    if match := _KEY_PATTERNS["mlp"].match(key):
        return ComponentRef(ComponentKind.MLP_OUT, layer=int(match.group(1)))
    if match := _KEY_PATTERNS["resid_pre"].match(key):
        return ComponentRef(ComponentKind.RESID_PRE, layer=int(match.group(1)))
    if match := _KEY_PATTERNS["resid_post"].match(key):
        return ComponentRef(ComponentKind.RESID_POST, layer=int(match.group(1)))
    if match := _KEY_PATTERNS["pattern"].match(key):
        return ComponentRef(ComponentKind.ATTN_PATTERN, layer=int(match.group(1)))
    raise ValueError(f"unknown component key {key}")


def universe_keys(model: Any) -> tuple[str, ...]:
    """The 54 head/MLP components in canonical order."""
    return tuple(canonical_component_id(ref) for ref in component_universe(model))


def is_head_key(key: str) -> bool:
    return bool(_KEY_PATTERNS["head"].match(key))


def is_mlp_key(key: str) -> bool:
    return bool(_KEY_PATTERNS["mlp"].match(key))


def key_layer(key: str) -> int:
    for pattern in _KEY_PATTERNS.values():
        if match := pattern.match(key):
            return int(match.group(1))
    raise ValueError(f"key {key} has no layer")


def _model_device(model: Any) -> Any:
    return getattr(getattr(model, "cfg", None), "device", "cpu")


@dataclass(frozen=True)
class PromptRun:
    """Final-position logits and the requested activation slices of one prompt."""

    prompt: Prompt
    logits: torch.Tensor  # [vocab] at p_t
    slices: Mapping[str, torch.Tensor]  # site label "KEY@POS" -> selected-shape tensor (batch 1)
    integrity: tuple[dict[str, Any], ...] = ()

    def slice(self, site: Site) -> torch.Tensor:
        return self.slices[site_label(site)]

    def vector(self, site: Site) -> torch.Tensor:
        return self.slice(site).reshape(-1) if not site[0].startswith("ATTN_PATTERN") else self.slice(site).squeeze(0).squeeze(1)


def site_label(site: Site) -> str:
    return f"{site[0]}@{site[1]}"


def parse_site(label: str) -> Site:
    key, _, position = label.rpartition("@")
    return key, int(position)


def _requests_for(sites: Sequence[Site], *, sequence_length: int) -> dict[Site, CaptureRequest]:
    requests: dict[Site, CaptureRequest] = {}
    for key, position in sites:
        if position < 0 or position >= sequence_length:
            raise ValueError(f"site {site_label((key, position))} is outside the prompt")
        ref = component_ref(key)
        if ref.kind is ComponentKind.ATTN_PATTERN:
            requests[(key, position)] = CaptureRequest(ref, positions=(position,))
        else:
            requests[(key, position)] = CaptureRequest(ref, positions=(position,))
    return requests


def capture_prompt(model: Any, prompt: Prompt, sites: Sequence[Site]) -> PromptRun:
    """One forward: final-position logits plus exactly the requested sites."""
    tokens = torch.tensor([prompt.token_ids], dtype=torch.long, device=_model_device(model))
    requests = _requests_for(tuple(dict.fromkeys(sites)), sequence_length=len(prompt.token_ids))
    plan = CapturePlan(tuple(requests.values()), ATTN_SETTINGS)
    result = run_capture(model, tokens, plan, prepend_bos=False)
    if not result.hook_settings.effective_use_attn_result or result.hook_settings.compatibility_mode:
        raise IncidentError("capture must run with use_attn_result and without compatibility mode")
    slices = {site_label(site): result.activations[request].tensor.clone() for site, request in requests.items()}
    return PromptRun(prompt, result.logits[0, prompt.p_t].detach().float().cpu().clone(), slices)


def _check_execution(execution: Any, ref: ComponentRef, position: int, replacement: torch.Tensor, source: ReplacementSource, model: Any) -> dict[str, Any]:
    problems = []
    expected_hook = resolve_component(ref, model).hook_name
    if execution.component != ref or execution.hook_name != expected_hook:
        problems.append("component/hook mismatch")
    if execution.operation is not InterventionOperation.REPLACE or execution.source is not source:
        problems.append("operation/source mismatch")
    if tuple(execution.shape) != tuple(replacement.shape) or execution.dtype != str(replacement.dtype):
        problems.append("shape/dtype mismatch")
    if tuple(execution.normalized_positions) != (position,):
        problems.append("position mismatch")
    if execution.outside_max_abs_change != 0.0:
        problems.append("outside change")
    if not torch.equal(execution.after.to(replacement.device), replacement):
        problems.append("inexact replacement")
    if problems:
        raise IncidentError(f"{component_key(ref)}@{position}: {', '.join(problems)}")
    return {"site": site_label((component_key(ref), position)), "hook": execution.hook_name, "source": source.value,
            "shape": list(execution.shape), "outside_max_abs_change": execution.outside_max_abs_change}


def run_patched(model: Any, prompt: Prompt, replacements: Mapping[Site, torch.Tensor], sources: Mapping[Site, ReplacementSource], *, capture_sites: Sequence[Site] = ()) -> PromptRun:
    """One forward with exact replacements at the given sites; optional captures of the same run."""
    if set(replacements) != set(sources):
        raise ValueError("every replacement needs exactly one declared source")
    tokens = torch.tensor([prompt.token_ids], dtype=torch.long, device=_model_device(model))
    device = _model_device(model)
    ordered = tuple(replacements)
    interventions = []
    for site in ordered:
        key, position = site
        if position < 0 or position >= len(prompt.token_ids):
            raise ValueError(f"site {site_label(site)} is outside the prompt")
        interventions.append(Intervention(component_ref(key), (position,), InterventionOperation.REPLACE, replacements[site].to(device), sources[site]))
    plan = InterventionPlan(tuple(interventions), ATTN_SETTINGS)
    requests = _requests_for(tuple(dict.fromkeys(capture_sites)), sequence_length=len(prompt.token_ids))
    captures = CapturePlan(tuple(requests.values()), ATTN_SETTINGS) if requests else None
    result = run_interventions(model, tokens, plan, prepend_bos=False, captures=captures)
    if len(result.executions) != len(ordered) or result.hook_settings.compatibility_mode or not result.hook_settings.effective_use_attn_result:
        raise IncidentError("intervention run did not execute every declared replacement")
    integrity = tuple(_check_execution(execution, component_ref(site[0]), site[1], replacements[site].to(device), sources[site], model)
                      for site, execution in zip(ordered, result.executions))
    slices = {site_label(site): result.activations[request].tensor.clone() for site, request in requests.items()}
    return PromptRun(prompt, result.logits[0, prompt.p_t].detach().float().cpu().clone(), slices, integrity)


def contrasts(logits: torch.Tensor, nouns: Sequence[Noun]) -> dict[str, float]:
    """c_N = log P(singular) − log P(plural) for every single-token noun from one logits vector."""
    log_probs = logits.float().log_softmax(dim=-1)
    result: dict[str, float] = {}
    for noun in nouns:
        if not noun.single_token:
            continue
        value = float(log_probs[noun.sg_ids[0]] - log_probs[noun.pl_ids[0]])
        if not torch.isfinite(torch.tensor(value)):
            raise IncidentError(f"non-finite contrast for {noun.key}")
        result[noun.lexical_key] = value
    return result


def pair_centered(slice_a: torch.Tensor, slice_b: torch.Tensor) -> torch.Tensor:
    """Midpoint of the two observed states (pair-centered neutralization)."""
    if slice_a.shape != slice_b.shape or slice_a.dtype != slice_b.dtype:
        raise ValueError("pair-centered neutralization requires equal shapes and dtypes")
    return (0.5 * (slice_a + slice_b)).to(slice_a.dtype)


def assert_prefix_identical(run_a: PromptRun, run_b: PromptRun, sites: Sequence[Site]) -> None:
    """Activations at positions before the cue must be bitwise identical within a pair."""
    for site in sites:
        if not torch.equal(run_a.slice(site), run_b.slice(site)):
            raise IncidentError(f"prefix activation {site_label(site)} differs within a matched pair")


@dataclass(frozen=True)
class CaseRow:
    """One (frame, noun) case: its full contrast shift and one intervention's aligned shift."""

    frame_id: str
    template_id: str
    lexical_key: str
    rule_class: str
    d_full: float
    d_patch: float

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "template_id": self.template_id, "lexical_key": self.lexical_key,
                "rule_class": self.rule_class, "d_full": self.d_full, "d_patch": self.d_patch}


def case_rows(frame: Frame, nouns: Sequence[Noun], c_a: Mapping[str, float], c_b: Mapping[str, float], c_a_from_b: Mapping[str, float], c_b_from_a: Mapping[str, float]) -> tuple[CaseRow, ...]:
    rows = []
    for noun in nouns:
        if not noun.single_token:
            continue
        key = noun.lexical_key
        rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, c_a[key] - c_b[key],
                            aligned_patch_shift(c_a[key], c_b[key], c_a_from_b[key], c_b_from_a[key])))
    return tuple(rows)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return float(sum(values) / len(values))


def stratified_recovery(rows: Sequence[CaseRow]) -> dict[str, Any]:
    """Aggregate recovery overall, per template, and per rule class, with denominator floors."""
    def summary(subset: Sequence[CaseRow], floor: float) -> dict[str, Any]:
        denominator = _mean([row.d_full for row in subset])
        numerator = _mean([row.d_patch for row in subset])
        valid = denominator >= floor
        return {"cases": len(subset), "mean_d_full": denominator, "mean_d_patch": numerator,
                "denominator_valid": valid, "recovery": (numerator / denominator) if valid else None}

    templates = sorted({row.template_id for row in rows}, key=TEMPLATE_ORDER.index)
    classes = sorted({row.rule_class for row in rows}, key=RULE_CLASSES.index)
    return {
        "overall": summary(rows, DENOMINATOR_FLOOR_OVERALL),
        "templates": {template: summary([row for row in rows if row.template_id == template], DENOMINATOR_FLOOR_STRATUM) for template in templates},
        "rule_classes": {rule: summary([row for row in rows if row.rule_class == rule], DENOMINATOR_FLOOR_STRATUM) for rule in classes},
    }


def deterministic_random_sets(universe: Sequence[str], size: int, *, count: int = RANDOM_SET_COUNT, seed: int = CONTROL_SEED) -> tuple[tuple[str, ...], ...]:
    """Size-``size`` component sets drawn uniformly without replacement, in a fixed order."""
    import random

    if size <= 0 or size > len(universe):
        raise ValueError("random set size must be within the universe")
    generator = random.Random(seed)
    ordered = tuple(universe)
    return tuple(tuple(sorted(generator.sample(ordered, size), key=ordered.index)) for _ in range(count))


# ---------------------------------------------------------------------------
# Model weights needed for exact readout accounting and the mechanism program.


@dataclass(frozen=True)
class Weights:
    """Detached float32 CPU copies of the weights the analyses and the program read."""

    W_E: torch.Tensor  # [vocab, d_model]
    W_U: torch.Tensor  # [d_model, vocab]
    b_U: torch.Tensor  # [vocab]
    ln_final_w: torch.Tensor
    ln_final_b: torch.Tensor
    eps: float
    attn_b_O: tuple[torch.Tensor, ...]  # per layer [d_model]
    ln2_0_w: torch.Tensor
    ln2_0_b: torch.Tensor
    mlp0_W_in: torch.Tensor  # [d_model, d_mlp]
    mlp0_b_in: torch.Tensor
    mlp0_W_out: torch.Tensor  # [d_mlp, d_model]
    mlp0_b_out: torch.Tensor
    act_fn: str

    @classmethod
    def from_model(cls, model: Any) -> "Weights":
        def grab(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to("cpu", torch.float32).clone()

        cfg = model.cfg
        if str(getattr(cfg, "normalization_type", "LN")) != "LN":
            raise IncidentError("the exact readout requires LayerNorm normalization")
        block0 = model.blocks[0]
        return cls(
            W_E=grab(model.embed.W_E), W_U=grab(model.unembed.W_U), b_U=grab(model.unembed.b_U),
            ln_final_w=grab(model.ln_final.w), ln_final_b=grab(model.ln_final.b), eps=float(cfg.eps),
            attn_b_O=tuple(grab(model.blocks[layer].attn.b_O) for layer in range(int(cfg.n_layers))),
            ln2_0_w=grab(block0.ln2.w), ln2_0_b=grab(block0.ln2.b),
            mlp0_W_in=grab(block0.mlp.W_in), mlp0_b_in=grab(block0.mlp.b_in),
            mlp0_W_out=grab(block0.mlp.W_out), mlp0_b_out=grab(block0.mlp.b_out),
            act_fn=str(cfg.act_fn),
        )

    def u(self, noun: Noun) -> torch.Tensor:
        """u_N = W_U[:, plural] − W_U[:, singular] in float64."""
        return (self.W_U[:, noun.pl_ids[0]] - self.W_U[:, noun.sg_ids[0]]).double()


def exact_layer_norm(r: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm with population variance (correction=0), in the input's dtype."""
    centered = r - r.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return weight * centered / torch.sqrt(variance + eps) + bias


def lexicon_vector(weights: Weights, token_id: int) -> torch.Tensor:
    """Weight-only L00.MLP output for a token: MLP_0(ln2_0(embed[token]))."""
    if weights.act_fn != "gelu":
        raise IncidentError(f"unsupported activation {weights.act_fn}; the lexicon assumes exact GELU")
    hidden = exact_layer_norm(weights.W_E[int(token_id)], weights.ln2_0_w, weights.ln2_0_b, weights.eps)
    pre = hidden @ weights.mlp0_W_in + weights.mlp0_b_in
    return torch.nn.functional.gelu(pre) @ weights.mlp0_W_out + weights.mlp0_b_out


# ---------------------------------------------------------------------------
# Clean-run cache with every site the discovery measurements read.


def standard_sites(model: Any, prompt: Prompt) -> tuple[Site, ...]:
    n_layers = int(model.cfg.n_layers)
    positions = sorted({prompt.p_c, prompt.p_t})
    sites: list[Site] = []
    for position in positions:
        sites.extend((key, position) for key in universe_keys(model))
        sites.append(("EMBED", position))
        sites.extend((f"RESID_PRE.L{layer}", position) for layer in range(n_layers))
        sites.append((f"RESID_POST.L{n_layers - 1}", position))
    sites.extend((f"ATTN_PATTERN.L{layer}", prompt.p_t) for layer in range(n_layers))
    for position in range(prompt.p_c):
        sites.append((f"RESID_POST.L{n_layers - 1}", position))
        sites.append(("EMBED", position))
    return tuple(dict.fromkeys(sites))


@dataclass
class PromptCache:
    """Clean runs and readouts for a fixed prompt set; the noun set is fixed per cache."""

    model: Any
    nouns: tuple[Noun, ...]
    runs: dict[str, PromptRun] = field(default_factory=dict)
    readouts: dict[str, dict[str, float]] = field(default_factory=dict)

    def run(self, prompt: Prompt) -> PromptRun:
        if prompt.key not in self.runs:
            self.runs[prompt.key] = capture_prompt(self.model, prompt, standard_sites(self.model, prompt))
            self.readouts[prompt.key] = contrasts(self.runs[prompt.key].logits, self.nouns)
        return self.runs[prompt.key]

    def c(self, prompt: Prompt) -> dict[str, float]:
        self.run(prompt)
        return self.readouts[prompt.key]

    def final_residual(self, prompt: Prompt) -> torch.Tensor:
        n_layers = int(self.model.cfg.n_layers)
        return self.run(prompt).vector((f"RESID_POST.L{n_layers - 1}", prompt.p_t))


def frame_prompts(frame: Frame) -> tuple[Prompt, Prompt]:
    return Prompt(frame, frame.cue_ids["sg"], "sg"), Prompt(frame, frame.cue_ids["pl"], "pl")


def prefix_sites(model: Any, prompt: Prompt) -> tuple[Site, ...]:
    n_layers = int(model.cfg.n_layers)
    return tuple((key, position) for position in range(prompt.p_c) for key in (f"RESID_POST.L{n_layers - 1}", "EMBED"))


# ---------------------------------------------------------------------------
# Role-addressed interventions.

RoleSite = tuple[str, str]  # (component key, "p_c" | "p_t")


def role_position(prompt: Prompt, role: str) -> int:
    if role == "p_c":
        return prompt.p_c
    if role == "p_t":
        return prompt.p_t
    raise ValueError(f"unknown role {role}")


def _site(prompt: Prompt, role_site: RoleSite) -> Site:
    return role_site[0], role_position(prompt, role_site[1])


def counterfactual_rows(model: Any, cache: PromptCache, frame: Frame, role_sites: Sequence[RoleSite], *,
                        freeze: Sequence[RoleSite] = (), source_frame: Frame | None = None, flip_source: bool = True,
                        capture_role_sites: Sequence[RoleSite] = ()) -> tuple[tuple[CaseRow, ...], dict[str, PromptRun]]:
    """Bidirectional replacement of ``role_sites`` from a source run, optionally freezing other sites clean.

    With ``source_frame`` None the source is the frame's own matched prompt (counterfactual). With a
    source frame and ``flip_source`` True the source is that frame's opposite-cue prompt; with
    ``flip_source`` False it is the same-cue prompt (resample controls). Returns the aligned rows and
    the two patched runs keyed ``a_from_b`` / ``b_from_a``.
    """
    sg, pl = frame_prompts(frame)
    src_sg, src_pl = frame_prompts(source_frame) if source_frame is not None else (sg, pl)
    if not flip_source and source_frame is None:
        raise ValueError("a same-cue replacement needs a source frame")
    run_sg, run_pl = cache.run(sg), cache.run(pl)
    run_src_sg, run_src_pl = cache.run(src_sg), cache.run(src_pl)
    source = ReplacementSource.REFERENCE if source_frame is None else ReplacementSource.RESAMPLE

    def build(target: Prompt, target_run: PromptRun, donor_run: PromptRun, donor: Prompt) -> tuple[dict[Site, torch.Tensor], dict[Site, ReplacementSource]]:
        replacements: dict[Site, torch.Tensor] = {}
        sources: dict[Site, ReplacementSource] = {}
        for role_site in role_sites:
            site = _site(target, role_site)
            replacements[site] = donor_run.slice(_site(donor, role_site))
            sources[site] = source
        for role_site in freeze:
            site = _site(target, role_site)
            if site in replacements:
                raise ValueError(f"{site_label(site)} cannot be both replaced and frozen")
            replacements[site] = target_run.slice(site)
            sources[site] = ReplacementSource.REFERENCE
        return replacements, sources

    donor_for_sg = (src_pl if flip_source else src_sg)
    donor_for_pl = (src_sg if flip_source else src_pl)
    captures = tuple(_site(sg, role_site) for role_site in capture_role_sites)
    rep, src = build(sg, run_sg, cache.run(donor_for_sg), donor_for_sg)
    a_from_b = run_patched(model, sg, rep, src, capture_sites=captures)
    captures = tuple(_site(pl, role_site) for role_site in capture_role_sites)
    rep, src = build(pl, run_pl, cache.run(donor_for_pl), donor_for_pl)
    b_from_a = run_patched(model, pl, rep, src, capture_sites=captures)
    rows = case_rows(frame, cache.nouns, cache.c(sg), cache.c(pl), contrasts(a_from_b.logits, cache.nouns), contrasts(b_from_a.logits, cache.nouns))
    return rows, {"a_from_b": a_from_b, "b_from_a": b_from_a}


def neutralized_runs(model: Any, cache: PromptCache, frame: Frame, role_sites: Sequence[RoleSite], *, capture_role_sites: Sequence[RoleSite] = ()) -> dict[str, PromptRun]:
    """Both prompts with ``role_sites`` replaced by their pair-centered midpoints."""
    sg, pl = frame_prompts(frame)
    run_sg, run_pl = cache.run(sg), cache.run(pl)
    results = {}
    for label, prompt in (("sg", sg), ("pl", pl)):
        replacements = {_site(prompt, role_site): pair_centered(run_sg.slice(_site(sg, role_site)), run_pl.slice(_site(pl, role_site))) for role_site in role_sites}
        sources = {site: ReplacementSource.MEAN for site in replacements}
        results[label] = run_patched(model, prompt, replacements, sources, capture_sites=tuple(_site(prompt, role_site) for role_site in capture_role_sites))
    return results


def neutralization_rows(cache: PromptCache, frame: Frame, runs: Mapping[str, PromptRun]) -> tuple[CaseRow, ...]:
    """Rows whose ``d_patch`` is the contrast *lost* under neutralization: d_full − d_neutralized."""
    sg, pl = frame_prompts(frame)
    c_a, c_b = cache.c(sg), cache.c(pl)
    c_a_n, c_b_n = contrasts(runs["sg"].logits, cache.nouns), contrasts(runs["pl"].logits, cache.nouns)
    rows = []
    for noun in cache.nouns:
        if not noun.single_token:
            continue
        key = noun.lexical_key
        full = c_a[key] - c_b[key]
        rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, full, full - (c_a_n[key] - c_b_n[key])))
    return tuple(rows)


def sign_retention(cache: PromptCache, frame: Frame, runs: Mapping[str, PromptRun]) -> dict[str, int]:
    c_a_n, c_b_n = contrasts(runs["sg"].logits, cache.nouns), contrasts(runs["pl"].logits, cache.nouns)
    keys = [noun.lexical_key for noun in cache.nouns if noun.single_token]
    return {"retained": sum(1 for key in keys if c_a_n[key] > 0 and c_b_n[key] < 0), "total": len(keys)}


def isolation_runs(model: Any, cache: PromptCache, frame: Frame, keep: Sequence[RoleSite]) -> dict[str, PromptRun]:
    """Hold ``keep`` clean and neutralize every other universe component at p_t (and p_c when distinct)."""
    kept = {site for site in keep}
    roles = ("p_t",) if frame.p_c == frame.p_t else ("p_c", "p_t")
    complement = [(key, role) for role in roles for key in universe_keys(model) if (key, role) not in kept]
    return neutralized_runs(model, cache, frame, complement)


def joint_rows(rows_by_frame: Mapping[str, Sequence[CaseRow]]) -> tuple[CaseRow, ...]:
    return tuple(row for rows in rows_by_frame.values() for row in rows)


# ---------------------------------------------------------------------------
# Site axes, number variables, and exact direct-effect accounting.


@dataclass(frozen=True)
class SiteAxis:
    """Centered, scaled projection parameters of one site."""

    label: str
    mu: torch.Tensor
    direction: torch.Tensor  # unit
    sigma: float

    def n(self, vector: torch.Tensor) -> float:
        return float(((vector.double() - self.mu.double()) @ self.direction.double()) / self.sigma)

    def projection(self, vector: torch.Tensor) -> float:
        return float(vector.double() @ self.direction.double())

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "sigma": self.sigma, "mu_norm": float(self.mu.norm()), "direction_norm": float(self.direction.norm())}


def site_axis(label: str, vectors_a: Sequence[torch.Tensor], vectors_b: Sequence[torch.Tensor]) -> SiteAxis:
    if len(vectors_a) != len(vectors_b) or not vectors_a:
        raise ValueError("site axis needs paired vectors")
    a = torch.stack([v.double().reshape(-1) for v in vectors_a])
    b = torch.stack([v.double().reshape(-1) for v in vectors_b])
    difference = (b - a).mean(dim=0)
    norm = float(difference.norm())
    if norm <= 0.0:
        raise IncidentError(f"site {label} has a zero number axis")
    direction = difference / norm
    mu = torch.cat([a, b]).mean(dim=0)
    sigma = 0.5 * float(((b - a) @ direction).mean())
    if sigma <= 0.0:
        raise IncidentError(f"site {label} has a non-positive scale")
    return SiteAxis(label, mu.float(), direction.float(), sigma)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.double().reshape(1, -1), right.double().reshape(1, -1)).item())


def summed_vector(run: PromptRun, sites: Sequence[Site]) -> torch.Tensor:
    return torch.stack([run.vector(site) for site in sites]).sum(dim=0)


def direct_effects(weights: Weights, run: PromptRun, nouns: Sequence[Noun], *, p_t: int, n_layers: int, universe: Sequence[str],
                   final_residual: torch.Tensor | None = None) -> dict[str, Any]:
    """Exact float64 decomposition of c_N through the final LayerNorm for one run.

    Terms: EMBED, every head result and MLP output at p_t, every attention output bias, and the β term.
    """
    terms: dict[str, torch.Tensor] = {"EMBED": run.vector(("EMBED", p_t)).double()}
    for key in universe:
        terms[key] = run.vector((key, p_t)).double()
    for layer, bias in enumerate(weights.attn_b_O):
        terms[f"b_O.L{layer}"] = bias.double()
    residual = (final_residual if final_residual is not None else run.vector((f"RESID_POST.L{n_layers - 1}", p_t))).double()
    reconstructed = torch.stack(list(terms.values())).sum(dim=0)
    residual_error = float((reconstructed - residual).abs().max())
    # The identity is stated on the decomposition's own whole; the captured residual is checked separately.
    residual = reconstructed
    scale = float(torch.sqrt(((residual - residual.mean()) ** 2).mean() + weights.eps))
    gamma, beta = weights.ln_final_w.double(), weights.ln_final_b.double()
    model_c = contrasts(run.logits, nouns)
    per_noun: dict[str, dict[str, float]] = {}
    max_identity_error = 0.0
    max_model_gap = 0.0
    for noun in nouns:
        if not noun.single_token:
            continue
        u = weights.u(noun)
        effects = {name: float(-u @ (gamma * (term - term.mean()) / scale)) for name, term in terms.items()}
        effects["beta"] = float(-u @ beta)
        c_recon = float(-u @ (gamma * (residual - residual.mean()) / scale + beta))
        identity_error = abs(sum(effects.values()) - c_recon)
        max_identity_error = max(max_identity_error, identity_error)
        max_model_gap = max(max_model_gap, abs(c_recon - model_c[noun.lexical_key]))
        effects["c_reconstructed"] = c_recon
        effects["c_model"] = model_c[noun.lexical_key]
        per_noun[noun.lexical_key] = effects
    return {"per_noun": per_noun, "scale": scale, "residual_sum_error": residual_error,
            "max_identity_error": max_identity_error, "max_model_gap": max_model_gap}


DIRECT_EFFECT_IDENTITY_TOLERANCE = 1e-4
DIRECT_EFFECT_MODEL_GAP_TOLERANCE = 1e-2
RESIDUAL_SUM_TOLERANCE = 1e-3


def check_direct_effects(effects: Mapping[str, Any], *, where: str) -> None:
    if effects["max_identity_error"] > DIRECT_EFFECT_IDENTITY_TOLERANCE:
        raise IncidentError(f"{where}: direct effects do not reconstruct the contrast ({effects['max_identity_error']:.2e} nats)")
    if effects["residual_sum_error"] > RESIDUAL_SUM_TOLERANCE:
        raise IncidentError(f"{where}: component outputs do not sum to the final residual ({effects['residual_sum_error']:.2e})")
    if effects["max_model_gap"] > DIRECT_EFFECT_MODEL_GAP_TOLERANCE:
        raise IncidentError(f"{where}: reconstructed contrast differs from the model's logits ({effects['max_model_gap']:.2e} nats)")


def pair_direct_effects(effects_a: Mapping[str, Any], effects_b: Mapping[str, Any]) -> dict[str, float]:
    """Mean over nouns of DE_term(x_A) − DE_term(x_B): each term's direct share of d_full in nats."""
    nouns = list(effects_a["per_noun"])
    names = [name for name in effects_a["per_noun"][nouns[0]] if name not in {"c_reconstructed", "c_model"}]
    return {name: _mean([effects_a["per_noun"][noun][name] - effects_b["per_noun"][noun][name] for noun in nouns]) for name in names}


# ---------------------------------------------------------------------------
# Tier A measurements A1–A9. Each returns a JSON-safe dict; the runner writes it
# before the next step is interpreted.

from .candidate_screening import score_behavior_case  # noqa: E402

SCREENING_REPORT_CONSTANTS = {  # from screening/behavior-candidates/report-2026-09-17.md
    "development_mean_d_full": 5.231,
    "validation_denominators": {"overall": 4.639, "cardinal": 4.606, "coordinated-adjective": 4.969, "quantifier": 4.343},
}
A1_CASE_TOLERANCE = 1e-6
A1_AGGREGATE_TOLERANCE = 1e-3
A1_PROMPT_LEVEL_TOLERANCE = 1e-4
A3_FULL_COUNTERFACTUAL_TOLERANCE = 1e-4
AXIS_SUM_TOLERANCE = 1e-3


@dataclass
class DiscoveryContext:
    model: Any
    weights: Weights
    manifest: ScreeningManifest | None
    frames: tuple[Frame, ...]
    nouns: tuple[Noun, ...]
    cache: PromptCache

    @property
    def n_layers(self) -> int:
        return int(self.model.cfg.n_layers)

    @property
    def universe(self) -> tuple[str, ...]:
        return universe_keys(self.model)

    def frames_of(self, template_id: str) -> tuple[Frame, ...]:
        return tuple(frame for frame in self.frames if frame.template_id == template_id)

    @property
    def coordinated(self) -> tuple[Frame, ...]:
        return self.frames_of(COORDINATED_TEMPLATE)


def new_discovery_context(model: Any, manifest: ScreeningManifest, nouns: Sequence[Noun]) -> DiscoveryContext:
    frames = derive_frames(manifest)
    return DiscoveryContext(model, Weights.from_model(model), manifest, frames, tuple(nouns), PromptCache(model, tuple(nouns)))


def _rows_summary(rows: Sequence[CaseRow]) -> dict[str, Any]:
    return stratified_recovery(rows)


def _recovery(rows: Sequence[CaseRow]) -> float | None:
    return stratified_recovery(rows)["overall"]["recovery"]


# -- A1 ---------------------------------------------------------------------

def _screening_measurements(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    screen = payload["behavioral"].get(PYTHIA_70M.model_id) or {}
    return dict(screen.get("measurements") or {})


def a1_baseline(ctx: DiscoveryContext, screening_results_path: Path | None) -> dict[str, Any]:
    """Recompute the behavioral baselines case by case and check them against the screen."""
    stored = _screening_measurements(screening_results_path)
    measurements: dict[str, Any] = {}
    max_case_gap = 0.0
    compared = 0
    for split in (Split.DEVELOPMENT, Split.HOLDOUT):
        for case in regular_plural_cases(ctx.manifest, split):
            measured = score_behavior_case(ctx.model, case).to_dict()
            measurements[case.case_id] = measured
            if stored is not None:
                reference = stored.get(case.case_id)
                if reference is None:
                    raise IncidentError(f"{case.case_id} is missing from the screening results")
                for condition in ("x_a", "x_b"):
                    gap = abs(measured[condition]["contrast"] - reference[condition]["contrast"])
                    max_case_gap = max(max_case_gap, gap)
                    if gap > A1_CASE_TOLERANCE:
                        raise IncidentError(f"{case.case_id}/{condition}: contrast differs from the screen by {gap:.2e}")
                compared += 1
    development = [case for case in regular_plural_cases(ctx.manifest, Split.DEVELOPMENT)]
    dev_d_full = [measurements[case.case_id]["d_full"] for case in development]
    aggregates = {"development_mean_d_full": _mean(dev_d_full)}
    validation = [case for case in development if case.compactness_partition is not None and case.compactness_partition.value == "validation"]
    aggregates["validation_denominators"] = {"overall": _mean([measurements[case.case_id]["d_full"] for case in validation])}
    for template in TEMPLATE_ORDER:
        aggregates["validation_denominators"][template] = _mean([measurements[case.case_id]["d_full"] for case in validation if case.template_id == template])
    gaps = {"development_mean_d_full": abs(aggregates["development_mean_d_full"] - SCREENING_REPORT_CONSTANTS["development_mean_d_full"])}
    for name, value in SCREENING_REPORT_CONSTANTS["validation_denominators"].items():
        gaps[f"validation_{name}"] = abs(aggregates["validation_denominators"][name] - value)
    for name, gap in gaps.items():
        if gap > A1_AGGREGATE_TOLERANCE:
            raise IncidentError(f"{name}: aggregate differs from the screening report by {gap:.2e}")
    # Prompt-level readout must agree with the case-level teacher-forced contrasts.
    max_prompt_gap = 0.0
    by_key = {noun.lexical_key: noun for noun in ctx.nouns}
    for case in development:
        frame = next(frame for frame in ctx.frames if frame.prompt_ids(frame.cue_ids["sg"]) == case.x_a.prompt_token_ids)
        sg, pl = frame_prompts(frame)
        noun = by_key[case.lexical_key]
        gap = max(abs(ctx.cache.c(sg)[noun.lexical_key] - measurements[case.case_id]["x_a"]["contrast"]),
                  abs(ctx.cache.c(pl)[noun.lexical_key] - measurements[case.case_id]["x_b"]["contrast"]))
        max_prompt_gap = max(max_prompt_gap, gap)
    if max_prompt_gap > A1_PROMPT_LEVEL_TOLERANCE:
        raise IncidentError(f"prompt-level readout differs from case-level scoring by {max_prompt_gap:.2e}")
    return {"screening_results_compared": compared, "max_case_gap": max_case_gap, "aggregates": aggregates, "aggregate_gaps": gaps,
            "max_prompt_level_gap": max_prompt_gap, "measurements": measurements}


# -- A2 ---------------------------------------------------------------------

def a2_position_map(ctx: DiscoveryContext) -> dict[str, Any]:
    """Singleton counterfactual replacement of every component at p_c and p_t, with prefix-identity checks."""
    for frame in ctx.frames:
        sg, pl = frame_prompts(frame)
        assert_prefix_identical(ctx.cache.run(sg), ctx.cache.run(pl), prefix_sites(ctx.model, sg))
    roles_by_frame = {frame.frame_id: (("p_t",) if frame.p_c == frame.p_t else ("p_c", "p_t")) for frame in ctx.frames}
    rows: dict[str, dict[str, list[CaseRow]]] = {"p_c": {}, "p_t": {}}
    for frame in ctx.frames:
        for role in roles_by_frame[frame.frame_id]:
            for key in ctx.universe:
                frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, [(key, role)])
                rows[role].setdefault(key, []).extend(frame_rows)
    result: dict[str, Any] = {"prefix_identity": "verified", "roles": {}}
    for role, by_key in rows.items():
        entries = {}
        for key, key_rows in by_key.items():
            summary = stratified_recovery(key_rows)
            entries[key] = {"mean_d_patch": summary["overall"]["mean_d_patch"], "recovery": summary["overall"]["recovery"],
                            "templates": {template: value["recovery"] for template, value in summary["templates"].items()},
                            "template_mean_d_patch": {template: value["mean_d_patch"] for template, value in summary["templates"].items()}}
        result["roles"][role] = entries
    coordinated_p_t = {key: entry["template_mean_d_patch"].get(COORDINATED_TEMPLATE, 0.0) for key, entry in result["roles"]["p_t"].items()}
    result["rankings"] = {
        "p_t_all_templates": sorted(ctx.universe, key=lambda key: (-result["roles"]["p_t"][key]["mean_d_patch"], ctx.universe.index(key))),
        "p_t_coordinated": sorted(ctx.universe, key=lambda key: (-coordinated_p_t[key], ctx.universe.index(key))),
        "p_c_coordinated": sorted(ctx.universe, key=lambda key: (-result["roles"]["p_c"][key]["mean_d_patch"], ctx.universe.index(key))),
    }
    return result


# -- A3 ---------------------------------------------------------------------

def a3_layer_profile(ctx: DiscoveryContext) -> dict[str, Any]:
    profile = {}
    for layer in range(ctx.n_layers):
        layer_rows: list[CaseRow] = []
        for frame in ctx.coordinated:
            rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, [(f"RESID_PRE.L{layer}", "p_c")])
            layer_rows.extend(rows)
        profile[f"L{layer}"] = stratified_recovery(layer_rows)["overall"]
    full = profile["L0"]["recovery"]
    if full is None or abs(full - 1.0) > A3_FULL_COUNTERFACTUAL_TOLERANCE:
        raise IncidentError(f"layer-0 residual replacement at p_c should equal the full counterfactual (recovery {full})")
    return {"recovery_by_layer": profile}


# -- A4 ---------------------------------------------------------------------

def a4_attention(ctx: DiscoveryContext) -> dict[str, Any]:
    n_heads = int(ctx.model.cfg.n_heads)
    to_cue: dict[str, list[float]] = {}
    argmax_key: dict[str, list[int]] = {}
    for frame in ctx.coordinated:
        for prompt in frame_prompts(frame):
            run = ctx.cache.run(prompt)
            for layer in range(ctx.n_layers):
                pattern = run.vector((f"ATTN_PATTERN.L{layer}", prompt.p_t))  # [n_heads, n_keys]
                for head in range(n_heads):
                    key = f"L{layer:02d}.H{head:02d}"
                    to_cue.setdefault(key, []).append(float(pattern[head, prompt.p_c]))
                    argmax_key.setdefault(key, []).append(int(pattern[head].argmax()))
    heads = {key: {"attention_to_cue": _mean(values), "attention_to_cue_by_prompt": values, "argmax_keys": argmax_key[key]} for key, values in to_cue.items()}
    ranking = sorted(heads, key=lambda key: (-heads[key]["attention_to_cue"], key))
    return {"heads": heads, "ranking_by_attention_to_cue": ranking}


# -- A5 ---------------------------------------------------------------------

def d_num_axes(ctx: DiscoveryContext) -> dict[str, Any]:
    """Mean pair-difference axes per template at p_c and p_t for every residual layer and the final residual."""
    axes: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for template in TEMPLATE_ORDER:
        axes[template] = {}
        frames = ctx.frames_of(template)
        for role in ("p_c", "p_t"):
            layer_axes = {}
            for layer in range(ctx.n_layers):
                key = f"RESID_PRE.L{layer}"
                diffs = [ctx.cache.run(pl).vector((key, role_position(pl, role))) - ctx.cache.run(sg).vector((key, role_position(sg, role))) for sg, pl in map(frame_prompts, frames)]
                layer_axes[key] = torch.stack(diffs).double().mean(dim=0)
            final = f"RESID_POST.L{ctx.n_layers - 1}"
            diffs = [ctx.cache.run(pl).vector((final, role_position(pl, role))) - ctx.cache.run(sg).vector((final, role_position(sg, role))) for sg, pl in map(frame_prompts, frames)]
            layer_axes[final] = torch.stack(diffs).double().mean(dim=0)
            axes[template][role] = layer_axes
    return axes


def a5_axes_and_direct_effects(ctx: DiscoveryContext, *, e_keys: Sequence[str], t_keys: Sequence[str], r_keys: Sequence[str], l_r: int, l_t: int | None) -> dict[str, Any]:
    """Number axes, site variables, component contributions, cosines, and exact direct effects."""
    axes = d_num_axes(ctx)
    final = f"RESID_POST.L{ctx.n_layers - 1}"
    norms = {template: {role: {key: float(vector.norm()) for key, vector in layer_axes.items()} for role, layer_axes in roles.items()} for template, roles in axes.items()}
    cosines = {"across_templates_p_t": {}, "across_frames_p_t": {}}
    for key in list(axes[TEMPLATE_ORDER[0]]["p_t"]):
        pairs = {}
        for i, left in enumerate(TEMPLATE_ORDER):
            for right in TEMPLATE_ORDER[i + 1:]:
                pairs[f"{left}|{right}"] = cosine(axes[left]["p_t"][key], axes[right]["p_t"][key])
        cosines["across_templates_p_t"][key] = pairs
        frame_cos = {}
        for template in TEMPLATE_ORDER:
            frames = ctx.frames_of(template)
            (sg1, pl1), (sg2, pl2) = frame_prompts(frames[0]), frame_prompts(frames[1])
            d1 = ctx.cache.run(pl1).vector((key, pl1.p_t)) - ctx.cache.run(sg1).vector((key, sg1.p_t))
            d2 = ctx.cache.run(pl2).vector((key, pl2.p_t)) - ctx.cache.run(sg2).vector((key, sg2.p_t))
            frame_cos[template] = cosine(d1, d2)
        cosines["across_frames_p_t"][key] = frame_cos
    # Site axes.
    e_a = [summed_vector(ctx.cache.run(sg), [(key, sg.p_c) for key in e_keys]) for sg, _ in map(frame_prompts, ctx.frames)]
    e_b = [summed_vector(ctx.cache.run(pl), [(key, pl.p_c) for key in e_keys]) for _, pl in map(frame_prompts, ctx.frames)]
    site_axes = {"E": site_axis("E", e_a, e_b)}
    if t_keys:
        t_a = [summed_vector(ctx.cache.run(sg), [(key, sg.p_t) for key in t_keys]) for sg, _ in map(frame_prompts, ctx.coordinated)]
        t_b = [summed_vector(ctx.cache.run(pl), [(key, pl.p_t) for key in t_keys]) for _, pl in map(frame_prompts, ctx.coordinated)]
        site_axes["T"] = site_axis("T", t_a, t_b)
    r_in_key = f"RESID_PRE.L{l_r}"
    r_a = [ctx.cache.run(sg).vector((r_in_key, sg.p_t)) for sg, _ in map(frame_prompts, ctx.frames)]
    r_b = [ctx.cache.run(pl).vector((r_in_key, pl.p_t)) for _, pl in map(frame_prompts, ctx.frames)]
    site_axes["R_in"] = site_axis("R_in", r_a, r_b)
    if r_keys:
        ro_a = [summed_vector(ctx.cache.run(sg), [(key, sg.p_t) for key in r_keys]) for sg, _ in map(frame_prompts, ctx.frames)]
        ro_b = [summed_vector(ctx.cache.run(pl), [(key, pl.p_t) for key in r_keys]) for _, pl in map(frame_prompts, ctx.frames)]
        site_axes["R_out"] = site_axis("R_out", ro_a, ro_b)
    variables = {label: {prompt.key: axis.n(_site_vector(ctx, prompt, label, e_keys, t_keys, r_keys, l_r)) for frame in ctx.frames for prompt in frame_prompts(frame)
                         if label != "T" or frame.template_id == COORDINATED_TEMPLATE}
                 for label, axis in site_axes.items()}
    # Component number contributions at L_R, per template and overall (exact decomposition of what arrives).
    contributions = {}
    direction = site_axes["R_in"].direction
    for template in list(TEMPLATE_ORDER) + ["all"]:
        frames = ctx.frames if template == "all" else ctx.frames_of(template)
        entries = {}
        keys = ["EMBED"] + [key for key in ctx.universe if key_layer(key) < l_r]
        for key in keys:
            diffs = [ctx.cache.run(pl).vector((key, pl.p_t)) - ctx.cache.run(sg).vector((key, sg.p_t)) for sg, pl in map(frame_prompts, frames)]
            entries[key] = float(torch.stack(diffs).double().mean(dim=0) @ direction.double())
        total = sum(entries.values())
        axis_projection = float(torch.stack([ctx.cache.run(pl).vector((r_in_key, pl.p_t)) - ctx.cache.run(sg).vector((r_in_key, sg.p_t)) for sg, pl in map(frame_prompts, frames)]).double().mean(dim=0) @ direction.double())
        if abs(total - axis_projection) > AXIS_SUM_TOLERANCE:
            raise IncidentError(f"{template}: component contributions ({total:.4f}) do not sum to the axis projection ({axis_projection:.4f})")
        contributions[template] = {"entries": entries, "total": total, "axis_projection": axis_projection}
    # Encoding branch decomposition at the transport layer's input, coordinated only.
    encoding = None
    if l_t is not None:
        lt_key = f"RESID_PRE.L{l_t}"
        axis_lt = axes[COORDINATED_TEMPLATE]["p_c"][lt_key]
        unit = axis_lt / axis_lt.norm()
        entries = {}
        for key in ["EMBED"] + [key for key in ctx.universe if key_layer(key) < l_t]:
            diffs = [ctx.cache.run(pl).vector((key, pl.p_c)) - ctx.cache.run(sg).vector((key, sg.p_c)) for sg, pl in map(frame_prompts, ctx.coordinated)]
            entries[key] = float(torch.stack(diffs).double().mean(dim=0) @ unit)
        total = float(axis_lt.norm())
        encoding = {"layer": l_t, "entries": entries, "axis_norm": total, "embedding_share": entries["EMBED"] / total,
                    "e_share": sum(entries[key] for key in e_keys if key in entries) / total}
    # Exact direct effects for every prompt; pair shares per frame and template.
    effects = {}
    pair_shares = {}
    for frame in ctx.frames:
        sg, pl = frame_prompts(frame)
        de_a = direct_effects(ctx.weights, ctx.cache.run(sg), ctx.nouns, p_t=sg.p_t, n_layers=ctx.n_layers, universe=ctx.universe)
        de_b = direct_effects(ctx.weights, ctx.cache.run(pl), ctx.nouns, p_t=pl.p_t, n_layers=ctx.n_layers, universe=ctx.universe)
        check_direct_effects(de_a, where=sg.key)
        check_direct_effects(de_b, where=pl.key)
        effects[sg.key] = {k: v for k, v in de_a.items() if k != "per_noun"}
        effects[pl.key] = {k: v for k, v in de_b.items() if k != "per_noun"}
        pair_shares[frame.frame_id] = pair_direct_effects(de_a, de_b)
    template_shares = {}
    for template in TEMPLATE_ORDER:
        frames = ctx.frames_of(template)
        names = list(pair_shares[frames[0].frame_id])
        template_shares[template] = {name: _mean([pair_shares[frame.frame_id][name] for frame in frames]) for name in names}
    return {"axis_norms": norms, "cosines": cosines, "site_axes": {label: axis.to_dict() for label, axis in site_axes.items()},
            "number_variables": variables, "contributions_at_L_R": contributions, "encoding_decomposition": encoding,
            "direct_effects": {"runs": effects, "pair_shares_by_frame": pair_shares, "pair_shares_by_template": template_shares},
            "_site_axes": site_axes}


def _site_vector(ctx: DiscoveryContext, prompt: Prompt, label: str, e_keys: Sequence[str], t_keys: Sequence[str], r_keys: Sequence[str], l_r: int) -> torch.Tensor:
    run = ctx.cache.run(prompt)
    if label == "E":
        return summed_vector(run, [(key, prompt.p_c) for key in e_keys])
    if label == "T":
        return summed_vector(run, [(key, prompt.p_t) for key in t_keys])
    if label == "R_in":
        return run.vector((f"RESID_PRE.L{l_r}", prompt.p_t))
    if label == "R_out":
        return summed_vector(run, [(key, prompt.p_t) for key in r_keys])
    raise ValueError(label)


# -- A6 ---------------------------------------------------------------------

def _projection_fraction(ctx: DiscoveryContext, frames: Sequence[Frame], runs_by_frame: Mapping[str, Mapping[str, PromptRun]], axis: SiteAxis, sites_for: Any) -> float:
    """Ratio of means: reproduced projection change over the full counterfactual projection change."""
    reproduced, full = [], []
    for frame in frames:
        sg, pl = frame_prompts(frame)
        n_a, n_b = axis.projection(sites_for(ctx.cache.run(sg), sg)), axis.projection(sites_for(ctx.cache.run(pl), pl))
        patched = runs_by_frame[frame.frame_id]
        n_a_from_b = axis.projection(sites_for(patched["a_from_b"], sg))
        n_b_from_a = axis.projection(sites_for(patched["b_from_a"], pl))
        reproduced.append(0.5 * ((n_a_from_b - n_a) + (n_b - n_b_from_a)))
        full.append(n_b - n_a)
    return _mean(reproduced) / _mean(full)


def a6_chain(ctx: DiscoveryContext, site_axes: Mapping[str, SiteAxis], *, e_keys: Sequence[str], t_keys: Sequence[str], t_candidates: Sequence[str], r_keys: Sequence[str], l_t: int) -> dict[str, Any]:
    """Chain, path, and mediation tests on the coordinated-adjective template.

    ``t_keys`` is the declared T set behind the T site axis; ``t_candidates`` are the ranked heads
    tested one by one (the first is the top head).
    """
    frames = ctx.coordinated
    e_sites = [(key, "p_c") for key in e_keys]
    r_sites = [(key, "p_t") for key in r_keys]
    result: dict[str, Any] = {"e_keys": list(e_keys), "t_keys": list(t_keys), "t_candidates": list(t_candidates), "r_keys": list(r_keys), "l_t": l_t}

    def t_vector(keys: Sequence[str]):
        return lambda run, prompt: summed_vector(run, [(key, prompt.p_t) for key in keys])

    def r_vector(run: PromptRun, prompt: Prompt) -> torch.Tensor:
        return summed_vector(run, [(key, prompt.p_t) for key in r_keys])

    capture_sites = list(dict.fromkeys([(key, "p_t") for key in list(t_candidates) + list(t_keys)] + r_sites))
    # (i) E alone, everything downstream recomputes.
    runs = {}
    rows: list[CaseRow] = []
    for frame in frames:
        frame_rows, patched = counterfactual_rows(ctx.model, ctx.cache, frame, e_sites, capture_role_sites=capture_sites)
        rows.extend(frame_rows)
        runs[frame.frame_id] = patched
    e_alone = {"recovery": stratified_recovery(rows)["overall"], "m_R": _projection_fraction(ctx, frames, runs, site_axes["R_out"], r_vector)}
    e_alone["m_T_by_head"] = {key: _projection_fraction(ctx, frames, runs, site_axis(key, [summed_vector(ctx.cache.run(sg), [(key, sg.p_t)]) for sg, _ in map(frame_prompts, frames)],
                                                                                         [summed_vector(ctx.cache.run(pl), [(key, pl.p_t)]) for _, pl in map(frame_prompts, frames)]), t_vector([key]))
                              for key in t_candidates}
    if "T" in site_axes and t_keys:
        e_alone["m_T"] = _projection_fraction(ctx, frames, runs, site_axes["T"], t_vector(list(t_keys)))
    result["e_alone"] = e_alone
    # (ii) E alone with each candidate T head frozen; and with the top-3 frozen jointly.
    frozen = {}
    freeze_sets = {key: [key] for key in t_candidates}
    freeze_sets["top3"] = list(t_candidates[:3])
    for label, keys in freeze_sets.items():
        runs_f, rows_f = {}, []
        for frame in frames:
            frame_rows, patched = counterfactual_rows(ctx.model, ctx.cache, frame, e_sites, freeze=[(key, "p_t") for key in keys], capture_role_sites=r_sites)
            rows_f.extend(frame_rows)
            runs_f[frame.frame_id] = patched
        frozen[label] = {"recovery": stratified_recovery(rows_f)["overall"], "m_R_given_T_frozen": _projection_fraction(ctx, frames, runs_f, site_axes["R_out"], r_vector)}
    result["e_alone_t_frozen"] = frozen
    # (iii) Residual at the transport layer's input at p_c, alone and with T frozen: blocked fraction.
    resid_site = [(f"RESID_PRE.L{l_t}", "p_c")]
    rows_u: list[CaseRow] = []
    for frame in frames:
        frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, resid_site)
        rows_u.extend(frame_rows)
    unfrozen = _recovery(rows_u)
    blocked = {}
    for label, keys in freeze_sets.items():
        rows_b: list[CaseRow] = []
        for frame in frames:
            frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, resid_site, freeze=[(key, "p_t") for key in keys])
            rows_b.extend(frame_rows)
        r_frozen = _recovery(rows_b)
        blocked[label] = {"recovery_frozen": r_frozen, "blocked_fraction": (unfrozen - r_frozen) / unfrozen if unfrozen else None}
    result["residual_patch"] = {"recovery_unfrozen": unfrozen, "frozen": blocked}
    # (iv) R outputs alone; and joint E∪T as sufficiency only.
    rows_r: list[CaseRow] = []
    for frame in frames:
        frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, r_sites)
        rows_r.extend(frame_rows)
    result["r_alone"] = {"recovery": _recovery(rows_r)}
    singles, cumulative = {}, {}
    for index, key in enumerate(t_candidates):
        rows_s, rows_c = [], []
        for frame in frames:
            frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, [(key, "p_t")])
            rows_s.extend(frame_rows)
            frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, [(k, "p_t") for k in t_candidates[:index + 1]])
            rows_c.extend(frame_rows)
        singles[key] = _recovery(rows_s)
        cumulative[f"top{index + 1}"] = _recovery(rows_c)
    result["t_alone"] = {"singles": singles, "cumulative": cumulative}
    rows_et: list[CaseRow] = []
    for frame in frames:
        frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, e_sites + [(key, "p_t") for key in t_candidates[:1]])
        rows_et.extend(frame_rows)
    result["joint_e_top_t_sufficiency_only"] = {"recovery": _recovery(rows_et)}
    return result


# -- A7 ---------------------------------------------------------------------

def a7_abstractness(ctx: DiscoveryContext, *, e_keys: Sequence[str], t_candidates: Sequence[str], r_keys: Sequence[str]) -> dict[str, Any]:
    """Cross-cue (cue-final) and cross-frame (all templates) resample replacements."""
    result: dict[str, Any] = {"cross_cue": {}, "cross_frame": {}}
    cardinal, quantifier = ctx.frames_of("cardinal"), ctx.frames_of("quantifier")
    for target_frames, source_frames, label in ((cardinal, quantifier, "cardinal<-quantifier"), (quantifier, cardinal, "quantifier<-cardinal")):
        entry = {}
        for flip, name in ((True, "opposite_number"), (False, "same_number")):
            rows: list[CaseRow] = []
            for target, source in zip(target_frames, source_frames):
                frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, target, [(key, "p_t") for key in e_keys], source_frame=source, flip_source=flip)
                rows.extend(frame_rows)
            entry[name] = _recovery(rows)
        result["cross_cue"][label] = entry
    sets = {"E": [(key, "p_c") for key in e_keys], "R": [(key, "p_t") for key in r_keys]}
    for key in t_candidates:
        sets[f"T:{key}"] = [(key, "p_t")]
    for label, role_sites in sets.items():
        per_template = {}
        for template in TEMPLATE_ORDER:
            frames = ctx.frames_of(template)
            entry = {}
            for flip, name in ((True, "opposite_cue"), (False, "same_cue")):
                rows = []
                for target, source in ((frames[0], frames[1]), (frames[1], frames[0])):
                    frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, target, role_sites, source_frame=source, flip_source=flip)
                    rows.extend(frame_rows)
                entry[name] = _recovery(rows)
            per_template[template] = entry
        result["cross_frame"][label] = per_template
    return result


# -- A8 ---------------------------------------------------------------------

def a8_neutralization(ctx: DiscoveryContext, *, e_keys: Sequence[str], t_candidates: Sequence[str], r_keys: Sequence[str]) -> dict[str, Any]:
    """Pair-centered neutralization: contrast loss, direct-effect compensation, conditional co-neutralization."""
    targets: dict[str, list[RoleSite]] = {f"E:{key}": [(key, "p_c")] for key in e_keys}
    targets.update({f"T:{key}": [(key, "p_t")] for key in t_candidates})
    targets.update({f"R:{key}": [(key, "p_t")] for key in r_keys})
    targets["candidate_set"] = [(key, "p_c") for key in e_keys] + [(key, "p_t") for key in t_candidates[:1]] + [(key, "p_t") for key in r_keys]
    capture = [(key, "p_t") for key in ctx.universe] + [("EMBED", "p_t"), (f"RESID_POST.L{ctx.n_layers - 1}", "p_t")]
    result: dict[str, Any] = {}
    for label, role_sites in targets.items():
        per_template = {}
        compensation_all = {}
        for template in TEMPLATE_ORDER:
            frames = ctx.frames_of(template)
            rows: list[CaseRow] = []
            retention = {"retained": 0, "total": 0}
            comp_terms: dict[str, list[float]] = {}
            clean_terms: list[dict[str, float]] = []
            for frame in frames:
                sites = list(dict.fromkeys((key, role) for key, role in role_sites))
                if frame.p_c == frame.p_t:
                    sites = list(dict.fromkeys((key, "p_t") for key, _ in sites))
                runs = neutralized_runs(ctx.model, ctx.cache, frame, sites, capture_role_sites=capture)
                rows.extend(neutralization_rows(ctx.cache, frame, runs))
                kept = sign_retention(ctx.cache, frame, runs)
                retention["retained"] += kept["retained"]
                retention["total"] += kept["total"]
                sg, pl = frame_prompts(frame)
                clean = pair_direct_effects(direct_effects(ctx.weights, ctx.cache.run(sg), ctx.nouns, p_t=sg.p_t, n_layers=ctx.n_layers, universe=ctx.universe),
                                            direct_effects(ctx.weights, ctx.cache.run(pl), ctx.nouns, p_t=pl.p_t, n_layers=ctx.n_layers, universe=ctx.universe))
                de_sg = direct_effects(ctx.weights, runs["sg"], ctx.nouns, p_t=sg.p_t, n_layers=ctx.n_layers, universe=ctx.universe)
                de_pl = direct_effects(ctx.weights, runs["pl"], ctx.nouns, p_t=pl.p_t, n_layers=ctx.n_layers, universe=ctx.universe)
                check_direct_effects(de_sg, where=f"{label}/{sg.key}")
                check_direct_effects(de_pl, where=f"{label}/{pl.key}")
                after = pair_direct_effects(de_sg, de_pl)
                clean_terms.append(clean)
                for name in clean:
                    comp_terms.setdefault(name, []).append(after[name] - clean[name])
            removed_keys = [key for key, _ in role_sites]
            deltas = {name: _mean(values) for name, values in comp_terms.items()}
            clean_removed = _mean([sum(clean_by_frame[key] for key in removed_keys if key in clean_by_frame) for clean_by_frame in clean_terms])
            others = {name: value for name, value in deltas.items() if name not in removed_keys}
            compensation = (sum(others.values()) / clean_removed) if clean_removed != 0.0 else None
            top = sorted(others, key=lambda name: -abs(others[name]))[:3]
            per_template[template] = {"loss": stratified_recovery(rows)["overall"], "sign_retention": retention,
                                      "clean_direct_effect_removed": clean_removed,
                                      "direct_effect_change_removed": sum(deltas[key] for key in removed_keys if key in deltas),
                                      "compensation_ratio": compensation, "top_compensators": {name: others[name] for name in top}}
            compensation_all[template] = top[0] if top else None
        result[label] = {"templates": per_template}
    # Conditional co-neutralization for the single top T head on the coordinated template.
    if t_candidates:
        top_t = t_candidates[0]
        frames = ctx.coordinated
        comp = result[f"T:{top_t}"]["templates"][COORDINATED_TEMPLATE]["top_compensators"]
        if comp:
            partner = next(iter(comp))
            if partner in ctx.universe:
                rows: list[CaseRow] = []
                for frame in frames:
                    runs = neutralized_runs(ctx.model, ctx.cache, frame, [(top_t, "p_t"), (partner, "p_t")])
                    rows.extend(neutralization_rows(ctx.cache, frame, runs))
                result["conditional_co_neutralization"] = {"primary": top_t, "partner": partner,
                                                           "loss_alone": result[f"T:{top_t}"]["templates"][COORDINATED_TEMPLATE]["loss"]["recovery"],
                                                           "loss_joint": stratified_recovery(rows)["overall"]["recovery"]}
    return result


# -- A9 ---------------------------------------------------------------------

def a9_isolation(ctx: DiscoveryContext, keep: Sequence[RoleSite]) -> dict[str, Any]:
    rows: list[CaseRow] = []
    retention = {"retained": 0, "total": 0}
    for frame in ctx.frames:
        kept = list(keep)
        if frame.p_c == frame.p_t:
            kept = list(dict.fromkeys((key, "p_t") for key, _ in kept))
        runs = isolation_runs(ctx.model, ctx.cache, frame, kept)
        sg, pl = frame_prompts(frame)
        c_a, c_b = ctx.cache.c(sg), ctx.cache.c(pl)
        c_a_i, c_b_i = contrasts(runs["sg"].logits, ctx.nouns), contrasts(runs["pl"].logits, ctx.nouns)
        for noun in ctx.nouns:
            if not noun.single_token:
                continue
            key = noun.lexical_key
            rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, c_a[key] - c_b[key], c_a_i[key] - c_b_i[key]))
        kept_signs = sign_retention(ctx.cache, frame, runs)
        retention["retained"] += kept_signs["retained"]
        retention["total"] += kept_signs["total"]
    summary = stratified_recovery(rows)
    return {"faithfulness": summary, "sign_retention": retention, "keep": [list(site) for site in keep]}


# ---------------------------------------------------------------------------
# A10/A11: hypothesis tree, encoding branch, set selection, program parameters, statement.

HYPOTHESIS_ROWS = ("H1", "H2", "H3")
PROGRAM_ELIGIBLE_KEYS = ("EMBED", "L00.MLP")
MAX_COMPONENTS_AT_P_T = 6
MAX_COMPONENTS_AT_P_C = 2
MAX_MECHANISM_VERSIONS = 3
TIER_A_RECOVERY_FLOOR_OVERALL = 0.70
TIER_A_RECOVERY_FLOOR_STRATUM = 0.60
TIER_A_ISOLATION_FLOOR = 0.50
PROGRAM_TEMPLATE_MEAN_TOLERANCE = 1.0


def hypothesis_tree(q1: float, q2: float, q3: float, q4: float, q5: float) -> tuple[str, bool]:
    """The design's decision tree; returns (row, ambiguity flag)."""
    for name, value in (("q1", q1), ("q2", q2), ("q3", q3), ("q4", q4), ("q5", q5)):
        if value is None or not math.isfinite(value):
            raise IncidentError(f"hypothesis quantity {name} is not finite")
    if q3 < 0.50 and q4 >= 0.70:
        return "H3", False
    if q1 >= 0.50 and q2 >= 0.50:
        return "H1", q3 < 0.50
    if q5 >= 0.70:
        return "H2", q3 < 0.50
    return "H2", True


def program_eligible(key: str) -> bool:
    return key in PROGRAM_ELIGIBLE_KEYS


def encoding_branch(decomposition: Mapping[str, Any], *, budget: int = MAX_COMPONENTS_AT_P_C) -> dict[str, Any]:
    """Apply the predeclared encoding rule to the p_c decomposition at the transport layer's input."""
    entries = dict(decomposition["entries"])
    total = float(decomposition["axis_norm"])
    e_keys = ["L00.MLP"]
    e_share = entries.get("L00.MLP", 0.0) / total
    embedding_share = entries["EMBED"] / total
    if e_share >= 0.5:
        return {"branch": "L00.MLP", "e_keys": e_keys, "e_share": e_share, "embedding_share": embedding_share, "flag": None}
    if embedding_share > 0.5:
        return {"branch": "EMBED", "e_keys": ["EMBED"], "e_share": embedding_share, "embedding_share": embedding_share, "flag": None}
    others = sorted((key for key in entries if key not in {"EMBED", "L00.MLP"}), key=lambda key: -entries[key])
    for key in others:
        if len(e_keys) >= budget:
            break
        e_keys.append(key)
        e_share += entries[key] / total
        if e_share >= 0.5:
            break
    flag = None if e_share >= 0.5 else "ENCODING_SHARE_BELOW_HALF"
    return {"branch": "L00.MLP", "e_keys": e_keys, "e_share": e_share, "embedding_share": embedding_share, "flag": flag}


@dataclass(frozen=True)
class MechanismSet:
    """S_M with roles: E at p_c, T heads and R MLPs at p_t (E sits at p_t in cue-final frames)."""

    branch: str
    e_keys: tuple[str, ...]
    t_keys: tuple[str, ...]
    r_keys: tuple[str, ...]

    @property
    def p_t_keys(self) -> tuple[str, ...]:
        return tuple(self.t_keys) + tuple(self.r_keys)

    @property
    def l_r(self) -> int | None:
        return min((key_layer(key) for key in self.r_keys), default=None)

    @property
    def l_t(self) -> int | None:
        return min((key_layer(key) for key in self.t_keys), default=None)

    def role_sites(self, *, include_e: bool = True) -> tuple[RoleSite, ...]:
        sites: list[RoleSite] = []
        if include_e and self.branch != "EMBED":
            sites.extend((key, "p_c") for key in self.e_keys)
        sites.extend((key, "p_t") for key in self.p_t_keys)
        return tuple(sites)

    def p_t_sites_for(self, frame: Frame) -> tuple[Site, ...]:
        keys = list(self.p_t_keys)
        if frame.p_c == frame.p_t and self.branch != "EMBED":
            keys = list(self.e_keys) + keys
        return tuple((key, frame.p_t) for key in dict.fromkeys(keys))

    @property
    def e_program_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.e_keys if program_eligible(key))

    @property
    def contextual_e_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.e_keys if not program_eligible(key))

    def to_dict(self) -> dict[str, Any]:
        return {"branch": self.branch, "e_keys": list(self.e_keys), "t_keys": list(self.t_keys), "r_keys": list(self.r_keys),
                "l_r": self.l_r, "l_t": self.l_t, "e_program_keys": list(self.e_program_keys), "contextual_e_keys": list(self.contextual_e_keys)}


def set_recovery(ctx: DiscoveryContext, mechanism: MechanismSet) -> dict[str, Any]:
    rows: list[CaseRow] = []
    for frame in ctx.frames:
        sites = mechanism.role_sites()
        if frame.p_c == frame.p_t:
            sites = tuple(dict.fromkeys((key, "p_t") for key, _ in sites))
        frame_rows, _ = counterfactual_rows(ctx.model, ctx.cache, frame, sites)
        rows.extend(frame_rows)
    return stratified_recovery(rows)


def recovery_floors_pass(summary: Mapping[str, Any], *, overall: float, stratum: float) -> tuple[bool, list[str]]:
    failures = []
    if summary["overall"]["recovery"] is None or summary["overall"]["recovery"] < overall:
        failures.append("overall")
    for group in ("templates", "rule_classes"):
        for name, entry in summary[group].items():
            if entry["recovery"] is None or entry["recovery"] < stratum:
                failures.append(f"{group}:{name}")
    return not failures, failures


# -- program parameters ------------------------------------------------------


@dataclass(frozen=True)
class ProgramParameters:
    """Everything the weight-only program needs, estimated on development data."""

    e_program_keys: tuple[str, ...]
    e_axis: SiteAxis
    k_t: float
    v_t: Mapping[str, torch.Tensor]  # per template increment vector (g_R · d̂_R)
    rho_frame: Mapping[str, torch.Tensor]
    rho_template: Mapping[str, torch.Tensor]
    t_axis: SiteAxis | None
    r_in_axis: SiteAxis | None = None
    r_out_axis: SiteAxis | None = None

    def site_axes(self) -> dict[str, SiteAxis]:
        axes = {"E": self.e_axis}
        if self.t_axis is not None:
            axes["T"] = self.t_axis
        if self.r_in_axis is not None:
            axes["R_in"] = self.r_in_axis
        if self.r_out_axis is not None:
            axes["R_out"] = self.r_out_axis
        return axes

    def gains(self) -> dict[str, float]:
        return {template: float(vector.norm()) for template, vector in self.v_t.items()}


def e_program_vector(weights: Weights, keys: Sequence[str], token_id: int) -> torch.Tensor:
    parts = []
    for key in keys:
        if key == "L00.MLP":
            parts.append(lexicon_vector(weights, token_id))
        elif key == "EMBED":
            parts.append(weights.W_E[int(token_id)])
        else:
            raise ValueError(f"{key} is not program-eligible")
    if not parts:
        raise ValueError("E_program needs at least one token-local path")
    return torch.stack(parts).sum(dim=0)


def estimate_program_parameters(ctx: DiscoveryContext, mechanism: MechanismSet) -> ProgramParameters:
    keys = mechanism.e_program_keys
    if not keys:
        raise IncidentError("no program-eligible encoding path is declared")
    e_a = [e_program_vector(ctx.weights, keys, frame.cue_ids["sg"]) for frame in ctx.frames]
    e_b = [e_program_vector(ctx.weights, keys, frame.cue_ids["pl"]) for frame in ctx.frames]
    e_axis = site_axis("E_program", e_a, e_b)
    t_axis = None
    k_t = 1.0
    if mechanism.t_keys:
        t_a = [summed_vector(ctx.cache.run(sg), [(key, sg.p_t) for key in mechanism.t_keys]) for sg, _ in map(frame_prompts, ctx.coordinated)]
        t_b = [summed_vector(ctx.cache.run(pl), [(key, pl.p_t) for key in mechanism.t_keys]) for _, pl in map(frame_prompts, ctx.coordinated)]
        t_axis = site_axis("T", t_a, t_b)
        n_c, n_t = [], []
        for frame in ctx.coordinated:
            for prompt in frame_prompts(frame):
                n_c.append(e_axis.n(e_program_vector(ctx.weights, keys, prompt.cue_token_id)))
                n_t.append(t_axis.n(summed_vector(ctx.cache.run(prompt), [(key, prompt.p_t) for key in mechanism.t_keys])))
        k_t = float(sum(c * t for c, t in zip(n_c, n_t)) / sum(c * c for c in n_c))
    v_t, rho_frame, rho_template = {}, {}, {}
    for template in TEMPLATE_ORDER:
        frames = ctx.frames_of(template)
        diffs = []
        for frame in frames:
            sg, pl = frame_prompts(frame)
            sites = mechanism.p_t_sites_for(frame)
            diffs.append(summed_vector(ctx.cache.run(pl), sites) - summed_vector(ctx.cache.run(sg), sites))
            rho_frame[frame.frame_id] = 0.5 * (ctx.cache.final_residual(sg) + ctx.cache.final_residual(pl))
        v_t[template] = 0.5 * torch.stack(diffs).mean(dim=0)
        rho_template[template] = torch.stack([rho_frame[frame.frame_id] for frame in frames]).mean(dim=0)
    r_in_axis = r_out_axis = None
    if mechanism.r_keys and mechanism.l_r is not None:
        r_in_key = f"RESID_PRE.L{mechanism.l_r}"
        r_in_axis = site_axis("R_in", [ctx.cache.run(sg).vector((r_in_key, sg.p_t)) for sg, _ in map(frame_prompts, ctx.frames)],
                              [ctx.cache.run(pl).vector((r_in_key, pl.p_t)) for _, pl in map(frame_prompts, ctx.frames)])
        r_out_axis = site_axis("R_out", [summed_vector(ctx.cache.run(sg), [(key, sg.p_t) for key in mechanism.r_keys]) for sg, _ in map(frame_prompts, ctx.frames)],
                               [summed_vector(ctx.cache.run(pl), [(key, pl.p_t) for key in mechanism.r_keys]) for _, pl in map(frame_prompts, ctx.frames)])
    return ProgramParameters(keys, e_axis, k_t, v_t, rho_frame, rho_template, t_axis, r_in_axis, r_out_axis)


def export_program_parameters(directory: Path, weights: Weights, parameters: ProgramParameters, *, vocab_size: int) -> dict[str, Any]:
    """Write the tensors the program may read, plus an index with digests; return the index."""
    from .provenance import tensor_digest

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "W_E": weights.W_E, "W_U": weights.W_U, "b_U": weights.b_U, "ln_final_w": weights.ln_final_w, "ln_final_b": weights.ln_final_b,
        "ln2_0_w": weights.ln2_0_w, "ln2_0_b": weights.ln2_0_b, "mlp0_W_in": weights.mlp0_W_in, "mlp0_b_in": weights.mlp0_b_in,
        "mlp0_W_out": weights.mlp0_W_out, "mlp0_b_out": weights.mlp0_b_out,
        "e_axis_mu": parameters.e_axis.mu, "e_axis_direction": parameters.e_axis.direction,
    }
    for template, vector in parameters.v_t.items():
        tensors[f"v_t.{template}"] = vector
    for frame_id, vector in parameters.rho_frame.items():
        tensors[f"rho_frame.{frame_id}"] = vector
    for template, vector in parameters.rho_template.items():
        tensors[f"rho_template.{template}"] = vector
    axis_scalars = {}
    for label, axis in parameters.site_axes().items():
        if label == "E":
            continue
        tensors[f"axis_mu.{label}"] = axis.mu
        tensors[f"axis_direction.{label}"] = axis.direction
        axis_scalars[label] = axis.sigma
    digests = {}
    for name, tensor in tensors.items():
        path = directory / f"{name}.pt"
        torch.save(tensor.detach().cpu().contiguous(), path)
        digests[name] = {"file": path.name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}
    index = {
        "eps": weights.eps, "act_fn": weights.act_fn, "vocab_size": vocab_size, "e_program_keys": list(parameters.e_program_keys),
        "e_axis_sigma": parameters.e_axis.sigma, "axis_sigmas": axis_scalars, "k_t": parameters.k_t, "templates": list(TEMPLATE_ORDER),
        "cue_final_templates": list(CUE_FINAL_TEMPLATES), "coordinated_template": COORDINATED_TEMPLATE,
        "frames_by_template": {template: [frame_id for frame_id in parameters.rho_frame if frame_id.startswith(template + "-")] for template in TEMPLATE_ORDER},
        "tensors": digests,
    }
    (directory / "parameters.json").write_text(canonical_json(index) + "\n", encoding="utf-8")
    return index


def load_program(parameters_dir: Path, program_path: Path) -> Any:
    """Import the standalone program module from the experiment directory and bind it to the parameters."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("mechanism_program", str(program_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MechanismProgram.load(Path(parameters_dir))


def program_development_floors(program: Any, ctx: DiscoveryContext) -> dict[str, Any]:
    """Every development pair sign; per-template mean within 1.0 nats of the measured mean d_full."""
    sign_failures = []
    per_template = {}
    for template in TEMPLATE_ORDER:
        measured, predicted = [], []
        for frame in ctx.frames_of(template):
            sg, pl = frame_prompts(frame)
            for noun in ctx.nouns:
                if not noun.single_token:
                    continue
                d_hat = program.predict_pair(template, frame.frame_id, sg.cue_token_id, pl.cue_token_id, noun.sg_ids[0], noun.pl_ids[0])
                d_meas = ctx.cache.c(sg)[noun.lexical_key] - ctx.cache.c(pl)[noun.lexical_key]
                measured.append(d_meas)
                predicted.append(d_hat)
                if d_hat <= 0.0:
                    sign_failures.append(f"{frame.frame_id}/{noun.lexical_key}")
        per_template[template] = {"mean_measured": _mean(measured), "mean_predicted": _mean(predicted),
                                  "gap": abs(_mean(measured) - _mean(predicted)),
                                  "mae": _mean([abs(m - p) for m, p in zip(measured, predicted)])}
    passed = not sign_failures and all(entry["gap"] <= PROGRAM_TEMPLATE_MEAN_TOLERANCE for entry in per_template.values())
    return {"passed": passed, "sign_failures": sign_failures, "templates": per_template}


def render_mechanism_statement(mechanism: MechanismSet, parameters: ProgramParameters, *, hypothesis: str, flagged: bool, capped: bool, encoding: Mapping[str, Any]) -> str:
    gains = parameters.gains()
    e_desc = ", ".join(mechanism.e_keys)
    t_desc = ", ".join(mechanism.t_keys) or "none (no transport declared)"
    r_desc = ", ".join(mechanism.r_keys)
    lines = [
        "VARIABLES",
        f"  n_c(x)  = ( E_program(cue token) − μ_E ) · d̂_E / σ_E          σ_E = {parameters.e_axis.sigma:.4f}; E_program = {', '.join(parameters.e_program_keys)}",
        f"  n_t(x)  = k_T · n_c(x) when p_c ≠ p_t (k_T = {parameters.k_t:.4f}); n_t(x) = n_c(x) otherwise",
        "  c(x)    = log P(singular | x) − log P(plural | x)",
        "",
        "STAGES",
        f"  E  [{e_desc}] at p_c : cue token → n_c   (encoding; branch {mechanism.branch}; share of the transport-input axis {encoding.get('e_share', float('nan')):.3f})",
        f"  T  [{t_desc}] at p_t : n_t := k_T · n_c  (transport; hypothesis {hypothesis}{' — AMBIGUOUS' if flagged else ''})",
        f"  R  [{r_desc}] at p_t : δ(x) = n_t(x) · v_T with g_R = |v_T|: " + ", ".join(f"{template} {gain:.3f}" for template, gain in gains.items()),
        "  D  direct paths: reported under residual accounting",
        "",
        "READOUT",
        "  ĉ(x)      = −u_N · LN( ρ + δ(x) )   exact final LayerNorm, ρ = frozen frame/template context",
        "  d̂_full(N) = ĉ(x_A) − ĉ(x_B)         predicted positive for every pair",
        "",
        f"PROGRAM  {'CAPPED — a contextual encoding component is required; the decompilation axis cannot pass' if capped else 'token-local E_program; decompilation axis live'}",
    ]
    if mechanism.contextual_e_keys:
        lines.append(f"  contextual E components in the circuit account only: {', '.join(mechanism.contextual_e_keys)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Discovery orchestration: A1–A11 in order, writing each step before the next.


def _write_step(state: dict[str, Any], results_path: Path | None, step: str, payload: Mapping[str, Any]) -> None:
    state["discovery"][step] = {key: value for key, value in payload.items() if not key.startswith("_")}
    if results_path is not None:
        write_results_state(results_path, state)


def candidate_roles_from_a2(ctx: DiscoveryContext, a2: Mapping[str, Any]) -> dict[str, Any]:
    """Top-three transport heads, the transport layer, provisional readout MLPs and L_R."""
    heads = [key for key in a2["rankings"]["p_t_coordinated"] if is_head_key(key)]
    t_candidates = heads[:3]
    l_t = key_layer(t_candidates[0])
    mlps_ranked = [key for key in a2["rankings"]["p_t_all_templates"] if is_mlp_key(key) and key != "L00.MLP"]
    r_cand = [key for key in mlps_ranked if key_layer(key) > l_t]
    flag = None
    if not r_cand:
        r_cand = mlps_ranked[:1]
        flag = "NO_MLP_BELOW_TRANSPORT_LAYER"
    r_cand = sorted(r_cand, key=key_layer)
    return {"t_candidates": t_candidates, "l_t": l_t, "r_candidates": r_cand, "l_r": min(key_layer(key) for key in r_cand), "flag": flag}


def evaluate_candidate_set(ctx: DiscoveryContext, mechanism: MechanismSet, *, program_path: Path, parameters_dir: Path) -> dict[str, Any]:
    """Tier A floors for one candidate S_M: recovery strata, isolation, roles, program floors."""
    recovery = set_recovery(ctx, mechanism)
    recovery_ok, recovery_failures = recovery_floors_pass(recovery, overall=TIER_A_RECOVERY_FLOOR_OVERALL, stratum=TIER_A_RECOVERY_FLOOR_STRATUM)
    isolation = a9_isolation(ctx, mechanism.role_sites())
    faithfulness = isolation["faithfulness"]["overall"]["recovery"]
    isolation_ok = faithfulness is not None and faithfulness >= TIER_A_ISOLATION_FLOOR
    roles_ok = bool(mechanism.e_keys) and bool(mechanism.r_keys) and bool(mechanism.t_keys)
    result = {"mechanism": mechanism.to_dict(), "recovery": recovery, "recovery_floors": {"passed": recovery_ok, "failures": recovery_failures},
              "isolation": isolation, "isolation_floor": {"passed": isolation_ok, "faithfulness": faithfulness},
              "roles_floor": {"passed": roles_ok}}
    program_result: dict[str, Any] = {"passed": False, "error": None}
    capped = False
    if roles_ok:
        try:
            parameters = estimate_program_parameters(ctx, mechanism)
            export_program_parameters(parameters_dir, ctx.weights, parameters, vocab_size=int(ctx.weights.W_E.shape[0]))
            program = load_program(parameters_dir, program_path)
            program_result = program_development_floors(program, ctx)
            program_result["k_t"] = parameters.k_t
            program_result["gains"] = parameters.gains()
            program_result["lexicon_sha256"] = program.lexicon_digest()
            result["_parameters"] = parameters
            result["_program"] = program
        except (IncidentError, ValueError) as error:
            program_result = {"passed": False, "error": str(error)}
        capped = bool(mechanism.contextual_e_keys) and not program_result["passed"]
    result["program_floors"] = program_result
    result["program_capped"] = capped
    circuit_ok = recovery_ok and isolation_ok and roles_ok
    result["passed"] = circuit_ok and (program_result["passed"] or capped)
    result["circuit_passed"] = circuit_ok
    return result


def select_mechanism_set(ctx: DiscoveryContext, a2: Mapping[str, Any], encoding: Mapping[str, Any], *, program_path: Path, parameters_dir: Path) -> dict[str, Any]:
    """Smallest cumulative top-k set at p_t (plus E at p_c) that passes the Tier A floors."""
    ranking = [key for key in a2["rankings"]["p_t_all_templates"] if key != "L00.MLP"]
    attempts = []
    chosen = None
    for k in range(0, MAX_COMPONENTS_AT_P_T + 1):
        top = ranking[:k]
        mechanism = MechanismSet(encoding["branch"], tuple(encoding["e_keys"]), tuple(key for key in top if is_head_key(key)), tuple(key for key in top if is_mlp_key(key)))
        evaluation = evaluate_candidate_set(ctx, mechanism, program_path=program_path, parameters_dir=parameters_dir)
        attempts.append({"k": k, **{key: value for key, value in evaluation.items() if not key.startswith("_")}})
        if evaluation["passed"]:
            chosen = {"k": k, "mechanism": mechanism, "evaluation": evaluation}
            break
    return {"attempts": attempts, "chosen": chosen, "ranking": ranking}


def run_discovery(ctx: DiscoveryContext, *, state: dict[str, Any], results_path: Path | None, program_path: Path, parameters_dir: Path,
                  screening_results_path: Path | None = None, log: Any = None) -> dict[str, Any]:
    """Execute A1–A11 and record mechanism version M1 (or NO_COMPACT_MECHANISM)."""
    say = log or (lambda message: None)
    if ctx.manifest is not None:
        say("A1 baseline replication")
        _write_step(state, results_path, "a1", a1_baseline(ctx, screening_results_path))
    say("A2 position map")
    a2 = a2_position_map(ctx)
    _write_step(state, results_path, "a2", a2)
    roles = candidate_roles_from_a2(ctx, a2)
    _write_step(state, results_path, "candidate_roles", roles)
    say("A3 layer profile")
    _write_step(state, results_path, "a3", a3_layer_profile(ctx))
    say("A4 attention")
    _write_step(state, results_path, "a4", a4_attention(ctx))
    say("A5 axes and direct effects (provisional roles)")
    a5 = a5_axes_and_direct_effects(ctx, e_keys=["L00.MLP"], t_keys=roles["t_candidates"][:1], r_keys=roles["r_candidates"], l_r=roles["l_r"], l_t=roles["l_t"])
    _write_step(state, results_path, "a5_provisional", a5)
    encoding = encoding_branch(a5["encoding_decomposition"])
    _write_step(state, results_path, "encoding_branch", encoding)
    e_keys = list(encoding["e_keys"])
    if encoding["branch"] == "EMBED":
        e_keys_for_chain = ["EMBED"]
    else:
        e_keys_for_chain = e_keys
    say("A6 chain, path, and mediation")
    a6 = a6_chain(ctx, a5["_site_axes"], e_keys=e_keys_for_chain, t_keys=roles["t_candidates"][:1], t_candidates=roles["t_candidates"], r_keys=roles["r_candidates"], l_t=roles["l_t"])
    _write_step(state, results_path, "a6", a6)
    top = roles["t_candidates"][0]
    quantities = {"q1": a6["t_alone"]["singles"][top], "q2": a6["residual_patch"]["frozen"][top]["blocked_fraction"],
                  "q3": a6["e_alone"]["m_R"], "q4": a6["r_alone"]["recovery"], "q5": a6["t_alone"]["cumulative"]["top3"]}
    row, flagged = hypothesis_tree(**quantities)
    _write_step(state, results_path, "hypothesis", {"quantities": quantities, "row": row, "flagged": flagged, "top_head": top})
    say("A7 abstractness")
    _write_step(state, results_path, "a7", a7_abstractness(ctx, e_keys=e_keys_for_chain, t_candidates=roles["t_candidates"], r_keys=roles["r_candidates"]))
    say("A8 neutralization and self-repair")
    _write_step(state, results_path, "a8", a8_neutralization(ctx, e_keys=e_keys_for_chain, t_candidates=roles["t_candidates"], r_keys=roles["r_candidates"]))
    say("A10 mechanism set selection")
    selection = select_mechanism_set(ctx, a2, encoding, program_path=program_path, parameters_dir=parameters_dir)
    _write_step(state, results_path, "a10_selection", {"attempts": selection["attempts"], "ranking": selection["ranking"], "chosen_k": selection["chosen"]["k"] if selection["chosen"] else None})
    if selection["chosen"] is None:
        version = {"version": "M1", "status": "rejected", "outcome": "NO_COMPACT_MECHANISM", "hypothesis": row, "hypothesis_flagged": flagged}
        state["mechanism_versions"].append(version)
        if results_path is not None:
            write_results_state(results_path, state)
        return version
    mechanism: MechanismSet = selection["chosen"]["mechanism"]
    evaluation = selection["chosen"]["evaluation"]
    say("A5 axes and direct effects (final roles)")
    a5_final = a5_axes_and_direct_effects(ctx, e_keys=list(mechanism.e_keys) if mechanism.branch != "EMBED" else ["EMBED"], t_keys=list(mechanism.t_keys),
                                          r_keys=list(mechanism.r_keys), l_r=mechanism.l_r, l_t=mechanism.l_t if mechanism.l_t is not None else roles["l_t"])
    _write_step(state, results_path, "a5_final", a5_final)
    parameters: ProgramParameters = evaluation["_parameters"]
    index = export_program_parameters(parameters_dir, ctx.weights, parameters, vocab_size=int(ctx.weights.W_E.shape[0]))
    program = load_program(parameters_dir, program_path)
    # The runner's lexicon check: captured E_program output equals the weight-only vector for the four cue tokens.
    lexicon_check = {}
    for frame in ctx.frames:
        for prompt in frame_prompts(frame):
            captured = summed_vector(ctx.cache.run(prompt), [(key, prompt.p_c) for key in mechanism.e_program_keys])
            gap = float((program.e_program_vector(prompt.cue_token_id).float() - captured).abs().max())
            lexicon_check[str(prompt.cue_token_id)] = max(gap, lexicon_check.get(str(prompt.cue_token_id), 0.0))
            if gap > 1e-5:
                raise IncidentError(f"E_program lexicon differs from the captured activation for token {prompt.cue_token_id} by {gap:.2e}")
    statement = render_mechanism_statement(mechanism, parameters, hypothesis=row, flagged=flagged, capped=evaluation["program_capped"], encoding=encoding)
    version = {
        "version": "M1", "status": "candidate", "mechanism": mechanism.to_dict(), "hypothesis": row, "hypothesis_flagged": flagged,
        "hypothesis_quantities": quantities, "encoding_branch": {key: value for key, value in encoding.items()},
        "program_capped": evaluation["program_capped"], "program_floors": evaluation["program_floors"],
        "recovery": evaluation["recovery"], "isolation": evaluation["isolation"]["faithfulness"], "sign_retention": evaluation["isolation"]["sign_retention"],
        "parameters_index_sha256": sha256_text(canonical_json(index)), "lexicon_sha256": program.lexicon_digest(), "lexicon_check_max_gap": lexicon_check,
        "statement": statement, "k": selection["chosen"]["k"],
    }
    state["mechanism_versions"].append(version)
    if results_path is not None:
        write_results_state(results_path, state)
    return version


# ---------------------------------------------------------------------------
# Prediction families: one evaluator shared by calibrate (holdout) and confirm
# (reserve + extension). Floors are the design's, restated as rates with
# exact-count evaluation (ceil(rate · n)); bands come from Tier B.

FLOOR_RATES = {
    "B1_accuracy": 103 / 120, "B1_flip": 96 / 120, "B2_positive": 114 / 120, "P3_retention": 84 / 120,
    "X1_positive": 108 / 120, "X1_variables": 11 / 12, "X3_sign_word": 5 / 6, "X3_sign_overall": 0.90, "X3_ambiguous": 5 / 6, "S3": 0.80,
}
FLOORS = {
    "P1_overall": 0.70, "P1_stratum": 0.60, "P2_random_floor": 0.05, "P2_multiplier": 2.0, "P3_overall": 0.50, "P3_template": 0.40,
    "P4": 0.50, "P5a_H1": 0.50, "P5a_H2": 0.70, "P5b_H1": 0.50, "P6_opposite": 0.50, "P6_same": 0.25, "P7_same": 0.25, "P7_opposite_factor": 0.5,
    "P8_loss": 0.30, "P9": 0.50, "P9_frozen_factor": 0.5, "S1": 0.70, "X3_spearman": 0.70, "X3_confident": 0.5, "X3_ambiguous_factor": 0.5,
    "X4_spearman": 0.70, "X4_mae_factor": 0.25,
}
BAND_MIN_WIDTH = 0.10
BAND_SE_MULTIPLIER = 1.5
BAND_D_FULL_NATS = 0.5
BOOTSTRAP_RESAMPLES = 1000
TAU_MIN_NATS = 0.5
TAU_RMSE_MULTIPLIER = 3.0
X_BAND_WORDS = 9
X_BAND_FRAMES = 5


def exact_count_floor(rate: float, n: int) -> int:
    return int(math.ceil(rate * n - 1e-9))


def primary_orientation_for(noun: Noun) -> str:
    """The screen's primary condition: singular prompt for simple-suffix nouns, plural prompt otherwise."""
    return "A" if noun.rule_class == "simple-suffix" else "B"


@dataclass
class EvalContext:
    """What every family needs: model, mechanism, program, frames, nouns, clean cache, locked axes."""

    model: Any
    weights: Weights
    mechanism: MechanismSet
    program: Any
    frames: tuple[Frame, ...]
    nouns: tuple[Noun, ...]
    cache: PromptCache
    site_axes: Mapping[str, SiteAxis]
    hypothesis: str

    @property
    def discovery_like(self) -> DiscoveryContext:
        return DiscoveryContext(self.model, self.weights, None, self.frames, self.nouns, self.cache)

    def frames_of(self, template_id: str) -> tuple[Frame, ...]:
        return tuple(frame for frame in self.frames if frame.template_id == template_id)

    @property
    def coordinated(self) -> tuple[Frame, ...]:
        return self.frames_of(COORDINATED_TEMPLATE)


def _rows_to_json(rows: Sequence[CaseRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def _behavior_family(ev: EvalContext) -> dict[str, Any]:
    """B1/B2 (and X1 on new frames): primary accuracy, flips, positive pairs, per-template d_full, variables."""
    accuracy = flips = positive = total = 0
    d_full_rows: list[CaseRow] = []
    variable_hits = {"n_c": 0, "n_t": 0, "prompts": 0}
    wrong_conditions = []
    for frame in ev.frames:
        sg, pl = frame_prompts(frame)
        c_a, c_b = ev.cache.c(sg), ev.cache.c(pl)
        for prompt in (sg, pl):
            run = ev.cache.run(prompt)
            expected = -1.0 if prompt.cue_label == "sg" else 1.0
            n_c = ev.site_axes["E"].n(summed_vector(run, [(key, prompt.p_c) for key in ev.mechanism.e_program_keys]))
            if frame.template_id == COORDINATED_TEMPLATE and "T" in ev.site_axes and ev.mechanism.t_keys:
                n_t = ev.site_axes["T"].n(summed_vector(run, [(key, prompt.p_t) for key in ev.mechanism.t_keys]))
            else:
                n_t = n_c
            variable_hits["prompts"] += 1
            variable_hits["n_c"] += int(math.copysign(1.0, n_c) == expected)
            variable_hits["n_t"] += int(math.copysign(1.0, n_t) == expected)
            for noun in ev.nouns:
                if not noun.single_token:
                    continue
                contrast = (c_a if prompt.cue_label == "sg" else c_b)[noun.lexical_key]
                correct = contrast > 0 if prompt.cue_label == "sg" else contrast < 0
                if not correct:
                    wrong_conditions.append({"frame_id": frame.frame_id, "noun": noun.lexical_key, "cue": prompt.cue_label,
                                             "n_t_sign_matches_cue": math.copysign(1.0, n_t) == expected})
        for noun in ev.nouns:
            if not noun.single_token:
                continue
            key = noun.lexical_key
            total += 1
            primary = c_a[key] if primary_orientation_for(noun) == "A" else c_b[key]
            accuracy += int(primary > 0 if primary_orientation_for(noun) == "A" else primary < 0)
            flips += int(c_a[key] > 0 and c_b[key] < 0)
            positive += int(c_a[key] - c_b[key] > 0)
            d_full_rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, c_a[key] - c_b[key], 0.0))
    template_means = {template: _mean([row.d_full for row in d_full_rows if row.template_id == template]) for template in TEMPLATE_ORDER if any(row.template_id == template for row in d_full_rows)}
    s3 = {"n_wrong": len(wrong_conditions), "n_t_matches": sum(1 for entry in wrong_conditions if entry["n_t_sign_matches_cue"])}
    return {"cases": total, "primary_correct": accuracy, "flips": flips, "positive_pairs": positive, "template_mean_d_full": template_means,
            "variables": variable_hits, "s3": s3, "rows": _rows_to_json(d_full_rows), "wrong_conditions": wrong_conditions}


def _rows_over(ev: EvalContext, frames: Sequence[Frame], role_sites: Sequence[RoleSite], **options: Any) -> tuple[CaseRow, ...]:
    rows: list[CaseRow] = []
    for frame in frames:
        sites = list(role_sites)
        if frame.p_c == frame.p_t:
            sites = list(dict.fromkeys((key, "p_t") for key, _ in sites))
        frame_rows, _ = counterfactual_rows(ev.model, ev.cache, frame, sites, **options)
        rows.extend(frame_rows)
    return tuple(rows)


def _p1(ev: EvalContext) -> dict[str, Any]:
    rows = _rows_over(ev, ev.frames, ev.mechanism.role_sites())
    return {"summary": stratified_recovery(rows), "rows": _rows_to_json(rows)}


def _p2(ev: EvalContext, universe: Sequence[str]) -> dict[str, Any]:
    n_c = len(ev.mechanism.e_keys) if ev.mechanism.branch != "EMBED" else 0
    n_t = len(ev.mechanism.p_t_keys)
    import random

    generator = random.Random(CONTROL_SEED)
    ordered = tuple(universe)
    recoveries, sets = [], []
    for _ in range(RANDOM_SET_COUNT):
        at_c = tuple(sorted(generator.sample(ordered, n_c), key=ordered.index)) if n_c else ()
        at_t = tuple(sorted(generator.sample(ordered, n_t), key=ordered.index)) if n_t else ()
        role_sites = [(key, "p_c") for key in at_c] + [(key, "p_t") for key in at_t]
        rows = _rows_over(ev, ev.frames, role_sites)
        recoveries.append(_recovery(rows))
        sets.append({"p_c": list(at_c), "p_t": list(at_t)})
    finite = sorted(value for value in recoveries if value is not None)
    median = finite[len(finite) // 2] if len(finite) % 2 else 0.5 * (finite[len(finite) // 2 - 1] + finite[len(finite) // 2])
    return {"median": median, "reference": max(median, FLOORS["P2_random_floor"]), "recoveries": recoveries, "sets": sets, "size": {"p_c": n_c, "p_t": n_t}}


def _p3(ev: EvalContext, frames: Sequence[Frame]) -> dict[str, Any]:
    rows: list[CaseRow] = []
    retention = {"retained": 0, "total": 0}
    for frame in frames:
        kept = list(ev.mechanism.role_sites())
        if frame.p_c == frame.p_t:
            kept = list(dict.fromkeys((key, "p_t") for key, _ in kept))
        runs = isolation_runs(ev.model, ev.cache, frame, kept)
        sg, pl = frame_prompts(frame)
        c_a, c_b = ev.cache.c(sg), ev.cache.c(pl)
        c_a_i, c_b_i = contrasts(runs["sg"].logits, ev.nouns), contrasts(runs["pl"].logits, ev.nouns)
        for noun in ev.nouns:
            if noun.single_token:
                key = noun.lexical_key
                rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, c_a[key] - c_b[key], c_a_i[key] - c_b_i[key]))
        kept_signs = sign_retention(ev.cache, frame, runs)
        retention["retained"] += kept_signs["retained"]
        retention["total"] += kept_signs["total"]
    return {"summary": stratified_recovery(rows), "sign_retention": retention, "rows": _rows_to_json(rows)}


def _p4(ev: EvalContext, frames: Sequence[Frame], universe: Sequence[str]) -> dict[str, Any]:
    coordinated = [frame for frame in frames if frame.template_id == COORDINATED_TEMPLATE]
    if ev.mechanism.branch == "EMBED":
        rows: list[CaseRow] = []
        for frame in coordinated:
            runs = neutralized_runs(ev.model, ev.cache, frame, [(key, "p_c") for key in universe])
            sg, pl = frame_prompts(frame)
            c_a, c_b = ev.cache.c(sg), ev.cache.c(pl)
            c_a_n, c_b_n = contrasts(runs["sg"].logits, ev.nouns), contrasts(runs["pl"].logits, ev.nouns)
            for noun in ev.nouns:
                if noun.single_token:
                    key = noun.lexical_key
                    rows.append(CaseRow(frame.frame_id, frame.template_id, key, noun.rule_class, c_a[key] - c_b[key], c_a_n[key] - c_b_n[key]))
        return {"variant": "P4prime", "value": _recovery(rows), "rows": _rows_to_json(rows)}
    rows = _rows_over(ev, coordinated, [(key, "p_c") for key in ev.mechanism.e_keys])
    return {"variant": "P4", "value": _recovery(rows), "rows": _rows_to_json(rows)}


def _p5(ev: EvalContext, frames: Sequence[Frame]) -> dict[str, Any]:
    coordinated = [frame for frame in frames if frame.template_id == COORDINATED_TEMPLATE]
    t_keys = list(ev.mechanism.t_keys)
    if not t_keys or ev.mechanism.l_t is None:
        return {"applicable": False}
    single = _rows_over(ev, coordinated, [(t_keys[0], "p_t")])
    declared = _rows_over(ev, coordinated, [(key, "p_t") for key in t_keys])
    resid = [(f"RESID_PRE.L{ev.mechanism.l_t}", "p_c")]
    unfrozen_rows = _rows_over(ev, coordinated, resid)
    frozen_rows = _rows_over(ev, coordinated, resid, freeze=[(key, "p_t") for key in t_keys])
    unfrozen, frozen = _recovery(unfrozen_rows), _recovery(frozen_rows)
    blocked = (unfrozen - frozen) / unfrozen if unfrozen else None
    return {"applicable": True, "a_single": _recovery(single), "a_declared": _recovery(declared), "b_unfrozen": unfrozen, "b_frozen": frozen,
            "b_blocked_fraction": blocked, "rows": {"single": _rows_to_json(single), "declared": _rows_to_json(declared),
                                                    "unfrozen": _rows_to_json(unfrozen_rows), "frozen": _rows_to_json(frozen_rows)}}


def _p6(ev: EvalContext) -> dict[str, Any]:
    cardinal, quantifier = ev.frames_of("cardinal"), ev.frames_of("quantifier")
    if len(cardinal) != len(quantifier) or not cardinal:
        return {"applicable": False}
    e_sites = [(key, "p_t") for key in ev.mechanism.e_keys] if ev.mechanism.branch != "EMBED" else [("EMBED", "p_t")]
    result = {"applicable": True}
    for flip, name in ((True, "opposite"), (False, "same")):
        rows: list[CaseRow] = []
        for targets, sources in ((cardinal, quantifier), (quantifier, cardinal)):
            for target, source in zip(targets, sources):
                frame_rows, _ = counterfactual_rows(ev.model, ev.cache, target, e_sites, source_frame=source, flip_source=flip)
                rows.extend(frame_rows)
        result[name] = _recovery(rows)
        result[f"rows_{name}"] = _rows_to_json(rows)
    return result


def _p7(ev: EvalContext, frames: Sequence[Frame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flip, name in ((True, "opposite"), (False, "same")):
        rows: list[CaseRow] = []
        for template in TEMPLATE_ORDER:
            pair = [frame for frame in frames if frame.template_id == template]
            if len(pair) != 2:
                continue
            for target, source in ((pair[0], pair[1]), (pair[1], pair[0])):
                sites = list(ev.mechanism.role_sites())
                if target.p_c == target.p_t:
                    sites = list(dict.fromkeys((key, "p_t") for key, _ in sites))
                frame_rows, _ = counterfactual_rows(ev.model, ev.cache, target, sites, source_frame=source, flip_source=flip)
                rows.extend(frame_rows)
        result[name] = _recovery(rows) if rows else None
        result[f"rows_{name}"] = _rows_to_json(rows)
    return result


def _p8(ev: EvalContext, frames: Sequence[Frame], universe: Sequence[str]) -> dict[str, Any]:
    """Neutralize T (coordinated) or E (cue-final): loss and compensation via direct effects."""
    n_layers = int(ev.model.cfg.n_layers)
    capture = [(key, "p_t") for key in universe] + [("EMBED", "p_t"), (f"RESID_POST.L{n_layers - 1}", "p_t")]
    rows: list[CaseRow] = []
    compensation_terms: list[float] = []
    for frame in frames:
        if frame.template_id == COORDINATED_TEMPLATE:
            if not ev.mechanism.t_keys:
                continue
            sites = [(key, "p_t") for key in ev.mechanism.t_keys]
        else:
            if ev.mechanism.branch == "EMBED":
                continue
            sites = [(key, "p_t") for key in ev.mechanism.e_keys]
        runs = neutralized_runs(ev.model, ev.cache, frame, sites, capture_role_sites=capture)
        rows.extend(neutralization_rows(ev.cache, frame, runs))
        sg, pl = frame_prompts(frame)
        clean = pair_direct_effects(direct_effects(ev.weights, ev.cache.run(sg), ev.nouns, p_t=sg.p_t, n_layers=n_layers, universe=universe),
                                    direct_effects(ev.weights, ev.cache.run(pl), ev.nouns, p_t=pl.p_t, n_layers=n_layers, universe=universe))
        after = pair_direct_effects(direct_effects(ev.weights, runs["sg"], ev.nouns, p_t=sg.p_t, n_layers=n_layers, universe=universe),
                                    direct_effects(ev.weights, runs["pl"], ev.nouns, p_t=pl.p_t, n_layers=n_layers, universe=universe))
        removed = [key for key, _ in sites]
        clean_removed = sum(clean[key] for key in removed if key in clean)
        others = sum(after[name] - clean[name] for name in clean if name not in removed)
        if clean_removed != 0.0:
            compensation_terms.append(others / clean_removed)
    return {"loss": _recovery(rows) if rows else None, "compensation_ratio": _mean(compensation_terms) if compensation_terms else None, "rows": _rows_to_json(rows)}


def _p9(ev: EvalContext, frames: Sequence[Frame]) -> dict[str, Any]:
    coordinated = [frame for frame in frames if frame.template_id == COORDINATED_TEMPLATE]
    t_keys, r_keys = list(ev.mechanism.t_keys), list(ev.mechanism.r_keys)
    if not t_keys or not r_keys or "T" not in ev.site_axes or "R_out" not in ev.site_axes:
        return {"applicable": False}
    e_sites = [(key, "p_c") for key in ev.mechanism.e_keys] if ev.mechanism.branch != "EMBED" else [("EMBED", "p_c")]
    capture = [(key, "p_t") for key in t_keys + r_keys]
    ctx = ev.discovery_like

    def t_vec(run: PromptRun, prompt: Prompt) -> torch.Tensor:
        return summed_vector(run, [(key, prompt.p_t) for key in t_keys])

    def r_vec(run: PromptRun, prompt: Prompt) -> torch.Tensor:
        return summed_vector(run, [(key, prompt.p_t) for key in r_keys])

    runs = {}
    for frame in coordinated:
        _, patched = counterfactual_rows(ev.model, ev.cache, frame, e_sites, capture_role_sites=capture)
        runs[frame.frame_id] = patched
    m_t = _projection_fraction(ctx, coordinated, runs, ev.site_axes["T"], t_vec)
    m_r = _projection_fraction(ctx, coordinated, runs, ev.site_axes["R_out"], r_vec)
    frozen = {}
    for frame in coordinated:
        _, patched = counterfactual_rows(ev.model, ev.cache, frame, e_sites, freeze=[(key, "p_t") for key in t_keys], capture_role_sites=[(key, "p_t") for key in r_keys])
        frozen[frame.frame_id] = patched
    m_r_frozen = _projection_fraction(ctx, coordinated, frozen, ev.site_axes["R_out"], r_vec)
    return {"applicable": True, "m_T": m_t, "m_R": m_r, "m_R_given_T_frozen": m_r_frozen}


def _s1(ev: EvalContext) -> dict[str, Any]:
    if ev.mechanism.l_r is None:
        return {"applicable": False}
    key = f"RESID_PRE.L{ev.mechanism.l_r}"
    axes = {}
    for template in TEMPLATE_ORDER:
        frames = ev.frames_of(template)
        if not frames:
            continue
        diffs = [ev.cache.run(pl).vector((key, pl.p_t)) - ev.cache.run(sg).vector((key, sg.p_t)) for sg, pl in map(frame_prompts, frames)]
        axes[template] = torch.stack(diffs).double().mean(dim=0)
    pairs = {}
    names = list(axes)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairs[f"{left}|{right}"] = cosine(axes[left], axes[right])
    return {"applicable": bool(pairs), "cosines": pairs, "minimum": min(pairs.values()) if pairs else None}


def _program_residual(ev: EvalContext) -> dict[str, Any]:
    """Per-condition residual c(x) − ĉ(x) on the frames (manifest frames use ρ_frame; new frames ρ_template)."""
    residuals = []
    for frame in ev.frames:
        sg, pl = frame_prompts(frame)
        frame_id = frame.frame_id if frame.origin == "manifest" else None
        for prompt in (sg, pl):
            measured = ev.cache.c(prompt)
            for noun in ev.nouns:
                if noun.single_token:
                    predicted = ev.program.predict_contrast(frame.template_id, frame_id, prompt.cue_token_id, noun.sg_ids[0], noun.pl_ids[0])
                    residuals.append({"frame_id": frame.frame_id, "cue": prompt.cue_label, "noun": noun.lexical_key,
                                      "measured": measured[noun.lexical_key], "predicted": predicted, "residual": measured[noun.lexical_key] - predicted})
    values = [entry["residual"] for entry in residuals]
    rmse = math.sqrt(_mean([value * value for value in values])) if values else None
    return {"conditions": len(residuals), "rmse": rmse, "mae": _mean([abs(value) for value in values]) if values else None, "entries": residuals}


def evaluate_circuit_families(ev: EvalContext, universe: Sequence[str]) -> dict[str, Any]:
    """Every circuit-axis family on ``ev.frames`` with ``ev.nouns``."""
    return {
        "behavior": _behavior_family(ev),
        "P1": _p1(ev), "P2": _p2(ev, universe), "P3": _p3(ev, ev.frames), "P4": _p4(ev, ev.frames, universe), "P5": _p5(ev, ev.frames),
        "P6": _p6(ev), "P7": _p7(ev, ev.frames), "P8": _p8(ev, ev.frames, universe), "P9": _p9(ev, ev.frames), "S1": _s1(ev),
        "program_residual": _program_residual(ev),
    }


def evaluate_new_frame_families(ev_new: EvalContext, universe: Sequence[str]) -> dict[str, Any]:
    """X1 and X2 on the extension frames (reserve nouns)."""
    return {
        "behavior": _behavior_family(ev_new),
        "P1": _p1(ev_new), "P3": _p3(ev_new, ev_new.frames), "P4": _p4(ev_new, ev_new.frames, universe), "P5": _p5(ev_new, ev_new.frames),
        "P8": _p8(ev_new, ev_new.frames, universe), "P9": _p9(ev_new, ev_new.frames), "program_residual": _program_residual(ev_new),
    }


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            rank = (index + end) / 2 + 1
            for position in range(index, end + 1):
                result[order[position]] = rank
            index = end + 1
        return result

    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("spearman needs two equal-length sequences with at least two entries")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else 0.0


def evaluate_cue_word_families(ev: EvalContext, extension: Extension, *, locked_full_shift: Mapping[str, float]) -> dict[str, Any]:
    """X3 (behavior) and X4 (E-patch) on the twelve cue words in the six original frames, reserve nouns."""
    frames_by_id = {frame.frame_id: frame for frame in ev.frames}
    words = list(extension.cue_words)
    per_word: dict[str, dict[str, Any]] = {word: {"token_id": token, "n_c": ev.program.n_c(token), "frames": {}} for word, token in words}
    e_program_sites = list(ev.mechanism.e_program_keys)
    for prompt in extension.cue_word_prompts:
        frame = frames_by_id[prompt.frame.frame_id]
        reference = Prompt(frame, extension.reference_cue_ids[frame.template_id], "sg")
        run_word = ev.cache.run(prompt)
        run_ref = ev.cache.run(reference)
        measured_word, measured_ref = ev.cache.c(prompt), ev.cache.c(reference)
        shifts, predicted_shifts, epatch_measured, epatch_predicted = [], [], [], []
        # Lexicon check: the captured E_program output equals the weight-only vector.
        captured = summed_vector(run_word, [(key, prompt.p_c) for key in e_program_sites])
        gap = float((ev.program.e_program_vector(prompt.cue_token_id).float() - captured).abs().max())
        if gap > 1e-5:
            raise IncidentError(f"E_program lexicon differs from the captured activation for {prompt.cue_label} by {gap:.2e}")
        replacements = {(key, prompt.p_c): run_word.slice((key, prompt.p_c)) for key in e_program_sites}
        sources = {site: ReplacementSource.RESAMPLE for site in replacements}
        patched = run_patched(ev.model, reference, replacements, sources)
        c_patched = contrasts(patched.logits, ev.nouns)
        for noun in ev.nouns:
            if not noun.single_token:
                continue
            key = noun.lexical_key
            shifts.append(measured_word[key] - measured_ref[key])
            predicted_shifts.append(ev.program.predict_shift(frame.template_id, frame.frame_id, prompt.cue_token_id, reference.cue_token_id, noun.sg_ids[0], noun.pl_ids[0]))
            epatch_measured.append(c_patched[key] - measured_ref[key])
            epatch_predicted.append(ev.program.predict_epatch_shift(frame.template_id, frame.frame_id, prompt.cue_token_id, reference.cue_token_id, noun.sg_ids[0], noun.pl_ids[0]))
        per_word[prompt.cue_label]["frames"][frame.frame_id] = {
            "template_id": frame.template_id, "mean_shift": _mean(shifts), "mean_predicted_shift": _mean(predicted_shifts),
            "epatch_mean_shift": _mean(epatch_measured), "epatch_mean_predicted": _mean(epatch_predicted), "lexicon_gap": gap,
        }
    # X3 aggregates.
    word_means = {word: _mean([entry["mean_shift"] for entry in per_word[word]["frames"].values()]) for word, _ in words}
    n_c_values = [per_word[word]["n_c"] for word, _ in words]
    # A positive n_c means "plural", which lowers c, so the predicted relationship is negative: correlate n_c with −shift.
    x3_spearman = spearman(n_c_values, [-word_means[word] for word, _ in words])
    confident = [word for word, _ in words if abs(per_word[word]["n_c"]) >= FLOORS["X3_confident"]]
    ambiguous = [word for word, _ in words if abs(per_word[word]["n_c"]) < FLOORS["X3_confident"]]
    sign_by_word = {}
    for word in confident:
        expected = -math.copysign(1.0, per_word[word]["n_c"])  # plural lexicon → negative shift of c
        frames = per_word[word]["frames"]
        sign_by_word[word] = {"agree": sum(1 for entry in frames.values() if math.copysign(1.0, entry["mean_shift"]) == expected), "total": len(frames)}
    ambiguous_by_word = {}
    for word in ambiguous:
        frames = per_word[word]["frames"]
        ambiguous_by_word[word] = {"small": sum(1 for entry in frames.values() if abs(entry["mean_shift"]) < FLOORS["X3_ambiguous_factor"] * locked_full_shift[entry["template_id"]]), "total": len(frames)}
    # X4 aggregates.
    epatch_word_measured = {word: _mean([entry["epatch_mean_shift"] for entry in per_word[word]["frames"].values()]) for word, _ in words}
    epatch_word_predicted = {word: _mean([entry["epatch_mean_predicted"] for entry in per_word[word]["frames"].values()]) for word, _ in words}
    x4_spearman = spearman([epatch_word_predicted[word] for word, _ in words], [epatch_word_measured[word] for word, _ in words])
    x4_mae_by_template = {}
    for template in TEMPLATE_ORDER:
        errors = [abs(entry["epatch_mean_shift"] - entry["epatch_mean_predicted"]) for word, _ in words for entry in per_word[word]["frames"].values() if entry["template_id"] == template]
        x4_mae_by_template[template] = _mean(errors) if errors else None
    return {"per_word": per_word, "X3": {"spearman": x3_spearman, "confident_words": confident, "ambiguous_words": ambiguous, "sign_by_word": sign_by_word,
                                         "ambiguous_by_word": ambiguous_by_word, "word_mean_shift": word_means},
            "X4": {"spearman": x4_spearman, "mae_by_template": x4_mae_by_template, "word_measured": epatch_word_measured, "word_predicted": epatch_word_predicted}}


# ---------------------------------------------------------------------------
# Floors (exact counts), bootstrap bands, and band hits.


def _rows_from_json(rows: Sequence[Mapping[str, Any]]) -> tuple[CaseRow, ...]:
    return tuple(CaseRow(row["frame_id"], row["template_id"], row["lexical_key"], row["rule_class"], row["d_full"], row["d_patch"]) for row in rows)


def circuit_floors(results: Mapping[str, Any], *, hypothesis: str, n_cases: int) -> dict[str, Any]:
    """Pass/fail for every circuit-axis family from the design's floors; B1 is the precondition."""
    out: dict[str, Any] = {}
    behavior = results["behavior"]
    out["B1"] = {"passed": behavior["primary_correct"] >= exact_count_floor(FLOOR_RATES["B1_accuracy"], n_cases) and behavior["flips"] >= exact_count_floor(FLOOR_RATES["B1_flip"], n_cases),
                 "primary_correct": behavior["primary_correct"], "flips": behavior["flips"], "n": n_cases,
                 "required": {"accuracy": exact_count_floor(FLOOR_RATES["B1_accuracy"], n_cases), "flips": exact_count_floor(FLOOR_RATES["B1_flip"], n_cases)}}
    out["B2"] = {"passed": behavior["positive_pairs"] >= exact_count_floor(FLOOR_RATES["B2_positive"], n_cases), "positive_pairs": behavior["positive_pairs"],
                 "required": exact_count_floor(FLOOR_RATES["B2_positive"], n_cases)}
    p1_ok, p1_fail = recovery_floors_pass(results["P1"]["summary"], overall=FLOORS["P1_overall"], stratum=FLOORS["P1_stratum"])
    out["P1"] = {"passed": p1_ok, "failures": p1_fail, "recovery": results["P1"]["summary"]["overall"]["recovery"]}
    r_set = results["P1"]["summary"]["overall"]["recovery"]
    reference = results["P2"]["reference"]
    out["P2"] = {"passed": r_set is not None and r_set >= FLOORS["P2_multiplier"] * reference, "reference": reference, "recovery": r_set}
    p3 = results["P3"]["summary"]
    retention = results["P3"]["sign_retention"]
    p3_ok = p3["overall"]["recovery"] is not None and p3["overall"]["recovery"] >= FLOORS["P3_overall"] and all(
        entry["recovery"] is not None and entry["recovery"] >= FLOORS["P3_template"] for entry in p3["templates"].values()) and \
        retention["retained"] >= exact_count_floor(FLOOR_RATES["P3_retention"], retention["total"])
    out["P3"] = {"passed": p3_ok, "faithfulness": p3["overall"]["recovery"], "retained": retention, "required_retained": exact_count_floor(FLOOR_RATES["P3_retention"], retention["total"])}
    out["P4"] = {"passed": results["P4"]["value"] is not None and results["P4"]["value"] >= FLOORS["P4"], "value": results["P4"]["value"], "variant": results["P4"]["variant"]}
    p5 = results["P5"]
    if p5.get("applicable"):
        if hypothesis == "H1":
            passed = p5["a_single"] is not None and p5["a_single"] >= FLOORS["P5a_H1"] and p5["b_blocked_fraction"] is not None and p5["b_blocked_fraction"] >= FLOORS["P5b_H1"]
        else:
            passed = p5["a_declared"] is not None and p5["a_declared"] >= FLOORS["P5a_H2"]
        out["P5"] = {"passed": passed, "a_single": p5["a_single"], "a_declared": p5["a_declared"], "b_blocked_fraction": p5["b_blocked_fraction"]}
    else:
        out["P5"] = {"passed": False, "applicable": False}
    p6 = results["P6"]
    out["P6"] = {"passed": p6.get("applicable", False) and p6["opposite"] is not None and p6["opposite"] >= FLOORS["P6_opposite"] and p6["same"] is not None and p6["same"] <= FLOORS["P6_same"],
                 "opposite": p6.get("opposite"), "same": p6.get("same")}
    p7 = results["P7"]
    out["P7"] = {"passed": p7["same"] is not None and p7["same"] <= FLOORS["P7_same"] and p7["opposite"] is not None and r_set is not None and p7["opposite"] >= FLOORS["P7_opposite_factor"] * r_set,
                 "same": p7["same"], "opposite": p7["opposite"]}
    p8 = results["P8"]
    out["P8"] = {"passed": p8["loss"] is not None and p8["loss"] >= FLOORS["P8_loss"], "loss": p8["loss"], "compensation_ratio": p8["compensation_ratio"]}
    p9 = results["P9"]
    if p9.get("applicable"):
        if hypothesis in ("H1", "H2"):
            passed = p9["m_T"] >= FLOORS["P9"] and p9["m_R"] >= FLOORS["P9"] and p9["m_R_given_T_frozen"] <= FLOORS["P9_frozen_factor"] * p9["m_R"]
        else:
            passed = True  # H3 is judged by its band
        out["P9"] = {"passed": passed, "m_T": p9["m_T"], "m_R": p9["m_R"], "m_R_given_T_frozen": p9["m_R_given_T_frozen"]}
    else:
        out["P9"] = {"passed": False, "applicable": False}
    s1 = results["S1"]
    out["S1"] = {"passed": bool(s1.get("applicable")) and s1["minimum"] is not None and s1["minimum"] >= FLOORS["S1"], "minimum": s1.get("minimum"), "secondary": True}
    s3 = behavior["s3"]
    out["S3"] = {"passed": s3["n_wrong"] == 0 or s3["n_t_matches"] >= exact_count_floor(FLOOR_RATES["S3"], s3["n_wrong"]), "n_wrong": s3["n_wrong"], "n_t_matches": s3["n_t_matches"], "secondary": True}
    primary = ["B2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    out["primary_failures"] = [family for family in primary if not out[family]["passed"]]
    out["precondition_passed"] = out["B1"]["passed"]
    out["circuit_passed"] = out["B1"]["passed"] and not out["primary_failures"]
    return out


def _stat_recovery(rows: Sequence[CaseRow]) -> float | None:
    if not rows:
        return None
    denominator = _mean([row.d_full for row in rows])
    return _mean([row.d_patch for row in rows]) / denominator if denominator else None


def bootstrap_rows(rows: Sequence[CaseRow], statistic: Any, generator: Any, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Resample (frame, noun) cases with replacement within each template; return the point and SE."""
    groups: dict[str, list[CaseRow]] = {}
    for row in rows:
        groups.setdefault(row.template_id, []).append(row)
    point = statistic(rows)
    values = []
    for _ in range(resamples):
        sample: list[CaseRow] = []
        for group in groups.values():
            sample.extend(generator.choice(group) for _ in range(len(group)))
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return {"point": point, "se": 0.0}
    mean = _mean(values)
    se = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"point": point, "se": se}


def _band(point: float | None, se: float, *, width_floor: float = BAND_MIN_WIDTH) -> dict[str, Any] | None:
    if point is None:
        return None
    half = max(width_floor, BAND_SE_MULTIPLIER * se)
    return {"point": point, "se": se, "low": point - half, "high": point + half}


def calibration_bands(results: Mapping[str, Any], *, seed: int = CONTROL_SEED) -> dict[str, Any]:
    """Tier B bands for every band-bearing family, in a fixed family order from one generator."""
    import random

    generator = random.Random(seed)
    bands: dict[str, Any] = {}
    # B2: template mean d_full ± 0.5 nats (no bootstrap needed).
    bands["B2"] = {template: {"point": value, "low": value - BAND_D_FULL_NATS, "high": value + BAND_D_FULL_NATS} for template, value in results["behavior"]["template_mean_d_full"].items()}
    p1_rows = _rows_from_json(results["P1"]["rows"])
    bands["P1"] = {"overall": _band(**bootstrap_rows(p1_rows, _stat_recovery, generator))}
    for template in TEMPLATE_ORDER:
        subset = [row for row in p1_rows if row.template_id == template]
        if subset:
            bands["P1"][f"template:{template}"] = _band(**bootstrap_rows(subset, _stat_recovery, generator))
    for rule in RULE_CLASSES:
        subset = [row for row in p1_rows if row.rule_class == rule]
        if subset:
            bands["P1"][f"rule:{rule}"] = _band(**bootstrap_rows(subset, _stat_recovery, generator))
    bands["P2"] = {"median": _band(results["P2"]["median"], 0.0)}
    p3_rows = _rows_from_json(results["P3"]["rows"])
    bands["P3"] = {"overall": _band(**bootstrap_rows(p3_rows, _stat_recovery, generator))}
    for template in TEMPLATE_ORDER:
        subset = [row for row in p3_rows if row.template_id == template]
        if subset:
            bands["P3"][f"template:{template}"] = _band(**bootstrap_rows(subset, _stat_recovery, generator))
    retention = results["P3"]["sign_retention"]
    bands["P3"]["retention_rate"] = _band(retention["retained"] / retention["total"] if retention["total"] else None, 0.0)
    bands["P4"] = {"value": _band(**bootstrap_rows(_rows_from_json(results["P4"]["rows"]), _stat_recovery, generator))}
    p5 = results["P5"]
    if p5.get("applicable"):
        bands["P5"] = {name: _band(**bootstrap_rows(_rows_from_json(p5["rows"][key]), _stat_recovery, generator)) for name, key in
                       (("a_single", "single"), ("a_declared", "declared"), ("b_unfrozen", "unfrozen"), ("b_frozen", "frozen"))}
        bands["P5"]["b_blocked_fraction"] = _band(p5["b_blocked_fraction"], 0.0)
    p6 = results["P6"]
    if p6.get("applicable"):
        bands["P6"] = {name: _band(**bootstrap_rows(_rows_from_json(p6[f"rows_{name}"]), _stat_recovery, generator)) for name in ("opposite", "same")}
    bands["P7"] = {name: _band(**bootstrap_rows(_rows_from_json(results["P7"][f"rows_{name}"]), _stat_recovery, generator)) for name in ("opposite", "same") if results["P7"][f"rows_{name}"]}
    bands["P8"] = {"loss": _band(**bootstrap_rows(_rows_from_json(results["P8"]["rows"]), _stat_recovery, generator)),
                   "compensation_ratio": _band(results["P8"]["compensation_ratio"], 0.0)}
    p9 = results["P9"]
    if p9.get("applicable"):
        bands["P9"] = {name: _band(p9[name], 0.0) for name in ("m_T", "m_R", "m_R_given_T_frozen")}
    return bands


def _inside(band: Mapping[str, Any] | None, value: float | None) -> bool | None:
    if band is None or value is None:
        return None
    return band["low"] <= value <= band["high"]


def band_hits(results: Mapping[str, Any], bands: Mapping[str, Any]) -> dict[str, Any]:
    """Whether each band-bearing family's confirmation value lies inside its locked band."""
    hits: dict[str, Any] = {}
    behavior_means = results["behavior"]["template_mean_d_full"]
    hits["B2"] = {template: _inside(band, behavior_means.get(template)) for template, band in bands.get("B2", {}).items()}
    p1 = results["P1"]["summary"]
    p1_values = {"overall": p1["overall"]["recovery"]}
    p1_values.update({f"template:{name}": entry["recovery"] for name, entry in p1["templates"].items()})
    p1_values.update({f"rule:{name}": entry["recovery"] for name, entry in p1["rule_classes"].items()})
    hits["P1"] = {name: _inside(band, p1_values.get(name)) for name, band in bands.get("P1", {}).items()}
    hits["P2"] = {"median": _inside(bands.get("P2", {}).get("median"), results["P2"]["median"])}
    p3 = results["P3"]["summary"]
    p3_values = {"overall": p3["overall"]["recovery"], "retention_rate": results["P3"]["sign_retention"]["retained"] / max(results["P3"]["sign_retention"]["total"], 1)}
    p3_values.update({f"template:{name}": entry["recovery"] for name, entry in p3["templates"].items()})
    hits["P3"] = {name: _inside(band, p3_values.get(name)) for name, band in bands.get("P3", {}).items()}
    hits["P4"] = {"value": _inside(bands.get("P4", {}).get("value"), results["P4"]["value"])}
    if results["P5"].get("applicable") and "P5" in bands:
        hits["P5"] = {name: _inside(band, results["P5"].get(name)) for name, band in bands["P5"].items()}
    if results["P6"].get("applicable") and "P6" in bands:
        hits["P6"] = {name: _inside(band, results["P6"].get(name)) for name, band in bands["P6"].items()}
    hits["P7"] = {name: _inside(band, results["P7"].get(name)) for name, band in bands.get("P7", {}).items()}
    hits["P8"] = {name: _inside(band, results["P8"].get(name)) for name, band in bands.get("P8", {}).items()}
    if results["P9"].get("applicable") and "P9" in bands:
        hits["P9"] = {name: _inside(band, results["P9"].get(name)) for name, band in bands["P9"].items()}
    missed = [f"{family}:{name}" for family, entries in hits.items() for name, hit in entries.items() if hit is False]
    return {"hits": hits, "missed": missed, "all_hit": not missed}


def tolerance_tau(rmse_b: float) -> float:
    return max(TAU_MIN_NATS, TAU_RMSE_MULTIPLIER * rmse_b)


def decompilation_floors(new_frames: Mapping[str, Any], cue_words: Mapping[str, Any], *, hypothesis: str, capped: bool, x_bands: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """X1–X4 floors (exact counts) on the extension results."""
    out: dict[str, Any] = {}
    behavior = new_frames["behavior"]
    n = behavior["cases"]
    variables = behavior["variables"]
    required_variables = exact_count_floor(FLOOR_RATES["X1_variables"], variables["prompts"])
    out["X1"] = {"passed": behavior["positive_pairs"] >= exact_count_floor(FLOOR_RATES["X1_positive"], n) and variables["n_c"] >= required_variables and variables["n_t"] >= required_variables,
                 "positive_pairs": behavior["positive_pairs"], "required_positive": exact_count_floor(FLOOR_RATES["X1_positive"], n), "variables": variables, "required_variables": required_variables}
    circuit = circuit_floors({**new_frames, "P2": {"reference": 0.0, "median": 0.0}, "P6": {"applicable": False}, "P7": {"same": 0.0, "opposite": 1.0, "rows_same": [], "rows_opposite": []}, "S1": {"applicable": False}},
                             hypothesis=hypothesis, n_cases=n)
    x2_families = ["P1", "P3", "P4", "P5", "P8", "P9"]
    out["X2"] = {"passed": all(circuit[family]["passed"] for family in x2_families), "families": {family: circuit[family] for family in x2_families}}
    x3 = cue_words["X3"]
    k = len(x3["confident_words"])
    sign_total = sum(entry["agree"] for entry in x3["sign_by_word"].values())
    sign_ok = all(entry["agree"] >= exact_count_floor(FLOOR_RATES["X3_sign_word"], entry["total"]) for entry in x3["sign_by_word"].values()) and \
        sign_total >= exact_count_floor(FLOOR_RATES["X3_sign_overall"], 6 * k)
    ambiguous_ok = all(entry["small"] >= exact_count_floor(FLOOR_RATES["X3_ambiguous"], entry["total"]) for entry in x3["ambiguous_by_word"].values())
    out["X3"] = {"passed": x3["spearman"] >= FLOORS["X3_spearman"] and sign_ok and ambiguous_ok, "spearman": x3["spearman"], "sign_ok": sign_ok, "ambiguous_ok": ambiguous_ok, "k": k}
    x4 = cue_words["X4"]
    mae_ok = all(value is not None and value <= FLOORS["X4_mae_factor"] * cue_words["locked_full_shift"][template] for template, value in x4["mae_by_template"].items())
    out["X4"] = {"passed": x4["spearman"] >= FLOORS["X4_spearman"] and mae_ok, "spearman": x4["spearman"], "mae_by_template": x4["mae_by_template"], "mae_ok": mae_ok}
    out["primary_failures"] = [family for family in ("X1", "X2", "X3", "X4") if not out[family]["passed"]]
    out["program_capped"] = capped
    out["program_passed"] = not capped and not out["primary_failures"]
    return out


def x_band_hits(cue_words: Mapping[str, Any], x_bands: Mapping[str, Any]) -> dict[str, Any]:
    """X3/X4: at least 9 of 12 words inside their tolerance interval in at least 5 of 6 frames."""
    hits = {}
    for family, measured_key in (("X3", "mean_shift"), ("X4", "epatch_mean_shift")):
        words_hit = 0
        detail = {}
        for word, entry in x_bands[family].items():
            frames_hit = 0
            for frame_id, interval in entry.items():
                measured = cue_words["per_word"][word]["frames"][frame_id][measured_key]
                if interval["low"] <= measured <= interval["high"]:
                    frames_hit += 1
            detail[word] = frames_hit
            if frames_hit >= X_BAND_FRAMES:
                words_hit += 1
        hits[family] = {"words_hit": words_hit, "hit": words_hit >= X_BAND_WORDS, "frames_hit_by_word": detail}
    return hits


def site_axes_from_parameters(parameters_dir: Path) -> dict[str, SiteAxis]:
    """Rebuild the locked site axes (E_program, T, R_in, R_out) from the exported tensors."""
    directory = Path(parameters_dir)
    index = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))

    def tensor(name: str) -> torch.Tensor:
        return torch.load(directory / index["tensors"][name]["file"], map_location="cpu", weights_only=True)

    axes = {"E": SiteAxis("E_program", tensor("e_axis_mu"), tensor("e_axis_direction"), float(index["e_axis_sigma"]))}
    for label, sigma in index.get("axis_sigmas", {}).items():
        axes[label] = SiteAxis(label, tensor(f"axis_mu.{label}"), tensor(f"axis_direction.{label}"), float(sigma))
    return axes


# ---------------------------------------------------------------------------
# Preregistration lock, X predictions at lock time, revision, outcome.

LOCK_SCHEMA_VERSION = 1
LOCK_RELATIVE_PATH = "experiments/005-regular-plural-mechanism/preregistration-lock.json"
SCIENTIFIC_PATH_PREFIXES = ("src/", "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
HYPOTHESIS_REGIONS = {  # the tree's own thresholds, stored so a confirmation result can land "inside a rejected row"
    "H1": {"P5.b_blocked_fraction": [0.5, 1.0], "P5.a_single": [0.5, 10.0]},
    "H2": {"P5.b_blocked_fraction": [-10.0, 0.5], "P5.a_single": [-10.0, 0.5]},
    "H3": {"P9.m_R": [-10.0, 0.5]},
}


def reserve_digest(manifest: ScreeningManifest) -> str:
    return sha256_text(canonical_json([case.case_id for case in regular_plural_cases(manifest, Split.FUTURE_RESERVE)]))


def lock_x_predictions(program: Any, extension: Extension, reserve: Sequence[Noun], *, tau: float, reserve_bands: Mapping[str, Any]) -> dict[str, Any]:
    """Program predictions for every extension prompt, computed without running any of them."""
    frames_by_id = {frame.frame_id: frame for frame in extension.original_frames}
    words = {}
    for word, token in extension.cue_words:
        entry = {"token_id": token, "n_c": program.n_c(token), "frames": {}}
        for frame in extension.original_frames:
            reference = extension.reference_cue_ids[frame.template_id]
            shift = _mean([program.predict_shift(frame.template_id, frame.frame_id, token, reference, noun.sg_ids[0], noun.pl_ids[0]) for noun in reserve])
            epatch = _mean([program.predict_epatch_shift(frame.template_id, frame.frame_id, token, reference, noun.sg_ids[0], noun.pl_ids[0]) for noun in reserve])
            entry["frames"][frame.frame_id] = {"template_id": frame.template_id, "predicted_shift": shift, "shift_interval": {"low": shift - tau, "high": shift + tau},
                                               "predicted_epatch_shift": epatch, "epatch_interval": {"low": epatch - tau, "high": epatch + tau}}
        words[word] = entry
    new_frames = {}
    for frame in extension.new_frames:
        pairs = [program.predict_pair(frame.template_id, None, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]) for noun in reserve]
        new_frames[frame.frame_id] = {"template_id": frame.template_id, "predicted_mean_d_full": _mean(pairs), "predicted_positive_pairs": sum(1 for value in pairs if value > 0), "pairs": len(pairs)}
    return {"tau": tau, "cue_words": words, "new_frames": new_frames, "new_frame_bands": {"B2": reserve_bands.get("B2", {})},
            "X3_band": {word: {frame_id: entry["shift_interval"] for frame_id, entry in data["frames"].items()} for word, data in words.items()},
            "X4_band": {word: {frame_id: entry["epatch_interval"] for frame_id, entry in data["frames"].items()} for word, data in words.items()}}


def build_candidate_lock(*, state: Mapping[str, Any], version: Mapping[str, Any], calibration: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str,
                         extension: Extension, program: Any, parameters_dir: Path, program_path: Path, protocol_code_commit: str) -> dict[str, Any]:
    index_text = (Path(parameters_dir) / "parameters.json").read_text(encoding="utf-8")
    reserve = pm_reserve = nouns_for(manifest, Split.FUTURE_RESERVE)
    rmse_b = calibration["program_residual"]["rmse"]
    tau = tolerance_tau(rmse_b)
    x_predictions = lock_x_predictions(program, extension, reserve, tau=tau, reserve_bands=calibration["bands"])
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "created_at": utc_now(),
        "manifest_path": MANIFEST_RELATIVE_PATH, "manifest_sha256": manifest_sha256,
        "extension_path": EXTENSION_RELATIVE_PATH, "extension_sha256": extension.content_sha256,
        "reserve_case_digest": reserve_digest(manifest), "reserve_noun_keys": [noun.key for noun in pm_reserve],
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED},
        "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
        "mechanism": {key: version[key] for key in ("version", "mechanism", "hypothesis", "hypothesis_flagged", "hypothesis_quantities", "encoding_branch", "program_capped", "k")},
        "statement": version["statement"], "statement_sha256": sha256_text(version["statement"]),
        "hypothesis_regions": HYPOTHESIS_REGIONS,
        "parameters_index_sha256": sha256_text(index_text), "parameters_index": json.loads(index_text),
        "lexicon_sha256": program.lexicon_digest(), "program_source_sha256": sha256_text(Path(program_path).read_text(encoding="utf-8")),
        "program_residual": {"development": version["program_floors"].get("templates"), "holdout_rmse": rmse_b, "holdout_mae": calibration["program_residual"]["mae"], "tau": tau},
        "floors": {"rates": FLOOR_RATES, "values": FLOORS},
        "bands": calibration["bands"],
        "calibration": {"passes": len(state["calibration"]["passes"]), "floors": calibration["floors"], "n_cases": calibration["n_cases"]},
        "x_predictions": x_predictions,
        "tier_a_results_sha256": state.get("state_sha256"), "discovery_steps": sorted(state["discovery"]),
    }
    lock["content_sha256"] = sha256_text(canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str, extension: Extension,
                  parameters_dir: Path, program_path: Path, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    """Refuse confirmation unless the committed lock matches every frozen input and the current state."""
    if not isinstance(lock, Mapping) or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise PhaseError("lock schema is not recognized")
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("content_sha256") != sha256_text(canonical_json(unsigned)):
        raise PhaseError("lock content digest mismatch")
    if not tracked:
        raise PhaseError("the lock must be tracked and committed")
    if git_state.get("dirty", True):
        raise PhaseError("confirm requires a clean Git tree")
    if lock["manifest_sha256"] != manifest_sha256 or lock["extension_sha256"] != extension.content_sha256:
        raise PhaseError("lock was built against a different manifest or extension")
    if lock["reserve_case_digest"] != reserve_digest(manifest):
        raise PhaseError("lock reserve digest differs from the manifest")
    if lock["run_id"] != state["run_id"]:
        raise PhaseError("lock belongs to a different results state")
    versions = [entry for entry in state["mechanism_versions"] if entry.get("status") == "candidate"]
    if not versions or versions[-1]["version"] != lock["mechanism"]["version"] or sha256_text(versions[-1]["statement"]) != lock["statement_sha256"]:
        raise PhaseError("lock mechanism does not match the current candidate version")
    if lock["calibration"]["passes"] != len(state["calibration"]["passes"]) or not state["calibration"]["passes"][-1].get("floors_passed"):
        raise PhaseError("lock calibration record does not match the results state")
    index_text = (Path(parameters_dir) / "parameters.json").read_text(encoding="utf-8")
    if sha256_text(index_text) != lock["parameters_index_sha256"]:
        raise PhaseError("program parameters differ from the locked digest")
    if sha256_text(Path(program_path).read_text(encoding="utf-8")) != lock["program_source_sha256"]:
        raise PhaseError("mechanism_program.py differs from the locked source")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and not path.endswith("preregistration-lock.json")]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    assert_not_executed(state, extension=extension, reserve=nouns_for(manifest, Split.FUTURE_RESERVE))


def revise_version(previous: Mapping[str, Any], calibration: Mapping[str, Any], a2: Mapping[str, Any]) -> dict[str, Any]:
    """The design's two mechanical revisions; returns the instruction for building M2 (no model here)."""
    number = int(previous["version"][1:]) + 1
    if number > MAX_MECHANISM_VERSIONS:
        raise PhaseError("the mechanism-version budget is exhausted")
    failures = list(calibration["floors"]["primary_failures"])
    if not calibration["floors"]["precondition_passed"]:
        return {"version": f"M{number}", "action": "reject", "reason": "behavior did not replicate on holdout"}
    hypothesis_specific = [family for family in failures if family in ("P5", "P9")]
    set_level = [family for family in failures if family not in ("P5", "P9")]
    if set_level:
        mechanism = previous["mechanism"]
        current = set(mechanism["t_keys"]) | set(mechanism["r_keys"])
        ranking = [key for key in a2["rankings"]["p_t_all_templates"] if key != "L00.MLP" and key not in current]
        if len(current) >= MAX_COMPONENTS_AT_P_T or not ranking:
            return {"version": f"M{number}", "action": "reject", "reason": "component budget exhausted"}
        return {"version": f"M{number}", "action": "extend", "add": ranking[:1], "failed": set_level}
    if hypothesis_specific:
        order = list(HYPOTHESIS_ROWS)
        current_row = previous["hypothesis"]
        remaining = [row for row in order if order.index(row) > order.index(current_row)]
        if not remaining:
            return {"version": f"M{number}", "action": "reject", "reason": "no hypothesis row remains"}
        return {"version": f"M{number}", "action": "next_row", "row": remaining[0], "failed": hypothesis_specific}
    return {"version": f"M{number}", "action": "reject", "reason": "no primary family failed; nothing to revise"}


def contested(results: Mapping[str, Any], *, hypothesis: str, bands: Mapping[str, Any]) -> list[str]:
    """Discriminating values outside the chosen band and inside a rejected row's region."""
    values = {}
    if results["P5"].get("applicable"):
        values["P5.b_blocked_fraction"] = results["P5"]["b_blocked_fraction"]
        values["P5.a_single"] = results["P5"]["a_single"]
    if results["P9"].get("applicable"):
        values["P9.m_R"] = results["P9"]["m_R"]
    band_lookup = {"P5.b_blocked_fraction": bands.get("P5", {}).get("b_blocked_fraction"), "P5.a_single": bands.get("P5", {}).get("a_single"), "P9.m_R": bands.get("P9", {}).get("m_R")}
    findings = []
    for name, value in values.items():
        if value is None:
            continue
        band = band_lookup.get(name)
        outside_chosen = band is not None and not (band["low"] <= value <= band["high"])
        if not outside_chosen:
            continue
        for row, regions in HYPOTHESIS_REGIONS.items():
            if row == hypothesis or name not in regions:
                continue
            low, high = regions[name]
            if low <= value <= high:
                findings.append(f"{name}={value:.3f} inside {row}")
    return findings


def outcome(*, circuit: Mapping[str, Any], hits: Mapping[str, Any], decompilation: Mapping[str, Any] | None, x_hits: Mapping[str, Any] | None, contested_findings: Sequence[str]) -> dict[str, Any]:
    circuit_pass = circuit["circuit_passed"]
    program_pass = bool(decompilation and decompilation["program_passed"])
    axes = {"circuit": "CIRCUIT_PASS" if circuit_pass else "CIRCUIT_FAIL", "decompilation": "PROGRAM_PASS" if program_pass else "PROGRAM_FAIL"}
    bands_hit = hits["all_hit"] and (x_hits is None or all(entry["hit"] for entry in x_hits.values()))
    if not circuit["precondition_passed"]:
        label = "BEHAVIOR_NOT_REPLICATED"
    elif not circuit_pass:
        label = "MECHANISM_NOT_SUPPORTED"
    elif not program_pass:
        label = "CIRCUIT_ONLY"
    elif contested_findings:
        label = "MECHANISM_CONTESTED"
    elif bands_hit:
        label = "MECHANISM_CONFIRMED"
    else:
        label = "MECHANISM_SUPPORTED_MISCALIBRATED"
    return {"label": label, "axes": axes, "circuit_failures": circuit.get("primary_failures", []), "program_failures": (decompilation or {}).get("primary_failures", []),
            "missed_bands": hits["missed"], "x_bands": x_hits, "contested": list(contested_findings)}


# ---------------------------------------------------------------------------
# Markdown report.


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _band_text(band: Mapping[str, Any] | None) -> str:
    if not band:
        return "—"
    return f"[{band['low']:.3f}, {band['high']:.3f}]"


def _family_table(floors: Mapping[str, Any], bands: Mapping[str, Any] | None, hits: Mapping[str, Any] | None) -> list[str]:
    lines = ["| Family | Value | Floor | Band | Hit |", "|---|---|---|---|---|"]
    order = ["B1", "B2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "S1", "S3"]
    for family in order:
        entry = floors.get(family)
        if entry is None:
            continue
        value_keys = [key for key in entry if key not in {"passed", "secondary", "applicable", "failures", "required", "required_retained", "variant", "retained"}]
        value = "; ".join(f"{key}={_fmt(entry[key])}" for key in value_keys[:4])
        band_entries = (bands or {}).get(family, {})
        band = "; ".join(f"{name}={_band_text(item)}" for name, item in list(band_entries.items())[:3]) if isinstance(band_entries, dict) else "—"
        hit_entries = (hits or {}).get("hits", {}).get(family, {}) if hits else {}
        hit = "; ".join(f"{name}={'yes' if item else 'no' if item is False else '—'}" for name, item in list(hit_entries.items())[:3]) if hit_entries else "—"
        lines.append(f"| {family}{' (secondary)' if entry.get('secondary') else ''} | {value} | {_fmt(entry['passed'])} | {band} | {hit} |")
    return lines


def render_report(state: Mapping[str, Any]) -> str:
    lines = ["# Experiment 005 Report", ""]
    lines += [f"- Run ID: `{state['run_id']}`", f"- Manifest sha256: `{state['manifest_sha256']}`", f"- Extension sha256: `{state['extension_sha256']}`",
              f"- Protocol/code commit at discover: `{state['protocol_code_commit']}`", f"- Model: `{state['model']['model_id']}` @ `{state['model']['revision']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    discovery = state.get("discovery", {})
    if "a0_contract_test" in discovery:
        lines += [f"- A0 contract test: {'passed' if discovery['a0_contract_test'].get('passed') else 'FAILED'}"]
    if "a1" in discovery and "aggregates" in discovery["a1"]:
        a1 = discovery["a1"]
        lines += [f"- A1 baseline: {a1['screening_results_compared']} cases compared to the screen (max gap {a1['max_case_gap']:.2e}); development mean d_full {a1['aggregates']['development_mean_d_full']:.3f}; prompt-level gap {a1['max_prompt_level_gap']:.2e}"]
    if "hypothesis" in discovery:
        h = discovery["hypothesis"]
        lines += [f"- Hypothesis tree: `{h['row']}`{' (AMBIGUOUS)' if h['flagged'] else ''} from " + ", ".join(f"{k}={_fmt(v)}" for k, v in h["quantities"].items()) + f" (top head `{h['top_head']}`)"]
    if "encoding_branch" in discovery:
        e = discovery["encoding_branch"]
        lines += [f"- Encoding branch: `{e['branch']}` with E = {e['e_keys']} (E share {_fmt(e['e_share'])}, embedding share {_fmt(e['embedding_share'])}{', flag ' + e['flag'] if e.get('flag') else ''})"]
    if "a2" in discovery:
        lines += ["", "### A2 rankings", "", f"- p_t (all templates): {', '.join(discovery['a2']['rankings']['p_t_all_templates'][:8])}",
                  f"- p_t (coordinated): {', '.join(discovery['a2']['rankings']['p_t_coordinated'][:8])}", f"- p_c (coordinated): {', '.join(discovery['a2']['rankings']['p_c_coordinated'][:8])}"]
    if "a3" in discovery:
        lines += ["", "### A3 residual profile at p_c (coordinated)", ""] + [f"- {layer}: recovery {_fmt(entry['recovery'])}" for layer, entry in discovery["a3"]["recovery_by_layer"].items()]
    if "a4" in discovery:
        lines += ["", "### A4 attention to the cue from p_t", ""] + [f"- {key}: {_fmt(discovery['a4']['heads'][key]['attention_to_cue'])}" for key in discovery["a4"]["ranking_by_attention_to_cue"][:6]]
    if "a6" in discovery:
        a6 = discovery["a6"]
        lines += ["", "### A6 chain and path", "", f"- E alone: recovery {_fmt(a6['e_alone']['recovery'].get('recovery'))}, m_T {_fmt(a6['e_alone'].get('m_T'))}, m_R {_fmt(a6['e_alone']['m_R'])}",
                  f"- residual patch at p_c: unfrozen {_fmt(a6['residual_patch']['recovery_unfrozen'])}; blocked fractions " + ", ".join(f"{k}={_fmt(v['blocked_fraction'])}" for k, v in a6["residual_patch"]["frozen"].items()),
                  f"- T alone: " + ", ".join(f"{k}={_fmt(v)}" for k, v in a6["t_alone"]["singles"].items()) + "; cumulative " + ", ".join(f"{k}={_fmt(v)}" for k, v in a6["t_alone"]["cumulative"].items()),
                  f"- R alone: {_fmt(a6['r_alone']['recovery'])}"]
    for entry in state.get("mechanism_versions", []):
        lines += ["", f"## Mechanism version {entry['version']} — {entry['status']}", ""]
        if entry.get("statement"):
            lines += ["```text", entry["statement"].rstrip(), "```", ""]
            recovery = entry.get("recovery", {}).get("overall", {})
            lines += [f"- Development recovery {_fmt(recovery.get('recovery'))}; isolation faithfulness {_fmt(entry.get('isolation', {}).get('overall', {}).get('recovery'))}; program floors {'passed' if entry.get('program_floors', {}).get('passed') else 'FAILED'}; program capped: {entry.get('program_capped')}"]
        else:
            lines += [f"- Outcome: {entry.get('outcome')} {entry.get('reason', '')}"]
    for index, entry in enumerate(state.get("calibration", {}).get("passes", []), start=1):
        lines += ["", f"## Calibration pass {index} — {entry['version']} on {entry['n_cases']} holdout cases", "",
                  f"- Floors: {'passed' if entry['floors_passed'] else 'FAILED ' + str(entry['floors'].get('primary_failures'))}; program residual RMSE_B {_fmt(entry['program_residual']['rmse'], 4)} (τ = {_fmt(tolerance_tau(entry['program_residual']['rmse']))})", ""]
        lines += _family_table(entry["floors"], entry["bands"], None)
    if state.get("lock"):
        lines += ["", "## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}` for {state['lock']['version']}"]
    confirmation = state.get("confirmation")
    if confirmation:
        verdict = confirmation["outcome"]
        lines += ["", f"## Confirmation — outcome `{verdict['label']}`", "", f"- Axes: circuit `{verdict['axes']['circuit']}`, decompilation `{verdict['axes']['decompilation']}`",
                  f"- Circuit failures: {verdict['circuit_failures'] or 'none'}; program failures: {verdict['program_failures'] or 'none'}; missed bands: {verdict['missed_bands'] or 'none'}; contested: {verdict['contested'] or 'none'}",
                  "", "### Reserve nouns (readout generalization)", ""]
        lines += _family_table(confirmation["circuit_floors"], None, confirmation["band_hits"])
        deco = confirmation["decompilation_floors"]
        lines += ["", "### Extension prompts (prompt-side generalization)", "",
                  f"- X1 frame invariance of behavior: {_fmt(deco['X1']['passed'])} (positive pairs {deco['X1']['positive_pairs']}/{deco['X1']['required_positive']} required; variables {deco['X1']['variables']})",
                  f"- X2 frame invariance of the circuit: {_fmt(deco['X2']['passed'])} (" + ", ".join(f"{k}={_fmt(v['passed'])}" for k, v in deco["X2"]["families"].items()) + ")",
                  f"- X3 cue lexicon: {_fmt(deco['X3']['passed'])} (Spearman {_fmt(deco['X3']['spearman'])}, {deco['X3']['k']} confident words, signs {deco['X3']['sign_ok']}, ambiguous {deco['X3']['ambiguous_ok']})",
                  f"- X4 E-patch prediction: {_fmt(deco['X4']['passed'])} (Spearman {_fmt(deco['X4']['spearman'])}, MAE by template " + ", ".join(f"{k}={_fmt(v)}" for k, v in deco["X4"]["mae_by_template"].items()) + ")",
                  f"- X band hits: " + ", ".join(f"{k}: {v['words_hit']}/12 words" for k, v in confirmation["x_band_hits"].items()),
                  "", "### Cue words", "", "| Word | n_c | mean shift | E-patch measured | E-patch predicted |", "|---|---|---|---|---|"]
        for word, entry in confirmation["cue_words"]["per_word"].items():
            frames = entry["frames"].values()
            lines.append(f"| {word} | {_fmt(entry['n_c'])} | {_fmt(_mean([f['mean_shift'] for f in frames]))} | {_fmt(_mean([f['epatch_mean_shift'] for f in frames]))} | {_fmt(_mean([f['epatch_mean_predicted'] for f in frames]))} |")
        reserve = confirmation["reserve"]
        lines += ["", "### Residual accounting (Q1)", "",
                  f"- Unexplained sufficiency residual 1 − R: {_fmt(1 - reserve['P1']['summary']['overall']['recovery'] if reserve['P1']['summary']['overall']['recovery'] is not None else None)}",
                  f"- Unexplained isolation residual 1 − F: {_fmt(1 - reserve['P3']['summary']['overall']['recovery'] if reserve['P3']['summary']['overall']['recovery'] is not None else None)}",
                  f"- Program residual on reserve conditions: RMSE {_fmt(confirmation['reserve_program_residual']['rmse'], 4)}, MAE {_fmt(confirmation['reserve_program_residual']['mae'], 4)}",
                  f"- Program residual on new frames: RMSE {_fmt(confirmation['new_frames_program_residual']['rmse'], 4)}, MAE {_fmt(confirmation['new_frames_program_residual']['mae'], 4)}"]
    lines += ["", "## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}",
              f"- Invalidated runs: {len(state['invalidated_runs'])}", ""]
    return "\n".join(lines)
