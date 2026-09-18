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
