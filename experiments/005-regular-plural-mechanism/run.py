#!/usr/bin/env python3
"""Run Experiment 005 through isolated phases.

``validate`` checks the manifest and the frozen extension without a model.
``freeze-extension`` builds ``extension-v1.json`` from tokenizer rules only and
refuses to overwrite it. The scientific phases ``discover``, ``calibrate``,
``revise``, ``lock``, ``confirm``, and ``report`` are enforced in order through
the results state; reserve nouns and extension prompts are executed only by
``confirm`` after the committed preregistration lock validates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import load_manifest
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/experiment-005"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("validate", help="check the manifest and frozen extension without loading a model")
    phases.add_parser("freeze-extension", help="build extension-v1.json from tokenizer rules only (refuses to overwrite)")
    return parser


def _git_tracked(path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name,
            "seed": pm.RUNTIME_SEED, "deterministic_algorithms": spec.deterministic_algorithms}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


@dataclass
class Runner:
    """Phase orchestration with injectable model, tokenizer, Git, and version providers."""

    root: Path = ROOT
    experiment_dir: Path = EXPERIMENT_DIR
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def manifest_path(self) -> Path:
        return self.root / pm.MANIFEST_RELATIVE_PATH

    @property
    def extension_path(self) -> Path:
        return self.root / pm.EXTENSION_RELATIVE_PATH

    def _manifest(self):
        manifest = load_manifest(self.manifest_path)
        return manifest, pm.manifest_digest_from_path(self.manifest_path)

    def validate(self) -> int:
        manifest, manifest_sha256 = self._manifest()
        frames = pm.derive_frames(manifest)
        nouns = {split: pm.nouns_for(manifest, split) for split in pm.Split}
        self.log(f"manifest ok: sha256 {manifest_sha256}; {len(frames)} frames; "
                 + ", ".join(f"{split.value}={len(items)} nouns" for split, items in nouns.items()))
        if not self.extension_path.exists():
            self.log("extension not frozen yet: run freeze-extension")
            return 1
        extension = pm.load_extension(self.extension_path, manifest, manifest_sha256)
        tracked = self.tracked(self.extension_path)
        self.log(f"extension ok: sha256 {extension.content_sha256}; {len(extension.new_frames)} new frames; "
                 f"{len(extension.cue_words)} cue words; {len(extension.cue_word_prompts)} cue-word prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_extension(self) -> int:
        manifest, manifest_sha256 = self._manifest()
        if self.extension_path.exists():
            self.log(f"refusing to overwrite frozen extension {self.extension_path}")
            return 1
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        digest = pm.freeze_extension(self.extension_path, tokenizer, manifest, manifest_sha256)
        self.log(f"froze extension {self.extension_path} sha256 {digest}; commit it before any scientific phase")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase == "validate":
        return runner.validate()
    if args.phase == "freeze-extension":
        return runner.freeze_extension()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
