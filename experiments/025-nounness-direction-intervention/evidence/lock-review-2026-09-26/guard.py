"""Review guard for the Experiment 025 lock review (read-only).

Import this module FIRST in every review script. It installs:
- an audit hook that refuses (and records) every read under outputs/experiment-023/, outputs/experiment-024/ and of any
  calibration-table.pt, every directory listing of those, and every repository write/delete/rename/link/mkdir/truncate
  (the repository's .venv included). Writes outside the repository are only logged.
- after `seal()` (called once torch and the project modules are imported): torch.nn.Module.__call__ refuses, every
  capture/intervention entry point (and every alias found by identity in any neural_decompiler module) refuses, and the
  confirm-only computations refuse (patched measurement, stage 2, the patch path, gates, C, D_attn, ell, A/B/G,
  descriptives, ul.measure_prompt, ul.contrast_of, NounSet.contrasts, ReadoutProgram.level1_detail/contrast/blocks_3_to_5,
  rr.cue_mse/fresh_pair_cells).
- `refuse_cr_geometry()` refuses the cue_rotation geometry functions until the reviewer's own values exist.
- `allow_spent_reference_capture(...)` (item 10 only) is the single door through which at most 6 plain captures of spent
  020 reference prompts `frame_id|ref|<reference id>` may run, every key checked before execution.
"""
from __future__ import annotations

import fcntl
import os
import sys
import threading
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
EVENTS: dict = {"forbidden_reads": [], "repo_writes_refused": [], "fs_ops_refused": [], "outside_writes": [], "module_calls_refused": 0,
                "refused": [], "aliases_patched": [], "captures_allowed": [], "captures_refused": []}
_STATE = threading.local()


def _depth() -> int:
    return getattr(_STATE, "depth", 0)


def _path(value, dir_fd=None):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        raw = os.fsdecode(value)
        if isinstance(dir_fd, int) and not os.path.isabs(raw):
            raw = os.path.join(fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode(), raw)
        return Path(raw).resolve()
    except (OSError, ValueError):
        return None


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def _forbidden_read(text: str) -> bool:
    return text.endswith("calibration-table.pt") or "/outputs/experiment-023" in text or "/outputs/experiment-024" in text


def _hook(event, args):
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return
        text = str(path)
        if _forbidden_read(text):
            EVENTS["forbidden_reads"].append(text)
            raise PermissionError(f"review may not open {text}")
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        if not writing:
            return
        if _inside(path, ROOT):
            EVENTS["repo_writes_refused"].append(text)
            raise PermissionError(f"review may not write {text}")
        EVENTS["outside_writes"].append(text)
        return
    if event in ("os.listdir", "os.scandir"):
        path = _path(args[0]) if args and args[0] is not None else None
        if path is not None and _forbidden_read(str(path) + "/"):
            EVENTS["forbidden_reads"].append(str(path))
            raise PermissionError(f"review may not list {path}")
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod", "os.chown", "os.utime", "os.chflags"):
        dir_fd = None
        if event == "os.mkdir" and len(args) > 2:
            dir_fd = args[2]
        elif event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1:
            dir_fd = args[1]
        path = _path(args[0], dir_fd) if args else None
        if path is None:
            return
        if _inside(path, ROOT):
            EVENTS["fs_ops_refused"].append(f"{event} {path}")
            raise PermissionError(f"review may not {event} {path}")
        return
    if event in ("os.rename", "os.link", "os.symlink"):
        if event == "os.symlink":
            candidates = (_path(args[1], args[2] if len(args) > 2 else None),)
        else:
            candidates = (_path(args[0], args[2] if len(args) > 2 else None), _path(args[1], args[3] if len(args) > 3 else None))
        if any(p is not None and _inside(p, ROOT) for p in candidates):
            EVENTS["fs_ops_refused"].append(f"{event} {candidates}")
            raise PermissionError(f"review may not {event} {candidates}")


sys.addaudithook(_hook)

_SEALED = {"done": False}
_ORIGINALS: dict = {}


def _refusal(name: str, original=None, passthrough: bool = False):
    def refuse(*args, **kwargs):
        if passthrough and _depth() > 0 and original is not None:
            return original(*args, **kwargs)
        EVENTS["refused"].append(name)
        raise RuntimeError(f"{name} refused during the lock review")

    refuse.__wrapped_name__ = name
    return refuse


def seal(confirm_only: bool = True) -> None:
    """Install every refusal. Idempotent. With confirm_only=False the pure cue_rotation/readout functions stay callable
    (for synthetic unit tests with no model and no measurement); forwards, captures and interventions always refuse."""
    if _SEALED["done"]:
        return
    import torch

    from neural_decompiler import capture, interventions
    from neural_decompiler import cue_rotation as cr
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler import readout_decompilation as rd
    from neural_decompiler import readout_routing as rr
    from neural_decompiler import upstream_localization as ul

    original_call = torch.nn.Module.__call__
    _ORIGINALS["Module.__call__"] = original_call

    def module_call(self, *args, **kwargs):
        if _depth() > 0:
            return original_call(self, *args, **kwargs)
        EVENTS["module_calls_refused"] += 1
        raise RuntimeError("a torch module was called during the lock review")

    torch.nn.Module.__call__ = module_call
    # run_capture may pass through only inside the single allowed door (a plain capture); run_patched and
    # run_interventions never pass.
    entry = {id(capture.run_capture): ("run_capture", capture.run_capture, True),
             id(interventions.run_interventions): ("run_interventions", interventions.run_interventions, False),
             id(pm.capture_prompt): ("capture_prompt", pm.capture_prompt, False),
             id(pm.run_patched): ("run_patched", pm.run_patched, False)}
    _ORIGINALS["capture_prompt"] = pm.capture_prompt
    refusals = {key: _refusal(name, original, passthrough) for key, (name, original, passthrough) in entry.items()}
    _ORIGINALS["entry"] = entry
    _ORIGINALS["refusals"] = refusals
    sweep()
    if not confirm_only:
        _SEALED["done"] = True
        return
    for name in ("measure_rotated", "stage_two", "patch_path_check", "identity_gates", "level1_gates", "run_gates", "recompute_c", "routing_distance", "per_cue",
                 "descriptives", "ladder", "vector_factors", "lexicon_of", "assert_measurements", "enforce_gates"):
        setattr(cr, name, _refusal(f"cr.{name}"))
    for name in ("measure_prompt", "contrast_of", "compose_dx3", "i3_error", "pair_compositions", "pair_gates"):
        if hasattr(ul, name):
            setattr(ul, name, _refusal(f"ul.{name}"))
    for name in ("cue_mse", "fresh_pair_cells", "cue_mse_torch"):
        setattr(rr, name, _refusal(f"rr.{name}"))
    for name in ("level1_detail", "contrast", "blocks_3_to_5"):
        setattr(rd.ReadoutProgram, name, _refusal(f"rd.ReadoutProgram.{name}"))
    rd.NounSet.contrasts = _refusal("rd.NounSet.contrasts")
    _SEALED["done"] = True


def sweep() -> None:
    """Replace every module attribute that *is* an original entry point (definition, alias or copy) by its refusal, in
    every loaded neural_decompiler module and in the runner. Safe to call again after new imports."""
    entry, refusals = _ORIGINALS["entry"], _ORIGINALS["refusals"]
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(("neural_decompiler", "experiment_025")):
            continue
        for attribute, value in list(vars(module).items()):
            key = id(value)
            if key in entry and value is entry[key][1]:
                setattr(module, attribute, refusals[key])
                EVENTS["aliases_patched"].append(f"{module_name}.{attribute}")


def refuse_model_load() -> None:
    """No model at all (for scripts that need none)."""
    from neural_decompiler import models

    refusal = _refusal("models.load_model")
    original = models.load_model
    models.load_model = refusal
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(("neural_decompiler", "experiment_025")):
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                setattr(module, attribute, refusal)
                EVENTS["aliases_patched"].append(f"{module_name}.{attribute}")


CR_GEOMETRY = ("geometry_block", "cue_vectors", "cue_geometry", "direction", "centroids", "control_directions", "sha_uniforms", "sha_gaussians",
               "plurality_direction", "plurality_tangent", "theta_for", "rotate", "project_off", "condition_directions", "condition_vectors",
               "geometry_checks", "score", "angle", "nearest_tokens", "unit")


def refuse_cr_geometry() -> None:
    from neural_decompiler import cue_rotation as cr

    for name in CR_GEOMETRY:
        setattr(cr, name, _refusal(f"cr.{name} (before the reviewer's own values exist)"))


def seal_model_forwards(model, *, door: bool = False) -> int:
    """Every module instance of a loaded model gets a refusing `forward` (direct .forward() routes); the class-level
    model entry points refuse too. With door=False nothing ever passes; with door=True a call passes only while the
    single guarded capture door (allow_spent_reference_capture) is executing."""
    count = 0
    for module_name, module in model.named_modules():
        original = module.forward
        object.__setattr__(module, "forward", _refusal(f"{module_name or type(module).__name__}.forward", original, door))
        count += 1
    for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
        if hasattr(type(model), method):
            setattr(type(model), method, _refusal(f"{type(model).__name__}.{method}", getattr(type(model), method), door))
    return count


def allow_spent_reference_capture(model, prompt, sites, *, ledger: frozenset, forbidden_prefixes: frozenset, budget: dict):
    """The single door for item 10: a plain capture of one spent 020 reference prompt. Every check runs before the
    forward; any other key is refused and never executed."""
    key = prompt.key
    frame = prompt.frame
    template = frame.template_id
    problems = []
    if prompt.cue_label != "ref" or key != f"{frame.frame_id}|ref|{prompt.cue_token_id}":
        problems.append("not a reference-prompt key frame_id|ref|<id>")
    if key not in ledger:
        problems.append("not in Experiment 020's ledger")
    if key in forbidden_prefixes:
        problems.append("an Experiment 025 manifest prompt key")
    if budget["total"] >= 6 or budget.setdefault(template, 0) >= 2:
        problems.append("budget exhausted (6 in all, 2 per template family)")
    if sorted(set(site[0] for site in sites)) != ["ATTN_PATTERN.L4", "ATTN_PATTERN.L5"] or any(site[1] != frame.p_t for site in sites):
        problems.append("only ATTN_PATTERN.L4/L5 at p_t may be captured")
    if problems:
        EVENTS["captures_refused"].append({"key": key, "problems": problems})
        raise RuntimeError(f"capture of {key} refused: {problems}")
    budget["total"] += 1
    budget[template] += 1
    EVENTS["captures_allowed"].append(key)
    _STATE.depth = _depth() + 1
    try:
        return _ORIGINALS["capture_prompt"](model, prompt, sites)
    finally:
        _STATE.depth = _depth() - 1


def summary() -> dict:
    return {key: (value if not isinstance(value, list) else list(value)) for key, value in EVENTS.items()}
