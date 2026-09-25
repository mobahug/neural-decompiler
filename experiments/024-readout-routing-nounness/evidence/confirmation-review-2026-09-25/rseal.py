"""Confirmation review 024 -- the model seal (reviewer's own code).

Phase 1 (during the load): torch.nn.Module.__call__ and every already-defined module class's own forward/__call__ are
wrapped by COUNTING delegates (they still work, so the count is honest).
Phase 2 (after the load, before any program is built): Module.__call__, every module class's own forward and __call__
(all subclasses, recursively, including the ones defined during the load) and torch.nn.Module.forward become REFUSING
stubs; every prompt-execution entry point in every loaded neural_decompiler module (capture_prompt, run_capture,
run_patched, run_interventions, record_execution, stage_two_022, measure_prompt, capture_frame_020, capture_reference,
calibration_rematerialize, stage_one_022) becomes a refusing stub. Every attempt is counted.
"""
from __future__ import annotations

import sys

import torch

REFUSED_NAMES = ("capture_prompt", "run_capture", "run_patched", "run_interventions", "record_execution", "stage_two_022", "measure_prompt",
                 "capture_frame_020", "capture_reference", "calibration_rematerialize", "stage_one_022")


class SealViolation(RuntimeError):
    pass


def _subclasses(cls):
    out, stack = set(), [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in out:
                out.add(sub)
                stack.append(sub)
    return out


class Seal:
    def __init__(self):
        self.load_counts = {"Module.__call__": 0, "forward": 0, "__call__ override": 0}
        self.refused = {"Module.__call__": 0, "forward": 0, "__call__ override": 0, "entry points": 0}
        self.entry_points = []
        self._saved = []
        self.phase = "idle"

    # -- phase 1 ------------------------------------------------------------------------------------------------------
    def count_during_load(self):
        original_call = torch.nn.Module.__call__
        counts = self.load_counts

        def counting_call(module, *args, **kwargs):
            counts["Module.__call__"] += 1
            return original_call(module, *args, **kwargs)

        self._saved.append((torch.nn.Module, "__call__", original_call))
        torch.nn.Module.__call__ = counting_call
        for cls in _subclasses(torch.nn.Module):
            for name, key in (("forward", "forward"), ("__call__", "__call__ override")):
                if name in vars(cls):
                    fn = vars(cls)[name]

                    def make(fn=fn, key=key):
                        def counting(*args, **kwargs):
                            counts[key] += 1
                            return fn(*args, **kwargs)
                        return counting

                    self._saved.append((cls, name, fn))
                    setattr(cls, name, make())
        self.phase = "load"

    # -- phase 2 ------------------------------------------------------------------------------------------------------
    def refuse_everything(self):
        refused = self.refused

        def stub(kind, label):
            def refusing(*args, **kwargs):
                refused[kind] += 1
                raise SealViolation(f"sealed: {label} was reached")
            refusing.__name__ = "refusing_stub"
            return refusing

        torch.nn.Module.__call__ = stub("Module.__call__", "torch.nn.Module.__call__")
        torch.nn.Module.forward = stub("forward", "torch.nn.Module.forward")
        n_classes = 0
        for cls in _subclasses(torch.nn.Module):
            touched = False
            if "forward" in vars(cls):
                setattr(cls, "forward", stub("forward", f"{cls.__module__}.{cls.__qualname__}.forward"))
                touched = True
            if "__call__" in vars(cls):
                setattr(cls, "__call__", stub("__call__ override", f"{cls.__module__}.{cls.__qualname__}.__call__"))
                touched = True
            n_classes += touched
        for modname, module in list(sys.modules.items()):
            if module is None or not (modname == "neural_decompiler" or modname.startswith("neural_decompiler.") or modname.startswith("run024")):
                continue
            for name in REFUSED_NAMES:
                if hasattr(module, name) and callable(getattr(module, name)):
                    setattr(module, name, stub("entry points", f"{modname}.{name}"))
                    self.entry_points.append(f"{modname}.{name}")
        self.phase = "sealed"
        return n_classes

    def verify_model_sealed(self, model):
        """Every module instance of the loaded model resolves forward and __call__ to a refusing stub."""
        bad = []
        for m in model.modules():
            f = type(m).forward
            c = type(m).__call__
            if getattr(f, "__name__", "") != "refusing_stub" or getattr(c, "__name__", "") != "refusing_stub":
                bad.append(f"{type(m).__module__}.{type(m).__qualname__}")
        return sorted(set(bad))
