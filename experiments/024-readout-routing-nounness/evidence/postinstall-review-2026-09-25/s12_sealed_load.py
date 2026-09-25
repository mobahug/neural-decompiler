"""Supplement to item 6: seal BEFORE the load. Every forward implementation and every __call__ override of every imported
torch module class (torch, transformers GPT-NeoX, transformer_lens bridge) is replaced by a refusing, counting stub, and
Module.__call__ too, before load_model runs; the load must succeed with 0 attempts, every class in the loaded model must
be one that was sealed beforehand, and I7 must still be bitwise equal."""
import guard  # noqa: F401
from guard import REPO, Seal, summary

import importlib.util
import json
import os
import sys

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


RUN_PY = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
spec = importlib.util.spec_from_file_location("run024_postinstall_review", RUN_PY)
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr, rd, ul = run.rr, run.rd, run.ul
from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime  # noqa: E402

seal = Seal()
seal.refuse_captures()


def refuse(spec):
    raise RuntimeError("not used")


runner = run.Runner(log=lambda m: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
state = runner._state_for("confirm", base.digests)
lock = json.loads((runner.root / rr.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, base.digests)
runner._check_runtime(base.inputs.closure)
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)

import torch  # noqa: E402
import transformer_lens.model_bridge  # noqa: E402,F401
import transformer_lens.model_bridge.generalized_components  # noqa: E402,F401
import transformers.activations  # noqa: E402,F401
import transformers.models.gpt_neox.modeling_gpt_neox  # noqa: E402,F401

attempts = {"forward": 0, "__call__": 0, "Module.__call__": 0}
sealed = set()


def all_subclasses(cls):
    out, stack = set(), [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in out:
                out.add(sub)
                stack.append(sub)
    return out


def make(kind, label):
    def refuse_stub(*args, **kwargs):
        attempts[kind] += 1
        raise RuntimeError(f"sealed before the load: {label}")
    return refuse_stub


for cls in all_subclasses(torch.nn.Module):
    if "forward" in vars(cls):
        cls.forward = make("forward", f"{cls.__qualname__}.forward")
        sealed.add(cls)
    if "__call__" in vars(cls):
        cls.__call__ = make("__call__", f"{cls.__qualname__}.__call__")
        sealed.add(cls)
torch.nn.Module.__call__ = make("Module.__call__", "Module.__call__")
print(f"sealed before the load: {len(sealed)} module classes (forward and/or __call__), plus Module.__call__")
model = load_model(PYTHIA_70M)
classes = {type(m) for m in model.modules()}
def resolved_forward(c):
    return getattr(c, "forward")


no_impl = sorted(f"{c.__module__}.{c.__qualname__}" for c in classes if resolved_forward(c) is torch.nn.Module.forward)  # _forward_unimplemented
uncovered = sorted(f"{c.__module__}.{c.__qualname__}" for c in classes
                   if resolved_forward(c) is not torch.nn.Module.forward and getattr(resolved_forward(c), "__name__", "") != "refuse_stub")
print(f"   classes with no forward implementation at all (torch's _forward_unimplemented): {no_impl}")
check("the load succeeded with every forward and module call refused from the start", True)
check("0 forward / __call__ / Module.__call__ attempts during the load", attempts == {"forward": 0, "__call__": 0, "Module.__call__": 0}, attempts)
check("every module class of the loaded model resolves forward to a pre-load refusing stub (or has no forward implementation)", not uncovered, uncovered)
print(f"   loaded-model module classes: {len(classes)}")
parameters_sha = rr.parameters_digest(model)
rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=parameters_sha, embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"], what="the lock")
progs = ul.ModelPrograms.from_model(model, base.inputs)
runner._check_nouns(progs, base.inputs)
with run.pytest_free_guard():
    rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=lock["dependencies"]["model"]["parameters_sha256"],
                                 embedding_sha256=rr.embedding_digest(progs.weights.W_E), what="the lock")
    reproduced = rr.reproduce_lock_quantities(progs.weights.W_E, record, confirmation, lock)
check("I7 under the pre-load seal: bitwise_equal True, nothing differing", reproduced["bitwise_equal"] is True and reproduced["differing"] == [], reproduced)
check("still 0 attempts after the programs and I7", attempts == {"forward": 0, "__call__": 0, "Module.__call__": 0} and seal.counts["capture_hits"] == 0, attempts)
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"SEALED-LOAD {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
