"""Item 4 (static part): every write-capable function in the package and run.py, and an over-approximate name-based call
reachability from confirm's pre-model / pre-prompt helpers. Own AST code; nothing is imported from the package."""
import guard  # noqa: F401
from guard import REPO, summary

import ast
import os
import re

SRC = os.path.join(REPO, "src/neural_decompiler")
RUN = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
modules = {f"neural_decompiler.{f[:-3]}": os.path.join(SRC, f) for f in sorted(os.listdir(SRC)) if f.endswith(".py")}
modules["run024"] = RUN
WRITE_RE = re.compile(r"write_text\(|write_bytes\(|\.mkdir\(|os\.replace\(|os\.rename\(|os\.unlink\(|\.unlink\(|torch\.save\(|tempfile\.mkstemp\(|shutil\.|os\.remove\(|"
                      r"rmtree\(|os\.fdopen\(|open\([^)]*['\"][wax]")

trees, funcs, classes, aliases, from_names = {}, {}, {}, {}, {}
for mod, path in modules.items():
    with open(path, encoding="utf-8") as h:
        source = h.read()
    tree = ast.parse(source)
    trees[mod] = (tree, source.splitlines())
    aliases[mod], from_names[mod] = {}, {}
    pkg = "neural_decompiler"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[mod][a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = pkg + ("." + base if base else "")
            for a in node.names:
                full = f"{base}.{a.name}"
                if full in modules:
                    aliases[mod][a.asname or a.name] = full
                else:
                    from_names[mod][a.asname or a.name] = (base, a.name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[(mod, node.name)] = node
        elif isinstance(node, ast.ClassDef):
            classes[(mod, node.name)] = node
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs[(mod, f"{node.name}.{item.name}")] = item

methods_by_name = {}
for (mod, qual) in funcs:
    methods_by_name.setdefault(qual.split(".")[-1], set()).add((mod, qual))


def enclosing(mod, lineno):
    best = None
    for (m, qual), node in funcs.items():
        if m == mod and node.lineno <= lineno <= node.end_lineno:
            if best is None or node.lineno > funcs[best].lineno:
                best = (m, qual)
    return best


writers = {}
for mod, (tree, lines) in trees.items():
    for i, line in enumerate(lines, 1):
        if WRITE_RE.search(line) and not line.strip().startswith("#") and not line.strip().startswith('"'):
            owner = enclosing(mod, i)
            writers.setdefault(owner, []).append(i)
print(f"write-capable functions (a write call inside): {len(writers)}")
for owner, lines in sorted(writers.items(), key=lambda kv: str(kv[0])):
    print(f"   {owner[0]}.{owner[1] if owner else '<module>'} lines {lines}")


def resolve_class(mod, name):
    if (mod, name) in classes:
        return (mod, name)
    if name in from_names[mod]:
        base, n = from_names[mod][name]
        if (base, n) in classes:
            return (base, n)
    return None


def refs(mod, qual):
    node = funcs[(mod, qual)]
    cls = qual.split(".")[0] if "." in qual else None
    out, unresolved = set(), set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            n = sub.id
            if (mod, n) in funcs:
                out.add((mod, n))
            elif resolve_class(mod, n):
                c = resolve_class(mod, n)
                out |= {k for k in funcs if k[0] == c[0] and k[1].startswith(c[1] + ".")}
            elif n in from_names[mod]:
                base, name = from_names[mod][n]
                if (base, name) in funcs:
                    out.add((base, name))
        elif isinstance(sub, ast.Attribute):
            v = sub.value
            if isinstance(v, ast.Name):
                if v.id in aliases[mod]:
                    target = aliases[mod][v.id]
                    if (target, sub.attr) in funcs:
                        out.add((target, sub.attr))
                    elif (target, sub.attr) in classes:
                        out |= {k for k in funcs if k[0] == target and k[1].startswith(sub.attr + ".")}
                    continue
                if v.id in ("self", "cls") and cls and (mod, f"{cls}.{sub.attr}") in funcs:
                    out.add((mod, f"{cls}.{sub.attr}"))
                    continue
                c = resolve_class(mod, v.id)
                if c and (c[0], f"{c[1]}.{sub.attr}") in funcs:
                    out.add((c[0], f"{c[1]}.{sub.attr}"))
                    continue
            if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id in aliases[mod]:
                target = aliases[mod][v.value.id]
                if (target, f"{v.attr}.{sub.attr}") in funcs:
                    out.add((target, f"{v.attr}.{sub.attr}"))
                    continue
            # unresolved attribute: over-approximate by method name across the package
            if sub.attr in methods_by_name:
                unresolved.add(sub.attr)
                out |= {k for k in methods_by_name[sub.attr] if "." in k[1]}
    return out, unresolved


ENTRIES = [("run024", q) for q in ("Runner._base", "Runner._confirmation", "Runner._state_for", "Runner._installed_record", "Runner._noun_keys",
                                   "Runner._check_runtime", "Runner._check_nouns", "Runner._inputs", "Runner._provenance", "Runner._record_022",
                                   "_git_tracked", "_git_changed_paths", "runtime_record", "pytest_free_guard.__enter__", "pytest_free_guard.__exit__")]
ENTRIES += [("neural_decompiler.readout_routing", q) for q in ("assert_ledger_isolated", "scientific_dependencies", "validate_lock", "verify_model_dependencies",
                                                               "parameters_digest", "embedding_digest", "reproduce_lock_quantities", "assert_phase_allowed")]
ENTRIES += [("neural_decompiler.provenance", "collect_git_state"), ("neural_decompiler.provenance", "collect_versions"),
            ("neural_decompiler.models", "seed_runtime"), ("neural_decompiler.models", "load_model"), ("neural_decompiler.upstream_localization", "ModelPrograms.from_model")]
reach, frontier, via_unresolved = set(ENTRIES), list(ENTRIES), set()
while frontier:
    item = frontier.pop()
    found, unresolved = refs(*item)
    via_unresolved |= unresolved
    for f in found - reach:
        reach.add(f)
        frontier.append(f)
hit = sorted((w for w in writers if w in reach), key=str)
print(f"reachable functions (over-approximation): {len(reach)}; unresolved method names widened: {len(via_unresolved)}")
print("write-capable functions reachable:", [f"{m}.{q}" for m, q in hit])
exact_hits = []
for m, q in hit:
    print(f"   REACHED (needs manual check): {m}.{q}")
s = summary()
print(f"STATIC {'NO WRITER REACHED' if not hit else 'WRITERS REACHED (see above; resolve by reading)'}; guard refused={s['refused']}")
