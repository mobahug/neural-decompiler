"""Final assembly: my reconstructed lock (static fields + the model-derived Y1 digests) serialized with my own canonical
JSON and compared byte for byte with the candidate; the preregistration rendered from my lock (text-only); re-hash of
every relevant file against the review's starting snapshot; git tree and HEAD unchanged. 022's table blocked."""
import sys
SCRATCH = "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lockreview023"
sys.path.insert(0, SCRATCH)
import guard  # noqa: E402  (first)

import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = guard.ROOT
OUT = Path(SCRATCH)
print("guard:", guard.prove_live()["live"], flush=True)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {str(detail)[:500]}", flush=True)


def cj(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, check=True, text=True).stdout


expected = json.loads((OUT / "expected_lock_partial.json").read_text(encoding="utf-8"))
model = json.loads((OUT / "review_model.json").read_text(encoding="utf-8"))
candidate_text = (ROOT / "outputs/experiment-023/candidate-lock.json").read_text(encoding="utf-8")
candidate = json.loads(candidate_text)
regenerated = (OUT / "regenerated-y1-table.f64").read_bytes()
expected["y1_table"].update({"file_sha256": sha(regenerated), "index_sha256": model["y1_index"]["sha256_mine"], "p0_dx3_sha256": model["y1_table"]["p0_dx3_sha256"],
                             "factors_sha256": model["y1_table"]["factors_sha256"],
                             # I5 is reproduced exactly (0.0 at no pair); the algebra maximum is a rounding-level diagnostic of b0c's own
                             # closed-form implementation and cannot be reproduced bit for bit by an independent one: taken from the candidate.
                             "gates": {"I5": {"max": model["checks"]["I5_max_abs"]["max"], "at": model["checks"]["I5_max_abs"]["at"]},
                                       "algebra": candidate["y1_table"]["gates"]["algebra"]}})
unsigned = {k: v for k, v in expected.items() if k != "content_sha256"}
expected["content_sha256"] = sha(cj(unsigned).encode("utf-8"))
mine_text = cj(expected) + "\n"
check("lock: my reconstruction serialized (own canonical JSON) is byte-identical to candidate-lock.json", mine_text == candidate_text,
      {"mine": sha(mine_text.encode()), "candidate": sha(candidate_text.encode()), "content_sha256": expected["content_sha256"]})
check("lock: algebra gate recorded 4.88e-15 at stacks|coordinated-adjective-015-1; my closed form 5.0e-15 at the same pair (tolerance 1e-12)",
      candidate["y1_table"]["gates"]["algebra"]["max"] <= 1e-12 and model["checks"]["b_closed_form_T-vs-022_attn_pt"]["max"] <= 1e-12,
      {"lock": candidate["y1_table"]["gates"]["algebra"], "mine": model["checks"]["b_closed_form_T-vs-022_attn_pt"]})
from neural_decompiler import block0_completion as b0c  # noqa: E402  (render_preregistration only: the allowed text check)

prereg_candidate = (ROOT / "outputs/experiment-023/candidate-preregistration.md").read_bytes()
check("preregistration rendered from MY lock == candidate-preregistration.md (bytes)", b0c.render_preregistration(expected).encode("utf-8") == prereg_candidate, sha(prereg_candidate))

BASELINE = {
    "outputs/experiment-023/candidate-calibration.json": "94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a",
    "outputs/experiment-023/candidate-exposed-cells.f64": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
    "outputs/experiment-023/candidate-exposed-cells.json": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe",
    "outputs/experiment-023/candidate-lock.json": "4bd5a5b14627768d49398c273fa1ce387db2e4739d7d31b7258072ef48beac9c",
    "outputs/experiment-023/candidate-locked-y1-table.f64": "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96",
    "outputs/experiment-023/candidate-locked-y1-table.json": "3bf85e186db7dfbf8c80f45557275058317e259cb5995dcd2f73e327dfd9644d",
    "outputs/experiment-023/candidate-preregistration.md": "dc198785b6350d20165f589236741f5f08ffc845da33840ca156c7d71780dc07",
    "outputs/experiment-023/draw-values.pt": "38ef8b5b49f0334b67849945c4dd9ecb54daff379607251670006f22ebb7d3bf",
    "outputs/experiment-023/results.json": "a3344e8380c67b330cf86c827290c80a84678cf8b94d3445ccae1dc753dd834e",
    "experiments/023-block0-completion/calibration-v1.json": "94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a",
    "experiments/023-block0-completion/confirmation-v1.json": "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4",
    "experiments/023-block0-completion/exposed-cells.json": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe",
    "experiments/023-block0-completion/exposed-cells.f64": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
    "experiments/023-block0-completion/run.py": "22b5cbe18a6dc78e4e0a62f994bababffe23c638c9c62ca7c177389b44644e2d",
    "experiments/023-block0-completion/evidence/archive/README.md": "1916614056c8d72a1583300aa637ca79370b2eb728c36b4fe9312389396b1873",
    "experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz": "69d726ac54b86ea98cc49a3e995de631875fb5417771b2212bd99cbe6fef6f86",
    "outputs/experiment-020/results.json": "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00",
}
now = {rel: sha((ROOT / rel).read_bytes()) for rel in BASELINE}
listing = sorted(p.name for p in (ROOT / "outputs/experiment-023").iterdir())
check("re-hash: every file unchanged since the review started; outputs/experiment-023 holds exactly the 9 files", now == BASELINE
      and listing == sorted(Path(rel).name for rel in BASELINE if rel.startswith("outputs/experiment-023/")), [rel for rel in BASELINE if now[rel] != BASELINE[rel]])
status = git("status", "--porcelain", "--untracked-files=all")
head = git("rev-parse", "HEAD").strip()
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
check("git: tree clean incl. untracked; HEAD == origin == 13b8d39", status == "" and head == remote == "13b8d392f9e88844e2673c530a37228337cc64b7", repr(status[:200]))
pyc = subprocess.run(["find", str(ROOT), "-path", str(ROOT / ".venv"), "-prune", "-o", "-name", "*.pyc", "-newer", str(OUT / "guard.py"), "-print"], capture_output=True, text=True).stdout.split()
check("no bytecode written into the repo during the review", pyc == [], pyc[:5])
check("guard: no refusal during this process; nothing under outputs/experiment-022 opened", guard.REFUSED == [] and not any("outputs/experiment-022" in p for p in guard.OPENED), guard.REFUSED)
print(f"\nFINAL: {sum(ok for _, ok in RESULTS)} PASS, {sum(not ok for _, ok in RESULTS)} FAIL", flush=True)
