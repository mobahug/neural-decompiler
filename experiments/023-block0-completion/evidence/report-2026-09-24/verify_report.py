"""Experiment 023 report verification: strictly read-only."""
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
D = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/report023")
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


out = ROOT / "outputs/experiment-023"
text = (out / "report.md").read_text(encoding="utf-8")
before = rd.load_results_state(D / "results-pre-report.json")
after = rd.load_results_state(out / "results.json")
check("pre-report state is the confirmed one", before["state_sha256"] == "938cb2469d6445b4ffcf04ca5937e8d26f4e80c80e7b96ad5458102f70fa5876")
record = json.loads((ROOT / b0c.CALIBRATION_RELATIVE_PATH).read_text())
arrays = b0c.draw_values_verified(torch.load(out / "draw-values.pt"), record)
check("the report is exactly the frozen renderer's output for the confirmed state", b0c.render_report(before, record, arrays) == text)
check("state: report complete; its sha256 == the file's text", after["phases"]["report"]["status"] == "complete" and after["report"]["sha256"] == pm.sha256_text(text)
      and Path(after["report"]["path"]) == out / "report.md", after["report"]["sha256"])
check("state: only the report fields changed", {k: v for k, v in after.items() if k not in ("report", "phases", "state_sha256")}
      == {k: v for k, v in before.items() if k not in ("report", "phases", "state_sha256")}
      and {p: e for p, e in after["phases"].items() if p != "report"} == {p: e for p, e in before["phases"].items() if p != "report"})
results = before["confirmation"]["conditions"]
lock = json.loads((ROOT / b0c.LOCK_RELATIVE_PATH).read_text())
for condition in b0c.CONDITIONS:
    e = results[condition]
    row = next(line for line in text.splitlines() if line.startswith(f"| {condition} | {e['g']:.6f}"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    g_values, defined = arrays[f"{condition}/g"], arrays[f"{condition}/defined"].bool()
    cdf = int(((g_values.double() <= e["g"]) & defined).sum()) / int(defined.sum())
    ok = (cells[1] == f"{e['g']:.6f}" and cells[2] == f"{lock['conditions'][condition]['envelope']['bound']:.6f}" and cells[4] == f"**{e['result']}**"
          and cells[5] == f"{e['gap']:.4f}" and cells[6:9] == [f"{e['R2_0']:.4f}", f"{e['R2_1']:.4f}", f"{e['R2_C']:.4f}"] and cells[9] == ("yes" if e["ceiling_limited"] else "no")
          and cells[10].startswith(f"{cdf:.4f}") and cells[11] == str(e["n_pairs"]))
    check(f"{condition}: row == results (g {e['g']!r}, F {lock['conditions'][condition]['envelope']['bound']!r}, {e['result']}, CDF {cdf})", ok)
    check(f"{condition}: reading == the frozen semantics", f"- {condition}: **{e['result']}** — {b0c.SEMANTICS['results'][e['result']]}." in text)
check("no aggregate label; aggregate note present", "Aggregate: none: the four condition results are the result, each read on its own; Y1 and Y2 are never pooled; no all-pass requirement" in text
      and before["confirmation"]["aggregate_label"] is None)
check("joint rate shown as descriptive only", "Joint rate, all four passing (descriptive only): 0.9178" in text)
check("scope sentence and the ceiling note present", b0c.SCOPE in text and "not a mathematical upper bound" in text)
check("no incident lines (the renderer's '- **… incident** at' form)", not re.search(r"^- \*\*.*[Ii]ncident\*\* at", text, flags=re.M))
check("calibration table == record (F, min, median, max, undefined, rates)", all(
    f"| {c} | ≥ v₍250₎ = {record['conditions'][c]['envelope']['bound']:.6f} | {record['conditions'][c]['summary']['min']:.6f} | "
    f"{record['conditions'][c]['summary']['median']:.6f} | {record['conditions'][c]['summary']['max']:.6f} | 0 | no | 0.9751 | 0.0249 | 0.0000 | 0.0000 |" in text for c in b0c.CONDITIONS))
check("HEAD unchanged; tree clean", subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "c7efec7f32709dc20ecb83cae16bb73927a14366"
      and subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True).stdout == "")
print("post-report state_sha256", after["state_sha256"], "| results.json file", rc.file_sha256(out / "results.json"), "| report.md", rc.file_sha256(out / "report.md"))
print("\nVERIFY REPORT:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
